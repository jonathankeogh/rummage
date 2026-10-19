//! Keeps the index in step with the chosen folders: new and changed files are read and embedded, gone ones dropped.
//!
//! Long files go in steps, as in DigUp: a first pass reads the start of every file (all of a short one; the first 30
//! pages of a big PDF or 40 passages of a long document), so everything is searchable soon, then later passes read
//! the rest of the long ones. A long file is searchable from its first step on.

use crate::crawl::{self, Found};
use crate::embed::{Embedder, Picture, prompts};
use crate::extract::{self, Part};
use crate::paths::ModelFiles;
use crate::store::{Kind, Modality, Store};
use crate::{ocr, picture};
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// PDFs longer than this go in steps.
const BIG_PDF_PAGES: usize = 50;
/// The first step of a big PDF, and of a long document.
const FIRST_PAGES: usize = 30;
const FIRST_PASSAGES: usize = 40;
/// Parts per step in the later passes.
const LATER_STEP: usize = 200;
/// Small files are grouped so the GPU gets full batches.
const GROUP_PARTS: usize = 32;

#[derive(Default, Debug)]
pub struct Sync {
    pub found: usize,
    pub new_or_changed: usize,
    pub removed: usize,
    /// PDFs left out because poppler isn't installed.
    pub pdfs_without_tools: usize,
    /// HEIC photos left out because nothing here converts them.
    pub heic_without_tool: usize,
    /// Sound and video left out because ffmpeg isn't installed.
    pub media_without_ffmpeg: usize,
    /// Pictures left out as icons (under 256 px) or as dataset folders.
    pub small_pictures: usize,
    pub dataset_files: usize,
}

/// Compares what's on disk under `roots` with the index: new and changed files become pending, gone ones go.
pub fn sync(store: &Store, roots: &[PathBuf], options: &crawl::Options) -> Result<Sync> {
    let walk = crawl::walk(roots, options);
    let found = walk.found;
    let mut summary = Sync {
        found: found.len(),
        pdfs_without_tools: walk.pdfs_without_tools,
        heic_without_tool: walk.heic_without_tool,
        media_without_ffmpeg: walk.media_without_ffmpeg,
        small_pictures: walk.small_pictures,
        dataset_files: walk.dataset_files,
        ..Default::default()
    };
    let tx = store.db.unchecked_transaction()?;
    for Found { path, kind, size, mtime } in &found {
        // Failed files are tried again: they're few, and a newer rummage may read them (scans, once it could).
        let unchanged = store
            .file(path)?
            .is_some_and(|f| f.size == *size && f.mtime == *mtime && f.state != "pending" && f.state != "failed");
        if !unchanged {
            store.upsert_pending(path, *kind, *size, *mtime)?;
            summary.new_or_changed += 1;
        }
    }
    let on_disk: std::collections::HashSet<&Path> = found.iter().map(|f| f.path.as_path()).collect();
    // Gone: deleted, excluded, or under no indexed folder any more. A folder missing altogether (an unplugged drive)
    // keeps its files until it's back.
    let missing: Vec<&PathBuf> = roots.iter().filter(|root| !root.exists()).collect();
    for (id, path) in store.files_under(Path::new("/"))? {
        if !on_disk.contains(path.as_path()) && !missing.iter().any(|root| path.starts_with(root)) {
            store.remove_file(id)?;
            summary.removed += 1;
        }
    }
    tx.commit()?;
    Ok(summary)
}

/// What indexing tells its caller as it goes.
pub enum Progress<'a> {
    /// The first pass finished a file: `done` of `total` files.
    File { done: usize, total: usize, path: &'a Path },
    /// A later pass read more of a long file: `done` of its `total` pages or passages.
    Later { path: &'a Path, kind: Kind, done: usize, total: usize },
}

/// Some of a file's parts, read and ready for the model.
struct Job {
    file_id: i64,
    path: PathBuf,
    name: String,
    parts: Vec<Part>,
    /// A PDF's pages without text in this step (scans): embedded as pictures, their words read by OCR.
    scans: Vec<usize>,
    /// A picture, sound or video file: embedded by what it shows or sounds like (see `embed`).
    picture: Option<Kind>,
    /// Parts read once this job is stored, and the file's total.
    done: usize,
    total: usize,
}

/// Embeds every pending file (the first pass), then the rest of every long file (the later passes). Returns how many
/// files the first pass took.
pub fn embed_pending(store: &Store, mut progress: impl FnMut(Progress)) -> Result<usize> {
    let pending = store.pending()?;
    if pending.is_empty() && store.partial()?.is_empty() {
        return Ok(0);
    }
    let models = ModelFiles::locate()?;
    let mut embedder = Embedder::load(&models, false)?;

    // First pass: the start of every file.
    let total = pending.len();
    let mut done = 0;
    let mut group: Vec<Job> = Vec::new();
    for (file_id, path, kind) in pending {
        match first_step(file_id, &path, kind) {
            Ok(job)
                if job.parts.is_empty() && job.scans.is_empty() && job.picture.is_none() && job.done >= job.total =>
            {
                store.finish(file_id, Some(if kind == Kind::Pdf { "no text (a scan?)" } else { "no text" }))?;
                done += 1;
                progress(Progress::File { done, total, path: &path });
            }
            Ok(job) => group.push(job),
            Err(error) => {
                store.finish(file_id, Some(&format!("{error:#}")))?;
                done += 1;
                progress(Progress::File { done, total, path: &path });
            }
        }
        if group.iter().map(|j| j.parts.len()).sum::<usize>() >= GROUP_PARTS {
            for job in embed(store, &mut embedder, std::mem::take(&mut group))? {
                done += 1;
                progress(Progress::File { done, total, path: &job });
            }
        }
    }
    for job in embed(store, &mut embedder, group)? {
        done += 1;
        progress(Progress::File { done, total, path: &job });
    }

    // Later passes: the rest of each long file, a step at a time, shortest file first.
    for (file_id, path, kind, mut parts_done, mut parts_total) in store.partial()? {
        let name = file_name(&path);
        while parts_done < parts_total {
            let range = parts_done..(parts_done + LATER_STEP).min(parts_total);
            let read = match extract::read_parts(&path, kind, range.clone()) {
                Ok(read) => read,
                Err(error) => {
                    // It may have changed or gone since; keep what's in and let the next sync sort it out.
                    eprintln!("rummage: stopped reading {}: {error:#}", path.display());
                    break;
                }
            };
            parts_total = read.total;
            let job = Job {
                file_id,
                path: path.clone(),
                name: name.clone(),
                parts: read.parts,
                scans: read.blank,
                picture: None,
                done: range.end.min(read.total),
                total: read.total,
            };
            embed(store, &mut embedder, vec![job])?;
            parts_done = range.end.min(read.total);
            progress(Progress::Later { path: &path, kind, done: parts_done, total: parts_total });
        }
    }
    Ok(done)
}

/// A file's first step: all of a short file; the start of a long one.
fn first_step(file_id: i64, path: &Path, kind: Kind) -> Result<Job> {
    if kind.is_picture() || kind.is_timed() {
        let name = file_name(path);
        return Ok(Job {
            file_id,
            path: path.to_path_buf(),
            name,
            parts: Vec::new(),
            scans: Vec::new(),
            picture: Some(kind),
            done: 1,
            total: 1,
        });
    }
    let step = match kind {
        Kind::Pdf => {
            let pages = extract::pdf_page_count(path)?.min(extract::MAX_PAGES);
            if pages > BIG_PDF_PAGES { FIRST_PAGES } else { pages }
        }
        Kind::Doc => FIRST_PASSAGES,
        // Code files are small (1 MB at most): all at once.
        Kind::Code => extract::MAX_PASSAGES,
        Kind::Image | Kind::Screenshot | Kind::Audio | Kind::Video => unreachable!("handled above"),
    };
    let read = extract::read_parts(path, kind, 0..step)?;
    Ok(Job {
        file_id,
        path: path.to_path_buf(),
        name: file_name(path),
        parts: read.parts,
        scans: read.blank,
        picture: None,
        done: step.min(read.total),
        total: read.total,
    })
}

fn file_name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Embeds and stores the jobs' parts (text passages and pages, scanned pages, pictures), records each file's progress,
/// and returns the paths of the files it took. A picture waits (stays pending) while the image encoder isn't there.
fn embed(store: &Store, embedder: &mut Embedder, jobs: Vec<Job>) -> Result<Vec<PathBuf>> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    let inputs: Vec<String> = jobs
        .iter()
        .flat_map(|job| job.parts.iter().map(|part| prompts::document(Some(&job.name), &part.text)))
        .collect();
    let vectors = embedder.embed_texts(&inputs)?;

    // Pictures, scanned pages, sound and video, one at a time, and the files that failed.
    let mut seen: Vec<Seen> = Vec::new();
    let mut failed: Vec<(usize, String)> = Vec::new();
    let mut waiting: Vec<usize> = Vec::new();
    for (j, job) in jobs.iter().enumerate() {
        if let Some(kind) = job.picture {
            if !embedder.sees_images() {
                waiting.push(j);
                continue;
            }
            let result = if kind.is_timed() {
                embed_timed(embedder, &job.path, kind).map(|stretches| {
                    for (start, end, vector, modality) in stretches {
                        seen.push(Seen { job: j, loc: start, end: Some(end), words: String::new(), vector, modality });
                    }
                })
            } else {
                embed_picture(embedder, &job.path, kind).map(|(vector, words)| {
                    seen.push(Seen { job: j, loc: 1.0, end: None, words, vector, modality: Modality::Image })
                })
            };
            if let Err(error) = result {
                failed.push((j, format!("{error:#}")));
            }
        }
        if embedder.sees_images() {
            for &page in &job.scans {
                match embed_scanned_page(embedder, &job.path, page) {
                    Ok((vector, words)) => seen.push(Seen {
                        job: j,
                        loc: page as f64,
                        end: None,
                        words,
                        vector,
                        modality: Modality::Image,
                    }),
                    Err(error) => eprintln!("rummage: page {page} of {}: {error:#}", job.path.display()),
                }
            }
        }
    }

    let tx = store.db.unchecked_transaction()?;
    let mut next = vectors.iter();
    let mut paths = Vec::with_capacity(jobs.len());
    for (j, job) in jobs.into_iter().enumerate() {
        for part in &job.parts {
            let vector = next.next().expect("one vector per part");
            let excerpt = extract::excerpt(&part.text, 200);
            store.add_segment(
                job.file_id,
                part.loc as f64,
                &job.name,
                &part.text,
                &excerpt,
                Some(vector),
                Modality::Text,
            )?;
        }
        for s in seen.iter().filter(|s| s.job == j) {
            // A picture's words in full (up to 4,000 characters): its preview shows them. A stretch of sound or video
            // says where it is.
            let excerpt: String = match s.end {
                Some(end) => format!(
                    "{} {}–{}",
                    if s.modality == Modality::Image { "frame" } else { "sound" },
                    crate::media::clock(s.loc),
                    crate::media::clock(end)
                ),
                None => s.words.chars().take(4000).collect(),
            };
            store.add_stretch(job.file_id, s.loc, s.end, &job.name, &s.words, &excerpt, Some(&s.vector), s.modality)?;
        }
        if waiting.contains(&j) {
            continue;
        }
        match failed.iter().find(|f| f.0 == j) {
            Some((_, error)) => store.finish(job.file_id, Some(error))?,
            None => store.progress(job.file_id, job.done, job.total)?,
        }
        paths.push(job.path);
    }
    tx.commit()?;
    Ok(paths)
}

/// A vector of part of a file that isn't text: a picture, a scanned page, a video frame, a stretch of sound.
struct Seen {
    job: usize,
    /// Page, or start in seconds; for a stretch, where it ends.
    loc: f64,
    end: Option<f64>,
    words: String,
    vector: Vec<f32>,
    modality: Modality,
}

/// A recording's or a video's vectors: (start, end, vector, modality). A video gives a frame per shot (70 vision
/// tokens) and its sound; a recording its sound, in 30 s windows.
fn embed_timed(embedder: &mut Embedder, path: &Path, kind: Kind) -> Result<Vec<(f64, f64, Vec<f32>, Modality)>> {
    let mut out = Vec::new();
    let (has_audio, has_video) = crate::media::streams(path);
    if kind == Kind::Video && has_video {
        let folder = std::env::temp_dir().join(format!("rummage-frames-{}-{}", std::process::id(), now_nanos()));
        let frames = crate::media::keyframes(path, &folder);
        let result = frames.and_then(|frames| {
            for frame in frames {
                let vector = embedder.embed_image(Picture::File(frame.path.clone()), crate::media::FRAME_BUDGET)?;
                out.push((frame.start, frame.end, vector, Modality::Image));
            }
            Ok(())
        });
        let _ = std::fs::remove_dir_all(&folder);
        result?;
    }
    if has_audio {
        crate::media::audio_windows(path, |window| {
            let vector = embedder.embed_audio(&window.samples)?;
            out.push((window.start, window.end, vector, Modality::Audio));
            Ok(())
        })?;
    }
    if out.is_empty() && !has_audio && !has_video {
        anyhow::bail!("no sound or picture in it");
    }
    Ok(out)
}

fn now_nanos() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos())
}

/// A picture's vector (photos at 140 vision tokens, screenshots at 280) and, for a screenshot, its words.
fn embed_picture(embedder: &mut Embedder, path: &Path, kind: Kind) -> Result<(Vec<f32>, String)> {
    let picture = if picture::stb_reads(path) {
        Picture::File(path.to_path_buf())
    } else {
        Picture::Pixels(picture::decode(path)?)
    };
    let budget = if kind == Kind::Screenshot { picture::SCREENSHOT_BUDGET } else { picture::PHOTO_BUDGET };
    let vector = embedder.embed_image(picture, budget)?;
    let words =
        if kind == Kind::Screenshot && ocr::available() { ocr::file(path).unwrap_or_default() } else { String::new() };
    Ok((vector, words))
}

/// A scanned PDF page: rendered, embedded as a picture (280 vision tokens), and read by OCR.
fn embed_scanned_page(embedder: &mut Embedder, path: &Path, page: usize) -> Result<(Vec<f32>, String)> {
    // Tall enough for OCR to read body text; the encoder gets it resized to its budget.
    let png = extract::pdf_page_png(path, page, 2000)?;
    let rgb = image::load_from_memory(&png)?.to_rgb8();
    let pixels = picture::Rgb { width: rgb.width() as usize, height: rgb.height() as usize, data: rgb.into_raw() };
    let vector = embedder.embed_image(Picture::Pixels(pixels), picture::PAGE_BUDGET)?;
    let words = if ocr::available() { ocr::png_bytes(&png).unwrap_or_default() } else { String::new() };
    Ok((vector, words))
}

/// Seconds as "3 s" or "2 min".
pub fn duration(started: Instant) -> String {
    let seconds = started.elapsed().as_secs_f64();
    if seconds < 90.0 { format!("{seconds:.1} s") } else { format!("{:.0} min", seconds / 60.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_leave_with_their_folder_but_not_with_an_unplugged_drive() {
        let dir = std::env::temp_dir().join(format!("rummage-sync-{}", std::process::id()));
        let (a, b) = (dir.join("a"), dir.join("b"));
        for folder in [&a, &b] {
            std::fs::create_dir_all(folder).unwrap();
            std::fs::write(folder.join("note.md"), "A note about the garden and what to plant in spring.").unwrap();
        }
        let store = Store::open(&dir.join("index")).unwrap();
        let options = crawl::Options { code: false, excluded: Vec::new() };
        assert_eq!(sync(&store, &[a.clone(), b.clone()], &options).unwrap().new_or_changed, 2);
        // b is no longer indexed: its file goes.
        assert_eq!(sync(&store, std::slice::from_ref(&a), &options).unwrap().removed, 1);
        // a is missing altogether, as an unplugged drive would be: its file stays.
        let moved = dir.join("a-away");
        std::fs::rename(&a, &moved).unwrap();
        assert_eq!(sync(&store, std::slice::from_ref(&a), &options).unwrap().removed, 0);
        assert_eq!(store.files_under(&a).unwrap().len(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
