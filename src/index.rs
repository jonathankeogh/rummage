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
    for root in roots {
        for (id, path) in store.files_under(root)? {
            if !on_disk.contains(path.as_path()) {
                store.remove_file(id)?;
                summary.removed += 1;
            }
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
    /// A picture file (photo or screenshot): its one vector is of what it shows.
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
    if kind.is_picture() {
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
        Kind::Image | Kind::Screenshot => unreachable!("pictures are handled above"),
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

    // Pictures and scanned pages, one at a time: (job, loc, words, vector), and the pictures that failed.
    let mut seen: Vec<(usize, f64, String, Vec<f32>)> = Vec::new();
    let mut failed: Vec<(usize, String)> = Vec::new();
    let mut waiting: Vec<usize> = Vec::new();
    for (j, job) in jobs.iter().enumerate() {
        if let Some(kind) = job.picture {
            if !embedder.sees_images() {
                waiting.push(j);
                continue;
            }
            match embed_picture(embedder, &job.path, kind) {
                Ok((vector, words)) => seen.push((j, 1.0, words, vector)),
                Err(error) => failed.push((j, format!("{error:#}"))),
            }
        }
        if embedder.sees_images() {
            for &page in &job.scans {
                match embed_scanned_page(embedder, &job.path, page) {
                    Ok((vector, words)) => seen.push((j, page as f64, words, vector)),
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
        for (_, loc, words, vector) in seen.iter().filter(|s| s.0 == j) {
            // A picture's words in full (up to 4,000 characters): its preview shows them.
            let excerpt: String = words.chars().take(4000).collect();
            store.add_segment(job.file_id, *loc, &job.name, words, &excerpt, Some(vector), Modality::Image)?;
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
