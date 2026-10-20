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
/// Text that's mostly digits (data dumps, a CSV saved as .txt) has only this many passages embedded: a digit costs the
/// model about a token, so such passages take several times longer than prose, and their vectors say little. All of it
/// stays searchable by its words.
const NUMBERS_EMBEDDED: usize = 3;

/// Recordings and videos go in passes of this many seconds (their progress is counted in seconds). One up to a minute
/// longer goes at once.
const PASS_SECONDS: usize = 600;

/// A file changed twice within this many seconds is busy (a log, a download, a document being worked on): it's read
/// once it has been still this long. RUMMAGE_BUSY_SECONDS shortens it for tests.
pub fn busy_seconds() -> i64 {
    std::env::var("RUMMAGE_BUSY_SECONDS").ok().and_then(|s| s.parse().ok()).unwrap_or(300)
}

/// Whether a changed file waits: its date moved less than `busy` seconds past the one a sweep last saw (or it was
/// read at), and it's still less than `busy` seconds old. A file that appears and stays still is read at once.
fn is_busy(last_seen: i64, now_dated: i64, now: i64, busy: i64) -> bool {
    last_seen > 0 && now_dated - last_seen < busy && now - now_dated < busy
}

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
    /// Files in the cloud only, not downloaded here: left out until they are (those already in stay).
    pub cloud_only: usize,
    /// Network and cloud mounts inside the folders, left out.
    pub remote_folders: Vec<PathBuf>,
    /// Photos libraries macOS didn't let it read (Full Disk Access).
    pub photos_refused: Vec<PathBuf>,
    /// Files changing often, left as they were read until they're still (see `is_busy`).
    pub busy: usize,
    /// New files whose content was already in (copies), and files moved or renamed: none read again.
    pub copies: usize,
    pub moved: usize,
    /// Files from before content hashes, hashed once.
    pub hashed: usize,
}

/// Compares what's on disk under `roots` with the index: new and changed files become pending, gone ones go. In the
/// service's runs (`background`), files that keep changing wait until they're still; yours read every change.
pub fn sync(store: &Store, roots: &[PathBuf], options: &crawl::Options, background: bool) -> Result<Sync> {
    let walk = crawl::walk(roots, options);
    let found = walk.found;
    let mut summary = Sync {
        found: found.len(),
        pdfs_without_tools: walk.pdfs_without_tools,
        heic_without_tool: walk.heic_without_tool,
        media_without_ffmpeg: walk.media_without_ffmpeg,
        small_pictures: walk.small_pictures,
        dataset_files: walk.dataset_files,
        cloud_only: walk.placeholders,
        remote_folders: walk.remote_folders,
        photos_refused: walk.refused,
        ..Default::default()
    };
    let tx = store.db.unchecked_transaction()?;
    // A file that went to the cloud after it was read (evicted) keeps what's in; one that's new or changed there waits
    // until it's downloaded.
    let mut kept: Vec<&Path> = Vec::new();
    for file in &walk.cloud {
        match store.file(&file.path)? {
            Some(f) if f.size == file.size && f.mtime == file.mtime && f.settled() => kept.push(&file.path),
            _ => summary.cloud_only += 1,
        }
    }
    let (now, busy) = (now_seconds(), busy_seconds());
    // When the first busy file is due, for the service's timer.
    let mut busy_until: Option<i64> = None;
    let on_disk: std::collections::HashSet<&Path> = found.iter().map(|f| f.path.as_path()).chain(kept).collect();
    // A folder missing altogether (an unplugged drive) keeps its files until it's back.
    let missing: Vec<&PathBuf> = roots.iter().filter(|root| !root.exists()).collect();
    let gone = |path: &Path| !on_disk.contains(path) && !missing.iter().any(|root| path.starts_with(root));
    for Found { path, kind, size, mtime } in &found {
        let known = store.file(path)?;
        // A failed file is tried again when it changes, or by a newer rummage (it may read it now); not at every sweep.
        if let Some(f) = known.as_ref().filter(|f| f.size == *size && f.mtime == *mtime && f.settled()) {
            // Read before content hashes: hashed once, and a duplicate read twice becomes a copy.
            if f.hash.is_none()
                && f.copy_of.is_none()
                && let Ok(hash) = content_hash(path)
            {
                store.set_hash(f.id, &hash)?;
                summary.hashed += 1;
                if let Some((holder, at, state)) = store.holder(&hash, f.id)?
                    && f.state == "done"
                    && state == "done"
                    && !gone(&at)
                {
                    store.make_copy(f.id, *kind, *size, *mtime, &hash, holder)?;
                }
            }
            continue;
        }
        if let Some(f) = known.as_ref().filter(|f| {
            let last_seen = f.seen_mtime.unwrap_or(f.mtime).max(f.mtime);
            background && f.state != "pending" && f.mtime != *mtime && is_busy(last_seen, *mtime, now, busy)
        }) {
            store.saw(f.id, *mtime)?;
            summary.busy += 1;
            busy_until = Some(busy_until.unwrap_or(i64::MAX).min(mtime + busy));
            continue;
        }
        // New or changed: its content decides. (A file that can't be read for its hash is read as new, and fails
        // with the reason.)
        let hash = content_hash(path).ok();
        if let (Some(f), Some(hash)) = (&known, &hash)
            && f.hash.as_deref() == Some(hash.as_str())
            && f.state != "failed"
            && f.state != "pending"
        {
            store.touched(f.id, *size, *mtime)?; // its date changed, not its content
            continue;
        }
        let holder = match &hash {
            Some(hash) => store.holder(hash, known.as_ref().map_or(-1, |f| f.id))?,
            None => None,
        };
        if let (Some(hash), Some((holder, at, _))) = (&hash, holder) {
            if known.is_none() && gone(&at) {
                store.moved(holder, path, *kind, *size, *mtime)?;
                summary.moved += 1;
            } else {
                match &known {
                    Some(f) => store.make_copy(f.id, *kind, *size, *mtime, hash, holder)?,
                    None => _ = store.add_copy(path, *kind, *size, *mtime, hash, holder)?,
                }
                summary.copies += 1;
            }
            continue;
        }
        let id = store.upsert_pending(path, *kind, *size, *mtime)?;
        if let Some(hash) = &hash {
            store.set_hash(id, hash)?;
        }
        summary.new_or_changed += 1;
    }
    store.set_meta("busy_until", &busy_until.map_or(String::new(), |t| t.to_string()))?;
    // Gone: deleted, excluded, or under no indexed folder any more (moved ones have their new place already).
    for (id, path) in store.files_under(Path::new("/"))? {
        if gone(&path) {
            store.remove_file(id)?;
            summary.removed += 1;
        }
    }
    store.set_meta("cloud_only", &summary.cloud_only.to_string())?;
    tx.commit()?;
    Ok(summary)
}

/// What indexing tells its caller as it goes.
pub enum Progress<'a> {
    /// The first pass finished a file: `done` of `total` files.
    File { done: usize, total: usize, path: &'a Path },
    /// A later pass read more of a long file: `done` of its `total` pages or passages.
    Later { path: &'a Path, kind: Kind, done: usize, total: usize },
    /// Something worth knowing about a file, said once.
    Note { path: &'a Path, note: &'static str },
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
    /// For sound and video in passes, the seconds this one covers (None: all of it).
    span: Option<(f64, f64)>,
    /// Only this many of `parts` are embedded; the rest go to the word index alone (number-heavy text).
    embedded: Option<usize>,
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
        let step = first_step(file_id, &path, kind);
        if let Ok(Job { embedded: Some(_), .. }) = &step {
            progress(Progress::Note { path: &path, note: "mostly numbers, searched by its words" });
        }
        match step {
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
            for (path, _) in embed(store, &mut embedder, std::mem::take(&mut group))? {
                done += 1;
                progress(Progress::File { done, total, path: &path });
            }
        }
    }
    for (path, _) in embed(store, &mut embedder, group)? {
        done += 1;
        progress(Progress::File { done, total, path: &path });
    }

    // Later passes: the rest of each long file, a step at a time, shortest file first.
    for (file_id, path, kind, mut parts_done, mut parts_total) in store.partial()? {
        let name = file_name(&path);
        // Sound and video: the next ten minutes at a time.
        while kind.is_timed() && parts_done < parts_total {
            let to = (parts_done + PASS_SECONDS).min(parts_total);
            let job = Job {
                file_id,
                path: path.clone(),
                name: name.clone(),
                parts: Vec::new(),
                scans: Vec::new(),
                picture: Some(kind),
                span: Some((parts_done as f64, to as f64)),
                embedded: None,
                done: to,
                total: parts_total,
            };
            let took = embed(store, &mut embedder, vec![job])?;
            if took.is_empty() || took.iter().any(|(_, failed)| *failed) {
                break; // failed (it's out), or the encoder isn't there
            }
            parts_done = to;
            progress(Progress::Later { path: &path, kind, done: parts_done, total: parts_total });
        }
        while !kind.is_timed() && parts_done < parts_total {
            let range = parts_done..(parts_done + LATER_STEP).min(parts_total);
            let read = match extract::read_parts(&path, kind, range.clone()) {
                Ok(read) => read,
                Err(error) => {
                    // The whole file goes, not just this step (if it changed, the next sync reads it again).
                    store.finish(file_id, Some(&format!("{error:#}")))?;
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
                span: None,
                embedded: None,
                done: range.end.min(read.total),
                total: read.total,
            };
            if embed(store, &mut embedder, vec![job])?.iter().any(|(_, failed)| *failed) {
                break; // it's out
            }
            parts_done = range.end.min(read.total);
            progress(Progress::Later { path: &path, kind, done: parts_done, total: parts_total });
        }
    }
    Ok(done)
}

/// A file's first step: all of a short file; the start of a long one.
fn first_step(file_id: i64, path: &Path, kind: Kind) -> Result<Job> {
    if kind.is_picture() || kind.is_timed() {
        // A long recording or video: its first ten minutes now, the rest in later passes.
        let length = if kind.is_timed() { crate::media::duration(path) } else { None };
        let (span, done, total) = match length {
            Some(length) if length > (PASS_SECONDS + 60) as f64 => {
                (Some((0.0, PASS_SECONDS as f64)), PASS_SECONDS, length.ceil() as usize)
            }
            _ => (None, 1, 1),
        };
        return Ok(Job {
            file_id,
            path: path.to_path_buf(),
            name: file_name(path),
            parts: Vec::new(),
            scans: Vec::new(),
            picture: Some(kind),
            span,
            embedded: None,
            done,
            total,
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
    let mut read = extract::read_parts(path, kind, 0..step)?;
    // Mostly numbers: all of it at once, a few passages embedded.
    let mut embedded = None;
    if kind == Kind::Doc && extract::mostly_digits(&read.parts) {
        if read.total > step {
            read = extract::read_parts(path, kind, 0..extract::MAX_PASSAGES)?;
        }
        embedded = Some(NUMBERS_EMBEDDED);
    }
    let done = if embedded.is_some() { read.total } else { step.min(read.total) };
    Ok(Job {
        file_id,
        path: path.to_path_buf(),
        name: file_name(path),
        parts: read.parts,
        scans: read.blank,
        picture: None,
        span: None,
        embedded,
        done,
        total: read.total,
    })
}

fn file_name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Embeds and stores the jobs' parts (text passages and pages, scanned pages, pictures), records each file's progress,
/// and returns the files it took, each with whether it failed. A picture waits (stays pending) while the image encoder isn't there. A
/// file with a part that fails is failed whole: what's in from it (earlier steps too) goes.
fn embed(store: &Store, embedder: &mut Embedder, jobs: Vec<Job>) -> Result<Vec<(PathBuf, bool)>> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    let inputs: Vec<String> = jobs
        .iter()
        .flat_map(|job| {
            let embedded = job.embedded.unwrap_or(usize::MAX);
            job.parts.iter().take(embedded).map(|part| prompts::document(Some(&job.name), &part.text))
        })
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
                embed_timed(embedder, &job.path, kind, job.span).map(|stretches| {
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
                    Err(error) => {
                        failed.push((j, format!("page {page}: {error:#}")));
                        break;
                    }
                }
            }
        }
    }

    let tx = store.db.unchecked_transaction()?;
    let mut next = vectors.iter();
    let mut paths = Vec::with_capacity(jobs.len());
    for (j, job) in jobs.into_iter().enumerate() {
        if let Some((_, error)) = failed.iter().find(|f| f.0 == j) {
            store.finish(job.file_id, Some(error))?;
            paths.push((job.path, true));
            continue;
        }
        for (i, part) in job.parts.iter().enumerate() {
            let vector = if i < job.embedded.unwrap_or(usize::MAX) {
                Some(next.next().expect("one vector per embedded part").as_slice())
            } else {
                None
            };
            let excerpt = extract::excerpt(&part.text, 200);
            store.add_segment(
                job.file_id,
                part.loc as f64,
                &job.name,
                &part.text,
                &excerpt,
                vector,
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
        store.progress(job.file_id, job.done, job.total)?;
        paths.push((job.path, false));
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

/// Videos shorter than this are one picture, their middle frame, without their sound (a Live Photo's clip, a GIF-like
/// loop, a recording started by accident): about a photo's cost instead of ten times more.
const CLIP_SECONDS: f64 = 5.0;

/// A recording's or a video's vectors: (start, end, vector, modality). A video gives a frame per shot (70 vision
/// tokens) and its sound; a recording its sound, in 30 s windows; a clip under 5 s its middle frame as a photo.
fn embed_timed(
    embedder: &mut Embedder,
    path: &Path,
    kind: Kind,
    span: Option<(f64, f64)>,
) -> Result<Vec<(f64, f64, Vec<f32>, Modality)>> {
    let mut out = Vec::new();
    let (has_audio, has_video) = crate::media::streams(path);
    if kind == Kind::Video
        && has_video
        && span.is_none()
        && let Some(length) = crate::media::duration(path).filter(|d| *d < CLIP_SECONDS)
    {
        // The stretch starts at the frame, so the preview shows it.
        let middle = length / 2.0;
        let rgb = crate::media::frame_at(path, middle, 1024)?.to_rgb8();
        let pixels = picture::Rgb { width: rgb.width() as usize, height: rgb.height() as usize, data: rgb.into_raw() };
        let vector = embedder.embed_image(Picture::Pixels(pixels), picture::PHOTO_BUDGET)?;
        return Ok(vec![(middle, length, vector, Modality::Image)]);
    }
    if kind == Kind::Video && has_video {
        let folder = std::env::temp_dir().join(format!("rummage-frames-{}-{}", std::process::id(), now_nanos()));
        let frames = crate::media::keyframes(path, &folder, span);
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
        crate::media::audio_windows(path, span, |window| {
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

/// A file's content as 32 hex digits of its BLAKE3: the same for every copy, whatever its name, place or date.
/// Reading costs about a minute per 100 GB on an SSD, and happens only when a file is new or changed.
pub fn content_hash(path: &Path) -> Result<String> {
    let mut hasher = blake3::Hasher::new();
    hasher.update_reader(std::fs::File::open(path)?)?;
    Ok(hasher.finalize().to_hex()[..32].to_string())
}

fn now_seconds() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
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
    fn a_file_changing_twice_in_five_minutes_waits_until_still() {
        let (now, busy) = (1_000_000, 300);
        // New: read at once.
        assert!(!is_busy(0, now - 1, now, busy));
        // Changed long after it was read: at once.
        assert!(!is_busy(now - 3600, now - 1, now, busy));
        // Changed a minute after it was read, a few seconds ago: waits…
        assert!(is_busy(now - 60, now - 5, now, busy));
        // …until its date is five minutes old.
        assert!(!is_busy(now - 400, now - 300, now, busy));
    }

    #[test]
    fn an_index_from_before_content_hashes_hashes_once_and_merges_duplicates() {
        let dir = std::env::temp_dir().join(format!("rummage-hashes-{}", std::process::id()));
        let (a, b) = (dir.join("a"), dir.join("b"));
        let text = "Notes on the allotment: beans by the fence, potatoes in the far bed.";
        let store = Store::open(&dir.join("index")).unwrap();
        // Both read and embedded, as before hashes.
        for folder in [&a, &b] {
            std::fs::create_dir_all(folder).unwrap();
            let path = folder.join("allotment.md");
            std::fs::write(&path, text).unwrap();
            let meta = std::fs::metadata(&path).unwrap();
            let mtime = meta.modified().unwrap().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
            let id = store.upsert_pending(&path, Kind::Doc, meta.len(), mtime).unwrap();
            store.add_segment(id, 1.0, "allotment.md", text, text, Some(&[0.1; 768]), Modality::Text).unwrap();
            store.progress(id, 1, 1).unwrap();
        }
        let options = crawl::Options { code: false, excluded: Vec::new() };
        let summary = sync(&store, &[a.clone(), b.clone()], &options, false).unwrap();
        assert_eq!((summary.hashed, summary.new_or_changed), (2, 0));
        let holders: i64 = store
            .db
            .query_row("SELECT COUNT(DISTINCT file_id) FROM segments WHERE length(vector) > 0", [], |r| r.get(0))
            .unwrap();
        let copies: i64 =
            store.db.query_row("SELECT COUNT(*) FROM files WHERE state = 'copy'", [], |r| r.get(0)).unwrap();
        assert_eq!((holders, copies), (1, 1));
        // Hashed once: the next sweep has nothing to do.
        assert_eq!(sync(&store, &[a, b], &options, false).unwrap().hashed, 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn files_leave_with_their_folder_but_not_with_an_unplugged_drive() {
        let dir = std::env::temp_dir().join(format!("rummage-sync-{}", std::process::id()));
        let (a, b) = (dir.join("a"), dir.join("b"));
        for folder in [&a, &b] {
            std::fs::create_dir_all(folder).unwrap();
            let text = format!("A note about the garden in {}: what to plant in spring.", folder.display());
            std::fs::write(folder.join("note.md"), text).unwrap();
        }
        let store = Store::open(&dir.join("index")).unwrap();
        let options = crawl::Options { code: false, excluded: Vec::new() };
        assert_eq!(sync(&store, &[a.clone(), b.clone()], &options, false).unwrap().new_or_changed, 2);
        // b is no longer indexed: its file goes.
        assert_eq!(sync(&store, std::slice::from_ref(&a), &options, false).unwrap().removed, 1);
        // a is missing altogether, as an unplugged drive would be: its file stays.
        let moved = dir.join("a-away");
        std::fs::rename(&a, &moved).unwrap();
        assert_eq!(sync(&store, std::slice::from_ref(&a), &options, false).unwrap().removed, 0);
        assert_eq!(store.files_under(&a).unwrap().len(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
