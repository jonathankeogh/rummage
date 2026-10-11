//! Keeps the index in step with the chosen folders: new and changed files are read and embedded, gone ones dropped.

use crate::crawl::{self, Found};
use crate::embed::{Embedder, prompts};
use crate::extract;
use crate::paths::ModelFiles;
use crate::store::Store;
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Default, Debug)]
pub struct Sync {
    pub found: usize,
    pub new_or_changed: usize,
    pub removed: usize,
}

/// Compares what's on disk under `roots` with the index: new and changed files become pending, gone ones go.
pub fn sync(store: &Store, roots: &[PathBuf], options: &crawl::Options) -> Result<Sync> {
    let found = crawl::walk(roots, options);
    let mut summary = Sync { found: found.len(), ..Default::default() };
    let tx = store.db.unchecked_transaction()?;
    for Found { path, kind, size, mtime } in &found {
        let unchanged = store.file(path)?.is_some_and(|f| f.size == *size && f.mtime == *mtime && f.state != "pending");
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

/// A file's passages, ready for the model.
struct Job {
    file_id: i64,
    name: String,
    passages: Vec<String>,
}

/// Embeds every pending file. `progress` hears (done, total, path) after each file.
pub fn embed_pending(store: &Store, mut progress: impl FnMut(usize, usize, &Path)) -> Result<usize> {
    let pending = store.pending()?;
    if pending.is_empty() {
        return Ok(0);
    }
    let models = ModelFiles::locate()?;
    let mut embedder = Embedder::load(&models, false)?;
    // Small files are grouped so the GPU gets full batches.
    const GROUP_PASSAGES: usize = 32;
    let mut group: Vec<(Job, PathBuf)> = Vec::new();
    let mut group_passages = 0;
    let total = pending.len();
    let mut done = 0;
    for (file_id, path, _kind) in pending {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        match extract::read_text(&path) {
            Ok(text) => {
                let passages =
                    extract::passages(&text, extract::PASSAGE_CHARS, extract::OVERLAP_CHARS, extract::MAX_PASSAGES);
                if passages.is_empty() {
                    store.finish(file_id, Some("no text"))?;
                    done += 1;
                    progress(done, total, &path);
                    continue;
                }
                group_passages += passages.len();
                group.push((Job { file_id, name, passages }, path));
            }
            Err(error) => {
                store.finish(file_id, Some(&error.to_string()))?;
                done += 1;
                progress(done, total, &path);
                continue;
            }
        }
        if group_passages >= GROUP_PASSAGES {
            let before = done;
            let mut finished = 0;
            embed_group(store, &mut embedder, std::mem::take(&mut group), &mut |path| {
                finished += 1;
                progress(before + finished, total, path)
            })?;
            done += finished;
            group_passages = 0;
        }
    }
    if !group.is_empty() {
        let before = done;
        let mut finished = 0;
        embed_group(store, &mut embedder, group, &mut |path| {
            finished += 1;
            progress(before + finished, total, path)
        })?;
        done += finished;
    }
    Ok(done)
}

fn embed_group(
    store: &Store,
    embedder: &mut Embedder,
    group: Vec<(Job, PathBuf)>,
    progress: &mut dyn FnMut(&Path),
) -> Result<()> {
    let inputs: Vec<String> =
        group.iter().flat_map(|(job, _)| job.passages.iter().map(|p| prompts::document(Some(&job.name), p))).collect();
    let vectors = embedder.embed_texts(&inputs)?;
    let tx = store.db.unchecked_transaction()?;
    let mut next = vectors.iter();
    for (job, path) in &group {
        for (index, passage) in job.passages.iter().enumerate() {
            let vector = next.next().expect("one vector per passage");
            store.add_segment(
                job.file_id,
                (index + 1) as f64,
                &job.name,
                passage,
                &extract::excerpt(passage, 200),
                vector,
            )?;
        }
        store.finish(job.file_id, None)?;
        progress(path);
    }
    tx.commit()?;
    Ok(())
}

/// Seconds as "3 s" or "2 min".
pub fn duration(started: Instant) -> String {
    let seconds = started.elapsed().as_secs_f64();
    if seconds < 90.0 { format!("{seconds:.1} s") } else { format!("{:.0} min", seconds / 60.0) }
}
