//! The index: one SQLite file with the files, a vector per passage, and a keyword index (FTS5) over names and text.

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// Text and documents (txt, md, docx, odt, html …), cut into passages.
    Doc,
    /// PDFs, a page at a time.
    Pdf,
    Code,
    /// Photos and other pictures, by what they show.
    Image,
    /// Screenshots: by what they show, and their words (OCR) for keyword search.
    Screenshot,
    /// Recordings, in 30 s windows.
    Audio,
    /// Videos: a frame per shot, and their sound.
    Video,
}

/// What a passage's vector was made from. A text query sits at different similarities to text and to pictures, so
/// each gets its own baseline when scoring.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Image,
    Audio,
}

impl Modality {
    pub fn as_str(self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::Image => "image",
            Modality::Audio => "audio",
        }
    }

    pub fn parse(s: &str) -> Modality {
        match s {
            "image" => Modality::Image,
            "audio" => Modality::Audio,
            _ => Modality::Text,
        }
    }
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Doc => "doc",
            Kind::Pdf => "pdf",
            Kind::Code => "code",
            Kind::Image => "image",
            Kind::Screenshot => "screenshot",
            Kind::Audio => "audio",
            Kind::Video => "video",
        }
    }

    /// Sound or video: `loc` is a moment, in seconds.
    pub fn is_timed(self) -> bool {
        matches!(self, Kind::Audio | Kind::Video)
    }

    /// What `loc` counts for this kind.
    pub fn unit(self) -> &'static str {
        match self {
            Kind::Pdf => "page",
            Kind::Image | Kind::Screenshot => "picture",
            Kind::Audio | Kind::Video => "second",
            Kind::Doc | Kind::Code => "passage",
        }
    }

    /// A short name for result lists (at most 5 characters).
    pub fn label(self) -> &'static str {
        match self {
            Kind::Image => "img",
            Kind::Screenshot => "shot",
            Kind::Video => "video",
            other => other.as_str(),
        }
    }

    /// Pictures, shown as pictures.
    pub fn is_picture(self) -> bool {
        matches!(self, Kind::Image | Kind::Screenshot)
    }

    pub fn parse(s: &str) -> Option<Kind> {
        match s {
            "doc" => Some(Kind::Doc),
            "pdf" => Some(Kind::Pdf),
            "code" => Some(Kind::Code),
            "image" => Some(Kind::Image),
            "screenshot" => Some(Kind::Screenshot),
            "audio" => Some(Kind::Audio),
            "video" => Some(Kind::Video),
            _ => None,
        }
    }
}

/// Which model and pipeline made the vectors. An index made with another one must embed again.
pub const FINGERPRINT: &str = "embeddinggemma-2@bfcd298+llama.cpp-b11461/vectors1/stemmed-words";

/// This rummage, as a failed file records it: a newer one tries the file again (it may read it now).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The keyword index: file names and text, contentless (the text stays in the files), English endings stemmed
/// ("invoices" finds "invoice"), accents ignored. A change here changes `FINGERPRINT`: its text can't be read back.
const WORDS_TABLE: &str = "CREATE VIRTUAL TABLE IF NOT EXISTS words USING fts5(
    name, text, content='', contentless_delete=1, tokenize='porter unicode61 remove_diacritics 2'
);";

pub struct Store {
    pub db: Connection,
    pub file: PathBuf,
}

pub struct FileRow {
    pub id: i64,
    pub size: u64,
    pub mtime: i64,
    pub state: String,
    /// For a failed file, the rummage that failed it.
    pub failed_by: Option<String>,
    /// The date the last sweep saw on disk, when that's newer than `mtime` (a busy file not read again yet).
    pub seen_mtime: Option<i64>,
}

impl FileRow {
    /// Read as far as it goes, or failed by this rummage: nothing to do until it changes.
    pub fn settled(&self) -> bool {
        match self.state.as_str() {
            "pending" => false,
            "failed" => self.failed_by.as_deref() == Some(VERSION),
            _ => true,
        }
    }
}

impl Store {
    pub fn open(dir: &Path) -> Result<Store> {
        std::fs::create_dir_all(dir).with_context(|| format!("can't create {}", dir.display()))?;
        let file = dir.join("index.sqlite");
        let db = Connection::open(&file)?;
        db.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA foreign_keys = ON;
             CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS roots (path TEXT PRIMARY KEY);
             CREATE TABLE IF NOT EXISTS files (
                 id INTEGER PRIMARY KEY,
                 path TEXT NOT NULL UNIQUE,
                 kind TEXT NOT NULL,
                 size INTEGER NOT NULL,
                 mtime INTEGER NOT NULL,
                 state TEXT NOT NULL,          -- pending | partial | done | failed
                 error TEXT,
                 parts_done INTEGER,           -- pages or passages embedded so far (a long file goes in steps)
                 parts_total INTEGER
             );
             CREATE TABLE IF NOT EXISTS segments (
                 id INTEGER PRIMARY KEY,
                 file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                 loc REAL,                     -- passage or page number from 1 (seconds later)
                 excerpt TEXT NOT NULL,
                 vector BLOB NOT NULL,         -- 768 little-endian f32; empty for keyword-only text (OCR words)
                 modality TEXT NOT NULL DEFAULT 'text',  -- text | image | audio: what the vector was made from
                 loc_end REAL                  -- where a stretch of sound or video ends, in seconds
             );
             CREATE INDEX IF NOT EXISTS segments_file ON segments(file_id);",
        )?;
        db.execute_batch(WORDS_TABLE)?;
        // Indexes made before long files went in steps lack the progress columns.
        let columns: Vec<String> = db
            .prepare("SELECT name FROM pragma_table_info('files')")?
            .query_map([], |r| r.get(0))?
            .collect::<Result<_, _>>()?;
        for column in ["parts_done", "parts_total"] {
            if !columns.iter().any(|c| c == column) {
                db.execute(&format!("ALTER TABLE files ADD COLUMN {column} INTEGER"), [])?;
            }
        }
        // And before pictures, the modality.
        let has_modality: bool =
            db.query_row("SELECT COUNT(*) > 0 FROM pragma_table_info('segments') WHERE name = 'modality'", [], |r| {
                r.get(0)
            })?;
        if !has_modality {
            db.execute("ALTER TABLE segments ADD COLUMN modality TEXT NOT NULL DEFAULT 'text'", [])?;
        }
        // And before sound and video, where a stretch ends.
        let has_end: bool =
            db.query_row("SELECT COUNT(*) > 0 FROM pragma_table_info('segments') WHERE name = 'loc_end'", [], |r| {
                r.get(0)
            })?;
        if !has_end {
            db.execute("ALTER TABLE segments ADD COLUMN loc_end REAL", [])?;
        }
        // And before failed files waited for a change, which rummage failed them.
        if !columns.iter().any(|c| c == "failed_by") {
            db.execute("ALTER TABLE files ADD COLUMN failed_by TEXT", [])?;
        }
        // And before files that keep changing waited, the date a sweep last saw.
        if !columns.iter().any(|c| c == "seen_mtime") {
            db.execute("ALTER TABLE files ADD COLUMN seen_mtime INTEGER", [])?;
        }
        let store = Store { db, file };
        match store.meta("fingerprint")? {
            None => store.set_meta("fingerprint", FINGERPRINT)?,
            Some(f) if f == FINGERPRINT => {}
            Some(f) => return Err(StaleIndex { made_by: f }.into()),
        }
        Ok(store)
    }

    pub fn meta(&self, key: &str) -> Result<Option<String>> {
        Ok(self.db.query_row("SELECT value FROM meta WHERE key = ?", [key], |r| r.get(0)).optional()?)
    }

    /// Folders `rummage index --exclude` left out, kept for later runs (and the service's).
    pub fn excluded(&self) -> Result<Vec<PathBuf>> {
        Ok(self.meta("excluded")?.map(|v| v.lines().map(PathBuf::from).collect()).unwrap_or_default())
    }

    pub fn add_excluded(&self, folders: &[PathBuf]) -> Result<()> {
        let mut all = self.excluded()?;
        for folder in folders {
            if !all.contains(folder) {
                all.push(folder.clone());
            }
        }
        let joined: Vec<String> = all.iter().map(|p| p.to_string_lossy().into_owned()).collect();
        self.set_meta("excluded", &joined.join("\n"))
    }

    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", [key, value])?;
        Ok(())
    }

    pub fn roots(&self) -> Result<Vec<PathBuf>> {
        let mut statement = self.db.prepare("SELECT path FROM roots ORDER BY path")?;
        let rows = statement.query_map([], |r| r.get::<_, String>(0))?;
        Ok(rows.filter_map(|r| r.ok()).map(PathBuf::from).collect())
    }

    pub fn add_root(&self, path: &Path) -> Result<()> {
        self.db.execute("INSERT OR IGNORE INTO roots (path) VALUES (?)", [path.to_string_lossy()])?;
        Ok(())
    }

    pub fn file(&self, path: &Path) -> Result<Option<FileRow>> {
        Ok(self
            .db
            .query_row(
                "SELECT id, size, mtime, state, failed_by, seen_mtime FROM files WHERE path = ?",
                [path.to_string_lossy()],
                |r| {
                    Ok(FileRow {
                        id: r.get(0)?,
                        size: r.get::<_, i64>(1)? as u64,
                        mtime: r.get(2)?,
                        state: r.get(3)?,
                        failed_by: r.get(4)?,
                        seen_mtime: r.get(5)?,
                    })
                },
            )
            .optional()?)
    }

    /// Adds a file, or marks a changed one pending again (its old passages go).
    pub fn upsert_pending(&self, path: &Path, kind: Kind, size: u64, mtime: i64) -> Result<i64> {
        if let Some(existing) = self.file(path)? {
            self.clear_segments(existing.id)?;
            self.db.execute(
                "UPDATE files SET kind = ?, size = ?, mtime = ?, state = 'pending', error = NULL, parts_done = NULL,
                 parts_total = NULL, seen_mtime = NULL WHERE id = ?",
                params![kind.as_str(), size as i64, mtime, existing.id],
            )?;
            return Ok(existing.id);
        }
        self.db.execute(
            "INSERT INTO files (path, kind, size, mtime, state) VALUES (?, ?, ?, ?, 'pending')",
            params![path.to_string_lossy(), kind.as_str(), size as i64, mtime],
        )?;
        Ok(self.db.last_insert_rowid())
    }

    /// Records the date a sweep saw on a busy file it left as it was.
    pub fn saw(&self, file_id: i64, mtime: i64) -> Result<()> {
        self.db.execute("UPDATE files SET seen_mtime = ? WHERE id = ?", params![mtime, file_id])?;
        Ok(())
    }

    pub fn pending(&self) -> Result<Vec<(i64, PathBuf, Kind)>> {
        let mut statement =
            self.db.prepare("SELECT id, path, kind FROM files WHERE state = 'pending' ORDER BY path")?;
        let rows =
            statement.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?)))?;
        Ok(rows
            .filter_map(|r| r.ok())
            .filter_map(|(id, path, kind)| Some((id, PathBuf::from(path), Kind::parse(&kind)?)))
            .collect())
    }

    /// Files read in part, with how far along: (id, path, kind, parts done, parts total), smallest first.
    pub fn partial(&self) -> Result<Vec<(i64, PathBuf, Kind, usize, usize)>> {
        let mut statement = self.db.prepare(
            "SELECT id, path, kind, parts_done, parts_total FROM files WHERE state = 'partial' ORDER BY parts_total",
        )?;
        let rows = statement.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, i64>(4)?,
            ))
        })?;
        Ok(rows
            .filter_map(|r| r.ok())
            .filter_map(|(id, path, kind, done, total)| {
                Some((id, PathBuf::from(path), Kind::parse(&kind)?, done as usize, total as usize))
            })
            .collect())
    }

    /// The long file later passes are reading, when every file has had its first: "Film.mp4 30 of 118 minutes",
    /// "Atlas.pdf page 230 of 900".
    pub fn reading_now(&self) -> Result<Option<String>> {
        let pending: i64 = self.db.query_row("SELECT COUNT(*) FROM files WHERE state = 'pending'", [], |r| r.get(0))?;
        if pending > 0 {
            return Ok(None);
        }
        let Some((path, kind, done, total)) = self
            .db
            .query_row(
                "SELECT path, kind, parts_done, parts_total FROM files WHERE state = 'partial' ORDER BY parts_total LIMIT 1",
                [],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?, r.get::<_, i64>(3)?)),
            )
            .optional()?
        else {
            return Ok(None);
        };
        let name = Path::new(&path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or(path);
        let kind = Kind::parse(&kind).unwrap_or(Kind::Doc);
        Ok(Some(match kind {
            Kind::Audio | Kind::Video => format!("{name} {} of {} minutes", done / 60, (total + 59) / 60),
            _ => format!("{name} {} {done} of {total}", kind.unit()),
        }))
    }

    /// Records how far a file has been read: `partial` until every part is in, then `done`.
    pub fn progress(&self, file_id: i64, done: usize, total: usize) -> Result<()> {
        let state = if done < total { "partial" } else { "done" };
        self.db.execute(
            "UPDATE files SET state = ?, error = NULL, parts_done = ?, parts_total = ? WHERE id = ?",
            params![state, done as i64, total as i64, file_id],
        )?;
        Ok(())
    }

    pub fn add_segment(
        &self,
        file_id: i64,
        loc: f64,
        name: &str,
        text: &str,
        excerpt: &str,
        vector: Option<&[f32]>,
        modality: Modality,
    ) -> Result<()> {
        self.add_stretch(file_id, loc, None, name, text, excerpt, vector, modality)
    }

    /// A passage that covers a stretch of time, from `loc` to `loc_end` seconds (sound, video).
    #[allow(clippy::too_many_arguments)]
    pub fn add_stretch(
        &self,
        file_id: i64,
        loc: f64,
        loc_end: Option<f64>,
        name: &str,
        text: &str,
        excerpt: &str,
        vector: Option<&[f32]>,
        modality: Modality,
    ) -> Result<()> {
        // No vector: the text is for keyword search only (a screenshot's words).
        let blob: Vec<u8> = vector.unwrap_or_default().iter().flat_map(|x| x.to_le_bytes()).collect();
        self.db.execute(
            "INSERT INTO segments (file_id, loc, loc_end, excerpt, vector, modality) VALUES (?, ?, ?, ?, ?, ?)",
            params![file_id, loc, loc_end, excerpt, blob, modality.as_str()],
        )?;
        let id = self.db.last_insert_rowid();
        self.db.execute("INSERT INTO words (rowid, name, text) VALUES (?, ?, ?)", params![id, name, text])?;
        Ok(())
    }

    /// Done; or failed with `error`, which also takes out whatever of it was in (a half-read file never shows).
    pub fn finish(&self, file_id: i64, error: Option<&str>) -> Result<()> {
        match error {
            None => {
                self.db.execute("UPDATE files SET state = 'done', error = NULL WHERE id = ?", [file_id])?;
            }
            Some(error) => {
                self.clear_segments(file_id)?;
                self.db.execute(
                    "UPDATE files SET state = 'failed', error = ?, failed_by = ?, parts_done = NULL, parts_total = NULL
                     WHERE id = ?",
                    params![error, VERSION, file_id],
                )?;
            }
        }
        Ok(())
    }

    fn clear_segments(&self, file_id: i64) -> Result<()> {
        self.db.execute("DELETE FROM words WHERE rowid IN (SELECT id FROM segments WHERE file_id = ?)", [file_id])?;
        self.db.execute("DELETE FROM segments WHERE file_id = ?", [file_id])?;
        Ok(())
    }

    pub fn remove_file(&self, file_id: i64) -> Result<()> {
        self.clear_segments(file_id)?;
        self.db.execute("DELETE FROM files WHERE id = ?", [file_id])?;
        Ok(())
    }

    /// Paths of indexed files under `root`.
    pub fn files_under(&self, root: &Path) -> Result<Vec<(i64, PathBuf)>> {
        let prefix = format!("{}/", root.to_string_lossy().trim_end_matches('/'));
        let mut statement = self.db.prepare("SELECT id, path FROM files WHERE substr(path, 1, length(?1)) = ?1")?;
        let rows = statement.query_map([&prefix], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
        Ok(rows.filter_map(|r| r.ok()).map(|(id, p)| (id, PathBuf::from(p))).collect())
    }

    /// Drops every vector so everything embeds again (after a model change).
    pub fn reset_vectors(&self) -> Result<()> {
        self.db.execute_batch(
            "DROP TABLE IF EXISTS words; DELETE FROM segments;
             UPDATE files SET state = 'pending', error = NULL, parts_done = NULL, parts_total = NULL;",
        )?;
        self.db.execute_batch(WORDS_TABLE)?;
        self.set_meta("fingerprint", FINGERPRINT)
    }
}

/// An index made by another model or pipeline: its vectors and words can't be used, only rebuilt.
#[derive(Debug)]
pub struct StaleIndex {
    pub made_by: String,
}

impl std::fmt::Display for StaleIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "this index was made by another model or pipeline ({}): the search service rebuilds it when it starts \
             (any search starts it), or run `rummage index --reindex`",
            self.made_by
        )
    }
}

impl std::error::Error for StaleIndex {}

/// Opens an index whose fingerprint may be stale, for `--reindex`.
pub fn open_for_reindex(dir: &Path) -> Result<Store> {
    let file = dir.join("index.sqlite");
    if file.exists() {
        let db = Connection::open(&file)?;
        db.execute("UPDATE meta SET value = ? WHERE key = 'fingerprint'", [FINGERPRINT]).ok();
    }
    let store = Store::open(dir)?;
    store.reset_vectors()?;
    Ok(store)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn says_which_long_file_later_passes_are_reading() {
        let dir = std::env::temp_dir().join(format!("rummage-store-{}", std::process::id()));
        let store = Store::open(&dir).unwrap();
        let film = store.upsert_pending(Path::new("/m/Film.mp4"), Kind::Video, 1, 1).unwrap();
        let atlas = store.upsert_pending(Path::new("/d/Atlas.pdf"), Kind::Pdf, 1, 1).unwrap();
        // While first passes go on, nothing to say.
        assert_eq!(store.reading_now().unwrap(), None);
        store.progress(film, 1800, 7080).unwrap();
        store.progress(atlas, 230, 900).unwrap();
        assert_eq!(store.reading_now().unwrap().as_deref(), Some("Atlas.pdf page 230 of 900"));
        store.progress(atlas, 900, 900).unwrap();
        assert_eq!(store.reading_now().unwrap().as_deref(), Some("Film.mp4 30 of 118 minutes"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
