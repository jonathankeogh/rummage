//! The index: one SQLite file with the files, a vector per passage, and a keyword index (FTS5) over names and text.

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, params};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Doc,
    Code,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Doc => "doc",
            Kind::Code => "code",
        }
    }

    pub fn parse(s: &str) -> Option<Kind> {
        match s {
            "doc" => Some(Kind::Doc),
            "code" => Some(Kind::Code),
            _ => None,
        }
    }
}

/// Which model and pipeline made the vectors. An index made with another one must embed again.
pub const FINGERPRINT: &str = "embeddinggemma-2@bfcd298+llama.cpp-b11461/vectors1";

pub struct Store {
    pub db: Connection,
    pub file: PathBuf,
}

pub struct FileRow {
    pub id: i64,
    pub size: u64,
    pub mtime: i64,
    pub state: String,
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
                 state TEXT NOT NULL,          -- pending | done | failed
                 error TEXT
             );
             CREATE TABLE IF NOT EXISTS segments (
                 id INTEGER PRIMARY KEY,
                 file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                 loc REAL,                     -- passage number from 1 (pages and seconds later)
                 excerpt TEXT NOT NULL,
                 vector BLOB NOT NULL          -- 768 little-endian f32
             );
             CREATE INDEX IF NOT EXISTS segments_file ON segments(file_id);
             CREATE VIRTUAL TABLE IF NOT EXISTS words USING fts5(
                 name, text, content='', contentless_delete=1, tokenize='unicode61 remove_diacritics 2'
             );",
        )?;
        let store = Store { db, file };
        match store.meta("fingerprint")? {
            None => store.set_meta("fingerprint", FINGERPRINT)?,
            Some(f) if f == FINGERPRINT => {}
            Some(f) => bail!("this index was made by another model or pipeline ({f}): run `rummage index --reindex`"),
        }
        Ok(store)
    }

    pub fn meta(&self, key: &str) -> Result<Option<String>> {
        Ok(self.db.query_row("SELECT value FROM meta WHERE key = ?", [key], |r| r.get(0)).optional()?)
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
            .query_row("SELECT id, size, mtime, state FROM files WHERE path = ?", [path.to_string_lossy()], |r| {
                Ok(FileRow { id: r.get(0)?, size: r.get::<_, i64>(1)? as u64, mtime: r.get(2)?, state: r.get(3)? })
            })
            .optional()?)
    }

    /// Adds a file, or marks a changed one pending again (its old passages go).
    pub fn upsert_pending(&self, path: &Path, kind: Kind, size: u64, mtime: i64) -> Result<i64> {
        if let Some(existing) = self.file(path)? {
            self.clear_segments(existing.id)?;
            self.db.execute(
                "UPDATE files SET kind = ?, size = ?, mtime = ?, state = 'pending', error = NULL WHERE id = ?",
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

    pub fn add_segment(
        &self,
        file_id: i64,
        loc: f64,
        name: &str,
        text: &str,
        excerpt: &str,
        vector: &[f32],
    ) -> Result<()> {
        let blob: Vec<u8> = vector.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.db.execute(
            "INSERT INTO segments (file_id, loc, excerpt, vector) VALUES (?, ?, ?, ?)",
            params![file_id, loc, excerpt, blob],
        )?;
        let id = self.db.last_insert_rowid();
        self.db.execute("INSERT INTO words (rowid, name, text) VALUES (?, ?, ?)", params![id, name, text])?;
        Ok(())
    }

    pub fn finish(&self, file_id: i64, error: Option<&str>) -> Result<()> {
        let state = if error.is_some() { "failed" } else { "done" };
        self.db.execute("UPDATE files SET state = ?, error = ? WHERE id = ?", params![state, error, file_id])?;
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
            "DELETE FROM words; DELETE FROM segments; UPDATE files SET state = 'pending', error = NULL;",
        )?;
        self.set_meta("fingerprint", FINGERPRINT)
    }
}

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
