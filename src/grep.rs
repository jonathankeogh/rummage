//! Live grep: the query's exact text in the files themselves, beside meaning search. Built on ripgrep's engine
//! (grep-searcher, grep-regex). It reads plain-text documents, or code for a "code: …" query, under the indexed
//! folders with the same skip rules as indexing but no size limits, so it also finds what the index left out (a big
//! log, a file past the passage limit, a code project indexed without --code).
//!
//! Files are tried newest first and grep stops at `limit` matching files, so it keeps the most recently changed
//! matches without reading everything. It also stops at a deadline, or when the caller cancels (a newer query).

use crate::store::Kind;
use anyhow::Result;
use grep_regex::RegexMatcherBuilder;
use grep_searcher::sinks::UTF8;
use grep_searcher::{BinaryDetection, SearcherBuilder};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// How many matching files grep keeps.
pub const LIMIT: usize = 20;

/// A file grep matched: where the first match is, and how many lines match (counted up to 1,000).
#[derive(Clone, Debug)]
pub struct GrepHit {
    pub path: PathBuf,
    pub kind: Kind,
    /// Seconds since 1970 it last changed.
    pub mtime: i64,
    /// The first matching line, from 1, and its text.
    pub line: usize,
    pub text: String,
    pub count: usize,
}

/// What to look for: `/…/` is a regular expression; anything else the literal text. Case is ignored unless the
/// query has a capital (smart case, as in ripgrep).
pub fn pattern(query: &str) -> Option<(String, bool)> {
    let query = query.trim();
    if query.len() > 2 && query.starts_with('/') && query.ends_with('/') {
        return Some((query[1..query.len() - 1].to_string(), true));
    }
    let literal = query.trim_matches('"');
    (literal.chars().count() >= 2).then(|| (literal.to_string(), false))
}

/// The newest `limit` files among `files` (path, kind, mtime) that contain `query`, newest first.
pub fn grep(
    files: &[(PathBuf, Kind, i64)],
    query: &str,
    limit: usize,
    deadline: Instant,
    cancel: &AtomicBool,
) -> Result<Vec<GrepHit>> {
    let Some((pattern, regex)) = pattern(query) else { return Ok(Vec::new()) };
    let matcher = RegexMatcherBuilder::new().case_smart(true).fixed_strings(!regex).build(&pattern)?;
    let mut searcher =
        SearcherBuilder::new().binary_detection(BinaryDetection::quit(b'\x00')).line_number(true).build();
    let mut order: Vec<&(PathBuf, Kind, i64)> = files.iter().collect();
    order.sort_by(|a, b| b.2.cmp(&a.2));
    let mut hits = Vec::new();
    for (path, kind, mtime) in order {
        if hits.len() >= limit || Instant::now() > deadline || cancel.load(Ordering::Relaxed) {
            break;
        }
        let mut first: Option<(usize, String)> = None;
        let mut count = 0;
        let searched = searcher.search_path(
            &matcher,
            path,
            UTF8(|line, text| {
                if first.is_none() {
                    first = Some((line as usize, text.trim_end().chars().take(300).collect()));
                }
                count += 1;
                Ok(count < 1000)
            }),
        );
        if let (Ok(()), Some((line, text))) = (searched, first) {
            hits.push(GrepHit { path: path.clone(), kind: *kind, mtime: *mtime, line, text, count });
        }
    }
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn patterns() {
        assert_eq!(pattern("for"), Some(("for".into(), false)));
        assert_eq!(pattern("\"card declined\""), Some(("card declined".into(), false)));
        assert_eq!(pattern("/INV\\d+/"), Some(("INV\\d+".into(), true)));
        assert_eq!(pattern("a"), None);
    }

    #[test]
    fn newest_matches_first_with_a_limit() {
        let dir = std::env::temp_dir().join(format!("rummage-grep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = |name: &str, text: &str, mtime: i64| {
            let path = dir.join(name);
            std::fs::write(&path, text).unwrap();
            (path, Kind::Doc, mtime)
        };
        let files = vec![
            file("old.txt", "they were fighting\n", 100),
            file("new.txt", "intro\nFighting broke out\nfighting again\n", 300),
            file("none.txt", "peace\n", 400),
            file("mid.log", "no FIGHTING here? yes\n", 200),
            file("bin.dat", "fighting\0binary", 500),
        ];
        let cancel = AtomicBool::new(false);
        let far = Instant::now() + Duration::from_secs(10);
        let hits = grep(&files, "fighting", 10, far, &cancel).unwrap();
        let names: Vec<_> = hits.iter().map(|h| h.path.file_name().unwrap().to_str().unwrap()).collect();
        assert_eq!(names, vec!["new.txt", "mid.log", "old.txt"], "newest first, binary skipped");
        assert_eq!((hits[0].line, hits[0].count, hits[0].text.as_str()), (2, 2, "Fighting broke out"));
        // Smart case: a capital makes it exact.
        assert_eq!(grep(&files, "FIGHTING", 10, far, &cancel).unwrap().len(), 1);
        // The limit keeps the newest.
        assert_eq!(grep(&files, "fighting", 1, far, &cancel).unwrap()[0].path, files[1].0);
        // Regex.
        assert_eq!(grep(&files, "/fight\\w+ (broke|again)/", 10, far, &cancel).unwrap().len(), 1);
        // Cancelled: nothing more is read.
        cancel.store(true, Ordering::Relaxed);
        assert!(grep(&files, "fighting", 10, far, &cancel).unwrap().is_empty());
    }
}
