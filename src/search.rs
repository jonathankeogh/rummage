//! Hybrid search: meaning (EmbeddingGemma's vectors) plus words (FTS5).
//!
//! Every vector is scored for each query, so the scores' spread comes for free: a passage counts as a meaning match
//! only when it stands out (z ≥ `z_gate`), which also gives an honest "nothing found". Words catch what a vector can't
//! hold exactly: codes, IDs and names. The scoring is DigUp's (MIT), tuned there on its evals.

use crate::embed::DIM;
use crate::store::{Kind, Store};
use anyhow::Result;
use rusqlite::params;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Hit {
    pub path: String,
    pub kind: Kind,
    pub score: f64,
    /// How far the best passage stands out for this query.
    pub z: Option<f64>,
    pub cosine: Option<f64>,
    /// 1-based rank among keyword matches.
    pub keyword_rank: Option<usize>,
    /// Passage number from 1.
    pub loc: Option<f64>,
    pub excerpt: String,
}

pub struct Options {
    pub kinds: Option<HashSet<Kind>>,
    pub limit: usize,
    pub z_gate: f64,
    pub word_weight: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options { kinds: None, limit: 10, z_gate: 2.0, word_weight: 2.0 }
    }
}

/// All vectors of an index in memory, one row each, for scoring every one against a query.
pub struct Vectors {
    matrix: Vec<f32>,
    row_segment: Vec<i64>,
    row_file: Vec<i64>,
    file_kind: HashMap<i64, Kind>,
}

impl Vectors {
    pub fn load(store: &Store) -> Result<Vectors> {
        let mut vectors =
            Vectors { matrix: Vec::new(), row_segment: Vec::new(), row_file: Vec::new(), file_kind: HashMap::new() };
        let mut statement = store.db.prepare("SELECT id, kind FROM files")?;
        for row in statement.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))? {
            let (id, kind) = row?;
            if let Some(kind) = Kind::parse(&kind) {
                vectors.file_kind.insert(id, kind);
            }
        }
        let mut statement = store.db.prepare("SELECT id, file_id, vector FROM segments ORDER BY id")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let blob: Vec<u8> = row.get(2)?;
            if blob.len() != DIM * 4 {
                continue;
            }
            vectors.matrix.extend(blob.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
            vectors.row_segment.push(row.get(0)?);
            vectors.row_file.push(row.get(1)?);
        }
        Ok(vectors)
    }

    pub fn len(&self) -> usize {
        self.row_segment.len()
    }

    pub fn is_empty(&self) -> bool {
        self.row_segment.is_empty()
    }

    /// Cosine of the query with every row (all vectors are L2-normalized, so a dot product).
    fn cosines(&self, query: &[f32]) -> Vec<f32> {
        self.matrix.chunks_exact(DIM).map(|row| row.iter().zip(query).map(|(a, b)| a * b).sum()).collect()
    }
}

/// Robust z-scores of `scores`: centred on the median, scaled by the median absolute deviation (× 1.4826, so it
/// matches the standard deviation of a normal spread). The matches themselves are outliers; with a mean and standard
/// deviation they'd pull the scale towards themselves, and in a small index two good matches hide each other. The
/// spread is floored so a flat field doesn't make everything stand out.
fn standardize(scores: &[f32]) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let median = |values: &mut Vec<f64>| {
        let middle = values.len() / 2;
        *values.select_nth_unstable_by(middle, |a, b| a.total_cmp(b)).1
    };
    let mut values: Vec<f64> = scores.iter().map(|&s| s as f64).collect();
    let center = median(&mut values);
    let mut deviations: Vec<f64> = scores.iter().map(|&s| (s as f64 - center).abs()).collect();
    let deviation = (median(&mut deviations) * 1.4826).max(0.02);
    scores.iter().map(|&s| (s as f64 - center) / deviation).collect()
}

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "to", "was", "were", "will", "with", "about", "where", "what", "which", "who", "how", "when",
    "my", "me", "i", "we", "you", "your", "this", "these", "those", "there", "they", "them", "some", "any",
];

/// The query's words worth matching: lowercased, two letters or more, no stopwords, each once.
pub fn terms(text: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 2 && !STOPWORDS.contains(w))
        .filter(|w| seen.insert(w.to_string()))
        .map(String::from)
        .collect()
}

/// Every word has a digit, and letters or 4+ characters: an invoice number, an error code. Words count for more.
fn is_lookup(terms: &[String]) -> bool {
    !terms.is_empty()
        && terms.iter().all(|t| {
            t.chars().any(|c| c.is_numeric()) && (t.chars().any(|c| c.is_alphabetic()) || t.chars().count() >= 4)
        })
}

/// "code: retry with backoff" → ("retry with backoff", true): a search of code, and nothing else.
pub fn split_code_prefix(text: &str) -> (&str, bool) {
    let trimmed = text.trim_start();
    match trimmed.get(..5) {
        Some(prefix) if prefix.eq_ignore_ascii_case("code:") => (trimmed[5..].trim(), true),
        _ => (text.trim(), false),
    }
}

pub fn search(
    store: &Store,
    vectors: &Vectors,
    text: &str,
    query: Option<&[f32]>,
    options: &Options,
) -> Result<Vec<Hit>> {
    let allowed = |file: i64| match (&options.kinds, vectors.file_kind.get(&file)) {
        (None, _) => true,
        (Some(kinds), Some(kind)) => kinds.contains(kind),
        (Some(_), None) => false,
    };

    // Meaning: every passage on this query's scale; each file keeps its best.
    let mut best: HashMap<i64, (f64, f64, usize)> = HashMap::new(); // file → (z, cosine, row)
    if let Some(query) = query.filter(|q| q.len() == DIM && !vectors.is_empty()) {
        let cosines = vectors.cosines(query);
        let z = standardize(&cosines);
        for row in 0..cosines.len() {
            let file = vectors.row_file[row];
            if allowed(file) && best.get(&file).is_none_or(|b| z[row] > b.0) {
                best.insert(file, (z[row], cosines[row] as f64, row));
            }
        }
    }

    // Words: which files hold which of the query's words, best keyword match first.
    let terms = terms(text);
    let patterns: Vec<String> = terms.iter().map(|t| format!("\"{}\"", t.replace('"', ""))).collect();
    let mut keyword: Vec<(i64, i64)> = Vec::new(); // (file, segment), best first, one per file
    let mut coverage: HashMap<i64, f64> = HashMap::new();
    if !patterns.is_empty() {
        let mut seen = HashSet::new();
        let mut statement = store.db.prepare(
            "SELECT s.file_id, s.id FROM words JOIN segments s ON s.id = words.rowid
             WHERE words MATCH ? ORDER BY bm25(words, 3.0, 1.0) LIMIT 1000",
        )?;
        let rows = statement.query_map([patterns.join(" OR ")], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))?;
        for row in rows {
            let (file, segment) = row?;
            if keyword.len() < 100 && allowed(file) && seen.insert(file) {
                keyword.push((file, segment));
            }
        }
        if !keyword.is_empty() {
            let files = keyword.iter().map(|(f, _)| f.to_string()).collect::<Vec<_>>().join(",");
            let mut statement = store.db.prepare(&format!(
                "SELECT DISTINCT s.file_id FROM words JOIN segments s ON s.id = words.rowid
                 WHERE words MATCH ? AND s.file_id IN ({files})"
            ))?;
            for pattern in &patterns {
                for file in statement.query_map([pattern], |r| r.get::<_, i64>(0))? {
                    *coverage.entry(file?).or_default() += 1.0 / patterns.len() as f64;
                }
            }
        }
    }

    // A file qualifies by standing out in meaning, or by holding at least half of the query's words. Score: meaning z
    // plus a bonus that grows with word coverage; an exact lookup leans on words.
    let (meaning_weight, words_weight) =
        if is_lookup(&terms) { (0.5, options.word_weight * 2.0) } else { (1.0, options.word_weight) };
    let mut candidates: HashSet<i64> = best.iter().filter(|(_, b)| b.0 >= options.z_gate).map(|(f, _)| *f).collect();
    candidates.extend(coverage.iter().filter(|(_, share)| **share >= 0.5).map(|(f, _)| *f));
    let mut scored: Vec<(i64, f64)> = candidates
        .into_iter()
        .map(|file| {
            let z = best.get(&file).map_or(0.0, |b| b.0.max(0.0));
            let words = coverage.get(&file).copied().unwrap_or(0.0).powf(1.5);
            (file, meaning_weight * z + words_weight * words)
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

    let keyword_rank: HashMap<i64, (usize, i64)> =
        keyword.iter().enumerate().map(|(i, (file, segment))| (*file, (i + 1, *segment))).collect();
    let mut hits = Vec::new();
    let mut statement = store.db.prepare(
        "SELECT f.path, f.kind, s.loc, s.excerpt FROM segments s JOIN files f ON f.id = s.file_id WHERE s.id = ?",
    )?;
    for (file, score) in scored.into_iter().take(options.limit) {
        let meaning = best.get(&file);
        let words = keyword_rank.get(&file);
        // Point at the best meaning passage if it stands out, else at the one the words matched.
        let segment = match (meaning, words) {
            (Some(m), w) if m.0 >= options.z_gate || w.is_none() => vectors.row_segment[m.2],
            (_, Some(w)) => w.1,
            _ => continue,
        };
        let row = statement.query_row(params![segment], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, Option<f64>>(2)?, r.get::<_, String>(3)?))
        })?;
        hits.push(Hit {
            path: row.0,
            kind: Kind::parse(&row.1).unwrap_or(Kind::Doc),
            score,
            z: meaning.map(|m| m.0),
            cosine: meaning.map(|m| m.1),
            keyword_rank: words.map(|w| w.0),
            loc: row.2,
            excerpt: row.3,
        });
    }
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terms_drop_stopwords_and_repeats() {
        assert_eq!(terms("Where is the lease clause about PETS, pets?"), vec!["lease", "clause", "pets"]);
    }

    #[test]
    fn code_prefix() {
        assert_eq!(split_code_prefix("  Code:  retry backoff "), ("retry backoff", true));
        assert_eq!(split_code_prefix("code:"), ("", true));
        assert_eq!(split_code_prefix("source code: retry"), ("source code: retry", false));
        assert_eq!(split_code_prefix("dé"), ("dé", false));
    }

    #[test]
    fn lookups_are_codes() {
        assert!(is_lookup(&terms("INV20931")));
        assert!(is_lookup(&terms("0x80070005 20931")));
        assert!(!is_lookup(&terms("error 0x80070005")));
        assert!(!is_lookup(&terms("a dog on the beach")));
    }

    #[test]
    fn two_matches_in_a_small_index_both_stand_out() {
        // The cosines of "a dog playing on the beach" against a 7-note test folder.
        let z = standardize(&[0.725, 0.716, 0.586, 0.558, 0.485, 0.559, 0.537]);
        assert!(z[0] >= 2.0 && z[1] >= 2.0, "{z:?}");
        assert!(z[2..].iter().all(|&v| v < 2.0), "{z:?}");
    }

    #[test]
    fn standardize_floors_the_spread() {
        let z = standardize(&[0.5, 0.5, 0.5, 0.51]);
        assert!(z[3] < 1.0, "a 0.01 bump in a flat field isn't a standout: {}", z[3]);
    }
}
