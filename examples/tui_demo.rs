//! Try the interface without an index: `cargo run --release --example tui_demo`.
//! Searches the notes in `examples/sample-files` by simple word overlap (no model).

use anyhow::Result;
use rummage::search::Hit;
use rummage::store::Kind;
use rummage::tui::{self, Backend};
use std::collections::HashSet;
use std::path::PathBuf;

const STOP_WORDS: [&str; 14] =
    ["the", "a", "an", "of", "on", "in", "to", "and", "is", "for", "with", "about", "my", "i"];

/// Words that mean nearly the same, so a query can match by "meaning" a little.
const RELATED: [(&str, &str); 8] = [
    ("pet", "dog"),
    ("pet", "cat"),
    ("coffee", "caffeine"),
    ("card", "payment"),
    ("retry", "backoff"),
    ("food", "recipe"),
    ("sea", "beach"),
    ("exercise", "gym"),
];

struct SampleBackend {
    files: Vec<(PathBuf, Kind, String)>,
}

impl SampleBackend {
    fn load() -> Result<SampleBackend> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/sample-files");
        let mut files = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            let kind = match path.extension().and_then(|e| e.to_str()) {
                Some("rs" | "py") => Kind::Code,
                _ => Kind::Doc,
            };
            let text = std::fs::read_to_string(&path)?;
            files.push((path, kind, text));
        }
        files.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(SampleBackend { files })
    }
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|w| w.to_lowercase())
        .filter(|w| !w.is_empty() && !STOP_WORDS.contains(&w.as_str()))
        .map(|w| if w.len() > 3 { w.trim_end_matches('s').to_string() } else { w })
        .collect()
}

impl Backend for SampleBackend {
    fn search(&mut self, query: &str, limit: usize) -> Result<Vec<Hit>> {
        // Like the real engine: "code:" searches code, anything else searches documents.
        let (kind, query) = match query.trim_start().strip_prefix("code:") {
            Some(rest) => (Kind::Code, rest),
            None => (Kind::Doc, query),
        };
        let query_words = words(query);
        let mut scored: Vec<(f64, usize, usize, usize)> = Vec::new();
        for (i, (path, file_kind, text)) in self.files.iter().enumerate() {
            if *file_kind != kind {
                continue;
            }
            let file_words: HashSet<String> =
                words(&format!("{} {}", path.file_stem().unwrap_or_default().to_string_lossy(), text))
                    .into_iter()
                    .collect();
            let exact = query_words.iter().filter(|w| file_words.contains(*w)).count();
            let related = query_words
                .iter()
                .filter(|w| {
                    RELATED
                        .iter()
                        .any(|(a, b)| (a == w && file_words.contains(*b)) || (b == w && file_words.contains(*a)))
                })
                .count();
            if exact + related > 0 {
                scored.push((exact as f64 + 0.5 * related as f64, exact, related, i));
            }
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut word_rank = 0;
        let hits = scored
            .into_iter()
            .take(limit)
            .filter_map(|(score, exact, related, i)| {
                let z = 1.5 + 1.1 * score;
                // Only a standout counts as a meaning match, as in the real search.
                if z < 2.0 && exact == 0 {
                    return None;
                }
                let keyword_rank = (exact > 0).then(|| {
                    word_rank += 1;
                    word_rank
                });
                let (path, kind, text) = &self.files[i];
                Some(Hit {
                    path: path.display().to_string(),
                    kind: *kind,
                    score,
                    z: (z >= 2.0 || related > 0).then_some(z),
                    cosine: Some(0.3 + 0.05 * score),
                    keyword_rank,
                    loc: Some(1.0),
                    loc_end: None,
                    line: None,
                    grep: false,
                    excerpt: text.chars().take(200).collect(),
                    also_in: Vec::new(),
                })
            })
            .collect();
        Ok(hits)
    }
}

fn main() -> Result<()> {
    tui::run(SampleBackend::load()?)
}
