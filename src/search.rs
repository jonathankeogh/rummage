//! Hybrid search: meaning (EmbeddingGemma's vectors) plus words (FTS5).
//!
//! Every vector is scored for each query, so the scores' spread comes for free: a passage counts as a meaning match
//! only when it stands out (z ≥ `z_gate`), which also gives an honest "nothing found". Words catch what a vector can't
//! hold exactly: codes, IDs and names. The scoring is DigUp's (MIT), tuned there on its evals.

use crate::embed::DIM;
use crate::store::{Kind, Store};
use anyhow::Result;
use half::f16;
use half::slice::HalfFloatSliceExt;
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
    /// How much a passage's closeness to the index's centre counts against it (see `Vectors::centrality`).
    pub hub_penalty: f64,
}

impl Default for Options {
    fn default() -> Self {
        // RUMMAGE_HUB_PENALTY overrides the tuned value, for evals.
        let hub_penalty = std::env::var("RUMMAGE_HUB_PENALTY").ok().and_then(|v| v.parse().ok()).unwrap_or(HUB_PENALTY);
        Options { kinds: None, limit: 10, z_gate: 2.0, word_weight: 2.0, hub_penalty }
    }
}

/// Tuned on DigUp's evals and real folders (see `search`).
pub const HUB_PENALTY: f64 = 1.0;

/// Everyday searches, mixed on purpose: their mean is where queries in general sit. A passage close to it is somewhat
/// similar to any query (a hub); a query close to it says little of its own.
pub const TYPICAL_QUERIES: &[&str] = &[
    "a dog on the beach",
    "my tax return from last year",
    "notes from the team meeting",
    "how to cook rice",
    "error message when installing the app",
    "photos from the wedding",
    "the contract with the landlord",
    "a function that parses dates",
    "chicken curry recipe",
    "flight booking confirmation",
    "invoice from the plumber",
    "lecture slides on linear algebra",
    "the bank statement for March",
    "a podcast about sleep",
    "my CV",
    "screenshot of the payment error",
    "train timetable to Galway",
    "the paper on transformers",
    "birthday party plans",
    "how to reset the router",
    "insurance policy for the car",
    "a poem about the sea",
    "budget spreadsheet",
    "the email about the deadline",
    "notes on the history of Rome",
    "a diagram of the network",
    "medical test results",
    "gym workout routine",
    "the lease renewal letter",
    "a video of the concert",
    "the recipe for brown bread",
    "a list of books to read",
    "the quarterly sales report",
    "instructions for the washing machine",
    "my passport scan",
    "a talk about climate change",
    "the code that sends emails",
    "minutes of the board meeting",
    "a map of the city",
    "a letter from the school",
];

/// Where typical queries sit (the mean of `TYPICAL_QUERIES`' vectors) and how spread out they are around it.
#[derive(Clone)]
pub struct QueryCentre {
    pub centre: Vec<f32>,
    /// Mean and spread of the typical queries' own closeness to the centre.
    pub mean: f64,
    pub spread: f64,
}

impl QueryCentre {
    /// From the typical queries' vectors.
    pub fn from_vectors(vectors: &[Vec<f32>]) -> QueryCentre {
        let n = vectors.len().max(1) as f32;
        let mut centre = vec![0.0f32; DIM];
        for v in vectors {
            for (c, x) in centre.iter_mut().zip(v) {
                *c += x / n;
            }
        }
        let at: Vec<f64> = vectors.iter().map(|v| dot(v, &centre) as f64).collect();
        let mean = at.iter().sum::<f64>() / at.len().max(1) as f64;
        let spread = (at.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / at.len().max(1) as f64).sqrt().max(1e-6);
        QueryCentre { centre, mean, spread }
    }
}

/// Dot product over 8 running sums, which the compiler turns into SIMD (a single running sum can't be reordered).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0.0f32; 8];
    let (a8, b8) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = a8.remainder().iter().zip(b8.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in a8.zip(b8) {
        for i in 0..8 {
            sums[i] += x[i] * y[i];
        }
    }
    sums.iter().sum::<f32>() + tail
}

/// All vectors of an index in memory, one row each, for scoring every one against a query.
pub struct Vectors {
    /// Every vector, row after row, in half precision: half the memory of f32, and cosines come out within about 0.001
    /// (DigUp keeps its vectors the same way).
    matrix: Vec<f16>,
    row_segment: Vec<i64>,
    row_file: Vec<i64>,
    file_kind: HashMap<i64, Kind>,
    /// Where typical queries sit, once known (see `set_query_centre`).
    query_centre: Option<QueryCentre>,
    /// Each row's closeness to where typical queries sit: its average similarity to typical queries. A passage with
    /// little meaning of its own (menus, boilerplate, a cover page) is somewhat similar to any query and turns up for
    /// unrelated ones: a "hub". All zero until the query centre is set.
    centrality: Vec<f32>,
}

impl Vectors {
    pub fn load(store: &Store) -> Result<Vectors> {
        let mut vectors = Vectors {
            matrix: Vec::new(),
            row_segment: Vec::new(),
            row_file: Vec::new(),
            file_kind: HashMap::new(),
            query_centre: None,
            centrality: Vec::new(),
        };
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
            vectors
                .matrix
                .extend(blob.chunks_exact(4).map(|b| f16::from_f32(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))));
            vectors.row_segment.push(row.get(0)?);
            vectors.row_file.push(row.get(1)?);
        }
        vectors.centrality = vec![0.0; vectors.row_segment.len()];
        Ok(vectors)
    }

    /// Measures every row against where typical queries sit, for the hub penalty.
    pub fn set_query_centre(&mut self, centre: QueryCentre) {
        self.centrality = self.cosines(&centre.centre);
        self.query_centre = Some(centre);
    }

    /// How much closer to where typical queries sit `query` is than typical queries are, in their standard
    /// deviations: above 0, vaguer than usual. None until the query centre is set.
    pub fn query_vagueness(&self, query: &[f32]) -> Option<f64> {
        let centre = self.query_centre.as_ref()?;
        Some((dot(query, &centre.centre) as f64 - centre.mean) / centre.spread)
    }

    pub fn len(&self) -> usize {
        self.row_segment.len()
    }

    pub fn is_empty(&self) -> bool {
        self.row_segment.is_empty()
    }

    /// Cosine of the query with every row (all vectors are L2-normalized, so a dot product).
    fn cosines(&self, query: &[f32]) -> Vec<f32> {
        // A block of rows at a time back to f32 (SIMD where the CPU has it), then plain dot products.
        const BLOCK: usize = 256;
        let mut floats = vec![0.0f32; BLOCK * DIM];
        let mut out = Vec::with_capacity(self.matrix.len() / DIM);
        for block in self.matrix.chunks(BLOCK * DIM) {
            let floats = &mut floats[..block.len()];
            block.convert_to_f32_slice(floats);
            out.extend(floats.chunks_exact(DIM).map(|row| dot(row, query)));
        }
        out
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

/// The query's words worth matching: lowercased, two letters or more, no stopwords, each once. A query of nothing but
/// stopwords ("for", "how to") keeps them: then the words are all there is to search for.
pub fn terms(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).filter(|w| w.chars().count() >= 2).collect();
    let content: Vec<&str> = words.iter().copied().filter(|w| !STOPWORDS.contains(w)).collect();
    let chosen = if content.is_empty() { words } else { content };
    let mut seen = HashSet::new();
    chosen.into_iter().filter(|w| seen.insert(*w)).map(String::from).collect()
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

    // A vague query ("for", "things") is close to where all queries sit and says little of its own: a meaning match
    // must stand out further, and counts for less, the vaguer it is. Descriptive queries (below 0) are unaffected.
    let vagueness = query
        .filter(|_| std::env::var("RUMMAGE_VAGUENESS").map_or(true, |v| v != "0"))
        .and_then(|q| vectors.query_vagueness(q))
        .unwrap_or(0.0)
        .max(0.0);
    let z_gate = options.z_gate + vagueness;

    // Meaning: every passage on this query's scale; each file keeps its best.
    let mut best: HashMap<i64, (f64, f64, usize)> = HashMap::new(); // file → (z, cosine, row)
    if let Some(query) = query.filter(|q| q.len() == DIM && !vectors.is_empty()) {
        let cosines = vectors.cosines(query);
        // Hubs lose the head start they get from being a little similar to everything.
        let adjusted: Vec<f32> =
            cosines.iter().zip(&vectors.centrality).map(|(&c, &h)| c - (options.hub_penalty as f32) * h).collect();
        let z = standardize(&adjusted);
        if std::env::var_os("RUMMAGE_DEBUG").is_some() {
            eprintln!("query vagueness {:.2}", vectors.query_vagueness(query).unwrap_or(f64::NAN));
        }
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
    let meaning_weight = meaning_weight / (1.0 + vagueness);
    let mut candidates: HashSet<i64> = best.iter().filter(|(_, b)| b.0 >= z_gate).map(|(f, _)| *f).collect();
    candidates.extend(coverage.iter().filter(|(_, share)| **share >= 0.5).map(|(f, _)| *f));
    let mut scored: Vec<(i64, f64)> = candidates
        .into_iter()
        .map(|file| {
            let z = best.get(&file).map_or(0.0, |b| b.0.max(0.0));
            let words = coverage.get(&file).copied().unwrap_or(0.0).powf(1.5);
            (file, meaning_weight * z + words_weight * words)
        })
        .collect();
    let keyword_rank: HashMap<i64, (usize, i64)> =
        keyword.iter().enumerate().map(|(i, (file, segment))| (*file, (i + 1, *segment))).collect();
    // Ties (common with word-only matches, which all score the same) go to the better keyword match.
    let rank_of = |file: &i64| keyword_rank.get(file).map_or(usize::MAX, |r| r.0);
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(rank_of(&a.0).cmp(&rank_of(&b.0))).then(a.0.cmp(&b.0)));
    let mut hits = Vec::new();
    let any_word = patterns.join(" OR ");
    let mut word_check = store.db.prepare("SELECT 1 FROM words WHERE words MATCH ? AND rowid = ?")?;
    let mut has_words =
        |segment: i64| -> Result<bool> { Ok(!any_word.is_empty() && word_check.exists(params![any_word, segment])?) };
    let mut statement = store.db.prepare(
        "SELECT f.path, f.kind, s.loc, s.excerpt FROM segments s JOIN files f ON f.id = s.file_id WHERE s.id = ?",
    )?;
    for (file, score) in scored.into_iter().take(options.limit) {
        let meaning = best.get(&file);
        let words = keyword_rank.get(&file);
        // Point at the best meaning passage if it stands out, else at the one the words matched. When the file also
        // has the words but the meaning passage holds none of them, the words' passage is the better place to land:
        // the reader sees what they typed, marked.
        let segment = match (meaning, words) {
            (Some(m), Some(w)) if m.0 >= z_gate => {
                let meaning_segment = vectors.row_segment[m.2];
                if meaning_segment == w.1 || has_words(meaning_segment)? { meaning_segment } else { w.1 }
            }
            (Some(m), None) => vectors.row_segment[m.2],
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
    fn stopword_only_queries_keep_their_words() {
        assert_eq!(terms("for"), vec!["for"]);
        assert_eq!(terms("How to"), vec!["how", "to"]);
        assert_eq!(terms("how to bake"), vec!["bake"]);
        assert!(terms("a").is_empty());
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

    fn unit(i: usize, j: usize) -> Vec<f32> {
        // Half along axis i, half along axis j, normalized.
        let mut v = vec![0.0; DIM];
        v[i] += std::f32::consts::FRAC_1_SQRT_2;
        v[j] += std::f32::consts::FRAC_1_SQRT_2;
        v
    }

    /// cargo test --release -- --ignored --nocapture scoring_speed
    #[test]
    #[ignore]
    fn scoring_speed() {
        let rows = 100_000;
        let mut seed = 1u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let matrix: Vec<f16> = (0..rows * DIM).map(|_| f16::from_f32(next() * 0.07)).collect();
        let vectors = Vectors {
            matrix,
            row_segment: vec![0; rows],
            row_file: vec![0; rows],
            file_kind: HashMap::new(),
            query_centre: None,
            centrality: Vec::new(),
        };
        let query: Vec<f32> = (0..DIM).map(|_| next() * 0.07).collect();
        let started = std::time::Instant::now();
        for _ in 0..10 {
            std::hint::black_box(vectors.cosines(&query));
        }
        eprintln!(
            "{rows} rows: {:.1} ms a query, {} MB",
            started.elapsed().as_secs_f64() * 100.0,
            rows * DIM * 2 >> 20
        );
    }

    #[test]
    fn query_centre_and_vagueness() {
        // Typical queries all share axis 0 and differ on another axis.
        let typical: Vec<Vec<f32>> = (1..=4).map(|j| unit(0, j)).collect();
        let centre = QueryCentre::from_vectors(&typical);
        assert!((centre.centre[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
        let mut vectors = Vectors {
            matrix: Vec::new(),
            row_segment: Vec::new(),
            row_file: Vec::new(),
            file_kind: HashMap::new(),
            query_centre: None,
            centrality: Vec::new(),
        };
        assert_eq!(vectors.query_vagueness(&unit(0, 1)), None, "unknown until the centre is set");
        // Two rows: a hub along the shared axis, and a passage off on its own.
        let mut hub = vec![0.0; DIM];
        hub[0] = 1.0;
        let own = unit(5, 6);
        vectors.matrix = [hub.clone(), own].concat().into_iter().map(f16::from_f32).collect();
        vectors.row_segment = vec![1, 2];
        vectors.row_file = vec![1, 2];
        vectors.set_query_centre(centre);
        assert!(vectors.centrality[0] > 0.5 && vectors.centrality[1].abs() < 1e-6, "{:?}", vectors.centrality);
        // A query right on the shared axis is vaguer than the typical ones; one off on its own is not vague at all.
        assert!(vectors.query_vagueness(&hub).unwrap() > 0.0);
        assert!(vectors.query_vagueness(&unit(7, 8)).unwrap() < 0.0);
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
