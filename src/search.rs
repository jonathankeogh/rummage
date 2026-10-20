//! Hybrid search: meaning (EmbeddingGemma's vectors) plus words (FTS5).
//!
//! Every vector is scored for each query, so the scores' spread comes for free: a passage counts as a meaning match
//! only when it stands out (z ≥ `z_gate`), which also gives an honest "nothing found". Words catch what a vector can't
//! hold exactly: codes, IDs and names. The scoring is DigUp's (MIT), tuned there on its evals.

use crate::embed::DIM;
use crate::grep::GrepHit;
use crate::store::{Kind, Modality, Store};
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
    /// Passage or page number from 1; for sound and video, the moment in seconds.
    pub loc: Option<f64>,
    /// Where that stretch of sound or video ends, in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loc_end: Option<f64>,
    /// The line grep matched, from 1, when that's where the hit points (no passage stood out).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
    /// Grep found the query's text in the file.
    #[serde(default)]
    pub grep: bool,
    pub excerpt: String,
}

pub struct Options {
    pub kinds: Option<HashSet<Kind>>,
    pub limit: usize,
    pub z_gate: f64,
    /// How much holding the query's words counts against meaning; None: `WORD_WEIGHT`, or `CODE_WORD_WEIGHT` for a
    /// search of code alone.
    pub word_weight: Option<f64>,
    /// How much a passage's closeness to the index's centre counts against it (see `Vectors::centrality`).
    pub hub_penalty: f64,
    /// The query is being typed: its last word also matches as the start of longer words ("invo" finds "invoice").
    pub prefix_last: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            kinds: None,
            limit: 10,
            z_gate: 2.0,
            word_weight: None,
            hub_penalty: tuned("HUB_PENALTY", HUB_PENALTY),
            prefix_last: false,
        }
    }
}

/// A ranking weight, overridden by RUMMAGE_<NAME> for evals (scripts/sweep.py); the service reads it when it starts.
pub fn tuned(name: &str, default: f64) -> f64 {
    std::env::var(format!("RUMMAGE_{name}")).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Tuned on DigUp's evals and real folders (see `search`).
pub const HUB_PENALTY: f64 = 1.0;

/// How much holding the query's words counts (DigUp's tuned 2.0; checked again 2026-10-19 with scripts/sweep.py: 1 or
/// less lost names and codes among pictures and recordings that look like them, 3 changed nothing).
pub const WORD_WEIGHT: f64 = 2.0;
/// Code's words count for less: they're everywhere in code, and a README or Makefile that mentions a few of them beat
/// the file that does what the query says (DigUp's 1.0; on its code eval R@1 0.88 → 0.97 here, 0.5 no better).
pub const CODE_WORD_WEIGHT: f64 = 1.0;

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
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    /// What each row's vector was made from: pictures and text get their own baselines.
    row_modality: Vec<Modality>,
    file_kind: HashMap<i64, Kind>,
    /// When each file last changed, and each indexed file's id by path (to join grep's hits).
    file_mtime: HashMap<i64, i64>,
    file_ids: HashMap<String, i64>,
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
            row_modality: Vec::new(),
            file_kind: HashMap::new(),
            file_mtime: HashMap::new(),
            file_ids: HashMap::new(),
            query_centre: None,
            centrality: Vec::new(),
        };
        let mut statement = store.db.prepare("SELECT id, kind, mtime, path FROM files")?;
        for row in
            statement.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(2)?, r.get::<_, String>(3)?)))?
        {
            let (id, mtime, path) = row?;
            vectors.file_mtime.insert(id, mtime);
            vectors.file_ids.insert(path, id);
        }
        for row in statement.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))? {
            let (id, kind) = row?;
            if let Some(kind) = Kind::parse(&kind) {
                vectors.file_kind.insert(id, kind);
            }
        }
        let mut statement = store.db.prepare("SELECT id, file_id, vector, modality FROM segments ORDER BY id")?;
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
            vectors.row_modality.push(Modality::parse(&row.get::<_, String>(3)?));
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
/// The spread of (hub-adjusted) cosines in a real index of thousands of passages, query after query: 0.020 to 0.035,
/// mostly about 0.025 (measured on a 2,883-passage index, 2026-10-19).
const PRIOR_SPREAD: f64 = 0.025;
/// How many passages' worth the typical spread counts for.
const PRIOR_WEIGHT: f64 = 50.0;

/// Subtracts each modality's median from its rows' scores (modalities under 20 rows take the median of all).
fn centre_by_modality(scores: &mut [f32], modality: &[Modality]) {
    let median = |mut values: Vec<f32>| -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        let middle = values.len() / 2;
        *values.select_nth_unstable_by(middle, |a, b| a.total_cmp(b)).1
    };
    if !modality.iter().any(|&m| m != Modality::Text) {
        return; // all text: nothing to line up (the z-score centres it anyway)
    }
    let all = median(scores.to_vec());
    let centres: HashMap<Modality, f32> = [Modality::Text, Modality::Image, Modality::Audio]
        .into_iter()
        .map(|m| {
            let rows: Vec<f32> = scores.iter().zip(modality).filter(|(_, r)| **r == m).map(|(&s, _)| s).collect();
            (m, if rows.len() >= 20 { median(rows) } else { all })
        })
        .collect();
    for (score, m) in scores.iter_mut().zip(modality) {
        *score -= centres[m];
    }
}

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
    let measured = median(&mut deviations) * 1.4826;
    // Few passages measure the spread badly (from 0.020 to 0.047 for 8 passages, query to query), and a clear best
    // match can then fail to stand out. So the measured spread is blended with a typical one, weighing in as if from
    // PRIOR_WEIGHT passages: a small index leans on the typical spread, a big one on its own.
    let n = scores.len() as f64;
    let deviation = ((n * measured + PRIOR_WEIGHT * PRIOR_SPREAD) / (n + PRIOR_WEIGHT)).max(0.02);
    if std::env::var_os("RUMMAGE_DEBUG").is_some() {
        eprintln!("spread {deviation:.4} (measured {measured:.4}) over {} passages", scores.len());
    }
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
    // A code ("inv-20931", "v1.2.3") or a name from code ("max_retries", "app.config") stays whole, so it matches as a
    // phrase, as it's written; other words split at punctuation ("pet-friendly" is two words).
    let mut words: Vec<&str> = Vec::new();
    for chunk in lower.split_whitespace() {
        let chunk = chunk.trim_matches(|c: char| !c.is_alphanumeric());
        let joined = chunk.contains(|c: char| !c.is_alphanumeric());
        let code = chunk.contains(|c: char| c.is_numeric()) || chunk.contains(['_', '.', '/']);
        if joined && code && chunk.chars().all(|c| c.is_alphanumeric() || "-_./:".contains(c)) {
            words.push(chunk);
        } else {
            words.extend(chunk.split(|c: char| !c.is_alphanumeric()).filter(|w| w.chars().count() >= 2));
        }
    }
    let content: Vec<&str> = words.iter().copied().filter(|w| !STOPWORDS.contains(w)).collect();
    let chosen = if content.is_empty() { words } else { content };
    let mut seen = HashSet::new();
    chosen.into_iter().filter(|w| seen.insert(*w)).map(String::from).collect()
}

/// FTS5 patterns for the query's words, each a phrase as written. A word in no file as typed matches as the start of
/// longer words, as the last one does while it's being typed (DigUp's): "grok" finds "Grok4" (stemming only covers
/// endings). A word that is in the index stays whole, so "cat" doesn't bring in "category"; so does a word of two
/// letters, the start of too many words to stand for one, unless it's the one being typed.
fn word_patterns(store: &Store, terms: &[String], typing: bool) -> Result<Vec<String>> {
    let mut indexed = store.db.prepare("SELECT 1 FROM words WHERE words MATCH ? LIMIT 1")?;
    let mut patterns = Vec::new();
    for (i, term) in terms.iter().enumerate() {
        let phrase = format!("\"{}\"", term.replace('"', ""));
        let whole_word = term.chars().all(char::is_alphanumeric);
        let being_typed = typing && i + 1 == terms.len();
        let prefix = whole_word && (being_typed || (term.chars().count() >= 3 && !indexed.exists([&phrase])?));
        patterns.push(if prefix { format!("{phrase}*") } else { phrase });
    }
    Ok(patterns)
}

/// Kinds a query asks for ("a photo of …", "the screenshot with …", "… in a pdf"): a nudge, not a filter.
pub fn kind_hints(text: &str) -> HashSet<Kind> {
    let mut hints = HashSet::new();
    for word in text.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
        let kind = match word {
            "photo" | "photos" | "picture" | "pictures" | "image" | "images" | "pic" | "pics" => Kind::Image,
            "screenshot" | "screenshots" | "screengrab" => Kind::Screenshot,
            "pdf" | "pdfs" | "paper" | "papers" => Kind::Pdf,
            "document" | "documents" | "note" | "notes" | "doc" | "docs" => Kind::Doc,
            "video" | "videos" | "clip" | "clips" | "movie" | "film" | "footage" => Kind::Video,
            "audio" | "podcast" | "recording" | "recordings" | "song" | "voice" | "sound" => Kind::Audio,
            _ => continue,
        };
        hints.insert(kind);
    }
    hints
}

/// How much a kind the query asks for counts (DigUp's +1).
const KIND_HINT_WEIGHT: f64 = 1.0;

/// Every word is a code: it has a digit (and letters or 4+ characters: an invoice number, 0x80070005), joins parts
/// with _ . / (max_retries, app.config), is camelCase (refreshAccessToken) or is a capitalised identifier of 4+
/// letters (ECONNRESET). A hyphenated word (payments-gw) can be part of a lookup, but alone it's as likely English
/// (pet-friendly). Words count for more.
fn is_lookup(text: &str) -> bool {
    let words: Vec<&str> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty() && !STOPWORDS.contains(&w.to_lowercase().as_str()))
        .collect();
    words.iter().all(|w| code(w).is_some()) && words.iter().any(|w| code(w) == Some(true))
}

/// Some(true) for a code, Some(false) for a hyphenated word, None for neither.
fn code(word: &str) -> Option<bool> {
    let chars: Vec<char> = word.chars().collect();
    let digit = chars.iter().any(|c| c.is_numeric());
    let letter = chars.iter().any(|c| c.is_alphabetic());
    let joined_by = |marks: &str| {
        chars.windows(3).any(|w| w[0].is_alphanumeric() && marks.contains(w[1]) && w[2].is_alphanumeric())
    };
    let camel = chars.windows(2).any(|w| w[0].is_lowercase() && w[1].is_uppercase());
    let capitals = chars.len() >= 4 && chars.iter().all(|c| c.is_uppercase() || c.is_numeric() || *c == '_');
    if (digit && (letter || chars.len() >= 4)) || (letter && joined_by("_./")) || camel || capitals {
        Some(true)
    } else {
        (letter && joined_by("-")).then_some(false)
    }
}

/// "code: retry with backoff" → ("retry with backoff", true): a search of code, and nothing else.
pub fn split_code_prefix(text: &str) -> (&str, bool) {
    let trimmed = text.trim_start();
    match trimmed.get(..5) {
        Some(prefix) if prefix.eq_ignore_ascii_case("code:") => (trimmed[5..].trim(), true),
        _ => (text.trim(), false),
    }
}

/// The meaning and word evidence for one query, worked out once: ranked now (`rank` with no grep hits), and again
/// when grep's hits come in.
pub struct Prepared {
    /// file → (z, cosine, row) of its best passage.
    best: HashMap<i64, (f64, f64, usize)>,
    /// (file, segment) of the best keyword matches, best first, one per file.
    keyword: Vec<(i64, i64)>,
    /// file → share of the query's words found in it.
    coverage: HashMap<i64, f64>,
    z_gate: f64,
    meaning_weight: f64,
    literal_weight: f64,
    /// The query's words as one FTS query (empty: none).
    any_word: String,
    kinds: Option<HashSet<Kind>>,
    /// Kinds the query names.
    hints: HashSet<Kind>,
    limit: usize,
}

/// Scores and ranks in one go, without grep.
pub fn search(
    store: &Store,
    vectors: &Vectors,
    text: &str,
    query: Option<&[f32]>,
    options: &Options,
) -> Result<Vec<Hit>> {
    let prepared = prepare(store, vectors, text, query, options)?;
    rank(store, vectors, &prepared, &[], now())
}

/// Seconds since 1970.
pub fn now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

/// Meaning (every vector scored, hubs penalised, each file's best passage) and words (the keyword index).
pub fn prepare(
    store: &Store,
    vectors: &Vectors,
    text: &str,
    query: Option<&[f32]>,
    options: &Options,
) -> Result<Prepared> {
    let allowed = |file: i64| match (&options.kinds, vectors.file_kind.get(&file)) {
        (None, _) => true,
        (Some(kinds), Some(kind)) => kinds.contains(kind),
        (Some(_), None) => false,
    };

    // A vague query ("for", "things") is close to where all queries sit and says little of its own: a meaning match
    // must stand out further, and counts for less, the vaguer it is. Descriptive queries (below 0) are unaffected.
    let vagueness = tuned("VAGUENESS", 1.0) * query.and_then(|q| vectors.query_vagueness(q)).unwrap_or(0.0).max(0.0);
    let z_gate = options.z_gate + vagueness;

    // Meaning: every passage on this query's scale; each file keeps its best.
    let mut best: HashMap<i64, (f64, f64, usize)> = HashMap::new();
    if let Some(query) = query.filter(|q| q.len() == DIM && !vectors.is_empty()) {
        let cosines = vectors.cosines(query);
        // Hubs lose the head start they get from being a little similar to everything.
        let mut adjusted: Vec<f32> =
            cosines.iter().zip(&vectors.centrality).map(|(&c, &h)| c - (options.hub_penalty as f32) * h).collect();
        // A text query sits lower against pictures than against text: each modality measured from its own median
        // (DigUp's per-modality means), once it has enough rows to say.
        centre_by_modality(&mut adjusted, &vectors.row_modality);
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
    let patterns = word_patterns(store, &terms, options.prefix_last && text.ends_with(char::is_alphanumeric))?;
    let mut keyword: Vec<(i64, i64)> = Vec::new();
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
    // An exact lookup (codes, numbers) leans on literal matches.
    let code_only = options.kinds.as_ref().is_some_and(|kinds| kinds.len() == 1 && kinds.contains(&Kind::Code));
    let word_weight = options.word_weight.unwrap_or_else(|| {
        if code_only { tuned("CODE_WORD_WEIGHT", CODE_WORD_WEIGHT) } else { tuned("WORD_WEIGHT", WORD_WEIGHT) }
    });
    let (meaning_weight, literal_weight) = if is_lookup(text) { (0.5, word_weight * 2.0) } else { (1.0, word_weight) };
    Ok(Prepared {
        best,
        keyword,
        coverage,
        z_gate,
        meaning_weight: meaning_weight / (1.0 + vagueness),
        literal_weight,
        any_word: patterns.join(" OR "),
        kinds: options.kinds.clone(),
        hints: kind_hints(text),
        limit: options.limit,
    })
}

/// How much being recent counts, at most (see `rank`).
const RECENCY_WEIGHT: f64 = 0.5;
/// What a meaning match below the gate still counts for, per unit of z.
const WEAK_MEANING: f64 = 0.5;
/// Recency halves every this many days.
const RECENCY_HALF_LIFE_DAYS: f64 = 30.0;

/// Ranks files on one score:
///
/// ```text
/// score = meaning + literal + recency
/// meaning = z if z stands out (≥ the gate), else max(z, 0) / 2       × 1 (0.5 for exact lookups, less when vague)
/// literal = max(word coverage^1.5, 1 if grep matched)                 × 2 (4 for exact lookups)
/// recency = 0.5^(days since it changed / 30)                          × 0.5
/// ```
///
/// Literal takes the larger of the keyword index and grep, not their sum: they usually find the same words. A file
/// makes the list when its meaning stands out, it holds half the query's words, or grep matched it.
pub fn rank(store: &Store, vectors: &Vectors, prepared: &Prepared, grep: &[GrepHit], now: i64) -> Result<Vec<Hit>> {
    let p = prepared;
    // Grep hits on indexed files join their file; the rest stand alone (keyed by a negative number).
    let grep: Vec<&GrepHit> =
        grep.iter().filter(|g| p.kinds.as_ref().is_none_or(|kinds| kinds.contains(&g.kind))).collect();
    let mut grep_of: HashMap<i64, &GrepHit> = HashMap::new();
    for (i, hit) in grep.iter().enumerate() {
        let file = vectors.file_ids.get(hit.path.to_string_lossy().as_ref()).copied().unwrap_or(-1 - i as i64);
        grep_of.entry(file).or_insert(hit);
    }
    let mut candidates: HashSet<i64> = p.best.iter().filter(|(_, b)| b.0 >= p.z_gate).map(|(f, _)| *f).collect();
    candidates.extend(p.coverage.iter().filter(|(_, share)| **share >= 0.5).map(|(f, _)| *f));
    candidates.extend(grep_of.keys());

    let (recency_weight, weak, hint_weight) = (
        tuned("RECENCY_WEIGHT", RECENCY_WEIGHT),
        tuned("WEAK_MEANING", WEAK_MEANING),
        tuned("KIND_HINT", KIND_HINT_WEIGHT),
    );
    let recency = |mtime: i64| {
        let days = (now - mtime).max(0) as f64 / 86_400.0;
        recency_weight * 0.5f64.powf(days / RECENCY_HALF_LIFE_DAYS)
    };
    let mut scored: Vec<(i64, f64)> = candidates
        .into_iter()
        .map(|file| {
            let z = p.best.get(&file).map_or(0.0, |b| b.0);
            let meaning = if z >= p.z_gate { z } else { z.max(0.0) * weak };
            let words = p.coverage.get(&file).copied().unwrap_or(0.0).powf(1.5);
            let literal = if grep_of.contains_key(&file) { 1.0f64.max(words) } else { words };
            let mtime = vectors.file_mtime.get(&file).copied().or(grep_of.get(&file).map(|g| g.mtime)).unwrap_or(0);
            let kind = vectors.file_kind.get(&file).copied().or(grep_of.get(&file).map(|g| g.kind));
            let hinted = if kind.is_some_and(|k| p.hints.contains(&k)) { hint_weight } else { 0.0 };
            (file, p.meaning_weight * meaning + p.literal_weight * literal + recency(mtime) + hinted)
        })
        .collect();
    let keyword_rank: HashMap<i64, (usize, i64)> =
        p.keyword.iter().enumerate().map(|(i, (file, segment))| (*file, (i + 1, *segment))).collect();
    // Ties (common with word-only matches) go to the better keyword match.
    let rank_of = |file: &i64| keyword_rank.get(file).map_or(usize::MAX, |r| r.0);
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(rank_of(&a.0).cmp(&rank_of(&b.0))).then(a.0.cmp(&b.0)));

    let mut hits = Vec::new();
    let mut word_check = store.db.prepare("SELECT 1 FROM words WHERE words MATCH ? AND rowid = ?")?;
    let mut has_words = |segment: i64| -> Result<bool> {
        Ok(!p.any_word.is_empty() && word_check.exists(params![p.any_word, segment])?)
    };
    let mut statement = store.db.prepare(
        "SELECT f.path, f.kind, s.loc, s.excerpt, s.loc_end FROM segments s JOIN files f ON f.id = s.file_id WHERE s.id = ?",
    )?;
    for (file, score) in scored.into_iter().take(p.limit) {
        let meaning = p.best.get(&file);
        let words = keyword_rank.get(&file);
        let grepped = grep_of.get(&file);
        // Point at the best meaning passage if it stands out, else at the one the words matched. When the file also
        // has the words but the meaning passage holds none of them, the words' passage is the better place to land:
        // the reader sees what they typed, marked. With neither, at the line grep found.
        let segment = match (meaning, words) {
            (Some(m), Some(w)) if m.0 >= p.z_gate => {
                let meaning_segment = vectors.row_segment[m.2];
                Some(if meaning_segment == w.1 || has_words(meaning_segment)? { meaning_segment } else { w.1 })
            }
            (Some(m), None) if m.0 >= p.z_gate || grepped.is_none() => Some(vectors.row_segment[m.2]),
            (_, Some(w)) => Some(w.1),
            _ => None,
        };
        let hit = match (segment, grepped) {
            (Some(segment), _) => {
                let row = statement.query_row(params![segment], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, Option<f64>>(2)?,
                        r.get::<_, String>(3)?,
                        r.get::<_, Option<f64>>(4)?,
                    ))
                })?;
                Hit {
                    path: row.0,
                    kind: Kind::parse(&row.1).unwrap_or(Kind::Doc),
                    score,
                    z: meaning.map(|m| m.0),
                    cosine: meaning.map(|m| m.1),
                    keyword_rank: words.map(|w| w.0),
                    loc: row.2,
                    loc_end: row.4,
                    line: None,
                    grep: grepped.is_some(),
                    excerpt: row.3,
                }
            }
            (None, Some(g)) => Hit {
                path: g.path.to_string_lossy().into_owned(),
                kind: g.kind,
                score,
                z: meaning.map(|m| m.0),
                cosine: meaning.map(|m| m.1),
                keyword_rank: None,
                loc: None,
                loc_end: None,
                line: Some(g.line),
                grep: true,
                excerpt: g.text.clone(),
            },
            (None, None) => continue,
        };
        hits.push(hit);
    }
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_match_as_written_or_as_the_start_of_longer_ones() {
        let dir = std::env::temp_dir().join(format!("rummage-patterns-{}", std::process::id()));
        let store = Store::open(&dir).unwrap();
        let file = store.upsert_pending(std::path::Path::new("/notes/pets.md"), Kind::Doc, 1, 1).unwrap();
        store
            .add_segment(file, 1.0, "pets.md", "The cat sleeps in the category box.", "", None, Modality::Text)
            .unwrap();
        let patterns = |words: &[&str], typing| {
            word_patterns(&store, &words.iter().map(|w| w.to_string()).collect::<Vec<_>>(), typing).unwrap()
        };
        // In the index as typed: whole, so "cat" doesn't bring in "category".
        assert_eq!(patterns(&["cat"], false), ["\"cat\""]);
        // In no file: the start of longer words; but not two letters.
        assert_eq!(patterns(&["categ", "ca"], false), ["\"categ\"*", "\"ca\""]);
        // Being typed: the last word is a start, whatever its length.
        assert_eq!(patterns(&["cat", "ca"], true), ["\"cat\"", "\"ca\"*"]);
        // Codes stay phrases.
        assert_eq!(patterns(&["inv-20931"], true), ["\"inv-20931\""]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

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
    fn codes_stay_whole() {
        assert_eq!(terms("invoice INV-20931?"), vec!["invoice", "inv-20931"]);
        assert!(is_lookup("INV-20931"));
        assert_eq!(terms("well-known pet-friendly"), vec!["well", "known", "pet", "friendly"]);
        assert_eq!(terms("where is max_retries set"), vec!["max_retries", "set"]);
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
        assert!(is_lookup("INV20931"));
        assert!(is_lookup("0x80070005 20931"));
        assert!(is_lookup("payments-gw ECONNRESET"));
        assert!(is_lookup("refreshAccessToken"));
        assert!(is_lookup("the max_retries"));
        assert!(!is_lookup("error 0x80070005"));
        assert!(!is_lookup("a dog on the beach"));
        assert!(!is_lookup("NASA budget"));
        assert!(!is_lookup("well-known Lisbon"));
        assert!(!is_lookup("Priya"));
        assert!(!is_lookup("pet-friendly"));
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
            row_modality: vec![Modality::Text; rows],
            file_kind: HashMap::new(),
            file_mtime: HashMap::new(),
            file_ids: HashMap::new(),
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
            row_modality: Vec::new(),
            file_kind: HashMap::new(),
            file_mtime: HashMap::new(),
            file_ids: HashMap::new(),
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
