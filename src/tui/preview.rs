//! The right-hand pane: the matched passage of a file, with the query's words marked.
//!
//! `Preview` is an enum so other kinds (image, PDF page, video frame) can be added next to `Text` and `Code`.

use crate::search::Hit;
use crate::store::Kind;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use std::io::Read;
use std::path::Path;
use std::sync::OnceLock;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Style as SynStyle, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

/// Passages are ~1800 characters and start 1600 apart (200 overlap).
pub const PASSAGE_STEP: usize = 1600;
const PASSAGE_LEN: usize = 1800;

/// What to show for the selected hit.
#[derive(Debug, Clone, PartialEq)]
pub enum Preview {
    /// A passage of prose.
    Text(Vec<Line<'static>>),
    /// A passage of source code, syntax highlighted.
    Code(Vec<Line<'static>>),
    /// The file can't be previewed; say why.
    Message(String),
}

impl Preview {
    /// Build the preview for `hit`, marking `terms` (see `query_terms`).
    pub fn load(hit: &Hit, terms: &[String]) -> Preview {
        let path = Path::new(&hit.path);
        let loc = hit.loc.map(|l| l.round().max(1.0) as usize).unwrap_or(1);
        let text = match read_prefix(path, loc) {
            Ok(Some(text)) => text,
            Ok(None) => return Preview::Message("binary file".into()),
            Err(_) if !hit.excerpt.is_empty() => hit.excerpt.clone(),
            Err(e) => return Preview::Message(format!("can't read file: {e}")),
        };
        let passage = passage(&text, loc);
        match hit.kind {
            Kind::Code => Preview::Code(code_lines(&passage, path, terms)),
            Kind::Doc => Preview::Text(
                passage
                    .lines()
                    .map(|l| Line::from(overlay(vec![(Style::default(), l.replace('\t', "    "))], terms)))
                    .collect(),
            ),
        }
    }
}

/// The lower-cased words of a query worth marking (a leading "code:" is not one).
pub fn query_terms(query: &str) -> Vec<String> {
    let query = query.trim_start();
    let query = query.strip_prefix("code:").unwrap_or(query);
    let mut terms: Vec<String> = Vec::new();
    for word in query.split(|c: char| !c.is_alphanumeric()) {
        let word = word.to_lowercase();
        if word.chars().count() >= 2 && !terms.contains(&word) {
            terms.push(word);
        }
    }
    terms
}

/// Read enough of a file to reach passage `loc`. `None` for binary files.
fn read_prefix(path: &Path, loc: usize) -> std::io::Result<Option<String>> {
    let limit = (loc * PASSAGE_STEP + PASSAGE_LEN + 1000) * 4;
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.take(limit as u64).read_to_end(&mut bytes)?;
    if bytes.contains(&0) {
        return Ok(None);
    }
    Ok(Some(String::from_utf8_lossy(&bytes).into_owned()))
}

/// Byte offset where passage `loc` (from 1) starts: about (loc-1)*1600 characters in, moved back to a line start.
pub fn passage_start(text: &str, loc: usize) -> usize {
    let chars = loc.saturating_sub(1) * PASSAGE_STEP;
    let Some((at, _)) = text.char_indices().nth(chars) else { return 0 };
    text[..at].rfind('\n').map_or(0, |i| i + 1)
}

/// The text of passage `loc`: from its start for ~1800 characters, ending at a line end.
pub fn passage(text: &str, loc: usize) -> String {
    let rest = &text[passage_start(text, loc)..];
    match rest.char_indices().nth(PASSAGE_LEN) {
        None => rest.to_string(),
        Some((end, _)) => {
            let cut = &rest[..end];
            cut[..cut.rfind('\n').map_or(end, |i| i + 1)].to_string()
        }
    }
}

/// Byte ranges of `line` where any of `terms` (lower-case) occurs, ignoring case; overlapping ranges are merged.
pub fn find_matches(line: &str, terms: &[String]) -> Vec<(usize, usize)> {
    // One lower-case char per char keeps byte offsets of the original line.
    let chars: Vec<(usize, char, usize)> =
        line.char_indices().map(|(i, c)| (i, c.to_lowercase().next().unwrap_or(c), c.len_utf8())).collect();
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for term in terms {
        let term: Vec<char> = term.chars().collect();
        if term.is_empty() || term.len() > chars.len() {
            continue;
        }
        for start in 0..=chars.len() - term.len() {
            if chars[start..start + term.len()].iter().zip(&term).all(|(a, b)| a.1 == *b) {
                let last = chars[start + term.len() - 1];
                ranges.push((chars[start].0, last.0 + last.2));
            }
        }
    }
    ranges.sort();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (s, e) in ranges {
        match merged.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => merged.push((s, e)),
        }
    }
    merged
}

fn mark(style: Style) -> Style {
    style.fg(Color::Black).bg(Color::Yellow).add_modifier(Modifier::BOLD)
}

/// Split styled segments of one line at the matches of `terms` and mark the matches.
pub fn overlay(segments: Vec<(Style, String)>, terms: &[String]) -> Vec<Span<'static>> {
    let line: String = segments.iter().map(|(_, s)| s.as_str()).collect();
    let matches = find_matches(&line, terms);
    let mut spans = Vec::new();
    let mut offset = 0;
    for (style, text) in segments {
        let (seg_start, seg_end) = (offset, offset + text.len());
        offset = seg_end;
        let mut at = seg_start;
        for &(s, e) in matches.iter().filter(|&&(s, e)| e > seg_start && s < seg_end) {
            let (s, e) = (s.max(seg_start), e.min(seg_end));
            if s > at {
                spans.push(Span::styled(text[at - seg_start..s - seg_start].to_string(), style));
            }
            spans.push(Span::styled(text[s - seg_start..e - seg_start].to_string(), mark(style)));
            at = e;
        }
        if at < seg_end {
            spans.push(Span::styled(text[at - seg_start..].to_string(), style));
        }
    }
    spans
}

fn syntaxes() -> &'static (SyntaxSet, ThemeSet) {
    static SETS: OnceLock<(SyntaxSet, ThemeSet)> = OnceLock::new();
    SETS.get_or_init(|| (SyntaxSet::load_defaults_newlines(), ThemeSet::load_defaults()))
}

fn code_lines(passage: &str, path: &Path, terms: &[String]) -> Vec<Line<'static>> {
    let (syntax_set, themes) = syntaxes();
    let syntax = path
        .extension()
        .and_then(|e| e.to_str())
        .and_then(|e| syntax_set.find_syntax_by_extension(e))
        .unwrap_or_else(|| syntax_set.find_syntax_plain_text());
    let mut highlighter = HighlightLines::new(syntax, &themes.themes["base16-ocean.dark"]);
    LinesWithEndings::from(passage)
        .map(|line| {
            let line = line.replace('\t', "    ");
            let plain = || vec![(Style::default(), line.trim_end_matches('\n').to_string())];
            let segments: Vec<(Style, String)> = match highlighter.highlight_line(&line, syntax_set) {
                Ok(ranges) => {
                    ranges.into_iter().map(|(s, t)| (syn_style(s), t.trim_end_matches('\n').to_string())).collect()
                }
                Err(_) => plain(),
            };
            Line::from(overlay(segments, terms))
        })
        .collect()
}

fn syn_style(s: SynStyle) -> Style {
    Style::default().fg(Color::Rgb(s.foreground.r, s.foreground.g, s.foreground.b))
}

/// Why a hit matched: "meaning z 3.1 · words #2", with whichever parts apply.
pub fn why(hit: &Hit) -> String {
    let mut parts = Vec::new();
    if let Some(z) = hit.z {
        parts.push(format!("meaning z {z:.1}"));
    }
    if let Some(rank) = hit.keyword_rank {
        parts.push(format!("words #{rank}"));
    }
    parts.join(" · ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passage_start_snaps_to_line_start() {
        let text = format!("{}\nsecond line\n{}", "a".repeat(1590), "b".repeat(100));
        assert_eq!(passage_start(&text, 1), 0);
        // char 1600 falls inside "second line"; snap back to the start of that line
        assert_eq!(passage_start(&text, 2), 1591);
        assert_eq!(passage_start("short", 5), 0);
    }

    #[test]
    fn passage_ends_at_line_end() {
        let text = "line of words\n".repeat(300);
        let p = passage(&text, 1);
        assert!(p.ends_with('\n') && p.chars().count() <= PASSAGE_LEN);
        assert_eq!(passage("tiny", 1), "tiny");
    }

    #[test]
    fn terms_drop_code_prefix_and_short_words() {
        assert_eq!(query_terms("code: Retry a failed-call, retry"), vec!["retry", "failed", "call"]);
    }

    #[test]
    fn matches_ignore_case_and_merge() {
        let terms = query_terms("dog beach");
        assert_eq!(find_matches("The DOG on the Beach", &terms), vec![(4, 7), (15, 20)]);
        assert_eq!(find_matches("abcd", &["ab".into(), "bc".into()]), vec![(0, 3)]);
        // multi-byte text keeps byte offsets of the original
        assert_eq!(find_matches("café Dog", &["dog".into()]), vec![(6, 9)]);
    }

    #[test]
    fn overlay_marks_only_matches() {
        let spans = overlay(vec![(Style::default(), "a dog ran".into())], &["dog".into()]);
        let texts: Vec<&str> = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(texts, vec!["a ", "dog", " ran"]);
        assert_eq!(spans[1].style, mark(Style::default()));
        // a match spanning two styled segments is marked in both
        let spans = overlay(vec![(Style::default(), "do".into()), (Style::default(), "g!".into())], &["dog".into()]);
        let texts: Vec<&str> = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(texts, vec!["do", "g", "!"]);
    }

    #[test]
    fn why_formats_both_parts() {
        let hit = Hit {
            path: "x".into(),
            kind: Kind::Doc,
            score: 1.0,
            z: Some(3.14),
            cosine: None,
            keyword_rank: Some(2),
            loc: None,
            excerpt: String::new(),
        };
        assert_eq!(why(&hit), "meaning z 3.1 · words #2");
        assert_eq!(why(&Hit { z: None, keyword_rank: None, ..hit }), "");
    }
}
