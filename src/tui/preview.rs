//! The right-hand pane: the file around the matched passage, with the query's words marked, opening at the passage.
//!
//! `Preview` is an enum so other kinds (image, PDF page, video frame) can be added next to `Text` and `Code`.

use crate::extract;
use crate::search::{self, Hit};
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
/// How much of the file around the passage the pane holds, in lines: enough to scroll through, little enough to
/// highlight at once.
const LINES_BEFORE: usize = 300;
const LINES_AFTER: usize = 1500;
/// Files are read up to this many bytes.
const READ_LIMIT: u64 = 16 << 20;

/// Lines of a file to show, and where the matched passage starts among them.
#[derive(Debug, Clone, PartialEq)]
pub struct Shown {
    pub lines: Vec<Line<'static>>,
    /// Index in `lines` of the passage's first line.
    pub focus: usize,
    /// Index in `lines` of the first marked line from the passage on: the pane opens with it in view. `None` when
    /// nothing is marked (a meaning match with none of the query's words); then the pane opens at the passage.
    pub anchor: Option<usize>,
}

/// What to show for the selected hit.
#[derive(Debug, Clone, PartialEq)]
pub enum Preview {
    /// Prose.
    Text(Shown),
    /// Source code, syntax highlighted.
    Code(Shown),
    /// The file can't be previewed; say why.
    Message(String),
}

impl Preview {
    /// Build the preview for `hit`, marking `terms` (see `query_terms`).
    pub fn load(hit: &Hit, terms: &[String]) -> Preview {
        let path = Path::new(&hit.path);
        let loc = hit.loc.map(|l| l.round().max(1.0) as usize).unwrap_or(1);
        if hit.kind.is_picture() {
            return if hit.excerpt.trim().is_empty() {
                Preview::Message("no words in this picture".into())
            } else {
                Preview::Text(shown(text_lines(&hit.excerpt, terms), 0))
            };
        }
        if hit.kind == Kind::Pdf {
            return match pdf_pages(path, loc, terms) {
                Ok(shown) => Preview::Text(shown),
                Err(e) if !hit.excerpt.is_empty() => {
                    Preview::Text(shown(text_lines(&format!("{}\n\n({e:#})", hit.excerpt), terms), 0))
                }
                Err(e) => Preview::Message(format!("can't read the PDF: {e:#}")),
            };
        }
        // A grep hit may sit deep in a file too big to read whole (it's why grep found it): stream to its line.
        if let (Some(line), false) = (hit.line, is_converted(path, hit.kind)) {
            return match read_lines_around(path, line) {
                Ok((window, focus)) => match hit.kind {
                    Kind::Code => Preview::Code(shown(code_lines(&window, path, terms), focus)),
                    _ => Preview::Text(shown(text_lines(&window, terms), focus)),
                },
                Err(e) => Preview::Message(format!("can't read file: {e:#}")),
            };
        }
        let text = match read_document(path, hit.kind) {
            Ok(Some(text)) => text,
            Ok(None) => return Preview::Message("binary file".into()),
            Err(_) if !hit.excerpt.is_empty() => hit.excerpt.clone(),
            Err(e) => return Preview::Message(format!("can't read file: {e:#}")),
        };
        let (window, focus) = match hit.line {
            Some(line) => window_at_line(&text, line),
            None => window(&text, loc),
        };
        match hit.kind {
            Kind::Code => Preview::Code(shown(code_lines(&window, path, terms), focus)),
            _ => Preview::Text(shown(text_lines(&window, terms), focus)),
        }
    }
}

/// `lines` opening at `focus`, or at the first marked line from there on.
fn shown(lines: Vec<Line<'static>>, focus: usize) -> Shown {
    let anchor = first_marked(&lines[focus.min(lines.len())..]).map(|i| focus + i).or(first_marked(&lines));
    Shown { lines, focus, anchor }
}

/// Prose lines with `terms` marked.
fn text_lines(text: &str, terms: &[String]) -> Vec<Line<'static>> {
    text.lines().map(|l| Line::from(overlay(vec![(Style::default(), l.replace('\t', "    "))], terms))).collect()
}

/// Pages of a PDF around page `page`: a few before, the page, more after, each under a "page N" rule.
fn pdf_pages(path: &Path, page: usize, terms: &[String]) -> anyhow::Result<Shown> {
    const BEFORE: usize = 2;
    const AFTER: usize = 6;
    let count = extract::pdf_page_count(path)?;
    let first = page.saturating_sub(BEFORE).max(1);
    let last = (page + AFTER).min(count).max(first);
    let rule = Style::default().fg(Color::DarkGray);
    let mut lines = Vec::new();
    let mut focus = 0;
    for (i, text) in extract::pdf_pages(path, first, last)?.iter().enumerate() {
        let number = first + i;
        if number == page {
            focus = lines.len();
        }
        lines.push(Line::styled(format!("── page {number} of {count} ──"), rule));
        if text.trim().is_empty() {
            lines.push(Line::styled("(no text on this page)", rule));
        }
        lines.extend(text_lines(&reflow(text.trim_end()), terms));
        lines.push(Line::from(""));
    }
    Ok(shown(lines, focus))
}

/// Joins a PDF's hard-wrapped lines back into paragraphs (a blank line ends one), mending words hyphenated across
/// lines, so the pane can wrap them at its own width.
pub fn reflow(text: &str) -> String {
    let mut out = String::new();
    let mut paragraph = String::new();
    let flush = |paragraph: &mut String, out: &mut String| {
        if !paragraph.is_empty() {
            out.push_str(paragraph);
            out.push('\n');
            paragraph.clear();
        }
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            flush(&mut paragraph, &mut out);
            out.push('\n');
            continue;
        }
        if paragraph.is_empty() {
            paragraph.push_str(line);
        } else if paragraph.ends_with('-') && line.starts_with(|c: char| c.is_lowercase()) {
            paragraph.pop();
            paragraph.push_str(line);
        } else {
            paragraph.push(' ');
            paragraph.push_str(line);
        }
    }
    flush(&mut paragraph, &mut out);
    out
}

/// A file's text as indexing read it: documents converted (docx, odt, html), anything else as text. `None` for
/// binary files.
fn read_document(path: &Path, kind: Kind) -> anyhow::Result<Option<String>> {
    if is_converted(path, kind) {
        return Ok(Some(extract::document_text(path)?));
    }
    Ok(read_text(path)?)
}

/// The query's words to mark: the same words the search matched (a leading "code:" is not one).
pub fn query_terms(query: &str) -> Vec<String> {
    search::terms(search::split_code_prefix(query).0)
}

/// The file's text, or `None` for binary files.
fn read_text(path: &Path) -> std::io::Result<Option<String>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.take(READ_LIMIT).read_to_end(&mut bytes)?;
    if bytes[..bytes.len().min(8192)].contains(&0) {
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

/// The lines of `text` around passage `loc`, and the index among them of the passage's first line.
pub fn window(text: &str, loc: usize) -> (String, usize) {
    let start_line = text[..passage_start(text, loc)].matches('\n').count();
    let first = start_line.saturating_sub(LINES_BEFORE);
    let lines: Vec<&str> = text.lines().skip(first).take(start_line - first + LINES_AFTER).collect();
    (lines.join("\n"), start_line - first)
}

/// The lines of `text` around line `line` (from 1, as grep counts), and the index among them of that line (the last
/// line when the text is shorter).
pub fn window_at_line(text: &str, line: usize) -> (String, usize) {
    let at = line.saturating_sub(1);
    let first = at.saturating_sub(LINES_BEFORE);
    let lines: Vec<&str> = text.lines().skip(first).take(at - first + LINES_AFTER).collect();
    let focus = (at - first).min(lines.len().saturating_sub(1));
    (lines.join("\n"), focus)
}

/// The lines around line `line` of a file of any size, read a line at a time (only the window is kept).
fn read_lines_around(path: &Path, line: usize) -> std::io::Result<(String, usize)> {
    use std::io::BufRead;
    let at = line.saturating_sub(1);
    let first = at.saturating_sub(LINES_BEFORE);
    let mut kept: Vec<String> = Vec::new();
    let reader = std::io::BufReader::new(std::fs::File::open(path)?);
    for (i, bytes) in reader.split(b'\n').enumerate().take(at + LINES_AFTER) {
        let bytes = bytes?;
        if i >= first {
            kept.push(String::from_utf8_lossy(&bytes).trim_end_matches('\r').to_string());
        }
    }
    let focus = (at - first).min(kept.len().saturating_sub(1));
    Ok((kept.join("\n"), focus))
}

/// Read through a converter (docx, odt, html, notebooks) rather than as text.
fn is_converted(path: &Path, kind: Kind) -> bool {
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    ext == "ipynb" || (kind == Kind::Doc && ["docx", "odt", "html", "htm", "xhtml"].contains(&ext.as_str()))
}

/// Byte ranges of `line` where any of `terms` (lower-case) occurs as a whole word, ignoring case; overlapping ranges
/// are merged.
pub fn find_matches(line: &str, terms: &[String]) -> Vec<(usize, usize)> {
    // One lower-case char per char keeps byte offsets of the original line.
    let chars: Vec<(usize, char, usize)> =
        line.char_indices().map(|(i, c)| (i, c.to_lowercase().next().unwrap_or(c), c.len_utf8())).collect();
    let word_char = |i: usize| chars.get(i).is_some_and(|c| c.1.is_alphanumeric());
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for term in terms {
        let term: Vec<char> = term.chars().collect();
        if term.is_empty() || term.len() > chars.len() {
            continue;
        }
        for start in 0..=chars.len() - term.len() {
            let end = start + term.len();
            let whole = (start == 0 || !word_char(start - 1)) && !word_char(end);
            if whole && chars[start..end].iter().zip(&term).all(|(a, b)| a.1 == *b) {
                let last = chars[end - 1];
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

/// Index of the first line with a marked word.
fn first_marked(lines: &[Line]) -> Option<usize> {
    lines.iter().position(|line| line.spans.iter().any(|s| s.style.bg == Some(Color::Yellow)))
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

fn code_lines(text: &str, path: &Path, terms: &[String]) -> Vec<Line<'static>> {
    let (syntax_set, themes) = syntaxes();
    let syntax = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| if e == "ipynb" { "py" } else { e })
        .and_then(|e| syntax_set.find_syntax_by_extension(e))
        .unwrap_or_else(|| syntax_set.find_syntax_plain_text());
    let mut highlighter = HighlightLines::new(syntax, &themes.themes["base16-ocean.dark"]);
    LinesWithEndings::from(text)
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
    match (hit.grep, hit.line) {
        (true, Some(line)) => parts.push(format!("grep, line {line}")),
        (true, None) => parts.push("grep".into()),
        _ => {}
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
    fn window_surrounds_the_passage() {
        let text: String = (0..5000).map(|i| format!("line {i} with some words in it\n")).collect();
        // Passage 3 starts about 3200 characters in: line 3200 / 30 chars a line ≈ 106.
        let (shown, focus) = window(&text, 3);
        let lines: Vec<&str> = shown.lines().collect();
        assert!(lines[0] == "line 0 with some words in it", "starts at the top: fewer than 300 lines before");
        assert!(lines[focus].starts_with("line 10"), "{}", lines[focus]);
        // Far into the file, the window starts 300 lines before the passage.
        let (shown, focus) = window(&text, 60);
        assert_eq!(focus, LINES_BEFORE);
        assert_eq!(shown.lines().count(), LINES_BEFORE + LINES_AFTER);
        assert_eq!(window("tiny", 1), ("tiny".to_string(), 0));
    }

    #[test]
    fn grep_hits_open_at_their_line() {
        let text: String = (1..=1000).map(|i| format!("line {i}\n")).collect();
        let (shown, focus) = window_at_line(&text, 500);
        assert_eq!(shown.lines().nth(focus), Some("line 500"));
        let (shown, focus) = window_at_line(&text, 3);
        assert_eq!((focus, shown.lines().next()), (2, Some("line 1")));
        // A line past the end: the window holds the end, and the focus stays inside it.
        let (shown, focus) = window_at_line(&text, 5000);
        assert!(shown.is_empty() && focus == 0);
        // Streamed from a file of any size.
        let path = std::env::temp_dir().join(format!("rummage-lines-{}.log", std::process::id()));
        std::fs::write(&path, &text).unwrap();
        let (shown, focus) = read_lines_around(&path, 700).unwrap();
        assert_eq!(shown.lines().nth(focus), Some("line 700"));
        assert_eq!(shown.lines().count(), LINES_BEFORE + 301, "300 before, the line, then to the end of the file");
    }

    #[test]
    fn terms_follow_the_search() {
        assert_eq!(query_terms("code: Retry a failed-call, retry"), vec!["retry", "failed", "call"]);
        assert_eq!(query_terms("a dog on the beach"), vec!["dog", "beach"]);
        assert_eq!(query_terms("how"), vec!["how"]);
    }

    #[test]
    fn matches_are_whole_words_ignoring_case() {
        let terms = query_terms("dog beach");
        assert_eq!(find_matches("The DOG on the Beach", &terms), vec![(4, 7), (15, 20)]);
        assert!(find_matches("dogma hotdogs", &terms).is_empty());
        assert_eq!(find_matches("how? show how-to", &["how".into()]), vec![(0, 3), (10, 13)]);
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
    fn reflow_joins_lines_into_paragraphs() {
        let text = "We denote its elements to\nemphasise that they\nare classes.\n\nA new para-\ngraph here.";
        assert_eq!(
            reflow(text),
            "We denote its elements to emphasise that they are classes.\n\nA new paragraph here.\n"
        );
        // A dash before a capital stays: it's punctuation, not a split word.
        assert_eq!(reflow("Projective-\nGeometry"), "Projective- Geometry\n");
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
            line: None,
            grep: false,
            excerpt: String::new(),
        };
        assert_eq!(why(&hit), "meaning z 3.1 · words #2");
        assert_eq!(why(&Hit { z: None, keyword_rank: None, ..hit }), "");
    }
}
