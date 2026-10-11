//! Reads a text file and cuts it into passages for the model.

use anyhow::{Result, bail};
use std::path::Path;

/// About 450 tokens a passage, with a little overlap so a sentence cut in two is still found (DigUp's sizes).
pub const PASSAGE_CHARS: usize = 1800;
pub const OVERLAP_CHARS: usize = 200;
/// A very long document keeps its first passages only (for now): 5,000 passages is about 8 MB of text.
pub const MAX_PASSAGES: usize = 5000;

/// The file's text, or an error for binary files.
pub fn read_text(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    if bytes[..bytes.len().min(8192)].contains(&0) {
        bail!("binary file");
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Splits `text` into ~`size`-character passages that overlap by ~`overlap`, breaking at whitespace.
pub fn passages(text: &str, size: usize, overlap: usize, limit: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut start = 0;
    while start < chars.len() && out.len() < limit {
        let mut end = (start + size).min(chars.len());
        // End on whitespace, unless that would make a tiny passage.
        if end < chars.len() {
            if let Some(space) = (start + size / 2..end).rev().find(|&i| chars[i].is_whitespace()) {
                end = space;
            }
        }
        let passage: String = chars[start..end].iter().collect::<String>().trim().to_string();
        if !passage.is_empty() {
            out.push(passage);
        }
        if end >= chars.len() {
            break;
        }
        // Start the next passage on a word boundary, `overlap` back.
        let mut next = end.saturating_sub(overlap).max(start + 1);
        while next < end && !chars[next - 1].is_whitespace() {
            next += 1;
        }
        start = next;
    }
    out
}

/// One line of at most `limit` characters, for result lists.
pub fn excerpt(text: &str, limit: usize) -> String {
    let line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if line.chars().count() <= limit {
        return line;
    }
    let mut cut: String = line.chars().take(limit).collect();
    if let Some(space) = cut.rfind(' ') {
        cut.truncate(space);
    }
    cut + "…"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passages_cover_the_text_and_overlap() {
        let text = (0..2000).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        let parts = passages(&text, 1800, 200, 100);
        assert!(parts.len() > 3);
        assert!(parts.iter().all(|p| p.chars().count() <= 1800));
        assert!(parts[0].starts_with("w0 "));
        assert!(parts.last().unwrap().ends_with("w1999"));
        // Neighbours share words.
        let last_word = parts[0].split(' ').last().unwrap();
        assert!(parts[1].split(' ').any(|w| w == last_word));
    }

    #[test]
    fn short_text_is_one_passage() {
        assert_eq!(passages("hello there", 1800, 200, 10), vec!["hello there"]);
        assert!(passages("   ", 1800, 200, 10).is_empty());
    }

    #[test]
    fn excerpt_is_one_short_line() {
        assert_eq!(excerpt("a\n b   c", 20), "a b c");
        assert_eq!(excerpt("alpha beta gamma delta", 12), "alpha beta…");
    }
}
