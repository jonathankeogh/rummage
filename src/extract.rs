//! Reads files as text and cuts them into the parts the model embeds: passages for text and documents, pages for
//! PDFs. Documents (docx, odt, html) are converted in Rust; PDFs go through poppler's `pdftotext` and `pdfinfo`,
//! packaged on every Unix (`brew install poppler`, `apt install poppler-utils`).

use crate::store::Kind;
use anyhow::{Context, Result, bail};
use quick_xml::events::Event;
use std::io::Read as _;
use std::ops::Range;
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

/// About 450 tokens a passage, with a little overlap so a sentence cut in two is still found (DigUp's sizes).
pub const PASSAGE_CHARS: usize = 1800;
pub const OVERLAP_CHARS: usize = 200;
/// A very long document keeps its first passages only (for now): 5,000 passages is about 8 MB of text.
pub const MAX_PASSAGES: usize = 5000;
/// PDFs keep their first 5,000 pages (DigUp's limit).
pub const MAX_PAGES: usize = 5000;

/// One part of a file for the model: a passage or a page, numbered from 1.
#[derive(Debug, PartialEq)]
pub struct Part {
    pub loc: usize,
    pub text: String,
}

/// Some of a file's parts, and how many it has in all.
pub struct Parts {
    pub parts: Vec<Part>,
    pub total: usize,
}

/// Parts `range` (from 0) of a file. A PDF's blank pages (scans, until OCR) are counted but have no part.
pub fn read_parts(path: &Path, kind: Kind, range: Range<usize>) -> Result<Parts> {
    if kind == Kind::Pdf {
        let total = pdf_page_count(path)?.min(MAX_PAGES);
        let (first, last) = (range.start + 1, range.end.min(total));
        if first > last {
            return Ok(Parts { parts: Vec::new(), total });
        }
        let parts = pdf_pages(path, first, last)?
            .into_iter()
            .enumerate()
            .map(|(i, text)| Part { loc: first + i, text: text.trim().to_string() })
            .filter(|p| !p.text.is_empty())
            .collect();
        return Ok(Parts { parts, total });
    }
    // Code is read as written (an html template keeps its markup); only notebooks are converted.
    let notebook = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("ipynb"));
    let text = if kind == Kind::Code && !notebook { read_text(path)? } else { document_text(path)? };
    let all = passages(&text, PASSAGE_CHARS, OVERLAP_CHARS, MAX_PASSAGES);
    let total = all.len();
    let parts = all
        .into_iter()
        .enumerate()
        .skip(range.start)
        .take(range.len())
        .map(|(i, text)| Part { loc: i + 1, text })
        .collect();
    Ok(Parts { parts, total })
}

/// A document's plain text, whatever its format: docx, odt and html are converted; anything else is read as text.
pub fn document_text(path: &Path) -> Result<String> {
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    match ext.as_str() {
        "docx" => office_text(path, "word/document.xml", Office::Word),
        "ipynb" => notebook_text(path),
        "odt" => office_text(path, "content.xml", Office::OpenDocument),
        "html" | "htm" | "xhtml" => {
            let bytes = std::fs::read(path)?;
            Ok(html2text::config::plain_no_decorate().string_from_read(&bytes[..], 100_000)?)
        }
        _ => read_text(path),
    }
}

/// A Jupyter notebook's cells, code and markdown, without their outputs (plots, tables, logs), each under a
/// `# %%` marker as in Python scripts that hold cells.
fn notebook_text(path: &Path) -> Result<String> {
    let notebook: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?).context("not a notebook (JSON)")?;
    let mut out = String::new();
    for cell in notebook["cells"].as_array().into_iter().flatten() {
        let kind = cell["cell_type"].as_str().unwrap_or("code");
        if kind != "code" && kind != "markdown" {
            continue;
        }
        let source = match &cell["source"] {
            serde_json::Value::Array(lines) => lines.iter().filter_map(|l| l.as_str()).collect::<String>(),
            serde_json::Value::String(text) => text.clone(),
            _ => continue,
        };
        out += if kind == "markdown" { "# %% [markdown]\n" } else { "# %%\n" };
        out += source.trim_end();
        out += "\n\n";
    }
    Ok(out)
}

enum Office {
    Word,
    OpenDocument,
}

/// The text of a zipped XML document: paragraphs become lines; tabs and breaks are kept.
fn office_text(path: &Path, member: &str, format: Office) -> Result<String> {
    let mut archive = zip::ZipArchive::new(std::fs::File::open(path)?).context("not a valid document (zip)")?;
    let mut xml = String::new();
    archive.by_name(member).with_context(|| format!("no {member} inside"))?.read_to_string(&mut xml)?;
    let (paragraph, text_run, tab, line_break, space): (&str, Option<&str>, &str, &str, &str) = match format {
        // Word keeps text in <w:t> runs; field codes (<w:instrText>) aren't text.
        Office::Word => ("w:p", Some("w:t"), "w:tab", "w:br", ""),
        Office::OpenDocument => ("text:p", None, "text:tab", "text:line-break", "text:s"),
    };
    let mut reader = quick_xml::Reader::from_str(&xml);
    let mut out = String::new();
    // Inside a Word text run, or how deep inside ODT paragraphs and headings: only text there is the document's.
    let mut in_run = false;
    let mut depth = 0usize;
    let collecting = |in_run: bool, depth: usize| if text_run.is_some() { in_run } else { depth > 0 };
    let is_block = |name: &str| name == paragraph || name == "text:h";
    loop {
        let (name, opening, closing) = match reader.read_event()? {
            Event::Eof => break,
            Event::Start(e) => (e.name().0.to_string(), true, false),
            Event::End(e) => (e.name().0.to_string(), false, true),
            Event::Empty(e) => (e.name().0.to_string(), true, true),
            Event::Text(e) if collecting(in_run, depth) => {
                out.push_str(&e.xml10_content());
                continue;
            }
            Event::GeneralRef(r) if collecting(in_run, depth) => {
                match r.resolve_char_ref()? {
                    Some(c) => out.push(c),
                    None => out.push_str(match r.as_ref() {
                        "amp" => "&",
                        "lt" => "<",
                        "gt" => ">",
                        "quot" => "\"",
                        "apos" => "'",
                        _ => "",
                    }),
                }
                continue;
            }
            _ => continue,
        };
        let name = name.as_str();
        if Some(name) == text_run {
            in_run = opening && !closing;
        } else if opening && name == tab {
            out.push('\t');
        } else if opening && name == line_break {
            out.push('\n');
        } else if opening && !space.is_empty() && name == space {
            out.push(' ');
        } else if is_block(name) {
            if opening && !closing {
                depth += 1;
            }
            if closing {
                depth = depth.saturating_sub(1);
                out.push('\n');
            }
        }
    }
    Ok(out)
}

/// Whether poppler's PDF tools are installed (checked once).
pub fn pdf_tools() -> bool {
    static FOUND: OnceLock<bool> = OnceLock::new();
    *FOUND.get_or_init(|| {
        Command::new("pdftotext").arg("-v").output().is_ok_and(|o| o.status.success() || !o.stderr.is_empty())
            && Command::new("pdfinfo").arg("-v").output().is_ok()
    })
}

/// How to get the PDF tools, for messages.
pub const PDF_TOOLS_HINT: &str =
    "install poppler (brew install poppler, apt install poppler-utils, dnf install poppler-utils)";

/// Renders page `page` (from 1) of a PDF to a PNG `height` pixels tall (its width follows), with `pdftoppm`.
pub fn pdf_page_png(path: &Path, page: usize, height: u32) -> Result<Vec<u8>> {
    let output = Command::new("pdftoppm")
        .args(["-png", "-singlefile", "-f", &page.to_string(), "-l", &page.to_string()])
        .args(["-scale-to-x", "-1", "-scale-to-y", &height.max(64).to_string()])
        .arg(path)
        .output()
        .context("can't run pdftoppm")?;
    if !output.status.success() || output.stdout.is_empty() {
        bail!("pdftoppm: {}", String::from_utf8_lossy(&output.stderr).trim());
    }
    Ok(output.stdout)
}

/// A PDF's page count, from `pdfinfo`.
pub fn pdf_page_count(path: &Path) -> Result<usize> {
    let output = Command::new("pdfinfo").arg(path).output().context("can't run pdfinfo")?;
    if !output.status.success() {
        bail!("pdfinfo: {}", String::from_utf8_lossy(&output.stderr).trim());
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|l| l.strip_prefix("Pages:").and_then(|n| n.trim().parse().ok()))
        .context("pdfinfo didn't say how many pages")
}

/// The text of pages `first..=last` (from 1), one string a page, from `pdftotext` (it ends each page with a form
/// feed).
pub fn pdf_pages(path: &Path, first: usize, last: usize) -> Result<Vec<String>> {
    let output = Command::new("pdftotext")
        .args(["-enc", "UTF-8", "-f", &first.to_string(), "-l", &last.to_string()])
        .arg(path)
        .arg("-")
        .output()
        .context("can't run pdftotext")?;
    if !output.status.success() {
        bail!("pdftotext: {}", String::from_utf8_lossy(&output.stderr).trim());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut pages: Vec<String> = text.split('\x0c').map(String::from).collect();
    pages.truncate(last + 1 - first);
    Ok(pages)
}

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
    use std::io::Write;
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rummage-extract-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    /// A zip with one member, like a docx or odt.
    fn zipped(path: &Path, member: &str, xml: &str) {
        let mut zip = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
        zip.start_file(member, zip::write::SimpleFileOptions::default()).unwrap();
        zip.write_all(xml.as_bytes()).unwrap();
        zip.finish().unwrap();
    }

    #[test]
    fn docx_paragraphs_become_lines() {
        let path = scratch("lease.docx");
        zipped(
            &path,
            "word/document.xml",
            r#"<?xml version="1.0"?><w:document xmlns:w="w"><w:body>
               <w:p><w:r><w:t>Clause 9: Pets.</w:t></w:r><w:r><w:t xml:space="preserve"> One cat &amp; one dog.</w:t></w:r></w:p>
               <w:p><w:r><w:instrText>PAGE</w:instrText><w:t>Rent</w:t><w:tab/><w:t>1,200 EUR</w:t></w:r></w:p>
               </w:body></w:document>"#,
        );
        assert_eq!(document_text(&path).unwrap(), "Clause 9: Pets. One cat & one dog.\nRent\t1,200 EUR\n");
    }

    #[test]
    fn odt_paragraphs_and_headings_become_lines() {
        let path = scratch("notes.odt");
        zipped(
            &path,
            "content.xml",
            r#"<?xml version="1.0"?><office:document-content xmlns:office="o" xmlns:text="t">
               <office:automatic-styles><style>not text</style></office:automatic-styles>
               <office:body><office:text><text:h>Sleep</text:h><text:p>Keep the room<text:s/>cool.</text:p></office:text></office:body>
               </office:document-content>"#,
        );
        assert_eq!(document_text(&path).unwrap(), "Sleep\nKeep the room cool.\n");
    }

    #[test]
    fn notebooks_keep_cells_not_outputs() {
        let path = scratch("explore.ipynb");
        std::fs::write(
            &path,
            r##"{"cells": [
                {"cell_type": "markdown", "source": ["# Class balance\n", "Count each class."]},
                {"cell_type": "code", "source": "df.label.value_counts().plot.bar()", "outputs": [{"data": {"image/png": "iVBORw0KGgo"}}]},
                {"cell_type": "raw", "source": "skip me"}
            ], "nbformat": 4}"##,
        )
        .unwrap();
        let text = document_text(&path).unwrap();
        assert_eq!(
            text,
            "# %% [markdown]\n# Class balance\nCount each class.\n\n# %%\ndf.label.value_counts().plot.bar()\n\n"
        );
        // Code passages of a notebook are its cells, not its JSON.
        let read = read_parts(&path, Kind::Code, 0..10).unwrap();
        assert!(read.parts[0].text.contains("value_counts") && !read.parts[0].text.contains("iVBOR"));
    }

    #[test]
    fn html_becomes_text() {
        let path = scratch("page.html");
        std::fs::write(&path, "<html><head><title>t</title><style>p{}</style></head><body><h1>Brown bread</h1><p>Bake <b>45</b> minutes.</p></body></html>").unwrap();
        let text = document_text(&path).unwrap();
        assert!(text.contains("Brown bread") && text.contains("Bake 45 minutes."), "{text}");
        assert!(!text.contains("p{}"), "{text}");
    }

    /// A small PDF with one line of text a page, its cross-reference table computed so poppler reads it cleanly.
    pub fn write_pdf(path: &Path, pages: &[&str]) {
        let n = pages.len();
        let mut objects: Vec<String> = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".into(),
            format!(
                "<< /Type /Pages /Kids [{}] /Count {n} >>",
                (0..n).map(|i| format!("{} 0 R", 4 + 2 * i)).collect::<Vec<_>>().join(" ")
            ),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".into(),
        ];
        for (i, text) in pages.iter().enumerate() {
            let stream = format!("BT /F1 12 Tf 20 100 Td ({text}) Tj ET");
            objects.push(format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Resources << /Font << /F1 3 0 R >> >> /Contents {} 0 R >>",
                5 + 2 * i
            ));
            objects.push(format!("<< /Length {} >>\nstream\n{stream}\nendstream", stream.len()));
        }
        let mut out = String::from("%PDF-1.4\n");
        let mut offsets = Vec::new();
        for (i, object) in objects.iter().enumerate() {
            offsets.push(out.len());
            out += &format!("{} 0 obj\n{object}\nendobj\n", i + 1);
        }
        let xref = out.len();
        out += &format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1);
        for offset in offsets {
            out += &format!("{offset:010} 00000 n \n");
        }
        out += &format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objects.len() + 1);
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn pdfs_are_read_a_page_at_a_time() {
        if !pdf_tools() {
            eprintln!("skipped: no poppler");
            return;
        }
        let path = scratch("three.pdf");
        write_pdf(&path, &["The first page talks about dogs.", "", "The third page is about cats."]);
        assert_eq!(pdf_page_count(&path).unwrap(), 3);
        let read = read_parts(&path, Kind::Pdf, 0..10).unwrap();
        assert_eq!(read.total, 3);
        // The blank page is counted but has no part.
        assert_eq!(
            read.parts,
            vec![
                Part { loc: 1, text: "The first page talks about dogs.".into() },
                Part { loc: 3, text: "The third page is about cats.".into() }
            ]
        );
        let later = read_parts(&path, Kind::Pdf, 2..3).unwrap();
        assert_eq!(later.parts.iter().map(|p| p.loc).collect::<Vec<_>>(), vec![3]);
        let png = pdf_page_png(&path, 1, 100).unwrap();
        assert!(png.starts_with(b"\x89PNG"));
    }

    #[test]
    fn documents_go_in_steps() {
        let path = scratch("long.txt");
        std::fs::write(&path, (0..3000).map(|i| format!("word{i} ")).collect::<String>()).unwrap();
        let first = read_parts(&path, Kind::Doc, 0..2).unwrap();
        assert_eq!(first.parts.iter().map(|p| p.loc).collect::<Vec<_>>(), vec![1, 2]);
        let rest = read_parts(&path, Kind::Doc, 2..100).unwrap();
        assert_eq!(rest.parts.first().map(|p| p.loc), Some(3));
        assert_eq!(rest.parts.len() + 2, first.total);
    }

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
