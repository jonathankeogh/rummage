//! The words in a picture, through tesseract (packaged on every Unix: `brew install tesseract`,
//! `apt install tesseract-ocr`). For screenshots and scanned PDF pages; the words go to keyword search.

use anyhow::{Result, bail};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

/// Whether tesseract is installed (checked once).
pub fn available() -> bool {
    static FOUND: OnceLock<bool> = OnceLock::new();
    *FOUND.get_or_init(|| Command::new("tesseract").arg("--version").output().is_ok_and(|o| o.status.success()))
}

/// How to get OCR, for messages.
pub const HINT: &str = "install tesseract (brew install tesseract, apt install tesseract-ocr, dnf install tesseract)";

/// Formats tesseract reads itself (through leptonica).
fn reads(path: &Path) -> bool {
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp" | "gif" | "webp")
}

/// The words in a picture file (others are converted to PNG first).
pub fn file(path: &Path) -> Result<String> {
    if reads(path) {
        return run(Command::new("tesseract").arg(path).args(["stdout", "--psm", "3"]), None);
    }
    let rgb = crate::picture::decode(path)?;
    let mut png = Vec::new();
    image::RgbImage::from_raw(rgb.width as u32, rgb.height as u32, rgb.data)
        .expect("a full buffer")
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)?;
    png_bytes(&png)
}

/// The words in a PNG held in memory (a rendered PDF page).
pub fn png_bytes(png: &[u8]) -> Result<String> {
    run(Command::new("tesseract").args(["stdin", "stdout", "--psm", "3"]), Some(png))
}

fn run(command: &mut Command, input: Option<&[u8]>) -> Result<String> {
    let mut child = command
        .stdin(if input.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let (Some(bytes), Some(mut stdin)) = (input, child.stdin.take()) {
        stdin.write_all(bytes)?;
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!("tesseract: {}", String::from_utf8_lossy(&output.stderr).trim());
    }
    // One line per line of text; blank runs squeezed.
    let text = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_words_in_a_picture() {
        if !available() {
            eprintln!("skipped: no tesseract");
            return;
        }
        // Black text drawn on white, big enough to read: "LITE" in 5x7 blocks of 12 px (letters with no lookalikes).
        let glyphs = [
            ["#....", "#....", "#....", "#....", "#....", "#....", "#####"], // L
            [".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."], // I
            ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."], // T
            ["#####", "#....", "#....", "####.", "#....", "#....", "#####"], // E
        ];
        let (cell, width, height) = (12u32, 4 * 6 * 12 + 48, 7 * 12 + 48);
        let mut image = image::RgbImage::from_pixel(width, height, image::Rgb([255, 255, 255]));
        for (g, glyph) in glyphs.iter().enumerate() {
            for (row, line) in glyph.iter().enumerate() {
                for (col, c) in line.chars().enumerate() {
                    if c == '#' {
                        for dy in 0..cell {
                            for dx in 0..cell {
                                let x = 24 + (g as u32 * 6 + col as u32) * cell + dx;
                                image.put_pixel(x, 24 + row as u32 * cell + dy, image::Rgb([0, 0, 0]));
                            }
                        }
                    }
                }
            }
        }
        let mut png = Vec::new();
        image.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
        let text = png_bytes(&png).unwrap();
        assert!(text.contains("LITE"), "{text:?}");
    }
}
