//! Pictures for the vision encoder: how big each goes in, resized exactly as the reference resizes it, and decoding
//! the formats llama.cpp's own decoder (stb_image: JPEG, PNG, BMP, GIF, TGA, PSD) doesn't read.
//!
//! Sizing and resampling are ported from DigUp (MIT), which checked them against Pillow bit for bit: the same pixels
//! go in, so the same vectors come out (tests/reference_vectors.rs).

use anyhow::{Context, Result, bail};
use std::path::Path;
use std::process::Command;

/// Vision tokens per kind of picture (DigUp's): photos 140, screenshots and PDF pages 280, video frames 70.
pub const PHOTO_BUDGET: usize = 140;
pub const SCREENSHOT_BUDGET: usize = 280;
pub const PAGE_BUDGET: usize = 280;

/// transformers' Gemma 4 image processor: scale to fill `budget` soft tokens (each 3 × 3 patches of 16 px) keeping the
/// aspect ratio, each side rounded down to a multiple of 48 px. None for an image too thin to hold a token.
pub fn target_size(height: usize, width: usize, budget: usize) -> Option<(usize, usize)> {
    let (patch, pooling) = (16usize, 3usize);
    let max_patches = budget * pooling * pooling;
    let factor = ((max_patches * patch * patch) as f64 / (height * width) as f64).sqrt();
    let side = pooling * patch;
    let mut target_height = (factor * height as f64 / side as f64).floor() as usize * side;
    let mut target_width = (factor * width as f64 / side as f64).floor() as usize * side;
    if target_height == 0 && target_width == 0 {
        return None;
    }
    let max_side = (max_patches / (pooling * pooling)) * side;
    if target_height == 0 {
        target_height = side;
        target_width = (width / height * side).min(max_side);
    } else if target_width == 0 {
        target_width = side;
        target_height = (height / width * side).min(max_side);
    }
    Some((target_height, target_width))
}

const PRECISION: i32 = 22;

fn filter(value: f64) -> f64 {
    let (x, a) = (value.abs(), -0.5);
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

/// For each output pixel: the first contributing input pixel and how many contribute (`bounds`), and their weights
/// in 22-bit fixed point, `size` per output pixel.
fn coefficients(in_size: usize, out_size: usize) -> (Vec<usize>, Vec<i32>, usize) {
    let scale = in_size as f64 / out_size as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let size = support.ceil() as usize * 2 + 1;
    let mut bounds = vec![0usize; out_size * 2];
    let mut weights = vec![0i32; out_size * size];
    let mut row = vec![0.0f64; size];
    for out in 0..out_size {
        let center = (out as f64 + 0.5) * scale;
        let first = ((center - support + 0.5) as isize).max(0) as usize;
        let count = (((center + support + 0.5) as isize).min(in_size as isize) as usize).saturating_sub(first);
        let mut total = 0.0;
        for x in 0..size {
            row[x] = if x < count { filter(((x + first) as f64 - center + 0.5) / filter_scale) } else { 0.0 };
            total += row[x];
        }
        for x in 0..size {
            let weight = if x < count && total != 0.0 { row[x] / total } else { row[x] };
            weights[out * size + x] =
                (weight * (1i64 << PRECISION) as f64 + if weight < 0.0 { -0.5 } else { 0.5 }) as i32;
        }
        bounds[out * 2] = first;
        bounds[out * 2 + 1] = count;
    }
    (bounds, weights, size)
}

fn clip(value: i32) -> u8 {
    (value >> PRECISION).clamp(0, 255) as u8
}

fn horizontal(src: &[u8], in_width: usize, rows: usize, out_width: usize) -> Vec<u8> {
    let (bounds, weights, size) = coefficients(in_width, out_width);
    let mut out = vec![0u8; out_width * rows * 3];
    for y in 0..rows {
        for x in 0..out_width {
            let mut p = (y * in_width + bounds[x * 2]) * 3;
            let (mut r, mut g, mut b) = (1i32 << (PRECISION - 1), 1i32 << (PRECISION - 1), 1i32 << (PRECISION - 1));
            for k in 0..bounds[x * 2 + 1] {
                let w = weights[x * size + k];
                r = r.wrapping_add((src[p] as i32).wrapping_mul(w));
                g = g.wrapping_add((src[p + 1] as i32).wrapping_mul(w));
                b = b.wrapping_add((src[p + 2] as i32).wrapping_mul(w));
                p += 3;
            }
            let o = (y * out_width + x) * 3;
            out[o] = clip(r);
            out[o + 1] = clip(g);
            out[o + 2] = clip(b);
        }
    }
    out
}

fn vertical(src: &[u8], in_height: usize, row_width: usize, out_height: usize) -> Vec<u8> {
    let (bounds, weights, size) = coefficients(in_height, out_height);
    let stride = row_width * 3;
    let mut out = vec![0u8; stride * out_height];
    let mut sums = vec![0i32; stride];
    for y in 0..out_height {
        sums.fill(1 << (PRECISION - 1));
        for k in 0..bounds[y * 2 + 1] {
            let row = &src[(bounds[y * 2] + k) * stride..][..stride];
            let w = weights[y * size + k];
            for (sum, &value) in sums.iter_mut().zip(row) {
                *sum = sum.wrapping_add((value as i32).wrapping_mul(w));
            }
        }
        for (o, &sum) in out[y * stride..][..stride].iter_mut().zip(&sums) {
            *o = clip(sum);
        }
    }
    out
}

/// Pillow's bicubic resampling of packed RGB (a = −0.5, the kernel widened when shrinking, 22-bit fixed point): what
/// the reference gets from PIL's Image.resize.
pub fn resize_bicubic(src: &[u8], width: usize, height: usize, to_width: usize, to_height: usize) -> Vec<u8> {
    match (to_width != width, to_height != height) {
        (true, true) => vertical(&horizontal(src, width, height, to_width), height, to_width, to_height),
        (true, false) => horizontal(src, width, height, to_width),
        (false, true) => vertical(src, height, width, to_height),
        (false, false) => src[..width * height * 3].to_vec(),
    }
}

/// A decoded picture: packed RGB.
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// Formats llama.cpp decodes itself (stb_image), exactly as the reference did.
pub fn stb_reads(path: &Path) -> bool {
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp" | "gif" | "tga" | "psd")
}

/// Decodes what stb_image doesn't: WebP and TIFF in Rust; HEIC (iPhone photos) through `sips` (macOS) or libheif's
/// `heif-convert` (Linux), when there.
pub fn decode(path: &Path) -> Result<Rgb> {
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    let image = if matches!(ext.as_str(), "heic" | "heif") {
        let png = heic_to_png(path)?;
        image::load_from_memory(&png)?
    } else {
        image::open(path).with_context(|| format!("can't decode {}", path.display()))?
    };
    let rgb = image.to_rgb8();
    Ok(Rgb { width: rgb.width() as usize, height: rgb.height() as usize, data: rgb.into_raw() })
}

/// A picture's size without decoding all of it (None if it can't be read).
pub fn dimensions(path: &Path) -> Option<(u32, u32)> {
    image::image_dimensions(path).ok()
}

/// The tool that converts HEIC here, if any.
pub fn heic_tool() -> Option<&'static str> {
    static TOOL: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
    *TOOL.get_or_init(|| {
        if cfg!(target_os = "macos") && Command::new("sips").arg("--help").output().is_ok() {
            Some("sips")
        } else if Command::new("heif-convert").arg("--version").output().is_ok() {
            Some("heif-convert")
        } else {
            None
        }
    })
}

/// How to get HEIC support, for messages.
pub const HEIC_HINT: &str = "install libheif (apt install libheif-examples, dnf install libheif-tools)";

fn heic_to_png(path: &Path) -> Result<Vec<u8>> {
    let out = std::env::temp_dir().join(format!("rummage-heic-{}-{}.png", std::process::id(), rand_suffix()));
    let status = match heic_tool() {
        Some("sips") => crate::tool::output(
            Command::new("sips").args(["-s", "format", "png"]).arg(path).arg("--out").arg(&out),
            crate::tool::limit(30),
        )?,
        Some(_) => crate::tool::output(Command::new("heif-convert").arg(path).arg(&out), crate::tool::limit(30))?,
        None => bail!("can't read HEIC: {HEIC_HINT}"),
    };
    if !status.status.success() {
        bail!("can't convert {}: {}", path.display(), String::from_utf8_lossy(&status.stderr).trim());
    }
    let png = std::fs::read(&out)?;
    let _ = std::fs::remove_file(&out);
    Ok(png)
}

fn rand_suffix() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u64(
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64),
    );
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_fill_the_budget_in_48px_steps() {
        // A 4:3 photo at 140 tokens: 140 * 9 patches of 16 px.
        let (h, w) = target_size(3024, 4032, 140).unwrap();
        assert!(h % 48 == 0 && w % 48 == 0, "{h}x{w}");
        assert!((h / 16) * (w / 16) <= 140 * 9, "{h}x{w} within the budget");
        assert_eq!((h, w), (480, 624), "30 x 39 = 1,170 patches of 1,260");
        // A very thin strip still gets one row of tokens.
        assert_eq!(target_size(10, 5000, 280).map(|(h, _)| h), Some(48));
    }

    #[test]
    fn bicubic_keeps_flat_colour_and_size() {
        let src: Vec<u8> = [200u8, 100, 50].repeat(97 * 61);
        let out = resize_bicubic(&src, 97, 61, 48, 96);
        assert_eq!(out.len(), 48 * 96 * 3);
        assert!(out.chunks(3).all(|p| p == [200, 100, 50]), "a flat image stays flat");
    }
}
