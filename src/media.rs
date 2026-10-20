//! Audio and video for the model, through ffmpeg and ffprobe (packaged on every Unix: `brew install ffmpeg`,
//! `apt install ffmpeg`). Ported from DigUp's portable path (MIT).
//!
//! Audio becomes 30 s windows of 16 kHz mono, 25 s apart (one model input hears at most 30 s), silent ones skipped.
//! Video becomes a frame every few seconds, one kept per shot, plus its soundtrack as audio windows.

use crate::tool;
use anyhow::{Context, Result, bail};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

pub const SAMPLE_RATE: usize = 16_000;
/// One model input hears 30 s; windows overlap by 5 s so nothing said across a cut is lost.
pub const WINDOW_SECONDS: f64 = 30.0;
pub const HOP_SECONDS: f64 = 25.0;
/// A tail shorter than this isn't worth a window of its own.
const MIN_TAIL_SECONDS: f64 = 2.0;
/// About −50 dBFS: silence isn't worth a vector.
const SILENCE_RMS: f64 = 0.003;
/// Seconds between sampled video frames (near-duplicates are dropped), and frames per video at most.
pub const FRAME_INTERVAL: f64 = 3.0;
const MAX_FRAMES: usize = 1200;
/// Vision tokens per video frame (DigUp's).
pub const FRAME_BUDGET: usize = 70;

/// Whether ffmpeg and ffprobe are installed (checked once).
pub fn available() -> bool {
    static FOUND: OnceLock<bool> = OnceLock::new();
    *FOUND.get_or_init(|| {
        ["ffmpeg", "ffprobe"].iter().all(|name| {
            tool::output(Command::new(name).arg("-version"), tool::limit(10)).is_ok_and(|o| o.status.success())
        })
    })
}

/// How to get audio and video, for messages.
pub const HINT: &str = "install ffmpeg (brew install ffmpeg, apt install ffmpeg, dnf install ffmpeg)";

fn probe(path: &Path, entries: &str) -> Result<String> {
    let output = tool::output(
        Command::new("ffprobe")
            .args(["-v", "error", "-show_entries", entries, "-of", "default=noprint_wrappers=1:nokey=1"])
            .arg(path),
        tool::limit(30),
    )?;
    if !output.status.success() {
        bail!("ffprobe: {}", String::from_utf8_lossy(&output.stderr).trim());
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Length in seconds.
pub fn duration(path: &Path) -> Option<f64> {
    probe(path, "format=duration").ok()?.trim().parse().ok().filter(|d: &f64| d.is_finite() && *d > 0.0)
}

/// A stretch of sound for a waveform: how loud each of `slices` equal parts of `from`..`to` seconds is, from 0 (silent)
/// to 1 (as loud as its loudest 1%). Decoded at a low rate, so minutes take a fraction of a second.
pub fn loudness(path: &Path, from: f64, to: f64, slices: usize) -> Result<Vec<f32>> {
    const RATE: usize = 4000;
    let output = tool::output(
        Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-nostdin",
                "-ss",
                &format!("{from:.3}"),
                "-t",
                &format!("{:.3}", (to - from).max(0.1)),
            ])
            .arg("-i")
            .arg(path)
            .args(["-vn", "-ac", "1", "-ar", &RATE.to_string(), "-f", "s16le", "-"]),
        tool::limit(60),
    )?;
    let samples: Vec<f32> =
        output.stdout.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect();
    if samples.is_empty() {
        bail!("no sound in {}", path.display());
    }
    let wanted = ((to - from) * RATE as f64) as usize;
    let per_slice = (wanted.max(samples.len()) as f64 / slices.max(1) as f64).max(1.0);
    let mut levels: Vec<f32> = (0..slices)
        .map(|i| {
            let start = (i as f64 * per_slice) as usize;
            let end = (((i + 1) as f64 * per_slice) as usize).min(samples.len());
            let part = samples.get(start..end).unwrap_or(&[]);
            if part.is_empty() { 0.0 } else { (part.iter().map(|s| s * s).sum::<f32>() / part.len() as f32).sqrt() }
        })
        .collect();
    // Scaled linearly to the loudest 1% (a square root, or the single loudest slice, flattens speech into a block).
    let mut sorted = levels.clone();
    sorted.sort_by(f32::total_cmp);
    let loud = sorted[(sorted.len() * 99 / 100).min(sorted.len() - 1)].max(1e-4);
    for level in &mut levels {
        *level = (*level / loud).min(1.0);
    }
    Ok(levels)
}

/// Which kinds of streams it has: (audio, video).
pub fn streams(path: &Path) -> (bool, bool) {
    let text = probe(path, "stream=codec_type").unwrap_or_default();
    (text.contains("audio"), text.contains("video"))
}

/// A stretch of audio for the model: its samples (16 kHz mono, −1 to 1) and where it sits, in seconds.
pub struct AudioWindow {
    pub start: f64,
    pub end: f64,
    pub samples: Vec<f32>,
}

/// Decodes the audio (a file's, or a video's soundtrack) and hands each non-silent window to `each` as it comes:
/// memory stays at one window, even for a three-hour podcast. With `span` (from, to), only the windows that start in
/// it, as a whole-file read would cut them (a long file goes in passes). Returns the length in seconds.
pub fn audio_windows(
    path: &Path,
    span: Option<(f64, f64)>,
    mut each: impl FnMut(AudioWindow) -> Result<()>,
) -> Result<f64> {
    let window = (WINDOW_SECONDS * SAMPLE_RATE as f64) as usize;
    let hop = (HOP_SECONDS * SAMPLE_RATE as f64) as usize;
    let mut command = Command::new("ffmpeg");
    command.args(["-v", "error", "-nostdin"]);
    let from = match span {
        Some((from, to)) => {
            // The last window that starts before `to` runs 5 s past it.
            let length = to - from + WINDOW_SECONDS - HOP_SECONDS;
            command.args(["-ss", &format!("{from:.3}"), "-t", &format!("{length:.3}")]);
            from
        }
        None => 0.0,
    };
    let mut child = command
        .arg("-i")
        .arg(path)
        .args(["-vn", "-ac", "1", "-ar", &SAMPLE_RATE.to_string(), "-f", "s16le", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("can't run ffmpeg")?;
    let mut stdout = child.stdout.take().expect("piped");
    // Given up on when a window's worth of sound takes over a minute to come (the model's time doesn't count).
    let reading = tool::Reading::default();
    let progress = reading.clone();
    let watchdog = tool::Watchdog::new(child, "ffmpeg", tool::limit(60), move || progress.progress());
    let mut bytes_read = 0u64;
    let mut buffer: Vec<i16> = Vec::new();
    let mut offset = 0usize; // sample index of buffer[0] in the file
    let mut odd: Option<u8> = None; // a byte left over when a read ends mid-sample
    let mut chunk = vec![0u8; 1 << 16];
    let emit = |buffer: &[i16], offset: usize, each: &mut dyn FnMut(AudioWindow) -> Result<()>| -> Result<()> {
        let rms =
            (buffer.iter().map(|&s| (s as f64 / 32768.0).powi(2)).sum::<f64>() / buffer.len().max(1) as f64).sqrt();
        if rms <= SILENCE_RMS {
            return Ok(());
        }
        each(AudioWindow {
            start: from + offset as f64 / SAMPLE_RATE as f64,
            end: from + (offset + buffer.len()) as f64 / SAMPLE_RATE as f64,
            samples: buffer.iter().map(|&s| s as f32 / 32768.0).collect(),
        })
    };
    loop {
        reading.at(bytes_read);
        let read = stdout.read(&mut chunk)?;
        reading.idle();
        if read == 0 {
            break;
        }
        bytes_read += read as u64;
        let mut bytes: Vec<u8> = Vec::with_capacity(read + 1);
        if let Some(first) = odd.take() {
            bytes.push(first);
        }
        bytes.extend_from_slice(&chunk[..read]);
        if bytes.len() % 2 == 1 {
            odd = bytes.pop();
        }
        buffer.extend(bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])));
        while buffer.len() >= window {
            emit(&buffer[..window], offset, &mut each)?;
            buffer.drain(..hop);
            offset += hop;
        }
    }
    let status = watchdog.finish()?;
    if !status.success() && offset == 0 && buffer.is_empty() {
        bail!("ffmpeg couldn't decode the audio");
    }
    // The tail: new audio beyond the last window's overlap, or the whole file if shorter than a window.
    let covered = if offset == 0 { 0 } else { window - hop };
    if buffer.len().saturating_sub(covered) >= (MIN_TAIL_SECONDS * SAMPLE_RATE as f64) as usize {
        emit(&buffer, offset, &mut each)?;
    }
    Ok(duration(path).unwrap_or(from + (offset + buffer.len()) as f64 / SAMPLE_RATE as f64))
}

/// A video frame for the model: a JPEG on disk and the stretch it stands for, in seconds.
pub struct Frame {
    pub path: PathBuf,
    pub start: f64,
    pub end: f64,
}

/// Frames of a video, one per shot: a frame every `FRAME_INTERVAL` seconds (more apart for very long videos, so at
/// most `MAX_FRAMES` in all), and one that looks like the last kept one (8×8 average hash within 4 bits) only extends
/// it. With `span` (from, to), only the frames in it (a long video goes in passes). The JPEGs are written into
/// `folder`, which the caller removes.
pub fn keyframes(path: &Path, folder: &Path, span: Option<(f64, f64)>) -> Result<Vec<Frame>> {
    let Some(duration) = duration(path) else { return Ok(Vec::new()) };
    let step = FRAME_INTERVAL.max(duration / MAX_FRAMES as f64);
    let first = 0.5f64.min(duration / 2.0);
    let (from, to) = span.map_or((0.0, duration), |(from, to)| (from, to.min(duration)));
    // The first slot in the span: frame n is the one shown at first + n * step, wherever the passes cut.
    let start = first + ((from - first) / step).ceil().max(0.0) * step;
    if start >= to {
        return Ok(Vec::new());
    }
    std::fs::create_dir_all(folder)?;
    // Round=up: the last frame at or before each slot. From the start, every frame is decoded anyway (and an MPEG-TS
    // has no index to seek with); a later pass seeks to where it starts.
    let mut command = Command::new("ffmpeg");
    command.args(["-v", "error", "-nostdin"]);
    if from > 0.0 {
        command.args(["-ss", &format!("{from:.3}"), "-t", &format!("{:.3}", to - from)]);
    }
    let child = command
        .arg("-i")
        .arg(path)
        .args([
            "-an",
            "-vf",
            &format!(
                "fps=fps={}:start_time={}:round=up,scale=1024:1024:force_original_aspect_ratio=decrease",
                1.0 / step,
                start - from
            ),
            "-q:v",
            "3",
            "-f",
            "image2",
        ])
        .arg(folder.join("frame-%06d.jpg"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("can't run ffmpeg")?;
    // Given up on when a frame takes over a minute to come.
    let written = folder.to_path_buf();
    let frames_so_far = move || Some(std::fs::read_dir(&written).map_or(0, |d| d.count() as u64));
    let status = tool::Watchdog::new(child, "ffmpeg", tool::limit(60), frames_so_far).finish()?;
    let mut names: Vec<PathBuf> = std::fs::read_dir(folder)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("frame-")))
        .collect();
    names.sort();
    if names.is_empty() && !status.success() {
        bail!("ffmpeg couldn't decode the video");
    }
    let mut frames: Vec<Frame> = Vec::new();
    let mut last_hash: Option<u64> = None;
    for (index, frame) in names.into_iter().enumerate() {
        let start = start + index as f64 * step;
        if start >= to {
            continue; // fps can fill a slot past the end
        }
        if let Ok(image) = image::open(&frame) {
            let hash = average_hash(&image.to_rgb8());
            if last_hash.is_some_and(|last| (hash ^ last).count_ones() <= 4) {
                continue;
            }
            last_hash = Some(hash);
        }
        frames.push(Frame { path: frame, start, end: to });
    }
    // Each kept frame stands for the stretch until the next.
    for i in 1..frames.len() {
        frames[i - 1].end = frames[i].start;
    }
    Ok(frames)
}

/// An 8×8 average hash of a picture's brightness: frames of one shot hash alike.
pub fn average_hash(image: &image::RgbImage) -> u64 {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let mut cells = [0u32; 64];
    for cy in 0..8 {
        let y0 = cy * height / 8;
        let y1 = height.min((y0 + 1).max((cy + 1) * height / 8));
        for cx in 0..8 {
            let x0 = cx * width / 8;
            let x1 = width.min((x0 + 1).max((cx + 1) * width / 8));
            let (mut sum, mut n) = (0u32, 0u32);
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = image.get_pixel(x as u32, y as u32);
                    sum += (p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114) / 1000;
                    n += 1;
                }
            }
            cells[cy * 8 + cx] = sum / n.max(1);
        }
    }
    let mean = cells.iter().sum::<u32>() / 64;
    cells.iter().enumerate().fold(0u64, |hash, (i, &c)| if c > mean { hash | 1 << i } else { hash })
}

/// The frame shown at `seconds`, as an RGB picture with its long side at most `pixels` (for the preview).
pub fn frame_at(path: &Path, seconds: f64, pixels: u32) -> Result<image::DynamicImage> {
    let output = tool::output(
        Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin", "-ss", &seconds.max(0.0).to_string(), "-i"])
            .arg(path)
            .args([
                "-frames:v",
                "1",
                "-vf",
                &format!("scale='min(iw,{pixels})':'min(ih,{pixels})':force_original_aspect_ratio=decrease"),
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ]),
        tool::limit(60),
    )?;
    if output.stdout.is_empty() {
        bail!("no frame at {seconds:.0} s");
    }
    Ok(image::load_from_memory(&output.stdout)?)
}

/// MPEG transport streams (.ts / .mts / .m2ts video) start a packet with 0x47 every 188 bytes, or every 192 with a
/// 4-byte timecode. A .ts that isn't one is TypeScript.
pub fn is_mpeg_transport_stream(path: &Path) -> bool {
    let mut bytes = [0u8; 600];
    let Ok(mut file) = std::fs::File::open(path) else { return false };
    let Ok(n) = file.read(&mut bytes) else { return false };
    n >= 400
        && ((bytes[0] == 0x47 && bytes[188] == 0x47 && bytes[376] == 0x47)
            || (bytes[4] == 0x47 && bytes[196] == 0x47 && bytes[388] == 0x47))
}

/// "2:14", or "1:02:14" past an hour.
pub fn clock(seconds: f64) -> String {
    let s = seconds.max(0.0).round() as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, s % 3600 / 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clocks() {
        assert_eq!(clock(134.4), "2:14");
        assert_eq!(clock(3734.0), "1:02:14");
        assert_eq!(clock(0.0), "0:00");
    }

    #[test]
    fn typescript_is_not_video() {
        let dir = std::env::temp_dir().join(format!("rummage-ts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let code = dir.join("app.ts");
        std::fs::write(&code, "export const greet = (name: string) => `hello ${name}`;\n".repeat(20)).unwrap();
        assert!(!is_mpeg_transport_stream(&code));
        let mut packets = vec![0u8; 188 * 4];
        for i in 0..4 {
            packets[i * 188] = 0x47;
        }
        let video = dir.join("clip.ts");
        std::fs::write(&video, packets).unwrap();
        assert!(is_mpeg_transport_stream(&video));
    }

    #[test]
    fn hashes_tell_shots_apart() {
        let flat = image::RgbImage::from_pixel(64, 48, image::Rgb([120, 120, 120]));
        let mut split = flat.clone();
        for y in 0..48 {
            for x in 0..32 {
                split.put_pixel(x, y, image::Rgb([250, 250, 250]));
            }
        }
        assert_eq!(average_hash(&flat), average_hash(&flat.clone()));
        assert!((average_hash(&flat) ^ average_hash(&split)).count_ones() > 4);
    }

    #[test]
    fn loudness_follows_the_sound() {
        if !available() {
            eprintln!("skipped: no ffmpeg");
            return;
        }
        // 2 s of silence, then 2 s of tone.
        let dir = std::env::temp_dir().join(format!("rummage-loudness-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("half.wav");
        let ok = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=2", "-f", "lavfi", "-i"])
            .args(["sine=frequency=440:sample_rate=16000:duration=2", "-filter_complex", "[0][1]concat=n=2:v=0:a=1"])
            .arg(&wav)
            .status()
            .unwrap();
        assert!(ok.success());
        let levels = loudness(&wav, 0.0, 4.0, 8).unwrap();
        assert_eq!(levels.len(), 8);
        assert!(levels[..3].iter().all(|l| *l < 0.05), "{levels:?}");
        assert!(levels[5..].iter().all(|l| *l > 0.8), "{levels:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn audio_windows_overlap_and_skip_silence() {
        if !available() {
            eprintln!("skipped: no ffmpeg");
            return;
        }
        let dir = std::env::temp_dir().join(format!("rummage-audio-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // 70 s: a tone for 0–30 s, silence for 30–60 s, a tone again for 60–70 s.
        let wav = dir.join("tone.wav");
        let status = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=30",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono:d=30",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=660:duration=10",
                "-filter_complex",
                "[0][1][2]concat=n=3:v=0:a=1",
                "-ac",
                "1",
                "-ar",
                "16000",
            ])
            .arg(&wav)
            .status()
            .unwrap();
        assert!(status.success());
        let mut windows = Vec::new();
        let length = audio_windows(&wav, None, |w| {
            windows.push((w.start, w.end, w.samples.len()));
            Ok(())
        })
        .unwrap();
        assert!((length - 70.0).abs() < 0.5, "{length}");
        // Windows at 0, 25 and 50 s (each 30 s); the 25 s one holds tone too; the tail from 75 s doesn't exist.
        let starts: Vec<f64> = windows.iter().map(|w| w.0).collect();
        assert_eq!(starts, vec![0.0, 25.0, 50.0], "{windows:?}");
        assert!(windows.iter().all(|w| w.2 == 30 * SAMPLE_RATE || w.1 >= 69.0));
        // Read in two passes, cut at 50 s: the same windows, each once.
        let mut passes = Vec::new();
        for span in [(0.0, 50.0), (50.0, 71.0)] {
            audio_windows(&wav, Some(span), |w| {
                passes.push((w.start, w.end, w.samples.len()));
                Ok(())
            })
            .unwrap();
        }
        let close = |a: &(f64, f64, usize), b: &(f64, f64, usize)| (a.0 - b.0).abs() < 0.01 && (a.1 - b.1).abs() < 0.1;
        assert!(passes.len() == windows.len() && passes.iter().zip(&windows).all(|(a, b)| close(a, b)), "{passes:?}");
    }
}
