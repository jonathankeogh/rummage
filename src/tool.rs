//! Running the outside programs rummage reads files with (poppler, tesseract, ffmpeg, sips), each with a time limit:
//! a file that makes one hang is given up on, not waited for forever.

use anyhow::{Context, Result, bail};
use std::io::{Read, Write};
use std::process::{Child, Command, Output, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// A limit of `seconds`, or of RUMMAGE_TOOL_SECONDS for every tool (tests shorten it).
pub fn limit(seconds: u64) -> Duration {
    let seconds = std::env::var("RUMMAGE_TOOL_SECONDS").ok().and_then(|s| s.parse().ok()).unwrap_or(seconds);
    Duration::from_secs(seconds)
}

/// The command's name, for messages.
fn name(command: &Command) -> String {
    command.get_program().to_string_lossy().into_owned()
}

/// Runs `command` to the end and returns its output, or an error once it's taken longer than `limit` (it's killed).
pub fn output(command: &mut Command, limit: Duration) -> Result<Output> {
    run(command, None, limit)
}

/// The same, with `input` on its standard input.
pub fn output_with_input(command: &mut Command, input: &[u8], limit: Duration) -> Result<Output> {
    run(command, Some(input.to_vec()), limit)
}

fn run(command: &mut Command, input: Option<Vec<u8>>, limit: Duration) -> Result<Output> {
    let program = name(command);
    let mut child = command
        .stdin(if input.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("can't run {program}"))?;
    // Fed and drained on threads, so a full pipe never blocks the wait.
    let feeder = match (input, child.stdin.take()) {
        (Some(bytes), Some(mut stdin)) => Some(std::thread::spawn(move || {
            let _ = stdin.write_all(&bytes);
        })),
        _ => None,
    };
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut bytes);
            }
            bytes
        })
    };
    let stdout = drain(child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let stderr = drain(child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if started.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            bail!("{program} took over {} s: given up", limit.as_secs());
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    if let Some(feeder) = feeder {
        let _ = feeder.join();
    }
    Ok(Output { status, stdout: stdout.join().unwrap_or_default(), stderr: stderr.join().unwrap_or_default() })
}

/// Keeps an eye on a running child whose output is read as it comes (ffmpeg decoding a long file): kills it when
/// `progress` stays the same for longer than `limit`. `progress` returns None while the child isn't what's being
/// waited for (the model is busy with what it gave), which doesn't count.
pub struct Watchdog {
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<Child>>,
    program: String,
    limit: Duration,
}

impl Watchdog {
    pub fn new(
        child: Child,
        program: &str,
        limit: Duration,
        progress: impl Fn() -> Option<u64> + Send + 'static,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let fired = Arc::new(AtomicBool::new(false));
        let (stop_seen, fired_set) = (stop.clone(), fired.clone());
        let thread = std::thread::spawn(move || {
            let mut child = child;
            let mut last = None;
            let mut since = Instant::now();
            while !stop_seen.load(Ordering::Relaxed) {
                if let Ok(Some(_)) = child.try_wait() {
                    break;
                }
                let now = progress();
                if now.is_none() || now != last {
                    last = now;
                    since = Instant::now();
                } else if since.elapsed() > limit {
                    fired_set.store(true, Ordering::Relaxed);
                    let _ = child.kill();
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            child
        });
        Watchdog { stop, fired, thread: Some(thread), program: program.to_string(), limit }
    }

    /// Stops watching and waits for the child to end. An error if it was killed for being stuck.
    pub fn finish(mut self) -> Result<std::process::ExitStatus> {
        self.stop.store(true, Ordering::Relaxed);
        let mut child = self.thread.take().expect("joined once").join().expect("the watchdog doesn't panic");
        let status = child.wait()?;
        if self.fired.load(Ordering::Relaxed) {
            bail!("{} made no progress for {} s: given up", self.program, self.limit.as_secs());
        }
        Ok(status)
    }

    /// Whether it killed the child.
    pub fn fired(&self) -> bool {
        self.fired.load(Ordering::Relaxed)
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(Ok(mut child)) = self.thread.take().map(|t| t.join()) {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// A count shared with a watchdog: what's been read so far, or None while not waiting on the child.
#[derive(Clone, Default)]
pub struct Reading(Arc<AtomicU64>);

impl Reading {
    const IDLE: u64 = u64::MAX;

    /// Waiting on the child, having read `bytes` so far.
    pub fn at(&self, bytes: u64) {
        self.0.store(bytes, Ordering::Relaxed);
    }

    /// Not waiting on the child.
    pub fn idle(&self) {
        self.0.store(Self::IDLE, Ordering::Relaxed);
    }

    pub fn progress(&self) -> Option<u64> {
        let value = self.0.load(Ordering::Relaxed);
        (value != Self::IDLE).then_some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hanging_program_is_killed_at_its_limit() {
        let started = Instant::now();
        let error = output(Command::new("sleep").arg("30"), Duration::from_secs(1)).unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(error.to_string().contains("sleep took over 1 s"), "{error}");
        let done = output(Command::new("echo").arg("hi"), Duration::from_secs(5)).unwrap();
        assert_eq!(done.stdout, b"hi\n");
        let fed = output_with_input(&mut Command::new("cat"), b"fed", Duration::from_secs(5)).unwrap();
        assert_eq!(fed.stdout, b"fed");
    }

    #[test]
    fn a_stuck_stream_is_killed_but_a_paused_reader_is_not() {
        // Stuck: nothing comes, and the reader is waiting.
        let child = Command::new("sleep").arg("30").stdout(Stdio::piped()).spawn().unwrap();
        let reading = Reading::default();
        reading.at(0);
        let progress = reading.clone();
        let started = Instant::now();
        let watchdog = Watchdog::new(child, "sleep", Duration::from_secs(1), move || progress.progress());
        while !watchdog.fired() && started.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(watchdog.finish().is_err());
        // Paused: the reader is busy elsewhere, so the quiet doesn't count.
        let child = Command::new("sleep").arg("2").spawn().unwrap();
        let reading = Reading::default();
        reading.idle();
        let progress = reading.clone();
        let watchdog = Watchdog::new(child, "sleep", Duration::from_secs(1), move || progress.progress());
        std::thread::sleep(Duration::from_millis(2500));
        assert!(watchdog.finish().unwrap().success());
    }
}
