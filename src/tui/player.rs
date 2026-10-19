//! Listening in the interface: the selected recording's or video's sound, from its moment, through mpv or ffplay
//! (ffmpeg's player) with no window. Pausing stops the player and remembers where; playing again, or skipping, starts
//! it there. That needs nothing from the player but a start time, so mpv and ffplay behave the same.

use crate::paths::on_path;
use anyhow::{Context, Result};
use std::process::{Child, Command, Stdio};
use std::time::Instant;

/// The players that can play from a moment without a window, best first.
const PLAYERS: [&str; 2] = ["mpv", "ffplay"];

/// What to tell someone without either player.
pub const HINT: &str = "to listen here, install mpv (or ffmpeg's ffplay); ⏎ opens it in your player";

/// The installed player, if any.
pub fn program() -> Option<&'static str> {
    PLAYERS.into_iter().find(|p| on_path(p))
}

pub struct Player {
    pub path: String,
    program: &'static str,
    child: Option<Child>,
    /// Where the current run started in the file, and when.
    from: f64,
    since: Instant,
    /// Where it was paused, while paused.
    paused: Option<f64>,
    /// The file's length, if known: playing stops there.
    length: Option<f64>,
    /// Where listening began (or the earliest it was skipped back to): the waveform marks what's been heard.
    pub heard_from: f64,
}

impl Player {
    /// Starts playing `path` at `seconds`.
    pub fn start(path: &str, seconds: f64, length: Option<f64>) -> Result<Player> {
        let program = program().context(HINT)?;
        let mut player = Player {
            path: path.to_string(),
            program,
            child: None,
            from: seconds,
            since: Instant::now(),
            paused: None,
            length,
            heard_from: seconds.max(0.0),
        };
        player.spawn(seconds)?;
        Ok(player)
    }

    fn spawn(&mut self, seconds: f64) -> Result<()> {
        self.stop();
        let at = format!("{:.2}", seconds.max(0.0));
        let mut command = Command::new(self.program);
        if self.program == "mpv" {
            command.args(["--no-video", "--no-terminal", "--really-quiet", &format!("--start={at}"), "--"]);
        } else {
            command.args(["-nodisp", "-autoexit", "-loglevel", "quiet", "-ss", &at]);
        }
        let child = command
            .arg(&self.path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("can't run {}", self.program))?;
        self.child = Some(child);
        self.from = seconds.max(0.0);
        self.since = Instant::now();
        self.paused = None;
        Ok(())
    }

    fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    /// Where it is in the file, in seconds.
    pub fn position(&self) -> f64 {
        let at = self.paused.unwrap_or(self.from + self.since.elapsed().as_secs_f64());
        self.length.map_or(at, |length| at.min(length))
    }

    /// Playing now (as of the last `poll`).
    pub fn is_playing(&self) -> bool {
        self.paused.is_none() && self.child.is_some()
    }

    /// Notices the player reaching the end; true while playing.
    pub fn poll(&mut self) -> bool {
        if self.paused.is_some() {
            return false;
        }
        // The player quits at the end of the file: it holds there, paused.
        if self.child.as_mut().is_none_or(|c| c.try_wait().ok().flatten().is_some()) {
            let end = self.position();
            self.child = None;
            self.paused = Some(end);
            return false;
        }
        true
    }

    /// Pauses, or plays on from where it was (from the start again once it reached the end).
    pub fn toggle(&mut self) -> Result<()> {
        if self.poll() {
            let at = self.position();
            self.stop();
            self.paused = Some(at);
            Ok(())
        } else {
            let at = self.position();
            let at = if self.length.is_some_and(|length| at >= length - 0.5) { 0.0 } else { at };
            self.spawn(at)
        }
    }

    /// Moves by `seconds` (back when negative), playing on if it was playing.
    pub fn skip(&mut self, seconds: f64) -> Result<()> {
        let to = (self.position() + seconds).max(0.0);
        let to = self.length.map_or(to, |length| to.min(length));
        self.heard_from = self.heard_from.min(to);
        if self.poll() {
            self.spawn(to)
        } else {
            self.paused = Some(to);
            Ok(())
        }
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.stop();
    }
}
