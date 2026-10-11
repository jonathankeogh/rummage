//! The terminal interface: type to search, see the matched passage, open the file.

pub mod app;
pub mod backend;
pub mod preview;
pub mod ui;

pub use backend::Backend;

use anyhow::{Context, Result};
use app::{Action, App};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{self, Event};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode};
use std::io::{Stdout, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

type Term = Terminal<CrosstermBackend<Stdout>>;

/// Open the interface and run until the user quits.
pub fn run<B: Backend>(backend: B) -> Result<()> {
    let (requests, responses) = app::spawn_worker(backend);
    let mut app = App::new(requests);

    // Put the terminal back before the panic message prints, so it stays readable.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = leave();
        default_hook(info);
    }));
    let mut terminal = enter().context("can't start the terminal interface")?;
    let result = event_loop(&mut terminal, &mut app, &responses);
    let _ = leave();
    result
}

fn enter() -> Result<Term> {
    enable_raw_mode()?;
    execute!(std::io::stdout(), EnterAlternateScreen)?;
    Ok(Terminal::new(CrosstermBackend::new(std::io::stdout()))?)
}

fn leave() -> Result<()> {
    execute!(std::io::stdout(), LeaveAlternateScreen)?;
    disable_raw_mode()?;
    Ok(())
}

fn event_loop(terminal: &mut Term, app: &mut App, responses: &std::sync::mpsc::Receiver<app::Response>) -> Result<()> {
    loop {
        while let Ok(response) = responses.try_recv() {
            app.apply_response(response);
        }
        app.refresh_preview();
        terminal.draw(|frame| ui::render(frame, app))?;
        // Poll briefly so results and the spinner show up without a key press.
        if !event::poll(Duration::from_millis(60))? {
            app.tick = app.tick.wrapping_add(1);
            continue;
        }
        let Event::Key(key) = event::read()? else { continue };
        match app.on_key(key) {
            Action::None => {}
            Action::Quit => return Ok(()),
            Action::Open(i) => {
                let (path, code) = (app.hits[i].path.clone(), app.hits[i].kind == crate::store::Kind::Code);
                if let Err(e) = open(terminal, &path, code) {
                    app.notify(format!("can't open: {e:#}"));
                }
            }
            Action::Reveal(i) => {
                if let Err(e) = reveal(&app.hits[i].path) {
                    app.notify(format!("can't reveal: {e:#}"));
                }
            }
            Action::Copy(i) => {
                let path = app.hits[i].path.clone();
                match copy_osc52(&path) {
                    Ok(()) => app.notify("path copied"),
                    Err(e) => app.notify(format!("can't copy: {e}")),
                }
            }
        }
    }
}

/// The program that opens files for the desktop.
fn opener() -> &'static str {
    if cfg!(target_os = "macos") { "open" } else { "xdg-open" }
}

/// Open a file: code in `$EDITOR` if set (the interface steps aside while it runs), anything else with the desktop.
fn open(terminal: &mut Term, path: &str, code: bool) -> Result<()> {
    let editor = std::env::var("EDITOR").ok().filter(|e| code && !e.trim().is_empty());
    if let Some(editor) = editor {
        let mut words = editor.split_whitespace();
        let mut command = Command::new(words.next().unwrap_or("vi"));
        command.args(words).arg(path);
        leave()?;
        let status = command.status();
        *terminal = enter()?;
        terminal.clear()?;
        status.with_context(|| format!("running {editor}"))?;
        return Ok(());
    }
    detached(opener(), &[path])
}

/// Show the file in the file manager (macOS selects it in Finder; elsewhere the parent folder opens).
fn reveal(path: &str) -> Result<()> {
    if cfg!(target_os = "macos") {
        detached("open", &["-R", path])
    } else {
        let parent = Path::new(path).parent().map_or(".".to_string(), |p| p.display().to_string());
        detached("xdg-open", &[&parent])
    }
}

/// Run a helper without letting its output reach the screen.
fn detached(program: &str, args: &[&str]) -> Result<()> {
    Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("running {program}"))?;
    Ok(())
}

/// Copy text to the clipboard through the terminal itself (OSC 52), which works over ssh.
fn copy_osc52(text: &str) -> std::io::Result<()> {
    let mut out = std::io::stdout();
    out.write_all(osc52(text).as_bytes())?;
    out.flush()
}

/// The escape sequence that sets the clipboard to `text`.
pub fn osc52(text: &str) -> String {
    format!("\x1b]52;c;{}\x07", base64(text.as_bytes()))
}

fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n =
            (chunk[0] as u32) << 16 | (*chunk.get(1).unwrap_or(&0) as u32) << 8 | *chunk.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(TABLE[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_known_values() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn osc52_sequence() {
        assert_eq!(osc52("/a"), "\x1b]52;c;L2E=\x07");
    }
}
