//! The terminal interface: type to search, see the matched passage, open the file.

pub mod app;
pub mod backend;
pub mod player;
pub mod preview;
pub mod ui;

pub use backend::Backend;

use anyhow::{Context, Result};
use app::{Action, App};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode};
use std::io::{Stdout, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

type Term = Terminal<CrosstermBackend<Stdout>>;

/// Open the interface and run until the user quits.
pub fn run<B: Backend>(backend: B) -> Result<()> {
    // Asked every so often, on its own thread, how indexing is going (a connection that fails just says nothing).
    let (progress_tx, progress) = std::sync::mpsc::channel();
    if let Some(mut probe) = backend.indexing_probe() {
        std::thread::spawn(move || {
            loop {
                if progress_tx.send(probe()).is_err() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(1500));
            }
        });
    }
    let (requests, responses) = app::spawn_worker(backend);
    let mut app = App::new(requests);

    // Put the terminal back before the panic message prints, so it stays readable.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = leave();
        default_hook(info);
    }));
    let mut terminal = enter().context("can't start the terminal interface")?;
    let result = event_loop(&mut terminal, &mut app, &responses, &progress);
    let _ = leave();
    result
}

fn enter() -> Result<Term> {
    enable_raw_mode()?;
    // Mouse reporting: clicks and the wheel come to the interface (hold Shift, or Option in some terminals, to select
    // text the usual way).
    execute!(std::io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
    Ok(Terminal::new(CrosstermBackend::new(std::io::stdout()))?)
}

fn leave() -> Result<()> {
    execute!(std::io::stdout(), DisableMouseCapture, LeaveAlternateScreen)?;
    disable_raw_mode()?;
    Ok(())
}

fn event_loop(
    terminal: &mut Term,
    app: &mut App,
    responses: &std::sync::mpsc::Receiver<app::Response>,
    progress: &std::sync::mpsc::Receiver<Option<backend::Indexing>>,
) -> Result<()> {
    loop {
        while let Ok(response) = responses.try_recv() {
            app.apply_response(response);
        }
        while let Ok(left) = progress.try_recv() {
            app.set_indexing(left);
        }
        app.refresh_preview();
        app.refresh_page_image();
        app.refresh_wave();
        terminal.draw(|frame| ui::render(frame, app))?;
        // Poll briefly so results and the spinner show up without a key press.
        if !event::poll(Duration::from_millis(60))? {
            app.tick = app.tick.wrapping_add(1);
            continue;
        }
        let action = match event::read()? {
            Event::Key(key) => app.on_key(key),
            Event::Mouse(mouse) => app.on_mouse(mouse),
            _ => continue,
        };
        match action {
            Action::None => {}
            Action::Quit => return Ok(()),
            Action::Open(i) => {
                let hit = &app.hits[i];
                let page = (hit.kind == crate::store::Kind::Pdf).then(|| hit.loc.map_or(1, |l| l.round() as usize));
                let moment = hit.kind.is_timed().then(|| hit.loc.unwrap_or(0.0));
                let (path, code) = (hit.path.clone(), hit.kind == crate::store::Kind::Code);
                if let Err(e) = open(terminal, &path, code, page, moment) {
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
fn open(terminal: &mut Term, path: &str, code: bool, page: Option<usize>, moment: Option<f64>) -> Result<()> {
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
    if let Some(page) = page {
        if let Some((viewer, args)) = pdf_viewer_at(page) {
            return detached(viewer, &args.iter().map(String::as_str).chain([path]).collect::<Vec<_>>());
        }
    }
    if let Some(seconds) = moment {
        if let Some((player, args)) = player_at(seconds) {
            return detached(player, &args.iter().map(String::as_str).chain([path]).collect::<Vec<_>>());
        }
    }
    detached(opener(), &[path])
}

/// A media player on PATH that can start at a moment, with the arguments that do it; without one, the file opens
/// from its start with the desktop's player.
fn player_at(seconds: f64) -> Option<(&'static str, Vec<String>)> {
    let at = format!("{:.0}", seconds.max(0.0).floor());
    let players: [(&str, Vec<String>); 2] =
        [("mpv", vec![format!("--start={at}"), "--force-window".into()]), ("vlc", vec![format!("--start-time={at}")])];
    let path = std::env::var_os("PATH")?;
    players.into_iter().find(|(name, _)| std::env::split_paths(&path).any(|dir| dir.join(name).is_file()))
}

/// A PDF viewer on PATH that can open at a page, with the arguments that do it. macOS's `open` can't, so there the
/// file opens at its start.
fn pdf_viewer_at(page: usize) -> Option<(&'static str, Vec<String>)> {
    let page = page.to_string();
    if cfg!(target_os = "macos") {
        return None;
    }
    let viewers: [(&str, Vec<String>); 3] = [
        ("zathura", vec!["--page".into(), page.clone()]),
        ("evince", vec!["--page-index".into(), page.clone()]),
        ("okular", vec!["--page".into(), page]),
    ];
    let path = std::env::var_os("PATH")?;
    viewers.into_iter().find(|(name, _)| std::env::split_paths(&path).any(|dir| dir.join(name).is_file()))
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
