//! The first run: which folders to index, asked in the terminal. Your home folder with its folders ticked (untick
//! what to leave out), or folders you name, each checked as it's given so a typo, a file, a system folder or one
//! macOS keeps private is explained there and then. Music is asked about separately (it's days of sound to read).

use crate::crawl;
use anyhow::Result;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::crossterm::{cursor, execute, terminal};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

/// What to index, as chosen.
pub struct Choice {
    pub roots: Vec<PathBuf>,
    /// Folders inside them left out (unticked).
    pub excluded: Vec<PathBuf>,
    pub music: bool,
    /// Keep the index up to date in the background, from login (`rummage watch --install`).
    pub background: bool,
}

/// Asks which folders to index. None when the person backs out.
pub fn ask() -> Result<Option<Choice>> {
    let home = crate::paths::home();
    eprintln!(
        "rummage hasn't indexed anything yet. Let's choose what it reads (only these folders, never anything else).\n"
    );
    if yes_no(&format!("Index your home folder ({})?", crate::paths::tilde(&home)), true)? {
        let folders = home_folders(&home);
        eprintln!("\nWhich of its folders? ↑↓ move, space ticks or unticks, ⏎ when done, esc to stop.");
        let Some(ticked) = checklist(&folders)? else { return Ok(None) };
        let mut excluded: Vec<PathBuf> =
            folders.iter().zip(&ticked).filter(|(_, t)| !**t).map(|(f, _)| f.clone()).collect();
        // Each ticked folder read once now: on macOS that's when it asks whether your terminal may (Documents,
        // Desktop, Downloads), while you're here to answer.
        for (folder, _) in folders.iter().zip(&ticked).filter(|(_, t)| **t) {
            if let Err(why) = check_folder(folder) {
                eprintln!("  {}: {why}; left out for now", crate::paths::tilde(folder));
                excluded.push(folder.clone());
            }
        }
        let music_ticked = folders.iter().zip(&ticked).any(|(f, t)| *t && f.file_name().is_some_and(|n| n == "Music"));
        let music = music_ticked
            && yes_no(
                "Include your music too (songs and music videos: days of sound to read, rarely searched)?",
                false,
            )?;
        let background = ask_background()?;
        return Ok(Some(Choice { roots: vec![home], excluded, music, background }));
    }
    let mut roots = Vec::new();
    loop {
        let prompt = if roots.is_empty() { "Which folder?" } else { "Another folder? (⏎ when done)" };
        let typed = line(prompt)?;
        if typed.is_empty() {
            if roots.is_empty() {
                return Ok(None);
            }
            break;
        }
        let folder = expand(&typed, &home);
        match check_folder(&folder) {
            Ok(folder) => {
                if let Some(kind) = crawl::network_file_system(&folder) {
                    let ask = format!(
                        "{} is on a network drive ({kind}): reading it fetches every file over the network. Add it?",
                        crate::paths::tilde(&folder)
                    );
                    if !yes_no(&ask, false)? {
                        continue;
                    }
                }
                eprintln!("  added {}", crate::paths::tilde(&folder));
                roots.push(folder);
            }
            Err(why) => eprintln!("  {}: {why}", crate::paths::tilde(&folder)),
        }
    }
    let background = ask_background()?;
    Ok(Some(Choice { roots, excluded: Vec::new(), music: false, background }))
}

fn ask_background() -> Result<bool> {
    yes_no("Keep the index up to date in the background, starting at login?", true)
}

/// A folder as typed: `~` for home, and the quotes or backslashes a dragged-in path comes with.
pub fn expand(typed: &str, home: &Path) -> PathBuf {
    let typed = typed.trim().trim_matches(|c| c == '"' || c == '\'').replace("\\ ", " ");
    let path = match typed.strip_prefix('~') {
        Some(rest) => home.join(rest.trim_start_matches('/')),
        None => PathBuf::from(&typed),
    };
    std::path::absolute(&path).unwrap_or(path)
}

/// The folder as rummage would index it, or why it can't: missing, a file, the system's, or not readable.
pub fn check_folder(folder: &Path) -> Result<PathBuf, String> {
    let folder = match folder.canonicalize() {
        Ok(folder) => folder,
        Err(_) => return Err("there's no such folder".into()),
    };
    if !folder.is_dir() {
        return Err("that's a file, not a folder".into());
    }
    if crawl::is_system_folder(&folder) {
        return Err("that's the system's, not yours: rummage reads your files".into());
    }
    match std::fs::read_dir(&folder) {
        Ok(_) => Ok(folder),
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied && cfg!(target_os = "macos") => {
            Err("macOS keeps it private: allow your terminal app in System Settings › Privacy & Security › Files and \
             Folders (or Full Disk Access), then try again"
                .into())
        }
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            Err("you don't have permission to read it".into())
        }
        Err(e) => Err(format!("it can't be read ({e})")),
    }
}

/// The folders in your home worth offering: not hidden, not ~/Library or ~/Applications (programs, not files).
fn home_folders(home: &Path) -> Vec<PathBuf> {
    let mut folders: Vec<PathBuf> = std::fs::read_dir(home)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
                .map(|e| e.path())
                .filter(|p| {
                    let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    !name.starts_with('.') && !["Library", "Applications", "snap"].contains(&name.as_str())
                })
                .collect()
        })
        .unwrap_or_default();
    folders.sort_by_key(|p| p.file_name().map(|n| n.to_string_lossy().to_lowercase()));
    folders
}

/// A list of folders to tick or untick, all ticked to start. None if left with esc.
fn checklist(folders: &[PathBuf]) -> Result<Option<Vec<bool>>> {
    let mut ticked = vec![true; folders.len()];
    if folders.is_empty() {
        return Ok(Some(ticked));
    }
    let mut at = 0;
    let mut err = std::io::stderr();
    terminal::enable_raw_mode()?;
    let draw = |err: &mut std::io::Stderr, ticked: &[bool], at: usize, first: bool| -> std::io::Result<()> {
        if !first {
            execute!(err, cursor::MoveUp(folders.len() as u16))?;
        }
        for (i, folder) in folders.iter().enumerate() {
            let name = folder.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let mark = if ticked[i] { "[x]" } else { "[ ]" };
            let pointer = if i == at { "›" } else { " " };
            write!(err, "\r\x1b[K {pointer} {mark} {name}\r\n")?;
        }
        err.flush()
    };
    let result = (|| -> Result<Option<Vec<bool>>> {
        draw(&mut err, &ticked, at, true)?;
        loop {
            let Event::Key(key) = event::read()? else { continue };
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match key.code {
                KeyCode::Up | KeyCode::Char('k') => at = at.saturating_sub(1),
                KeyCode::Down | KeyCode::Char('j') => at = (at + 1).min(folders.len() - 1),
                KeyCode::Char(' ') => ticked[at] = !ticked[at],
                KeyCode::Enter => return Ok(Some(ticked.clone())),
                KeyCode::Esc => return Ok(None),
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return Ok(None),
                _ => continue,
            }
            draw(&mut err, &ticked, at, false)?;
        }
    })();
    terminal::disable_raw_mode()?;
    result
}

fn line(prompt: &str) -> Result<String> {
    eprint!("{prompt} ");
    std::io::stderr().flush()?;
    let mut answer = String::new();
    std::io::stdin().lock().read_line(&mut answer)?;
    Ok(answer.trim().to_string())
}

fn yes_no(question: &str, default: bool) -> Result<bool> {
    let answer = line(&format!("{question} {}", if default { "[Y/n]" } else { "[y/N]" }))?.to_lowercase();
    Ok(if answer.is_empty() { default } else { answer.starts_with('y') })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_folders_are_understood() {
        let home = Path::new("/Users/sam");
        assert_eq!(expand("~/Documents", home), Path::new("/Users/sam/Documents"));
        assert_eq!(expand("~", home), Path::new("/Users/sam"));
        // Dragged into a terminal: quoted, or with its spaces escaped.
        assert_eq!(expand("'/Volumes/My Drive' ", home), Path::new("/Volumes/My Drive"));
        assert_eq!(expand("/Volumes/My\\ Drive", home), Path::new("/Volumes/My Drive"));
    }

    #[test]
    fn folders_are_checked_with_a_reason() {
        let dir = std::env::temp_dir().join(format!("rummage-setup-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("note.md"), "a note").unwrap();
        assert!(check_folder(&dir).is_ok());
        assert_eq!(check_folder(&dir.join("missing")).unwrap_err(), "there's no such folder");
        assert_eq!(check_folder(&dir.join("note.md")).unwrap_err(), "that's a file, not a folder");
        assert!(check_folder(Path::new("/usr")).unwrap_err().contains("system"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
