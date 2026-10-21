//! Watch mode: the search service keeps the index up to date. It watches the indexed folders (inotify on Linux,
//! FSEvents on macOS, kqueue on the BSDs, through `notify`), waits for changes to settle, and runs `rummage index` in
//! a child process at low priority: the same skip rules and two passes as running it yourself, and the 555 MB picture
//! and sound encoder loaded only while there's something to read. The service goes on answering searches meanwhile,
//! from the index as it fills in.

use crate::store::Store;
use anyhow::{Context, Result, bail};
use notify::{EventKind, RecursiveMode, Watcher};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Changes are picked up once the folders have been quiet this long (a download or a save is many events)…
const SETTLE: Duration = Duration::from_secs(2);
/// …or after this long at most, for a folder that never goes quiet.
const MOST: Duration = Duration::from_secs(30);
/// How often the watcher looks for folders added to the index since it started.
const ROOTS_EVERY: Duration = Duration::from_secs(30);

/// Holds the index's lock until dropped: one indexing run at a time, yours or the service's. A run that finds it held
/// waits for the other to finish.
pub fn lock_index(dir: &Path, quiet: bool) -> Result<File> {
    std::fs::create_dir_all(dir)?;
    let file = File::create(dir.join("index.lock")).context("can't create the index lock")?;
    if file.try_lock().is_err() {
        if !quiet {
            eprintln!(
                "waiting for another `rummage index` to finish (the search service keeping the index up to date)…"
            );
        }
        file.lock().context("can't lock the index")?;
    }
    Ok(file)
}

/// Runs `rummage index` for the service, one run at a time; changes that come in during a run get another.
pub struct Indexer {
    dir: PathBuf,
    running: AtomicBool,
    again: AtomicBool,
    /// The next run starts the index over (it was made by another version).
    rebuild: AtomicBool,
    lock: Mutex<()>,
}

impl Indexer {
    pub fn new(dir: &Path) -> Arc<Indexer> {
        Arc::new(Indexer {
            dir: dir.to_path_buf(),
            running: AtomicBool::new(false),
            again: AtomicBool::new(false),
            rebuild: AtomicBool::new(false),
            lock: Mutex::new(()),
        })
    }

    /// An indexing run is going.
    pub fn running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Brings the index up to date, now or right after the run in progress.
    pub fn run(self: &Arc<Self>) {
        let _guard = self.lock.lock().unwrap();
        if self.running.swap(true, Ordering::SeqCst) {
            self.again.store(true, Ordering::SeqCst);
            return;
        }
        let indexer = self.clone();
        std::thread::spawn(move || {
            loop {
                let rebuild = indexer.rebuild.swap(false, Ordering::SeqCst);
                if let Err(error) = indexer.run_once(rebuild) {
                    eprintln!("rummage serve: indexing: {error:#}");
                }
                let _guard = indexer.lock.lock().unwrap();
                if !indexer.again.swap(false, Ordering::SeqCst) {
                    indexer.running.store(false, Ordering::SeqCst);
                    return;
                }
            }
        });
    }

    /// Starts the index over, from the folders it had.
    pub fn rebuild(self: &Arc<Self>) {
        self.rebuild.store(true, Ordering::SeqCst);
        self.run();
    }

    fn run_once(&self, rebuild: bool) -> Result<()> {
        let started = Instant::now();
        let exe = std::env::current_exe()?;
        // Low priority, so searches and everything else come first.
        let mut command = Command::new("nice");
        command.args(["-n", "10"]).arg(&exe).arg("--index").arg(&self.dir).args(["index", "--background"]);
        if rebuild {
            command.arg("--reindex");
        }
        let log = std::fs::OpenOptions::new().create(true).append(true).open(crate::serve::log_path(&self.dir))?;
        let status = command
            .stdin(Stdio::null())
            .stdout(log.try_clone()?)
            .stderr(log)
            .status()
            .context("can't run rummage index")?;
        if !status.success() {
            bail!("rummage index stopped ({status})");
        }
        if started.elapsed() > Duration::from_secs(2) {
            eprintln!("rummage serve: index up to date ({:.0} s)", started.elapsed().as_secs_f64());
        }
        Ok(())
    }
}

/// What the watcher is doing, for `rummage status`.
#[derive(Default)]
pub struct Watching {
    /// Folders watched.
    pub folders: AtomicUsize,
    /// Why some couldn't be (Linux's limit on watches, say), if so.
    pub problem: Mutex<Option<String>>,
}

/// Watches the index's folders on its own thread, running `indexer` once changes settle. Folders added to the index
/// later are picked up within half a minute.
pub fn watch(dir: &Path, indexer: Arc<Indexer>) -> Result<Arc<Watching>> {
    let (tx, rx) = mpsc::channel::<notify::Result<notify::Event>>();
    let mut watcher = notify::recommended_watcher(move |event| {
        let _ = tx.send(event);
    })?;
    let watching = Arc::new(Watching::default());
    let store = Store::open(dir)?;
    let index_dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let state = watching.clone();
    std::thread::spawn(move || {
        // Folders watched, and every folder tried (one that can't be watched isn't tried again and again).
        let mut watched: Vec<PathBuf> = Vec::new();
        let mut tried: Vec<PathBuf> = Vec::new();
        let mut look_for_roots = Instant::now() - ROOTS_EVERY;
        // The busy file's time the timer last ran the indexer for.
        let mut swept_for: Option<i64> = None;
        loop {
            if look_for_roots.elapsed() >= ROOTS_EVERY {
                look_for_roots = Instant::now();
                let new: Vec<PathBuf> =
                    store.roots().unwrap_or_default().into_iter().filter(|r| !tried.contains(r)).collect();
                for root in new {
                    tried.push(root.clone());
                    match watcher.watch(&root, RecursiveMode::Recursive) {
                        Ok(()) => watched.push(root.clone()),
                        Err(error) => {
                            let problem = watch_problem(&root, &error);
                            eprintln!("rummage serve: {problem}");
                            *state.problem.lock().unwrap() = Some(problem);
                        }
                    }
                }
                state.folders.store(watched.len(), Ordering::Relaxed);
            }
            // A file left alone for changing often is due: sweep, even if nothing else has changed since.
            let due = store.meta("busy_until").ok().flatten().and_then(|t| t.parse::<i64>().ok());
            let wait = match due {
                Some(due) if swept_for != Some(due) => {
                    let left = due - unix_now();
                    if left <= 0 {
                        swept_for = Some(due);
                        indexer.run();
                        continue;
                    }
                    ROOTS_EVERY.min(Duration::from_secs(left as u64 + 1))
                }
                _ => ROOTS_EVERY,
            };
            // Wait for a change that matters, then for the folders to settle.
            let Ok(event) = rx.recv_timeout(wait) else { continue };
            if !matters(&event, &watched, &index_dir) {
                continue;
            }
            let first = Instant::now();
            while first.elapsed() < MOST {
                match rx.recv_timeout(SETTLE) {
                    Ok(_) => continue,
                    Err(mpsc::RecvTimeoutError::Timeout) => break,
                    Err(mpsc::RecvTimeoutError::Disconnected) => return,
                }
            }
            indexer.run();
        }
    });
    Ok(watching)
}

/// A change to a file rummage might index: not a read, not in the index's own folder, not hidden below a watched
/// folder, not a download or save still being written.
fn matters(event: &notify::Result<notify::Event>, roots: &[PathBuf], index_dir: &Path) -> bool {
    let Ok(event) = event else { return true };
    if matches!(event.kind, EventKind::Access(_)) {
        return false;
    }
    event.paths.iter().any(|path| {
        if path.starts_with(index_dir) {
            return false;
        }
        let below = roots.iter().find_map(|root| path.strip_prefix(root).ok()).unwrap_or(path);
        let hidden = below.components().any(|c| c.as_os_str().to_string_lossy().starts_with('.'));
        let name = path.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
        let unfinished = UNFINISHED.iter().any(|e| name.ends_with(e)) || name.starts_with("~$");
        // Safari downloads into a "….download" folder.
        let in_download = below.components().any(|c| c.as_os_str().to_string_lossy().ends_with(".download"));
        !hidden && !unfinished && !in_download
    })
}

/// Endings of files still being downloaded or written: browsers', qBittorrent's, editors'. (`~$` starts Office's lock
/// files.)
const UNFINISHED: &[&str] = &[".part", ".crdownload", ".download", ".opdownload", ".!qb", ".tmp", ".swp", "~"];

fn unix_now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

fn watch_problem(root: &Path, error: &notify::Error) -> String {
    let too_many = matches!(&error.kind, notify::ErrorKind::MaxFilesWatch)
        || matches!(&error.kind, notify::ErrorKind::Io(e) if e.raw_os_error() == Some(28));
    if too_many {
        format!(
            "can't watch all of {}: the system's limit on watched folders is reached (raise it with \
             `sudo sysctl fs.inotify.max_user_watches=524288`); its changes are picked up each time the service starts",
            root.display()
        )
    } else {
        format!("can't watch {}: {error}; its changes are picked up each time the service starts", root.display())
    }
}

/// Settings kept from the shell that installs the service, besides PATH: where the model and the index live, and the
/// CPU switch. Without them the service would look elsewhere than this shell's rummage does.
const KEPT_SETTINGS: &[&str] =
    &["RUMMAGE_MODELS", "RUMMAGE_CPU", "HF_HUB_CACHE", "HF_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"];

/// The service's environment: PATH (so it finds ffmpeg, poppler and tesseract where this shell does) and the
/// settings above that are set here.
fn service_environment() -> Vec<(&'static str, String)> {
    let mut environment = vec![("PATH", std::env::var("PATH").unwrap_or_default())];
    for name in KEPT_SETTINGS {
        if let Ok(value) = std::env::var(name) {
            environment.push((name, value));
        }
    }
    environment
}

/// Starts `rummage watch` at login: a launchd agent on macOS, a systemd user service elsewhere, with this shell's
/// PATH and rummage settings (`service_environment`).
pub fn install(dir: &Path) -> Result<()> {
    let exe = stable_path(&std::env::current_exe()?.canonicalize()?);
    let dir = {
        std::fs::create_dir_all(dir)?;
        dir.canonicalize()?
    };
    let environment = service_environment();
    let home = PathBuf::from(std::env::var_os("HOME").context("no HOME")?);
    if cfg!(target_os = "macos") {
        let plist = home.join("Library/LaunchAgents").join(format!("{LAUNCHD_LABEL}.plist"));
        std::fs::create_dir_all(plist.parent().unwrap())?;
        let log = crate::serve::log_path(&dir);
        std::fs::write(
            &plist,
            format!(
                r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LAUNCHD_LABEL}</string>
  <key>ProgramArguments</key>
  <array><string>{}</string><string>--index</string><string>{}</string><string>watch</string></array>
  <key>EnvironmentVariables</key><dict>{}</dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ProcessType</key><string>Background</string>
  <key>StandardErrorPath</key><string>{}</string>
</dict>
</plist>
"#,
                xml(&exe.to_string_lossy()),
                xml(&dir.to_string_lossy()),
                environment
                    .iter()
                    .map(|(name, value)| format!("<key>{name}</key><string>{}</string>", xml(value)))
                    .collect::<String>(),
                xml(&log.to_string_lossy())
            ),
        )?;
        let domain = format!("gui/{}", unsafe { getuid() });
        let _ = Command::new("launchctl").args(["bootout", &domain]).arg(&plist).stderr(Stdio::null()).status();
        let status = Command::new("launchctl").args(["bootstrap", &domain]).arg(&plist).status()?;
        if !status.success() {
            bail!("launchctl couldn't load {}", plist.display());
        }
        println!("rummage watch starts at login now, and is running ({})", plist.display());
        println!(
            "macOS may now ask whether rummage can read your Documents, Desktop or Downloads: click Allow (it reads \
             only the folders you chose). If you don't, rummage status says which it can't read."
        );
    } else {
        let unit = home.join(".config/systemd/user").join(SYSTEMD_UNIT);
        std::fs::create_dir_all(unit.parent().unwrap())?;
        std::fs::write(
            &unit,
            format!(
                "[Unit]\nDescription=rummage: keeps the search index up to date\n\n[Service]\n\
                 ExecStart=\"{}\" --index \"{}\" watch\n{}Restart=on-failure\nNice=10\n\n\
                 [Install]\nWantedBy=default.target\n",
                exe.display(),
                dir.display(),
                environment.iter().map(|(name, value)| format!("Environment=\"{name}={value}\"\n")).collect::<String>()
            ),
        )?;
        for args in [vec!["--user", "daemon-reload"], vec!["--user", "enable", "--now", SYSTEMD_UNIT]] {
            let status = Command::new("systemctl").args(&args).status().context("can't run systemctl")?;
            if !status.success() {
                bail!("systemctl {} failed", args.join(" "));
            }
        }
        println!("rummage watch starts at login now, and is running ({})", unit.display());
    }
    Ok(())
}

/// A path to this program that outlasts updates. Homebrew keeps each version in its own folder
/// (`…/Cellar/rummage/0.2.0/bin/rummage`) and deletes old ones, so a login item pointing there breaks after an
/// upgrade; `…/opt/rummage/bin/rummage` always points at the version installed.
fn stable_path(exe: &Path) -> PathBuf {
    let parts: Vec<_> = exe.components().collect();
    let Some(cellar) = parts.iter().position(|c| c.as_os_str() == "Cellar") else { return exe.to_path_buf() };
    let (Some(formula), Some(name)) = (parts.get(cellar + 1), exe.file_name()) else { return exe.to_path_buf() };
    let prefix: PathBuf = parts[..cellar].iter().collect();
    let opt = prefix.join("opt").join(formula).join("bin").join(name);
    if opt.exists() { opt } else { exe.to_path_buf() }
}

/// Undoes `install`.
pub fn uninstall() -> Result<()> {
    let home = PathBuf::from(std::env::var_os("HOME").context("no HOME")?);
    if cfg!(target_os = "macos") {
        let plist = home.join("Library/LaunchAgents").join(format!("{LAUNCHD_LABEL}.plist"));
        if !plist.exists() {
            println!("rummage watch wasn't set to start at login");
            return Ok(());
        }
        let domain = format!("gui/{}", unsafe { getuid() });
        let _ = Command::new("launchctl").args(["bootout", &domain]).arg(&plist).stderr(Stdio::null()).status();
        std::fs::remove_file(&plist)?;
    } else {
        let unit = home.join(".config/systemd/user").join(SYSTEMD_UNIT);
        if !unit.exists() {
            println!("rummage watch wasn't set to start at login");
            return Ok(());
        }
        let _ = Command::new("systemctl").args(["--user", "disable", "--now", SYSTEMD_UNIT]).status();
        std::fs::remove_file(&unit)?;
        let _ = Command::new("systemctl").args(["--user", "daemon-reload"]).status();
    }
    println!("rummage watch no longer starts at login (searches still start the service when needed)");
    Ok(())
}

const LAUNCHD_LABEL: &str = "io.github.rummage.watch";
const SYSTEMD_UNIT: &str = "rummage-watch.service";

unsafe extern "C" {
    fn getuid() -> u32;
}

fn xml(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(kind: EventKind, path: &str) -> notify::Result<notify::Event> {
        Ok(notify::Event::new(kind).add_path(PathBuf::from(path)))
    }

    #[test]
    fn only_changes_rummage_might_index_count() {
        let roots = [PathBuf::from("/home/a/.cache/notes")];
        let index = Path::new("/home/a/.local/share/rummage");
        let create = EventKind::Create(notify::event::CreateKind::File);
        assert!(
            matters(&event(create, "/home/a/.cache/notes/trip.md"), &roots, index),
            "hidden above the folder is fine"
        );
        assert!(!matters(&event(create, "/home/a/.cache/notes/.git/index"), &roots, index), "hidden below it isn't");
        assert!(!matters(&event(create, "/home/a/.cache/notes/film.mp4.part"), &roots, index), "still downloading");
        assert!(!matters(&event(create, "/home/a/.cache/notes/film.mkv.!qB"), &roots, index), "qBittorrent's");
        assert!(
            !matters(&event(create, "/home/a/.cache/notes/report.pdf.download/report.pdf"), &roots, index),
            "Safari's"
        );
        assert!(!matters(&event(create, "/home/a/.cache/notes/~$budget.xlsx"), &roots, index), "Office's lock");
        assert!(!matters(&event(create, "/home/a/.local/share/rummage/index.sqlite-wal"), &roots, index), "the index");
        let read = EventKind::Access(notify::event::AccessKind::Read);
        assert!(!matters(&event(read, "/home/a/.cache/notes/trip.md"), &roots, index), "a read changes nothing");
    }

    #[test]
    fn login_items_point_past_homebrew_versions() {
        let brew = std::env::temp_dir().join(format!("rummage-brew-{}", std::process::id()));
        let versioned = brew.join("Cellar/rummage/0.2.0/bin/rummage");
        std::fs::create_dir_all(versioned.parent().unwrap()).unwrap();
        std::fs::write(&versioned, "").unwrap();
        // No opt link (not Homebrew after all): the path as it is.
        assert_eq!(stable_path(&versioned), versioned);
        let opt = brew.join("opt/rummage/bin/rummage");
        std::fs::create_dir_all(opt.parent().unwrap()).unwrap();
        std::fs::write(&opt, "").unwrap();
        assert_eq!(stable_path(&versioned), opt);
        // Built from source: unchanged.
        assert_eq!(
            stable_path(Path::new("/home/a/rummage/target/release/rummage")),
            Path::new("/home/a/rummage/target/release/rummage")
        );
        std::fs::remove_dir_all(&brew).unwrap();
    }

    #[test]
    fn one_indexing_run_at_a_time() {
        let dir = std::env::temp_dir().join(format!("rummage-lock-{}", std::process::id()));
        let held = lock_index(&dir, true).unwrap();
        let second = File::create(dir.join("index.lock")).unwrap();
        assert!(second.try_lock().is_err(), "a second run waits");
        drop(held);
        assert!(second.try_lock().is_ok(), "and goes once the first is done");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
