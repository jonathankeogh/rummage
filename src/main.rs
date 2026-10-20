use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use rummage::crawl;
use rummage::extract;
use rummage::index;
use rummage::paths::{self, ModelFiles, tilde};
use rummage::serve;
use rummage::store::{self, Kind, Store};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// Describe a file, find it: local multimodal search for the terminal.
#[derive(Parser)]
#[command(name = "rummage", version)]
struct Cli {
    /// Index folder [default: $XDG_DATA_HOME/rummage or ~/.local/share/rummage]
    #[arg(long, global = true)]
    index: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Add folders and bring the index up to date (no folders: the ones added before)
    Index {
        folders: Vec<PathBuf>,
        /// Also read source code in code projects (find it with "code: …")
        #[arg(long)]
        code: bool,
        /// Don't index this folder (repeatable)
        #[arg(long)]
        exclude: Vec<PathBuf>,
        /// Drop every vector and embed everything again
        #[arg(long)]
        reindex: bool,
        /// Run for the search service: summary lines only, no progress counter
        #[arg(long, hide = true)]
        background: bool,
    },
    /// Search by meaning and words. Start with "code:" to search code
    Search {
        #[arg(required = true)]
        words: Vec<String>,
        #[arg(short = 'n', default_value_t = 10)]
        limit: usize,
        /// Only these kinds: doc, pdf, code (comma-separated) [default: doc,pdf]
        #[arg(long, value_delimiter = ',')]
        kind: Vec<String>,
        /// How far a meaning match must stand out
        #[arg(long, default_value_t = 2.0)]
        zgate: f64,
        #[arg(long)]
        json: bool,
        /// Search in this process instead of through the background service
        #[arg(long)]
        no_serve: bool,
        /// Meaning and indexed words only: don't grep the files
        #[arg(long)]
        no_grep: bool,
    },
    /// What's in the index
    Status,
    /// Keep the model loaded so searches are instant (searches start this in the background on their own)
    Serve {
        /// Quit after this many idle minutes (0: never)
        #[arg(long, default_value_t = 15)]
        linger: u64,
    },
    /// Stop the background service
    Stop,
    /// Keep the index up to date at all times: the search service, never quitting (it indexes new and changed files
    /// whenever it runs; this keeps it running)
    Watch {
        /// Start it at login (launchd on macOS, a systemd user service on Linux)
        #[arg(long, conflicts_with = "uninstall")]
        install: bool,
        /// Stop starting it at login
        #[arg(long)]
        uninstall: bool,
    },
}

/// `rummage search … | head` stops quietly when `head` has read enough, as other Unix tools do (Rust otherwise ignores
/// SIGPIPE and the next print panics). Only for commands that print results: the service and the interface write to
/// sockets and must outlive a peer that goes away.
fn quiet_when_piped_output_closes() {
    unsafe extern "C" {
        fn signal(signum: i32, handler: usize) -> usize;
    }
    const SIGPIPE: i32 = 13;
    const SIG_DFL: usize = 0;
    unsafe { signal(SIGPIPE, SIG_DFL) };
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rummage: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    if matches!(cli.command, Some(Command::Search { .. } | Command::Status)) {
        quiet_when_piped_output_closes();
    }
    let dir = cli.index.clone().unwrap_or_else(paths::index_dir);
    match cli.command {
        Some(Command::Index { folders, code, exclude, reindex, background }) => {
            index_command(&dir, folders, code, exclude, reindex, background)
        }
        Some(Command::Search { words, limit, kind, zgate, json, no_serve, no_grep }) => {
            search_command(&dir, words, limit, kind, zgate, json, no_serve, no_grep)
        }
        Some(Command::Status) => status(&dir),
        Some(Command::Serve { linger }) => {
            serve::serve(&dir, (linger > 0).then(|| std::time::Duration::from_secs(linger * 60)))
        }
        Some(Command::Stop) => match serve::Client::connect(&dir) {
            Ok(mut client) => {
                client.request(&serve::Request::Quit)?;
                eprintln!("stopped the search service");
                Ok(())
            }
            Err(_) => {
                eprintln!("the search service isn't running");
                Ok(())
            }
        },
        Some(Command::Watch { install: true, .. }) => rummage::watch::install(&dir),
        Some(Command::Watch { uninstall: true, .. }) => rummage::watch::uninstall(),
        Some(Command::Watch { .. }) => {
            // A service a search started quits when idle: this one takes its place and stays.
            if let Ok(mut client) = serve::Client::connect(&dir) {
                client.request(&serve::Request::Quit).ok();
                for _ in 0..50 {
                    if serve::Client::connect(&dir).is_err() {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
            serve::serve(&dir, None)
        }
        None => interface(&dir),
    }
}

fn index_command(
    dir: &PathBuf,
    folders: Vec<PathBuf>,
    code: bool,
    exclude: Vec<PathBuf>,
    reindex: bool,
    background: bool,
) -> Result<()> {
    // One indexing run at a time: yours, or the search service's keeping up with changes.
    let _lock = rummage::watch::lock_index(dir, background)?;
    let store = if reindex { store::open_for_reindex(dir)? } else { Store::open(dir)? };
    for folder in &folders {
        let folder = folder.canonicalize().with_context(|| format!("no folder {}", folder.display()))?;
        if !folder.is_dir() {
            bail!("{} isn't a folder", folder.display());
        }
        store.add_root(&folder)?;
    }
    if code {
        store.set_meta("code", "1")?;
    }
    let roots = store.roots()?;
    if roots.is_empty() {
        bail!("no folders yet: rummage index <folder>...");
    }
    // Left out for good: later runs (and the service's) leave them out too.
    store.add_excluded(&exclude.iter().filter_map(|e| e.canonicalize().ok()).collect::<Vec<_>>())?;
    let options =
        crawl::Options { code: code || store.meta("code")?.as_deref() == Some("1"), excluded: store.excluded()? };
    let started = Instant::now();
    let sync = index::sync(&store, &roots, &options)?;
    eprintln!(
        "{} files in {} folder{}: {} new or changed, {} gone ({})",
        sync.found,
        roots.len(),
        if roots.len() == 1 { "" } else { "s" },
        sync.new_or_changed,
        sync.removed,
        index::duration(started)
    );
    let started = Instant::now();
    if sync.media_without_ffmpeg > 0 {
        eprintln!(
            "left out {} recordings and videos: {}, then run rummage index again",
            sync.media_without_ffmpeg,
            rummage::media::HINT
        );
    }
    if sync.heic_without_tool > 0 {
        eprintln!(
            "left out {} HEIC photos: {}, then run rummage index again",
            sync.heic_without_tool,
            rummage::picture::HEIC_HINT
        );
    }
    if sync.small_pictures + sync.dataset_files > 0 {
        eprintln!(
            "left out {} icons and thumbnails (under 256 px) and {} pictures in dataset folders",
            sync.small_pictures, sync.dataset_files
        );
    }
    let pending_count = |kinds: &str| -> Result<i64> {
        Ok(store.db.query_row(
            &format!("SELECT COUNT(*) FROM files WHERE state = 'pending' AND kind IN ({kinds})"),
            [],
            |r| r.get(0),
        )?)
    };
    let pictures_pending = pending_count("'image', 'screenshot', 'audio', 'video'")?;
    if pictures_pending > 0 && ModelFiles::locate().is_ok_and(|m| m.media.is_none()) {
        eprintln!(
            "{pictures_pending} pictures, recordings and videos wait for the image and audio encoder (555 MB): uvx --from huggingface_hub hf download {} {} --revision {}",
            paths::MODEL_REPOSITORY,
            paths::MEDIA_MODEL,
            paths::MODEL_REVISION
        );
    }
    if pending_count("'screenshot'")? > 0 && !rummage::ocr::available() {
        eprintln!("screenshots are read by what they show only; for their words, {}", rummage::ocr::HINT);
    }
    if sync.pdfs_without_tools > 0 {
        eprintln!(
            "left out {} PDFs: {}, then run rummage index again",
            sync.pdfs_without_tools,
            extract::PDF_TOOLS_HINT
        );
    }
    let mut first_pass_over = false;
    let done = index::embed_pending(&store, |progress| {
        if background {
            // The service's log gets the summary, not a counter redrawn in place.
            if let index::Progress::Later { .. } = progress {
                first_pass_over = true;
            }
            return;
        }
        // Just the name: a full path wraps in a narrow terminal and the line can't be redrawn.
        let short = |path: &std::path::Path| -> String {
            path.file_name().map(|n| n.to_string_lossy().chars().take(60).collect()).unwrap_or_default()
        };
        match progress {
            index::Progress::File { done, total, path } => eprint!("\r\x1b[K[{done}/{total}] {}", short(path)),
            index::Progress::Later { path, kind, done, total } => {
                if !first_pass_over {
                    first_pass_over = true;
                    eprintln!(
                        "\r\x1b[Keverything is searchable; reading the rest of long files ({})",
                        index::duration(started)
                    );
                }
                eprint!("\r\x1b[K{} {} of {} {}s", short(path), done, total, kind.unit());
            }
        }
        std::io::stderr().flush().ok();
    })?;
    if done > 0 || first_pass_over {
        eprintln!("\r\x1b[Kindexed {done} files in {}", index::duration(started));
    }
    Ok(())
}

fn search_command(
    dir: &PathBuf,
    words: Vec<String>,
    limit: usize,
    kind: Vec<String>,
    zgate: f64,
    json: bool,
    no_serve: bool,
    no_grep: bool,
) -> Result<()> {
    let query = words.join(" ");
    let kinds: Vec<Kind> =
        kind.iter().map(|k| Kind::parse(k).with_context(|| format!("unknown kind {k}"))).collect::<Result<_>>()?;
    let kinds = (!kinds.is_empty()).then_some(kinds);
    let started = Instant::now();
    let (hits, timing) = if no_serve {
        let mut engine = serve::Engine::open(dir)?;
        let loaded = started.elapsed();
        let hits = if no_grep {
            engine.search_start(&query, kinds.as_deref(), limit, zgate, false)?.0
        } else {
            engine.search(&query, kinds.as_deref(), limit, zgate)?
        };
        let timing = format!(
            "{} vectors · start {:.2} s · search {:.0} ms, in this process",
            engine.vector_count(),
            loaded.as_secs_f64(),
            (started.elapsed() - loaded).as_secs_f64() * 1000.0
        );
        (hits, timing)
    } else {
        let mut client = serve::Client::connect_or_start(dir)?;
        let connected = started.elapsed();
        let response = client.search(&query, kinds, limit, zgate, !no_grep)?;
        let mut timing = format!(
            "{} vectors · search {:.0} ms in the service, {:.0} ms here",
            response.vectors,
            response.millis,
            (started.elapsed() - connected).as_secs_f64() * 1000.0
        );
        if connected.as_secs_f64() > 0.2 {
            timing += &format!(" · started the service in {:.1} s", connected.as_secs_f64());
        }
        (response.hits, timing)
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&hits)?);
        return Ok(());
    }
    if hits.is_empty() {
        println!("no good matches");
    }
    for (rank, hit) in hits.iter().enumerate() {
        let path = PathBuf::from(&hit.path);
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let place = match (hit.kind, hit.loc) {
            _ if hit.line.is_some() => format!("  [line {}]", hit.line.unwrap_or(0)),
            (Kind::Pdf, Some(loc)) => format!("  [page {loc:.0}]"),
            (kind, Some(loc)) if kind.is_timed() => format!("  [@{}]", rummage::media::clock(loc)),
            (_, Some(loc)) if loc > 1.0 => format!("  [passage {loc:.0}]"),
            _ => String::new(),
        };
        let why = [
            hit.z.filter(|z| *z >= zgate).map(|z| format!("z {z:.1}")),
            hit.keyword_rank.map(|r| format!("words #{r}")),
            hit.grep.then(|| "grep".to_string()),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(", ");
        println!("{:>2}. {:<5} {name}{place}   ({why})", rank + 1, hit.kind.as_str());
        println!("      {}", tilde(path.parent().unwrap_or(&path)));
        println!("      “{}”", hit.excerpt.chars().take(160).collect::<String>());
    }
    eprintln!("({timing})");
    Ok(())
}

/// `rummage` alone: the search interface, on this index's background service.
fn interface(dir: &PathBuf) -> Result<()> {
    if !std::io::IsTerminal::is_terminal(&std::io::stdout()) {
        bail!("the interface needs a terminal; for scripts use `rummage search … --json`");
    }
    match Store::open(dir) {
        Ok(store) if store.roots()?.is_empty() => {
            bail!("nothing indexed yet: rummage index <folder>... (rummage --help for more)")
        }
        Ok(_) => {}
        // Made by another version: the service rebuilds it, and the interface shows how far it's got.
        Err(error) if error.is::<store::StaleIndex>() => {}
        Err(error) => return Err(error),
    }
    rummage::tui::run(serve::ServiceBackend::connect(dir)?)
}

fn status(dir: &PathBuf) -> Result<()> {
    // An index that can't be used (made by another version) still leaves the rest worth showing.
    match Store::open(dir) {
        Ok(store) => index_status(dir, &store)?,
        Err(error) => println!("index: {}\n  {error:#}", tilde(dir)),
    }
    match ModelFiles::locate() {
        Ok(models) => println!("model files: {}", tilde(models.text.parent().unwrap_or(&models.text))),
        Err(error) => println!("model files: missing\n{error}"),
    }
    let backend = if cfg!(target_os = "macos") {
        "Metal"
    } else if cfg!(feature = "cuda") {
        "CUDA"
    } else if cfg!(feature = "vulkan") {
        "Vulkan"
    } else {
        "CPU"
    };
    println!("runs on: {backend}");
    println!("tools (optional; each adds a kind of file or a way to open one):");
    for (found, name, adds, hint) in tools() {
        match found {
            Some(which) => println!("  {name:<10} {:<17} {adds}", format!("found {which}")),
            None => println!("  {name:<10} {:<17} {adds}: {hint}", "missing"),
        }
    }
    Ok(())
}

/// The index's folders, files by kind and state, size and service.
fn index_status(dir: &PathBuf, store: &Store) -> Result<()> {
    println!("index: {}", tilde(&store.file));
    let roots = store.roots()?;
    println!(
        "folders: {}",
        if roots.is_empty() {
            "none yet".into()
        } else {
            roots.iter().map(|r| tilde(r)).collect::<Vec<_>>().join(", ")
        }
    );
    let segments: i64 = store.db.query_row("SELECT COUNT(*) FROM segments", [], |r| r.get(0))?;
    println!("vectors: {segments}");
    let mut statement =
        store.db.prepare("SELECT kind, state, COUNT(*) FROM files GROUP BY kind, state ORDER BY kind, state")?;
    let rows =
        statement.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?)))?;
    for row in rows {
        let (kind, state, count) = row?;
        println!("  {kind:<10} {state:<8} {count}");
    }
    println!("model: {}", store.meta("fingerprint")?.unwrap_or_default());
    let size = std::fs::metadata(&store.file).map(|m| m.len()).unwrap_or(0);
    println!("size: {:.1} MB", size as f64 / 1e6);
    match serve::Client::connect(dir).and_then(|mut c| c.request(&serve::Request::Ping)) {
        Ok(ping) => {
            let folders = if ping.watching == 1 { "folder" } else { "folders" };
            print!("search service: running, watching {} {folders} for changes", ping.watching);
            match ping.indexing {
                Some(left) => println!("; indexing, {left} files to go"),
                None => println!(),
            }
            if let Some(problem) = ping.watch_problem {
                println!("  {problem}");
            }
        }
        Err(_) => println!("search service: not running (a search starts it, and it catches up with changes then)"),
    }
    Ok(())
}

/// The programs rummage uses when they're there: whether each is found (and which), what it adds, how to get it.
fn tools() -> Vec<(Option<String>, &'static str, &'static str, &'static str)> {
    let first = |names: &[&str]| names.iter().find(|n| paths::on_path(n)).map(|n| n.to_string());
    let mut tools = vec![
        (
            extract::pdf_tools().then(|| "pdftotext".to_string()),
            "poppler",
            "PDFs, page by page",
            extract::PDF_TOOLS_HINT,
        ),
        (
            rummage::ocr::available().then(|| "tesseract".to_string()),
            "tesseract",
            "words in screenshots and scans",
            rummage::ocr::HINT,
        ),
        (
            rummage::media::available().then(|| "ffmpeg".to_string()),
            "ffmpeg",
            "recordings and videos",
            rummage::media::HINT,
        ),
        (
            rummage::picture::heic_tool().map(String::from),
            "libheif",
            "iPhone photos (HEIC)",
            rummage::picture::HEIC_HINT,
        ),
        (
            first(&["mpv", "ffplay"]),
            "player",
            "listening in the interface (^s)",
            "install mpv (brew install mpv, apt install mpv), or ffmpeg's ffplay",
        ),
        (
            first(&["mpv", "vlc"]),
            "mpv, vlc",
            "opening a recording at the moment (⏎)",
            "install mpv or VLC; otherwise it opens from the start",
        ),
    ];
    if !cfg!(target_os = "macos") {
        tools.push((
            first(&["zathura", "evince", "okular"]),
            "viewer",
            "opening a PDF at the page (⏎)",
            "install zathura, evince or okular; otherwise it opens at the first page",
        ));
    }
    tools
}
