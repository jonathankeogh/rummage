use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use rummage::crawl;
use rummage::embed::{Embedder, prompts};
use rummage::index;
use rummage::paths::{self, ModelFiles, tilde};
use rummage::search::{self, Vectors};
use rummage::store::{self, Kind, Store};
use std::collections::HashSet;
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
    },
    /// Search by meaning and words. Start with "code:" to search code
    Search {
        #[arg(required = true)]
        words: Vec<String>,
        #[arg(short = 'n', default_value_t = 10)]
        limit: usize,
        /// Only these kinds: doc, code (comma-separated)
        #[arg(long, value_delimiter = ',')]
        kind: Vec<String>,
        /// How far a meaning match must stand out
        #[arg(long, default_value_t = 2.0)]
        zgate: f64,
        #[arg(long)]
        json: bool,
    },
    /// What's in the index
    Status,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rummage: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let dir = cli.index.clone().unwrap_or_else(paths::index_dir);
    match cli.command {
        Some(Command::Index { folders, code, exclude, reindex }) => {
            index_command(&dir, folders, code, exclude, reindex)
        }
        Some(Command::Search { words, limit, kind, zgate, json }) => {
            search_command(&dir, words, limit, kind, zgate, json)
        }
        Some(Command::Status) => status(&dir),
        None => {
            // The interface comes in milestone 3.
            use clap::CommandFactory;
            Cli::command().print_help()?;
            Ok(())
        }
    }
}

fn index_command(dir: &PathBuf, folders: Vec<PathBuf>, code: bool, exclude: Vec<PathBuf>, reindex: bool) -> Result<()> {
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
    let options = crawl::Options {
        code: code || store.meta("code")?.as_deref() == Some("1"),
        excluded: exclude.iter().filter_map(|e| e.canonicalize().ok()).collect(),
    };
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
    let done = index::embed_pending(&store, |done, total, path| {
        eprint!("\r\x1b[K[{done}/{total}] {}", tilde(path));
        std::io::stderr().flush().ok();
    })?;
    if done > 0 {
        eprintln!("\r\x1b[Kembedded {done} files in {}", index::duration(started));
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
) -> Result<()> {
    let mut query = words.join(" ");
    let code = query.trim_start().to_lowercase().starts_with("code:");
    if code {
        query = query.trim_start()[5..].trim().to_string();
    }
    let kinds: HashSet<Kind> = if code {
        [Kind::Code].into()
    } else if kind.is_empty() {
        [Kind::Doc].into()
    } else {
        kind.iter().map(|k| Kind::parse(k).with_context(|| format!("unknown kind {k}"))).collect::<Result<_>>()?
    };
    let store = Store::open(dir)?;
    let started = Instant::now();
    let vectors = Vectors::load(&store)?;
    let loaded = started.elapsed();
    let mut embedder = Embedder::load(&ModelFiles::locate()?, true)?;
    let encoder_ready = started.elapsed();
    let vector = embedder.embed_texts(&[prompts::query(&query)])?.pop();
    let encoded = started.elapsed();
    let options = search::Options { kinds: Some(kinds), limit, z_gate: zgate, ..Default::default() };
    let hits = search::search(&store, &vectors, &query, vector.as_deref(), &options)?;
    let searched = started.elapsed();
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
            (_, Some(loc)) if loc > 1.0 => format!("  [passage {loc:.0}]"),
            _ => String::new(),
        };
        let why = [
            hit.z.filter(|z| *z >= zgate).map(|z| format!("z {z:.1}")),
            hit.keyword_rank.map(|r| format!("words #{r}")),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(", ");
        println!("{:>2}. {:<5} {name}{place}   ({why})", rank + 1, hit.kind.as_str());
        println!("      {}", tilde(path.parent().unwrap_or(&path)));
        println!("      “{}”", hit.excerpt.chars().take(160).collect::<String>());
    }
    eprintln!(
        "({} vectors · load {:.0} ms · model start {:.2} s · query {:.0} ms · search {:.0} ms)",
        vectors.len(),
        loaded.as_secs_f64() * 1000.0,
        (encoder_ready - loaded).as_secs_f64(),
        (encoded - encoder_ready).as_secs_f64() * 1000.0,
        (searched - encoded).as_secs_f64() * 1000.0
    );
    Ok(())
}

fn status(dir: &PathBuf) -> Result<()> {
    let store = Store::open(dir)?;
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
        println!("  {kind:<5} {state:<8} {count}");
    }
    println!("model: {}", store.meta("fingerprint")?.unwrap_or_default());
    match ModelFiles::locate() {
        Ok(models) => println!("model files: {}", tilde(models.text.parent().unwrap_or(&models.text))),
        Err(error) => println!("model files: missing\n{error}"),
    }
    let size = std::fs::metadata(&store.file).map(|m| m.len()).unwrap_or(0);
    println!("size: {:.1} MB", size as f64 / 1e6);
    Ok(())
}
