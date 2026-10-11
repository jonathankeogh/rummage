//! `rummage serve`: keeps the query model and every vector in memory, so a search takes milliseconds, not the second
//! it takes to load the model. Clients talk to it over a Unix socket, one JSON object per line each way:
//!
//! ```text
//! {"op":"search","query":"dog on the beach","limit":10}   → {"ok":true,"hits":[…],"vectors":333,"millis":12.5}
//! {"op":"ping"}                                           → {"ok":true,"vectors":333}
//! {"op":"quit"}                                           → {"ok":true}
//! ```
//!
//! One service per index. `rummage search` (and the interface) start it when it isn't running; it quits after
//! `linger` without a request. It notices when `rummage index` changes the index and reloads the vectors.

use crate::embed::{Embedder, prompts};
use crate::paths::ModelFiles;
use crate::search::{self, Hit, Vectors};
use crate::store::{Kind, Store};
use anyhow::{Context, Result, bail};
use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(serde::Serialize, serde::Deserialize, Debug)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum Request {
    Ping,
    Search {
        query: String,
        #[serde(default = "default_limit")]
        limit: usize,
        /// Only these kinds; none = documents and PDFs (or code for a "code: …" query).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        kinds: Option<Vec<Kind>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        zgate: Option<f64>,
    },
    Quit,
}

fn default_limit() -> usize {
    10
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Default)]
pub struct Response {
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hits: Vec<Hit>,
    #[serde(default)]
    pub vectors: usize,
    /// Time spent on the request inside the service.
    #[serde(default)]
    pub millis: f64,
}

/// What a search needs, loaded once: the index, its vectors and the query model.
pub struct Engine {
    store: Store,
    vectors: Vectors,
    embedder: Embedder,
    data_version: i64,
}

impl Engine {
    pub fn open(dir: &Path) -> Result<Engine> {
        let store = Store::open(dir)?;
        let vectors = Vectors::load(&store)?;
        let embedder = Embedder::load(&ModelFiles::locate()?, true)?;
        let data_version = data_version(&store)?;
        Ok(Engine { store, vectors, embedder, data_version })
    }

    pub fn vector_count(&self) -> usize {
        self.vectors.len()
    }

    /// Reloads the vectors when another process (`rummage index`) has changed the index since the last look.
    fn refresh(&mut self) -> Result<()> {
        let version = data_version(&self.store)?;
        if version != self.data_version {
            self.vectors = Vectors::load(&self.store)?;
            self.data_version = version;
        }
        Ok(())
    }

    /// A search as typed: "code: …" searches code; otherwise `kinds`, or documents and PDFs when none are given.
    pub fn search(&mut self, text: &str, kinds: Option<&[Kind]>, limit: usize, z_gate: f64) -> Result<Vec<Hit>> {
        self.refresh()?;
        let (query, code) = search::split_code_prefix(text);
        let kinds: HashSet<Kind> = match (code, kinds) {
            (true, _) => [Kind::Code].into(),
            (false, Some(kinds)) if !kinds.is_empty() => kinds.iter().copied().collect(),
            _ => [Kind::Doc, Kind::Pdf].into(),
        };
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let vector = self.embedder.embed_texts(&[prompts::query(query)])?.pop();
        let options = search::Options { kinds: Some(kinds), limit, z_gate, ..Default::default() };
        search::search(&self.store, &self.vectors, query, vector.as_deref(), &options)
    }
}

/// SQLite's counter that moves whenever another connection commits to the file.
fn data_version(store: &Store) -> Result<i64> {
    Ok(store.db.query_row("PRAGMA data_version", [], |r| r.get(0))?)
}

/// The service's socket for this index: in `$XDG_RUNTIME_DIR` or the temp folder, named after the index folder (a
/// socket path must stay under ~100 bytes, so the index folder can't hold it).
pub fn socket_path(dir: &Path) -> PathBuf {
    let base = std::env::var_os("XDG_RUNTIME_DIR").filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = base.unwrap_or_else(std::env::temp_dir);
    let user = std::env::var("USER").unwrap_or_default();
    let dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    // FNV-1a: stable across builds, unlike std's hasher.
    let hash = format!("{user}\0{}", dir.display())
        .bytes()
        .fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    base.join(format!("rummage-{:012x}.sock", hash & 0xffff_ffff_ffff))
}

/// Runs the service in the foreground until `quit`, or until `linger` passes without a request.
pub fn serve(dir: &Path, linger: Option<Duration>) -> Result<()> {
    let socket = socket_path(dir);
    if UnixStream::connect(&socket).is_ok() {
        eprintln!("rummage serve: already running for this index ({})", socket.display());
        return Ok(());
    }
    let started = Instant::now();
    let engine = Engine::open(dir)?;
    // Bind only once the model is loaded: a client that can connect gets an answer.
    std::fs::remove_file(&socket).ok();
    let listener = UnixListener::bind(&socket).with_context(|| format!("can't listen on {}", socket.display()))?;
    std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600))?;
    eprintln!(
        "rummage serve: {} vectors, ready in {:.1} s on {}",
        engine.vector_count(),
        started.elapsed().as_secs_f64(),
        socket.display()
    );

    let engine = Arc::new(Mutex::new(engine));
    let last_request = Arc::new(AtomicU64::new(now_seconds()));
    if let Some(linger) = linger {
        let (last_request, socket) = (last_request.clone(), socket.clone());
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_secs(5));
                if now_seconds().saturating_sub(last_request.load(Ordering::Relaxed)) >= linger.as_secs() {
                    eprintln!("rummage serve: idle for {} s, quitting", linger.as_secs());
                    quit(&socket);
                }
            }
        });
    }
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let (engine, last_request, socket) = (engine.clone(), last_request.clone(), socket.clone());
        std::thread::spawn(move || {
            if let Err(error) = handle(stream, &engine, &last_request, &socket) {
                eprintln!("rummage serve: {error:#}");
            }
        });
    }
    Ok(())
}

fn handle(stream: UnixStream, engine: &Mutex<Engine>, last_request: &AtomicU64, socket: &Path) -> Result<()> {
    let mut writer = stream.try_clone()?;
    for line in BufReader::new(stream).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        last_request.store(now_seconds(), Ordering::Relaxed);
        let started = Instant::now();
        let mut response = match serde_json::from_str::<Request>(&line) {
            Err(error) => Response { error: Some(format!("bad request: {error}")), ..Default::default() },
            Ok(Request::Quit) => {
                writeln!(writer, "{}", serde_json::to_string(&Response { ok: true, ..Default::default() })?)?;
                quit(socket);
            }
            Ok(Request::Ping) => {
                let engine = engine.lock().unwrap();
                Response { ok: true, vectors: engine.vector_count(), ..Default::default() }
            }
            Ok(Request::Search { query, limit, kinds, zgate }) => {
                let mut engine = engine.lock().unwrap();
                match engine.search(&query, kinds.as_deref(), limit, zgate.unwrap_or(2.0)) {
                    Ok(hits) => Response { ok: true, hits, vectors: engine.vector_count(), ..Default::default() },
                    Err(error) => Response { error: Some(format!("{error:#}")), ..Default::default() },
                }
            }
        };
        response.millis = started.elapsed().as_secs_f64() * 1000.0;
        writeln!(writer, "{}", serde_json::to_string(&response)?)?;
    }
    Ok(())
}

/// Removes the socket and exits at once. Not `std::process::exit`: that runs C++ static destructors, and llama.cpp's
/// Metal backend asserts when its device is torn down while the model (held by another thread) still uses it. The OS
/// frees everything anyway.
fn quit(socket: &Path) -> ! {
    unsafe extern "C" {
        fn _exit(status: i32) -> !;
    }
    std::fs::remove_file(socket).ok();
    unsafe { _exit(0) }
}

fn now_seconds() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// A connection to the service.
pub struct Client {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
}

/// How long a client waits for a service it started: loading the model takes under a second, but the first run on a
/// GPU compiles its kernels (about 15 s with Metal).
const START_TIMEOUT: Duration = Duration::from_secs(60);

impl Client {
    /// Connects to this index's service, starting it in the background first when it isn't running.
    pub fn connect_or_start(dir: &Path) -> Result<Client> {
        if let Ok(client) = Client::connect(dir) {
            return Ok(client);
        }
        start_in_background(dir)?;
        let deadline = Instant::now() + START_TIMEOUT;
        loop {
            if let Ok(mut client) = Client::connect(dir) {
                if client.request(&Request::Ping).is_ok_and(|r| r.ok) {
                    return Ok(client);
                }
            }
            if Instant::now() > deadline {
                bail!(
                    "the search service didn't start within {} s; see {}",
                    START_TIMEOUT.as_secs(),
                    log_path(dir).display()
                );
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Connects to this index's service if it's running.
    pub fn connect(dir: &Path) -> Result<Client> {
        let stream = UnixStream::connect(socket_path(dir))?;
        Ok(Client { reader: BufReader::new(stream.try_clone()?), writer: stream })
    }

    pub fn request(&mut self, request: &Request) -> Result<Response> {
        writeln!(self.writer, "{}", serde_json::to_string(request)?)?;
        let mut line = String::new();
        if self.reader.read_line(&mut line)? == 0 {
            bail!("the search service closed the connection");
        }
        let response: Response = serde_json::from_str(&line)?;
        if let Some(error) = &response.error {
            bail!("{error}");
        }
        Ok(response)
    }

    pub fn search(&mut self, query: &str, kinds: Option<Vec<Kind>>, limit: usize, zgate: f64) -> Result<Response> {
        self.request(&Request::Search { query: query.to_string(), limit, kinds, zgate: Some(zgate) })
    }
}

/// The interface's search backend: this index's service. It reconnects (restarting the service if needed) when a
/// request fails, since an interface left open longer than the service's linger outlives it.
pub struct ServiceBackend {
    dir: PathBuf,
    client: Option<Client>,
}

impl ServiceBackend {
    /// Connects now (starting the service if needed), so problems show before the interface takes the screen.
    pub fn connect(dir: &Path) -> Result<ServiceBackend> {
        Ok(ServiceBackend { dir: dir.to_path_buf(), client: Some(Client::connect_or_start(dir)?) })
    }

    fn search_once(&mut self, query: &str, limit: usize) -> Result<Vec<Hit>> {
        let client = match &mut self.client {
            Some(client) => client,
            None => self.client.insert(Client::connect_or_start(&self.dir)?),
        };
        Ok(client.search(query, None, limit, 2.0)?.hits)
    }
}

impl crate::tui::backend::Backend for ServiceBackend {
    fn search(&mut self, query: &str, limit: usize) -> Result<Vec<Hit>> {
        self.search_once(query, limit).or_else(|_| {
            self.client = None;
            self.search_once(query, limit)
        })
    }
}

/// The service's log, next to the index.
pub fn log_path(dir: &Path) -> PathBuf {
    dir.join("serve.log")
}

/// Starts `rummage serve` for `dir` detached from this terminal (its own process group, so Ctrl-C here doesn't stop
/// it), logging to `serve.log`.
fn start_in_background(dir: &Path) -> Result<()> {
    use std::os::unix::process::CommandExt;
    std::fs::create_dir_all(dir)?;
    let log = std::fs::OpenOptions::new().create(true).append(true).open(log_path(dir))?;
    std::process::Command::new(std::env::current_exe()?)
        .arg("--index")
        .arg(dir)
        .arg("serve")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(log)
        .process_group(0)
        .spawn()
        .context("can't start the search service")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_are_one_json_line() {
        let request = Request::Search { query: "dog".into(), limit: 5, kinds: Some(vec![Kind::Doc]), zgate: None };
        let line = serde_json::to_string(&request).unwrap();
        assert_eq!(line, r#"{"op":"search","query":"dog","limit":5,"kinds":["doc"]}"#);
        let back: Request = serde_json::from_str(r#"{"op":"search","query":"dog"}"#).unwrap();
        assert!(matches!(back, Request::Search { limit: 10, kinds: None, zgate: None, .. }));
        assert!(matches!(serde_json::from_str::<Request>(r#"{"op":"ping"}"#).unwrap(), Request::Ping));
    }

    #[test]
    fn socket_paths_are_short_and_per_index() {
        let a = socket_path(Path::new("/some/very/long/index/folder/that/goes/on/and/on/and/on/for/a/while/rummage"));
        let b = socket_path(Path::new("/another/index"));
        assert_ne!(a, b);
        assert!(a.as_os_str().len() < 104, "{}", a.display());
    }
}
