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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(serde::Serialize, serde::Deserialize, Debug)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum Request {
    Ping,
    Search {
        /// Echoed in the responses, so a client can drop answers to a query it has moved on from.
        #[serde(default)]
        id: u64,
        query: String,
        #[serde(default = "default_limit")]
        limit: usize,
        /// Also grep the files: a second, final response follows with grep's matches merged in.
        #[serde(default = "yes", skip_serializing_if = "is_true")]
        grep: bool,
        /// Only these kinds; none = documents and PDFs (or code for a "code: …" query).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        kinds: Option<Vec<Kind>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        zgate: Option<f64>,
    },
    Quit,
}

fn yes() -> bool {
    true
}

fn is_true(b: &bool) -> bool {
    *b
}

fn default_limit() -> usize {
    10
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct Response {
    pub ok: bool,
    /// The search this answers.
    #[serde(default)]
    pub id: u64,
    /// The last response to its search: a search with grep gets a first one (meaning and words) and a final one.
    #[serde(rename = "final", default = "yes")]
    pub done: bool,
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

impl Default for Response {
    /// Not ok, and final: every response but a search's first is its last.
    fn default() -> Response {
        Response { ok: false, id: 0, done: true, error: None, hits: Vec::new(), vectors: 0, millis: 0.0 }
    }
}

/// What a search needs, loaded once: the index, its vectors and the query model.
pub struct Engine {
    store: Store,
    vectors: Vectors,
    embedder: Embedder,
    data_version: i64,
    /// Where typical queries sit, worked out once (about 50 ms) for the hub penalty.
    query_centre: search::QueryCentre,
    /// Moves on every reload of the vectors: a search's second phase only runs on the vectors of its first.
    generation: u64,
}

/// What a search's second phase needs: the first phase's evidence, and what to grep.
pub struct Pending {
    prepared: search::Prepared,
    generation: u64,
    /// The query as grep takes it (no "code:"), where, and for which kind of file.
    pub query: String,
    pub roots: Vec<PathBuf>,
    pub code: bool,
}

impl Engine {
    pub fn open(dir: &Path) -> Result<Engine> {
        let store = Store::open(dir)?;
        let mut vectors = Vectors::load(&store)?;
        let mut embedder = Embedder::load(&ModelFiles::locate()?, true)?;
        let typical: Vec<String> = search::TYPICAL_QUERIES.iter().map(|q| prompts::query(q)).collect();
        let query_centre = search::QueryCentre::from_vectors(&embedder.embed_texts(&typical)?);
        vectors.set_query_centre(query_centre.clone());
        let data_version = data_version(&store)?;
        Ok(Engine { store, vectors, embedder, data_version, query_centre, generation: 0 })
    }

    pub fn vector_count(&self) -> usize {
        self.vectors.len()
    }

    /// Reloads the vectors when another process (`rummage index`) has changed the index since the last look.
    fn refresh(&mut self) -> Result<()> {
        let version = data_version(&self.store)?;
        if version != self.data_version {
            self.vectors = Vectors::load(&self.store)?;
            self.vectors.set_query_centre(self.query_centre.clone());
            self.generation += 1;
            self.data_version = version;
        }
        Ok(())
    }

    /// A search as typed: "code: …" searches code; otherwise `kinds`, or documents and PDFs when none are given.
    /// A whole search, grep included (it waits for grep, up to `GREP_DEADLINE`).
    pub fn search(&mut self, text: &str, kinds: Option<&[Kind]>, limit: usize, z_gate: f64) -> Result<Vec<Hit>> {
        let (hits, pending) = self.search_start(text, kinds, limit, z_gate)?;
        let Some(pending) = pending else { return Ok(hits) };
        let deadline = Instant::now() + GREP_DEADLINE;
        let files = crate::crawl::grep_files(&pending.roots, pending.code);
        let grepped = crate::grep::grep(&files, &pending.query, crate::grep::LIMIT, deadline, &AtomicBool::new(false))?;
        Ok(self.search_finish(&pending, &grepped)?.unwrap_or(hits))
    }

    /// A search's first phase, meaning and words: its hits now, and what grep needs for the second (none when there's
    /// nothing to grep for). "code: …" searches code; otherwise `kinds`, or documents and PDFs when none are given.
    pub fn search_start(
        &mut self,
        text: &str,
        kinds: Option<&[Kind]>,
        limit: usize,
        z_gate: f64,
    ) -> Result<(Vec<Hit>, Option<Pending>)> {
        self.refresh()?;
        let (query, code) = search::split_code_prefix(text);
        let kinds: HashSet<Kind> = match (code, kinds) {
            (true, _) => [Kind::Code].into(),
            (false, Some(kinds)) if !kinds.is_empty() => kinds.iter().copied().collect(),
            _ => [Kind::Doc, Kind::Pdf, Kind::Image, Kind::Screenshot, Kind::Audio, Kind::Video].into(),
        };
        if query.is_empty() {
            return Ok((Vec::new(), None));
        }
        let vector = self.embedder.embed_texts(&[prompts::query(query)])?.pop();
        let options = search::Options { kinds: Some(kinds.clone()), limit, z_gate, ..Default::default() };
        let prepared = search::prepare(&self.store, &self.vectors, query, vector.as_deref(), &options)?;
        let hits = search::rank(&self.store, &self.vectors, &prepared, &[], search::now())?;
        // Grep reads text and code files; PDFs and pictures alone have nothing for it.
        let greppable = kinds.contains(&Kind::Doc) || kinds.contains(&Kind::Code);
        let pending = (greppable && crate::grep::pattern(query).is_some()).then(|| Pending {
            prepared,
            generation: self.generation,
            query: query.to_string(),
            roots: self.store.roots().unwrap_or_default(),
            code,
        });
        Ok((hits, pending))
    }

    /// A search's second phase: everything ranked again with grep's matches. None if the vectors were reloaded in
    /// between (the first phase's hits stand then).
    pub fn search_finish(&self, pending: &Pending, grepped: &[crate::grep::GrepHit]) -> Result<Option<Vec<Hit>>> {
        if pending.generation != self.generation {
            return Ok(None);
        }
        Ok(Some(search::rank(&self.store, &self.vectors, &pending.prepared, grepped, search::now())?))
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
    let files = Arc::new(GrepFiles::default());
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
        let (engine, files, last_request, socket) =
            (engine.clone(), files.clone(), last_request.clone(), socket.clone());
        std::thread::spawn(move || {
            if let Err(error) = handle(stream, &engine, &files, &last_request, &socket) {
                eprintln!("rummage serve: {error:#}");
            }
        });
    }
    Ok(())
}

/// How long grep may take before the final response goes out with what it has.
pub const GREP_DEADLINE: Duration = Duration::from_secs(3);
/// How long a listing of the files grep reads is reused before walking the folders again.
const FILE_LIST_FRESH: Duration = Duration::from_secs(60);

/// The files grep reads, listed per folder set and kind, kept for a minute: walking a big folder takes a while, and
/// searches come a keystroke apart.
#[derive(Default)]
pub struct GrepFiles {
    lists: Mutex<std::collections::HashMap<(Vec<PathBuf>, bool), (Instant, Arc<Vec<(PathBuf, Kind, i64)>>)>>,
}

impl GrepFiles {
    fn get(&self, roots: &[PathBuf], code: bool) -> Arc<Vec<(PathBuf, Kind, i64)>> {
        let key = (roots.to_vec(), code);
        if let Some((at, list)) = self.lists.lock().unwrap().get(&key) {
            if at.elapsed() < FILE_LIST_FRESH {
                return list.clone();
            }
        }
        let list = Arc::new(crate::crawl::grep_files(roots, code));
        self.lists.lock().unwrap().insert(key, (Instant::now(), list.clone()));
        list
    }
}

fn handle(
    stream: UnixStream,
    engine: &Mutex<Engine>,
    files: &Arc<GrepFiles>,
    last_request: &AtomicU64,
    socket: &Path,
) -> Result<()> {
    let mut writer = stream.try_clone()?;
    // Requests are read on their own thread, so a newer one can cancel a grep still running.
    let (lines_tx, lines) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in BufReader::new(stream).lines() {
            let Ok(line) = line else { break };
            if lines_tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut send =
        |response: &Response| -> Result<()> { Ok(writeln!(writer, "{}", serde_json::to_string(response)?)?) };
    let mut next: Option<String> = None;
    loop {
        let line = match next.take() {
            Some(line) => line,
            None => match lines.recv() {
                Ok(line) => line,
                Err(_) => return Ok(()),
            },
        };
        if line.trim().is_empty() {
            continue;
        }
        last_request.store(now_seconds(), Ordering::Relaxed);
        let started = Instant::now();
        let millis = || started.elapsed().as_secs_f64() * 1000.0;
        let request = match serde_json::from_str::<Request>(&line) {
            Ok(request) => request,
            Err(error) => {
                send(&Response { error: Some(format!("bad request: {error}")), ..Default::default() })?;
                continue;
            }
        };
        match request {
            Request::Quit => {
                send(&Response { ok: true, ..Default::default() })?;
                quit(socket);
            }
            Request::Ping => {
                let vectors = engine.lock().unwrap().vector_count();
                send(&Response { ok: true, vectors, ..Default::default() })?;
            }
            Request::Search { id, query, limit, grep, kinds, zgate } => {
                let started_search =
                    engine.lock().unwrap().search_start(&query, kinds.as_deref(), limit, zgate.unwrap_or(2.0));
                let vectors = engine.lock().unwrap().vector_count();
                let (hits, pending) = match started_search {
                    Ok(result) => result,
                    Err(error) => {
                        send(&Response { id, error: Some(format!("{error:#}")), ..Default::default() })?;
                        continue;
                    }
                };
                let pending = pending.filter(|_| grep);
                send(&Response {
                    ok: true,
                    id,
                    done: pending.is_none(),
                    hits,
                    vectors,
                    millis: millis(),
                    ..Default::default()
                })?;
                let Some(pending) = pending else { continue };

                // Second phase: grep on its own thread; a newer request cancels it (and gets no final for this one).
                let cancel = Arc::new(AtomicBool::new(false));
                let (done_tx, done) = std::sync::mpsc::channel();
                {
                    let (files, cancel) = (files.clone(), cancel.clone());
                    let (roots, code, query) = (pending.roots.clone(), pending.code, pending.query.clone());
                    std::thread::spawn(move || {
                        let deadline = Instant::now() + GREP_DEADLINE;
                        let list = files.get(&roots, code);
                        let _ = done_tx.send(crate::grep::grep(&list, &query, crate::grep::LIMIT, deadline, &cancel));
                    });
                }
                loop {
                    match done.recv_timeout(Duration::from_millis(10)) {
                        Ok(grepped) => {
                            let ranked = grepped.and_then(|g| engine.lock().unwrap().search_finish(&pending, &g));
                            let response = match ranked {
                                Ok(Some(hits)) => {
                                    Response { ok: true, id, hits, vectors, millis: millis(), ..Default::default() }
                                }
                                Ok(None) => Response { ok: true, id, vectors, millis: millis(), ..Default::default() },
                                Err(error) => Response { id, error: Some(format!("{error:#}")), ..Default::default() },
                            };
                            send(&response)?;
                            break;
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            if let Ok(line) = lines.try_recv() {
                                cancel.store(true, Ordering::Relaxed);
                                next = Some(line);
                                break;
                            }
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
            }
        }
    }
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
    /// The id of the last search sent.
    last_id: u64,
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
        Ok(Client { reader: BufReader::new(stream.try_clone()?), writer: stream, last_id: 0 })
    }

    /// Sends a ping or quit and reads its answer (answers left over from an abandoned search are skipped).
    pub fn request(&mut self, request: &Request) -> Result<Response> {
        writeln!(self.writer, "{}", serde_json::to_string(request)?)?;
        loop {
            let response = self.read()?;
            if response.id == 0 {
                return Ok(response);
            }
        }
    }

    fn read(&mut self) -> Result<Response> {
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

    /// A whole search: waits for the final response (grep included unless `grep` is false).
    pub fn search(
        &mut self,
        query: &str,
        kinds: Option<Vec<Kind>>,
        limit: usize,
        zgate: f64,
        grep: bool,
    ) -> Result<Response> {
        Ok(self.search_stream(query, kinds, limit, zgate, grep, &mut |_| true)?.expect("never abandoned"))
    }

    /// A search whose first response goes to `first` before the final one arrives. If `first` returns false the
    /// search is abandoned (None): its final response is skipped by the next search, being for an older id.
    pub fn search_stream(
        &mut self,
        query: &str,
        kinds: Option<Vec<Kind>>,
        limit: usize,
        zgate: f64,
        grep: bool,
        first: &mut dyn FnMut(&Response) -> bool,
    ) -> Result<Option<Response>> {
        self.last_id += 1;
        let id = self.last_id;
        let request = Request::Search { id, query: query.to_string(), limit, grep, kinds, zgate: Some(zgate) };
        writeln!(self.writer, "{}", serde_json::to_string(&request)?)?;
        loop {
            let response = self.read()?;
            if response.id != id {
                continue;
            }
            if response.done {
                return Ok(Some(response));
            }
            if !first(&response) {
                return Ok(None);
            }
        }
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

    fn search_once(
        &mut self,
        query: &str,
        limit: usize,
        first: &mut dyn FnMut(Vec<Hit>) -> bool,
    ) -> Result<Option<Vec<Hit>>> {
        let client = match &mut self.client {
            Some(client) => client,
            None => self.client.insert(Client::connect_or_start(&self.dir)?),
        };
        let done = client.search_stream(query, None, limit, 2.0, true, &mut |r| first(r.hits.clone()))?;
        Ok(done.map(|r| r.hits))
    }
}

impl crate::tui::backend::Backend for ServiceBackend {
    fn search(&mut self, query: &str, limit: usize) -> Result<Vec<Hit>> {
        Ok(self.search_stream(query, limit, &mut |_| true)?.unwrap_or_default())
    }

    fn search_stream(
        &mut self,
        query: &str,
        limit: usize,
        first: &mut dyn FnMut(Vec<Hit>) -> bool,
    ) -> Result<Option<Vec<Hit>>> {
        match self.search_once(query, limit, first) {
            Ok(result) => Ok(result),
            Err(_) => {
                // The service may have quit (idle) while the interface stayed open: start it again and retry once.
                self.client = None;
                self.search_once(query, limit, first)
            }
        }
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
        let request = Request::Search {
            id: 7,
            query: "dog".into(),
            limit: 5,
            grep: true,
            kinds: Some(vec![Kind::Doc]),
            zgate: None,
        };
        let line = serde_json::to_string(&request).unwrap();
        assert_eq!(line, r#"{"op":"search","id":7,"query":"dog","limit":5,"kinds":["doc"]}"#);
        let back: Request = serde_json::from_str(r#"{"op":"search","query":"dog"}"#).unwrap();
        assert!(matches!(back, Request::Search { id: 0, limit: 10, grep: true, kinds: None, zgate: None, .. }));
        let response: Response = serde_json::from_str(r#"{"ok":true}"#).unwrap();
        assert!(response.done, "an answer without \"final\" is final");
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
