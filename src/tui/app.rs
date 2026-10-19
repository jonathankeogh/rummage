//! Interface state: the query, results, selection and the search worker that fills them in.

use super::backend::Backend;
use super::preview::{Preview, query_terms};
use crate::search::Hit;
use crate::store::Kind;
use ratatui::crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::layout::{Position, Rect};
use ratatui::widgets::ListState;
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

/// How many hits to ask the backend for.
pub const LIMIT: usize = 50;
/// How long the query must sit still before a search starts.
pub const DEBOUNCE: Duration = Duration::from_millis(80);
/// Two clicks on one row this close together open it.
const DOUBLE_CLICK: Duration = Duration::from_millis(400);
/// Rows a wheel notch or Shift-arrow scrolls the preview.
const SCROLL_STEP: isize = 3;

/// A search to run.
pub struct Request {
    pub id: u64,
    pub query: String,
}

/// Search results, tagged with the request they answer: early ones (meaning and words) and then the final ones
/// (grep's matches merged in).
pub struct Response {
    pub id: u64,
    pub result: Result<Vec<Hit>, String>,
    pub done: bool,
}

/// Start the search thread. It waits for the query to settle, then searches; of several queued requests only the
/// newest runs, and a newer query drops the search in flight (its grep is cancelled).
pub fn spawn_worker<B: Backend>(mut backend: B) -> (Sender<Request>, Receiver<Response>) {
    let (request_tx, request_rx) = mpsc::channel::<Request>();
    let (response_tx, response_rx) = mpsc::channel::<Response>();
    std::thread::spawn(move || {
        let mut queued: Option<Request> = None;
        loop {
            let mut request = match queued.take() {
                Some(request) => request,
                None => match request_rx.recv() {
                    Ok(request) => request,
                    Err(_) => return,
                },
            };
            loop {
                match request_rx.recv_timeout(DEBOUNCE) {
                    Ok(newer) => request = newer,
                    Err(mpsc::RecvTimeoutError::Timeout) => break,
                    Err(mpsc::RecvTimeoutError::Disconnected) => return,
                }
            }
            let id = request.id;
            let mut first = |hits: Vec<Hit>| {
                let _ = response_tx.send(Response { id, result: Ok(hits), done: false });
                // A newer query waiting: drop this search.
                match request_rx.try_recv() {
                    Ok(newer) => {
                        queued = Some(newer);
                        false
                    }
                    Err(_) => true,
                }
            };
            let result = backend.search_stream(&request.query, LIMIT, &mut first).map_err(|e| format!("{e:#}"));
            let response = match result {
                Ok(Some(hits)) => Response { id, result: Ok(hits), done: true },
                Ok(None) => continue,
                Err(e) => Response { id, result: Err(e), done: true },
            };
            if response_tx.send(response).is_err() {
                return;
            }
        }
    });
    (request_tx, response_rx)
}

/// What the main loop should do after a key.
#[derive(Debug, PartialEq)]
pub enum Action {
    None,
    Quit,
    Open(usize),
    Reveal(usize),
    Copy(usize),
}

/// Which hits to show.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    All,
    Only(Kind),
}

impl Filter {
    fn next(self) -> Filter {
        match self {
            Filter::All => Filter::Only(Kind::Doc),
            Filter::Only(Kind::Doc) => Filter::Only(Kind::Pdf),
            Filter::Only(Kind::Pdf) => Filter::Only(Kind::Image),
            Filter::Only(Kind::Image) => Filter::Only(Kind::Screenshot),
            Filter::Only(Kind::Screenshot) => Filter::Only(Kind::Code),
            Filter::Only(Kind::Code) => Filter::All,
        }
    }

    /// Name for the title bar.
    pub fn label(self) -> &'static str {
        match self {
            Filter::All => "all",
            Filter::Only(kind) => kind.as_str(),
        }
    }
}

pub struct App {
    pub query: String,
    pub hits: Vec<Hit>,
    pub filter: Filter,
    /// Position within `visible()`.
    pub selected: usize,
    pub help: bool,
    /// The last search failed with this.
    pub error: Option<String>,
    /// A message for the footer, with when it appeared.
    pub notice: Option<(String, Instant)>,
    pub tick: usize,
    requests: Sender<Request>,
    /// The newest request sent; answers to older ones are stale.
    latest: u64,
    /// The newest request answered finally (grep included).
    answered: u64,
    /// The selection moved since the current results came in: grep's later hits then go below, not re-sorted.
    moved: bool,
    /// The selected hit's preview and what it was made for.
    preview: Option<(PreviewKey, Preview)>,
    /// The preview's first shown row (after wrapping); `None` until drawn, which opens it at the passage.
    pub scroll: Cell<Option<usize>>,
    /// Wrapped-row counts of the current preview, for the width they were measured at (see `ui`).
    pub metrics: Cell<Option<Metrics>>,
    /// Where the last draw put the results list and the preview, for mouse clicks.
    pub areas: Cell<Areas>,
    /// The results list's scroll position, kept between draws.
    pub list: RefCell<ListState>,
    last_click: Option<(Instant, usize)>,
    /// Show PDF pages as pictures rather than text (Ctrl-v).
    pub page_view: bool,
    /// How this terminal draws pictures (kitty, iTerm2, sixel, or half-blocks anywhere).
    pub picker: Option<Picker>,
    /// The selected PDF page, drawn for the preview's size: what it was made for, and the picture or why not.
    pub page_image: RefCell<Option<(PageKey, Result<StatefulProtocol, String>)>>,
}

/// How this terminal draws pictures: asked the first time a page is shown (not at start, so a terminal that never
/// answers costs nothing until then). `RUMMAGE_PICTURES=halfblocks` skips the question: half-blocks work in any
/// terminal with colour.
fn ask_terminal() -> Picker {
    if std::env::var("RUMMAGE_PICTURES").is_ok_and(|v| v == "halfblocks") {
        return Picker::halfblocks();
    }
    let options = ratatui_image::picker::cap_parser::QueryStdioOptions {
        timeout: Duration::from_millis(500),
        ..Default::default()
    };
    Picker::from_query_stdio_with_options(options).unwrap_or_else(|_| Picker::halfblocks())
}

#[derive(Clone, Debug, PartialEq)]
pub struct PageKey {
    path: String,
    page: usize,
    width: u16,
    height: u16,
}

/// The preview's size in rows at one width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Metrics {
    pub width: u16,
    /// Rows before the line the pane opens at (the first marked line, else the passage's start).
    pub landing: usize,
    pub total: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Areas {
    pub results: Rect,
    pub preview: Rect,
}

#[derive(PartialEq)]
struct PreviewKey {
    path: String,
    loc: Option<i64>,
    query: String,
}

impl App {
    pub fn new(requests: Sender<Request>) -> App {
        App {
            query: String::new(),
            hits: Vec::new(),
            filter: Filter::All,
            selected: 0,
            help: false,
            error: None,
            notice: None,
            tick: 0,
            requests,
            latest: 0,
            answered: 0,
            moved: false,
            preview: None,
            scroll: Cell::new(None),
            metrics: Cell::new(None),
            areas: Cell::new(Areas::default()),
            list: RefCell::new(ListState::default()),
            last_click: None,
            page_view: false,
            picker: None,
            page_image: RefCell::new(None),
        }
    }

    /// A search is running or queued.
    pub fn searching(&self) -> bool {
        self.answered < self.latest
    }

    /// Indices into `hits` that pass the kind filter.
    pub fn visible(&self) -> Vec<usize> {
        (0..self.hits.len())
            .filter(|&i| matches!(self.filter, Filter::All) || self.filter == Filter::Only(self.hits[i].kind))
            .collect()
    }

    /// Index into `hits` of the selected row.
    pub fn selected_hit(&self) -> Option<usize> {
        self.visible().get(self.selected).copied()
    }

    /// Ask for results for the current query, making any answer to an older query stale.
    fn query_changed(&mut self) {
        self.latest += 1;
        if self.query.trim().is_empty() {
            self.answered = self.latest;
            self.hits.clear();
            self.error = None;
            self.selected = 0;
        } else {
            let _ = self.requests.send(Request { id: self.latest, query: self.query.clone() });
        }
    }

    /// Take a search answer. Returns false (and changes nothing) if it is for an old query. The final answer
    /// re-sorts the list, unless the selection has moved since the early one: then its new hits go below.
    pub fn apply_response(&mut self, response: Response) -> bool {
        if response.id != self.latest {
            return false;
        }
        if response.done {
            self.answered = response.id;
        }
        match response.result {
            Ok(hits) if response.done && self.moved => {
                let shown: HashSet<String> = self.hits.iter().map(|h| h.path.clone()).collect();
                self.hits.extend(hits.into_iter().filter(|h| !shown.contains(&h.path)));
                self.error = None;
                return true;
            }
            Ok(hits) => {
                self.hits = hits;
                self.error = None;
            }
            Err(e) => {
                self.hits.clear();
                self.error = Some(e);
            }
        }
        self.selected = 0;
        self.moved = false;
        *self.list.borrow_mut() = ListState::default();
        true
    }

    fn move_selection(&mut self, delta: isize) {
        let len = self.visible().len();
        self.moved = true;
        if len > 0 {
            self.selected = (self.selected as isize + delta).clamp(0, len as isize - 1) as usize;
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) -> Action {
        if key.kind == KeyEventKind::Release {
            return Action::None;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if self.help {
            // Any key closes help; Ctrl-c still quits.
            self.help = false;
            return if ctrl && key.code == KeyCode::Char('c') { Action::Quit } else { Action::None };
        }
        let selected = self.selected_hit();
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        match key.code {
            KeyCode::Esc => return Action::Quit,
            KeyCode::Up if shift => self.scroll_preview(-SCROLL_STEP),
            KeyCode::Down if shift => self.scroll_preview(SCROLL_STEP),
            KeyCode::PageUp => self.scroll_preview(-self.page()),
            KeyCode::PageDown => self.scroll_preview(self.page()),
            KeyCode::Char('c') if ctrl => return Action::Quit,
            KeyCode::F(1) => self.help = true,
            KeyCode::Char('?') if self.query.is_empty() => self.help = true,
            KeyCode::Up => self.move_selection(-1),
            KeyCode::Down => self.move_selection(1),
            KeyCode::Char('p') if ctrl => self.move_selection(-1),
            KeyCode::Char('n') if ctrl => self.move_selection(1),
            KeyCode::Tab => {
                self.filter = self.filter.next();
                self.selected = 0;
            }
            KeyCode::Enter => return selected.map_or(Action::None, Action::Open),
            KeyCode::Char('o') if ctrl => return selected.map_or(Action::None, Action::Reveal),
            KeyCode::Char('v') if ctrl => {
                self.page_view = !self.page_view;
                let kind = selected.map(|i| self.hits[i].kind);
                if kind.is_some_and(|k| k != Kind::Pdf && !k.is_picture()) {
                    self.notify("^v switches PDFs and pictures between picture and text");
                }
            }
            KeyCode::Char('y') if ctrl => return selected.map_or(Action::None, Action::Copy),
            KeyCode::Char('u') if ctrl => {
                self.query.clear();
                self.query_changed();
            }
            KeyCode::Char('w') if ctrl => {
                let trimmed = self.query.trim_end().len();
                self.query.truncate(self.query[..trimmed].rfind(' ').map_or(0, |i| i + 1));
                self.query_changed();
            }
            KeyCode::Backspace => {
                if self.query.pop().is_some() {
                    self.query_changed();
                }
            }
            KeyCode::Char(c) if !ctrl && !key.modifiers.contains(KeyModifiers::ALT) => {
                self.query.push(c);
                self.query_changed();
            }
            _ => {}
        }
        Action::None
    }

    /// Wheel over the results moves the selection; over the preview it scrolls. A click selects a row; a second click
    /// on it opens it. Any click closes help.
    pub fn on_mouse(&mut self, mouse: MouseEvent) -> Action {
        let at = Position::new(mouse.column, mouse.row);
        let areas = self.areas.get();
        if self.help {
            if matches!(mouse.kind, MouseEventKind::Down(_)) {
                self.help = false;
            }
            return Action::None;
        }
        match mouse.kind {
            MouseEventKind::ScrollDown if areas.preview.contains(at) => self.scroll_preview(SCROLL_STEP),
            MouseEventKind::ScrollUp if areas.preview.contains(at) => self.scroll_preview(-SCROLL_STEP),
            MouseEventKind::ScrollDown if areas.results.contains(at) => self.move_selection(1),
            MouseEventKind::ScrollUp if areas.results.contains(at) => self.move_selection(-1),
            MouseEventKind::Down(MouseButton::Left) if areas.results.contains(at) => {
                let row = self.list.borrow().offset() + (mouse.row - areas.results.y) as usize;
                let visible = self.visible();
                let Some(&hit) = visible.get(row) else { return Action::None };
                let now = Instant::now();
                let double = self.last_click.is_some_and(|(t, r)| r == row && now.duration_since(t) < DOUBLE_CLICK);
                self.selected = row;
                self.moved = true;
                if double {
                    self.last_click = None;
                    return Action::Open(hit);
                }
                self.last_click = Some((now, row));
            }
            _ => {}
        }
        Action::None
    }

    /// Scroll the preview by `rows` (clamped when drawn).
    pub fn scroll_preview(&mut self, rows: isize) {
        let from = self.scroll.get().unwrap_or(0) as isize;
        self.scroll.set(Some((from + rows).max(0) as usize));
    }

    /// A page of preview, less two rows of overlap.
    fn page(&self) -> isize {
        (self.areas.get().preview.height as isize - 2).max(1)
    }

    /// Show a message in the footer for a moment.
    pub fn notify(&mut self, message: impl Into<String>) {
        self.notice = Some((message.into(), Instant::now()));
    }

    /// Make the preview match the selection and query, reading the file only when they changed.
    pub fn refresh_preview(&mut self) {
        let Some(i) = self.selected_hit() else {
            self.preview = None;
            return;
        };
        let hit = &self.hits[i];
        let key = PreviewKey { path: hit.path.clone(), loc: hit.loc.map(|l| l as i64), query: self.query.clone() };
        if self.preview.as_ref().is_some_and(|(k, _)| *k == key) {
            return;
        }
        let preview = Preview::load(hit, &query_terms(&self.query));
        self.preview = Some((key, preview));
        self.scroll.set(None);
        self.metrics.set(None);
    }

    /// Draws the selected PDF page for the preview's current size, when the page view is on and it changed.
    /// Whether the selected hit is shown as a picture: pictures unless switched to their words, PDF pages when switched
    /// to the page.
    pub fn shows_picture(&self, kind: Kind) -> bool {
        (kind.is_picture() && !self.page_view) || (kind == Kind::Pdf && self.page_view)
    }

    pub fn refresh_page_image(&mut self) {
        let Some(i) = self.selected_hit().filter(|&i| self.shows_picture(self.hits[i].kind)) else {
            return;
        };
        let picker = self.picker.get_or_insert_with(ask_terminal);
        let area = self.areas.get().preview;
        if area.width == 0 || area.height == 0 {
            return;
        }
        let hit = &self.hits[i];
        let key = PageKey {
            path: hit.path.clone(),
            page: hit.loc.map_or(1, |l| l.round().max(1.0) as usize),
            width: area.width,
            height: area.height,
        };
        if self.page_image.borrow().as_ref().is_some_and(|(k, _)| *k == key) {
            return;
        }
        let height_px = area.height as u32 * picker.font_size().height as u32;
        let path = std::path::Path::new(&key.path);
        let image = if hit.kind.is_picture() {
            crate::picture::decode(path).and_then(|rgb| {
                let buffer = image::RgbImage::from_raw(rgb.width as u32, rgb.height as u32, rgb.data)
                    .ok_or_else(|| anyhow::anyhow!("odd picture size"))?;
                Ok(image::DynamicImage::ImageRgb8(buffer))
            })
        } else {
            crate::extract::pdf_page_png(path, key.page, height_px).and_then(|png| Ok(image::load_from_memory(&png)?))
        };
        let image = image.map(|image| picker.new_resize_protocol(image)).map_err(|e| format!("can't draw it: {e:#}"));
        *self.page_image.borrow_mut() = Some((key, image));
    }

    pub fn preview(&self) -> Option<&Preview> {
        self.preview.as_ref().map(|(_, p)| p)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use ratatui::crossterm::event::KeyEvent;

    pub fn hit(path: &str, kind: Kind) -> Hit {
        Hit {
            path: path.into(),
            kind,
            score: 1.0,
            z: Some(3.1),
            cosine: None,
            keyword_rank: Some(2),
            loc: Some(1.0),
            line: None,
            grep: false,
            excerpt: String::new(),
        }
    }

    fn app() -> (App, Receiver<Request>) {
        let (tx, rx) = mpsc::channel();
        (App::new(tx), rx)
    }

    fn type_str(app: &mut App, s: &str) {
        for c in s.chars() {
            app.on_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        }
    }

    #[test]
    fn stale_results_are_dropped() {
        let (mut app, rx) = app();
        type_str(&mut app, "do");
        let ids: Vec<u64> = rx.try_iter().map(|r| r.id).collect();
        assert_eq!(ids, vec![1, 2]);
        assert!(app.searching());
        // the answer to "d" arrives after "do" was typed: ignored
        assert!(!app.apply_response(Response { id: 1, result: Ok(vec![hit("/a", Kind::Doc)]), done: true }));
        assert!(app.hits.is_empty() && app.searching());
        assert!(app.apply_response(Response { id: 2, result: Ok(vec![hit("/b", Kind::Doc)]), done: true }));
        assert_eq!(app.hits[0].path, "/b");
        assert!(!app.searching());
    }

    #[test]
    fn clearing_the_query_invalidates_pending_search() {
        let (mut app, _rx) = app();
        type_str(&mut app, "a");
        app.on_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert!(!app.searching());
        assert!(!app.apply_response(Response { id: 1, result: Ok(vec![hit("/a", Kind::Doc)]), done: true }));
        assert!(app.hits.is_empty());
    }

    #[test]
    fn filter_cycles_and_limits_rows() {
        let (mut app, _rx) = app();
        app.hits = vec![
            hit("/a.md", Kind::Doc),
            hit("/b.rs", Kind::Code),
            hit("/c.md", Kind::Doc),
            hit("/d.pdf", Kind::Pdf),
            hit("/e.jpg", Kind::Image),
            hit("/Screenshot f.png", Kind::Screenshot),
        ];
        assert_eq!(app.visible(), vec![0, 1, 2, 3, 4, 5]);
        // all → docs → PDFs → pictures → screenshots → code → all
        for expected in [vec![0, 2], vec![3], vec![4], vec![5], vec![1]] {
            app.on_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
            assert_eq!(app.visible(), expected);
        }
        app.on_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.filter, Filter::All);
    }

    #[test]
    fn keys_move_and_act() {
        let (mut app, _rx) = app();
        app.hits = vec![hit("/a.md", Kind::Doc), hit("/b.rs", Kind::Code)];
        app.on_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        app.on_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(app.selected, 1);
        app.on_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL));
        assert_eq!(app.selected, 0);
        assert_eq!(app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)), Action::Open(0));
        assert_eq!(app.on_key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::CONTROL)), Action::Copy(0));
        assert_eq!(app.on_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)), Action::Quit);
        // '?' opens help only on an empty query, otherwise it is typed
        app.on_key(KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE));
        assert!(app.help);
        app.on_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
        assert!(!app.help && app.query.is_empty());
        type_str(&mut app, "why?");
        assert_eq!(app.query, "why?");
    }

    fn mouse(kind: MouseEventKind, column: u16, row: u16) -> MouseEvent {
        MouseEvent { kind, column, row, modifiers: KeyModifiers::NONE }
    }

    #[test]
    fn mouse_selects_opens_and_scrolls() {
        let (mut app, _rx) = app();
        app.hits = vec![hit("/a", Kind::Doc), hit("/b", Kind::Doc), hit("/c", Kind::Doc)];
        app.areas.set(Areas { results: Rect::new(1, 1, 40, 10), preview: Rect::new(42, 1, 40, 10) });
        let click = || mouse(MouseEventKind::Down(MouseButton::Left), 5, 3);
        // One click selects row 2; a second click on it opens it.
        assert_eq!(app.on_mouse(click()), Action::None);
        assert_eq!(app.selected, 2);
        assert_eq!(app.on_mouse(click()), Action::Open(2));
        // A click below the last row does nothing.
        assert_eq!(app.on_mouse(mouse(MouseEventKind::Down(MouseButton::Left), 5, 8)), Action::None);
        assert_eq!(app.selected, 2);
        // The wheel moves the selection over the results and scrolls over the preview.
        app.on_mouse(mouse(MouseEventKind::ScrollUp, 5, 5));
        assert_eq!(app.selected, 1);
        app.scroll.set(Some(10));
        app.on_mouse(mouse(MouseEventKind::ScrollDown, 50, 5));
        assert_eq!(app.scroll.get(), Some(13));
        app.on_mouse(mouse(MouseEventKind::ScrollUp, 50, 5));
        app.on_mouse(mouse(MouseEventKind::ScrollUp, 50, 5));
        assert_eq!(app.scroll.get(), Some(7));
    }

    #[test]
    fn shift_arrows_and_pages_scroll_the_preview() {
        let (mut app, _rx) = app();
        app.hits = vec![hit("/a", Kind::Doc), hit("/b", Kind::Doc)];
        app.areas.set(Areas { preview: Rect::new(0, 0, 40, 12), ..Areas::default() });
        app.scroll.set(Some(0));
        app.on_key(KeyEvent::new(KeyCode::Down, KeyModifiers::SHIFT));
        assert_eq!((app.scroll.get(), app.selected), (Some(3), 0), "Shift-↓ scrolls, the selection stays");
        app.on_key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE));
        assert_eq!(app.scroll.get(), Some(13));
        app.on_key(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE));
        app.on_key(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE));
        assert_eq!(app.scroll.get(), Some(0), "never above the top");
    }

    #[test]
    fn worker_runs_only_the_newest_request() {
        struct Echo;
        impl Backend for Echo {
            fn search(&mut self, query: &str, _limit: usize) -> anyhow::Result<Vec<Hit>> {
                Ok(vec![hit(query, Kind::Doc)])
            }
        }
        let (tx, rx) = spawn_worker(Echo);
        for (id, q) in [(1, "d"), (2, "do"), (3, "dog")] {
            tx.send(Request { id, query: q.into() }).unwrap();
        }
        let response = rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(response.id, 3);
        assert_eq!(response.result.unwrap()[0].path, "dog");
        assert!(rx.recv_timeout(Duration::from_millis(300)).is_err());
    }
}
