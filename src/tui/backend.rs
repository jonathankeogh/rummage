//! What the interface needs from a search engine.

use crate::search::Hit;

/// A source of search results. The interface calls it from one background thread, never from the UI thread.
pub trait Backend: Send + 'static {
    /// Best hits for `query`, at most `limit`, best first. A query starting with "code:" is passed through as typed.
    fn search(&mut self, query: &str, limit: usize) -> anyhow::Result<Vec<Hit>>;
}
