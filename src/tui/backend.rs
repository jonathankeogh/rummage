//! What the interface needs from a search engine.

use crate::search::Hit;

/// A source of search results. The interface calls it from one background thread, never from the UI thread.
pub trait Backend: Send + 'static {
    /// Best hits for `query`, at most `limit`, best first. A query starting with "code:" is passed through as typed.
    fn search(&mut self, query: &str, limit: usize) -> anyhow::Result<Vec<Hit>>;

    /// A search in two steps: early hits go to `first` (meaning and words), then the final ones come back (grep's
    /// matches merged in). If `first` returns false the caller has moved on and the search is dropped (None).
    fn search_stream(
        &mut self,
        query: &str,
        limit: usize,
        _first: &mut dyn FnMut(Vec<Hit>) -> bool,
    ) -> anyhow::Result<Option<Vec<Hit>>> {
        self.search(query, limit).map(Some)
    }
}
