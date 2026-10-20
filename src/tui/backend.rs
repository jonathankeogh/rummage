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

    /// A way to ask, from another thread, how indexing is going while the index is being brought up to date.
    fn indexing_probe(&self) -> Option<Box<dyn FnMut() -> Option<Indexing> + Send>> {
        None
    }
}

/// How far the service's indexing has got.
#[derive(Clone, Debug, PartialEq)]
pub struct Indexing {
    /// Files still to read, in part or whole.
    pub left: usize,
    /// The long file later passes are reading, with how far ("Film.mp4 30 of 118 minutes").
    pub reading: Option<String>,
}
