use std::thread::sleep;
use std::time::Duration;

/// Call `f` until it succeeds, waiting twice as long after each failure.
pub fn retry_with_backoff<T, E>(mut f: impl FnMut() -> Result<T, E>, attempts: u32) -> Result<T, E> {
    let mut delay = Duration::from_millis(100);
    for attempt in 1..=attempts {
        match f() {
            Ok(value) => return Ok(value),
            Err(e) if attempt == attempts => return Err(e),
            Err(_) => {
                sleep(delay);
                delay *= 2;
            }
        }
    }
    unreachable!("attempts must be at least 1")
}
