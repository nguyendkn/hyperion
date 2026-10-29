//! Counts requests being served so a shutdown can wait for them. A request takes an
//! [`InflightGuard`] when it starts; the count drops when the guard does, on every exit path
//! (return, error, panic unwind, a dropped client connection that cancels the handler future).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Notify;

#[derive(Default)]
struct Inner {
    count: AtomicUsize,
    idle: Notify,
}

/// Shared handle; clones count the same requests.
#[derive(Clone, Default)]
pub struct InflightTracker(Arc<Inner>);

/// Held for the life of one request.
pub struct InflightGuard(Arc<Inner>);

impl InflightTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enter(&self) -> InflightGuard {
        self.0.count.fetch_add(1, Ordering::AcqRel);
        InflightGuard(self.0.clone())
    }

    pub fn current(&self) -> usize {
        self.0.count.load(Ordering::Acquire)
    }

    /// Wait until no request is in flight; `true` if that happened within `timeout`.
    pub async fn wait_idle(&self, timeout: Duration) -> bool {
        let wait = async {
            loop {
                // Register for the wake-up before checking the count, so a guard dropping
                // between the check and the await cannot be missed.
                let notified = self.0.idle.notified();
                if self.current() == 0 {
                    return;
                }
                notified.await;
            }
        };
        tokio::time::timeout(timeout, wait).await.is_ok()
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        if self.0.count.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.idle.notify_waiters();
        }
    }
}
