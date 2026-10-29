//! The in-flight counter goes down on every exit path of a request, so a shutdown drain can
//! trust it: normal completion, an early return, a panic, and a handler future cancelled when the
//! client disconnects.

use std::time::Duration;

use helios_core::inflight::InflightTracker;

#[test]
fn guard_drop_decrements() {
    let t = InflightTracker::new();
    let a = t.enter();
    let b = t.enter();
    assert_eq!(t.current(), 2);
    drop(a);
    assert_eq!(t.current(), 1);
    drop(b);
    assert_eq!(t.current(), 0);
}

#[test]
fn guard_drop_decrements_on_panic_unwind() {
    let t = InflightTracker::new();
    let t2 = t.clone();
    let r = std::panic::catch_unwind(move || {
        let _g = t2.enter();
        panic!("handler bug");
    });
    assert!(r.is_err());
    assert_eq!(t.current(), 0);
}

#[tokio::test]
async fn guard_drop_decrements_when_the_future_is_cancelled() {
    let t = InflightTracker::new();
    let t2 = t.clone();
    let task = tokio::spawn(async move {
        let _g = t2.enter();
        tokio::time::sleep(Duration::from_secs(3600)).await;
    });
    while t.current() == 0 {
        tokio::task::yield_now().await;
    }
    task.abort();
    let _ = task.await;
    assert_eq!(t.current(), 0);
}

#[tokio::test]
async fn wait_idle_returns_at_once_when_nothing_is_in_flight() {
    assert!(InflightTracker::new().wait_idle(Duration::from_millis(50)).await);
}
