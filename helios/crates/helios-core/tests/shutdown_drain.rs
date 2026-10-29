//! A shutdown waits for requests in flight, but only up to its deadline.

use std::time::{Duration, Instant};

use helios_core::inflight::InflightTracker;

#[tokio::test]
async fn drain_waits_for_the_last_request_then_returns() {
    let t = InflightTracker::new();
    let g1 = t.enter();
    let g2 = t.enter();
    let waiter = {
        let t = t.clone();
        tokio::spawn(async move { t.wait_idle(Duration::from_secs(10)).await })
    };
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(!waiter.is_finished(), "must still wait with two requests in flight");
    drop(g1);
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(!waiter.is_finished(), "must still wait with one request in flight");
    drop(g2);
    assert!(waiter.await.unwrap(), "must report idle once the last guard is gone");
}

#[tokio::test]
async fn drain_gives_up_at_the_deadline_and_says_so() {
    let t = InflightTracker::new();
    let _stuck = t.enter();
    let start = Instant::now();
    let idle = t.wait_idle(Duration::from_millis(100)).await;
    assert!(!idle, "a stuck request must make the drain report a timeout");
    assert!(start.elapsed() < Duration::from_secs(5), "the deadline bounds the wait");
    assert_eq!(t.current(), 1);
}

#[tokio::test]
async fn a_request_that_starts_during_the_drain_is_waited_for_too() {
    let t = InflightTracker::new();
    let first = t.enter();
    let waiter = {
        let t = t.clone();
        tokio::spawn(async move { t.wait_idle(Duration::from_secs(10)).await })
    };
    tokio::time::sleep(Duration::from_millis(20)).await;
    let second = t.enter();
    drop(first);
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(!waiter.is_finished());
    drop(second);
    assert!(waiter.await.unwrap());
}
