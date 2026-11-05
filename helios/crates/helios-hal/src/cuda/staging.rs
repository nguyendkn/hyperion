//! Pinned staging for the host-device transfers of the step path (owner decision D4, 2026-11-05).
//!
//! A `cuMemcpyHtoD` from pageable memory is not ordered with the (non-blocking) stream the kernels run on: the driver may return before the DMA to the
//! device has finished, and a kernel launched at once can read a half-written input (this was the cause of the random disagreement of the host-path ops
//! with the CPU reference, 2026-11-04). Every transfer of the step path therefore goes through a ring of pinned host slots allocated before Ready:
//! the bytes are copied into a slot, `cuMemcpyHtoDAsync` queues the copy on the step stream (so it is ordered with the launches that follow), and an
//! event recorded behind it tells when the slot may be written again. The host never reuses a slot before that event has completed (it waits for it).
//! Device-to-host copies use the same slots with `cuMemcpyDtoHAsync` and wait for their event before reading.
//!
//! [`Ring`] is the bookkeeping, free of CUDA so its rules are unit-tested on any host.

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StagingError {
    #[error("a transfer of {bytes} bytes does not fit a staging slot of {slot_bytes} bytes")]
    TooLarge { bytes: usize, slot_bytes: usize },
    #[error("the pinned staging ring was not allocated before the first step (CudaDevice::enable_staging)")]
    NotEnabled,
}

/// Which slot a transfer uses and whether the host must first wait for the event of that slot's previous use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Acquired {
    pub slot: usize,
    pub wait: bool,
}

/// The slot rotation: slot `i` is in use from the moment its transfer is queued until the host has seen its event complete.
#[derive(Debug)]
pub struct Ring {
    slots: usize,
    slot_bytes: usize,
    next: usize,
    pending: Vec<bool>,
}

impl Ring {
    pub fn new(slots: usize, slot_bytes: usize) -> Self {
        assert!(slots >= 2 && slot_bytes > 0, "a ring needs at least two slots of at least one byte");
        Self { slots, slot_bytes, next: 0, pending: vec![false; slots] }
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Total pinned bytes the ring needs.
    pub fn total_bytes(&self) -> usize {
        self.slots * self.slot_bytes
    }

    /// Take the next slot for a transfer of `bytes`. `wait` is true when that slot was queued before and its event has not been seen complete: the caller
    /// waits for the event and calls [`Ring::completed`] before writing the slot.
    pub fn acquire(&mut self, bytes: usize) -> Result<Acquired, StagingError> {
        if bytes > self.slot_bytes {
            return Err(StagingError::TooLarge { bytes, slot_bytes: self.slot_bytes });
        }
        let slot = self.next;
        self.next = (self.next + 1) % self.slots;
        Ok(Acquired { slot, wait: self.pending[slot] })
    }

    /// The transfer of `slot` and the event behind it are queued.
    pub fn recorded(&mut self, slot: usize) {
        self.pending[slot] = true;
    }

    /// The host has seen the event of `slot` complete: the slot may be written again.
    pub fn completed(&mut self, slot: usize) {
        self.pending[slot] = false;
    }

    /// Slots whose last transfer has not been seen complete.
    pub fn in_flight(&self) -> usize {
        self.pending.iter().filter(|p| **p).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_rotate_in_order_and_a_slot_is_free_until_it_is_queued() {
        let mut ring = Ring::new(3, 64);
        for expected in [0, 1, 2, 0, 1] {
            let a = ring.acquire(8).unwrap();
            assert_eq!(a.slot, expected);
            ring.recorded(a.slot);
            ring.completed(a.slot);
        }
        assert_eq!(ring.in_flight(), 0);
    }

    #[test]
    fn a_slot_queued_but_not_seen_complete_must_be_waited_for_before_reuse() {
        let mut ring = Ring::new(2, 64);
        let (a, b) = (ring.acquire(4).unwrap(), ring.acquire(4).unwrap());
        assert_eq!((a.wait, b.wait), (false, false), "fresh slots need no wait");
        ring.recorded(a.slot);
        ring.recorded(b.slot);
        assert_eq!(ring.in_flight(), 2);
        let again = ring.acquire(4).unwrap();
        assert_eq!(again.slot, a.slot);
        assert!(again.wait, "the first slot is reused while its event is not known to be complete");
        ring.completed(again.slot);
        ring.recorded(again.slot);
        let next = ring.acquire(4).unwrap();
        assert_eq!(next.slot, b.slot);
        assert!(next.wait);
        // A slot whose completion was seen in the meantime needs no wait.
        ring.completed(next.slot);
        ring.recorded(next.slot);
        ring.completed(again.slot);
        assert!(!ring.acquire(4).unwrap().wait);
    }

    #[test]
    fn a_transfer_larger_than_a_slot_is_refused_with_a_typed_error_not_split_or_truncated() {
        let mut ring = Ring::new(2, 16);
        assert_eq!(ring.acquire(17), Err(StagingError::TooLarge { bytes: 17, slot_bytes: 16 }));
        assert!(ring.acquire(16).is_ok(), "exactly the slot size fits");
        assert_eq!(ring.total_bytes(), 32);
    }

    #[test]
    fn a_refused_transfer_does_not_consume_a_slot() {
        let mut ring = Ring::new(2, 16);
        let _ = ring.acquire(100);
        assert_eq!(ring.acquire(1).unwrap().slot, 0);
    }
}
