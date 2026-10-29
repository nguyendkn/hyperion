//! The rule "after the model is loaded, serving allocates nothing, loads no kernel module and captures
//! no graph": everything a decode step needs is built at load time, so a serving request never pays
//! for (or fails on) device allocation, module loading or capture. The device carries a flag; the
//! places that do such work ask it first, and a violation is a typed error, not a silent slow path.
//!
//! The flag is set for the duration of a generation by [`WorkRestriction::forbid`]. It is plain
//! host logic (no CUDA), so it is unit-tested everywhere.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Work that must happen at load time was attempted while serving.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{what} is not allowed while serving: it must be done when the model is loaded")]
pub struct NewGpuWorkRefused {
    pub what: &'static str,
}

/// A shared on/off switch; cloning shares it.
#[derive(Debug, Clone, Default)]
pub struct WorkRestriction {
    forbidden: Arc<AtomicBool>,
}

impl WorkRestriction {
    /// Forbid new GPU work until the returned guard is dropped (then the previous state returns).
    pub fn forbid(&self) -> RestrictionGuard {
        RestrictionGuard { flag: self.forbidden.clone(), previous: self.forbidden.swap(true, Ordering::SeqCst) }
    }

    /// Whether new GPU work is currently forbidden.
    pub fn is_active(&self) -> bool {
        self.forbidden.load(Ordering::SeqCst)
    }

    /// `Ok` unless new GPU work is forbidden right now; `what` names the attempted work.
    pub fn check(&self, what: &'static str) -> Result<(), NewGpuWorkRefused> {
        if self.is_active() {
            Err(NewGpuWorkRefused { what })
        } else {
            Ok(())
        }
    }
}

/// Restores the previous state of the switch when dropped.
#[must_use = "the restriction ends when the guard is dropped"]
pub struct RestrictionGuard {
    flag: Arc<AtomicBool>,
    previous: bool,
}

impl Drop for RestrictionGuard {
    fn drop(&mut self) {
        self.flag.store(self.previous, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn work_is_allowed_until_forbidden_and_allowed_again_after() {
        let restriction = WorkRestriction::default();
        assert!(restriction.check("device memory allocation").is_ok());
        {
            let _guard = restriction.forbid();
            let error = restriction.check("graph capture").unwrap_err();
            assert_eq!(error, NewGpuWorkRefused { what: "graph capture" });
            assert!(error.to_string().contains("graph capture") && error.to_string().contains("when the model is loaded"));
        }
        assert!(restriction.check("graph capture").is_ok());
    }

    #[test]
    fn a_nested_guard_restores_the_outer_state_not_just_off() {
        let restriction = WorkRestriction::default();
        let outer = restriction.forbid();
        drop(restriction.forbid());
        assert!(restriction.is_active(), "the outer guard is still alive");
        drop(outer);
        assert!(!restriction.is_active());
    }

    #[test]
    fn clones_share_the_switch() {
        let restriction = WorkRestriction::default();
        let other = restriction.clone();
        let _guard = restriction.forbid();
        assert!(other.is_active());
    }
}
