//! Runtime hardware triage — decide whether a package's `target_arch` can run
//! on the *actual* device the factory just probed.
//!
//! The rule is one-way forward compatibility:
//! - `target <= actual` → run as-is, or downgrade kernels.
//! - `target >  actual` → reject. A sm_120 package on sm_90 hardware has no
//!   safe fallback; the kernels literally use instructions the GPU lacks.

use thiserror::Error;

/// Numeric major of a compute capability string, e.g. "sm_120" → 120.
pub fn cap_major(cap: &str) -> Option<u32> {
    cap.strip_prefix("sm_")
        .or_else(|| cap.strip_prefix("sm"))
        .and_then(|n| n.parse::<u32>().ok())
}

/// Decision produced by triage before any weight is loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriageOutcome {
    /// Exact match — run the package's native kernels.
    Exact,
    /// Package is older than the device — run, optionally downgrading kernels
    /// (e.g. run sm_90 kernels on sm_120 hardware).
    Downgrade,
}

#[derive(Debug, Error)]
pub enum TriageError {
    #[error("package targets {target} but device is {actual}: forward-incompatible")]
    UpgradeRequired {
        target: String,
        actual: String,
    },
    #[error("unparseable compute capability: {0}")]
    BadCapability(String),
}

/// A finished triage decision. `helios-core` reads `outcome` to decide whether
/// to allow a kernel-level recipe swap before calling `load_model_weights`.
#[derive(Debug, Clone)]
pub struct Triage {
    pub outcome: TriageOutcome,
    pub target: String,
    pub actual: String,
}

impl Triage {
    /// Compare a package's `target_arch` against the device's probed
    /// `compute_capability`. Both strings use the `sm_<major>` convention.
    pub fn evaluate(target: &str, actual: &str) -> Result<Self, TriageError> {
        let t = cap_major(target).ok_or_else(|| TriageError::BadCapability(target.into()))?;
        let a = cap_major(actual).ok_or_else(|| TriageError::BadCapability(actual.into()))?;
        let outcome = if t == a {
            TriageOutcome::Exact
        } else if t < a {
            TriageOutcome::Downgrade
        } else {
            return Err(TriageError::UpgradeRequired {
                target: target.into(),
                actual: actual.into(),
            });
        };
        Ok(Self {
            outcome,
            target: target.to_string(),
            actual: actual.to_string(),
        })
    }
}
