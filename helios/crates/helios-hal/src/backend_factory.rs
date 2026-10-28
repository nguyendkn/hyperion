//! Backend factory — probes the real device at runtime and returns the
//! concrete [`ComputeBackend`] for it. This is the only place in the HAL that
//! knows about backend *kinds*; everything else goes through the trait.
//!
//! ## Build-time vs runtime
//! Feature flags decide which backends are *compiled in*; this factory decides
//! which compiled-in backend is *used*. A `multi` build carries several and
//! picks at startup; a per-host binary carries exactly one and the probe just
//! confirms it matches the device.

use anyhow::{anyhow, Result};
use tracing::info;

use crate::triage::Triage;
use crate::traits::{ComputeBackend, DynBackend};

/// The set of backends compiled into this build. Exactly one is `true` when
/// built per-host; several may be `true` under the `multi` feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Compile-only stub — returns the trait without a real GPU. Active under
    /// the `default` (stub) feature so the workspace typechecks without CUDA.
    Stub,
    /// Hopper SM90 — TensorRT-LLM via FFI.
    #[cfg(feature = "sm90")]
    HopperSm90,
    /// Blackwell SM120 — native CUDA Graphs.
    #[cfg(feature = "sm120")]
    BlackwellSm120,
}

impl BackendKind {
    /// The compute capability string this backend claims to target. Used in
    /// triage against the device's actual probed capability.
    pub fn target_arch(self) -> &'static str {
        match self {
            BackendKind::Stub => "sm_0",
            #[cfg(feature = "sm90")]
            BackendKind::HopperSm90 => "sm_90",
            #[cfg(feature = "sm120")]
            BackendKind::BlackwellSm120 => "sm_120",
        }
    }
}

/// Probe the device and return a boxed backend ready for `helios-core` to
/// drive. The probe order is feature-gated so a per-host build doesn't link
/// probe code for backends it can't compile.
///
/// `package_target` is the `target_arch` from the manifest; triage is run here
/// rather than in `core` so the factory can refuse before handing out a
/// backend the package can never run on.
pub fn create_backend(package_target: &str) -> Result<(DynBackend, String, Triage)> {
    // Try each compiled-in backend in order. The first that probes a real
    // device wins; this is the natural precedence (newer arch first under
    // `multi`) and keeps the per-host single-feature path a straight line.
    let candidates: &[BackendKind] = &[
        #[cfg(feature = "sm120")]
        BackendKind::BlackwellSm120,
        #[cfg(feature = "sm90")]
        BackendKind::HopperSm90,
        BackendKind::Stub,
    ];

    for &kind in candidates {
        let (backend, actual_cap) = instantiate(kind)?;
        let triage = Triage::evaluate(package_target, &actual_cap)?;
        info!(backend = ?kind, actual = %actual_cap, target = %package_target, outcome = ?triage.outcome, "triaged");
        return Ok((backend, actual_cap, triage));
    }

    Err(anyhow!("no backend compiled in or probeable for this build"))
}

/// Construct one backend and probe its hardware. Kept small per-kind so each
/// backend module owns its own probe + init sequence. The `Stub` path always
/// succeeds and reports `sm_0` so triage (sm_0 vs sm_0) is `Exact`.
fn instantiate(kind: BackendKind) -> Result<(DynBackend, String)> {
    match kind {
        BackendKind::Stub => {
            let b = crate::backends::stub::StubBackend::new();
            let info = b.initialize_device()?;
            Ok((Box::new(b), info.compute_capability))
        }
        #[cfg(feature = "sm90")]
        BackendKind::HopperSm90 => {
            let b = crate::backends::hopper_sm90::HopperBackend::new()?;
            let info = b.initialize_device()?;
            Ok((Box::new(b), info.compute_capability))
        }
        #[cfg(feature = "sm120")]
        BackendKind::BlackwellSm120 => {
            let b = crate::backends::blackwell_sm120::BlackwellBackend::new()?;
            let info = b.initialize_device()?;
            Ok((Box::new(b), info.compute_capability))
        }
    }
}
