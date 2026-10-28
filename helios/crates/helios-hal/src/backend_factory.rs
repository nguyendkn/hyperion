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

use hyperion_specification::v1::Manifest;

use crate::traits::{ComputeBackend, DynBackend, HardwareInfo};
use crate::triage::{triage, ProbedBackend, Triage};

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

/// A backend that probed a real device and passed triage.
pub struct CreatedBackend {
    pub backend: DynBackend,
    /// The capability the device reported when probed (e.g. `sm_90`).
    pub device_arch: String,
    pub triage: Triage,
    /// Probe result (total VRAM, stream limit) for capacity checks.
    pub hardware: HardwareInfo,
}

/// Probe each compiled-in backend in precedence order (sm120, sm90, stub) and
/// return the first one that both probes a real device and passes triage for
/// this manifest. A backend that fails to probe, or whose triage refuses, is
/// skipped and its reason is kept; if none qualifies the error lists every
/// refusal, so "why did nothing run" is answerable from one message.
///
/// Triage runs here rather than in `core` so the factory can refuse before
/// handing out a backend the package can never run on.
pub fn create_backend(manifest: &Manifest) -> Result<CreatedBackend> {
    let candidates: &[BackendKind] = &[
        #[cfg(feature = "sm120")]
        BackendKind::BlackwellSm120,
        #[cfg(feature = "sm90")]
        BackendKind::HopperSm90,
        BackendKind::Stub,
    ];

    let mut refusals: Vec<String> = Vec::new();
    for &kind in candidates {
        let (backend, hardware) = match instantiate(kind) {
            Ok(found) => found,
            Err(e) => {
                info!(backend = ?kind, error = %e, "backend could not probe a device; skipping");
                refusals.push(format!("{kind:?}: no device ({e})"));
                continue;
            }
        };
        let device_arch = hardware.compute_capability.clone();
        let probed = ProbedBackend { backend_arch: kind.target_arch(), device_arch: device_arch.clone() };
        match triage(manifest, &probed) {
            Ok(decision) => {
                info!(backend = ?kind, actual = %device_arch, target = %manifest.target_arch, outcome = ?decision.outcome, "triaged");
                return Ok(CreatedBackend { backend, device_arch, triage: decision, hardware });
            }
            Err(e) => {
                info!(backend = ?kind, actual = %device_arch, error = %e, "triage refused backend");
                refusals.push(format!("{kind:?}: {e}"));
            }
        }
    }

    Err(anyhow!(
        "no compiled-in backend can run this package (target_arch {}, tensor_format {}):
  {}",
        manifest.target_arch,
        crate::triage::format_name(manifest.tensor_format),
        refusals.join("
  ")
    ))
}

/// Construct one backend and probe its hardware. Kept small per-kind so each
/// backend module owns its own probe + init sequence. The `Stub` path always
/// succeeds and reports `sm_0`, so a real package (sm_90 / sm_120) is refused
/// by triage instead of silently pretending to work.
fn instantiate(kind: BackendKind) -> Result<(DynBackend, HardwareInfo)> {
    match kind {
        BackendKind::Stub => {
            let b = crate::backends::stub::StubBackend::new();
            let info = b.initialize_device()?;
            Ok((Box::new(b), info))
        }
        #[cfg(feature = "sm90")]
        BackendKind::HopperSm90 => {
            let b = crate::backends::hopper_sm90::HopperBackend::new()?;
            let info = b.initialize_device()?;
            Ok((Box::new(b), info))
        }
        #[cfg(feature = "sm120")]
        BackendKind::BlackwellSm120 => {
            let b = crate::backends::blackwell_sm120::BlackwellBackend::new()?;
            let info = b.initialize_device()?;
            Ok((Box::new(b), info))
        }
    }
}
