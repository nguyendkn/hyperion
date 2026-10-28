//! Runtime hardware triage: decide whether a package can run on the backend
//! the factory just probed. Pure functions, no GPU needed to test them.
//!
//! The rule is **exact match only** (CLAUDE.md rule 6): the manifest's
//! `target_arch` must equal the probed device's capability string, the backend
//! must be the one written for that device, and the manifest's `tensor_format`
//! must be one the backend can run. There is no forward-compat downgrade: a
//! package is compiled per target by Ignition.
//!
//! The arch -> tensor-format table lives here and only here. Ignition's
//! recipes declare `supported_formats` too; a committed JSON dump of this table
//! (`tests/data/format-table.json`) is compared by a Python test so the two
//! sides cannot drift.

use hyperion_specification::v1::{Manifest, TensorFormat};
use thiserror::Error;

/// Capability the stub backend reports; not a real compile target.
const STUB_ARCH: &str = "sm_0";

/// Compute-capability string -> tensor formats the matching backend can run.
/// Unknown archs map to an empty list, which triage treats as unsupported.
pub fn supported_formats(arch: &str) -> &'static [TensorFormat] {
    match arch {
        // Hopper has no FP4 (CLAUDE.md rule 7).
        "sm_90" => &[TensorFormat::Bf16, TensorFormat::Fp8],
        "sm_120" => &[TensorFormat::Bf16, TensorFormat::Fp8, TensorFormat::Fp4],
        // The stub runs no kernels, so it does not constrain the format.
        "sm_0" => &[TensorFormat::Bf16, TensorFormat::Fp8, TensorFormat::Fp4],
        _ => &[],
    }
}

/// Wire name of a tensor format, as written in the manifest.
pub fn format_name(format: TensorFormat) -> &'static str {
    match format {
        TensorFormat::Bf16 => "bf16",
        TensorFormat::Fp8 => "fp8",
        TensorFormat::Fp4 => "fp4",
    }
}

/// Every known arch with its formats, in a stable order. Used for the
/// committed JSON dump that the Python cross-check test reads.
pub fn format_table() -> Vec<(&'static str, Vec<&'static str>)> {
    ["sm_0", "sm_90", "sm_120"]
        .into_iter()
        .map(|arch| (arch, supported_formats(arch).iter().map(|f| format_name(*f)).collect()))
        .collect()
}

/// `sm_<digits>`; anything else is not a capability string.
fn valid_capability(cap: &str) -> bool {
    cap.strip_prefix("sm_")
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// What the factory learned about a backend by probing real hardware.
#[derive(Debug, Clone)]
pub struct ProbedBackend {
    /// The arch the backend implementation is written for (e.g. `sm_120` for
    /// the Blackwell backend), independent of the device it ended up on.
    pub backend_arch: &'static str,
    /// The capability the device actually reported when probed.
    pub device_arch: String,
}

/// Decision produced by triage before any weight is loaded. Only an exact
/// match is ever accepted, so there is a single variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriageOutcome {
    /// Package arch, device arch and backend arch all agree.
    Exact,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TriageError {
    #[error("package targets {target} but the device is {device}: recompile with `hyperion-compile --hw {device}`")]
    ArchMismatch { target: String, device: String },
    #[error("the {backend} backend cannot drive a {device} device: build helios-cli with the backend feature for {device}")]
    BackendArchMismatch { backend: String, device: String },
    #[error("package targets {target} but this build has no GPU backend for it (only the stub): rebuild helios-cli with the backend feature for {target}")]
    NoGpuBackend { target: String },
    #[error("package tensor_format {format} is not supported on {arch} (supported: {supported}): recompile with a supported format")]
    FormatUnsupported { format: String, arch: String, supported: String },
    #[error("unparseable compute capability: {0}")]
    BadCapability(String),
}

/// A finished triage decision.
#[derive(Debug, Clone)]
pub struct Triage {
    pub outcome: TriageOutcome,
    pub target: String,
    pub actual: String,
}

/// Check a manifest against a probed backend. Order matters for the message:
/// a backend that does not match its own device is reported first, since the
/// package cannot be judged against a backend that should not be running.
pub fn triage(manifest: &Manifest, probed: &ProbedBackend) -> Result<Triage, TriageError> {
    for cap in [manifest.target_arch.as_str(), probed.backend_arch, probed.device_arch.as_str()] {
        if !valid_capability(cap) {
            return Err(TriageError::BadCapability(cap.into()));
        }
    }
    if probed.backend_arch != probed.device_arch {
        return Err(TriageError::BackendArchMismatch {
            backend: probed.backend_arch.into(),
            device: probed.device_arch.clone(),
        });
    }
    if probed.device_arch == STUB_ARCH && manifest.target_arch != STUB_ARCH {
        return Err(TriageError::NoGpuBackend { target: manifest.target_arch.clone() });
    }
    if manifest.target_arch != probed.device_arch {
        return Err(TriageError::ArchMismatch {
            target: manifest.target_arch.clone(),
            device: probed.device_arch.clone(),
        });
    }
    let formats = supported_formats(&probed.device_arch);
    if !formats.contains(&manifest.tensor_format) {
        return Err(TriageError::FormatUnsupported {
            format: format_name(manifest.tensor_format).into(),
            arch: probed.device_arch.clone(),
            supported: if formats.is_empty() {
                "none: unknown arch".into()
            } else {
                formats.iter().map(|f| format_name(*f)).collect::<Vec<_>>().join(", ")
            },
        });
    }
    Ok(Triage {
        outcome: TriageOutcome::Exact,
        target: manifest.target_arch.clone(),
        actual: probed.device_arch.clone(),
    })
}
