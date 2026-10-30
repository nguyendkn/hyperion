//! # hyperion-specification
//!
//! Versioned data contract for `.hyperion` model packages. Each major version
//! lives in its own module (`v1`, `2`, …) so the wire format never breaks: a
//! package written under `v1` must deserialize under `v1` forever, regardless
//! of later additions.
//!
//! ## Why a dedicated crate
//! The manifest is read by *both* Ignition (Python, at compile time, via the
//! committed JSON schema `schema/manifest.v1.json`) and Helios (Rust, at load
//! time). Keeping the Rust
//! definition here lets the runtime verify a package against an exact, typed
//! contract instead of free-form JSON, and lets FFI expose the same structs.

pub mod identity;
pub mod sha256;
pub mod v1;

#[cfg(feature = "schema")]
pub mod schema;

use thiserror::Error;

/// Errors raised when a manifest cannot be parsed or its version is unknown.
#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("unsupported manifest version: {0}")]
    UnknownVersion(u32),

    #[error("malformed manifest: {0}")]
    Malformed(#[from] serde_json::Error),

    #[error("bincode decode failure: {0}")]
    Decode(#[from] bincode::Error),

    #[error("invalid payload table: {0}")]
    InvalidPayloads(String),

    /// The manifest was written under a contract this build does not read. The package is not
    /// converted: it is exported again by a current Ignition.
    #[error("unsupported manifest contract ({found}); this build reads {supported}: re-export the package with Ignition >= {min_ignition}")]
    UnsupportedContractVersion { found: String, supported: String, min_ignition: &'static str },

    #[error("invalid manifest: {0}")]
    Invalid(String),
}

/// Discriminated top-level wrapper so loaders can branch on version before
/// committing to a concrete struct. New versions add a variant here — never
/// mutate an existing one. On the wire the tag is the variant name, e.g.
/// `"manifest_version": "V1"`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "manifest_version")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum Manifest {
    V1(v1::Manifest),
}

impl Manifest {
    /// Current (highest) major version this build understands.
    pub const LATEST: u32 = 1;

    /// Parse a manifest from its stored bytes. The version tag and the contract revision are checked
    /// first, so a package from an older or newer writer is refused with
    /// [`ManifestError::UnsupportedContractVersion`] (and its re-export hint) rather than with a
    /// complaint about whichever field happens to be missing.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, ManifestError> {
        let value: serde_json::Value = serde_json::from_slice(bytes)?;
        let supported = format!("manifest_version V1, contract_revision {}", v1::CONTRACT_REVISION);
        let refuse = |found: String| ManifestError::UnsupportedContractVersion { found, supported: supported.clone(), min_ignition: v1::MIN_IGNITION_VERSION };
        let version = value.get("manifest_version").and_then(|v| v.as_str());
        if version != Some("V1") {
            return Err(refuse(format!("manifest_version {}", version.map_or("missing".to_string(), |v| format!("`{v}`")))));
        }
        let revision = value.get("contract_revision").and_then(|v| v.as_u64());
        if revision != Some(u64::from(v1::CONTRACT_REVISION)) {
            return Err(refuse(format!("contract_revision {}", revision.map_or("missing".to_string(), |r| r.to_string()))));
        }
        let manifest: Manifest = serde_json::from_value(value)?;
        let Manifest::V1(inner) = &manifest;
        inner.validate()?;
        Ok(manifest)
    }
}
