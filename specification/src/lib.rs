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
}
