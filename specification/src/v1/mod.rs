//! Manifest version 1 — the original `.hyperion` package format.
//!
//! Stability contract: once `v1` ships, these structs are frozen. New fields
//! are added in a new major version, not here. Optional fields use `Option<T>`
//! and `#[serde(default)]` so older packages still load after minor additions.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Top-level descriptor embedded in every `.hyperion` package.
///
/// Written by Ignition at compile time, read by Helios at load time. The
/// `target_arch` field is what drives HAL backend selection — see
/// `helios-hal::backend_factory`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Semantic version of the model itself, e.g. "llama3-8b-v1".
    pub model_id: String,

    /// Minimum compute capability the package was compiled for, e.g. "sm_90".
    /// Runtime may run on a *higher* capability (forward-compat) but never lower.
    pub target_arch: String,

    /// Tokenizer configuration serialized alongside the weights. Required — a
    /// package without a tokenizer is broken; we reject it at parse time here
    /// rather than surfacing a 501 per-request at runtime.
    pub tokenizer: TokenizerConfig,

    /// Per-layer weight metadata + byte offsets into the package blob.
    pub layers: Vec<LayerMapping>,

    /// Hardware recipe applied at compile time (FP8/FP4/BF16, CUDA Graph capture).
    /// Stored as free-form k=v so recipes evolve without a schema change.
    #[serde(default)]
    pub hardware_recipe: HashMap<String, String>,

    /// Ed25519 signature over the weight blob, verified at load. Absent =
    /// unsigned package (allowed in dev builds, rejected by release loader).
    #[serde(default)]
    pub signature: Option<Vec<u8>>,
}

/// Tokenizer metadata. Kept intentionally thin — the actual vocab/merges are
/// in a separate asset in the package, referenced by `asset_path`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Family: "bpe", "spm", "rune", … drives the decoder impl in runtime.
    pub family: String,

    /// Path inside the package to the tokenizer asset (vocab + merges).
    pub asset_path: PathBuf,

    /// Maximum sequence length the model was compiled with. Decode beyond this
    /// is rejected to avoid KV cache overrun.
    #[serde(default)]
    pub max_seq_len: Option<u32>,
}

/// Maps one transformer layer to its weight tensor(s) inside the package blob.
/// Byte offsets let the loader mmap the blob and slice without copying.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMapping {
    /// 0-based layer index in the model graph.
    pub index: u32,

    /// Logical name, e.g. "attention.q_proj" — matches the kernel registry.
    pub name: String,

    /// On-package byte range where this tensor's weights begin.
    pub byte_offset: u64,

    /// Length in bytes of the weight blob for this tensor.
    pub byte_length: u64,

    /// Tensor metadata describing dtype/shape — see [`TensorMetadata`].
    pub tensor: TensorMetadata,
}

/// Shape + dtype of a single tensor. Mirrors `helios_hal::TensorData` but
/// stays decoupled from device pointers — this is on-package metadata only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    /// "bf16" | "fp8" | "fp4" | "fp6" — matches kernel dtype flags.
    pub dtype: String,

    /// Logical shape in row-major order, e.g. [4096, 4096].
    pub shape: Vec<usize>,
}
