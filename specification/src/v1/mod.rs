//! Manifest version 1 — the original `.hyperion` package format.
//!
//! Stability contract: v1 stays editable until the first package is released.
//! From then on these structs are frozen — new fields go in a new major
//! version, not here. Unknown fields are rejected (`deny_unknown_fields`) so a
//! typo or a field from a newer writer fails loudly instead of being dropped.
//! Optional fields use `Option<T>` and `#[serde(default)]`.
//!
//! These structs are the schema source of truth: with the `schema` feature,
//! `specification/schema/manifest.v1.json` is generated from them.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Top-level descriptor embedded in every `.hyperion` package.
///
/// Written by Ignition at compile time, read by Helios at load time. The
/// `target_arch` field is what drives HAL backend selection — see
/// `helios-hal::backend_factory`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Manifest {
    /// Identifier of the compiled model, e.g. "llama3".
    pub model_id: String,

    /// Compute capability the package was compiled for, e.g. "sm_90". The
    /// runtime accepts the package only on hardware that reports exactly this.
    pub target_arch: String,

    /// Model dimensions the runtime needs to size caches and buffers. Backends
    /// read these instead of hard-coding them.
    pub model: ModelConfig,

    /// Tokenizer configuration serialized alongside the weights.
    #[serde(default)]
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

/// Architecture dimensions of the compiled model. All fields are required: a
/// manifest that cannot say how big its KV cache and vocabulary are is not
/// loadable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ModelConfig {
    /// Number of transformer layers.
    pub num_layers: u32,

    /// Number of attention (query) heads.
    pub num_attention_heads: u32,

    /// Number of key/value heads (equals `num_attention_heads` without GQA).
    pub num_kv_heads: u32,

    /// Dimension of one attention head.
    pub head_dim: u32,

    /// Vocabulary size (logits width).
    pub vocab_size: u32,

    /// Maximum sequence length the model was compiled with. Requests beyond
    /// this are rejected to avoid KV cache overrun.
    pub max_seq_len: u32,
}

/// Tokenizer metadata. Kept intentionally thin — the actual vocab/merges are
/// in a separate asset in the package, referenced by `asset_path`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TokenizerConfig {
    /// Family: "bpe", "spm", "rune", … drives the decoder impl in runtime.
    pub family: String,

    /// Path inside the package to the tokenizer asset (vocab + merges). Empty
    /// means the package ships without a tokenizer.
    pub asset_path: PathBuf,
}

/// Maps one transformer layer to its weight tensor(s) inside the package blob.
/// Byte offsets let the loader mmap the blob and slice without copying.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TensorMetadata {
    /// "bf16" | "fp8" | "fp4" | "fp6" — matches kernel dtype flags.
    pub dtype: String,

    /// Logical shape in row-major order, e.g. [4096, 4096].
    pub shape: Vec<usize>,
}
