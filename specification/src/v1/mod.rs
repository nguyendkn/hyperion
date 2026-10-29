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

    /// Weight packing format of the package. Together with `target_arch` this
    /// is what the runtime checks before loading (e.g. fp4 needs Blackwell).
    pub tensor_format: TensorFormat,

    /// Model dimensions the runtime needs to size caches and buffers. Backends
    /// read these instead of hard-coding them.
    pub model: ModelConfig,

    /// Tokenizer configuration serialized alongside the weights.
    #[serde(default)]
    pub tokenizer: TokenizerConfig,

    /// Per-layer weight metadata + byte offsets into the package blob.
    pub layers: Vec<LayerMapping>,

    /// The weights files of the package, in the order `layers[].payload` indexes them, each with its
    /// size and sha256. Absent (empty) means one file, [`DEFAULT_PAYLOAD_FILE`], with nothing recorded
    /// about it, which is what a header-only compile and every older manifest look like.
    /// A checkpoint split over several files (`model-00001-of-00004.safetensors` ...) needs the table.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub payloads: Vec<Payload>,

    /// Hardware recipe applied at compile time (FP8/FP4/BF16, CUDA Graph capture).
    /// Stored as free-form k=v so recipes evolve without a schema change.
    #[serde(default)]
    pub hardware_recipe: HashMap<String, String>,

    /// Ed25519 signature over the weight blob, verified at load. Absent =
    /// unsigned package (allowed in dev builds, rejected by release loader).
    #[serde(default)]
    pub signature: Option<Vec<u8>>,
}

/// Weight packing format of a package. Per-tensor dtypes live in
/// [`TensorMetadata`]; this is the package-level format the recipe selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum TensorFormat {
    Bf16,
    Fp8,
    Fp4,
}

/// How the attention layers lay out their KV cache. Decides the geometry that
/// `num_kv_heads` and `head_dim` describe; a variant with a different cache
/// shape (e.g. latent attention) is added here, never guessed from the dims.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum AttentionKind {
    /// Multi-head: `num_kv_heads == num_attention_heads`.
    Mha,
    /// Grouped-query: `num_kv_heads < num_attention_heads`, divides it evenly.
    Gqa,
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

    /// KV cache layout; see [`AttentionKind`].
    pub attention: AttentionKind,

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

    /// Index into the manifest's payload table of the file that holds this tensor (0, the default,
    /// is the first or only payload).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub payload: u32,

    /// Byte range where this tensor's weights begin, counted from the start of the data section of its payload file.
    pub byte_offset: u64,

    /// Length in bytes of the weight blob for this tensor.
    pub byte_length: u64,

    /// Tensor metadata describing dtype/shape — see [`TensorMetadata`].
    pub tensor: TensorMetadata,
}

fn is_zero(value: &u32) -> bool {
    *value == 0
}

/// The name of the weights file of a package that has no payload table.
pub const DEFAULT_PAYLOAD_FILE: &str = "model.safetensors";

/// One weights file of the package. The file sits beside the package under exactly this name; the
/// name is part of what is signed, so a swapped or edited file is detectable once the runtime
/// checks `sha256` (planned with package signing; the runtime does not verify it yet).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Payload {
    /// File name only: no directory separator, no `:`, not `.` or `..`. See [`is_bare_file_name`].
    #[cfg_attr(feature = "schema", schemars(length(min = 1, max = 255), regex(pattern = r"^[^/\\:]+$")))]
    pub file: String,

    /// Size of the file in bytes.
    pub byte_length: u64,

    /// Lowercase hexadecimal SHA-256 of the whole file.
    #[cfg_attr(feature = "schema", schemars(regex(pattern = r"^[0-9a-f]{64}$")))]
    pub sha256: String,
}

/// Whether `name` is a plain file name that cannot leave the package directory and opens an
/// ordinary file on every platform: non-empty, at most 255 bytes, not `.` or `..`, without a path
/// separator (`/`, `\`), a drive or stream separator (`:`) or a NUL, not ending in a dot or a space
/// (Windows drops those), and not a Windows device name (`CON`, `PRN`, `AUX`, `NUL`, `COM0`-`COM9`,
/// `LPT0`-`LPT9`, `CONIN$`, `CONOUT$`, in any case and with any extension, so `NUL.safetensors` is
/// refused): opening one of those on Windows reaches a device, not a file.
pub fn is_bare_file_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 255
        && name != "."
        && name != ".."
        && !name.ends_with('.')
        && !name.ends_with(' ')
        && !name.chars().any(|c| matches!(c, '/' | '\\' | ':' | '\0'))
        && !is_windows_device_name(name)
}

/// Whether Windows treats `name` as a device: the part before the first dot, without trailing
/// spaces, is a reserved name (case-insensitive).
fn is_windows_device_name(name: &str) -> bool {
    let stem = name.split('.').next().unwrap_or(name).trim_end_matches(' ').to_uppercase();
    if matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$") {
        return true;
    }
    let mut chars = stem.chars();
    let prefix: String = chars.by_ref().take(3).collect();
    matches!(prefix.as_str(), "COM" | "LPT")
        && matches!(chars.next(), Some('0'..='9' | '\u{b9}' | '\u{b2}' | '\u{b3}'))
        && chars.next().is_none()
}

impl Manifest {
    /// The weights file names in payload-index order: the table's files, or just
    /// [`DEFAULT_PAYLOAD_FILE`] when the manifest has no table.
    pub fn payload_files(&self) -> Vec<&str> {
        if self.payloads.is_empty() {
            vec![DEFAULT_PAYLOAD_FILE]
        } else {
            self.payloads.iter().map(|p| p.file.as_str()).collect()
        }
    }

    /// Check the payload table and every layer's reference to it: names are bare and unique, hashes are
    /// 64 lowercase hex digits, and each layer points at an existing payload. Call it after parsing
    /// a manifest that came from outside, before any path is built from it.
    pub fn validate_payloads(&self) -> Result<(), crate::ManifestError> {
        let invalid = |message: String| Err(crate::ManifestError::InvalidPayloads(message));
        let mut seen = std::collections::HashSet::new();
        for (index, payload) in self.payloads.iter().enumerate() {
            if !is_bare_file_name(&payload.file) {
                return invalid(format!("payload {index}: `{}` is not a bare file name", payload.file));
            }
            if !seen.insert(payload.file.as_str()) {
                return invalid(format!("payload {index}: `{}` is listed twice", payload.file));
            }
            let hex = &payload.sha256;
            if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
                return invalid(format!("payload {index} (`{}`): sha256 must be 64 lowercase hex digits", payload.file));
            }
        }
        let count = self.payload_files().len();
        for layer in &self.layers {
            if layer.payload as usize >= count {
                return invalid(format!("layer `{}` refers to payload {} but the manifest has {count}", layer.name, layer.payload));
            }
        }
        Ok(())
    }
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
