//! `cache_identity`: what a stored KV block or a compiled graph may be reused under. Derived from the
//! manifest, never stored in it: a stored value could drift from the content it claims to describe,
//! and a package whose weights, layout or target changed must not keep its old identity.

use crate::sha256::{sha256_hex, Sha256};
use crate::v1::{Manifest, TensorFormat};

/// Domain separation: the same bytes hashed for another purpose give another digest.
const DOMAIN: &[u8] = b"hyperion.cache-identity.v1";

fn field(hasher: &mut Sha256, bytes: &[u8]) {
    // Length-prefixed, so ("ab", "c") and ("a", "bc") differ.
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

impl Manifest {
    /// Hash of the model configuration as serialised (field order is the declaration order).
    pub fn model_config_hash(&self) -> String {
        sha256_hex(&serde_json::to_vec(&self.model).expect("a ModelConfig always serialises"))
    }

    /// Hash of how the weights are laid out: every tensor's name, payload, offset, length, dtype and shape, in order.
    pub fn layout_fingerprint(&self) -> String {
        let mut h = Sha256::new();
        for layer in &self.layers {
            field(&mut h, &layer.index.to_le_bytes());
            field(&mut h, layer.name.as_bytes());
            field(&mut h, &layer.payload.to_le_bytes());
            field(&mut h, &layer.byte_offset.to_le_bytes());
            field(&mut h, &layer.byte_length.to_le_bytes());
            field(&mut h, layer.tensor.dtype.as_bytes());
            for dim in &layer.tensor.shape {
                field(&mut h, &(*dim as u64).to_le_bytes());
            }
        }
        h.finalize_hex()
    }

    /// 64 lowercase hex digits identifying everything a cached block depends on: the payload hashes,
    /// the layout, the target architecture, the tensor format and the model configuration. Equal
    /// identities mean the same weights run the same way on the same hardware.
    pub fn cache_identity(&self) -> String {
        let mut h = Sha256::new();
        h.update(DOMAIN);
        h.update(&[0]);
        field(&mut h, &(self.payloads.len() as u64).to_le_bytes());
        for payload in &self.payloads {
            field(&mut h, payload.sha256.as_bytes());
        }
        field(&mut h, self.layout_fingerprint().as_bytes());
        field(&mut h, self.target_arch.as_bytes());
        let format = match self.tensor_format {
            TensorFormat::Bf16 => "bf16",
            TensorFormat::Fp8 => "fp8",
            TensorFormat::Fp4 => "fp4",
        };
        field(&mut h, format.as_bytes());
        field(&mut h, self.model_config_hash().as_bytes());
        h.finalize_hex()
    }
}
