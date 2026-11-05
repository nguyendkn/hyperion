//! Manifest envelope of the non-LLM families (ADR 0012). The LLM `v1::Manifest` is not touched: `parse_any` looks at the `family` key and sends
//! manifests without it (every existing one) to `Manifest::from_json_bytes` unchanged.
use crate::family_config::{SttConfig, TtsConfig};
use crate::v1::{is_bare_file_name, LayerMapping, Payload, QualifiedCheckpoint, TensorFormat};
use crate::{Manifest, ManifestError};
use serde::{Deserialize, Serialize};

/// Contract revision of the family envelope; a reader refuses any other value before looking at other fields.
pub const FAMILY_CONTRACT_REVISION: u32 = 1;

/// Families that use this envelope. `decoder_lm` is the implicit family of the LLM manifest and is deliberately not a variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum ModelFamily {
    SttQwen3Asr,
    TtsVieneuV3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct FamilyManifest {
    pub family: ModelFamily,
    pub contract_revision: u32,
    pub model_id: String,
    pub target_arch: String,
    pub tensor_format: TensorFormat,
    /// SPDX id of the weights' licence as verified by the packager (ADR 0013/0014 evidence rules).
    pub license_spdx: String,
    /// Owner gate: false until the owner approves the model for production (unknown training data, unverified licence). The server refuses an
    /// unapproved model unless started with an explicit override.
    #[serde(default)]
    pub production_approved: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qualified_checkpoint: Option<QualifiedCheckpoint>,
    pub payloads: Vec<Payload>,
    /// Weight tensors with byte offsets into their payload (the LLM `LayerMapping`, reused).
    pub tensors: Vec<LayerMapping>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stt: Option<SttConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tts: Option<TtsConfig>,
    #[serde(default)]
    pub signature: Option<Vec<u8>>,
}

impl FamilyManifest {
    pub fn validate(&self) -> Result<(), ManifestError> {
        let bad = |m: String| Err(ManifestError::Invalid(m));
        let (want_stt, want_tts) = (self.family == ModelFamily::SttQwen3Asr, self.family == ModelFamily::TtsVieneuV3);
        if self.stt.is_some() != want_stt || self.tts.is_some() != want_tts {
            return bad(format!("family {:?} needs exactly its own section (stt: {}, tts: {})", self.family, self.stt.is_some(), self.tts.is_some()));
        }
        if self.payloads.is_empty() {
            return bad("a speech package lists its payload files".into());
        }
        for (i, p) in self.payloads.iter().enumerate() {
            if !is_bare_file_name(&p.file) {
                return bad(format!("payload {i}: `{}` is not a bare file name", p.file));
            }
            if p.sha256.len() != 64 || !p.sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
                return bad(format!("payload {i} (`{}`): sha256 must be 64 lowercase hex digits", p.file));
            }
        }
        if let Some(t) = self.tensors.iter().find(|t| t.payload as usize >= self.payloads.len()) {
            return bad(format!("tensor `{}` refers to payload {} but the manifest has {}", t.name, t.payload, self.payloads.len()));
        }
        if let Some(t) = &self.tts {
            if t.voices.is_empty() || t.voices.iter().any(|v| v.is_empty()) {
                return bad("a TTS package ships at least one named preset voice".into());
            }
        }
        Ok(())
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, ManifestError> {
        let value: serde_json::Value = serde_json::from_slice(bytes)?;
        check_revision(&value)?;
        let m: FamilyManifest = serde_json::from_value(value)?;
        m.validate()?;
        Ok(m)
    }
}

fn check_revision(value: &serde_json::Value) -> Result<(), ManifestError> {
    let found = value.get("contract_revision").and_then(|v| v.as_u64());
    if found != Some(u64::from(FAMILY_CONTRACT_REVISION)) {
        return Err(ManifestError::UnsupportedContractVersion {
            found: format!("family contract_revision {}", found.map_or("missing".to_string(), |r| r.to_string())),
            supported: format!("family contract_revision {FAMILY_CONTRACT_REVISION}"),
            min_ignition: crate::v1::MIN_IGNITION_VERSION,
        });
    }
    Ok(())
}

/// Either kind of package.
#[derive(Debug, Clone)]
pub enum AnyManifest {
    Llm(Manifest),
    Family(FamilyManifest),
}

/// Dispatch on the `family` key. Absent: the LLM manifest, parsed by the unchanged `Manifest::from_json_bytes`. `"decoder_lm"` written out is
/// refused (the LLM manifest has no such key; two spellings of one thing would mean two goldens).
pub fn parse_any(bytes: &[u8]) -> Result<AnyManifest, ManifestError> {
    let value: serde_json::Value = serde_json::from_slice(bytes)?;
    match value.get("family") {
        None => Manifest::from_json_bytes(bytes).map(AnyManifest::Llm),
        Some(serde_json::Value::String(s)) if s == "decoder_lm" => Err(ManifestError::Invalid("`family: decoder_lm` is implicit: omit the key".into())),
        Some(_) => FamilyManifest::from_json_bytes(bytes).map(AnyManifest::Family),
    }
}
