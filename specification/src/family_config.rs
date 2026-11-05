//! Family sections of the speech manifests (ADR 0012, 0013, 0014). Plain data: every number comes from the model's real config at the
//! pinned revision (Ignition refuses a missing source), nothing is defaulted. A new architecture is a new enum variant plus its Ignition compiler.
use crate::v1::ModelConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum WindowKind {
    /// Periodic Hann, `0.5 - 0.5 cos(2 pi i / n)`.
    HannPeriodic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum MelScale {
    /// Slaney scale and Slaney area normalisation (librosa default, used by the Whisper feature extractor).
    Slaney,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum FeatureNormalization {
    /// `log10`, floor at (max - 8), then (x + 4) / 4.
    WhisperLogMax8,
    /// `log10` only.
    None,
}

/// Audio front end exactly as the reference defines it; shared by every speech family (`helios-audio::mel::MelConfig` is built from it).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AudioFrontendConfig {
    pub sample_rate: u32,
    pub n_fft: u32,
    pub hop: u32,
    pub n_mels: u32,
    pub f_min: f64,
    pub f_max: f64,
    pub window: WindowKind,
    pub mel_scale: MelScale,
    pub normalization: FeatureNormalization,
    /// Longest chunk the front end hands to the encoder, in seconds.
    pub chunk_seconds: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum SttArchitecture {
    /// Audio encoder (conv2d subsampling + transformer) whose output is spliced into a Qwen3 decoder's input embeddings.
    Qwen3Asr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AudioEncoderConfig {
    pub num_layers: u32,
    pub hidden_size: u32,
    pub num_heads: u32,
    pub intermediate_size: u32,
    /// Channels after the stride-2 convolutions.
    pub conv_channels: u32,
    /// Total time downsampling of the convolutions (8 for three stride-2 layers).
    pub subsample: u32,
    /// Mel frames per convolution chunk (the reference's `2 * n_window`; 100).
    pub chunk_mel_frames: u32,
    /// Mel frames per attention block (the reference's `n_window_infer`; 800, 104 tokens after the convolutions); attention is bidirectional inside a block.
    pub attention_window_mel_frames: u32,
    /// Width of the projector output, equal to the decoder hidden size.
    pub output_dim: u32,
    /// Length of the sinusoidal position table.
    pub max_source_positions: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SttConfig {
    pub architecture: SttArchitecture,
    pub frontend: AudioFrontendConfig,
    pub encoder: AudioEncoderConfig,
    /// The decoder is a complete `decoder_lm` description, so the unchanged LLM path serves it.
    pub decoder: ModelConfig,
    /// Token id of the placeholder whose embedding row is replaced by an encoder frame (the splice rule of the reference).
    pub audio_token_id: u32,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum TtsArchitecture {
    /// LLM backbone emitting multi-codebook codec frames, small depth decoder per frame, causal transformer codec.
    VieneuV3,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CodecConfig {
    pub sample_rate: u32,
    /// Frames per second times 1000 (12.5 Hz is 12500), an integer so the manifest has no float to compare.
    pub frame_rate_milli_hz: u32,
    pub num_codebooks: u32,
    pub codebook_size: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DepthDecoderConfig {
    pub num_layers: u32,
    pub hidden_size: u32,
    pub num_heads: u32,
    pub intermediate_size: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TtsConfig {
    pub architecture: TtsArchitecture,
    pub backbone: ModelConfig,
    pub depth_decoder: DepthDecoderConfig,
    pub codec: CodecConfig,
    /// Preset voices the package ships; a request naming another voice is a 400 (cloning is not supported).
    pub voices: Vec<String>,
}
