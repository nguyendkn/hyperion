//! Family sections of the speech manifests (ADR 0012, 0013, 0014). Plain data: every number comes from the model's real config at the
//! pinned revision (Ignition refuses a missing source), nothing is defaulted. A new architecture is a new enum variant plus its Ignition compiler.
use crate::v1::ModelConfig;
use serde::{Deserialize, Serialize};

/// Audio front end exactly as the reference defines it (`helios-audio::mel::MelConfig` is built from this).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AudioFrontend {
    pub sample_rate: u32,
    pub n_fft: u32,
    pub hop: u32,
    pub n_mels: u32,
    pub f_min: f64,
    pub f_max: f64,
    /// Whisper-style floor at (max - 8) and (x + 4) / 4.
    pub whisper_norm: bool,
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
    /// Attention window in encoder frames, shortest and longest (bidirectional inside the window).
    pub window_frames_min: u32,
    pub window_frames_max: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SttConfig {
    pub architecture: SttArchitecture,
    pub frontend: AudioFrontend,
    pub encoder: AudioEncoderConfig,
    /// The decoder is a complete `decoder_lm` description, so the unchanged LLM path serves it.
    pub decoder: ModelConfig,
    /// Token id of the placeholder whose embedding row is replaced by an encoder frame (the splice rule of the reference).
    pub audio_token_id: u32,
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
