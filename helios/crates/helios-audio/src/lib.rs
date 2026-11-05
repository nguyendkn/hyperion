//! Audio front end of the speech families (ADR 0012). Pure Rust, no dependency, deterministic: the same input gives the same bits on every host
//! (no threads, no fused multiply-add reliance beyond what `f64` arithmetic does, no hardware intrinsics).

pub mod error;
pub mod fft;
pub mod mel;
pub mod resample;
pub mod wav;

pub use error::AudioError;
pub use wav::Audio;
