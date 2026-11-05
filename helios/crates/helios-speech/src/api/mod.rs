//! Pure request and response logic of the audio endpoints (ADR 0012 section 5): no HTTP framework, no GPU. The axum handlers are thin wrappers
//! registered by one hook line; everything that can be wrong with a request is decided here and tested on the host.
pub mod error;
pub mod multipart;
pub mod stt_format;
pub mod stt_request;
pub mod tts_request;
