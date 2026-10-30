//! Blackwell SM120 backend — native CUDA Graph capture for static decode.
//!
//! The defining posture of this generation: capture a prefill-then-decode
//! graph at load time and *replay* it each step, so the decode hot path issues
//! one `cudaGraphLaunch` instead of hundreds of kernel launches. CPU-side
//! overhead drops to near zero — the "zero-overhead" target the architecture
//! doc requires.
//!
//! Only the device probe and the capabilities are real today: weight upload, the KV pool and the graph capture return
//! `HalError::Unsupported` (the CUDA device/memory/graph layer is the shared core in `crate::cuda`). The placeholder kernel
//! module that stood in for FlashInfer went with the unwired trait methods it served.

mod backend;

pub use backend::BlackwellBackend;
