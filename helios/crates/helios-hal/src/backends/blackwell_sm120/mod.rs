//! Blackwell SM120 backend — native CUDA Graph capture for static decode.
//!
//! The defining posture of this generation: capture a prefill-then-decode
//! graph at load time and *replay* it each step, so the decode hot path issues
//! one `cudaGraphLaunch` instead of hundreds of kernel launches. CPU-side
//! overhead drops to near zero — the "zero-overhead" target the architecture
//! doc requires.
//!
//! Submodules:
//! - the CUDA device/memory layer is the shared core in `crate::cuda`.
//! - [`graph`]   — `StaticCudaGraph` lifecycle wrapper (capture / replay).
//! - [`kernels`] — kernel-launch surface; stands in for FlashInfer's
//!   attention/matmul kernels via the package's PTX until those are vendored.

pub mod graph;
pub mod kernels;

mod backend;

pub use backend::BlackwellBackend;
