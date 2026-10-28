//! Blackwell SM120 backend — native CUDA Graph capture for static decode.
//!
//! The defining posture of this generation: capture a prefill-then-decode
//! graph at load time and *replay* it each step, so the decode hot path issues
//! one `cudaGraphLaunch` instead of hundreds of kernel launches. CPU-side
//! overhead drops to near zero — the "zero-overhead" target the architecture
//! doc requires.
//!
//! Submodules:
//! - [`device`]  — driver-level probe + memory allocation, real FFI.
//! - [`graph`]   — `StaticCudaGraph` lifecycle wrapper (capture / replay).
//! - [`kernels`] — kernel-launch surface; stands in for FlashInfer's
//!   attention/matmul kernels via the package's PTX until those are vendored.

pub mod device;
pub mod ffi;
pub mod graph;
pub mod kernels;

mod backend;

pub use backend::BlackwellBackend;
