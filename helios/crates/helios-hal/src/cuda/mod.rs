//! Shared CUDA Driver-API core (ADR 0001): context, stream, device memory and
//! weight upload, used by every NVIDIA generation. Only kernels are per
//! architecture (`backends/<arch>/`). Real driver calls throughout, loaded at run
//! time (`sys`, `loader`): nothing links against the CUDA toolkit or the driver, so
//! the module builds on any host, and a host without a driver gets a typed
//! `LoadError` when a backend probes the device. It exists only behind the `cuda`
//! feature (enabled by `sm90` and `sm120`).

pub mod cublas;
pub mod device;
pub mod event;
mod ffi;
pub mod graph;
pub mod loader;
pub mod module;
pub mod profile;
pub mod sys;
pub mod weights;

pub use cublas::Cublas;
pub use device::CudaDevice;
pub use module::{Kernel, KernelModule};
pub use ffi::eval_cuda;
pub use graph::GraphExec;
pub use weights::DeviceWeights;
