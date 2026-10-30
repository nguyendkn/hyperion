//! Shared CUDA Driver-API core (ADR 0001): context, stream, device memory and
//! weight upload, used by every NVIDIA generation. Only kernels are per
//! architecture (`backends/<arch>/`). Real driver calls throughout, loaded at run
//! time (`sys`, `loader`): nothing links against the CUDA toolkit or the driver, so
//! the module builds on any host, and a host without a driver gets a typed
//! `LoadError` when a backend probes the device. It exists only behind the `cuda`
//! feature (enabled by `sm90` and `sm120`).

pub mod cublas;
pub mod cublaslt;
pub mod device;
pub mod event;
mod ffi;
pub mod graph;
pub mod loader;
pub mod module;
pub mod nccl;
pub mod profile;
pub mod sys;
pub mod vmm;
pub mod work_guard;
pub mod weights;

pub use cublas::Cublas;
pub use cublaslt::CublasLt;

/// One line naming the driver and cuBLAS libraries that were actually loaded and the driver version.
pub fn loaded_libraries() -> String {
    let driver = sys::loaded_driver().map_or_else(|| "driver library not loaded".to_string(), |(name, version)| format!("driver library {name} (CUDA driver version {version})"));
    let cublas = cublas::loaded_library().map_or_else(|| "cuBLAS library not loaded".to_string(), |name| format!("cuBLAS library {name}"));
    let lt = cublaslt::loaded_library().map_or_else(|| "cuBLASLt library not loaded".to_string(), |name| format!("cuBLASLt library {name}"));
    let nccl = nccl::loaded_library().map_or_else(|| "NCCL library not loaded".to_string(), |name| format!("NCCL library {name}"));
    format!("{driver}; {cublas}; {lt}; {nccl}")
}
pub use device::CudaDevice;
pub use module::{Kernel, KernelModule};
pub use ffi::eval_cuda;
pub use graph::GraphExec;
pub use vmm::VmmRange;
pub use weights::{DeviceWeights, Group};
