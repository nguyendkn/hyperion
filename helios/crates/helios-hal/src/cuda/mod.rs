//! Shared CUDA Driver-API core (ADR 0001): context, stream, device memory and
//! weight upload, used by every NVIDIA generation. Only kernels are per
//! architecture (`backends/<arch>/`). Real driver calls throughout: without the
//! CUDA driver library this module does not link, which is why it exists only
//! behind the `cuda` feature (enabled by `sm90` and `sm120`).

pub mod device;
mod ffi;
pub mod weights;

pub use device::CudaDevice;
pub use ffi::eval_cuda;
pub use weights::DeviceWeights;
