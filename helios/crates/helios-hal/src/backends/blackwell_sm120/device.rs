//! CUDA Driver FFI surface for the SM120 backend — device probing, memory
//! allocation, stream handle. All calls go through `cuda-driver-sys`, which
//! binds the real CUDA Driver headers. There is **no mock here**: on a host
//! without the CUDA Driver library present, *this module will not link*, and
//! a build with `--features sm120` will fail at the link step. That is
//! intended — CI and dev hosts use `--features stub` instead.
//!
//! We use the **driver** API (not runtime) because CUDA Graph capture lives
//! in the driver. Going straight to the driver matches the zero-overhead
//! posture and avoids the runtime's global state.

use anyhow::Result;
use cuda_driver_sys::{
    cuCtxSetCurrent, cuDeviceGet, cuDeviceGetAttribute, cuDeviceGetCount, cuDevicePrimaryCtxRetain,
    cuDeviceTotalMem_v2, cuInit, cuMemAlloc_v2, cuMemFree_v2, cuStreamCreate, cuStreamDestroy_v2,
    CUcontext, CUdevice, CUdeviceptr, CUstream, CUdevice_attribute,
};
use std::os::raw::c_int;

use super::ffi::eval_cuda;

/// `CU_STREAM_NON_BLOCKING = 1` — the driver's `cuStreamCreate` takes the
/// flags as a raw `c_uint`, not the `CUstream_flags` enum, so we pass the
/// literal value the enum would carry.
const CU_STREAM_NON_BLOCKING: std::os::raw::c_uint = 1;

/// Owned handle to a CUDA device + its primary context. The context is
/// retained for the backend's lifetime and torn down explicitly.
pub struct CudaDevice {
    device: CUdevice,
    ctx: CUcontext,
    /// Cached at probe time — the scheduler reads these without re-querying.
    cap_major: i32,
    cap_minor: i32,
    total_vram: usize,
    stream: CUstream,
}

// Safety: the device handle and context are owned by this struct and only
// touched through `&mut` methods behind a mutex in the backend; the driver
// is thread-safe w.r.t. distinct contexts. `Send` is the only bound needed.
unsafe impl Send for CudaDevice {}

impl CudaDevice {
    /// Probe the first available device, create a primary context, read its
    /// properties, and stand up a non-blocking capture stream.
    pub fn probe() -> Result<Self> {
        unsafe {
            eval_cuda("cuInit", cuInit(0))?;

            let mut count: c_int = 0;
            eval_cuda("cuDeviceGetCount", cuDeviceGetCount(&mut count))?;
            if count < 1 {
                return Err(anyhow::anyhow!("no CUDA devices present"));
            }

            let mut device: CUdevice = 0;
            eval_cuda("cuDeviceGet", cuDeviceGet(&mut device, 0))?;

            let mut cap_major: c_int = 0;
            let mut cap_minor: c_int = 0;
            eval_cuda(
                "cuDeviceGetAttribute",
                cuDeviceGetAttribute(
                    &mut cap_major,
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    device,
                ),
            )?;
            eval_cuda(
                "cuDeviceGetAttribute",
                cuDeviceGetAttribute(
                    &mut cap_minor,
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    device,
                ),
            )?;

            let mut total_vram: usize = 0;
            eval_cuda("cuDeviceTotalMem", cuDeviceTotalMem_v2(&mut total_vram, device))?;

            let mut primary_ctx: CUcontext = std::ptr::null_mut();
            eval_cuda(
                "cuDevicePrimaryCtxRetain",
                cuDevicePrimaryCtxRetain(&mut primary_ctx, device),
            )?;
            eval_cuda("cuCtxSetCurrent", cuCtxSetCurrent(primary_ctx))?;

            let mut stream: CUstream = std::ptr::null_mut();
            eval_cuda(
                "cuStreamCreate",
                cuStreamCreate(&mut stream, CU_STREAM_NON_BLOCKING),
            )?;

            Ok(Self {
                device,
                ctx: primary_ctx,
                cap_major,
                cap_minor,
                total_vram,
                stream,
            })
        }
    }

    /// Format the probed capability as "sm_<major><minor>", matching the
    /// manifest's `target_arch` convention.
    pub fn compute_capability_string(&self) -> String {
        format!("sm_{}{}", self.cap_major, self.cap_minor)
    }

    pub fn total_vram_bytes(&self) -> u64 {
        self.total_vram as u64
    }

    /// Conservative max-stream budget. The driver has no single getter; 64
    /// leaves headroom for graph-internal streams while bounding the
    /// scheduler's prefill/decode parallelism sensibly.
    pub fn max_streams(&self) -> u32 {
        64
    }

    /// Borrow the capture stream. Used by the graph + kernel layers; the
    /// borrow never outlives the device.
    pub fn stream(&self) -> CUstream {
        self.stream
    }

    /// Make the context current on this thread. Graph capture + kernel
    /// launches require it; called by graph/kernels before driver work.
    pub fn set_current(&self) -> Result<()> {
        unsafe {
            eval_cuda("cuCtxSetCurrent", cuCtxSetCurrent(self.ctx))?;
        }
        Ok(())
    }

    /// Allocate `bytes` of device memory, returning the raw device pointer.
    pub fn alloc(&mut self, bytes: usize) -> Result<u64> {
        let mut ptr: CUdeviceptr = 0;
        unsafe {
            eval_cuda("cuMemAlloc", cuMemAlloc_v2(&mut ptr, bytes))?;
        }
        Ok(ptr)
    }

    /// Free a device pointer previously returned by [`Self::alloc`].
    pub fn free(&mut self, ptr: u64) -> Result<()> {
        unsafe {
            eval_cuda("cuMemFree", cuMemFree_v2(ptr))?;
        }
        Ok(())
    }

    /// Release the primary context and destroy the stream. Called on shutdown.
    pub fn teardown(self) -> Result<()> {
        unsafe {
            eval_cuda("cuStreamDestroy", cuStreamDestroy_v2(self.stream))?;
            // `cuDevicePrimaryCtxRelease` has no v2 in the driver; the v1
            // signature is the current one.
            eval_cuda(
                "cuDevicePrimaryCtxRelease",
                cuda_driver_sys::cuDevicePrimaryCtxRelease(self.device),
            )?;
            // `self.ctx` is retained by the primary-ctx refcount; released above.
            let _ = self.ctx;
        }
        Ok(())
    }

    /// Same cleanup as [`teardown`], but operating on `&mut self` — for the
    /// backend path where the device lives behind a `Mutex` (and so can't
    /// move out). The device handle is left in a released state; callers
    /// must not use the device after this.
    pub fn teardown_ref(&mut self) -> Result<()> {
        unsafe {
            eval_cuda("cuStreamDestroy", cuStreamDestroy_v2(self.stream))?;
            eval_cuda(
                "cuDevicePrimaryCtxRelease",
                cuda_driver_sys::cuDevicePrimaryCtxRelease(self.device),
            )?;
            self.stream = std::ptr::null_mut();
            self.ctx = std::ptr::null_mut();
        }
        Ok(())
    }
}
