//! CUDA Driver surface shared by every NVIDIA backend — device probing, memory
//! allocation, host/device copies, stream handle. All calls go through `super::sys`, which loads
//! the real driver at run time. There is **no mock here**: on a host without the driver the
//! probe fails with a typed `LoadError` (the build itself needs no CUDA), and the backend
//! factory falls back to the next backend.
//!
//! We use the **driver** API (not runtime) because CUDA Graph capture lives
//! in the driver. Going straight to the driver matches the zero-overhead
//! posture and avoids the runtime's global state.

use anyhow::Result;

use super::sys::{
    cuCtxSetCurrent, cuCtxSynchronize, cuDeviceGet, cuMemGetInfo_v2, cuDeviceGetAttribute, cuDeviceGetCount, cuDevicePrimaryCtxRelease_v2,
    cuDevicePrimaryCtxRetain, cuDeviceTotalMem_v2, cuInit, cuMemAlloc_v2, cuMemFree_v2, cuMemcpyDtoDAsync_v2,
    cuMemcpyDtoH_v2, cuMemcpyHtoD_v2, cuStreamCreate, cuStreamDestroy_v2, load_driver, CUcontext, CUdevice, CUdeviceptr,
    CUstream, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuEventCreate, cuEventDestroy_v2, cuEventRecord, cuEventSynchronize, cuMemFreeHost, cuMemHostAlloc,
    cuMemcpyDtoHAsync_v2, cuMemcpyHtoDAsync_v2, cuMemsetD8Async, CUevent,
};

const CU_EVENT_DISABLE_TIMING: std::os::raw::c_uint = 2;
use std::os::raw::{c_int, c_void};

use super::ffi::eval_cuda;
use super::staging::{Ring, StagingError};
use super::work_guard::{RestrictionGuard, WorkRestriction};

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
    sm_count: u32,
    stream: CUstream,
    restriction: WorkRestriction,
    /// The pinned staging ring of the step path (see `staging.rs`), allocated by [`CudaDevice::enable_staging`] before Ready.
    staging: std::cell::RefCell<Option<Staging>>,
}

/// Pinned host memory, the events behind each slot, and the ring rules.
struct Staging {
    ring: Ring,
    host: *mut u8,
    events: Vec<CUevent>,
}

// Safety: the device handle and context are owned by this struct and only
// touched through `&mut` methods behind a mutex in the backend; the driver
// is thread-safe w.r.t. distinct contexts. `Send` is the only bound needed.
unsafe impl Send for CudaDevice {}

impl CudaDevice {
    /// Probe the first available device, create a primary context, read its
    /// properties, and stand up a non-blocking capture stream.
    pub fn probe() -> Result<Self> {
        Self::probe_ordinal(0)
    }

    /// [`CudaDevice::probe`] for the device with this ordinal (a rank of a multi-GPU node binds its own).
    pub fn probe_ordinal(ordinal: i32) -> Result<Self> {
        // The driver library is loaded here, at the first probe: a host without one gets a typed
        // `LoadError` (downcast the error to tell it from a CUDA failure), never a link failure.
        load_driver()?;
        unsafe {
            eval_cuda("cuInit", cuInit(0))?;

            let mut count: c_int = 0;
            eval_cuda("cuDeviceGetCount", cuDeviceGetCount(&mut count))?;
            if ordinal < 0 || count <= ordinal {
                return Err(anyhow::anyhow!("CUDA device {ordinal} is not present ({count} device(s))"));
            }

            let mut device: CUdevice = 0;
            eval_cuda("cuDeviceGet", cuDeviceGet(&mut device, ordinal))?;

            let mut cap_major: c_int = 0;
            let mut cap_minor: c_int = 0;
            eval_cuda(
                "cuDeviceGetAttribute",
                cuDeviceGetAttribute(
                    &mut cap_major,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    device,
                ),
            )?;
            eval_cuda(
                "cuDeviceGetAttribute",
                cuDeviceGetAttribute(
                    &mut cap_minor,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    device,
                ),
            )?;

            let mut sm_count: c_int = 0;
            eval_cuda(
                "cuDeviceGetAttribute",
                cuDeviceGetAttribute(&mut sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
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
                sm_count: sm_count.max(1) as u32,
                stream,
                restriction: WorkRestriction::default(),
                staging: std::cell::RefCell::new(None),
            })
        }
    }

    /// Format the probed capability as "sm_<major><minor>", matching the
    /// manifest's `target_arch` convention.
    /// The driver's handle of this device (its ordinal), for calls that name a device.
    pub fn ordinal(&self) -> i32 {
        self.device
    }

    pub fn compute_capability_string(&self) -> String {
        format!("sm_{}{}", self.cap_major, self.cap_minor)
    }

    /// Free and total device memory in bytes right now, as the driver reports it for this context
    /// (`cuMemGetInfo`). Unlike [`Self::total_vram_bytes`] this reflects what the weights and the
    /// driver already use, so it is the number a KV cache can be sized from after the weights load.
    pub fn memory_info(&self) -> Result<(u64, u64)> {
        self.set_current()?;
        let (mut free, mut total) = (0usize, 0usize);
        unsafe {
            eval_cuda("cuMemGetInfo", cuMemGetInfo_v2(&mut free, &mut total))?;
        }
        Ok((free as u64, total as u64))
    }

    /// Number of streaming multiprocessors, used to size grids that should fill the GPU.
    pub fn multiprocessor_count(&self) -> u32 {
        self.sm_count
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
        self.check_new_gpu_work("device memory allocation")?;
        let mut ptr: CUdeviceptr = 0;
        unsafe {
            eval_cuda("cuMemAlloc", cuMemAlloc_v2(&mut ptr, bytes))?;
        }
        Ok(ptr)
    }

    /// Forbid device allocation, kernel module loading, graph capture and cuBLAS handle creation until the
    /// guard is dropped: the serving path runs under it, so anything it would have to build at request
    /// time is a typed error instead (see `work_guard`).
    pub fn forbid_new_gpu_work(&self) -> RestrictionGuard {
        self.restriction.forbid()
    }

    /// `Ok` unless new GPU work is forbidden; `what` names the attempted work.
    pub fn check_new_gpu_work(&self, what: &'static str) -> Result<()> {
        Ok(self.restriction.check(what)?)
    }

    /// Block until all work queued on the context has finished.
    pub fn synchronize(&self) -> Result<()> {
        self.set_current()?;
        unsafe {
            eval_cuda("cuCtxSynchronize", cuCtxSynchronize())?;
        }
        Ok(())
    }

    /// Allocate the pinned staging ring of the step path: `slots` slots of `slot_bytes` bytes and one event per slot. Allocates, so it is part of the
    /// preallocation before Ready (refused while new GPU work is forbidden). A second call is a no-op.
    pub fn enable_staging(&self, slots: usize, slot_bytes: usize) -> Result<()> {
        if self.staging.borrow().is_some() {
            return Ok(());
        }
        self.check_new_gpu_work("pinned staging allocation")?;
        self.set_current()?;
        let ring = Ring::new(slots, slot_bytes);
        let mut host: *mut c_void = std::ptr::null_mut();
        unsafe {
            eval_cuda("cuMemHostAlloc", cuMemHostAlloc(&mut host, ring.total_bytes(), 0))?;
        }
        let mut events = Vec::with_capacity(slots);
        for _ in 0..slots {
            let mut event: CUevent = std::ptr::null_mut();
            unsafe {
                eval_cuda("cuEventCreate", cuEventCreate(&mut event, CU_EVENT_DISABLE_TIMING))?;
            }
            events.push(event);
        }
        *self.staging.borrow_mut() = Some(Staging { ring, host: host as *mut u8, events });
        Ok(())
    }

    /// Bytes of one staging slot (the largest single transfer of the step path), or `None` before [`CudaDevice::enable_staging`].
    pub fn staging_slot_bytes(&self) -> Option<usize> {
        self.staging.borrow().as_ref().map(|s| s.ring.slot_bytes())
    }

    /// Copy host bytes to a device allocation, **ordered on the step stream**: the bytes go through a pinned slot and `cuMemcpyHtoDAsync`, so a kernel
    /// launched afterwards on this stream sees them. The step path uses this and never [`CudaDevice::htod`].
    pub fn htod_staged(&self, dst: u64, src: &[u8]) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        self.set_current()?;
        let mut guard = self.staging.borrow_mut();
        let staging = guard.as_mut().ok_or(StagingError::NotEnabled)?;
        let acquired = staging.ring.acquire(src.len())?;
        if acquired.wait {
            unsafe { eval_cuda("cuEventSynchronize", cuEventSynchronize(staging.events[acquired.slot]))? };
            staging.ring.completed(acquired.slot);
        }
        let host = unsafe { staging.host.add(acquired.slot * staging.ring.slot_bytes()) };
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), host, src.len());
            eval_cuda("cuMemcpyHtoDAsync", cuMemcpyHtoDAsync_v2(dst, host as *const c_void, src.len(), self.stream))?;
            eval_cuda("cuEventRecord", cuEventRecord(staging.events[acquired.slot], self.stream))?;
        }
        staging.ring.recorded(acquired.slot);
        Ok(())
    }

    /// Copy a device allocation to host memory through a pinned slot, **after everything queued on the step stream**, and wait for it.
    pub fn dtoh_staged(&self, dst: &mut [u8], src: u64) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        self.set_current()?;
        let mut guard = self.staging.borrow_mut();
        let staging = guard.as_mut().ok_or(StagingError::NotEnabled)?;
        let acquired = staging.ring.acquire(dst.len())?;
        if acquired.wait {
            unsafe { eval_cuda("cuEventSynchronize", cuEventSynchronize(staging.events[acquired.slot]))? };
            staging.ring.completed(acquired.slot);
        }
        let host = unsafe { staging.host.add(acquired.slot * staging.ring.slot_bytes()) };
        unsafe {
            eval_cuda("cuMemcpyDtoHAsync", cuMemcpyDtoHAsync_v2(host as *mut c_void, src, dst.len(), self.stream))?;
            eval_cuda("cuEventRecord", cuEventRecord(staging.events[acquired.slot], self.stream))?;
            staging.ring.recorded(acquired.slot);
            eval_cuda("cuEventSynchronize", cuEventSynchronize(staging.events[acquired.slot]))?;
            staging.ring.completed(acquired.slot);
            std::ptr::copy_nonoverlapping(host as *const u8, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    /// Fill `bytes` bytes at `dst` with `value`, ordered on the step stream (no host memory involved).
    pub fn memset_staged(&self, dst: u64, value: u8, bytes: usize) -> Result<()> {
        self.set_current()?;
        unsafe { eval_cuda("cuMemsetD8Async", cuMemsetD8Async(dst, value, bytes, self.stream)) }
    }

    /// Copy host bytes to a device allocation. **Not ordered with the step stream** (a copy from pageable memory may still be in flight when a kernel
    /// starts): for load time and tests that follow it with a synchronize; the step path uses [`CudaDevice::htod_staged`].
    pub fn htod(&self, dst: u64, src: &[u8]) -> Result<()> {
        self.set_current()?;
        unsafe {
            eval_cuda("cuMemcpyHtoD", cuMemcpyHtoD_v2(dst, src.as_ptr() as *const c_void, src.len()))?;
        }
        Ok(())
    }

    /// Copy `bytes` between device allocations on this device's stream, so it is ordered with the
    /// kernels and cuBLAS calls queued on the same stream (the plain copy is not).
    pub fn dtod_async(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        self.set_current()?;
        unsafe {
            eval_cuda("cuMemcpyDtoDAsync", cuMemcpyDtoDAsync_v2(dst, src, bytes, self.stream))?;
        }
        Ok(())
    }

    /// Copy a device allocation back to host memory.
    pub fn dtoh(&self, dst: &mut [u8], src: u64) -> Result<()> {
        self.set_current()?;
        unsafe {
            eval_cuda("cuMemcpyDtoH", cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut c_void, src, dst.len()))?;
        }
        Ok(())
    }

    /// Free a device pointer previously returned by [`Self::alloc`].
    pub fn free(&mut self, ptr: u64) -> Result<()> {
        unsafe {
            eval_cuda("cuMemFree", cuMemFree_v2(ptr))?;
        }
        Ok(())
    }

    /// Release the primary context and destroy the stream. Called on shutdown.
    pub fn teardown(mut self) -> Result<()> {
        self.teardown_ref()
    }

    /// Same cleanup as [`teardown`], but operating on `&mut self` — for the
    /// backend path where the device lives behind a `Mutex` (and so can't
    /// move out). The device handle is left in a released state; callers
    /// must not use the device after this.
    pub fn teardown_ref(&mut self) -> Result<()> {
        // Idempotent: a second call (explicit `shutdown` then `Drop`) is a no-op.
        if self.ctx.is_null() {
            return Ok(());
        }
        if let Some(staging) = self.staging.get_mut().take() {
            unsafe {
                for event in staging.events {
                    cuEventDestroy_v2(event);
                }
                cuMemFreeHost(staging.host as *mut c_void);
            }
        }
        unsafe {
            eval_cuda("cuStreamDestroy", cuStreamDestroy_v2(self.stream))?;
            eval_cuda(
                "cuDevicePrimaryCtxRelease",
                cuDevicePrimaryCtxRelease_v2(self.device),
            )?;
            self.stream = std::ptr::null_mut();
            self.ctx = std::ptr::null_mut();
        }
        Ok(())
    }
}

impl Drop for CudaDevice {
    /// A device dropped without an explicit `shutdown` still releases its primary
    /// context. Without this a backend that probed a device and was then refused
    /// kept ~0.5 GiB of GPU memory until the process exited (seen on an H100).
    fn drop(&mut self) {
        let _ = self.teardown_ref();
    }
}
