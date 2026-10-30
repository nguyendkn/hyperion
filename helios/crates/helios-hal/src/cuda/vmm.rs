//! Virtual memory management of the driver (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`): one reserved range of device addresses
//! that is backed by physical memory in granules, for the KV pool of ADR 0002. Real driver calls through the run-time loaded table.
//!
//! The order of use is the ADR's: reserve the range, (capture graphs against its base address), measure the free memory, map everything
//! once, then serve. Reserving and mapping are "new GPU work": a device that is serving refuses both with the typed error of
//! `work_guard`, so nothing is mapped on demand.

use std::os::raw::c_ulonglong;

use anyhow::{anyhow, Result};

use super::device::CudaDevice;
use super::ffi::eval_cuda;
use super::sys::*;

fn device_prop(ordinal: i32) -> CUmemAllocationProp {
    CUmemAllocationProp {
        kind: CU_MEM_ALLOCATION_TYPE_PINNED,
        requested_handle_types: 0,
        location: CUmemLocation { kind: CU_MEM_LOCATION_TYPE_DEVICE, id: ordinal },
        win32_handle_meta_data: std::ptr::null_mut(),
        alloc_flags: [0; 8],
    }
}

/// `(minimum, recommended)` allocation granularity of the device's memory, in bytes.
pub fn granularity(dev: &CudaDevice) -> Result<(usize, usize)> {
    dev.set_current()?;
    let prop = device_prop(dev.ordinal());
    let (mut minimum, mut recommended) = (0usize, 0usize);
    unsafe {
        eval_cuda("cuMemGetAllocationGranularity", cuMemGetAllocationGranularity(&mut minimum, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM))?;
        eval_cuda("cuMemGetAllocationGranularity", cuMemGetAllocationGranularity(&mut recommended, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED))?;
    }
    Ok((minimum, recommended))
}

/// One reserved virtual range and the physical granules mapped into its start so far.
pub struct VmmRange {
    base: u64,
    reserved: usize,
    granule: usize,
    ordinal: i32,
    /// Physical allocations, each mapped at `base + sum of the earlier sizes`.
    handles: Vec<(c_ulonglong, usize)>,
    mapped: usize,
}

impl VmmRange {
    /// Reserve `bytes` (rounded up to a granule) of address space. No physical memory is used.
    pub fn reserve(dev: &CudaDevice, bytes: usize) -> Result<Self> {
        dev.check_new_gpu_work("reserving the KV address range")?;
        let (_, granule) = granularity(dev)?;
        let reserved = bytes.div_ceil(granule) * granule;
        let mut base = 0u64;
        unsafe { eval_cuda("cuMemAddressReserve", cuMemAddressReserve(&mut base, reserved, granule, 0, 0))? };
        Ok(Self { base, reserved, granule, ordinal: dev.ordinal(), handles: Vec::new(), mapped: 0 })
    }

    pub fn base(&self) -> u64 {
        self.base
    }

    pub fn reserved_bytes(&self) -> usize {
        self.reserved
    }

    pub fn mapped_bytes(&self) -> usize {
        self.mapped
    }

    pub fn granule(&self) -> usize {
        self.granule
    }

    /// Back the next `bytes` (a multiple of the granule) of the range with physical memory and make it readable and writable by this
    /// device. A typed error (never a panic) when the device has no memory left or the range is exhausted.
    pub fn map(&mut self, dev: &CudaDevice, bytes: usize) -> Result<()> {
        dev.check_new_gpu_work("mapping KV memory")?;
        if bytes == 0 || bytes % self.granule != 0 {
            return Err(anyhow!("a mapping must be a positive multiple of the {}-byte granule, got {bytes}", self.granule));
        }
        if self.mapped + bytes > self.reserved {
            return Err(anyhow!("the reserved range holds {} bytes, {} are mapped and {bytes} more were asked for", self.reserved, self.mapped));
        }
        dev.set_current()?;
        let prop = device_prop(self.ordinal);
        let mut handle: c_ulonglong = 0;
        unsafe {
            eval_cuda("cuMemCreate", cuMemCreate(&mut handle, bytes, &prop, 0))?;
            let at = self.base + self.mapped as u64;
            if let Err(e) = eval_cuda("cuMemMap", cuMemMap(at, bytes, 0, handle, 0)) {
                cuMemRelease(handle);
                return Err(e);
            }
            let access = CUmemAccessDesc { location: prop.location, flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE };
            if let Err(e) = eval_cuda("cuMemSetAccess", cuMemSetAccess(at, bytes, &access, 1)) {
                cuMemUnmap(at, bytes);
                cuMemRelease(handle);
                return Err(e);
            }
        }
        self.handles.push((handle, bytes));
        self.mapped += bytes;
        Ok(())
    }
}

impl Drop for VmmRange {
    fn drop(&mut self) {
        // Unmap in the order mapped, release the physical memory, then the addresses. Errors cannot be reported from a destructor.
        let mut at = self.base;
        unsafe {
            for (handle, bytes) in self.handles.drain(..) {
                cuMemUnmap(at, bytes);
                cuMemRelease(handle);
                at += bytes as u64;
            }
            cuMemAddressFree(self.base, self.reserved);
        }
    }
}
