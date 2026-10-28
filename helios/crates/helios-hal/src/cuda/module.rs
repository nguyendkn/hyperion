//! Load PTX and launch kernels through the Driver API. The PTX is generated from the
//! CUDA sources in `helios-hal/kernels/` (see `scripts/build-ptx.sh`) and embedded in
//! the binary; the driver JIT-compiles it for the installed GPU when the module is
//! loaded, so one file serves every generation.

use std::ffi::{c_void, CString};

use anyhow::{anyhow, Context, Result};
use cuda_driver_sys::{
    cuLaunchKernel, cuModuleGetFunction, cuModuleLoadData, cuModuleUnload, CUfunction, CUmodule,
};

use super::device::CudaDevice;
use super::ffi::eval_cuda;

/// A loaded PTX module. Unloaded on drop; drop it before the device releases its context
/// (declare it before the device in a struct).
pub struct KernelModule {
    module: CUmodule,
}

// Safety: the module handle is only used behind the backend's mutex.
unsafe impl Send for KernelModule {}

/// A kernel entry point inside a [`KernelModule`]. Valid while the module is alive.
#[derive(Clone, Copy)]
pub struct Kernel {
    function: CUfunction,
}

impl KernelModule {
    /// JIT-compile and load `ptx` on the device's context.
    pub fn load(dev: &CudaDevice, ptx: &str) -> Result<Self> {
        dev.set_current()?;
        let image = CString::new(ptx).context("PTX must not contain NUL bytes")?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            eval_cuda("cuModuleLoadData", cuModuleLoadData(&mut module, image.as_ptr() as *const c_void))?;
        }
        Ok(Self { module })
    }

    pub fn function(&self, name: &str) -> Result<Kernel> {
        let name_c = CString::new(name).map_err(|_| anyhow!("kernel name must not contain NUL"))?;
        let mut function: CUfunction = std::ptr::null_mut();
        unsafe {
            eval_cuda("cuModuleGetFunction", cuModuleGetFunction(&mut function, self.module, name_c.as_ptr()))?;
        }
        Ok(Kernel { function })
    }
}

impl Drop for KernelModule {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe {
                cuModuleUnload(self.module);
            }
        }
    }
}

impl Kernel {
    /// Launch on the device's stream. `params` are pointers to the kernel arguments, in
    /// order, each pointing at a value that outlives the call.
    ///
    /// # Safety
    /// `params` must match the kernel's signature exactly (count, types, sizes).
    pub unsafe fn launch(
        &self,
        dev: &CudaDevice,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        dev.set_current()?;
        eval_cuda(
            "cuLaunchKernel",
            cuLaunchKernel(
                self.function,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_bytes,
                dev.stream(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            ),
        )
    }
}
