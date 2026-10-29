//! The CUDA Driver API, loaded at run time from `libcuda.so.1` (Linux) or `nvcuda.dll` (Windows).
//! Nothing here links against the CUDA toolkit or the driver: a build needs neither, and a
//! machine without a driver gets [`LoadError`] from [`load_driver`] (the backends call it when
//! they probe the device) instead of a link or start-up failure.
//!
//! Functions are looked up under their explicit versioned names (`cuMemAlloc_v2`,
//! `cuStreamBeginCapture_v2`, ...), the names the driver exports for the current ABI, so a header
//! macro that renames a function can never silently pick a different signature. The wrappers
//! below keep the C names and return the raw `CUresult`; [`super::eval_cuda`] turns it into a
//! `Result`. Called before the driver is loaded they return `CUDA_ERROR_NOT_INITIALIZED`.

use std::ffi::{c_char, c_int, c_uint, c_ulonglong, c_void};
use std::sync::OnceLock;

use super::loader::{Library, LoadError};

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;

pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;

#[cfg(windows)]
const DRIVER_LIBRARIES: &[&str] = &["nvcuda.dll"];
#[cfg(not(windows))]
const DRIVER_LIBRARIES: &[&str] = &["libcuda.so.1", "libcuda.so"];

macro_rules! driver_api {
    ($( $name:ident ( $($arg:ident : $ty:ty),* ) ; )*) => {
        /// The function pointers of every Driver API call the engine makes.
        #[allow(non_snake_case)]
        struct Driver {
            $( $name: unsafe extern "C" fn($($ty),*) -> CUresult, )*
        }

        #[allow(non_snake_case)]
        impl Driver {
            fn load(library: &Library) -> Result<Self, LoadError> {
                // Safety: each field's type is the exact C signature of the function it is named after.
                Ok(Self { $( $name: unsafe { library.symbol(stringify!($name))? }, )* })
            }
        }

        $(
            #[allow(non_snake_case, clippy::missing_safety_doc)]
            pub unsafe fn $name($($arg: $ty),*) -> CUresult {
                match DRIVER.get() {
                    Some(Ok(driver)) => (driver.$name)($($arg),*),
                    _ => CUDA_ERROR_NOT_INITIALIZED,
                }
            }
        )*
    };
}

static DRIVER: OnceLock<Result<Driver, LoadError>> = OnceLock::new();

/// Load the driver library and resolve every function (once per process). The error says which
/// library names were tried, or which function the driver lacks.
pub fn load_driver() -> Result<(), LoadError> {
    let loaded = DRIVER.get_or_init(|| {
        let library = Library::open(DRIVER_LIBRARIES)?;
        let driver = Driver::load(&library)?;
        let mut version: c_int = 0;
        // Safety: the signature is that of the C function; the pointer is a local.
        let status = unsafe { (driver.cuDriverGetVersion)(&mut version) };
        tracing::info!(library = library.name(), cuda_driver_version = if status == CUDA_SUCCESS { version } else { -1 }, "CUDA driver library loaded");
        Ok(driver)
    });
    match loaded {
        Ok(_) => Ok(()),
        Err(error) => Err(error.clone()),
    }
}

driver_api! {
    cuInit(flags: c_uint);
    cuDriverGetVersion(version: *mut c_int);
    cuGetErrorString(error: CUresult, text: *mut *const c_char);
    cuDeviceGetCount(count: *mut c_int);
    cuDeviceGet(device: *mut CUdevice, ordinal: c_int);
    cuDeviceGetAttribute(value: *mut c_int, attribute: c_int, device: CUdevice);
    cuDeviceTotalMem_v2(bytes: *mut usize, device: CUdevice);
    cuDevicePrimaryCtxRetain(context: *mut CUcontext, device: CUdevice);
    cuDevicePrimaryCtxRelease_v2(device: CUdevice);
    cuCtxSetCurrent(context: CUcontext);
    cuCtxSynchronize();
    cuMemAlloc_v2(pointer: *mut CUdeviceptr, bytes: usize);
    cuMemFree_v2(pointer: CUdeviceptr);
    cuMemGetInfo_v2(free: *mut usize, total: *mut usize);
    cuMemcpyHtoD_v2(destination: CUdeviceptr, source: *const c_void, bytes: usize);
    cuMemcpyDtoH_v2(destination: *mut c_void, source: CUdeviceptr, bytes: usize);
    cuMemcpyDtoDAsync_v2(destination: CUdeviceptr, source: CUdeviceptr, bytes: usize, stream: CUstream);
    cuStreamCreate(stream: *mut CUstream, flags: c_uint);
    cuStreamDestroy_v2(stream: CUstream);
    cuStreamSynchronize(stream: CUstream);
    cuStreamBeginCapture_v2(stream: CUstream, mode: c_int);
    cuStreamEndCapture(stream: CUstream, graph: *mut CUgraph);
    cuModuleLoadData(module: *mut CUmodule, image: *const c_void);
    cuModuleGetFunction(function: *mut CUfunction, module: CUmodule, name: *const c_char);
    cuModuleUnload(module: CUmodule);
    cuLaunchKernel(function: CUfunction, grid_x: c_uint, grid_y: c_uint, grid_z: c_uint, block_x: c_uint, block_y: c_uint, block_z: c_uint, shared_bytes: c_uint, stream: CUstream, params: *mut *mut c_void, extra: *mut *mut c_void);
    cuEventCreate(event: *mut CUevent, flags: c_uint);
    cuEventDestroy_v2(event: CUevent);
    cuEventRecord(event: CUevent, stream: CUstream);
    cuEventSynchronize(event: CUevent);
    cuEventElapsedTime(milliseconds: *mut f32, start: CUevent, end: CUevent);
    cuGraphInstantiateWithFlags(exec: *mut CUgraphExec, graph: CUgraph, flags: c_ulonglong);
    cuGraphLaunch(exec: CUgraphExec, stream: CUstream);
    cuGraphExecDestroy(exec: CUgraphExec);
    cuGraphDestroy(graph: CUgraph);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_a_driver_the_error_is_typed_and_calls_report_not_initialized() {
        match load_driver() {
            Ok(()) => {
                // A machine with an NVIDIA driver: the table is real, so a harmless call works.
                let mut count: c_int = -1;
                let status = unsafe { cuInit(0) };
                if status == CUDA_SUCCESS {
                    assert_eq!(unsafe { cuDeviceGetCount(&mut count) }, CUDA_SUCCESS);
                    assert!(count >= 0);
                }
            }
            Err(LoadError::LibraryMissing { tried, .. }) => {
                assert!(!tried.is_empty());
                assert_eq!(unsafe { cuInit(0) }, CUDA_ERROR_NOT_INITIALIZED, "a call before a successful load must not crash");
            }
            Err(other) => panic!("a driver that loads must export every function the engine uses: {other}"),
        }
    }
}
