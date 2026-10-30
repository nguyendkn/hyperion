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
pub const CU_MEM_ALLOCATION_TYPE_PINNED: c_int = 1;
pub const CU_MEM_LOCATION_TYPE_DEVICE: c_int = 1;
pub const CU_MEM_ALLOC_GRANULARITY_MINIMUM: c_int = 0;
pub const CU_MEM_ALLOC_GRANULARITY_RECOMMENDED: c_int = 1;
pub const CU_MEM_ACCESS_FLAGS_PROT_READWRITE: c_int = 3;

/// `CUmemLocation`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemLocation {
    pub kind: c_int,
    pub id: c_int,
}

/// `CUmemAllocationProp` (32 bytes): the type of allocation, its location and flags.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemAllocationProp {
    pub kind: c_int,
    pub requested_handle_types: c_int,
    pub location: CUmemLocation,
    pub win32_handle_meta_data: *mut c_void,
    /// `allocFlags`: compression type, GPUDirect RDMA capability, usage, 4 reserved bytes.
    pub alloc_flags: [u8; 8],
}

/// `CUmemAccessDesc`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemAccessDesc {
    pub location: CUmemLocation,
    pub flags: c_int,
}

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
/// The driver library that was loaded and its CUDA driver version (for logs and test output).
static DRIVER_INFO: OnceLock<(String, i32)> = OnceLock::new();

/// `(library name, CUDA driver version)` of the driver library, once it has been loaded.
pub fn loaded_driver() -> Option<(String, i32)> {
    DRIVER_INFO.get().cloned()
}

/// Load the driver library and resolve every function (once per process). The error says which
/// library names were tried, or which function the driver lacks.
pub fn load_driver() -> Result<(), LoadError> {
    let loaded = DRIVER.get_or_init(|| {
        let library = Library::open(DRIVER_LIBRARIES)?;
        let driver = Driver::load(&library)?;
        let mut version: c_int = 0;
        // Safety: the signature is that of the C function; the pointer is a local.
        let status = unsafe { (driver.cuDriverGetVersion)(&mut version) };
        let version = if status == CUDA_SUCCESS { version } else { -1 };
        let _ = DRIVER_INFO.set((library.name().to_string(), version));
        tracing::info!(library = library.name(), cuda_driver_version = version, "CUDA driver library loaded");
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
    cuMemGetAllocationGranularity(granularity: *mut usize, prop: *const CUmemAllocationProp, option: c_int);
    cuMemAddressReserve(pointer: *mut CUdeviceptr, size: usize, alignment: usize, address: CUdeviceptr, flags: c_ulonglong);
    cuMemAddressFree(pointer: CUdeviceptr, size: usize);
    cuMemCreate(handle: *mut c_ulonglong, size: usize, prop: *const CUmemAllocationProp, flags: c_ulonglong);
    cuMemRelease(handle: c_ulonglong);
    cuMemMap(pointer: CUdeviceptr, size: usize, offset: usize, handle: c_ulonglong, flags: c_ulonglong);
    cuMemUnmap(pointer: CUdeviceptr, size: usize);
    cuMemSetAccess(pointer: CUdeviceptr, size: usize, desc: *const CUmemAccessDesc, count: usize);
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
    cuGraphUpload(exec: CUgraphExec, stream: CUstream);
    cuGraphLaunch(exec: CUgraphExec, stream: CUstream);
    cuGraphExecDestroy(exec: CUgraphExec);
    cuGraphDestroy(graph: CUgraph);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_vmm_structs_have_the_sizes_the_driver_header_gives_them() {
        assert_eq!(std::mem::size_of::<CUmemLocation>(), 8);
        assert_eq!(std::mem::size_of::<CUmemAllocationProp>(), 32);
        assert_eq!(std::mem::size_of::<CUmemAccessDesc>(), 12);
    }

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
