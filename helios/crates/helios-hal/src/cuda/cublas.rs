//! cuBLAS for the matrix multiplies (ADR 0001: cuBLAS first, custom kernels later).
//! A minimal hand-written binding of the four entry points we need, loaded at run time
//! like the driver (`libcublas.so.13`, then `.so.12`; `cublas64_13.dll`, then `cublas64_12.dll`): there is no mock, and a host
//! without cuBLAS gets a typed `LoadError` when the handle is created.
//!
//! Layout note: cuBLAS is column-major, our tensors are row-major. A row-major
//! `[rows, cols]` matrix is the column-major `[cols, rows]` transpose, so the
//! linear layer `y[T,out] = x[T,in] * W[out,in]^T` is computed as the
//! column-major product `y^T[out,T] = op(W)[out,in] * x^T[in,T]` with `op(W) = T`.

use std::os::raw::{c_int, c_void};
use std::sync::OnceLock;

use anyhow::{anyhow, Result};

use super::cublaslt::CublasLt;
use super::device::CudaDevice;
use super::loader::{Library, LoadError};

type Handle = *mut c_void;

const STATUS_SUCCESS: c_int = 0;
const OP_N: c_int = 0;
const OP_T: c_int = 1;
const CUDA_R_32F: c_int = 0;
const CUDA_R_16BF: c_int = 14;
const COMPUTE_32F: c_int = 68;
const GEMM_DEFAULT: c_int = -1;

#[cfg(windows)]
const CUBLAS_LIBRARIES: &[&str] = &["cublas64_13.dll", "cublas64_12.dll"];
#[cfg(not(windows))]
const CUBLAS_LIBRARIES: &[&str] = &["libcublas.so.13", "libcublas.so.12", "libcublas.so"];

type GemmEx = unsafe extern "C" fn(
    Handle,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const c_void,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
) -> c_int;

/// The cuBLAS entry points we use, resolved once under their explicit names.
struct Api {
    create: unsafe extern "C" fn(*mut Handle) -> c_int,
    destroy: unsafe extern "C" fn(Handle) -> c_int,
    set_stream: unsafe extern "C" fn(Handle, *mut c_void) -> c_int,
    gemm_ex: GemmEx,
    /// `cublasSetWorkspace_v2`, optional: a library without it just keeps its own workspace policy.
    set_workspace: Option<unsafe extern "C" fn(Handle, *mut c_void, usize) -> c_int>,
}

static API: OnceLock<std::result::Result<Api, LoadError>> = OnceLock::new();
static LIBRARY_NAME: OnceLock<String> = OnceLock::new();

/// The cuBLAS library that was loaded, once it has been.
pub fn loaded_library() -> Option<String> {
    LIBRARY_NAME.get().cloned()
}

fn api() -> std::result::Result<&'static Api, LoadError> {
    let loaded = API.get_or_init(|| {
        // The newest major version first: a CUDA 13 toolkit ships `libcublas.so.13`, a 12 one `.so.12`.
        let library = Library::open(CUBLAS_LIBRARIES)?;
        let _ = LIBRARY_NAME.set(library.name().to_string());
        tracing::info!(library = library.name(), "cuBLAS library loaded");
        // Safety: each type is the exact C signature of the function named.
        unsafe {
            Ok(Api {
                create: library.symbol("cublasCreate_v2")?,
                destroy: library.symbol("cublasDestroy_v2")?,
                set_stream: library.symbol("cublasSetStream_v2")?,
                gemm_ex: library.symbol("cublasGemmEx")?,
                set_workspace: library.symbol("cublasSetWorkspace_v2").ok(),
            })
        }
    });
    loaded.as_ref().map_err(Clone::clone)
}

fn check(call: &str, status: c_int) -> Result<()> {
    if status == STATUS_SUCCESS {
        Ok(())
    } else {
        Err(anyhow!("cuBLAS {call} failed with status {status}"))
    }
}

/// A cuBLAS handle bound to the device's stream. It must be dropped before the
/// device releases its context (declare it before the device in a struct).
pub struct Cublas {
    handle: Handle,
    api: &'static Api,
    /// cuBLASLt, for projections that add their bias in the epilogue; absent when its library cannot be loaded.
    lt: Option<CublasLt>,
}

// Safety: the handle is only used behind the backend's mutex.
unsafe impl Send for Cublas {}

impl Cublas {
    pub fn new(dev: &CudaDevice) -> Result<Self> {
        dev.check_new_gpu_work("cuBLAS handle creation")?;
        // A host without cuBLAS gets a typed `LoadError` here (downcast to tell it from a cuBLAS failure).
        let api = api()?;
        dev.set_current()?;
        let mut handle: Handle = std::ptr::null_mut();
        unsafe {
            check("cublasCreate", (api.create)(&mut handle))?;
            if let Err(e) = check("cublasSetStream", (api.set_stream)(handle, dev.stream() as *mut c_void)) {
                (api.destroy)(handle);
                return Err(e);
            }
        }
        // cuBLASLt is an optimisation: without it the decode step keeps the plain GEMM and the bias/cast kernel.
        let lt = match CublasLt::new(dev) {
            Ok(lt) => Some(lt),
            Err(error) => {
                tracing::warn!(%error, "cuBLASLt is not available: decode keeps the plain cuBLAS path");
                None
            }
        };
        Ok(Self { handle, api, lt })
    }

    /// The cuBLASLt handle, when its library was found.
    pub fn lt(&self) -> Option<&CublasLt> {
        self.lt.as_ref()
    }

    /// Give the handle a fixed, preallocated workspace of `bytes` at device address `ptr`, so no cuBLAS
    /// call allocates (which is not allowed while a graph is being captured, nor after the engine is ready).
    /// An error if the cuBLAS library has no `cublasSetWorkspace_v2`.
    pub fn set_workspace(&self, ptr: u64, bytes: usize) -> Result<()> {
        let set = self.api.set_workspace.ok_or_else(|| anyhow!("this cuBLAS library has no cublasSetWorkspace_v2"))?;
        if let Some(lt) = &self.lt {
            lt.set_workspace(ptr, bytes);
        }
        unsafe { check("cublasSetWorkspace", set(self.handle, ptr as *mut c_void, bytes)) }
    }

    /// `y[tokens, out] = x[tokens, in] * W[out, in]^T` with bf16 `W` and `x` and f32
    /// `y` (f32 accumulation). All three are row-major device pointers.
    pub fn linear_bf16_f32(&self, w: u64, x: u64, y: u64, tokens: usize, out: usize, inn: usize) -> Result<()> {
        let dim = |d: usize| c_int::try_from(d).map_err(|_| anyhow!("dimension {d} too large for cuBLAS"));
        let (t, o, i) = (dim(tokens)?, dim(out)?, dim(inn)?);
        let (alpha, beta) = (1.0f32, 0.0f32);
        unsafe {
            check(
                "cublasGemmEx",
                (self.api.gemm_ex)(
                    self.handle,
                    OP_T,
                    OP_N,
                    o,
                    t,
                    i,
                    &alpha as *const f32 as *const c_void,
                    w as *const c_void,
                    CUDA_R_16BF,
                    i,
                    x as *const c_void,
                    CUDA_R_16BF,
                    i,
                    &beta as *const f32 as *const c_void,
                    y as *mut c_void,
                    CUDA_R_32F,
                    o,
                    COMPUTE_32F,
                    GEMM_DEFAULT,
                ),
            )
        }
    }
}

impl Drop for Cublas {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                (self.api.destroy)(self.handle);
            }
        }
    }
}
