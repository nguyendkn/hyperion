//! cuBLAS for the matrix multiplies (ADR 0001: cuBLAS first, custom kernels later).
//! A minimal hand-written binding of the four entry points we need: there is no
//! mock, so without `libcublas` a build that enables the `cuda` feature fails at
//! link time, like the driver itself.
//!
//! Layout note: cuBLAS is column-major, our tensors are row-major. A row-major
//! `[rows, cols]` matrix is the column-major `[cols, rows]` transpose, so the
//! linear layer `y[T,out] = x[T,in] * W[out,in]^T` is computed as the
//! column-major product `y^T[out,T] = op(W)[out,in] * x^T[in,T]` with `op(W) = T`.

use std::os::raw::{c_int, c_void};

use anyhow::{anyhow, Result};

use super::device::CudaDevice;

type Handle = *mut c_void;

const STATUS_SUCCESS: c_int = 0;
const OP_N: c_int = 0;
const OP_T: c_int = 1;
const CUDA_R_32F: c_int = 0;
const CUDA_R_16BF: c_int = 14;
const COMPUTE_32F: c_int = 68;
const GEMM_DEFAULT: c_int = -1;

#[link(name = "cublas")]
extern "C" {
    fn cublasCreate_v2(handle: *mut Handle) -> c_int;
    fn cublasDestroy_v2(handle: Handle) -> c_int;
    fn cublasSetStream_v2(handle: Handle, stream: *mut c_void) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn cublasGemmEx(
        handle: Handle,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        a_type: c_int,
        lda: c_int,
        b: *const c_void,
        b_type: c_int,
        ldb: c_int,
        beta: *const c_void,
        c: *mut c_void,
        c_type: c_int,
        ldc: c_int,
        compute_type: c_int,
        algo: c_int,
    ) -> c_int;
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
}

// Safety: the handle is only used behind the backend's mutex.
unsafe impl Send for Cublas {}

impl Cublas {
    pub fn new(dev: &CudaDevice) -> Result<Self> {
        dev.set_current()?;
        let mut handle: Handle = std::ptr::null_mut();
        unsafe {
            check("cublasCreate", cublasCreate_v2(&mut handle))?;
            if let Err(e) = check("cublasSetStream", cublasSetStream_v2(handle, dev.stream() as *mut c_void)) {
                cublasDestroy_v2(handle);
                return Err(e);
            }
        }
        Ok(Self { handle })
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
                cublasGemmEx(
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
                cublasDestroy_v2(self.handle);
            }
        }
    }
}
