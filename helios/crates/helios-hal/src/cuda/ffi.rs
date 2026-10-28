//! Shared FFI helpers for the SM120 backend. `eval_cuda` converts the CUDA
//! Driver's `CUresult` return convention (a status enum) into a Rust
//! `Result`, with the driver's own error string attached when available.
//!
//! Keeping this here (rather than inline at each call site) avoids an
//! `expect()`/`unwrap`-laden call layer and makes error messages uniform
//! across device / graph / kernel code.

use anyhow::{anyhow, Result};
use cuda_driver_sys::{cuGetErrorString, CUresult};

/// The success variant of the driver's `CUresult` enum. bindgen emits the
/// variant under the enum's alias `CUresult`, not at crate root, so we spell
/// it through the alias.
const CUDA_SUCCESS: CUresult = CUresult::CUDA_SUCCESS;

/// Convert a `CUresult` into `Result<()>`, attaching the driver's error
/// string when the result is not `CUDA_SUCCESS`. `where_` is a short label
/// for the call site (e.g. "cuInit") so the error reads like a stack trace.
///
/// `CUresult` is an enum in `cuda-driver-sys` (a bindgen `cudaError_enum`),
/// so comparing against `CUDA_SUCCESS` is a plain variant match.
pub fn eval_cuda(where_: &str, status: CUresult) -> Result<()> {
    if status == CUDA_SUCCESS {
        return Ok(());
    }
    let msg = unsafe {
        let mut raw: *const std::os::raw::c_char = std::ptr::null();
        let r = cuGetErrorString(status, &mut raw);
        if r == CUDA_SUCCESS && !raw.is_null() {
            std::ffi::CStr::from_ptr(raw)
                .to_str()
                .unwrap_or("<invalid utf-8>")
                .to_string()
        } else {
            format!("<cuGetErrorString failed: code={}>", status as i32)
        }
    };
    Err(anyhow!("CUDA {where_} failed: {msg} (code={})", status as i32))
}
