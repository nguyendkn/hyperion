//! cuBLASLt for the matrix multiplies of a decode step: bf16 output with the bias added in the epilogue, so a
//! projection is one launch (no f32 intermediate, no separate bias/cast kernel). Loaded at run time like the driver
//! and cuBLAS (`libcublasLt.so.13`, then `.so.12`; `cublasLt64_13.dll`, then `cublasLt64_12.dll`), with a typed
//! `LoadError` when the library is missing; the caller then keeps the plain cuBLAS path.
//!
//! Algorithm choice. The algorithm is the heuristic's first answer for the shape, always: it decides the order of the f32 sums, so it must
//! be reproducible without any stored state (no cache file, no timing at load). Its id is logged once per shape when the shape is planned,
//! which the model load does for every shape a generation uses. [`CublasLt::benchmark`] times the heuristic's candidates for a shape without
//! changing the choice; `scripts/bench-cublaslt.sh` runs it on a GPU host (a measurement showed no candidate beating the first on the decode
//! shapes of Qwen2.5-0.5B and 7B).
//!
//! The arithmetic is the one `Cublas::linear_bf16_f32` plus the `bias_cast` kernel perform: f32 accumulation, the
//! bias added in f32, one rounding to bf16. Same layout convention as `cublas.rs`: `y[T,out] = x[T,in] * W[out,in]^T`
//! is the column-major product `y^T = op(W) * x^T`.
//!
//! Each distinct shape builds its descriptors and picks an algorithm once (a host-only query) and keeps them; a
//! call only sets the bias pointer and launches. The model load warms every shape a generation uses, so serving
//! creates nothing.

use std::collections::HashMap;
use std::os::raw::{c_int, c_void};
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Result};

use super::device::CudaDevice;
use super::loader::{Library, LoadError};

type Handle = *mut c_void;

const STATUS_SUCCESS: c_int = 0;
const CUDA_R_32F: c_int = 0;
const CUDA_R_16BF: c_int = 14;
const COMPUTE_32F: c_int = 68;
const OP_N: i32 = 0;
const OP_T: i32 = 1;
// cublasLtMatmulDescAttributes_t
const DESC_TRANSA: c_int = 3;
const DESC_TRANSB: c_int = 4;
const DESC_EPILOGUE: c_int = 7;
const DESC_BIAS_POINTER: c_int = 8;
// cublasLtEpilogue_t
const EPILOGUE_DEFAULT: u32 = 1;
const EPILOGUE_BIAS: u32 = 4;
// cublasLtMatmulPreferenceAttributes_t
const PREF_MAX_WORKSPACE_BYTES: c_int = 1;
/// Algorithms the heuristic is asked for per shape (the first is used; all are timed by `benchmark`).
pub const CANDIDATES: usize = 8;
/// Launches of a candidate before it is timed.
const WARM_UPS: usize = 3;

#[cfg(windows)]
const LT_LIBRARIES: &[&str] = &["cublasLt64_13.dll", "cublasLt64_12.dll"];
#[cfg(not(windows))]
const LT_LIBRARIES: &[&str] = &["libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"];

/// `cublasLtMatmulHeuristicResult_t`: the algorithm (64 opaque bytes) and what it needs.
#[repr(C)]
#[derive(Clone, Copy)]
struct HeuristicResult {
    algo: [u64; 8],
    workspace_size: usize,
    state: c_int,
    waves_count: f32,
    reserved: [c_int; 4],
}

struct Api {
    create: unsafe extern "C" fn(*mut Handle) -> c_int,
    destroy: unsafe extern "C" fn(Handle) -> c_int,
    desc_create: unsafe extern "C" fn(*mut Handle, c_int, c_int) -> c_int,
    desc_destroy: unsafe extern "C" fn(Handle) -> c_int,
    desc_set: unsafe extern "C" fn(Handle, c_int, *const c_void, usize) -> c_int,
    layout_create: unsafe extern "C" fn(*mut Handle, c_int, u64, u64, i64) -> c_int,
    layout_destroy: unsafe extern "C" fn(Handle) -> c_int,
    pref_create: unsafe extern "C" fn(*mut Handle) -> c_int,
    pref_destroy: unsafe extern "C" fn(Handle) -> c_int,
    pref_set: unsafe extern "C" fn(Handle, c_int, *const c_void, usize) -> c_int,
    heuristic: unsafe extern "C" fn(Handle, Handle, Handle, Handle, Handle, Handle, Handle, c_int, *mut HeuristicResult, *mut c_int) -> c_int,
    version: Option<unsafe extern "C" fn() -> usize>,
    matmul: unsafe extern "C" fn(
        Handle,
        Handle,
        *const c_void,
        *const c_void,
        Handle,
        *const c_void,
        Handle,
        *const c_void,
        *const c_void,
        Handle,
        *mut c_void,
        Handle,
        *const HeuristicResult,
        *mut c_void,
        usize,
        *mut c_void,
    ) -> c_int,
}

static API: OnceLock<std::result::Result<Api, LoadError>> = OnceLock::new();
static LIBRARY_NAME: OnceLock<String> = OnceLock::new();

/// The cuBLASLt library that was loaded, once it has been.
pub fn loaded_library() -> Option<String> {
    LIBRARY_NAME.get().cloned()
}

fn api() -> std::result::Result<&'static Api, LoadError> {
    let loaded = API.get_or_init(|| {
        let library = Library::open(LT_LIBRARIES)?;
        let _ = LIBRARY_NAME.set(library.name().to_string());
        // Safety: each type is the exact C signature of the function named.
        unsafe {
            let api = Api {
                create: library.symbol("cublasLtCreate")?,
                destroy: library.symbol("cublasLtDestroy")?,
                desc_create: library.symbol("cublasLtMatmulDescCreate")?,
                desc_destroy: library.symbol("cublasLtMatmulDescDestroy")?,
                desc_set: library.symbol("cublasLtMatmulDescSetAttribute")?,
                layout_create: library.symbol("cublasLtMatrixLayoutCreate")?,
                layout_destroy: library.symbol("cublasLtMatrixLayoutDestroy")?,
                pref_create: library.symbol("cublasLtMatmulPreferenceCreate")?,
                pref_destroy: library.symbol("cublasLtMatmulPreferenceDestroy")?,
                pref_set: library.symbol("cublasLtMatmulPreferenceSetAttribute")?,
                heuristic: library.symbol("cublasLtMatmulAlgoGetHeuristic")?,
                version: library.symbol("cublasLtGetVersion").ok(),
                matmul: library.symbol("cublasLtMatmul")?,
            };
            let version = api.version.map_or_else(|| "unknown".to_string(), |v| v().to_string());
            tracing::info!(library = library.name(), cublaslt_version = version, "cuBLASLt library loaded");
            Ok(api)
        }
    });
    loaded.as_ref().map_err(Clone::clone)
}

fn check(call: &str, status: c_int) -> Result<()> {
    if status == STATUS_SUCCESS {
        Ok(())
    } else {
        Err(anyhow!("cuBLASLt {call} failed with status {status}"))
    }
}

/// Descriptors and the chosen algorithm of one shape.
struct Plan {
    desc: Handle,
    a: Handle,
    b: Handle,
    d: Handle,
    algo: HeuristicResult,
    has_bias: bool,
    /// The heuristic's answers, best first (`algo` is the first; the rest are only timed by `benchmark`).
    candidates: Vec<HeuristicResult>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Shape {
    tokens: usize,
    out: usize,
    inn: usize,
    has_bias: bool,
}

/// A cuBLASLt handle bound to the device's stream, with the shapes it has planned. Drop it before the device.
pub struct CublasLt {
    handle: Handle,
    api: &'static Api,
    stream: *mut c_void,
    /// Device address and size of the workspace every call is given.
    workspace: Mutex<(u64, usize)>,
    plans: Mutex<HashMap<Shape, Plan>>,
}

// Safety: the handle, the descriptors and the stream are only used behind the backend's mutexes.
unsafe impl Send for CublasLt {}

impl CublasLt {
    /// A `LoadError` (downcast to tell) when the library or one of its functions is missing.
    pub fn new(dev: &CudaDevice) -> Result<Self> {
        dev.check_new_gpu_work("cuBLASLt handle creation")?;
        let api = api()?;
        dev.set_current()?;
        let mut handle: Handle = std::ptr::null_mut();
        unsafe { check("cublasLtCreate", (api.create)(&mut handle))? };
        Ok(Self { handle, api, stream: dev.stream() as *mut c_void, workspace: Mutex::new((0, 0)), plans: Mutex::new(HashMap::new()) })
    }

    /// The scratch memory algorithms may use (the same buffer cuBLAS was given). Set before the first call.
    pub fn set_workspace(&self, ptr: u64, bytes: usize) {
        *self.workspace.lock().expect("workspace mutex poisoned") = (ptr, bytes);
    }

    fn plan(&self, shape: Shape) -> Result<Plan> {
        let dim = |d: usize| u64::try_from(d).map_err(|_| anyhow!("dimension {d} too large"));
        let (t, o, i) = (dim(shape.tokens)?, dim(shape.out)?, dim(shape.inn)?);
        let api = self.api;
        let (workspace_bytes, mut plan) = (self.workspace.lock().expect("workspace mutex poisoned").1, Plan { desc: std::ptr::null_mut(), a: std::ptr::null_mut(), b: std::ptr::null_mut(), d: std::ptr::null_mut(), algo: HeuristicResult { algo: [0; 8], workspace_size: 0, state: 0, waves_count: 0.0, reserved: [0; 4] }, has_bias: shape.has_bias, candidates: Vec::new() });
        let mut pref: Handle = std::ptr::null_mut();
        let built = (|| -> Result<()> {
            unsafe {
                check("MatmulDescCreate", (api.desc_create)(&mut plan.desc, COMPUTE_32F, CUDA_R_32F))?;
                for (attribute, value) in [(DESC_TRANSA, OP_T), (DESC_TRANSB, OP_N)] {
                    check("MatmulDescSetAttribute", (api.desc_set)(plan.desc, attribute, &value as *const i32 as *const c_void, 4))?;
                }
                let epilogue = if shape.has_bias { EPILOGUE_BIAS } else { EPILOGUE_DEFAULT };
                check("MatmulDescSetAttribute", (api.desc_set)(plan.desc, DESC_EPILOGUE, &epilogue as *const u32 as *const c_void, 4))?;
                // W is column-major [in, out], x is [in, tokens], y is [out, tokens]; all bf16.
                check("MatrixLayoutCreate", (api.layout_create)(&mut plan.a, CUDA_R_16BF, i, o, i as i64))?;
                check("MatrixLayoutCreate", (api.layout_create)(&mut plan.b, CUDA_R_16BF, i, t, i as i64))?;
                check("MatrixLayoutCreate", (api.layout_create)(&mut plan.d, CUDA_R_16BF, o, t, o as i64))?;
                check("MatmulPreferenceCreate", (api.pref_create)(&mut pref))?;
                let limit = workspace_bytes as u64;
                check("MatmulPreferenceSetAttribute", (api.pref_set)(pref, PREF_MAX_WORKSPACE_BYTES, &limit as *const u64 as *const c_void, 8))?;
                let mut found: c_int = 0;
                let mut results = [plan.algo; CANDIDATES];
                check("MatmulAlgoGetHeuristic", (api.heuristic)(self.handle, plan.desc, plan.a, plan.b, plan.d, plan.d, pref, CANDIDATES as c_int, results.as_mut_ptr(), &mut found))?;
                if found < 1 {
                    return Err(anyhow!("cuBLASLt has no algorithm for a {} x {} x {} bf16 product", shape.out, shape.tokens, shape.inn));
                }
                plan.candidates = results[..(found as usize).min(CANDIDATES)].to_vec();
                plan.algo = plan.candidates[0];
            }
            Ok(())
        })();
        unsafe {
            if !pref.is_null() {
                (api.pref_destroy)(pref);
            }
        }
        match built {
            Ok(()) => {
                tracing::info!("cuBLASLt {}: algorithm {} (heuristic, first of {})", shape_key(&shape), algo_hex(&plan.algo.algo), plan.candidates.len());
                Ok(plan)
            }
            Err(e) => {
                self.destroy(&plan);
                Err(e)
            }
        }
    }

    fn destroy(&self, plan: &Plan) {
        unsafe {
            for layout in [plan.a, plan.b, plan.d] {
                if !layout.is_null() {
                    (self.api.layout_destroy)(layout);
                }
            }
            if !plan.desc.is_null() {
                (self.api.desc_destroy)(plan.desc);
            }
        }
    }

    /// Time the heuristic's candidates for one shape (the buffers are only used for timing; `y` is overwritten) and return, per candidate in
    /// the heuristic's order, its id and microseconds per call (`None` when the candidate cannot run the shape). Changes nothing: the
    /// algorithm used by `linear_bf16` stays the heuristic's first. Run by `scripts/bench-cublaslt.sh`.
    #[allow(clippy::too_many_arguments)]
    pub fn benchmark(&self, dev: &CudaDevice, w: u64, x: u64, y: u64, bias: u64, tokens: usize, out: usize, inn: usize, runs: usize) -> Result<Vec<(String, Option<f64>)>> {
        // Plans are created by the first call; make sure this one exists (and has its heuristic candidates).
        self.linear_bf16(w, x, y, bias, tokens, out, inn)?;
        let shape = Shape { tokens, out, inn, has_bias: bias != 0 };
        let plans = self.plans.lock().expect("plan mutex poisoned");
        let plan = &plans[&shape];
        let (workspace, workspace_bytes) = *self.workspace.lock().expect("workspace mutex poisoned");
        let (alpha, beta) = (1.0f32, 0.0f32);
        let start = super::event::Event::new_timed(dev)?;
        let end = super::event::Event::new_timed(dev)?;
        let mut results = Vec::new();
        for candidate in &plan.candidates {
            let run = |count: usize| -> Result<()> {
                for _ in 0..count {
                    unsafe {
                        if plan.has_bias {
                            check("MatmulDescSetAttribute", (self.api.desc_set)(plan.desc, DESC_BIAS_POINTER, &bias as *const u64 as *const c_void, 8))?;
                        }
                        let status = (self.api.matmul)(self.handle, plan.desc, &alpha as *const f32 as *const c_void, w as *const c_void, plan.a, x as *const c_void, plan.b, &beta as *const f32 as *const c_void, y as *const c_void, plan.d, y as *mut c_void, plan.d, candidate, workspace as *mut c_void, workspace_bytes, self.stream);
                        check("cublasLtMatmul", status)?;
                    }
                }
                Ok(())
            };
            // A candidate that cannot run this shape is reported, not an error.
            if run(WARM_UPS).is_err() {
                results.push((algo_hex(&candidate.algo), None));
                continue;
            }
            dev.synchronize()?;
            start.record(dev)?;
            run(runs)?;
            end.record(dev)?;
            end.wait()?;
            results.push((algo_hex(&candidate.algo), Some(super::event::Event::elapsed_ms(&start, &end)? * 1000.0 / runs as f64)));
        }
        Ok(results)
    }

    /// `y[tokens, out] = x[tokens, in] * W[out, in]^T (+ bias)` with bf16 `W`, `x`, `y` and `bias` (`0` for none), f32
    /// accumulation, the bias added in f32 and one rounding to bf16. All are row-major device pointers.
    pub fn linear_bf16(&self, w: u64, x: u64, y: u64, bias: u64, tokens: usize, out: usize, inn: usize) -> Result<()> {
        let shape = Shape { tokens, out, inn, has_bias: bias != 0 };
        let mut plans = self.plans.lock().expect("plan mutex poisoned");
        if !plans.contains_key(&shape) {
            let plan = self.plan(shape)?;
            plans.insert(shape, plan);
        }
        let plan = &plans[&shape];
        let (workspace, workspace_bytes) = *self.workspace.lock().expect("workspace mutex poisoned");
        let (alpha, beta) = (1.0f32, 0.0f32);
        unsafe {
            if plan.has_bias {
                check("MatmulDescSetAttribute", (self.api.desc_set)(plan.desc, DESC_BIAS_POINTER, &bias as *const u64 as *const c_void, 8))?;
            }
            check(
                "cublasLtMatmul",
                (self.api.matmul)(
                    self.handle,
                    plan.desc,
                    &alpha as *const f32 as *const c_void,
                    w as *const c_void,
                    plan.a,
                    x as *const c_void,
                    plan.b,
                    &beta as *const f32 as *const c_void,
                    y as *const c_void,
                    plan.d,
                    y as *mut c_void,
                    plan.d,
                    &plan.algo,
                    workspace as *mut c_void,
                    workspace_bytes,
                    self.stream,
                ),
            )
        }
    }
}

impl Drop for CublasLt {
    fn drop(&mut self) {
        let plans = std::mem::take(self.plans.get_mut().expect("plan mutex poisoned"));
        for (_, plan) in plans {
            self.destroy(&plan);
        }
        if !self.handle.is_null() {
            unsafe {
                (self.api.destroy)(self.handle);
            }
        }
    }
}

fn shape_key(shape: &Shape) -> String {
    format!("{}x{}x{}{}", shape.tokens, shape.out, shape.inn, if shape.has_bias { "+bias" } else { "" })
}

fn algo_hex(algo: &[u64; 8]) -> String {
    algo.iter().map(|w| format!("{w:016x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_algorithm_id_is_128_hex_digits() {
        let algo = [1u64, 2, 3, u64::MAX, 0, 5, 6, 7];
        assert_eq!(algo_hex(&algo).len(), 128);
        assert!(algo_hex(&algo).starts_with("0000000000000001"));
    }

    #[test]
    fn the_shape_key_names_the_shape_and_the_bias() {
        assert_eq!(shape_key(&Shape { tokens: 1, out: 4608, inn: 3584, has_bias: true }), "1x4608x3584+bias");
        assert_eq!(shape_key(&Shape { tokens: 1, out: 3584, inn: 3584, has_bias: false }), "1x3584x3584");
    }

    #[test]
    fn the_library_is_loaded_or_the_failure_is_a_typed_load_error_naming_it() {
        match api() {
            Ok(_) => assert!(loaded_library().is_some()),
            Err(LoadError::LibraryMissing { tried, .. }) => assert!(tried.iter().all(|name| name.to_lowercase().contains("cublaslt")), "{tried:?}"),
            Err(other) => panic!("a missing function would mean a wrong symbol name: {other}"),
        }
    }

    #[test]
    fn the_heuristic_result_has_the_size_cublaslt_expects() {
        // 8 u64 of algorithm, size_t workspace, int state, float waves, 4 int reserved.
        assert_eq!(std::mem::size_of::<HeuristicResult>(), 64 + 8 + 4 + 4 + 16);
    }
}
