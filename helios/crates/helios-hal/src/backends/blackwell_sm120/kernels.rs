//! Kernel launch surface for SM120 — prefill + decode. These functions are
//! the seam where the FlashInfer-style attention/matmul kernels plug in once
//! their PTX is vendored into the package. Today they:
//!
//! - `prefill_launch`  — begin graph capture, issue (placeholder) kernel
//!   work on the stream, end capture, and *instantiate* the result graph.
//!   Returns a device pointer to the logits buffer (allocated here; the
//!   real pipeline will have it as a kernel output).
//! - `decode_replay`   — launch the instantiated graph once per decode step
//!   and read back the argmax'd token id from the logits buffer.
//!
//! The "FlashInfer" kernel bodies themselves are PTX that the package will
//! embed; the graph captures whatever runs on the stream during capture,
//! including those PTX kernels once they're loaded via `cuModuleLoad`. Until
//! then the functions below exercise the *graph machinery* end-to-end with
//! no kernel work on the stream — a graph with zero nodes still captures,
//! instantiates, and launches, which proves the FFI path is sound.

use anyhow::Result;
use cuda_driver_sys::{cuMemAlloc_v2, cuMemFree_v2, cuMemcpyDtoH_v2, CUdeviceptr};

use super::device::CudaDevice;
use super::ffi::eval_cuda;
use super::graph::StaticCudaGraph;

/// Vocabulary dimension hard-coded for the placeholder. Real value comes
/// from the manifest's tokenizer config once wired.
pub const VOCAB_SIZE: usize = 128_256;

/// Output logits pointer per request. Allocated on prefill, read on decode.
/// Held at module scope under the lint escape because the real pipeline owns
/// it from a per-request struct; this is the FFI smoke path.
static mut LOGITS_PTR: CUdeviceptr = 0;

/// Capture the prefill graph (with placeholder kernel work — none for now)
/// and allocate + return the logits buffer pointer.
pub fn prefill_launch(
    dev: &CudaDevice,
    graph: &mut StaticCudaGraph,
    _tokens: &[u32],
) -> Result<u64> {
    // 1. Allocate the logits buffer: [batch, vocab] of bf16.
    let bytes = VOCAB_SIZE * 2; // one row, bf16
    let mut logits: CUdeviceptr = 0;
    unsafe {
        eval_cuda("cuMemAlloc (logits)", cuMemAlloc_v2(&mut logits, bytes))?;
    }

    // 2. Capture: begin, issue kernel work on the stream (none for now), end.
    graph.begin_capture(dev)?;
    // TODO: launch FlashInfer prefill kernel(s) here once PTX is vendored.
    graph.end_capture(dev)?;

    // 3. Instantiate for replay during decode.
    graph.instantiate()?;

    // 4. Stash the logits pointer for decode_replay to read. Module-scope
    // storage is the smoke-path simplification; the real pipeline owns this
    // from a per-request struct held in the scheduler.
    unsafe {
        LOGITS_PTR = logits;
    }
    Ok(logits)
}

/// Replay the captured decode graph and return the argmax'd token id.
///
/// Real implementation does: launch graph → `cuMemcpy` logits row to host →
/// argmax. The argmax-from-device step needs a small host staging buffer and
/// a host-side scan — both trivial once the codec lands. Today, after the
/// graph launch we read back zero-extended bf16 values, take argmax over
/// `VOCAB_SIZE` half-precision lanes, and return that token id.
pub fn decode_replay(dev: &CudaDevice, graph: &StaticCudaGraph) -> Result<u32> {
    graph.launch(dev)?;

    // Read logits row back to a host buffer and argmax. bf16 → u16 view.
    let bytes = VOCAB_SIZE * 2;
    let mut host_buf = vec![0u8; bytes];
    unsafe {
        let logits = LOGITS_PTR;
        if logits == 0 {
            return Err(anyhow::anyhow!("decode_replay called before prefill_launch"));
        }
        eval_cuda(
            "cuMemcpyDtoH",
            cuMemcpyDtoH_v2(host_buf.as_mut_ptr().cast(), logits, bytes),
        )?;
    }

    // Argmax over bf16 lanes. We compare as u16 because bf16's bitwise order
    // preserves IEEE-754 total ordering for non-negative values, which is the
    // common case for logits after a softmax-ish layer. A precise argmax
    // would convert bf16→f32 first; the codec vendor will swap this in.
    let u16_view: &[u16] = bytewise_as_u16(&host_buf);
    let best = u16_view
        .iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
        .map(|(i, _)| i as u32)
        .unwrap_or(0);
    Ok(best)
}

/// View 2 bytes at a time as u16. No `bytemuck` dep — a plain byte-slice to
/// u16-slice cast via `slice::from_raw_parts` after an alignment check.
fn bytewise_as_u16(bytes: &[u8]) -> &[u16] {
    assert_eq!(bytes.len() % 2, 0, "logits buffer must be even-length");
    let ptr = bytes.as_ptr();
    assert!(
        ptr.align_offset(2) == 0,
        "logits buffer must be 2-byte aligned"
    );
    // SAFETY: length is even and ptr is 2-aligned (checked above). The
    // resulting slice borrows from `bytes` with the same lifetime.
    unsafe { std::slice::from_raw_parts(ptr.cast::<u16>(), bytes.len() / 2) }
}

// Own the logits pointer's free on module unload. The backend Shutdown path
// cleans up the device; logits live as long as the engine.
struct LogitsGuard;
impl Drop for LogitsGuard {
    fn drop(&mut self) {
        unsafe {
            if LOGITS_PTR != 0 {
                let _ = eval_cuda("cuMemFree (logits)", cuMemFree_v2(LOGITS_PTR));
                LOGITS_PTR = 0;
            }
        }
    }
}
// Static: kept for the engine's lifetime.
static _LOGITS_GUARD: LogitsGuard = LogitsGuard;
