//! Kernel launch surface for SM120 — prefill + decode. These functions are the seam where the
//! FlashInfer-style attention/matmul kernels plug in once their binaries are vendored. Today they
//! exercise the shared graph machinery (`crate::cuda::GraphExec`, the same code the SM90 backend
//! uses) with no kernel work on the stream: a graph with zero nodes still captures, instantiates
//! and launches. Not run on any GPU yet (there is no Blackwell hardware).
//!
//! - `prefill_launch` — allocate the logits buffer, capture the (empty) decode graph.
//! - `decode_replay`  — launch the graph once and read back the argmax token id.

use anyhow::{anyhow, Result};

use crate::cuda::{CudaDevice, GraphExec};
use crate::reference::bf16_to_f32;
use crate::sampling::argmax;

/// Vocabulary dimension hard-coded for the placeholder. Real value comes
/// from the manifest's tokenizer config once wired.
pub const VOCAB_SIZE: usize = 128_256;

/// What the decode path keeps between prefill and decode: the instantiated graph and the logits
/// buffer it fills. Owned by the backend (no global state).
#[derive(Default)]
pub struct DecodeState {
    graph: Option<GraphExec>,
    logits: Option<u64>,
}

impl DecodeState {
    /// Free the logits buffer and drop the graph. Call before the device is torn down.
    pub fn release(&mut self, dev: &mut CudaDevice) {
        if let Some(pointer) = self.logits.take() {
            let _ = dev.free(pointer);
        }
        self.graph = None;
    }
}

/// Capture the decode graph (no kernel work yet) and allocate the logits buffer; returns its pointer.
pub fn prefill_launch(dev: &mut CudaDevice, state: &mut DecodeState, _tokens: &[u32]) -> Result<u64> {
    state.release(dev);
    let logits = dev.alloc(VOCAB_SIZE * 2)?; // one row, bf16
    // TODO: launch the FlashInfer prefill and decode kernels inside the closure once vendored.
    let graph = match GraphExec::capture(dev, || Ok(())) {
        Ok(graph) => graph,
        Err(error) => {
            let _ = dev.free(logits);
            return Err(error);
        }
    };
    state.graph = Some(graph);
    state.logits = Some(logits);
    Ok(logits)
}

/// Replay the captured decode graph and return the argmax token id of the logits row. The logits
/// are converted from bf16 before comparing, so negative values order correctly.
pub fn decode_replay(dev: &CudaDevice, state: &DecodeState) -> Result<u32> {
    let (Some(graph), Some(logits)) = (state.graph.as_ref(), state.logits) else {
        return Err(anyhow!("decode_replay called before prefill_launch"));
    };
    graph.launch(dev)?;
    dev.synchronize()?; // the host copy below is not ordered with the stream the graph ran on
    let mut host = vec![0u8; VOCAB_SIZE * 2];
    dev.dtoh(&mut host, logits)?;
    let values: Vec<f32> = host.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
    Ok(argmax(&values) as u32)
}
