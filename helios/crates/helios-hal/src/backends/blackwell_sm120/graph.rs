//! `StaticCudaGraph` — lifecycle wrapper around a CUDA Graph + its
//! instantiated executable. This is the SM120's defining mechanism: capture
//! the decode plan once at load time, *instantiate* it into an executable
//! graph, then *launch* (replay) that executable each decode step.
//!
//! Lifecycle (driver calls):
//!   empty  →  [`begin_capture`]  →  [kernels run, recorded]  →  [`end_capture`]
//!           →  captured graph
//!           →  [`instantiate`]  →  executable graph
//!           →  [`launch`] (per decode step)  →  …
//!           →  [`destroy`] (shutdown)
//!
//! All calls are real FFI through `cuda-driver-sys`. No mock. On a host
//! without the driver, this module does not link — same posture as `device.rs`.

use anyhow::{anyhow, Result};
use cuda_driver_sys::{
    cuGraphDestroy, cuGraphExecDestroy, cuGraphInstantiate, cuGraphLaunch,
    cuStreamBeginCapture_v2, cuStreamEndCapture, cuStreamSynchronize, CUgraph, CUgraphExec,
    CUstreamCaptureMode,
};

use crate::cuda::{eval_cuda, CudaDevice};

/// Capture mode passed to `cuStreamBeginCapture`. We use
/// `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`: kernels running on other threads
/// (e.g. the scheduler's prefill) are allowed to proceed; capture only
/// records the work issued on this stream. That matches the decode-only
/// capture the backend wants.
const CAPTURE_MODE: CUstreamCaptureMode =
    CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;

/// Owned CUDA Graph handle. Distinct from the executable below.
pub struct StaticCudaGraph {
    graph: CUgraph,
    /// `None` until `instantiate` runs. The executable is what we *launch*.
    executable: Option<CUgraphExec>,
    /// True once `end_capture` has run; cleared when `instantiate` consumes
    /// the graph node list. Guards against launching a half-captured graph.
    captured: bool,
}

// Safety: the graph + executable are host-side opaque handles; the driver
// serialises access. We hold one `StaticCudaGraph` behind a mutex in the
// backend, so `Send` suffices.
unsafe impl Send for StaticCudaGraph {}

impl StaticCudaGraph {
    /// Construct an empty wrapper. No driver allocation yet — `graph` is
    /// null until capture begins. Callers must populate it via
    /// [`begin_capture`]/[`end_capture`] before [`instantiate`].
    pub fn empty() -> Self {
        Self {
            graph: std::ptr::null_mut(),
            executable: None,
            captured: false,
        }
    }

    /// Begin recording on the device's stream. Subsequent kernel launches on
    /// that stream are *captured* into the graph rather than executed. Caller
    /// must pair this with [`end_capture`] before any launch.
    pub fn begin_capture(&self, dev: &CudaDevice) -> Result<()> {
        dev.set_current()?;
        unsafe {
            eval_cuda(
                "cuStreamBeginCapture",
                cuStreamBeginCapture_v2(dev.stream(), CAPTURE_MODE),
            )?;
        }
        Ok(())
    }

    /// Stop recording, materialise the captured kernel sequence into the
    /// owned graph handle. After this, `self.captured == true`.
    pub fn end_capture(&mut self, dev: &CudaDevice) -> Result<()> {
        unsafe {
            eval_cuda(
                "cuStreamEndCapture",
                cuStreamEndCapture(dev.stream(), &mut self.graph),
            )?;
        }
        if self.graph.is_null() {
            return Err(anyhow!(
                "cuStreamEndCapture returned null graph — capture was empty or failed"
            ));
        }
        self.captured = true;
        Ok(())
    }

    /// Instantiate the captured graph into an executable. Requires that
    /// [`end_capture`] has run. Once instantiated, the executable can be
    /// launched any number of times without re-running capture.
    pub fn instantiate(&mut self) -> Result<()> {
        if !self.captured {
            return Err(anyhow!("cannot instantiate: graph not captured"));
        }
        let mut exec: CUgraphExec = std::ptr::null_mut();
        unsafe {
            // The driver's `cuGraphInstantiate` takes an error-node out-param
            // and a log buffer for instantiation diagnostics. We pass nulls
            // and a zero-size buffer; a failed instantiate surfaces via the
            // returned `CUresult` and `eval_cuda` instead, which is enough to
            // debug capture failures at this stage.
            eval_cuda(
                "cuGraphInstantiate",
                cuGraphInstantiate(
                    &mut exec,
                    self.graph,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                ),
            )?;
        }
        self.executable = Some(exec);
        Ok(())
    }

    /// Replay the instantiated graph on the device's stream. One
    /// `cudaGraphLaunch` instead of hundreds of per-op launches — the SM120
    /// payoff on the decode hot path.
    pub fn launch(&self, dev: &CudaDevice) -> Result<()> {
        let exec = self
            .executable
            .ok_or_else(|| anyhow!("cannot launch: graph not instantiated"))?;
        dev.set_current()?;
        unsafe {
            eval_cuda("cuGraphLaunch", cuGraphLaunch(exec, dev.stream()))?;
            // Synchronous replay is the conservative default. The scheduler
            // may switch to event-signalled async once it tracks per-request
            // completion. For now we wait so a step's logits are ready before
            // the next.
            eval_cuda("cuStreamSynchronize", cuStreamSynchronize(dev.stream()))?;
        }
        Ok(())
    }

    /// Placeholder entry retained for future use: when weights load, the
    /// backend will call `begin_capture`/`end_capture`/`instantiate` from
    /// `load_model_weights`. Left as a no-op seam so callers compile today.
    pub fn capture_empty_graph(&mut self) {
        self.captured = true;
    }

    /// Destroy the graph + executable, releasing driver-side state. Safe to
    /// call even if `instantiate` never ran.
    pub fn destroy(&mut self) {
        unsafe {
            if let Some(exec) = self.executable.take() {
                let _ = eval_cuda("cuGraphExecDestroy", cuGraphExecDestroy(exec));
            }
            if !self.graph.is_null() {
                let _ = eval_cuda("cuGraphDestroy", cuGraphDestroy(self.graph));
                self.graph = std::ptr::null_mut();
            }
        }
        self.captured = false;
    }
}

impl Drop for StaticCudaGraph {
    fn drop(&mut self) {
        self.destroy();
    }
}
