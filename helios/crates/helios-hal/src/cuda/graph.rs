//! CUDA graph capture of the device's stream: record the launches made by a closure once, then
//! replay them with a single launch call.
//!
//! `cuda-driver-sys` 0.3 binds `cuGraphInstantiate` with its pre-CUDA-12 five-argument form
//! (error node, log buffer, size), while CUDA 12 and later export it with a flags argument.
//! `cuGraphInstantiateWithFlags` has had one signature since CUDA 11.4, so it is declared here
//! instead of trusting the generated binding.

use std::os::raw::c_ulonglong;

use anyhow::{anyhow, Result};
use cuda_driver_sys::{
    cuGraphDestroy, cuGraphExecDestroy, cuGraphLaunch, cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph,
    CUgraphExec, CUresult, CUstreamCaptureMode,
};

use super::device::CudaDevice;
use super::ffi::eval_cuda;

extern "C" {
    fn cuGraphInstantiateWithFlags(exec: *mut CUgraphExec, graph: CUgraph, flags: c_ulonglong) -> CUresult;
}

/// An instantiated graph, destroyed on drop. It must be dropped before the device is torn down.
pub struct GraphExec {
    exec: CUgraphExec,
}

// Safety: the exec handle is only used through the device mutex of the backend.
unsafe impl Send for GraphExec {}

impl GraphExec {
    /// Run `record` while the device's stream is capturing: the launches it makes are recorded,
    /// not executed. `record` must not allocate or free device memory, synchronise, or use the
    /// legacy default stream, or the driver rejects the capture (and this returns its error).
    pub fn capture(dev: &CudaDevice, record: impl FnOnce() -> Result<()>) -> Result<Self> {
        dev.set_current()?;
        let stream = dev.stream();
        unsafe {
            eval_cuda(
                "cuStreamBeginCapture",
                cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
            )?;
        }
        let recorded = record();
        let mut graph: CUgraph = std::ptr::null_mut();
        // Always end the capture, even after a failure, so the stream is usable again.
        let ended = unsafe { eval_cuda("cuStreamEndCapture", cuStreamEndCapture(stream, &mut graph)) };
        recorded?;
        ended?;
        if graph.is_null() {
            return Err(anyhow!("stream capture produced no graph"));
        }
        let mut exec: CUgraphExec = std::ptr::null_mut();
        let instantiated = unsafe { eval_cuda("cuGraphInstantiateWithFlags", cuGraphInstantiateWithFlags(&mut exec, graph, 0)) };
        unsafe {
            cuGraphDestroy(graph);
        }
        instantiated?;
        Ok(Self { exec })
    }

    /// Replay the recorded launches on the device's stream (asynchronous, like any launch).
    pub fn launch(&self, dev: &CudaDevice) -> Result<()> {
        dev.set_current()?;
        unsafe { eval_cuda("cuGraphLaunch", cuGraphLaunch(self.exec, dev.stream())) }
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            unsafe {
                cuGraphExecDestroy(self.exec);
            }
        }
    }
}
