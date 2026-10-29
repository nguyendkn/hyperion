//! In-process GPU timing with CUDA events: mark points on the device's stream while a step is
//! being launched, then read how long the GPU spent between consecutive marks. Used only when a
//! caller asks for a profile; the normal decode path creates no events.

use std::os::raw::c_uint;

use anyhow::Result;
use cuda_driver_sys::{cuEventCreate, cuEventDestroy_v2, cuEventElapsedTime, cuEventRecord, cuEventSynchronize, CUevent};

use super::device::CudaDevice;
use super::ffi::eval_cuda;

const CU_EVENT_DEFAULT: c_uint = 0;

/// Event marks of one step: `start`, then one mark per finished group of launches.
pub struct StepMarks {
    events: Vec<(&'static str, CUevent)>,
}

impl StepMarks {
    /// Record the step's start on the device's stream.
    pub fn start(dev: &CudaDevice) -> Result<Self> {
        let mut marks = Self { events: Vec::new() };
        marks.mark(dev, "start")?;
        Ok(marks)
    }

    /// Record that everything launched since the previous mark belongs to `group`.
    pub fn mark(&mut self, dev: &CudaDevice, group: &'static str) -> Result<()> {
        dev.set_current()?;
        let mut event: CUevent = std::ptr::null_mut();
        unsafe {
            eval_cuda("cuEventCreate", cuEventCreate(&mut event, CU_EVENT_DEFAULT))?;
        }
        self.events.push((group, event));
        unsafe { eval_cuda("cuEventRecord", cuEventRecord(event, dev.stream())) }
    }

    /// Wait for the last mark and return the GPU milliseconds between consecutive marks, by group
    /// (the same group may appear many times; the caller sums). Includes any idle gap between two
    /// kernels, since an event is stamped when the stream reaches it.
    pub fn read(&self) -> Result<Vec<(&'static str, f64)>> {
        let Some((_, last)) = self.events.last() else { return Ok(Vec::new()) };
        unsafe {
            eval_cuda("cuEventSynchronize", cuEventSynchronize(*last))?;
        }
        let mut out = Vec::with_capacity(self.events.len());
        for pair in self.events.windows(2) {
            let mut ms = 0f32;
            unsafe {
                eval_cuda("cuEventElapsedTime", cuEventElapsedTime(&mut ms, pair[0].1, pair[1].1))?;
            }
            out.push((pair[1].0, f64::from(ms)));
        }
        Ok(out)
    }
}

impl Drop for StepMarks {
    fn drop(&mut self) {
        for (_, event) in self.events.drain(..) {
            unsafe {
                cuEventDestroy_v2(event);
            }
        }
    }
}
