//! A CUDA event on the device's stream: lets the host wait for one point of the queued work
//! without waiting for everything queued after it.

use std::os::raw::c_uint;

use anyhow::Result;
use cuda_driver_sys::{cuEventCreate, cuEventDestroy_v2, cuEventElapsedTime, cuEventRecord, cuEventSynchronize, CUevent};

use super::device::CudaDevice;
use super::ffi::eval_cuda;

const CU_EVENT_DEFAULT: c_uint = 0;
const CU_EVENT_DISABLE_TIMING: c_uint = 2;

pub struct Event {
    event: CUevent,
}

// Safety: used only behind the backend's device mutex.
unsafe impl Send for Event {}

impl Event {
    /// An event that only orders work (cheaper: it carries no timestamp).
    pub fn new(dev: &CudaDevice) -> Result<Self> {
        Self::with_flags(dev, CU_EVENT_DISABLE_TIMING)
    }

    /// An event that also records a timestamp, for [`Event::elapsed_ms`].
    pub fn new_timed(dev: &CudaDevice) -> Result<Self> {
        Self::with_flags(dev, CU_EVENT_DEFAULT)
    }

    fn with_flags(dev: &CudaDevice, flags: c_uint) -> Result<Self> {
        dev.set_current()?;
        let mut event: CUevent = std::ptr::null_mut();
        unsafe {
            eval_cuda("cuEventCreate", cuEventCreate(&mut event, flags))?;
        }
        Ok(Self { event })
    }

    /// GPU milliseconds between two timed events, both already reached (call [`Event::wait`] on `end` first).
    pub fn elapsed_ms(start: &Event, end: &Event) -> Result<f64> {
        let mut ms = 0f32;
        unsafe {
            eval_cuda("cuEventElapsedTime", cuEventElapsedTime(&mut ms, start.event, end.event))?;
        }
        Ok(f64::from(ms))
    }

    /// Complete when everything queued on the stream so far has finished.
    pub fn record(&self, dev: &CudaDevice) -> Result<()> {
        dev.set_current()?;
        unsafe { eval_cuda("cuEventRecord", cuEventRecord(self.event, dev.stream())) }
    }

    /// Block the host until the recorded point has been reached.
    pub fn wait(&self) -> Result<()> {
        unsafe { eval_cuda("cuEventSynchronize", cuEventSynchronize(self.event)) }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            cuEventDestroy_v2(self.event);
        }
    }
}
