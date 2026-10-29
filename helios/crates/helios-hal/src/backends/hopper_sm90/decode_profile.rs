//! What a profiled generation reports: GPU time per group of kernels (CUDA events on the stream)
//! and host time per phase of a decode step. Verification and measurement only.

/// Totals over the profiled decode steps (the prompt is not included).
#[derive(Default, Debug)]
pub struct DecodeProfile {
    pub steps: usize,
    /// GPU milliseconds between event marks, summed per group; includes idle gaps between
    /// kernels. Filled for `DecodeMode::Direct` only (a replayed graph cannot carry marks).
    pub gpu_ms_by_group: Vec<(&'static str, f64)>,
    pub host_upload_ms: f64,
    /// Time the host spent issuing the step's launches (or the single graph launch).
    pub host_issue_ms: f64,
    /// Time blocked in the synchronisation after the launches.
    pub host_wait_ms: f64,
    /// Time copying the logits back and converting them.
    pub host_read_ms: f64,
    /// Time picking the next token from the logits on the host.
    pub host_sample_ms: f64,
}

/// Time of a window of decode steps measured inside one run: CUDA events around the window give
/// the GPU time, the host clock gives the wall time. No subtraction of two runs is involved and the
/// prompt is outside the window.
#[derive(Debug, Clone, Copy)]
pub struct DecodeTiming {
    /// Decode steps inside the window.
    pub steps: usize,
    /// GPU milliseconds from the start event to the end event (idle gaps included).
    pub gpu_ms: f64,
    /// Host milliseconds over the same window, from the start mark to the end event being reached.
    pub wall_ms: f64,
}

impl DecodeTiming {
    pub fn gpu_ms_per_token(&self) -> f64 {
        self.gpu_ms / self.steps as f64
    }

    pub fn wall_ms_per_token(&self) -> f64 {
        self.wall_ms / self.steps as f64
    }
}

impl DecodeProfile {
    pub(super) fn add_gpu(&mut self, group: &'static str, ms: f64) {
        match self.gpu_ms_by_group.iter_mut().find(|(g, _)| *g == group) {
            Some((_, total)) => *total += ms,
            None => self.gpu_ms_by_group.push((group, ms)),
        }
    }
}
