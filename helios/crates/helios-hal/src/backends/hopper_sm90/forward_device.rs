//! Generation on the device: the public entry points of [`HopperBackend`] and the loop that runs a
//! prompt and its decode steps on a loaded [`Session`] (see `session.rs`, where the buffers, kernels and
//! graphs are built once at load time). Activations, the per-layer K/V cache and the RoPE tables live in
//! GPU memory, every op is one launch on a device pointer (custom kernels and cuBLAS share a stream, so
//! nothing waits between ops), and only token ids, or for sampling and verification the final logits,
//! travel back to the host. It computes exactly what the host-sliced path in `forward.rs` computes, with
//! the same kernels and the same rounding, and a hardware test requires the logits to be bit-identical to
//! that path.
//!
//! A generation runs with new GPU work forbidden (no allocation, module load, graph capture or cuBLAS
//! handle creation): everything it uses already exists.

use std::cell::RefCell;
use std::time::Instant;

use anyhow::{anyhow, Result};

use super::backend::HopperBackend;
use super::decode_profile::{DecodeProfile, DecodeTiming};
use super::forward::Qwen2Dims;
use super::session::Session;
use crate::cuda::profile::StepMarks;
use crate::cuda::{Cublas, CudaDevice};
use crate::sampling::{sample, Rng, SamplingConfig};
use crate::traits::StopReason;

/// Steps run before the timed window starts. With the graphs captured at load time nothing needs
/// warming up; the first steps still absorb clock ramp-up and the first replay of each graph.
const TIMING_WARM_UP: usize = 3;

/// How decode steps (one token after the prompt) are issued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeMode {
    /// Every step launches its kernels one by one, the arguments carrying the sequence length.
    Direct,
    /// Every step replays the CUDA graph captured when the model was loaded. The per-token values
    /// (token id, position, ring base) live in device memory, so the same graph is valid for every
    /// step of every generation.
    Graph,
}

impl HopperBackend {
    /// Greedy or sampled generation entirely on the device; see the module docs. Same contract as
    /// [`HopperBackend::generate_streaming_host`]. Decode steps are launched one by one.
    ///
    /// Greedy decoding chains the steps on the GPU: the argmax kernel writes each token as the
    /// next step's input, so only 4-byte token ids reach the host and the host runs one step
    /// behind the device (an end-of-sequence token is noticed one step late and that extra step
    /// is discarded). Sampled decoding still copies the logits back each step.
    pub fn generate_streaming(
        &self,
        d: &Qwen2Dims,
        prompt: &[u32],
        max_new: usize,
        eos: Option<u32>,
        sampling: &SamplingConfig,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<StopReason> {
        self.run_generation(d, prompt, max_new, eos, sampling, on_token, DecodeMode::Direct, None, None, None, None)
    }

    /// Time `steps` greedy decode steps inside one run: CUDA events and the host clock around the
    /// window, after warm-up steps, so neither the prompt nor the first replays are in the figure.
    pub fn generate_timed(&self, d: &Qwen2Dims, prompt: &[u32], steps: usize, mode: DecodeMode) -> Result<DecodeTiming> {
        let mut timing = None;
        let max_new = 1 + TIMING_WARM_UP + steps;
        self.run_generation(d, prompt, max_new, None, &SamplingConfig::greedy(), &mut |_| {}, mode, None, None, None, Some(&mut timing))?;
        timing.ok_or_else(|| anyhow!("no timing produced (steps must be at least 1)"))
    }

    /// Greedy generation of `max_new` tokens with the decode steps profiled; see [`DecodeProfile`].
    pub fn generate_profiled(&self, d: &Qwen2Dims, prompt: &[u32], max_new: usize, mode: DecodeMode) -> Result<DecodeProfile> {
        let mut profile = DecodeProfile::default();
        self.run_generation(d, prompt, max_new, None, &SamplingConfig::greedy(), &mut |_| {}, mode, None, None, Some(&mut profile), None)?;
        Ok(profile)
    }

    /// [`Self::generate_streaming`] with an explicit decode mode.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_mode(
        &self,
        d: &Qwen2Dims,
        prompt: &[u32],
        max_new: usize,
        eos: Option<u32>,
        sampling: &SamplingConfig,
        mode: DecodeMode,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<StopReason> {
        self.run_generation(d, prompt, max_new, eos, sampling, on_token, mode, None, None, None, None)
    }

    /// Like [`Self::generate_streaming`] with an explicit decode mode; when `trace` is given, the
    /// logits of every step (prompt first) are appended to it. For verification and measurement.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_traced(
        &self,
        d: &Qwen2Dims,
        prompt: &[u32],
        max_new: usize,
        eos: Option<u32>,
        sampling: &SamplingConfig,
        mode: DecodeMode,
        trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<StopReason> {
        self.run_generation(d, prompt, max_new, eos, sampling, &mut |_| {}, mode, None, trace, None, None)
    }

    /// Teacher-forced decode: the prompt, then the given tokens fed one per step regardless of what
    /// the model would have picked. `trace` receives `forced.len()` logits rows: the one after the
    /// prompt, then the one after each forced token except the last. Comparing those rows with a
    /// reference run on the same tokens is not disturbed by a legitimate near-tie flip of the
    /// argmax, which would otherwise make every later step incomparable.
    pub fn generate_forced(&self, d: &Qwen2Dims, prompt: &[u32], forced: &[u32], mode: DecodeMode, trace: &mut Vec<Vec<f32>>) -> Result<StopReason> {
        self.run_generation(d, prompt, forced.len(), None, &SamplingConfig::greedy(), &mut |_| {}, mode, Some(forced), Some(trace), None, None)
    }

    /// Logits of the last prompt token through the device pipeline (prefill only); the
    /// verification entry point that is compared bit for bit with the host path.
    pub fn forward_prefill_logits_device(&self, d: &Qwen2Dims, prompt: &[u32]) -> Result<Vec<f32>> {
        let mut trace = Vec::new();
        self.run_generation(d, prompt, 0, None, &SamplingConfig::greedy(), &mut |_| {}, DecodeMode::Direct, None, Some(&mut trace), None, None)?;
        trace.into_iter().next().ok_or_else(|| anyhow!("no logits produced"))
    }

    /// The longest prompt plus generated tokens the loaded model's decode session holds, if one is built.
    pub fn context_capacity(&self) -> Option<usize> {
        self.session.lock().expect("session mutex poisoned").as_ref().map(|s| s.capacity)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_generation(
        &self,
        d: &Qwen2Dims,
        prompt: &[u32],
        max_new: usize,
        eos: Option<u32>,
        sampling: &SamplingConfig,
        on_token: &mut dyn FnMut(u32),
        mode: DecodeMode,
        forced: Option<&[u32]>,
        trace: Option<&mut Vec<Vec<f32>>>,
        profile: Option<&mut DecodeProfile>,
        timing: Option<&mut Option<DecodeTiming>>,
    ) -> Result<StopReason> {
        if prompt.is_empty() {
            return Err(anyhow!("no prompt tokens"));
        }
        let dev = self.device.lock().expect("device mutex poisoned");
        let session = self.session.lock().expect("session mutex poisoned");
        let session = session.as_ref().ok_or_else(|| anyhow!("no model loaded: call load_model_weights first"))?;
        if session.dims != *d {
            return Err(anyhow!("the loaded model's dimensions differ from the ones passed to generate"));
        }
        let cublas = self.cublas.lock().expect("cublas mutex poisoned");
        let cublas = cublas.as_ref().ok_or_else(|| anyhow!("the cuBLAS handle was not created when the model was loaded"))?;
        // Everything the generation uses was built at load time; refuse to build anything now.
        let _no_new_gpu_work = dev.forbid_new_gpu_work();
        session.generate(&dev, cublas, prompt, max_new, eos, sampling, on_token, mode, forced, trace, profile, timing)
    }
}

impl Session {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate(
        &self,
        dev: &CudaDevice,
        cublas: &Cublas,
        prompt: &[u32],
        max_new: usize,
        eos: Option<u32>,
        sampling: &SamplingConfig,
        on_token: &mut dyn FnMut(u32),
        mode: DecodeMode,
        forced: Option<&[u32]>,
        mut trace: Option<&mut Vec<Vec<f32>>>,
        mut profile: Option<&mut DecodeProfile>,
        timing: Option<&mut Option<DecodeTiming>>,
    ) -> Result<StopReason> {
        if let Some(forced) = forced {
            if forced.len() < max_new {
                return Err(anyhow!("{} forced tokens given for {max_new} steps", forced.len()));
            }
            if trace.is_none() {
                return Err(anyhow!("forced tokens need a trace: the logits are the point"));
            }
        }
        let total = prompt.len() + max_new;
        if total > self.capacity {
            return Err(anyhow!("prompt + max_new = {total} exceeds the {} tokens of context this model was loaded with", self.capacity));
        }
        let b = &self.b;

        // One forward step over `ids` placed at `start..start + ids.len()`; returns the logits. A step
        // after the first token is a decode step: the split-K kernels, position on the device.
        let step = |ids: &[u32], start: usize| -> Result<Vec<f32>> {
            self.upload(dev, ids, start)?;
            self.launches(dev, cublas, ids.len(), start, start > 0, None)?;
            dev.synchronize()?;
            self.read_logits(dev)
        };

        if trace.is_none() && profile.is_none() && sampling.is_greedy() {
            if max_new == 0 {
                return Ok(StopReason::MaxTokens);
            }
            // Greedy: the argmax kernel picks each token and feeds it to the next step, so the
            // host never needs the logits. It reads token ids from a device ring, one step late.
            let ring_base = prompt.len() - 1;
            // Prefill, then token 0 from its logits. The position buffer must hold the last
            // prompt position for the chain; wait for the prefill before overwriting it.
            self.upload(dev, prompt, 0)?;
            self.launches(dev, cublas, prompt.len(), 0, false, None)?;
            dev.synchronize()?;
            dev.htod(b.positions, &(ring_base as u32).to_le_bytes())?;
            dev.htod(b.ring_base, &(ring_base as u32).to_le_bytes())?;
            self.pick(dev)?;
            let events = &self.events;
            events[0].record(dev)?;
            let read_token = |index: usize| -> Result<u32> {
                events[index % events.len()].wait()?;
                let mut raw = [0u8; 4];
                dev.dtoh(&mut raw, b.ring + (index * 4) as u64)?;
                Ok(u32::from_le_bytes(raw))
            };
            let window = match timing {
                Some(out) if max_new > TIMING_WARM_UP + 1 => Some(out),
                _ => None,
            };
            let mut window_started = None;
            for index in 1..max_new {
                // Enqueue the step that produces token `index`, then read token `index - 1`
                // while the device works on it.
                match mode {
                    DecodeMode::Graph => self.chain_graph()?.launch(dev)?,
                    DecodeMode::Direct => {
                        self.launches(dev, cublas, 1, 0, true, None)?;
                        self.pick(dev)?;
                    }
                }
                events[index % events.len()].record(dev)?;
                if window.is_some() {
                    if index == TIMING_WARM_UP {
                        self.window.0.record(dev)?;
                        window_started = Some(Instant::now());
                    } else if index == max_new - 1 {
                        self.window.1.record(dev)?;
                    }
                }
                let token = read_token(index - 1)?;
                on_token(token);
                if Some(token) == eos {
                    dev.synchronize()?;
                    return Ok(StopReason::EndOfSequence);
                }
            }
            let token = read_token(max_new - 1)?;
            on_token(token);
            dev.synchronize()?;
            if let (Some(out), Some(started)) = (window, window_started) {
                self.window.1.wait()?;
                let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
                *out = Some(DecodeTiming {
                    steps: max_new - 1 - TIMING_WARM_UP,
                    gpu_ms: crate::cuda::event::Event::elapsed_ms(&self.window.0, &self.window.1)?,
                    wall_ms,
                });
            }
            return Ok(if Some(token) == eos { StopReason::EndOfSequence } else { StopReason::MaxTokens });
        }

        let mut rng = Rng::new(sampling.seed);
        let mut logits = step(prompt, 0)?;
        if let Some(trace) = trace.as_mut() {
            trace.push(logits.clone());
        }
        let mut position = prompt.len();
        for produced in 0..max_new {
            let sampled_at = Instant::now();
            let next = match forced {
                Some(forced) => forced[produced],
                None => sample(&logits, sampling, &mut rng),
            };
            if produced > 0 {
                if let Some(p) = profile.as_mut() {
                    p.host_sample_ms += sampled_at.elapsed().as_secs_f64() * 1000.0;
                }
            }
            on_token(next);
            if Some(next) == eos {
                return Ok(StopReason::EndOfSequence);
            }
            if produced + 1 < max_new {
                let timed = profile.is_some();
                let t0 = Instant::now();
                self.upload(dev, &[next], position)?;
                let t1 = Instant::now();
                let marks = if timed && mode == DecodeMode::Direct { Some(RefCell::new(StepMarks::start(dev)?)) } else { None };
                match mode {
                    DecodeMode::Graph => self.logits_graph()?.launch(dev)?,
                    DecodeMode::Direct => self.launches(dev, cublas, 1, position, true, marks.as_ref())?,
                }
                let t2 = Instant::now();
                dev.synchronize()?;
                let t3 = Instant::now();
                logits = self.read_logits(dev)?;
                let t4 = Instant::now();
                if timed {
                    if let Some(p) = profile.as_mut() {
                        let ms = |a: Instant, b: Instant| b.duration_since(a).as_secs_f64() * 1000.0;
                        p.steps += 1;
                        p.host_upload_ms += ms(t0, t1);
                        p.host_issue_ms += ms(t1, t2);
                        p.host_wait_ms += ms(t2, t3);
                        p.host_read_ms += ms(t3, t4);
                        if let Some(m) = &marks {
                            for (group, gpu_ms) in m.borrow().read()? {
                                p.add_gpu(group, gpu_ms);
                            }
                        }
                    }
                }
                if let Some(trace) = trace.as_mut() {
                    trace.push(logits.clone());
                }
                position += 1;
            }
        }
        Ok(StopReason::MaxTokens)
    }
}
