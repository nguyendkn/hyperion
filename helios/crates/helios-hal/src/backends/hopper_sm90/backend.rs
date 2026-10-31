//! The `HopperBackend` struct + its `ComputeBackend` impl.
//!
//! Built on the shared CUDA Driver-API core (ADR 0001). Today it can probe a
//! real device and upload a model's weights; kernels, the KV cache and the
//! forward pass are not written yet and return a clear "not wired" error, so
//! the engine can never reach `Serving` on this backend by accident.

use std::collections::HashMap;
use std::fs::File;
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};

use std::ffi::c_void;

use crate::cuda::{Cublas, CudaDevice, DeviceWeights, Group, KernelModule};
use super::forward::Qwen2Dims;
use super::session::Session;
use crate::sampling::SamplingConfig;
use crate::contract::{
    BackendConfig, Capabilities, GraphSet, HalError, KvPool, KvReservation, KvSpec, LoadedModel, MemoryPolicy, RankInfo, StepTicket, WeightSource,
};
use crate::step::{StepOutput, StepPlan};
use crate::traits::{ComputeBackend, HardwareInfo, StopReason};
use crate::weights_plan::{self, PayloadFile};

const NAME: &str = "sm_90";

/// Fat binary built from `kernels/rmsnorm.cu` (rebuild with `scripts/build-kernels.sh`).
pub(super) const RMSNORM_IMAGE: &[u8] = include_bytes!("../../../kernels/rmsnorm.fatbin");

/// Threads per block for the row-wise kernels; a power of two, as the reduction needs.
pub(super) const ROW_BLOCK: u32 = 256;

/// Field order is drop order: the decode session (graphs, events) first, then the cuBLAS handle,
/// then the weights' owner, then the device (which releases the context they all live in).
pub struct HopperBackend {
    pub(super) session: Mutex<Option<Session>>,
    pub(super) cublas: Mutex<Option<Cublas>>,
    pub(super) modules: Mutex<HashMap<&'static str, KernelModule>>,
    pub(super) weights: Mutex<Option<DeviceWeights>>,
    /// Model dimensions and the tokens that end a generation, taken from the manifest at load time.
    runtime: Mutex<Option<(Qwen2Dims, Vec<u32>)>>,
    /// Set by a verification run to load a package that needs QK-norm before that path is verified (`QK_NORM_VERIFIED`).
    allow_qk_norm: std::sync::atomic::AtomicBool,
    pub(super) device: Mutex<CudaDevice>,
}

/// The weight groups the decode step reads as one matrix: per layer `qkv` (Q, K, V rows, and their biases when the model has
/// them) and `gate_up`. Names are the manifest's tensor names.
pub(super) fn fused_groups(d: &Qwen2Dims) -> Vec<Group> {
    let mut groups = Vec::with_capacity(3 * d.layers);
    for l in 0..d.layers {
        let n = |part: &str| format!("layers.{l}.{part}");
        groups.push(Group { name: n("qkv.weight"), members: ["q_proj", "k_proj", "v_proj"].iter().map(|p| n(&format!("self_attn.{p}.weight"))).collect() });
        if d.qkv_bias {
            groups.push(Group { name: n("qkv.bias"), members: ["q_proj", "k_proj", "v_proj"].iter().map(|p| n(&format!("self_attn.{p}.bias"))).collect() });
        }
        groups.push(Group { name: n("gate_up.weight"), members: ["gate_proj", "up_proj"].iter().map(|p| n(&format!("mlp.{p}.weight"))).collect() });
    }
    groups
}

impl HopperBackend {
    /// Probe a real CUDA device. Fails when the driver or a device is absent, so
    /// the factory skips this backend on hosts without a GPU.
    pub fn new() -> Result<Self> {
        Ok(Self { session: Mutex::new(None), cublas: Mutex::new(None), modules: Mutex::new(HashMap::new()), weights: Mutex::new(None), runtime: Mutex::new(None), allow_qk_norm: std::sync::atomic::AtomicBool::new(false), device: Mutex::new(CudaDevice::probe()?) })
    }

    /// Let this backend load a package that needs QK-norm although that path has not been verified against an oracle yet
    /// (`QK_NORM_VERIFIED`): for the run that does the verifying. Call it before `load_model`.
    pub fn allow_unverified_qk_norm(&self) {
        self.allow_qk_norm.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Copy one uploaded tensor back to the host. A verification aid for hardware
    /// tests (device bytes must equal file bytes); not used on any serving path.
    pub fn read_back_weight(&self, name: &str) -> Result<Vec<u8>> {
        let dev = self.device.lock().expect("device mutex poisoned");
        let weights = self.weights.lock().expect("weights mutex poisoned");
        weights.as_ref().ok_or_else(|| anyhow!("no weights loaded"))?.download(&dev, name)
    }

    /// `y[tokens, out] = x[tokens, in] * W[out, in]^T` on the device with a loaded weight
    /// `weight` (bf16, row-major `[out, in]`), bf16 input bits, f32 output. The first real
    /// op on this backend; it checks the weight's size against the requested shape.
    pub fn linear_bf16_f32(
        &self,
        weight: &str,
        x: &[u16],
        tokens: usize,
        out_features: usize,
        in_features: usize,
    ) -> Result<Vec<f32>> {
        if x.len() != tokens * in_features {
            return Err(anyhow!("input has {} elements, expected tokens*in = {}", x.len(), tokens * in_features));
        }
        let mut dev = self.device.lock().expect("device mutex poisoned");
        let (w_ptr, w_len) = self
            .weights
            .lock()
            .expect("weights mutex poisoned")
            .as_ref()
            .and_then(|w| w.pointer(weight))
            .ok_or_else(|| anyhow!("weight {weight} is not loaded"))?;
        if w_len != out_features * in_features * 2 {
            return Err(anyhow!("weight {weight} holds {w_len} bytes, [{out_features}, {in_features}] bf16 needs {}", out_features * in_features * 2));
        }
        let mut cublas = self.cublas.lock().expect("cublas mutex poisoned");
        if cublas.is_none() {
            *cublas = Some(Cublas::new(&dev)?);
        }

        let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let y_len = tokens * out_features * 4;
        let x_ptr = dev.alloc(x_bytes.len().max(1))?;
        let y_ptr = match dev.alloc(y_len.max(1)) {
            Ok(p) => p,
            Err(e) => {
                let _ = dev.free(x_ptr);
                return Err(e);
            }
        };
        let result = (|| -> Result<Vec<f32>> {
            dev.htod(x_ptr, &x_bytes)?;
            cublas.as_ref().expect("created above").linear_bf16_f32(w_ptr, x_ptr, y_ptr, tokens, out_features, in_features)?;
            dev.synchronize()?;
            let mut raw = vec![0u8; y_len];
            dev.dtoh(&mut raw, y_ptr)?;
            Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        })();
        let _ = dev.free(x_ptr);
        let _ = dev.free(y_ptr);
        result
    }

    /// Time the cuBLASLt heuristic's candidate algorithms for a one-token `[out, inn]` bf16 projection (with or without a bias) on zero-filled
    /// scratch buffers: microseconds per call for each candidate, in the heuristic's order. Changes nothing; see `scripts/bench-cublaslt.sh`.
    pub fn benchmark_cublaslt_candidates(&self, out: usize, inn: usize, has_bias: bool, runs: usize) -> Result<Vec<(String, Option<f64>)>> {
        const WORKSPACE: usize = 32 << 20;
        let mut dev = self.device.lock().expect("device mutex poisoned");
        let mut cublas = self.cublas.lock().expect("cublas mutex poisoned");
        if cublas.is_none() {
            *cublas = Some(Cublas::new(&dev)?);
        }
        let lt = cublas.as_ref().and_then(|c| c.lt()).ok_or_else(|| anyhow!("cuBLASLt is not available"))?;
        let sizes = [out * inn * 2, inn * 2, out * 2, out * 2, WORKSPACE];
        let mut buffers = Vec::new();
        for size in sizes {
            let ptr = dev.alloc(size)?;
            dev.htod(ptr, &vec![0u8; size])?;
            buffers.push(ptr);
        }
        lt.set_workspace(buffers[4], WORKSPACE);
        let result = lt.benchmark(&dev, buffers[0], buffers[1], buffers[2], if has_bias { buffers[3] } else { 0 }, 1, out, inn, runs);
        for ptr in buffers {
            let _ = dev.free(ptr);
        }
        result
    }

    /// RMSNorm of `x` (`[tokens, hidden]`, bf16 bits) with the loaded bf16 `weight`, run by
    /// the custom kernel in `kernels/rmsnorm.cu`; returns bf16 bits.
    pub fn rmsnorm_bf16(&self, weight: &str, x: &[u16], tokens: usize, hidden: usize, eps: f32) -> Result<Vec<u16>> {
        if x.len() != tokens * hidden {
            return Err(anyhow!("input has {} elements, expected tokens*hidden = {}", x.len(), tokens * hidden));
        }
        let mut dev = self.device.lock().expect("device mutex poisoned");
        let (w_ptr, w_len) = self
            .weights
            .lock()
            .expect("weights mutex poisoned")
            .as_ref()
            .and_then(|w| w.pointer(weight))
            .ok_or_else(|| anyhow!("weight {weight} is not loaded"))?;
        if w_len != hidden * 2 {
            return Err(anyhow!("weight {weight} holds {w_len} bytes, bf16 [{hidden}] needs {}", hidden * 2));
        }
        let kernel = self.kernel(&dev, "rmsnorm", RMSNORM_IMAGE, "rmsnorm_bf16")?;

        let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x_ptr = dev.alloc(bytes.len().max(1))?;
        let y_ptr = match dev.alloc(bytes.len().max(1)) {
            Ok(p) => p,
            Err(e) => {
                let _ = dev.free(x_ptr);
                return Err(e);
            }
        };
        let result = (|| -> Result<Vec<u16>> {
            dev.htod(x_ptr, &bytes)?;
            let (mut a, mut b, mut c) = (x_ptr, w_ptr, y_ptr);
            let (mut hidden_arg, mut eps_arg) = (i32::try_from(hidden)?, eps);
            let mut params: [*mut c_void; 5] = [
                &mut a as *mut u64 as *mut c_void,
                &mut b as *mut u64 as *mut c_void,
                &mut c as *mut u64 as *mut c_void,
                &mut hidden_arg as *mut i32 as *mut c_void,
                &mut eps_arg as *mut f32 as *mut c_void,
            ];
            // Safety: the parameters match `rmsnorm_bf16(x, w, y, hidden, eps)`.
            unsafe { kernel.launch(&dev, (u32::try_from(tokens)?, 1, 1), (ROW_BLOCK, 1, 1), ROW_BLOCK * 4, &mut params)? };
            dev.synchronize()?;
            let mut raw = vec![0u8; bytes.len()];
            dev.dtoh(&mut raw, y_ptr)?;
            Ok(raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
        })();
        let _ = dev.free(x_ptr);
        let _ = dev.free(y_ptr);
        result
    }

    /// Free and total device memory in bytes (`cuMemGetInfo`), for example right after the weights
    /// loaded: what is left is what a KV cache can use.
    pub fn device_memory(&self) -> Result<(u64, u64)> {
        self.device.lock().expect("device mutex poisoned").memory_info()
    }

    /// Number of tensors and total bytes currently on the device.
    pub fn loaded_weights(&self) -> (usize, usize) {
        let weights = self.weights.lock().expect("weights mutex poisoned");
        weights.as_ref().map_or((0, 0), |w| (w.len(), w.total_bytes()))
    }
}

impl ComputeBackend for HopperBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities::known("sm_90").expect("the sm_90 arch is known")
    }

    fn initialize_device(&self, _rank: &RankInfo) -> Result<HardwareInfo> {
        let dev = self.device.lock().expect("device mutex poisoned");
        Ok(HardwareInfo {
            compute_capability: dev.compute_capability_string(),
            total_vram_bytes: dev.total_vram_bytes(),
            free_vram_bytes: dev.memory_info()?.0,
            max_cuda_streams: dev.max_streams(),
        })
    }

    fn load_model(&self, config: &BackendConfig, weights: &WeightSource) -> Result<LoadedModel> {
        let files = &weights.files;
        let mut facts = Vec::with_capacity(files.len());
        for file in files {
            let mut handle = File::open(file).with_context(|| format!("open weights {}", file.display()))?;
            let file_len = handle.metadata()?.len();
            let data_start = weights_plan::data_start(&mut handle)?;
            facts.push(PayloadFile { data_start, file_len });
        }
        let plans = weights_plan::plan(config, &facts)?;

        // Everything the forward pass needs to know about the model is in the signed manifest; nothing is
        // read from files beside the package. A model the pass does not implement is a typed refusal.
        let allow_qk_norm = self.allow_qk_norm.load(std::sync::atomic::Ordering::Relaxed);
        let runtime = (Qwen2Dims::from_manifest_allowing(&config.model, allow_qk_norm || super::forward::QK_NORM_VERIFIED)?, config.eos_token_ids.clone());

        let mut dev = self.device.lock().expect("device mutex poisoned");
        // A model loaded before is released first (its session refers to its weights).
        if let Some(mut old) = self.session.lock().expect("session mutex poisoned").take() {
            old.free(&mut dev);
        }
        // The fused projections of a decode step (Q|K|V and gate|up) are read from one matrix each: the tensors are placed
        // back to back in one allocation as they are uploaded, so there is no second copy of any weight.
        let groups = fused_groups(&runtime.0);
        let uploaded = DeviceWeights::upload_grouped(&mut dev, files, &plans, &groups)?;
        {
            let mut slot = self.weights.lock().expect("weights mutex poisoned");
            if let Some(mut old) = slot.replace(uploaded) {
                let _ = old.free(&mut dev);
            }
        }
        // Everything a generation needs (buffers, K/V cache, kernels, cuBLAS workspace, decode graphs) is
        // built now, sized from the device memory left after the weights; serving builds nothing.
        let session = Session::build(self, &mut dev, runtime.0.clone(), config.model.max_seq_len as usize)?;
        *self.session.lock().expect("session mutex poisoned") = Some(session);
        *self.runtime.lock().expect("runtime mutex poisoned") = Some(runtime);
        Ok(LoadedModel::new(NAME, ()))
    }

    // The page-index KV pool, the graphs bound to it and the StepPlan path are the next change (W1.2); until then the
    // session built by `load_model` owns a contiguous KV cache and `generate` is the way to run a prompt.
    fn reserve_kv(&self, _model: &LoadedModel, _spec: &KvSpec) -> Result<KvReservation> {
        Err(HalError::Unsupported("reserve_kv on sm_90").into())
    }

    fn capture_graphs(&self, _model: &LoadedModel, _kv: &KvReservation) -> Result<GraphSet> {
        Err(HalError::Unsupported("capture_graphs on sm_90").into())
    }

    fn commit_kv(&self, _kv: KvReservation, _graphs: &GraphSet, _policy: &MemoryPolicy) -> Result<KvPool> {
        Err(HalError::Unsupported("commit_kv on sm_90").into())
    }

    fn submit_step(&self, _pool: &KvPool, _plan: &StepPlan) -> Result<StepTicket> {
        Err(HalError::Unsupported("submit_step on sm_90").into())
    }

    fn take_output(&self, _ticket: StepTicket) -> Result<StepOutput> {
        Err(HalError::Unsupported("take_output on sm_90").into())
    }

    fn generate(
        &self,
        prompt: &[u32],
        max_new: usize,
        sampling: &SamplingConfig,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<StopReason> {
        let (dims, eos) = self
            .runtime
            .lock()
            .expect("runtime mutex poisoned")
            .clone()
            .ok_or_else(|| anyhow!("no model loaded: call load_model first"))?;
        self.generate_streaming(&dims, prompt, max_new, &eos, sampling, on_token)
    }

    fn shutdown(&self) -> Result<()> {
        // The session (graphs, buffers), the cuBLAS handle and the weights first, then the context they live in.
        let mut dev = self.device.lock().expect("device mutex poisoned");
        if let Some(mut session) = self.session.lock().expect("session mutex poisoned").take() {
            session.free(&mut dev);
        }
        self.cublas.lock().expect("cublas mutex poisoned").take();
        self.modules.lock().expect("modules mutex poisoned").clear();
        if let Some(mut weights) = self.weights.lock().expect("weights mutex poisoned").take() {
            weights.free(&mut dev)?;
        }
        dev.teardown_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(qkv_bias: bool) -> Qwen2Dims {
        Qwen2Dims { layers: 2, hidden: 64, intermediate: 128, heads: 4, kv_heads: 2, head_dim: 16, vocab: 128, rope_theta: 1e4, rms_eps: 1e-6, tied_head: false, qkv_bias, qk_norm: false }
    }

    #[test]
    fn each_layer_fuses_qkv_and_gate_up_in_the_order_the_projections_read_them() {
        let groups = fused_groups(&dims(true));
        assert_eq!(groups.len(), 6, "weights, bias and gate/up for two layers");
        let first = &groups[0];
        assert_eq!(first.name, "layers.0.qkv.weight");
        assert_eq!(first.members, ["layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.v_proj.weight"]);
        assert_eq!(groups[1].members[2], "layers.0.self_attn.v_proj.bias");
        assert_eq!(groups[2].members, ["layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight"], "gate first, then up: the SwiGLU reads them in that order");
    }

    #[test]
    fn a_model_without_biases_has_no_bias_group() {
        let groups = fused_groups(&dims(false));
        assert_eq!(groups.len(), 4);
        assert!(groups.iter().all(|g| !g.name.ends_with("qkv.bias")));
    }
}
