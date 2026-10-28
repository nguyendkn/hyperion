//! The `HopperBackend` struct + its `ComputeBackend` impl.
//!
//! Built on the shared CUDA Driver-API core (ADR 0001). Today it can probe a
//! real device and upload a model's weights; kernels, the KV cache and the
//! forward pass are not written yet and return a clear "not wired" error, so
//! the engine can never reach `Serving` on this backend by accident.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};

use std::ffi::c_void;

use crate::cuda::{Cublas, CudaDevice, DeviceWeights, KernelModule};
use crate::traits::{ComputeBackend, DevicePointer, HardwareInfo, TensorData};
use crate::weights_plan;

/// PTX generated from `kernels/rmsnorm.cu` (rebuild with `scripts/build-ptx.sh`).
const RMSNORM_PTX: &str = include_str!("../../../kernels/rmsnorm.ptx");

/// Threads per block for the row-wise kernels; a power of two, as the reduction needs.
const ROW_BLOCK: u32 = 256;

/// Name of the checkpoint file next to the package, as written by Ignition's
/// source model (the manifest's layer offsets index its data section).
const WEIGHTS_FILE: &str = "model.safetensors";

/// Field order is drop order: cuBLAS handle first, then the weights' owner, then the
/// device (which releases the context they all live in).
pub struct HopperBackend {
    pub(super) cublas: Mutex<Option<Cublas>>,
    pub(super) modules: Mutex<HashMap<&'static str, KernelModule>>,
    pub(super) weights: Mutex<Option<DeviceWeights>>,
    pub(super) device: Mutex<CudaDevice>,
}

impl HopperBackend {
    /// Probe a real CUDA device. Fails when the driver or a device is absent, so
    /// the factory skips this backend on hosts without a GPU.
    pub fn new() -> Result<Self> {
        Ok(Self { cublas: Mutex::new(None), modules: Mutex::new(HashMap::new()), weights: Mutex::new(None), device: Mutex::new(CudaDevice::probe()?) })
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
        let kernel = self.kernel(&dev, "rmsnorm", RMSNORM_PTX, "rmsnorm_bf16")?;

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

    /// Number of tensors and total bytes currently on the device.
    pub fn loaded_weights(&self) -> (usize, usize) {
        let weights = self.weights.lock().expect("weights mutex poisoned");
        weights.as_ref().map_or((0, 0), |w| (w.len(), w.total_bytes()))
    }
}

impl ComputeBackend for HopperBackend {
    fn initialize_device(&self) -> Result<HardwareInfo> {
        let dev = self.device.lock().expect("device mutex poisoned");
        Ok(HardwareInfo {
            compute_capability: dev.compute_capability_string(),
            total_vram_bytes: dev.total_vram_bytes(),
            max_cuda_streams: dev.max_streams(),
        })
    }

    fn load_model_weights(
        &self,
        model_path: &Path,
        manifest: &hyperion_specification::v1::Manifest,
    ) -> Result<()> {
        let file = model_path.with_file_name(WEIGHTS_FILE);
        let mut handle = File::open(&file).with_context(|| format!("open weights {}", file.display()))?;
        let file_len = handle.metadata()?.len();
        let data_start = weights_plan::data_start(&mut handle)?;
        let plans = weights_plan::plan(manifest, data_start, file_len)?;

        let mut dev = self.device.lock().expect("device mutex poisoned");
        let uploaded = DeviceWeights::upload(&mut dev, &file, &plans)?;
        let mut slot = self.weights.lock().expect("weights mutex poisoned");
        if let Some(mut old) = slot.replace(uploaded) {
            let _ = old.free(&mut dev);
        }
        Ok(())
    }

    fn allocate_kv_cache(
        &self,
        _num_blocks: usize,
        _block_size_tokens: usize,
    ) -> Result<Vec<DevicePointer>> {
        Err(anyhow!("sm90 backend: KV cache not wired yet"))
    }

    fn execute_prefill(
        &self,
        _input_tokens: &[u32],
        _cache_pointers: &[DevicePointer],
    ) -> Result<TensorData> {
        Err(anyhow!("sm90 backend: forward pass not wired yet"))
    }

    fn execute_decode_step(&self, _current_step: usize, _active_slots: &[u32]) -> Result<u32> {
        Err(anyhow!("sm90 backend: forward pass not wired yet"))
    }

    fn shutdown(&self) -> Result<()> {
        // cuBLAS handle and weights first, then the context they live in.
        self.cublas.lock().expect("cublas mutex poisoned").take();
        self.modules.lock().expect("modules mutex poisoned").clear();
        let mut dev = self.device.lock().expect("device mutex poisoned");
        if let Some(mut weights) = self.weights.lock().expect("weights mutex poisoned").take() {
            weights.free(&mut dev)?;
        }
        dev.teardown_ref()
    }
}
