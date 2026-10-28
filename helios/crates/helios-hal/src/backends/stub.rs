//! Stub backend — implements [`ComputeBackend`] without touching a GPU.
//! Lets the workspace typecheck and run core unit tests on hosts without CUDA.
//! Reports `sm_0` so triage against a real package's `target_arch` fails
//! loudly (`UpgradeRequired`) rather than silently pretending to work.

use std::path::Path;

use anyhow::Result;

use crate::traits::{
    ComputeBackend, DevicePointer, HardwareInfo, TensorData,
};

#[derive(Default)]
pub struct StubBackend;

impl StubBackend {
    pub fn new() -> Self {
        Self
    }
}

impl ComputeBackend for StubBackend {
    fn initialize_device(&self) -> Result<HardwareInfo> {
        Ok(HardwareInfo {
            compute_capability: "sm_0".into(),
            total_vram_bytes: 0,
            max_cuda_streams: 0,
        })
    }

    fn load_model_weights(&self, _model_path: &Path, _manifest: &hyperion_specification::v1::Manifest) -> Result<()> {
        Ok(())
    }

    fn allocate_kv_cache(
        &self,
        num_blocks: usize,
        _block_size_tokens: usize,
    ) -> Result<Vec<DevicePointer>> {
        Ok(vec![DevicePointer { address: 0, byte_size: 0 }; num_blocks])
    }

    fn execute_prefill(
        &self,
        input_tokens: &[u32],
        _cache_pointers: &[DevicePointer],
    ) -> Result<TensorData> {
        Ok(TensorData {
            dtype: "bf16".into(),
            shape: vec![input_tokens.len()],
            data_ptr: DevicePointer { address: 0, byte_size: 0 },
        })
    }

    fn execute_decode_step(&self, _current_step: usize, _active_slots: &[u32]) -> Result<u32> {
        Ok(0)
    }
}
