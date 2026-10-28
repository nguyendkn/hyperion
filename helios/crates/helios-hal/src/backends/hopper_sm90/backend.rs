//! The `HopperBackend` struct + its `ComputeBackend` impl.

use std::path::Path;

use anyhow::{anyhow, Result};

use crate::traits::{ComputeBackend, DevicePointer, HardwareInfo, TensorData};

pub struct HopperBackend {
    // Owned CUDA context handle once FFI is wired.
    _context: (),
}

impl HopperBackend {
    pub fn new() -> Result<Self> {
        // TODO(build.rs): link TensorRT-LLM + CUDA Driver, generate bindings,
        // allocate a primary stream + context here. For now we construct so
        // the factory compiles.
        Ok(Self { _context: () })
    }
}

impl ComputeBackend for HopperBackend {
    fn initialize_device(&self) -> Result<HardwareInfo> {
        // No probe is wired yet. Reporting a hard-coded "sm_90" would claim an
        // arch nobody measured, so refuse until a real device query exists
        // (the factory skips this backend and reports why).
        Err(anyhow!("sm90 backend cannot probe a device: CUDA/TensorRT-LLM FFI not wired"))
    }

    fn load_model_weights(
        &self,
        _model_path: &Path,
        _manifest: &hyperion_specification::v1::Manifest,
    ) -> Result<()> {
        Err(anyhow!("sm90 backend not wired: build.rs FFI linkage pending"))
    }

    fn allocate_kv_cache(
        &self,
        _num_blocks: usize,
        _block_size_tokens: usize,
    ) -> Result<Vec<DevicePointer>> {
        Err(anyhow!("sm90 backend not wired"))
    }

    fn execute_prefill(
        &self,
        _input_tokens: &[u32],
        _cache_pointers: &[DevicePointer],
    ) -> Result<TensorData> {
        Err(anyhow!("sm90 backend not wired"))
    }

    fn execute_decode_step(&self, _current_step: usize, _active_slots: &[u32]) -> Result<u32> {
        Err(anyhow!("sm90 backend not wired"))
    }
}
