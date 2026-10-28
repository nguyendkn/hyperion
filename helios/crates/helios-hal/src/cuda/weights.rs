//! Upload model weights to the device and read them back. Real driver calls: on a
//! host without the CUDA driver this module does not link (see `device.rs`).
//!
//! Each tensor is read from the checkpoint file at the position computed by
//! `weights_plan`, copied into its own device allocation, and tracked by name.
//! `download` exists so a test can prove the bytes on the device equal the bytes
//! in the file; it is not on any serving path.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{anyhow, Context, Result};

use super::device::CudaDevice;
use crate::weights_plan::TensorPlan;

/// Device allocations for every tensor of one model.
#[derive(Default)]
pub struct DeviceWeights {
    tensors: HashMap<String, (u64, usize)>,
}

impl DeviceWeights {
    /// Copy every planned tensor from `file` to the device. If any step fails the
    /// allocations made so far are freed before the error is returned, so a failed
    /// load leaves no device memory behind.
    pub fn upload(dev: &mut CudaDevice, file: &Path, plans: &[TensorPlan]) -> Result<Self> {
        let mut source = File::open(file).with_context(|| format!("open {}", file.display()))?;
        let mut weights = Self::default();
        for plan in plans {
            if let Err(e) = weights.upload_one(dev, &mut source, plan) {
                let _ = weights.free(dev);
                return Err(e.context(format!("uploading tensor {}", plan.name)));
            }
        }
        Ok(weights)
    }

    fn upload_one(&mut self, dev: &mut CudaDevice, source: &mut File, plan: &TensorPlan) -> Result<()> {
        let len = usize::try_from(plan.len).map_err(|_| anyhow!("tensor too large for this platform"))?;
        let mut bytes = vec![0u8; len];
        source.seek(SeekFrom::Start(plan.file_offset))?;
        source.read_exact(&mut bytes).context("read tensor bytes")?;
        let ptr = dev.alloc(len.max(1))?;
        // Track the allocation first so a failed copy still frees it.
        self.tensors.insert(plan.name.clone(), (ptr, len));
        dev.htod(ptr, &bytes)
    }

    /// Copy a tensor back to the host (verification only).
    pub fn download(&self, dev: &CudaDevice, name: &str) -> Result<Vec<u8>> {
        let (ptr, len) = self.tensors.get(name).ok_or_else(|| anyhow!("no tensor named {name} on the device"))?;
        let mut out = vec![0u8; *len];
        dev.dtoh(&mut out, *ptr)?;
        Ok(out)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn total_bytes(&self) -> usize {
        self.tensors.values().map(|(_, len)| len).sum()
    }

    /// Release every device allocation.
    pub fn free(&mut self, dev: &mut CudaDevice) -> Result<()> {
        let mut first_error = None;
        for (_, (ptr, _)) in self.tensors.drain() {
            if let Err(e) = dev.free(ptr) {
                first_error.get_or_insert(e);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}
