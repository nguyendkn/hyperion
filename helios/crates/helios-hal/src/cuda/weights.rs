//! Upload model weights to the device and read them back. Real driver calls: on a
//! host without the CUDA driver this module does not link (see `device.rs`).
//!
//! Each tensor is read from the checkpoint file at the position computed by
//! `weights_plan`, copied into its own device allocation, and tracked by name.
//! Tensors named in a [`Group`] are placed one after another in a single allocation (the
//! concatenation a fused matrix multiply needs), each member being a view into it: nothing is
//! copied twice and no second copy of the weights exists.
//! `download` exists so a test can prove the bytes on the device equal the bytes
//! in the file; it is not on any serving path.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};

use super::device::CudaDevice;
use crate::weights_plan::TensorPlan;

/// Tensors to be laid out back to back in one allocation, in this order, under the name `name`.
/// Row-major `[out, in]` matrices with the same `in` stack along `out` by concatenating their bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    pub name: String,
    pub members: Vec<String>,
}

/// Device allocations for every tensor of one model.
#[derive(Default)]
pub struct DeviceWeights {
    tensors: HashMap<String, (u64, usize)>,
    /// The fused buffers, by group name.
    groups: HashMap<String, (u64, usize)>,
    /// Every allocation made, to free (a group's members are views into one of them).
    allocations: Vec<u64>,
}

impl DeviceWeights {
    /// Copy every planned tensor from its payload file (`files[plan.payload]`) to the device. If any
    /// step fails the allocations made so far are freed before the error is returned, so a failed
    /// load leaves no device memory behind.
    pub fn upload(dev: &mut CudaDevice, files: &[PathBuf], plans: &[TensorPlan]) -> Result<Self> {
        Self::upload_grouped(dev, files, plans, &[])
    }

    /// [`Self::upload`], with the tensors of each group placed contiguously in one allocation. Every member of a group
    /// must be a planned tensor.
    pub fn upload_grouped(dev: &mut CudaDevice, files: &[PathBuf], plans: &[TensorPlan], groups: &[Group]) -> Result<Self> {
        let mut weights = Self::default();
        let outcome = weights.upload_all(dev, files, plans, groups);
        if let Err(e) = outcome {
            let _ = weights.free(dev);
            return Err(e);
        }
        Ok(weights)
    }

    fn upload_all(&mut self, dev: &mut CudaDevice, files: &[PathBuf], plans: &[TensorPlan], groups: &[Group]) -> Result<()> {
        // Reserve each group's buffer and name the place of every member in it.
        let mut placed: HashMap<&str, (u64, usize)> = HashMap::new();
        for group in groups {
            let mut total = 0usize;
            let mut offsets = Vec::with_capacity(group.members.len());
            for member in &group.members {
                let plan = plans.iter().find(|p| &p.name == member).ok_or_else(|| anyhow!("group {} names tensor {member}, which is not in the package", group.name))?;
                let len = usize::try_from(plan.len).map_err(|_| anyhow!("tensor too large for this platform"))?;
                offsets.push(total);
                total += len;
            }
            let base = dev.alloc(total.max(1))?;
            self.allocations.push(base);
            self.groups.insert(group.name.clone(), (base, total));
            for (member, offset) in group.members.iter().zip(offsets) {
                placed.insert(member.as_str(), (base + offset as u64, offset));
            }
        }
        let mut sources = Vec::with_capacity(files.len());
        for file in files {
            sources.push(File::open(file).with_context(|| format!("open {}", file.display()))?);
        }
        for plan in plans {
            let Some(source) = sources.get_mut(plan.payload) else {
                return Err(anyhow!("tensor {} refers to payload {} but only {} file(s) are open", plan.name, plan.payload, files.len()));
            };
            let at = placed.get(plan.name.as_str()).map(|(ptr, _)| *ptr);
            self.upload_one(dev, source, plan, at).with_context(|| format!("uploading tensor {}", plan.name))?;
        }
        Ok(())
    }

    /// Copy one tensor to `at` (a place inside a group's buffer), or to an allocation of its own.
    fn upload_one(&mut self, dev: &mut CudaDevice, source: &mut File, plan: &TensorPlan, at: Option<u64>) -> Result<()> {
        let len = usize::try_from(plan.len).map_err(|_| anyhow!("tensor too large for this platform"))?;
        let mut bytes = vec![0u8; len];
        source.seek(SeekFrom::Start(plan.file_offset))?;
        source.read_exact(&mut bytes).context("read tensor bytes")?;
        let ptr = match at {
            Some(ptr) => ptr,
            None => {
                let ptr = dev.alloc(len.max(1))?;
                self.allocations.push(ptr);
                ptr
            }
        };
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

    /// Device pointer and byte length of a tensor, if loaded.
    pub fn pointer(&self, name: &str) -> Option<(u64, usize)> {
        self.tensors.get(name).copied()
    }

    /// Device pointer and byte length of a group's fused buffer.
    pub fn group(&self, name: &str) -> Option<(u64, usize)> {
        self.groups.get(name).copied()
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
        self.tensors.clear();
        self.groups.clear();
        for ptr in self.allocations.drain(..) {
            if let Err(e) = dev.free(ptr) {
                first_error.get_or_insert(e);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}
