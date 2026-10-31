//! The multi-rank stub's `Communicator` (ADR 0006): the same trait the NCCL communicator implements, over host memory. A buffer is an opaque
//! `u64` handle issued by [`StubComm::alloc`] (a device address for NCCL, an arena handle here), so the trait needs no change. Only `F32`
//! buffers are supported; the other element types are refused with `HalError::Unsupported`. A named test double: `[RUN on stub]`, never
//! evidence about a GPU.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::contract::HalError;
use crate::multi_rank_stub::{GroupError, Shared};
use crate::traits::{CommDType, Communicator};

pub struct StubComm {
    rank: u32,
    shared: Arc<Shared>,
    buffers: Mutex<HashMap<u64, Vec<f32>>>,
    next: AtomicU64,
}

impl StubComm {
    pub(crate) fn new(rank: u32, shared: Arc<Shared>) -> Self {
        Self { rank, shared, buffers: Mutex::new(HashMap::new()), next: AtomicU64::new(1) }
    }

    /// A buffer holding `data`; the handle is what the collectives take.
    pub fn alloc(&self, data: Vec<f32>) -> u64 {
        let handle = self.next.fetch_add(1, Ordering::Relaxed);
        self.buffers.lock().unwrap().insert(handle, data);
        handle
    }

    pub fn read(&self, handle: u64) -> Result<Vec<f32>> {
        self.buffers.lock().unwrap().get(&handle).cloned().ok_or_else(|| HalError::Invalid(format!("no buffer {handle}")).into())
    }

    fn take(&self, handle: u64, count: usize) -> Result<Vec<f32>> {
        let data = self.read(handle)?;
        if data.len() < count {
            return Err(HalError::Invalid(format!("buffer {handle} holds {} elements, {count} asked", data.len())).into());
        }
        Ok(data[..count].to_vec())
    }

    fn put(&self, handle: u64, values: &[f32]) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        let data = buffers.get_mut(&handle).ok_or_else(|| HalError::Invalid(format!("no buffer {handle}")))?;
        if data.len() < values.len() {
            return Err(HalError::Invalid(format!("buffer {handle} holds {} elements, {} written", data.len(), values.len())).into());
        }
        data[..values.len()].copy_from_slice(values);
        Ok(())
    }
}

fn f32_only(dtype: CommDType) -> Result<()> {
    if dtype == CommDType::F32 {
        Ok(())
    } else {
        Err(HalError::Unsupported("the multi-rank stub moves F32 buffers only").into())
    }
}

impl Communicator for StubComm {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn world_size(&self) -> u32 {
        self.shared.world as u32
    }

    fn all_reduce_sum(&self, buf: u64, count: usize, dtype: CommDType) -> Result<()> {
        f32_only(dtype)?;
        let all = self.shared.exchange(self.rank as usize, self.take(buf, count)?)?;
        if all.iter().any(|v| v.len() != count) {
            return Err(GroupError::LengthMismatch.into());
        }
        let sum: Vec<f32> = (0..count).map(|i| all.iter().map(|v| v[i]).sum()).collect();
        self.put(buf, &sum)
    }

    fn all_gather(&self, send: u64, recv: u64, count: usize, dtype: CommDType) -> Result<()> {
        f32_only(dtype)?;
        let all = self.shared.exchange(self.rank as usize, self.take(send, count)?)?;
        if all.iter().any(|v| v.len() != count) {
            return Err(GroupError::LengthMismatch.into());
        }
        self.put(recv, &all.iter().flatten().copied().collect::<Vec<_>>())
    }

    fn reduce_scatter_sum(&self, send: u64, recv: u64, count: usize, dtype: CommDType) -> Result<()> {
        f32_only(dtype)?;
        let world = self.shared.world;
        let all = self.shared.exchange(self.rank as usize, self.take(send, count * world)?)?;
        if all.iter().any(|v| v.len() != count * world) {
            return Err(GroupError::LengthMismatch.into());
        }
        let mine = self.rank as usize * count;
        let sum: Vec<f32> = (mine..mine + count).map(|i| all.iter().map(|v| v[i]).sum()).collect();
        self.put(recv, &sum)
    }

    fn send(&self, buf: u64, count: usize, dtype: CommDType, peer: u32) -> Result<()> {
        f32_only(dtype)?;
        Ok(self.shared.post(self.rank, peer, self.take(buf, count)?)?)
    }

    fn recv(&self, buf: u64, count: usize, dtype: CommDType, peer: u32) -> Result<()> {
        f32_only(dtype)?;
        let data = self.shared.take(peer, self.rank)?;
        if data.len() != count {
            return Err(GroupError::LengthMismatch.into());
        }
        self.put(buf, &data)
    }
}
