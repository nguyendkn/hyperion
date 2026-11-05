//! The real rank group of a node (ADR 0004, ADR 0009): one persistent thread per GPU, each owning its `CudaDevice` (context and stream) and its NCCL
//! communicator. The communicators are created inside one `ncclGroupStart/End` bracket by the coordinating thread, then moved to their rank threads. A closure
//! sent to every rank runs on that rank's thread with that rank's context, the way the multi-rank stub's `StubGroup` does it without a GPU.
//!
//! Failure handling: a rank whose closure returns an error or panics is reported by index; a rank that does not answer within the call timeout is reported
//! as [`GroupError::Timeout`] (the peers of a collective that never completes end in `CollectiveError::Timeout` of the communicator, see `cuda/nccl.rs`).

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{anyhow, Result};
use thiserror::Error;

use crate::cuda::nccl::{self, NcclComm, RankVersions, RendezvousError};
use crate::cuda::CudaDevice;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GroupError {
    #[error("rank {rank} did not answer within {waited:?}")]
    Timeout { rank: u32, waited: Duration },
    #[error("rank {rank} panicked")]
    Panicked { rank: u32 },
    #[error("rank {rank} is gone (its thread ended)")]
    Gone { rank: u32 },
}

/// What one rank thread owns and what a closure sees.
pub struct RankCtx {
    pub rank: u32,
    pub world_size: u32,
    pub device: CudaDevice,
    pub comm: NcclComm,
}

/// The versions the group refuses to run without (`None` = whatever is installed). A version other than the installed one is a configuration the rendezvous
/// refuses before any communicator exists.
#[derive(Debug, Clone, Default)]
pub struct ExpectedVersions {
    pub nccl: Option<i32>,
    pub driver: Option<i32>,
}

type Job = Box<dyn FnOnce(&mut RankCtx) + Send>;

pub struct RankGroup {
    senders: Vec<Sender<Job>>,
    handles: Vec<JoinHandle<()>>,
    world: usize,
    nccl_version: i32,
}

impl RankGroup {
    /// Start one thread per GPU ordinal (`ordinals[r]` is the device of rank `r`). Checks the NCCL and driver versions against `expected` first (typed
    /// `RendezvousError` on a mismatch), creates the communicators in one group and hands each to its rank thread.
    pub fn start(ordinals: &[u32], expected: &ExpectedVersions) -> Result<Self> {
        let world = ordinals.len();
        if world < 2 {
            return Err(anyhow!("a rank group needs at least two ranks, got {world}"));
        }
        nccl::load()?;
        let installed = nccl::version()?;
        let (_, driver) = crate::cuda::sys::loaded_driver().unwrap_or_default();
        // Every rank runs in this process, so each reports what is installed; a configured expectation different from it is what the rendezvous refuses.
        if let Some(want) = expected.nccl.filter(|w| *w != installed) {
            return Err(RendezvousError::VersionMismatch { rank: 0, what: "NCCL", expected: nccl::format_version(want), found: nccl::format_version(installed) }.into());
        }
        if let Some(want) = expected.driver.filter(|w| *w != driver) {
            return Err(RendezvousError::VersionMismatch { rank: 0, what: "driver", expected: want.to_string(), found: driver.to_string() }.into());
        }
        let reports: Vec<RankVersions> = (0..world as u32).map(|rank| RankVersions { rank, nccl: installed, driver }).collect();
        nccl::check_versions(&reports)?;
        let devices: Vec<CudaDevice> = ordinals.iter().map(|&o| CudaDevice::probe_ordinal(o as i32)).collect::<Result<_>>()?;
        let id = nccl::unique_id()?;
        let refs: Vec<(u32, &CudaDevice)> = devices.iter().enumerate().map(|(r, d)| (r as u32, d)).collect();
        let comms = nccl::init_local_group(world as u32, id, &refs)?;
        drop(refs);
        let mut senders = Vec::new();
        let mut handles = Vec::new();
        for (rank, (device, comm)) in devices.into_iter().zip(comms).enumerate() {
            let (tx, rx): (Sender<Job>, Receiver<Job>) = mpsc::channel();
            senders.push(tx);
            handles.push(
                thread::Builder::new()
                    .name(format!("hyperion-rank-{rank}"))
                    .spawn(move || {
                        let mut ctx = RankCtx { rank: rank as u32, world_size: world as u32, device, comm };
                        let _ = ctx.device.set_current();
                        while let Ok(job) = rx.recv() {
                            job(&mut ctx);
                        }
                        // The channel closed: the communicator goes before its device.
                        drop(ctx.comm);
                        let _ = ctx.device.teardown_ref();
                    })?,
            );
        }
        Ok(Self { senders, handles, world, nccl_version: installed })
    }

    pub fn world(&self) -> usize {
        self.world
    }

    /// The installed NCCL version, as NCCL numbers it (`nccl::format_version` prints it).
    pub fn nccl_version(&self) -> i32 {
        self.nccl_version
    }

    /// Run `f` on every rank's thread and collect the answers in rank order.
    pub fn run_all<T: Send + 'static>(&self, timeout: Duration, f: impl Fn(&mut RankCtx) -> Result<T> + Send + Sync + 'static) -> Vec<Result<T>> {
        let f = Arc::new(f);
        let receivers: Vec<Receiver<std::thread::Result<Result<T>>>> = self
            .senders
            .iter()
            .map(|sender| {
                let (tx, rx) = mpsc::channel();
                let f = Arc::clone(&f);
                let job: Job = Box::new(move |ctx| {
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(ctx)));
                    let _ = tx.send(outcome);
                });
                let _ = sender.send(job);
                rx
            })
            .collect();
        receivers
            .into_iter()
            .enumerate()
            .map(|(rank, rx)| match rx.recv_timeout(timeout) {
                Ok(Ok(result)) => result,
                Ok(Err(_)) => Err(GroupError::Panicked { rank: rank as u32 }.into()),
                Err(mpsc::RecvTimeoutError::Timeout) => Err(GroupError::Timeout { rank: rank as u32, waited: timeout }.into()),
                Err(mpsc::RecvTimeoutError::Disconnected) => Err(GroupError::Gone { rank: rank as u32 }.into()),
            })
            .collect()
    }
}

impl Drop for RankGroup {
    fn drop(&mut self) {
        self.senders.clear();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}
