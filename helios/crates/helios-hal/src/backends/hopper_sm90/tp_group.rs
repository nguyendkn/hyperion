//! Tensor parallelism on one node (gate K3, steps T1-T5; ADRs 0004, 0005, 0006, 0007, 0009): one [`HopperBackend`] per GPU, each driven by its own
//! persistent thread, joined by NCCL communicators created in one group. The scheduler's `StepPlan` is encoded once and every rank decodes it (the plan hash
//! catches a corrupted copy) and applies it: the same page-table deltas, the same rows, the same sampling state. Each rank holds its slice of the weights and
//! its KV heads; the two all-reduces per layer and the all-gather of the vocab-parallel head run inside the rank's graphs.
//!
//! The ranks sample on the gathered logits independently and deterministically, so they stay in lockstep with no further communication; the output of rank
//! 0 is returned and the others' tokens are compared with it (a rank that disagrees is a typed error).

use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{anyhow, Result};
use hyperion_specification::v1::Manifest;
use thiserror::Error;

use super::backend::HopperBackend;
use super::forward_device::DecodeMode;
use crate::contract::{BackendConfig, Deployment, KvPool, WeightSource};
use crate::cuda::nccl::{self, RankVersions};
use crate::cuda::CudaDevice;
use crate::rank_group::ExpectedVersions;
use crate::sharding::rank_slices;
use crate::step::{StepOutput, StepPlan};
use crate::traits::{ComputeBackend, Communicator};

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TpError {
    #[error("rank {rank} did not answer within {waited:?}")]
    Timeout { rank: usize, waited: Duration },
    #[error("rank {rank} panicked")]
    Panicked { rank: usize },
    #[error("rank {rank} is gone (its thread ended)")]
    Gone { rank: usize },
    #[error("the ranks disagree: rank 0 produced {zero:?}, rank {rank} produced {other:?}")]
    Diverged { rank: usize, zero: Vec<Vec<u32>>, other: Vec<Vec<u32>> },
    #[error("the ranks mapped pools of different sizes ({pages:?} pages): every rank must hold the same number")]
    UnequalPools { pages: Vec<usize> },
}

struct RankState {
    rank: usize,
    backend: Arc<HopperBackend>,
    pool: Option<KvPool>,
}

type Job = Box<dyn FnOnce(&mut RankState) + Send>;

struct Worker {
    tx: Sender<Job>,
    handle: Option<JoinHandle<()>>,
    backend: Arc<HopperBackend>,
}

/// A tensor-parallel group of `size` GPUs of one node serving one model.
pub struct TpBackend {
    workers: Vec<Worker>,
    timeout: Duration,
}

impl TpBackend {
    /// One rank per entry of `ordinals` (GPU ordinals of the node). Checks the NCCL version against `expected`, creates the devices and the communicators
    /// (one group), and starts the rank threads.
    pub fn start(ordinals: &[u32], expected: &ExpectedVersions) -> Result<Self> {
        let size = ordinals.len();
        if size < 2 {
            return Err(anyhow!("tensor parallelism needs at least two ranks, got {size}"));
        }
        nccl::load()?;
        let installed = nccl::version()?;
        let (_, driver) = crate::cuda::sys::loaded_driver().unwrap_or_default();
        if let Some(want) = expected.nccl.filter(|w| *w != installed) {
            return Err(nccl::RendezvousError::VersionMismatch { rank: 0, what: "NCCL", expected: nccl::format_version(want), found: nccl::format_version(installed) }.into());
        }
        let reports: Vec<RankVersions> = (0..size as u32).map(|rank| RankVersions { rank, nccl: installed, driver }).collect();
        nccl::check_versions(&reports)?;
        let backends: Vec<Arc<HopperBackend>> = ordinals.iter().map(|&o| Ok(Arc::new(HopperBackend::with_device(CudaDevice::probe_ordinal(o as i32)?)))).collect::<Result<_>>()?;
        let id = nccl::unique_id()?;
        let comms = {
            let guards: Vec<_> = backends.iter().map(|b| b.device.lock().expect("device mutex poisoned")).collect();
            let refs: Vec<(u32, &CudaDevice)> = guards.iter().enumerate().map(|(r, g)| (r as u32, &**g)).collect();
            nccl::init_local_group(size as u32, id, &refs)?
        };
        let mut workers = Vec::new();
        for (rank, (backend, comm)) in backends.into_iter().zip(comms).enumerate() {
            let comm: Arc<dyn Communicator> = Arc::new(comm);
            backend.set_tensor_parallel(size, rank, comm);
            let (tx, rx): (Sender<Job>, Receiver<Job>) = mpsc::channel();
            let state_backend = Arc::clone(&backend);
            let handle = thread::Builder::new().name(format!("hyperion-tp-rank-{rank}")).spawn(move || {
                let mut state = RankState { rank, backend: state_backend, pool: None };
                // This thread drives this GPU: its context is current here for every driver call the rank makes (an allocation does not select it itself).
                let _ = state.backend.device.lock().expect("device mutex poisoned").set_current();
                while let Ok(job) = rx.recv() {
                    job(&mut state);
                }
                // The pool goes before the device it maps memory on.
                state.pool = None;
                let _ = state.backend.shutdown();
            })?;
            workers.push(Worker { tx, handle: Some(handle), backend });
        }
        Ok(Self { workers, timeout: Duration::from_secs(900) })
    }

    pub fn size(&self) -> usize {
        self.workers.len()
    }

    /// The backend of rank `rank` (for inspection by tests: weight bytes, session facts).
    pub fn rank_backend(&self, rank: usize) -> &HopperBackend {
        &self.workers[rank].backend
    }

    /// Run `f` on every rank's thread at once and collect the answers in rank order.
    pub fn run_all<T: Send + 'static>(&self, f: impl Fn(&mut RankStateView) -> Result<T> + Send + Sync + 'static) -> Vec<Result<T>> {
        let f = Arc::new(f);
        let receivers: Vec<Receiver<thread::Result<Result<T>>>> = self
            .workers
            .iter()
            .map(|worker| {
                let (tx, rx) = mpsc::channel();
                let f = Arc::clone(&f);
                let job: Job = Box::new(move |state| {
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut view = RankStateView { rank: state.rank, backend: &state.backend, pool: &mut state.pool };
                        f(&mut view)
                    }));
                    let _ = tx.send(outcome);
                });
                let _ = worker.tx.send(job);
                rx
            })
            .collect();
        receivers
            .into_iter()
            .enumerate()
            .map(|(rank, rx)| match rx.recv_timeout(self.timeout) {
                Ok(Ok(result)) => result,
                Ok(Err(_)) => Err(TpError::Panicked { rank }.into()),
                Err(mpsc::RecvTimeoutError::Timeout) => Err(TpError::Timeout { rank, waited: self.timeout }.into()),
                Err(mpsc::RecvTimeoutError::Disconnected) => Err(TpError::Gone { rank }.into()),
            })
            .collect()
    }

    fn all_ok<T>(results: Vec<Result<T>>) -> Result<Vec<T>> {
        let mut out = Vec::new();
        for (rank, r) in results.into_iter().enumerate() {
            out.push(r.map_err(|e| e.context(format!("rank {rank}")))?);
        }
        Ok(out)
    }

    /// Load the package's weights, every rank reading its slices (T2). Returns the bytes each rank uploaded.
    pub fn load_package(&self, package: &Path, manifest: &Manifest) -> Result<Vec<usize>> {
        let size = self.size() as u32;
        let (package, manifest) = (package.to_path_buf(), Arc::new(manifest.clone()));
        let answers = self.run_all(move |rank| {
            let config = BackendConfig::for_deployment(&manifest, Deployment { tp_size: size, pp_size: 1, tp_rank: rank.rank as u32, pp_stage: 0 })?;
            let weights = WeightSource { files: config.payload_files.iter().map(|name| package.with_file_name(name)).collect(), slices: rank_slices(&config)? };
            let started = std::time::Instant::now();
            rank.backend.load_model(&config, &weights)?;
            let bytes = rank.backend.weight_bytes();
            tracing::info!(rank = rank.rank, weight_mib = bytes as f64 / (1u64 << 20) as f64, load_ms = started.elapsed().as_millis() as u64, "rank loaded its weight slices");
            Ok(bytes)
        });
        Self::all_ok(answers)
    }

    /// Reserve, capture (every rank at once), warm the collectives and map the pools; the pools must hold the same number of pages on every rank (T5).
    /// Returns the pages of the pool of each rank.
    pub fn open_kv(&self, max_batch: usize, max_tokens: usize) -> Result<Vec<usize>> {
        let answers = self.run_all(move |rank| {
            let pool = rank.backend.open_kv_serving(max_batch, max_tokens)?;
            let pages = pool.info().pages;
            *rank.pool = Some(pool);
            Ok(pages)
        });
        let pages = Self::all_ok(answers)?;
        if pages.iter().any(|p| *p != pages[0]) {
            return Err(TpError::UnequalPools { pages }.into());
        }
        Ok(pages)
    }

    /// Pages of the scheduler's pool (the same on every rank).
    pub fn pool_pages(&self) -> Result<usize> {
        let answers = self.run_all(|rank| Ok(rank.pool.as_ref().ok_or_else(|| anyhow!("no pool: open_kv first"))?.info().pages));
        Ok(Self::all_ok(answers)?[0])
    }

    /// One step: the plan is encoded once, every rank decodes (hash-checked) and applies it, graph or direct launches; rank 0's output is returned.
    pub fn submit_step(&self, plan: &StepPlan, mode: DecodeMode) -> Result<StepOutput> {
        let bytes = Arc::new(plan.encode());
        let answers = self.run_all(move |rank| {
            let plan = StepPlan::decode(&bytes).map_err(|e| anyhow!("rank {} refused the plan: {e}", rank.rank))?;
            let pool = rank.pool.as_ref().ok_or_else(|| anyhow!("no pool: open_kv first"))?;
            let ticket = rank.backend.submit_step_mode(pool, &plan, mode)?;
            rank.backend.take_output(ticket)
        });
        let outputs = Self::all_ok(answers)?;
        let zero: Vec<Vec<u32>> = outputs[0].rows.iter().map(|r| r.tokens.clone()).collect();
        for (rank, out) in outputs.iter().enumerate().skip(1) {
            let other: Vec<Vec<u32>> = out.rows.iter().map(|r| r.tokens.clone()).collect();
            if other != zero {
                return Err(TpError::Diverged { rank, zero, other }.into());
            }
        }
        Ok(outputs.into_iter().next().expect("at least two ranks"))
    }
}

/// What a job sees of its rank.
pub struct RankStateView<'a> {
    pub rank: usize,
    pub backend: &'a HopperBackend,
    pub pool: &'a mut Option<KvPool>,
}

impl Drop for TpBackend {
    fn drop(&mut self) {
        for worker in &mut self.workers {
            // Dropping the sender ends the thread's loop; the pool and the backend are released there.
            let (dead, _) = mpsc::channel();
            worker.tx = dead;
        }
        for worker in &mut self.workers {
            if let Some(handle) = worker.handle.take() {
                let _ = handle.join();
            }
        }
    }
}
