//! The data types of the backend contract (ADR 0006): what a backend declares (`Capabilities`), what it is told (`BackendConfig`,
//! `WeightSource`, `RankInfo`, `KvSpec`, `MemoryPolicy`, `StepPlan`) and the tokens it hands back so that the order of the load sequence
//! is in the types (`LoadedModel`, `KvReservation`, `GraphSet`, `KvPool`, `StepTicket`). Plain data and opaque tokens: no CUDA here.
//!
//! The tokens carry an opaque backend-private payload. Only this crate can make them (the constructors are `pub(crate)`), so code outside
//! the HAL (`helios-core`, `helios-cli`) can obtain a [`KvPool`] only from `commit_kv` and a [`StepTicket`] only from `submit_step`: a
//! backend cannot be driven with an unmapped or half-mapped pool, and nothing maps memory after the pool exists.
//!
//! Outside this crate a pool cannot be made by hand, so a step cannot be submitted without going through the load order:
//!
//! ```compile_fail
//! use helios_hal::{KvPool, KvPoolInfo};
//! let info = KvPoolInfo { page_tokens: 64, pages: 1, bytes_per_page: 0, free_before: 0, free_after: 0 };
//! let _pool = KvPool::new("stub", info, ()); // `KvPool::new` is `pub(crate)`
//! ```
//!
//! and a reservation (virtual addresses only, nothing mapped) is not a pool:
//!
//! ```compile_fail
//! use helios_hal::{ComputeBackend, KvReservation, StepPlan};
//! fn serve(backend: &dyn ComputeBackend, reservation: &KvReservation, plan: &StepPlan) {
//!     let _ = backend.submit_step(reservation, plan); // expected `&KvPool`
//! }
//! ```

use std::any::Any;
use std::path::PathBuf;
use std::sync::Arc;

use hyperion_specification::v1::{Manifest, ModelConfig, ShardClass, TensorFormat};
use thiserror::Error;

/// Typed errors of the contract. Nothing in a backend panics for these.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HalError {
    /// The backend does not implement the operation (a stub, or a path not built yet).
    #[error("this backend does not support `{0}`")]
    Unsupported(&'static str),
    /// A token was handed to a backend that did not make it.
    #[error("a {0} from another backend was passed in")]
    ForeignToken(&'static str),
    /// A request the backend cannot honour as configured (a TP size the model does not allow, a page count the pool cannot hold).
    #[error("invalid request: {0}")]
    Invalid(String),
    /// A tensor-parallel size the model does not allow (ADR 0005). Raised before any CUDA call.
    #[error("tensor-parallel size {tp} is not supported by this model (supported: {supported:?})")]
    UnsupportedTp { tp: u32, supported: Vec<u32> },
    /// The KV pool has no free page for the request; the scheduler decides what to evict. Never answered by mapping more memory.
    #[error("the KV pool is full: {needed} pages needed, {free} free")]
    PoolFull { needed: usize, free: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    /// Discrete device memory.
    Device,
    /// Memory shared with the host.
    Unified,
    /// No device memory at all (the stub).
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectiveKind {
    AllReduce,
    AllGather,
    ReduceScatter,
    SendRecv,
}

/// What a backend can do, as data: the allowlists that used to live in `triage.rs` move here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    /// The compute capability string the backend is written for (`sm_90`).
    pub arch: &'static str,
    /// Tensor formats it can run.
    pub formats: Vec<TensorFormat>,
    pub memory_kind: MemoryKind,
    /// Head dimensions its attention kernels support.
    pub head_dims: Vec<u32>,
    /// How many CUDA graphs it can hold at once (0 = unlimited or none).
    pub max_graph_captures: u32,
    /// Whether peer-to-peer access between ranks is available.
    pub peer_access: bool,
    /// The collectives it can capture in a graph.
    pub collectives: Vec<CollectiveKind>,
}

impl Capabilities {
    /// The capabilities of the backend written for `arch`, as data (CLAUDE.md rules 6 and 7: Hopper has no FP4). `None` for an arch no
    /// backend is written for. A backend starts from this and adds what only it knows (the collectives it can capture).
    pub fn known(arch: &str) -> Option<Self> {
        let (arch, formats, memory_kind) = match arch {
            "sm_90" => ("sm_90", vec![TensorFormat::Bf16, TensorFormat::Fp8], MemoryKind::Device),
            "sm_120" => ("sm_120", vec![TensorFormat::Bf16, TensorFormat::Fp8, TensorFormat::Fp4], MemoryKind::Device),
            // The stub runs no kernels, so it does not constrain the format.
            "sm_0" => ("sm_0", vec![TensorFormat::Bf16, TensorFormat::Fp8, TensorFormat::Fp4], MemoryKind::None),
            _ => return None,
        };
        Some(Self { arch, formats, memory_kind, head_dims: vec![64, 128], max_graph_captures: 0, peer_access: false, collectives: Vec::new() })
    }

    /// Every arch with a backend, in a stable order (the committed format-table dump reads this).
    pub const KNOWN_ARCHS: [&'static str; 3] = ["sm_0", "sm_90", "sm_120"];
}

/// One rank of a device group: a GPU and the thread that drives it (ADR 0004).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankInfo {
    /// Global rank across the whole group.
    pub rank: u32,
    pub world_size: u32,
    /// The GPU ordinal on its node.
    pub ordinal: u32,
    pub node_rank: u32,
    pub numa_node: Option<u32>,
    /// The other ranks of the group.
    pub peers: Vec<u32>,
}

impl RankInfo {
    /// The only rank of a single-GPU run.
    pub fn single() -> Self {
        Self { rank: 0, world_size: 1, ordinal: 0, node_rank: 0, numa_node: None, peers: Vec::new() }
    }
}

/// The ranks of a deployment and how they are connected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceGroup {
    pub ranks: Vec<RankInfo>,
    /// `nvidia-smi topo -m` style text, for the log and the ledger.
    pub topology: String,
}

impl DeviceGroup {
    pub fn single() -> Self {
        Self { ranks: vec![RankInfo::single()], topology: String::new() }
    }

    pub fn world_size(&self) -> usize {
        self.ranks.len()
    }
}

/// How a model is spread over ranks, a load-time choice (ADR 0005, decision 20).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Deployment {
    pub tp_size: u32,
    pub pp_size: u32,
    /// This rank's index inside its tensor-parallel group.
    pub tp_rank: u32,
    /// This rank's pipeline stage.
    pub pp_stage: u32,
}

impl Deployment {
    pub const SINGLE: Deployment = Deployment { tp_size: 1, pp_size: 1, tp_rank: 0, pp_stage: 0 };
}

/// One tensor of the package as a backend needs to know it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    /// Index into [`BackendConfig::payload_files`].
    pub payload: usize,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// How the tensor may be split between tensor-parallel ranks; `None` means it is not split.
    pub sharding: Option<ShardClass>,
}

/// Everything a backend is told about the model, built from the manifest inside the HAL (`BackendConfig::from`): a backend never sees
/// the signed on-disk format.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendConfig {
    pub model_id: String,
    pub model: ModelConfig,
    pub tensor_format: TensorFormat,
    pub target_arch: String,
    /// Token ids that end a generation.
    pub eos_token_ids: Vec<u32>,
    /// Weights file names, in the order `TensorSpec::payload` indexes them (bare names, validated).
    pub payload_files: Vec<String>,
    /// The byte length the manifest declares for each payload file (`None`: a package without a payload table, whose one file has no declared size).
    pub payload_lengths: Vec<Option<u64>>,
    pub tensors: Vec<TensorSpec>,
    /// The tensor-parallel sizes the model supports (`[1]` when the manifest records none).
    pub supported_tp_sizes: Vec<u32>,
    pub deployment: Deployment,
}

impl BackendConfig {
    /// Build the config of a single-rank run from a manifest. Refuses a manifest whose payload table is invalid.
    pub fn from_manifest(manifest: &Manifest) -> Result<Self, HalError> {
        Self::for_deployment(manifest, Deployment::SINGLE)
    }

    /// As [`Self::from_manifest`], for one rank of a deployment. The deployment is checked against the model (a TP size the model does
    /// not allow is refused here, before any CUDA call: W1.5).
    pub fn for_deployment(manifest: &Manifest, deployment: Deployment) -> Result<Self, HalError> {
        // Payload names come from the signed manifest: refuse any that could leave the package directory before a path is built from them.
        manifest.validate_payloads().map_err(|e| HalError::Invalid(e.to_string()))?;
        let config = Self {
            model_id: manifest.model_id.clone(),
            model: manifest.model.clone(),
            tensor_format: manifest.tensor_format,
            target_arch: manifest.target_arch.clone(),
            eos_token_ids: manifest.generation.eos_token_ids.clone(),
            payload_files: manifest.payload_files().into_iter().map(str::to_string).collect(),
            payload_lengths: if manifest.payloads.is_empty() { vec![None] } else { manifest.payloads.iter().map(|p| Some(p.byte_length)).collect() },
            tensors: manifest
                .layers
                .iter()
                .map(|l| TensorSpec {
                    name: l.name.clone(),
                    payload: l.payload as usize,
                    byte_offset: l.byte_offset,
                    byte_length: l.byte_length,
                    dtype: l.tensor.dtype.clone(),
                    shape: l.tensor.shape.clone(),
                    sharding: l.sharding,
                })
                .collect(),
            supported_tp_sizes: manifest.parallelism.as_ref().map_or_else(|| vec![1], |p| p.supported_tp_sizes.clone()),
            deployment,
        };
        config.check_deployment()?;
        Ok(config)
    }

    /// Refuse a deployment the model does not allow, before any device is touched: a tensor-parallel size outside the supported list,
    /// a rank or stage outside its group, a pipeline deeper than the model.
    pub fn check_deployment(&self) -> Result<(), HalError> {
        let d = &self.deployment;
        if !self.supported_tp_sizes.contains(&d.tp_size) {
            return Err(HalError::UnsupportedTp { tp: d.tp_size, supported: self.supported_tp_sizes.clone() });
        }
        if d.tp_rank >= d.tp_size || d.pp_stage >= d.pp_size.max(1) || d.pp_size == 0 {
            return Err(HalError::Invalid(format!("rank {} of tp {} / stage {} of pp {} is outside its group", d.tp_rank, d.tp_size, d.pp_stage, d.pp_size)));
        }
        if d.pp_size > self.model.num_layers {
            return Err(HalError::Invalid(format!("pipeline of {} stages for a model of {} layers", d.pp_size, self.model.num_layers)));
        }
        Ok(())
    }
}

/// A rank's share of one tensor: `[start, end)` along `axis` (0 = rows/output, 1 = columns/input) of the canonical tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSlice {
    pub axis: usize,
    pub start: usize,
    pub end: usize,
}

/// Where the weights are and which part of each tensor this rank reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightSource {
    /// Absolute paths of the payload files, in the order of `BackendConfig::payload_files`.
    pub files: Vec<PathBuf>,
    /// Tensor name -> the slice this rank reads. Empty: every rank reads whole tensors (single GPU).
    pub slices: Vec<(String, TensorSlice)>,
}

impl WeightSource {
    /// The files of a package at `package` (the path of the `.hyperion` file; the payloads sit beside it, ADR 0003).
    pub fn from_package(package: &std::path::Path, config: &BackendConfig) -> Self {
        Self { files: config.payload_files.iter().map(|name| package.with_file_name(name)).collect(), slices: Vec::new() }
    }
}

/// What the KV pool is for: the geometry of one token's K and V and the page size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvSpec {
    /// Tokens per page (ADR 0002 default 64).
    pub page_tokens: usize,
    /// The longest sequence the pool must be able to hold at least (0: as many pages as fit).
    pub min_tokens: usize,
    /// The largest decode bucket (`--max-batch`, a power of two): a graph is captured for each of 1, 2, 4, ... up to it (A6).
    pub max_batch: usize,
}

impl Default for KvSpec {
    fn default() -> Self {
        Self { page_tokens: 64, min_tokens: 0, max_batch: 1 }
    }
}

/// How `commit_kv` sizes the pool from the memory that is free once the weights and graphs exist (ADR 0002).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPolicy {
    /// What stays free for the driver, cuBLAS and NCCL after the pool is mapped.
    pub reserve_bytes: usize,
    /// The model's maximum length caps the pool even when more memory is free (0: no cap).
    pub max_tokens: usize,
}

impl Default for MemoryPolicy {
    fn default() -> Self {
        Self { reserve_bytes: 2 << 30, max_tokens: 0 }
    }
}

macro_rules! token {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        pub struct $name {
            backend: &'static str,
            payload: Arc<dyn Any + Send + Sync>,
        }
        impl $name {
            pub(crate) fn new<T: Any + Send + Sync>(backend: &'static str, payload: T) -> Self {
                Self { backend, payload: Arc::new(payload) }
            }
            /// The backend that made this token.
            pub fn backend(&self) -> &'static str {
                self.backend
            }
            /// The payload, if `backend` made it and it is a `T`.
            pub(crate) fn payload<T: Any + Send + Sync>(&self, backend: &'static str) -> Result<&T, HalError> {
                if self.backend != backend {
                    return Err(HalError::ForeignToken(stringify!($name)));
                }
                self.payload.downcast_ref::<T>().ok_or(HalError::ForeignToken(stringify!($name)))
            }
        }
        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, concat!(stringify!($name), "({})"), self.backend)
            }
        }
    };
}

token!(
    /// The weights are on the device. Made by `load_model`.
    LoadedModel
);
token!(
    /// A virtual address range for the KV pool, with no physical memory behind it. Made by `reserve_kv`, consumed by `commit_kv`.
    KvReservation
);
token!(
    /// The captured graphs, bound to the reservation's addresses. Made by `capture_graphs`.
    GraphSet
);
token!(
    /// A submitted step. Made by `submit_step`, consumed by `take_output`.
    StepTicket
);

/// What the pool ended up holding, reported by the backend after `commit_kv`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPoolInfo {
    pub page_tokens: usize,
    pub pages: usize,
    pub bytes_per_page: usize,
    /// Free device memory measured before the pool was mapped, and what was left after.
    pub free_before: u64,
    pub free_after: u64,
}

/// The mapped KV pool: serving is possible, growth is not. Made only by `commit_kv` (the constructor is `pub(crate)`).
pub struct KvPool {
    backend: &'static str,
    info: KvPoolInfo,
    payload: Arc<dyn Any + Send + Sync>,
}

impl KvPool {
    pub(crate) fn new<T: Any + Send + Sync>(backend: &'static str, info: KvPoolInfo, payload: T) -> Self {
        Self { backend, info, payload: Arc::new(payload) }
    }

    pub fn backend(&self) -> &'static str {
        self.backend
    }

    /// Pages and sizes of the pool, for the scheduler.
    pub fn info(&self) -> &KvPoolInfo {
        &self.info
    }

    pub(crate) fn payload<T: Any + Send + Sync>(&self, backend: &'static str) -> Result<&T, HalError> {
        if self.backend != backend {
            return Err(HalError::ForeignToken("KvPool"));
        }
        self.payload.downcast_ref::<T>().ok_or(HalError::ForeignToken("KvPool"))
    }
}

impl std::fmt::Debug for KvPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KvPool({}, {} pages of {} tokens)", self.backend, self.info.pages, self.info.page_tokens)
    }
}
