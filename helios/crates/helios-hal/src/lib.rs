//! # helios-hal
//!
//! Hardware Abstraction Layer for the Helios runtime. Defines the one trait
//! every GPU backend must implement ([`ComputeBackend`]) and the runtime
//! factory that probes the real device and selects a concrete backend.
//!
//! ## Design invariants
//! - `helios-core` depends on this crate's *trait* only, never on a concrete
//!   backend. Backends are compiled behind feature flags and instantiated by
//!   the factory, so the core stays device-agnostic.
//! - Exact match is the rule: a package runs only on the arch it was compiled
//!   for, on the backend written for that arch, in a tensor format that
//!   backend supports. No cross-generation fallback — see [`triage`].
//! - No GPU code lives in this crate. Backend modules (`backends/*`) hold the
//!   FFI/kernel calls; here we only mediate.

pub mod contract;
pub mod multi_rank_comm;
pub mod pool_plan;
pub mod decode_buckets;
pub mod prefill_capacities;
pub mod multi_rank_stub;
pub mod sharding;
pub mod step;
pub mod step_ledger;
pub mod traits;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod oracle;
pub mod reference;
pub mod sampling;
pub mod weights_plan;
pub mod triage;
pub mod backend_factory;
mod backends;

#[cfg(feature = "sm90")]
pub use backends::hopper_sm90::{DecodeMode, DecodeProfile, DecodeTiming, HopperBackend, KvCache, Qwen2Dims, PrefillError, UnsupportedModel};
#[cfg(feature = "sm90")]
pub use backends::hopper_sm90::{LayoutBench, PageLayout};
pub use backend_factory::{create_backend, BackendKind, CreatedBackend};
pub use sampling::{sample, Rng, SamplingConfig};
pub use contract::{
    BackendConfig, Capabilities, CollectiveKind, Deployment, DeviceGroup, GraphSet, HalError, KvPool, KvPoolInfo, KvReservation, KvSpec, LoadedModel, MemoryKind,
    MemoryPolicy, RankInfo, StepTicket, TensorSlice, TensorSpec, WeightSource,
};
pub use step::{Capacity, Input, PageDelta, PlanError, Row, RowOutput, SamplingInit, SlotInit, StepOutput, StepPlan, Want};
pub use traits::{load_package, CommDType, Communicator, ComputeBackend, DevicePointer, DynBackend, HardwareInfo, StopReason};
pub use triage::{
    format_name, format_table, supported_formats, triage, ProbedBackend, Triage, TriageError,
    TriageOutcome,
};
