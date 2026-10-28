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
//! - Forward compatibility is the rule: a package compiled for `sm_90` may run
//!   on `sm_120` (downgrade kernels), but a package compiled for `sm_120` is
//!   rejected on `sm_90` no matter what — see [`Triage`].
//! - No GPU code lives in this crate. Backend modules (`backends/*`) hold the
//!   FFI/kernel calls; here we only mediate.

pub mod traits;
pub mod triage;
pub mod backend_factory;
mod backends;

pub use backend_factory::{create_backend, BackendKind};
pub use traits::{ComputeBackend, DevicePointer, HardwareInfo, TensorData, DynBackend};
pub use triage::{Triage, TriageOutcome};
