//! Backend implementations, feature-gated per compute capability.
//!
//! Adding a backend = one new file + one new feature flag + a match arm in
//! `backend_factory::instantiate`. Existing files are not touched — this is
//! the extension point the architecture doc promises ("add file, don't edit").

pub mod stub;

#[cfg(feature = "sm90")]
pub mod hopper_sm90;

#[cfg(feature = "sm120")]
pub mod blackwell_sm120;
