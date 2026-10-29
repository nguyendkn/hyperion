//! Probing for a device when the CUDA driver may be absent. The driver is loaded at run time, so a
//! build never needs it; what matters is that a host without one gets a typed error, which the
//! backend factory reports and falls back from, instead of a crash.
#![cfg(feature = "cuda")]

use helios_hal::cuda::loader::LoadError;
use helios_hal::cuda::CudaDevice;

#[test]
fn probing_without_a_driver_gives_a_typed_error_and_with_one_succeeds_or_reports_cuda() {
    match CudaDevice::probe() {
        // A GPU host: nothing to check here (the hardware tests cover it).
        Ok(device) => drop(device),
        Err(error) => match error.downcast_ref::<LoadError>() {
            Some(LoadError::LibraryMissing { tried, .. }) => assert!(!tried.is_empty(), "the error must list the names tried"),
            Some(other) => panic!("a driver that loads must export every function the engine needs: {other}"),
            // A driver is installed but CUDA itself failed (no device, no permission): a CUDA error, not a load error.
            None => assert!(error.to_string().contains("CUDA"), "unexpected error: {error}"),
        },
    }
}
