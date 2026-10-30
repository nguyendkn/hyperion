//! NCCL, loaded at run time like the driver and cuBLAS (ADR 0009, decision 22: NCCL only): `libnccl.so.2` through `dlopen`, every function
//! hand-declared and resolved under its explicit name, no crate and no link-time dependency. A missing library or symbol is a typed
//! `LoadError`; a single-rank run never loads it. `nccl.dll` is not a target, so on Windows the load fails the same typed way.
//!
//! Communicators are created inside one `ncclGroupStart` / `ncclGroupEnd` bracket by the thread that owns the group, never as blocking
//! `ncclCommInitRank` calls from separate threads (a documented deadlock). Collectives are enqueued on the rank's stream, so a stream
//! capture records them. A hang ends in a typed timeout and a communicator abort, not a stuck process.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use thiserror::Error;

use super::device::CudaDevice;
use super::loader::{Library, LoadError};
use super::sys::{cuStreamQuery, CUDA_SUCCESS};
use crate::traits::{CommDType, Communicator};

#[cfg(windows)]
const NCCL_LIBRARIES: &[&str] = &["nccl.dll"];
#[cfg(not(windows))]
const NCCL_LIBRARIES: &[&str] = &["libnccl.so.2"];

type Comm = *mut c_void;
const SUCCESS: c_int = 0;
const IN_PROGRESS: c_int = 7;
const CUDA_ERROR_NOT_READY: c_int = 600;
const OP_SUM: c_int = 0;

/// `ncclUniqueId`: 128 opaque bytes made on rank 0 and sent to every other rank over the control plane.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct UniqueId {
    pub internal: [u8; 128],
}

fn data_type(dtype: CommDType) -> c_int {
    match dtype {
        CommDType::F16 => 6,
        CommDType::F32 => 7,
        CommDType::Bf16 => 9,
    }
}

type Send6 = unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, Comm, *mut c_void) -> c_int;

struct Api {
    get_version: unsafe extern "C" fn(*mut c_int) -> c_int,
    get_unique_id: unsafe extern "C" fn(*mut UniqueId) -> c_int,
    comm_init_rank: unsafe extern "C" fn(*mut Comm, c_int, UniqueId, c_int) -> c_int,
    comm_destroy: unsafe extern "C" fn(Comm) -> c_int,
    comm_abort: unsafe extern "C" fn(Comm) -> c_int,
    comm_async_error: unsafe extern "C" fn(Comm, *mut c_int) -> c_int,
    group_start: unsafe extern "C" fn() -> c_int,
    group_end: unsafe extern "C" fn() -> c_int,
    all_reduce: unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, c_int, Comm, *mut c_void) -> c_int,
    all_gather: Send6,
    reduce_scatter: unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, c_int, Comm, *mut c_void) -> c_int,
    send: unsafe extern "C" fn(*const c_void, usize, c_int, c_int, Comm, *mut c_void) -> c_int,
    recv: unsafe extern "C" fn(*mut c_void, usize, c_int, c_int, Comm, *mut c_void) -> c_int,
    error_string: unsafe extern "C" fn(c_int) -> *const c_char,
}

static API: OnceLock<Result<Api, LoadError>> = OnceLock::new();
static LIBRARY_NAME: OnceLock<String> = OnceLock::new();

/// The NCCL library that was loaded, once it has been.
pub fn loaded_library() -> Option<String> {
    LIBRARY_NAME.get().cloned()
}

fn api() -> Result<&'static Api, LoadError> {
    let loaded = API.get_or_init(|| {
        let library = Library::open(NCCL_LIBRARIES)?;
        let _ = LIBRARY_NAME.set(library.name().to_string());
        tracing::info!(library = library.name(), "NCCL library loaded");
        // Safety: each type is the exact C signature of the function named.
        unsafe {
            Ok(Api {
                get_version: library.symbol("ncclGetVersion")?,
                get_unique_id: library.symbol("ncclGetUniqueId")?,
                comm_init_rank: library.symbol("ncclCommInitRank")?,
                comm_destroy: library.symbol("ncclCommDestroy")?,
                comm_abort: library.symbol("ncclCommAbort")?,
                comm_async_error: library.symbol("ncclCommGetAsyncError")?,
                group_start: library.symbol("ncclGroupStart")?,
                group_end: library.symbol("ncclGroupEnd")?,
                all_reduce: library.symbol("ncclAllReduce")?,
                all_gather: library.symbol("ncclAllGather")?,
                reduce_scatter: library.symbol("ncclReduceScatter")?,
                send: library.symbol("ncclSend")?,
                recv: library.symbol("ncclRecv")?,
                error_string: library.symbol("ncclGetErrorString")?,
            })
        }
    });
    loaded.as_ref().map_err(Clone::clone)
}

/// Load the library and resolve every function now (a typed error when it is absent); returns the library name.
pub fn load() -> Result<String, LoadError> {
    api()?;
    Ok(loaded_library().unwrap_or_default())
}

fn check(api: &Api, call: &str, result: c_int) -> Result<()> {
    if result == SUCCESS {
        return Ok(());
    }
    let text = unsafe {
        let raw = (api.error_string)(result);
        if raw.is_null() { String::from("unknown error") } else { CStr::from_ptr(raw).to_string_lossy().into_owned() }
    };
    Err(anyhow!("NCCL {call} failed: {text} (code {result})"))
}

/// `(major, minor, patch)` of an `ncclGetVersion` code: `major * 10000 + minor * 100 + patch` from 2.9 on.
pub fn decode_version(code: i32) -> (u32, u32, u32) {
    let code = code.max(0) as u32;
    if code >= 20900 { (code / 10000, code / 100 % 100, code % 100) } else { (code / 1000, code / 100 % 10, code % 100) }
}

pub fn format_version(code: i32) -> String {
    let (major, minor, patch) = decode_version(code);
    format!("{major}.{minor}.{patch}")
}

/// The NCCL version of the loaded library.
pub fn version() -> Result<i32> {
    let api = api()?;
    let mut code: c_int = 0;
    check(api, "ncclGetVersion", unsafe { (api.get_version)(&mut code) })?;
    Ok(code)
}

/// What one rank reports at rendezvous.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankVersions {
    pub rank: u32,
    pub nccl: i32,
    pub driver: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RendezvousError {
    #[error("rendezvous refused: rank {rank} runs {what} {found} but rank 0 runs {expected}")]
    VersionMismatch { rank: u32, what: &'static str, expected: String, found: String },
}

/// The group is formed only when every rank runs rank 0's NCCL version and driver version.
pub fn check_versions(ranks: &[RankVersions]) -> Result<(), RendezvousError> {
    let Some(reference) = ranks.first() else { return Ok(()) };
    for r in ranks {
        if r.nccl != reference.nccl {
            return Err(RendezvousError::VersionMismatch { rank: r.rank, what: "NCCL", expected: format_version(reference.nccl), found: format_version(r.nccl) });
        }
        if r.driver != reference.driver {
            return Err(RendezvousError::VersionMismatch { rank: r.rank, what: "driver", expected: reference.driver.to_string(), found: r.driver.to_string() });
        }
    }
    Ok(())
}

/// Every `NCCL_*` variable set in this process, sorted: logged at startup by every rank and named in every multi-GPU result.
pub fn nccl_environment() -> Vec<(String, String)> {
    environment_from(std::env::vars())
}

fn environment_from(vars: impl Iterator<Item = (String, String)>) -> Vec<(String, String)> {
    let mut found: Vec<_> = vars.filter(|(name, _)| name.starts_with("NCCL_")).collect();
    found.sort();
    found
}

/// Differences between the environments of the ranks (against rank 0), one warning line each.
pub fn environment_differences(per_rank: &[(u32, Vec<(String, String)>)]) -> Vec<String> {
    let Some((_, reference)) = per_rank.first() else { return Vec::new() };
    let mut warnings = Vec::new();
    for (rank, vars) in per_rank.iter().skip(1) {
        for (name, value) in vars {
            match reference.iter().find(|(n, _)| n == name) {
                Some((_, v)) if v == value => {}
                Some((_, v)) => warnings.push(format!("rank {rank} sets {name}={value}, rank 0 sets {name}={v}")),
                None => warnings.push(format!("rank {rank} sets {name}={value}, rank 0 does not set it")),
            }
        }
        for (name, value) in reference {
            if !vars.iter().any(|(n, _)| n == name) {
                warnings.push(format!("rank 0 sets {name}={value}, rank {rank} does not set it"));
            }
        }
    }
    warnings
}

/// A hang or a failure of a collective, as a typed error.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CollectiveError {
    #[error("rank {rank}: a collective did not finish within {waited:?}; the communicator was aborted")]
    Timeout { rank: u32, waited: Duration },
    #[error("rank {rank}: NCCL reported error {code} on the communicator")]
    Async { rank: u32, code: i32 },
}

/// A fresh unique id (call on rank 0 and send it to the others).
pub fn unique_id() -> Result<UniqueId> {
    let api = api()?;
    let mut id = UniqueId { internal: [0; 128] };
    check(api, "ncclGetUniqueId", unsafe { (api.get_unique_id)(&mut id) })?;
    Ok(id)
}

/// One rank's communicator, bound to the rank's stream. Collectives are enqueued on that stream.
pub struct NcclComm {
    api: &'static Api,
    comm: Comm,
    stream: u64,
    rank: u32,
    world: u32,
}

// Safety: a communicator is used by the one thread that drives its rank; the handle is an opaque token.
unsafe impl Send for NcclComm {}
unsafe impl Sync for NcclComm {}

/// Create the communicators of all local ranks inside one group bracket (ADR 0009). `ranks` are `(global rank, device)`; the calling thread
/// makes each device current in turn, as NCCL requires for the grouped one-thread-many-devices form.
pub fn init_local_group(world: u32, id: UniqueId, ranks: &[(u32, &CudaDevice)]) -> Result<Vec<NcclComm>> {
    let api = api()?;
    let mut comms: Vec<Comm> = vec![std::ptr::null_mut(); ranks.len()];
    check(api, "ncclGroupStart", unsafe { (api.group_start)() })?;
    let mut failure = None;
    for (slot, (rank, device)) in ranks.iter().enumerate() {
        if let Err(e) = device.set_current() {
            failure = Some(e);
            break;
        }
        if let Err(e) = check(api, "ncclCommInitRank", unsafe { (api.comm_init_rank)(&mut comms[slot], world as c_int, id, *rank as c_int) }) {
            failure = Some(e);
            break;
        }
    }
    // The bracket is always closed, so a failed rank does not leave the calling thread inside a group.
    let ended = check(api, "ncclGroupEnd", unsafe { (api.group_end)() });
    if let Some(error) = failure.or(ended.err()) {
        for comm in comms.iter().filter(|c| !c.is_null()) {
            unsafe { (api.comm_abort)(*comm) };
        }
        return Err(error);
    }
    Ok(ranks.iter().zip(comms).map(|((rank, device), comm)| NcclComm { api, comm, stream: device.stream() as u64, rank: *rank, world }).collect())
}

impl NcclComm {
    /// Wait for the stream's enqueued work, up to `timeout`. A collective whose peer never arrives ends in
    /// [`CollectiveError::Timeout`] and the communicator is aborted, instead of a process that never returns.
    pub fn wait(&self, timeout: Duration) -> Result<()> {
        let started = Instant::now();
        loop {
            let status = unsafe { cuStreamQuery(self.stream as *mut c_void) };
            if status == CUDA_SUCCESS {
                return Ok(());
            }
            if status != CUDA_ERROR_NOT_READY {
                return Err(anyhow!("CUDA cuStreamQuery failed with code {status}"));
            }
            let mut async_error: c_int = SUCCESS;
            let polled = unsafe { (self.api.comm_async_error)(self.comm, &mut async_error) };
            if polled == SUCCESS && async_error != SUCCESS && async_error != IN_PROGRESS {
                unsafe { (self.api.comm_abort)(self.comm) };
                return Err(CollectiveError::Async { rank: self.rank, code: async_error }.into());
            }
            if started.elapsed() >= timeout {
                unsafe { (self.api.comm_abort)(self.comm) };
                return Err(CollectiveError::Timeout { rank: self.rank, waited: timeout }.into());
            }
            std::thread::sleep(Duration::from_micros(200));
        }
    }
}

impl Communicator for NcclComm {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn world_size(&self) -> u32 {
        self.world
    }

    fn all_reduce_sum(&self, buf: u64, count: usize, dtype: CommDType) -> Result<()> {
        let result = unsafe { (self.api.all_reduce)(buf as *const c_void, buf as *mut c_void, count, data_type(dtype), OP_SUM, self.comm, self.stream as *mut c_void) };
        check(self.api, "ncclAllReduce", result)
    }

    fn all_gather(&self, send: u64, recv: u64, count: usize, dtype: CommDType) -> Result<()> {
        let result = unsafe { (self.api.all_gather)(send as *const c_void, recv as *mut c_void, count, data_type(dtype), self.comm, self.stream as *mut c_void) };
        check(self.api, "ncclAllGather", result)
    }

    fn reduce_scatter_sum(&self, send: u64, recv: u64, count: usize, dtype: CommDType) -> Result<()> {
        let result = unsafe { (self.api.reduce_scatter)(send as *const c_void, recv as *mut c_void, count, data_type(dtype), OP_SUM, self.comm, self.stream as *mut c_void) };
        check(self.api, "ncclReduceScatter", result)
    }

    fn send(&self, buf: u64, count: usize, dtype: CommDType, peer: u32) -> Result<()> {
        check(self.api, "ncclSend", unsafe { (self.api.send)(buf as *const c_void, count, data_type(dtype), peer as c_int, self.comm, self.stream as *mut c_void) })
    }

    fn recv(&self, buf: u64, count: usize, dtype: CommDType, peer: u32) -> Result<()> {
        check(self.api, "ncclRecv", unsafe { (self.api.recv)(buf as *mut c_void, count, data_type(dtype), peer as c_int, self.comm, self.stream as *mut c_void) })
    }
}

impl Drop for NcclComm {
    fn drop(&mut self) {
        if !self.comm.is_null() {
            unsafe { (self.api.comm_destroy)(self.comm) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_library_is_loaded_or_the_failure_is_a_typed_load_error_naming_it() {
        match load() {
            Ok(name) => assert!(name.contains("nccl")),
            Err(LoadError::LibraryMissing { tried, .. }) => assert_eq!(tried, NCCL_LIBRARIES.iter().map(|s| s.to_string()).collect::<Vec<_>>()),
            Err(LoadError::SymbolMissing { library, symbol }) => assert!(symbol.starts_with("nccl"), "{library}: {symbol}"),
        }
    }

    #[test]
    fn a_collective_without_the_library_is_an_error_not_a_panic() {
        if load().is_err() {
            assert!(version().is_err());
            assert!(unique_id().is_err());
        }
    }

    #[test]
    fn version_codes_decode_in_both_encodings() {
        assert_eq!(decode_version(22105), (2, 21, 5));
        assert_eq!(decode_version(21903), (2, 19, 3));
        assert_eq!(decode_version(2805), (2, 8, 5), "before 2.9 the code was major * 1000 + minor * 100 + patch");
        assert_eq!(format_version(22105), "2.21.5");
    }

    #[test]
    fn rendezvous_refuses_a_rank_with_another_nccl_or_driver_version() {
        let same = |rank| RankVersions { rank, nccl: 22105, driver: 13000 };
        assert!(check_versions(&[same(0), same(1), same(2)]).is_ok());
        let old = RankVersions { rank: 2, nccl: 21903, driver: 13000 };
        let error = check_versions(&[same(0), same(1), old]).unwrap_err();
        assert_eq!(error, RendezvousError::VersionMismatch { rank: 2, what: "NCCL", expected: "2.21.5".into(), found: "2.19.3".into() });
        assert!(error.to_string().contains("rank 2") && error.to_string().contains("2.19.3") && error.to_string().contains("2.21.5"));
        let other_driver = RankVersions { rank: 1, nccl: 22105, driver: 12040 };
        assert!(matches!(check_versions(&[same(0), other_driver]), Err(RendezvousError::VersionMismatch { what: "driver", rank: 1, .. })));
    }

    #[test]
    fn only_nccl_variables_are_recorded_and_differences_between_ranks_are_named() {
        let env = |pairs: &[(&str, &str)]| environment_from(pairs.iter().map(|(a, b)| (a.to_string(), b.to_string())));
        let rank0 = env(&[("PATH", "/bin"), ("NCCL_DEBUG", "INFO"), ("NCCL_ALGO", "Ring")]);
        assert_eq!(rank0, vec![("NCCL_ALGO".to_string(), "Ring".to_string()), ("NCCL_DEBUG".to_string(), "INFO".to_string())]);
        let rank1 = env(&[("NCCL_DEBUG", "WARN"), ("NCCL_P2P_DISABLE", "1")]);
        let warnings = environment_differences(&[(0, rank0.clone()), (1, rank1)]);
        assert_eq!(warnings.len(), 3, "{warnings:?}");
        assert!(warnings.iter().any(|w| w.contains("NCCL_DEBUG=WARN") && w.contains("NCCL_DEBUG=INFO")));
        assert!(warnings.iter().any(|w| w.contains("NCCL_P2P_DISABLE") && w.contains("rank 0 does not set it")));
        assert!(warnings.iter().any(|w| w.contains("NCCL_ALGO=Ring") && w.contains("rank 1 does not set it")));
        assert!(environment_differences(&[(0, rank0.clone()), (1, rank0)]).is_empty());
    }
}
