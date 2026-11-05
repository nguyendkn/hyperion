//! Hardware tests of the real rank group (gate K3, step T1): one thread and one CUDA context per GPU, communicators created inside one group, a closure run
//! on every rank, version checks refused by configuration, and a failing rank reported by index.
//!
//! ```text
//! cargo test --release -p helios-hal --features sm90 --test rank_group_hw -- --ignored --nocapture --test-threads=1
//! ```
#![cfg(feature = "sm90")]

use std::time::Duration;

use helios_hal::cuda::nccl::{self, RendezvousError};
use helios_hal::rank_group::{ExpectedVersions, RankGroup};
use helios_hal::{CommDType, Communicator};

const CALL: Duration = Duration::from_secs(60);

#[test]
#[ignore = "needs two GPUs on one node and libnccl.so.2"]
fn each_rank_runs_on_its_own_thread_and_device_and_an_all_reduce_across_them_is_exact() {
    let group = RankGroup::start(&[0, 1], &ExpectedVersions::default()).expect("group of two GPUs");
    println!("rank group of {} ranks, NCCL {}", group.world(), nccl::format_version(group.nccl_version()));
    let answers = group.run_all(CALL, |ctx| {
        let thread = std::thread::current().name().unwrap_or("?").to_string();
        let count = 4096usize;
        let mine: Vec<u8> = (0..count).flat_map(|i| (((i % 17) as f32) + ctx.rank as f32).to_le_bytes()).collect();
        let buf = ctx.device.alloc(count * 4)?;
        ctx.device.htod(buf, &mine)?;
        ctx.comm.all_reduce_sum(buf, count, CommDType::F32)?;
        ctx.comm.wait(CALL)?;
        let mut out = vec![0u8; count * 4];
        ctx.device.dtoh(&mut out, buf)?;
        ctx.device.free(buf)?;
        let got: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let want: Vec<f32> = (0..count).map(|i| 2.0 * ((i % 17) as f32) + 1.0).collect();
        assert_eq!(got, want, "rank {}: the all-reduce of the two ranks' data", ctx.rank);
        Ok((thread, ctx.device.ordinal(), ctx.rank))
    });
    let answers: Vec<_> = answers.into_iter().map(|a| a.expect("the rank answers")).collect();
    for (thread, ordinal, rank) in &answers {
        println!("rank {rank}: thread {thread}, device ordinal {ordinal}");
    }
    assert_eq!(answers[0].1, 0);
    assert_eq!(answers[1].1, 1);
    assert_ne!(answers[0].0, answers[1].0, "one thread per rank");
}

#[test]
#[ignore = "needs two GPUs on one node and libnccl.so.2"]
fn a_group_configured_for_another_nccl_or_driver_version_is_refused_before_any_communicator_exists() {
    nccl::load().expect("nccl");
    let installed = nccl::version().expect("version");
    let error = match RankGroup::start(&[0, 1], &ExpectedVersions { nccl: Some(installed + 100), driver: None }) {
        Ok(_) => panic!("a group expecting another NCCL must not start"),
        Err(e) => e,
    };
    match error.downcast_ref::<RendezvousError>() {
        Some(RendezvousError::VersionMismatch { what: "NCCL", expected, found, .. }) => println!("refused: expected NCCL {expected}, installed {found}"),
        other => panic!("expected a NCCL version mismatch, got {other:?} ({error:#})"),
    }
    let error = match RankGroup::start(&[0, 1], &ExpectedVersions { nccl: None, driver: Some(1) }) {
        Ok(_) => panic!("a group expecting another driver must not start"),
        Err(e) => e,
    };
    assert!(matches!(error.downcast_ref::<RendezvousError>(), Some(RendezvousError::VersionMismatch { what: "driver", .. })), "{error:#}");
    // And the matching configuration starts.
    RankGroup::start(&[0, 1], &ExpectedVersions { nccl: Some(installed), driver: None }).expect("the installed version is accepted");
}

#[test]
#[ignore = "needs two GPUs on one node and libnccl.so.2"]
fn a_rank_that_fails_or_panics_is_reported_by_index_and_the_others_still_answer() {
    let group = RankGroup::start(&[0, 1], &ExpectedVersions::default()).expect("group");
    let answers = group.run_all(CALL, |ctx| {
        if ctx.rank == 1 {
            return Err(anyhow::anyhow!("rank 1 refuses"));
        }
        Ok(ctx.rank)
    });
    assert_eq!(answers[0].as_ref().unwrap(), &0);
    assert!(answers[1].as_ref().unwrap_err().to_string().contains("rank 1 refuses"));
    let answers = group.run_all(CALL, |ctx| {
        if ctx.rank == 0 {
            panic!("rank 0 panics");
        }
        Ok(ctx.rank)
    });
    assert!(answers[0].as_ref().unwrap_err().to_string().contains("rank 0 panicked"), "{:?}", answers[0].as_ref().err());
    assert_eq!(answers[1].as_ref().unwrap(), &1);
    // The group is still usable after a rank's error.
    let again = group.run_all(CALL, |ctx| Ok(ctx.rank));
    assert!(again.iter().all(|a| a.is_ok()));
}
