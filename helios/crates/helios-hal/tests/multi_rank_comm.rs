//! The two-rank stub group driven through the `Communicator` trait only (ADR 0006's verification plan). Every result is `[RUN on stub]`.

use std::time::{Duration, Instant};

use helios_hal::multi_rank_comm::StubComm;
use helios_hal::multi_rank_stub::{GroupError, RankCtx, StubGroup};
use helios_hal::{CommDType, Communicator, HalError};

const LONG: Duration = Duration::from_secs(20);

/// What a rank runs: nothing but the trait object (`StubComm::alloc` and `read` only stand in for device allocation and copies).
fn through_the_trait(ctx: &mut RankCtx) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), GroupError> {
    let comm: &StubComm = &ctx.comm;
    let dynamic: &dyn Communicator = comm;
    let (rank, world) = (dynamic.rank() as usize, dynamic.world_size() as usize);
    assert_eq!(world, 2);
    let fail = |e: anyhow::Error| match e.downcast::<GroupError>() {
        Ok(group) => group,
        Err(other) => GroupError::Plan(other.to_string()),
    };
    // All-reduce: rank r holds [r+1, r+3].
    let reduce = comm.alloc(vec![rank as f32 + 1.0, rank as f32 + 3.0]);
    dynamic.all_reduce_sum(reduce, 2, CommDType::F32).map_err(fail)?;
    // All-gather: each rank contributes one element, both end with [10, 11].
    let (send, gathered) = (comm.alloc(vec![10.0 + rank as f32]), comm.alloc(vec![0.0; 2]));
    dynamic.all_gather(send, gathered, 1, CommDType::F32).map_err(fail)?;
    // Reduce-scatter: both hold [1, 2, 3, 4] scaled by (rank+1); rank r keeps block r of the sum [3, 6, 9, 12].
    let (input, scattered) = (comm.alloc((1..=4).map(|v| v as f32 * (rank as f32 + 1.0)).collect()), comm.alloc(vec![0.0; 2]));
    dynamic.reduce_scatter_sum(input, scattered, 2, CommDType::F32).map_err(fail)?;
    // Send / recv: rank 0 sends [7, 8, 9] to rank 1, rank 1 sends [1] back.
    let message = if rank == 0 {
        dynamic.send(comm.alloc(vec![7.0, 8.0, 9.0]), 3, CommDType::F32, 1).map_err(fail)?;
        let back = comm.alloc(vec![0.0]);
        dynamic.recv(back, 1, CommDType::F32, 1).map_err(fail)?;
        back
    } else {
        let got = comm.alloc(vec![0.0; 3]);
        dynamic.recv(got, 3, CommDType::F32, 0).map_err(fail)?;
        dynamic.send(comm.alloc(vec![1.0]), 1, CommDType::F32, 0).map_err(fail)?;
        got
    };
    let read = |h| comm.read(h).map_err(fail);
    Ok((read(reduce)?, read(gathered)?, read(scattered)?, read(message)?))
}

#[test]
fn the_collectives_and_point_to_point_have_the_real_semantics_through_the_trait() {
    let group = StubGroup::spawn(2, LONG);
    let answers = group.run_all(LONG, through_the_trait);
    let (r0, r1) = (answers[0].clone().unwrap(), answers[1].clone().unwrap());
    // Sum of [1, 3] and [2, 4].
    assert_eq!(r0.0, vec![3.0, 7.0]);
    assert_eq!(r1.0, vec![3.0, 7.0]);
    assert_eq!(r0.1, vec![10.0, 11.0]);
    assert_eq!(r1.1, vec![10.0, 11.0]);
    assert_eq!(r0.2, vec![3.0, 6.0], "rank 0 keeps block 0");
    assert_eq!(r1.2, vec![9.0, 12.0], "rank 1 keeps block 1");
    assert_eq!(r0.3, vec![1.0], "rank 0 received rank 1's message");
    assert_eq!(r1.3, vec![7.0, 8.0, 9.0], "rank 1 received rank 0's message");
}

#[test]
fn an_element_type_the_stub_does_not_move_is_a_typed_refusal() {
    let group = StubGroup::spawn(2, LONG);
    let answers = group.run_all(LONG, |ctx| {
        let buf = ctx.comm.alloc(vec![0.0; 4]);
        let error = Communicator::all_reduce_sum(&ctx.comm, buf, 4, CommDType::Bf16).unwrap_err();
        assert_eq!(error.downcast_ref::<HalError>(), Some(&HalError::Unsupported("the multi-rank stub moves F32 buffers only")));
        Ok(())
    });
    assert!(answers.iter().all(Result::is_ok));
}

#[test]
fn a_rank_that_dies_while_its_peer_waits_in_recv_ends_the_wait_at_once() {
    let group = StubGroup::spawn(2, Duration::from_secs(30));
    let started = Instant::now();
    let answers = group.run_all(Duration::from_secs(30), |ctx| {
        if ctx.rank == 1 {
            panic!("rank 1 dies before it sends");
        }
        let buf = ctx.comm.alloc(vec![0.0]);
        Communicator::recv(&ctx.comm, buf, 1, CommDType::F32, 1).map_err(|e| e.downcast::<GroupError>().unwrap())
    });
    assert!(started.elapsed() < Duration::from_secs(5), "{:?}", started.elapsed());
    assert_eq!(answers[0], Err(GroupError::RankFailed(1)));
}

#[test]
fn a_slow_sender_is_waited_for_within_the_timeout_and_reported_beyond_it() {
    let exchange = |ctx: &mut RankCtx| {
        let buf = ctx.comm.alloc(vec![5.0]);
        if ctx.rank == 1 {
            std::thread::sleep(Duration::from_millis(300));
            Communicator::send(&ctx.comm, buf, 1, CommDType::F32, 0).map_err(|e| e.downcast::<GroupError>().unwrap())?;
            Ok(0.0)
        } else {
            let into = ctx.comm.alloc(vec![0.0]);
            Communicator::recv(&ctx.comm, into, 1, CommDType::F32, 1).map_err(|e| e.downcast::<GroupError>().unwrap())?;
            Ok(ctx.comm.read(into).unwrap()[0])
        }
    };
    let patient = StubGroup::spawn(2, Duration::from_secs(5));
    assert_eq!(patient.run_all(Duration::from_secs(10), exchange)[0], Ok(5.0));
    let impatient = StubGroup::spawn(2, Duration::from_millis(100));
    assert!(matches!(impatient.run_all(Duration::from_secs(10), exchange)[0], Err(GroupError::Timeout(_))));
}

#[test]
fn a_rendezvous_timeout_names_the_ranks_that_never_joined() {
    let group = StubGroup::spawn(4, Duration::from_millis(200));
    let answers = group.run_all(Duration::from_secs(10), |ctx| {
        if ctx.rank == 1 || ctx.rank == 3 {
            // These two ranks never join (their processes did not start).
            std::thread::sleep(Duration::from_millis(600));
            return Ok(());
        }
        ctx.rendezvous("nccl 2.21.5")
    });
    for rank in [0, 2] {
        match &answers[rank] {
            Err(GroupError::RendezvousTimeout { missing, .. }) => assert_eq!(missing, &vec![1, 3], "rank {rank}"),
            // The first rank to time out marks the group dead; its peer may see that first.
            Err(GroupError::RankFailed(_)) => {}
            other => panic!("rank {rank}: {other:?}"),
        }
    }
    assert!(answers.iter().any(|a| matches!(a, Err(GroupError::RendezvousTimeout { missing, .. }) if missing == &vec![1, 3])), "someone names ranks 1 and 3");
}
