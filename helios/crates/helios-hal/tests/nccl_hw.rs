//! Hardware tests of NCCL on two GPUs of one node (gate J3, ADR 0009): communicators created inside one group by one thread, then one thread per rank for
//! the collectives; results against the host; an all-reduce captured in a CUDA graph and replayed 1000 times; a typed timeout when a rank does not take
//! part. Tolerance (declared before the first run): the data are small integers in f32, so every result must be **bit-identical** to the host
//! computation, and each of the 1000 replays must produce the same bits.
//!
//! ```text
//! cargo test --release -p helios-hal --features sm90 --test nccl_hw -- --ignored --nocapture --test-threads=1
//! ```
#![cfg(feature = "sm90")]

use std::time::Duration;

use helios_hal::cuda::nccl::{self, CollectiveError, NcclComm, RankVersions};
use helios_hal::cuda::{CudaDevice, GraphExec};
use helios_hal::{CommDType, Communicator};

const WAIT: Duration = Duration::from_secs(60);

/// The value rank `rank` holds at element `i`: small integers, so f32 sums are exact.
fn value(rank: u32, i: usize) -> f32 {
    ((i * 7 + rank as usize * 13) % 251) as f32 - 100.0
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn upload(dev: &mut CudaDevice, data: &[f32]) -> u64 {
    let ptr = dev.alloc(data.len() * 4).expect("alloc");
    dev.htod(ptr, &bytes(data)).expect("htod");
    ptr
}

fn download(dev: &CudaDevice, ptr: u64, count: usize) -> Vec<f32> {
    let mut out = vec![0u8; count * 4];
    dev.dtoh(&mut out, ptr).expect("dtoh");
    floats(&out)
}

/// Two devices, a unique id, and the communicators made inside one group by this thread.
fn two_ranks() -> (CudaDevice, CudaDevice, Vec<NcclComm>) {
    println!("NCCL library: {}, version {}", nccl::load().expect("libnccl.so.2 must be present"), nccl::format_version(nccl::version().expect("version")));
    let d0 = CudaDevice::probe_ordinal(0).expect("device 0");
    let d1 = CudaDevice::probe_ordinal(1).expect("device 1 (this test needs two GPUs)");
    let id = nccl::unique_id().expect("unique id");
    let comms = nccl::init_local_group(2, id, &[(0, &d0), (1, &d1)]).expect("grouped init");
    println!("NCCL environment: {:?}", nccl::nccl_environment());
    (d0, d1, comms)
}

/// Run `body(rank, device, communicator)` on one thread per rank (each thread makes its device current first).
fn on_ranks<R: Send>(d0: CudaDevice, d1: CudaDevice, comms: Vec<NcclComm>, body: impl Fn(u32, CudaDevice, NcclComm) -> R + Sync) -> Vec<R> {
    let mut devices = vec![Some(d0), Some(d1)];
    let mut comms: Vec<Option<NcclComm>> = comms.into_iter().map(Some).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2)
            .map(|r| {
                let (device, comm) = (devices[r].take().unwrap(), comms[r].take().unwrap());
                let body = &body;
                scope.spawn(move || {
                    device.set_current().expect("set_current");
                    body(r as u32, device, comm)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a rank thread panicked")).collect()
    })
}

#[test]
#[ignore = "needs two H100s on one node and libnccl.so.2"]
fn all_reduce_all_gather_and_reduce_scatter_equal_the_host_results() {
    let (d0, d1, comms) = two_ranks();
    // The rendezvous check accepts equal versions (one process, so they are equal; a mismatch cannot be configured on one node, the pure check in
    // nccl.rs covers the refusal).
    let versions: Vec<RankVersions> = (0..2).map(|rank| RankVersions { rank, nccl: nccl::version().unwrap(), driver: 0 }).collect();
    nccl::check_versions(&versions).expect("equal versions are accepted");
    let lines = on_ranks(d0, d1, comms, |rank, mut dev, comm| {
        assert_eq!((comm.rank(), comm.world_size()), (rank, 2));
        let mut lines = Vec::new();
        for count in [1usize, 255, 4096, 1 << 20] {
            let mine: Vec<f32> = (0..count).map(|i| value(rank, i)).collect();
            // all-reduce
            let buf = upload(&mut dev, &mine);
            comm.all_reduce_sum(buf, count, CommDType::F32).expect("all_reduce");
            comm.wait(WAIT).expect("all_reduce finished");
            let want: Vec<f32> = (0..count).map(|i| value(0, i) + value(1, i)).collect();
            assert_eq!(bytes(&download(&dev, buf, count)), bytes(&want), "rank {rank}: all-reduce of {count} elements");
            // all-gather: rank order
            let send = upload(&mut dev, &mine);
            let recv = dev.alloc(count * 2 * 4).expect("alloc");
            comm.all_gather(send, recv, count, CommDType::F32).expect("all_gather");
            comm.wait(WAIT).expect("all_gather finished");
            let want: Vec<f32> = (0..2).flat_map(|r| (0..count).map(move |i| value(r, i))).collect();
            assert_eq!(bytes(&download(&dev, recv, count * 2)), bytes(&want), "rank {rank}: all-gather of {count} elements");
            // reduce-scatter: rank r keeps block r of the sum
            let both: Vec<f32> = (0..count * 2).map(|i| value(rank, i)).collect();
            let send = upload(&mut dev, &both);
            let out = dev.alloc(count * 4).expect("alloc");
            comm.reduce_scatter_sum(send, out, count, CommDType::F32).expect("reduce_scatter");
            comm.wait(WAIT).expect("reduce_scatter finished");
            let want: Vec<f32> = (0..count).map(|i| value(0, rank as usize * count + i) + value(1, rank as usize * count + i)).collect();
            assert_eq!(bytes(&download(&dev, out, count)), bytes(&want), "rank {rank}: reduce-scatter of {count} elements");
            for p in [buf, send, recv, out] {
                dev.free(p).expect("free");
            }
            lines.push(format!("rank {rank}: {count} f32 elements: all-reduce, all-gather and reduce-scatter equal the host, bit for bit"));
        }
        lines
    });
    for l in lines.iter().flatten() {
        println!("{l}");
    }
}

#[test]
#[ignore = "needs two H100s on one node and libnccl.so.2"]
fn an_all_reduce_captured_in_a_graph_replays_bit_identically_one_thousand_times() {
    let (d0, d1, comms) = two_ranks();
    let summary = on_ranks(d0, d1, comms, |rank, mut dev, comm| {
        let count = 1usize << 16;
        let mine: Vec<f32> = (0..count).map(|i| value(rank, i)).collect();
        let pristine = upload(&mut dev, &mine);
        let work = dev.alloc(count * 4).expect("alloc");
        // The graph restores the input and reduces it in place, so every replay starts from the same bits.
        let graph = GraphExec::capture(&dev, || {
            dev.dtod_async(work, pristine, count * 4)?;
            comm.all_reduce_sum(work, count, CommDType::F32)
        })
        .expect("the all-reduce is captured");
        let want = bytes(&(0..count).map(|i| value(0, i) + value(1, i)).collect::<Vec<f32>>());
        for replay in 0..1000 {
            graph.launch(&dev).expect("launch");
            comm.wait(WAIT).expect("replay finished");
            assert!(bytes(&download(&dev, work, count)) == want, "rank {rank}: replay {replay} differs from the host sum");
        }
        drop(graph);
        format!("rank {rank}: {count} f32 elements, 1000 replays of a graph holding a copy and an all-reduce: bit-identical to the host sum every time")
    });
    for l in summary {
        println!("{l}");
    }
}

#[test]
#[ignore = "needs two H100s on one node and libnccl.so.2"]
fn a_rank_that_does_not_take_part_ends_in_a_typed_timeout() {
    let (d0, d1, comms) = two_ranks();
    let outcome = on_ranks(d0, d1, comms, |rank, mut dev, comm| {
        let buf = upload(&mut dev, &vec![1.0f32; 1024]);
        let result = if rank == 0 {
            comm.all_reduce_sum(buf, 1024, CommDType::F32).expect("enqueue");
            Some(comm.wait(Duration::from_secs(3)))
        } else {
            // The withheld rank never enqueues its half.
            std::thread::sleep(Duration::from_secs(6));
            None
        };
        // A communicator that timed out is aborted; destroying it or its peer could block, so both are left to the process end.
        std::mem::forget(comm);
        std::mem::forget(dev);
        result
    });
    let error = outcome[0].as_ref().expect("rank 0 waited").as_ref().expect_err("the wait must time out");
    match error.downcast_ref::<CollectiveError>() {
        Some(CollectiveError::Timeout { rank, waited }) => println!("typed timeout: rank {rank} waited {waited:?}: {error}"),
        other => panic!("expected CollectiveError::Timeout, got {other:?} ({error:#})"),
    }
}
