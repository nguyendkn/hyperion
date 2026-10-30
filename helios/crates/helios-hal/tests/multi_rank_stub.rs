//! The multi-rank stub (ADR 0006): a named test double, so every result here is `[RUN on stub]`. It tests the control plane, the sharding
//! plans combined with the collectives, and failure handling. It says nothing about a GPU.

use std::time::{Duration, Instant};

use helios_hal::multi_rank_stub::{GroupError, StubGroup};
use helios_hal::sharding::{extract, rank_slice};
use helios_hal::{BackendConfig, Capacity, Deployment, Input, PageDelta, Row, StepPlan, TensorSlice, TensorSpec, Want};
use hyperion_specification::v1::{Activation, AttentionKind, ModelConfig, RopeStyle, ShardClass, TensorFormat};

const LONG: Duration = Duration::from_secs(20);

fn plan(step: u64) -> StepPlan {
    StepPlan {
        step,
        capacity: Capacity::Decode { bucket: 2 },
        rows: vec![
            Row { slot: 0, input: Input::FromDevice, n_tokens: 1, position: 70, table_len: 2, want: Want::Token },
            Row { slot: 1, input: Input::Tokens(vec![9, 8, 7]), n_tokens: 3, position: 0, table_len: 1, want: Want::Logits },
        ],
        deltas: vec![PageDelta { slot: 0, index: 1, page: 12 }, PageDelta { slot: 1, index: 0, page: 3 }],
        inits: Vec::new(),
    }
}

#[test]
fn a_plan_is_broadcast_to_every_rank_and_every_rank_applies_the_same_deltas() {
    let group = StubGroup::spawn(4, LONG);
    let output = group.step(&plan(1)).expect("every rank accepts the plan and answers the same");
    assert_eq!(output.step, 1);
    assert_eq!(output.rows[0].tokens, vec![0]);
    let tables = group.run_all(LONG, |ctx| Ok(ctx.page_tables.clone()));
    let first = tables[0].as_ref().unwrap();
    assert_eq!(first[&0], vec![None, Some(12)]);
    assert_eq!(first[&1], vec![Some(3)]);
    for table in &tables {
        assert_eq!(table.as_ref().unwrap(), first, "every rank mirrors the same page tables");
    }
    // A later plan releasing a page is seen by all.
    let mut release = plan(2);
    release.deltas = vec![PageDelta { slot: 0, index: 1, page: helios_hal::step::RELEASED }];
    group.step(&release).unwrap();
    let tables = group.run_all(LONG, |ctx| Ok(ctx.page_tables[&0].clone()));
    assert!(tables.iter().all(|t| t.as_ref().unwrap() == &vec![None, None]));
}

#[test]
fn a_rank_that_receives_a_corrupted_plan_refuses_it_by_hash() {
    let group = StubGroup::spawn(3, LONG);
    let good = plan(1).encode();
    let mut bad = good.clone();
    *bad.last_mut().unwrap() ^= 0xFF;
    let answers = group.run_all(LONG, move |ctx| {
        let bytes = if ctx.rank == 2 { &bad } else { &good };
        ctx.apply_plan(bytes).map(|(_, hash)| hash)
    });
    assert!(answers[0].is_ok() && answers[1].is_ok());
    assert!(matches!(&answers[2], Err(GroupError::Plan(message)) if message.contains("hash")), "{:?}", answers[2]);
}

#[test]
fn the_collectives_have_the_real_semantics() {
    let group = StubGroup::spawn(4, LONG);
    // Rank r holds [r, r + 10, r + 20, r + 30].
    let answers = group.run_all(LONG, |ctx| {
        let mine: Vec<f32> = (0..4).map(|i| (ctx.rank + 10 * i) as f32).collect();
        Ok((ctx.all_reduce_sum(&mine)?, ctx.all_gather(&mine[..2])?, ctx.reduce_scatter_sum(&mine)?))
    });
    for (rank, answer) in answers.into_iter().enumerate() {
        let (reduced, gathered, scattered) = answer.unwrap();
        // The sum over ranks 0..4 of r + 10 i is 6 + 40 i.
        assert_eq!(reduced, vec![6.0, 46.0, 86.0, 126.0], "rank {rank}");
        assert_eq!(gathered, vec![0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0], "rank {rank}");
        assert_eq!(scattered, vec![6.0 + 40.0 * rank as f32], "rank {rank} keeps block {rank}");
    }
}

fn config(tensors: Vec<TensorSpec>, tp: u32) -> BackendConfig {
    BackendConfig {
        model_id: "toy".into(),
        model: ModelConfig {
            num_layers: 1,
            attention: AttentionKind::Gqa,
            num_attention_heads: 4,
            num_kv_heads: 4,
            head_dim: 4,
            vocab_size: 8,
            max_seq_len: 16,
            hidden_size: 16,
            intermediate_size: 16,
            rope_theta: 1e4,
            rope_style: RopeStyle::RotateHalf,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            qkv_bias: false,
            qk_norm: false,
            activation: Activation::Silu,
            attention_scale: None,
        },
        tensor_format: TensorFormat::Bf16,
        target_arch: "sm_0".into(),
        eos_token_ids: vec![0],
        payload_files: vec!["m.safetensors".into()],
        payload_lengths: vec![None],
        tensors,
        supported_tp_sizes: vec![1, 2, 4],
        deployment: Deployment { tp_size: tp, pp_size: 1, tp_rank: 0, pp_stage: 0 },
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// y = W x on the host, W is `[rows, cols]`.
fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    (0..rows).map(|r| (0..cols).map(|c| w[r * cols + c] * x[c]).sum()).collect()
}

/// The sharding plan and the collectives together: a column-split matrix needs an all-gather, a row-split one an all-reduce, and both give
/// the same answer as the unsplit product. Weights are small integers so float sums are exact in any order.
#[test]
fn column_and_row_parallel_products_equal_the_whole_product() {
    let (rows, cols, tp) = (8usize, 16usize, 4usize);
    let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 7) % 11) as f32 - 5.0).collect();
    let x: Vec<f32> = (0..cols).map(|i| (i % 5) as f32 - 2.0).collect();
    let expected = matvec(&w, rows, cols, &x);
    let spec = |name: &str, sharding| TensorSpec { name: name.into(), payload: 0, byte_offset: 0, byte_length: 0, dtype: "f32".into(), shape: vec![rows, cols], sharding: Some(sharding) };
    let column = spec("layers.0.mlp.up_proj.weight", ShardClass::Column);
    let row = spec("layers.0.mlp.down_proj.weight", ShardClass::Row);
    let base = config(vec![column.clone(), row.clone()], tp as u32);
    let (w_bytes, x) = (f32_bytes(&w), x);
    let answers = group_run(tp, {
        let (w_bytes, x, base, column, row) = (w_bytes.clone(), x.clone(), base.clone(), column.clone(), row.clone());
        move |ctx| {
            let mut c = base.clone();
            c.deployment.tp_rank = ctx.rank as u32;
            // Column parallel: this rank owns rows [r*2, r*2+2) and computes those outputs, an all-gather assembles y.
            let slice: TensorSlice = rank_slice(&c, &column).unwrap().unwrap();
            let mine = f32s(&extract(&w_bytes, &column.shape, 4, &slice));
            let partial = matvec(&mine, slice.end - slice.start, cols, &x);
            let gathered = ctx.all_gather(&partial)?;
            // Row parallel: this rank owns columns [r*4, r*4+4) and the matching part of x; an all-reduce sums the partial products.
            let slice: TensorSlice = rank_slice(&c, &row).unwrap().unwrap();
            let mine = f32s(&extract(&w_bytes, &row.shape, 4, &slice));
            let width = slice.end - slice.start;
            let partial = matvec(&mine, rows, width, &x[slice.start..slice.end]);
            let reduced = ctx.all_reduce_sum(&partial)?;
            Ok((gathered, reduced))
        }
    });
    for (rank, (gathered, reduced)) in answers.into_iter().enumerate() {
        assert_eq!(gathered, expected, "column parallel + all-gather on rank {rank}");
        assert_eq!(reduced, expected, "row parallel + all-reduce on rank {rank}");
    }
}

fn group_run<T: Send + 'static>(world: usize, f: impl Fn(&mut helios_hal::multi_rank_stub::RankCtx) -> Result<T, GroupError> + Send + Sync + 'static) -> Vec<T> {
    StubGroup::spawn(world, LONG).run_all(LONG, f).into_iter().map(|a| a.expect("every rank answers")).collect()
}

#[test]
fn a_rank_that_dies_fails_its_peers_at_once_not_after_the_timeout() {
    let group = StubGroup::spawn(4, Duration::from_secs(30));
    let started = Instant::now();
    let answers = group.run_all(Duration::from_secs(30), |ctx| {
        if ctx.rank == 1 {
            panic!("rank 1 dies before the collective");
        }
        ctx.all_reduce_sum(&[1.0])
    });
    assert!(started.elapsed() < Duration::from_secs(5), "the peers were told at once, not after 30 s: {:?}", started.elapsed());
    assert_eq!(answers[1], Err(GroupError::RankFailed(1)));
    for rank in [0, 2, 3] {
        assert_eq!(answers[rank], Err(GroupError::RankFailed(1)), "rank {rank}");
    }
    // The group stays dead: no later collective pretends to work.
    assert!(group.run_all(Duration::from_secs(5), |ctx| ctx.all_reduce_sum(&[1.0])).iter().all(Result::is_err));
}

#[test]
fn a_slow_rank_is_waited_for_within_the_timeout_and_reported_beyond_it() {
    let slow = |ctx: &mut helios_hal::multi_rank_stub::RankCtx| {
        if ctx.rank == 2 {
            std::thread::sleep(Duration::from_millis(300));
        }
        ctx.all_reduce_sum(&[ctx.rank as f32])
    };
    let patient = StubGroup::spawn(4, Duration::from_secs(5));
    assert!(patient.run_all(Duration::from_secs(10), slow).iter().all(|a| a.as_ref().unwrap() == &vec![6.0]));

    let impatient = StubGroup::spawn(4, Duration::from_millis(100));
    let answers = impatient.run_all(Duration::from_secs(10), slow);
    for rank in [0, 1, 3] {
        assert!(matches!(answers[rank], Err(GroupError::Timeout(_)) | Err(GroupError::RankFailed(_))), "rank {rank}: {:?}", answers[rank]);
    }
}

#[test]
fn a_rendezvous_with_a_missing_rank_times_out() {
    let group = StubGroup::spawn(4, Duration::from_millis(200));
    let started = Instant::now();
    let answers = group.run_all(Duration::from_secs(10), |ctx| {
        if ctx.rank == 3 {
            // This rank never joins (its process did not start).
            std::thread::sleep(Duration::from_millis(600));
            return Ok(());
        }
        ctx.rendezvous("nccl 2.21.5")
    });
    assert!(started.elapsed() < Duration::from_secs(5));
    for rank in 0..3 {
        assert!(matches!(answers[rank], Err(GroupError::Timeout(_)) | Err(GroupError::RankFailed(_))), "rank {rank}: {:?}", answers[rank]);
    }
}

#[test]
fn a_rendezvous_refuses_ranks_that_report_different_library_versions() {
    let group = StubGroup::spawn(3, LONG);
    let answers = group.run_all(LONG, |ctx| ctx.rendezvous(if ctx.rank == 2 { "nccl 2.19.3" } else { "nccl 2.21.5" }));
    for (rank, answer) in answers.iter().enumerate() {
        assert_eq!(answer, &Err(GroupError::VersionMismatch { rank: 2, expected: "nccl 2.21.5".into(), found: "nccl 2.19.3".into() }), "rank {rank}");
    }
    let agreeing = StubGroup::spawn(3, LONG);
    assert!(agreeing.run_all(LONG, |ctx| ctx.rendezvous("nccl 2.21.5")).iter().all(Result::is_ok));
}
