//! Continuous batching in the scheduler (`[RUN on stub]`): decode steps gather the running sequences into buckets, a row's plan depends only on
//! its own sequence, starvation is bounded by the policy ratio, cancel and page pressure behave. The stub computes nothing, so these cover plans
//! and bookkeeping, never numerics.

use helios_core::kv::{BlockHash, KvEvent};
use helios_core::simulator::{agent_sessions, replay, Report, Trace, TraceRequest};
use helios_core::step_scheduler::StepConfig;
use helios_hal::step::{Capacity, Input, StepPlan, Want};
use helios_hal::{BackendConfig, ComputeBackend, KvPool, KvSpec, MemoryPolicy, RankInfo, WeightSource};
use hyperion_specification::v1::{
    Activation, AttentionKind, GenerationConfig, Manifest, ModelConfig, RopeStyle, TensorFormat, TokenizerConfig, CONTRACT_REVISION,
};

const IDENTITY: BlockHash = BlockHash([9; 32]);

fn stub_manifest() -> Manifest {
    Manifest {
        model_id: "stub".into(),
        target_arch: "sm_0".into(),
        tensor_format: TensorFormat::Bf16,
        model: ModelConfig {
            num_layers: 1,
            attention: AttentionKind::Mha,
            num_attention_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            vocab_size: 1,
            max_seq_len: 8192,
            hidden_size: 8,
            intermediate_size: 16,
            rope_theta: 10000.0,
            rope_style: RopeStyle::RotateHalf,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            qkv_bias: true,
            qk_norm: false,
            activation: Activation::Silu,
            attention_scale: None,
        },
        contract_revision: CONTRACT_REVISION,
        generation: GenerationConfig { eos_token_ids: vec![0], sampling_defaults: None },
        chat_template: None,
        qualified_checkpoint: None,
        tokenizer: TokenizerConfig::default(),
        layers: vec![],
        payloads: Vec::new(),
        parallelism: None,
        hardware_recipe: Default::default(),
        signature: None,
    }
}

/// The real stub backend taken through the whole load order, with a pool of `tokens` tokens.
fn open(tokens: usize) -> (Box<dyn ComputeBackend>, KvPool) {
    let backend = helios_hal::create_backend(&stub_manifest()).expect("stub backend").backend;
    let config = BackendConfig::from_manifest(&stub_manifest()).unwrap();
    backend.initialize_device(&RankInfo::single()).unwrap();
    let model = backend.load_model(&config, &WeightSource { files: Vec::new(), slices: Vec::new() }).unwrap();
    let reservation = backend.reserve_kv(&model, &KvSpec { max_batch: 32, ..KvSpec::default() }).unwrap();
    let graphs = backend.capture_graphs(&model, &reservation).unwrap();
    let pool = backend.commit_kv(reservation, &graphs, &MemoryPolicy { reserve_bytes: 0, max_tokens: tokens }).unwrap();
    (backend, pool)
}

fn config(max_batch: usize, max_queue: usize, prefill_steps_per_decode: usize) -> StepConfig {
    // The stub always answers token 0, so no end-of-sequence id is set: every request runs to its length.
    StepConfig {
        prefill_capacities: vec![128, 256, 512, 1024, 2048, 4096],
        max_slots: max_batch,
        max_queue,
        eos_token_ids: Vec::new(),
        max_seq_len: 8192,
        decode_buckets: helios_hal::decode_buckets::buckets(max_batch).unwrap(),
        prefill_steps_per_decode,
    }
}

fn run_with(trace: &Trace, tokens: usize, max_batch: usize, max_queue: usize, ratio: usize) -> Report {
    let (backend, pool) = open(tokens);
    replay(trace, config(max_batch, max_queue, ratio), IDENTITY, 64, backend.as_ref(), &pool).unwrap()
}

fn same_arrival(n: usize, prompt: usize, max_new: u32) -> Trace {
    Trace { requests: (0..n as u32).map(|i| TraceRequest { arrive: 0, tokens: (i * 10_000..i * 10_000 + prompt as u32).collect(), max_new, cancel_at: None }).collect() }
}

fn plans(report: &Report) -> Vec<StepPlan> {
    report.plans.iter().map(|p| StepPlan::decode(p).unwrap()).collect()
}

#[test]
fn concurrent_sequences_decode_together_in_the_smallest_bucket_that_holds_them() {
    let report = run_with(&same_arrival(8, 100, 30), 64 * 400, 8, 16, 1);
    assert_eq!((report.finished, report.failed), (8, 0));
    assert!(report.generated.values().all(|t| t.len() == 30));
    assert_eq!(report.peak_running, 8);
    assert!(report.decode_rows.get(&8).copied().unwrap_or(0) > 0, "full batches ran: {:?}", report.decode_rows);
    for plan in plans(&report) {
        plan.validate().unwrap();
        if let Capacity::Decode { bucket } = plan.capacity {
            let expected = [1usize, 2, 4, 8].into_iter().find(|b| *b >= plan.rows.len()).unwrap();
            assert_eq!(bucket as usize, expected, "{} rows", plan.rows.len());
            assert!(plan.rows.windows(2).all(|w| w[0].slot < w[1].slot), "rows are ordered by slot");
            assert!(plan.rows.iter().all(|r| r.input == Input::FromDevice && r.want == Want::Token));
        }
    }
}

#[test]
fn a_batch_of_one_is_the_old_batch_one_plan() {
    let report = run_with(&same_arrival(1, 300, 12), 64 * 100, 1, 4, 1);
    assert_eq!(report.decode_rows.keys().copied().collect::<Vec<_>>(), vec![1]);
    assert!(plans(&report).iter().all(|p| p.rows.len() == 1 && !matches!(p.capacity, Capacity::Decode { bucket } if bucket != 1)));
}

#[test]
fn a_long_prompt_delays_the_running_sequences_by_at_most_the_policy_ratio() {
    // One short request is decoding when a very long prompt arrives; chunk steps and decode steps alternate at the configured ratio.
    for ratio in [1usize, 2] {
        let trace = Trace {
            requests: vec![
                TraceRequest { arrive: 0, tokens: (0..100).collect(), max_new: 200, cancel_at: None },
                TraceRequest { arrive: 10, tokens: (50_000..50_000 + 8000).collect(), max_new: 4, cancel_at: None },
            ],
        };
        let report = run_with(&trace, 64 * 400, 4, 8, ratio);
        assert_eq!(report.finished, 2);
        let mut streak = 0usize;
        let mut worst = 0usize;
        let mut seen_decode = false;
        for plan in plans(&report) {
            match plan.capacity {
                Capacity::Prefill { .. } if seen_decode => {
                    streak += 1;
                    worst = worst.max(streak);
                }
                Capacity::Decode { .. } => {
                    seen_decode = true;
                    streak = 0;
                }
                _ => {}
            }
        }
        assert!(worst <= ratio, "ratio {ratio}: {worst} chunk steps in a row while a sequence was decoding");
    }
}

#[test]
fn cancelling_one_row_mid_batch_leaves_the_others_untouched_and_frees_its_pages() {
    let mut trace = same_arrival(4, 200, 60);
    trace.requests[2].cancel_at = Some(30);
    let report = run_with(&trace, 64 * 200, 4, 8, 1);
    assert_eq!((report.cancelled, report.finished, report.failed), (1, 3, 0));
    assert!(report.generated.iter().filter(|(id, _)| **id != 3).all(|(_, t)| t.len() == 60));
    assert!(report.generated.get(&3).map_or(0, Vec::len) < 60, "the cancelled request stopped early");
    assert_eq!(report.final_pages_used, report.final_registered_blocks, "only cached blocks remain: nothing leaks");
}

#[test]
fn a_request_that_cannot_get_pages_waits_in_the_queue_and_is_admitted_when_pages_free() {
    // 12 pages; each request needs 4 pages of prompt and grows to 5: at most two run together, the other two wait for them.
    let report = run_with(&same_arrival(4, 250, 60), 64 * 12, 4, 8, 1);
    assert_eq!((report.finished, report.failed, report.rejected), (4, 0, 0));
    assert!(report.peak_running < 4, "page pressure held some requests back: {}", report.peak_running);
    assert!(report.peak_pages <= 12);
    let waits: Vec<u64> = (1..=4).map(|id| report.first_token_ticks[&id]).collect();
    assert!(waits[3] > waits[0], "the last request queued behind the first: {waits:?}");
}

#[test]
fn a_rows_plan_does_not_depend_on_the_other_rows() {
    // Request 1 alone, then with seven neighbours of other contents: its decode rows (slot, position, table length, input) are the same.
    let own: Vec<u32> = (0..300).collect();
    let alone = Trace { requests: vec![TraceRequest { arrive: 0, tokens: own.clone(), max_new: 40, cancel_at: None }] };
    let mut crowded = alone.clone();
    for i in 1..8u32 {
        crowded.requests.push(TraceRequest { arrive: 0, tokens: (i * 20_000..i * 20_000 + 100 + i * 37).collect(), max_new: 40, cancel_at: None });
    }
    let rows_of_slot0 = |report: &Report| -> Vec<(u32, u32)> {
        plans(report).iter().filter(|p| matches!(p.capacity, Capacity::Decode { .. })).flat_map(|p| p.rows.iter().filter(|r| r.slot == 0).map(|r| (r.position, r.table_len)).collect::<Vec<_>>()).collect()
    };
    let a = rows_of_slot0(&run_with(&alone, 64 * 400, 8, 16, 1));
    let b = rows_of_slot0(&run_with(&crowded, 64 * 400, 8, 16, 1));
    assert_eq!(a.len(), 39, "one decode row per token after the first");
    assert_eq!(a, b);
}

#[test]
fn the_same_concurrent_trace_gives_byte_identical_plans_and_events_stay_gap_free() {
    let trace = agent_sessions(6, 3, 600, 150, 24, 5, 40);
    let first = run_with(&trace, 64 * 500, 8, 32, 1);
    let second = run_with(&trace, 64 * 500, 8, 32, 1);
    assert_eq!(first.plans, second.plans);
    assert_eq!(first.finished, 18);
    let ids: Vec<u64> = first.kv_events.iter().map(KvEvent::id).collect();
    assert!(ids.iter().enumerate().all(|(i, id)| *id == i as u64 + 1));
    assert!(first.hit_rate > 0.3, "the sessions share a system prompt: {}", first.hit_rate);
}

#[test]
fn shared_blocks_stay_while_any_row_uses_them_and_pages_return_when_all_are_done() {
    // Eight requests share a 640-token prefix (10 pages): it is stored once, and once all have finished nothing but cached blocks remains.
    let shared: Vec<u32> = (0..640).collect();
    let requests = (0..8u32).map(|i| TraceRequest { arrive: i as u64, tokens: shared.iter().copied().chain(100_000 + i * 100..100_000 + i * 100 + 30).collect(), max_new: 20, cancel_at: None }).collect();
    let report = run_with(&Trace { requests }, 64 * 200, 8, 16, 1);
    assert_eq!((report.finished, report.failed), (8, 0));
    assert!(report.hit_rate > 0.5, "{}", report.hit_rate);
    assert!(report.peak_pages < 8 * 11, "the shared prefix was not duplicated per row: {} pages", report.peak_pages);
}
