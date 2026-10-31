//! The scheduler on the plan path, replayed by the deterministic simulator against the single-rank CPU stub backend (`[RUN on stub]`: the
//! stub returns token 0 and computes nothing, so these tests cover the plans, the page bookkeeping and the prefix cache, never numerics).

use helios_core::kv::{BlockHash, KvEvent, Mirror, StreamHeader};
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
    let reservation = backend.reserve_kv(&model, &KvSpec::default()).unwrap();
    let graphs = backend.capture_graphs(&model, &reservation).unwrap();
    let pool = backend.commit_kv(reservation, &graphs, &MemoryPolicy { reserve_bytes: 0, max_tokens: tokens }).unwrap();
    (backend, pool)
}

fn config(max_slots: usize, max_queue: usize) -> StepConfig {
    // The stub always answers token 0, so no end-of-sequence id is set: every request runs to its length.
    StepConfig { prefill_capacities: vec![128, 256, 512, 1024, 2048, 4096], max_slots, max_queue, eos_token_ids: Vec::new(), max_seq_len: 8192 }
}

fn run(trace: &Trace, tokens: usize, max_slots: usize, max_queue: usize) -> Report {
    let (backend, pool) = open(tokens);
    replay(trace, config(max_slots, max_queue), IDENTITY, 64, backend.as_ref(), &pool).unwrap()
}

fn agent_trace() -> Trace {
    agent_sessions(4, 3, 1000, 200, 32, 40, 120)
}

#[test]
fn the_same_trace_gives_byte_identical_plans() {
    let trace = agent_trace();
    let (a, b) = (run(&trace, 64 * 600, 2, 8), run(&trace, 64 * 600, 2, 8));
    assert!(a.plans.len() > 100);
    assert_eq!(a.plans, b.plans, "the same trace gives the same plans, byte for byte");
    assert_eq!(a, b);
    // A different trace gives different plans (the comparison is not vacuous).
    let other = agent_sessions(4, 3, 1000, 201, 32, 40, 120);
    assert_ne!(run(&other, 64 * 600, 2, 8).plans, a.plans);
}

#[test]
fn every_request_finishes_with_exactly_the_tokens_it_asked_for() {
    let trace = agent_trace();
    let report = run(&trace, 64 * 600, 2, 8);
    assert_eq!((report.finished, report.failed, report.rejected, report.cancelled), (12, 0, 0, 0));
    assert_eq!(report.generated.len(), 12);
    assert!(report.generated.values().all(|tokens| tokens.len() == 32));
}

#[test]
fn the_prefix_cache_serves_the_shared_system_prompt_and_the_earlier_turns() {
    let report = run(&agent_trace(), 64 * 600, 2, 8);
    // 12 prompts of 1200 to 1860 tokens; the 1000-token system prompt is shared by all, each turn extends the last one of its session.
    assert!(report.hit_rate > 0.5, "hit rate {}", report.hit_rate);
    let decoded: Vec<StepPlan> = report.plans.iter().map(|p| StepPlan::decode(p).unwrap()).collect();
    // Some prefill starts past position 0: the cache skipped the cached prefix, in whole blocks of 64.
    let later_starts: Vec<u32> = decoded.iter().filter(|p| matches!(p.capacity, Capacity::Prefill { .. })).flat_map(|p| p.rows.iter().map(|r| r.position)).filter(|&pos| pos > 0).collect();
    assert!(!later_starts.is_empty());
    assert!(later_starts.iter().all(|pos| pos % 64 == 0), "{later_starts:?}");
}

#[test]
fn the_plans_have_the_shape_adr_0007_describes() {
    let trace = Trace { requests: vec![TraceRequest { arrive: 0, tokens: (1..=1000).collect(), max_new: 3, cancel_at: None }] };
    let report = run(&trace, 64 * 64, 1, 4);
    let plans: Vec<StepPlan> = report.plans.iter().map(|p| StepPlan::decode(p).unwrap()).collect();
    assert_eq!(plans.len(), 1 + 2, "one prefill chunk (1000 tokens fit the 1024 graph) then two decode steps");
    let first = &plans[0];
    assert_eq!(first.capacity, Capacity::Prefill { rows: 1024 });
    assert_eq!(first.rows[0].want, Want::Token, "the last chunk samples");
    assert_eq!((first.rows[0].n_tokens, first.rows[0].position, first.rows[0].table_len), (1000, 0, 16));
    assert_eq!(first.deltas.len(), 16, "the whole page table of the new slot");
    assert_eq!(first.inits.len(), 1);
    for plan in &plans[1..] {
        assert_eq!(plan.capacity, Capacity::Decode { bucket: 1 });
        assert_eq!(plan.rows[0].input, Input::FromDevice, "a decode row takes its input from the device");
        assert_eq!(plan.rows[0].n_tokens, 1);
        assert!(plan.inits.is_empty());
    }
    assert_eq!(plans[1].rows[0].position, 1000);
    assert_eq!(plans[2].rows[0].position, 1001);
    assert!(plans.iter().enumerate().all(|(i, p)| p.step == i as u64 + 1), "step numbers rise by one");
}

#[test]
fn a_long_prompt_is_split_into_chunks_of_the_largest_capacity() {
    let trace = Trace { requests: vec![TraceRequest { arrive: 0, tokens: (0..4200).collect(), max_new: 1, cancel_at: None }] };
    let report = run(&trace, 64 * 200, 1, 4);
    let plans: Vec<StepPlan> = report.plans.iter().map(|p| StepPlan::decode(p).unwrap()).collect();
    let chunks: Vec<(u32, u32, Want)> = plans.iter().map(|p| (p.rows[0].position, p.rows[0].n_tokens, p.rows[0].want)).collect();
    assert_eq!(chunks, vec![(0, 4096, Want::Nothing), (4096, 104, Want::Token)]);
    assert_eq!(plans[0].capacity, Capacity::Prefill { rows: 4096 });
    assert_eq!(plans[1].capacity, Capacity::Prefill { rows: 128 }, "the tail uses the smallest graph that holds it");
}

#[test]
fn a_cancelled_request_frees_its_pages_at_once() {
    let mut trace = Trace { requests: vec![TraceRequest { arrive: 0, tokens: (50_000..51_000).collect(), max_new: 500, cancel_at: Some(5) }] };
    trace.requests.push(TraceRequest { arrive: 0, tokens: (60_000..60_300).collect(), max_new: 4, cancel_at: None });
    let report = run(&trace, 64 * 100, 2, 4);
    assert_eq!((report.cancelled, report.finished), (1, 1));
    // Nothing leaks: the pages left are exactly the registered blocks the finished request left cached (one page per block of 64).
    assert_eq!(report.final_pages_used, report.final_registered_blocks);
}

#[test]
fn a_full_queue_refuses_the_next_request() {
    let requests = (0..20u32).map(|i| TraceRequest { arrive: 0, tokens: (i * 1000..i * 1000 + 100).collect(), max_new: 2, cancel_at: None }).collect();
    let report = run(&Trace { requests }, 64 * 100, 1, 4);
    assert_eq!(report.rejected, 16, "all 20 arrive together and only 4 fit the waiting queue");
    assert_eq!(report.finished, 4);
}

#[test]
fn under_page_pressure_cached_blocks_are_evicted_and_a_consumer_rebuilds_the_index_from_the_events() {
    // 60 pages for requests of 20 pages each, one after another: the cache must give pages back.
    let requests = (0..6u32).map(|i| TraceRequest { arrive: i as u64 * 100, tokens: (i * 10_000..i * 10_000 + 1200).collect(), max_new: 8, cancel_at: None }).collect();
    let report = run(&Trace { requests }, 64 * 60, 1, 4);
    assert_eq!((report.finished, report.failed), (6, 0));
    assert!(report.kv_events.iter().any(|e| matches!(e, KvEvent::BlockRemoved { .. })), "blocks were evicted");
    let ids: Vec<u64> = report.kv_events.iter().map(KvEvent::id).collect();
    assert!(ids.iter().enumerate().all(|(i, id)| *id == i as u64 + 1), "event ids are gap-free");
    let mut mirror = Mirror::new(StreamHeader { block_size: 64, identity: IDENTITY });
    for event in &report.kv_events {
        mirror.apply(event).unwrap();
    }
    assert_eq!(mirror.blocks.len(), report.final_registered_blocks, "the index rebuilt from events alone has the blocks the cache holds");
}

#[test]
fn a_prompt_that_cannot_fit_the_pool_fails_cleanly_instead_of_hanging() {
    let trace = Trace { requests: vec![TraceRequest { arrive: 0, tokens: (0..2000).collect(), max_new: 2, cancel_at: None }] };
    let report = run(&trace, 64 * 8, 1, 4);
    assert_eq!((report.failed, report.finished), (1, 0));
}
