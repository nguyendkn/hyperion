//! The async scheduler on the plan path with the real stub backend taken through the load order (`[RUN on stub]`): requests stream tokens and
//! finish, concurrent requests all complete, a client that goes away does not stop the others, a prompt that does not fit is an error event.

use std::sync::Arc;
use std::time::Duration;

use helios_core::kv::BlockHash;
use helios_core::scheduler::{ingress_channel, InferenceRequest, PlanPath, SamplingParams, Scheduler, StreamEvent};
use helios_core::step_scheduler::StepConfig;
use helios_core::{FinishReason, StateMachine};
use helios_hal::{BackendConfig, ComputeBackend, KvSpec, MemoryPolicy, RankInfo, WeightSource};
use hyperion_specification::v1::{
    Activation, AttentionKind, GenerationConfig, Manifest, ModelConfig, RopeStyle, TensorFormat, TokenizerConfig, CONTRACT_REVISION,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

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
            max_seq_len: 4096,
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

fn plan_path(backend: &dyn ComputeBackend, max_queue: usize) -> PlanPath {
    let config = BackendConfig::from_manifest(&stub_manifest()).unwrap();
    backend.initialize_device(&RankInfo::single()).unwrap();
    let model = backend.load_model(&config, &WeightSource { files: Vec::new(), slices: Vec::new() }).unwrap();
    let reservation = backend.reserve_kv(&model, &KvSpec::default()).unwrap();
    let graphs = backend.capture_graphs(&model, &reservation).unwrap();
    let pool = backend.commit_kv(reservation, &graphs, &MemoryPolicy { reserve_bytes: 0, max_tokens: 64 * 200 }).unwrap();
    // The stub always answers token 0: no end-of-sequence id, so a request runs to its length.
    let step = StepConfig { prefill_capacities: vec![128, 512, 4096], max_slots: 2, max_queue, eos_token_ids: Vec::new(), max_seq_len: 4096 };
    PlanPath { pool: Arc::new(pool), step, identity: BlockHash([3; 32]), hash_block_tokens: 64 }
}

fn request(id: u64, prompt: usize, max_tokens: u32) -> (InferenceRequest, mpsc::UnboundedReceiver<StreamEvent>) {
    let (tx, rx) = mpsc::unbounded_channel();
    let request = InferenceRequest { request_id: id, token_ids: (0..prompt as u32).map(|t| t + id as u32 * 10_000).collect(), sampling_params: SamplingParams { temperature: None, top_p: None, max_tokens }, response_tx: tx };
    (request, rx)
}

async fn next(rx: &mut mpsc::UnboundedReceiver<StreamEvent>) -> Option<StreamEvent> {
    tokio::time::timeout(Duration::from_secs(10), rx.recv()).await.expect("the scheduler must answer within 10 s")
}

async fn start(max_queue: usize) -> (helios_core::IngressSender, CancellationToken, tokio::task::JoinHandle<anyhow::Result<()>>) {
    let backend: Arc<helios_hal::DynBackend> = Arc::new(helios_hal::create_backend(&stub_manifest()).expect("stub backend").backend);
    let path = plan_path(backend.as_ref().as_ref(), max_queue);
    let (tx, rx) = ingress_channel();
    let cancel = CancellationToken::new();
    let scheduler = Scheduler::new(backend, StateMachine::new(), rx).with_plan_path(path);
    let task = tokio::spawn(scheduler.run(cancel.clone()));
    (tx, cancel, task)
}

#[tokio::test]
async fn a_request_streams_its_tokens_one_at_a_time_and_ends_with_done() {
    let (tx, cancel, task) = start(8).await;
    let (req, mut events) = request(1, 300, 5);
    tx.send(req).await.unwrap();
    let mut tokens = 0;
    loop {
        match next(&mut events).await.expect("a terminal event comes before the stream closes") {
            StreamEvent::Tokens(ids) => tokens += ids.len(),
            StreamEvent::Done { finish_reason, generated_tokens } => {
                assert_eq!((finish_reason, generated_tokens, tokens), (FinishReason::Length, 5, 5));
                break;
            }
            StreamEvent::Error(message) => panic!("{message}"),
        }
    }
    assert!(next(&mut events).await.is_none(), "the terminal event ends the stream");
    cancel.cancel();
    task.await.unwrap().unwrap();
}

#[tokio::test]
async fn concurrent_requests_all_finish_and_a_vanished_client_does_not_stop_the_others() {
    let (tx, cancel, task) = start(8).await;
    let mut streams = Vec::new();
    for id in 1..=5u64 {
        let (req, rx) = request(id, 200, 6);
        tx.send(req).await.unwrap();
        if id == 3 {
            drop(rx); // this client went away before reading anything
        } else {
            streams.push((id, rx));
        }
    }
    for (id, mut rx) in streams {
        let mut count = 0;
        loop {
            match next(&mut rx).await.expect("terminal") {
                StreamEvent::Tokens(ids) => count += ids.len(),
                StreamEvent::Done { generated_tokens, .. } => {
                    assert_eq!((generated_tokens as usize, count), (6, 6), "request {id}");
                    break;
                }
                StreamEvent::Error(message) => panic!("request {id}: {message}"),
            }
        }
    }
    cancel.cancel();
    task.await.unwrap().unwrap();
}

#[tokio::test]
async fn a_prompt_longer_than_the_context_is_an_error_event_and_the_scheduler_keeps_serving() {
    let (tx, cancel, task) = start(8).await;
    let (long, mut events) = request(1, 5000, 3);
    tx.send(long).await.unwrap();
    assert!(matches!(next(&mut events).await, Some(StreamEvent::Error(m)) if m.contains("does not fit")));
    let (ok, mut events) = request(2, 100, 2);
    tx.send(ok).await.unwrap();
    assert!(matches!(next(&mut events).await, Some(StreamEvent::Tokens(_))));
    cancel.cancel();
    task.await.unwrap().unwrap();
}
