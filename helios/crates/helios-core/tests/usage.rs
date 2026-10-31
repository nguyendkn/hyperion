//! `usage`, `stream_options.include_usage` and the default `max_tokens` over real HTTP, against the real scheduler on the plan path with the
//! stub backend (`[RUN on stub]`: the stub answers token 0, so the counts are what is checked, not the text).

use std::sync::Arc;

use helios_core::api::{effective_max_tokens, router, AppState};
use helios_core::scheduler::{ingress_channel, PlanPath, Scheduler};
use helios_core::step_scheduler::StepConfig;
use helios_core::tokenizer::{ChatMessageForEncode, PromptTokenizer};
use helios_core::StateMachine;
use helios_hal::{BackendConfig, ComputeBackend, KvSpec, MemoryPolicy, RankInfo, WeightSource};
use hyperion_specification::v1::{
    Activation, AttentionKind, GenerationConfig, Manifest, ModelConfig, RopeStyle, TensorFormat, TokenizerConfig, CONTRACT_REVISION,
};
use helios_core::kv::BlockHash;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
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
    let step = StepConfig { prefill_capacities: vec![128, 512, 4096], max_slots: 2, max_queue, eos_token_ids: Vec::new(), max_seq_len: 4096, decode_buckets: vec![1, 2], prefill_steps_per_decode: 1 };
    PlanPath { pool: Arc::new(pool), step, identity: BlockHash([3; 32]), hash_block_tokens: 64 }
}

const CHAT: &str = "hello world";

async fn serve(default_max_tokens: u32) -> (std::net::SocketAddr, PromptTokenizer, CancellationToken) {
    let backend: Arc<helios_hal::DynBackend> = Arc::new(helios_hal::create_backend(&stub_manifest()).expect("stub backend").backend);
    let path = plan_path(backend.as_ref().as_ref(), 8);
    let (tx, rx) = ingress_channel();
    let cancel = CancellationToken::new();
    tokio::spawn(Scheduler::new(backend, StateMachine::new(), rx).with_plan_path(path).run(cancel.clone()));
    let sm = StateMachine::new();
    sm.mark_ready().unwrap();
    let tokenizer = PromptTokenizer::new("tests/fixtures/dummy_tokenizer.json", Some(4096));
    let mut app = AppState::from(Arc::new(sm));
    app.tokenizer = Some(tokenizer.clone());
    app.ingress_tx = Some(tx);
    app.default_max_tokens = default_max_tokens;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let service = router().with_state(app);
    tokio::spawn(async move { axum::serve(listener, service).await.unwrap() });
    (addr, tokenizer, cancel)
}

async fn post(addr: std::net::SocketAddr, body: serde_json::Value) -> (u16, String) {
    let body = body.to_string();
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let head = format!("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len());
    stream.write_all(head.as_bytes()).await.unwrap();
    let mut raw = String::new();
    stream.read_to_string(&mut raw).await.unwrap();
    let status = raw.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, raw.split("\r\n\r\n").nth(1).unwrap_or("").to_string())
}

/// The JSON of every `data:` line of an SSE body (chunked transfer framing lines are skipped), `[DONE]` as `null`.
fn sse_events(body: &str) -> Vec<serde_json::Value> {
    body.lines().filter_map(|l| l.strip_prefix("data:")).map(|d| d.trim()).map(|d| if d == "[DONE]" { serde_json::Value::Null } else { serde_json::from_str(d).unwrap() }).collect()
}

#[tokio::test]
async fn usage_equals_the_tokenizer_count_and_the_default_max_tokens_applies() {
    let (addr, tokenizer, cancel) = serve(7).await;
    let prompt = tokenizer.encode_messages(&[ChatMessageForEncode { role: "user", content: CHAT }]).await.unwrap().len() as u64;
    let (status, body) = post(addr, serde_json::json!({ "model": "m", "messages": [{ "role": "user", "content": CHAT }] })).await;
    let json: serde_json::Value = serde_json::from_str(body.trim()).unwrap_or_else(|_| panic!("{body}"));
    assert_eq!(status, 200, "{json}");
    assert_eq!(json["usage"]["prompt_tokens"], prompt);
    assert_eq!(json["usage"]["completion_tokens"], 7, "max_tokens omitted: the default");
    assert_eq!(json["usage"]["total_tokens"], prompt + 7);
    let (_, body) = post(addr, serde_json::json!({ "model": "m", "max_tokens": 3, "messages": [{ "role": "user", "content": CHAT }] })).await;
    let json: serde_json::Value = serde_json::from_str(body.trim()).unwrap();
    assert_eq!(json["usage"]["completion_tokens"], 3, "an explicit max_tokens wins over the default");
    cancel.cancel();
}

#[tokio::test]
async fn the_final_usage_chunk_appears_only_when_the_request_asks_for_it() {
    let (addr, tokenizer, cancel) = serve(5).await;
    let prompt = tokenizer.encode_messages(&[ChatMessageForEncode { role: "user", content: CHAT }]).await.unwrap().len() as u64;
    let ask = |include: Option<bool>| {
        let mut body = serde_json::json!({ "model": "m", "stream": true, "messages": [{ "role": "user", "content": CHAT }] });
        if let Some(include_usage) = include {
            body["stream_options"] = serde_json::json!({ "include_usage": include_usage });
        }
        body
    };
    for plain in [None, Some(false)] {
        let (status, body) = post(addr, ask(plain)).await;
        let events = sse_events(&body);
        assert_eq!(status, 200);
        assert!(events.iter().all(|e| e.is_null() || e.get("usage").is_none()), "no usage unless asked: {body}");
    }
    let (_, body) = post(addr, ask(Some(true))).await;
    let events = sse_events(&body);
    assert!(events.last().unwrap().is_null(), "[DONE] is last");
    let usage = &events[events.len() - 2];
    assert_eq!(usage["choices"], serde_json::json!([]));
    assert_eq!(usage["usage"], serde_json::json!({ "prompt_tokens": prompt, "completion_tokens": 5, "total_tokens": prompt + 5 }));
    assert!(events[..events.len() - 2].iter().all(|e| e.get("usage").is_none()), "the counts come once, at the end");
    cancel.cancel();
}

#[test]
fn the_default_is_capped_by_the_context_left_after_the_prompt() {
    assert_eq!(effective_max_tokens(Some(5), 1024, 100, 90), 5, "an explicit value is the client's");
    assert_eq!(effective_max_tokens(None, 1024, 8192, 100), 1024);
    assert_eq!(effective_max_tokens(None, 1024, 100, 90), 10);
    assert_eq!(effective_max_tokens(None, 1024, 100, 100), 1, "never zero");
}
