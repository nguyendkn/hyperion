//! The scheduler really serves requests: it hands each one to the backend and streams the
//! result back. Here the real stub backend is used, which cannot generate (its forward pass
//! does not exist), so the observable behaviour is a clear error event instead of silence. The
//! success path needs a GPU backend and is covered by the hardware tests and the end-to-end
//! run documented in docs/unverified-ledger.md.

use std::sync::Arc;
use std::time::Duration;

use helios_core::scheduler::{ingress_channel, InferenceRequest, SamplingParams, Scheduler, StreamEvent};
use helios_core::StateMachine;
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
            max_seq_len: 8,
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

fn request(id: u64) -> (InferenceRequest, mpsc::UnboundedReceiver<StreamEvent>) {
    let (tx, rx) = mpsc::unbounded_channel();
    let request = InferenceRequest {
        request_id: id,
        token_ids: vec![1, 2, 3],
        sampling_params: SamplingParams { temperature: None, top_p: None, max_tokens: 4 },
        response_tx: tx,
    };
    (request, rx)
}

async fn next(rx: &mut mpsc::UnboundedReceiver<StreamEvent>) -> Option<StreamEvent> {
    tokio::time::timeout(Duration::from_secs(5), rx.recv()).await.expect("the scheduler must answer within 5 s")
}

#[tokio::test]
async fn a_backend_that_cannot_generate_gets_a_clear_error_event() {
    let backend = helios_hal::create_backend(&stub_manifest()).expect("stub backend").backend;
    let (tx, rx_in) = ingress_channel();
    let cancel = CancellationToken::new();
    let task = tokio::spawn(Scheduler::new(Arc::new(backend), StateMachine::new(), rx_in).run(cancel.clone()));

    let (req, mut events) = request(1);
    tx.send(req).await.unwrap();
    match next(&mut events).await {
        Some(StreamEvent::Error(message)) => assert!(message.contains("does not support `generate`"), "{message}"),
        other => panic!("expected an Error event, got {other:?}"),
    }
    // The terminal event ends the stream.
    assert!(next(&mut events).await.is_none());

    // The scheduler keeps serving after a failed request.
    let (req, mut events) = request(2);
    tx.send(req).await.unwrap();
    assert!(matches!(next(&mut events).await, Some(StreamEvent::Error(_))));

    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), task).await.expect("exits on cancel").unwrap().unwrap();
}

#[tokio::test]
async fn the_scheduler_exits_when_the_ingress_is_closed() {
    let backend = helios_hal::create_backend(&stub_manifest()).expect("stub backend").backend;
    let (tx, rx_in) = ingress_channel();
    let task = tokio::spawn(Scheduler::new(Arc::new(backend), StateMachine::new(), rx_in).run(CancellationToken::new()));
    drop(tx);
    tokio::time::timeout(Duration::from_secs(5), task).await.expect("exits when no sender is left").unwrap().unwrap();
}
