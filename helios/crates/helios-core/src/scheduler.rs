//! Continuous-batching scheduler. Splits work into prefill (long, batch-1-ish)
//! and decode (short, fully batched) phases and interleaves them on the single
//! device. Owns the `Arc<dyn ComputeBackend>` and the request queue.
//!
//! This is a skeleton: the algorithm (iteration loop, ready-set selection,
//! preemption on KV pressure) lands once the HAL backends are wired enough to
//! return real logits. The structure here defines the seams — including the
//! `InferenceRequest` contract the API handler enqueues and the `StreamEvent`
//! contract the scheduler streams back.

use std::sync::Arc;

use helios_hal::DynBackend;
use tokio::sync::mpsc;

use crate::state::StateMachine;

/// Backpressure bound on the ingress (API → scheduler) queue. The handler
/// gets a `SendError` (mapped to HTTP 503/429) the moment the queue is full,
/// instead of unbounded buffering inflating host memory under load. 32 is a
/// first guess — small enough to surface overload fast, large enough that a
/// brief client burst doesn't trip it. Tunable once we have real load shape.
pub const INGRESS_CAPACITY: usize = 32;

/// Numerically-stable sampling knobs handed to the scheduler per request.
/// Mirrors a subset of the OpenAI `ChatCompletionRequest` fields; unknown
/// knobs (top_k, frequency_penalty, …) land later via an extensible enum or
/// a nested struct — YAGNI until a caller needs them.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Softmax temperature. `None` → greedy / deterministic (treat as 1.0 only
    /// if `top_p` is also unset; the scheduler picks greedy when both are
    /// absent to make smoke testing deterministic).
    pub temperature: Option<f32>,
    /// Nucleus cumulative probability threshold. `None` → no top-p filter.
    pub top_p: Option<f32>,
    /// Hard cap on generated tokens *after* the prompt. The scheduler stops
    /// decoding at this count even if no EOS is sampled.
    pub max_tokens: u32,
}

/// Why a finished stream ended. Mirrors the OpenAI `finish_reason` vocabulary
/// so the handler can pass it straight through without a translation layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// `max_tokens` hit before an EOS was sampled.
    Length,
    /// The model sampled its end-of-sequence id.
    Stop,
}

/// One message on the response stream. The scheduler owns the lifecycle: it
/// emits zero or more `Tokens` followed by exactly one terminal event
/// (`Done` or `Error`). After the terminal event the sender is dropped, so
/// the handler's `recv()` returns `None` and it knows the stream is closed.
#[derive(Debug)]
pub enum StreamEvent {
    /// A decoded chunk of token ids. Currently one id per step; grouping is
    /// a later optimization (speculative decode, chunked output).
    Tokens(Vec<u32>),
    /// Stream completed normally. Carries the reason so the handler can set
    /// `finish_reason` on the final `Choice`.
    Done {
        finish_reason: FinishReason,
        /// Total generated token count, for the handler's usage accounting.
        generated_tokens: u32,
    },
    /// Stream aborted with an error (prefill failure, device error, …). The
    /// handler maps this to a 500-range response or an SSE `error` event.
    Error(String),
}

/// A single request submitted by the API handler after tokenization. This is
/// the wire contract across the API↔Scheduler channel — anything not listed
/// here is invisible to the scheduler, so it must be the *minimal* set needed
/// to drive prefill + decode and return a result.
#[derive(Debug)]
pub struct InferenceRequest {
    /// Monotonic request id — comes from `api::chrono_like_id`. Cheaper than
    /// a UUID and unique within one server run, which is all the scheduler
    /// needs for slot tracking and log correlation.
    pub request_id: u64,
    /// Token ids from `PromptTokenizer::encode_messages`. The scheduler owns
    /// these from prefill onward; the API handler drops its copy after
    /// enqueuing.
    pub token_ids: Vec<u32>,
    /// Sampling configuration for the decode phase.
    pub sampling_params: SamplingParams,
    /// One-shot response sink. The scheduler emits `StreamEvent`s here; the
    /// terminal event (`Done`/`Error`) ends the stream and the sender is
    /// dropped so the handler's `recv()` returns `None`.
    ///
    /// `UnboundedSender` keeps the scheduler from ever blocking on a slow
    /// reader — at the cost of unbounded buffering. Acceptable for the
    /// current single-request vertical slice; once we fan out multiple
    /// in-flight requests and care about KV/head memory pressure, this
    /// becomes a bounded channel with a credit protocol. The *ingress* side
    /// (API → scheduler) is already bounded — see [`INGRESS_CAPACITY`].
    pub response_tx: mpsc::UnboundedSender<StreamEvent>,
}

/// Holds everything the scheduler needs. Built by the CLI once the HAL factory
/// has returned a backend and triage has cleared. Owns the ingress receiver;
/// the matching sender is held by `AppState` so the API handler can enqueue.
pub struct Scheduler {
    pub backend: Arc<DynBackend>,
    pub state: Arc<StateMachine>,
    /// When set, requests run through StepPlans on this pool; otherwise through the backend generate.
    pub(crate) plan_path: Option<PlanPath>,
    /// Ingress queue. Dropped when `run()` returns, which immediately fails
    /// any in-flight `send()` on the API side → handler returns 503.
    pub(crate) ingress: mpsc::Receiver<InferenceRequest>,
}

/// What the plan path needs: the KV pool the backend committed (`commit_kv`), the scheduler limits and the identity the prefix hashes chain from.
pub struct PlanPath {
    pub pool: Arc<helios_hal::KvPool>,
    pub step: crate::step_scheduler::StepConfig,
    pub identity: crate::kv::BlockHash,
    /// Tokens per prefix-hash block (a multiple of the page size of the pool).
    pub hash_block_tokens: usize,
}

/// The sender half handed to the API layer. Stored in `AppState` so the
/// handler can enqueue `InferenceRequest`s without owning the scheduler.
pub type IngressSender = mpsc::Sender<InferenceRequest>;

/// Constructs the bounded ingress channel and returns both halves. The CLI
/// keeps the `Scheduler` (with the receiver) and threads the sender into
/// `AppState`. Splitting construction here keeps the channel lifetime tied to
/// the scheduler task, not to a struct that outlives it.
pub fn ingress_channel() -> (IngressSender, mpsc::Receiver<InferenceRequest>) {
    mpsc::channel(INGRESS_CAPACITY)
}

impl Scheduler {
    /// Build from a backend, a state machine, and the receiver end of the
    /// ingress channel. The sender end goes into `AppState`.
    ///
    /// Takes `Arc<DynBackend>` (not `DynBackend`) so the CLI can keep its own
    /// clone to call `shutdown()` on the backend after the server stops —
    /// the scheduler's task must not be the sole owner of device teardown.
    pub fn new(
        backend: Arc<DynBackend>,
        state: StateMachine,
        ingress: mpsc::Receiver<InferenceRequest>,
    ) -> Self {
        Self {
            backend,
            state: Arc::new(state),
            ingress,
            plan_path: None,
        }
    }

    /// Serve through StepPlans on the committed KV pool (`submit_step` / `take_output`) instead of `generate`.
    pub fn with_plan_path(mut self, plan_path: PlanPath) -> Self {
        self.plan_path = Some(plan_path);
        self
    }

    /// Main scheduling loop. TODO: repeatedly pick the next prefill or decode
    /// batch, call the backend, and feed logits back to waiting requests.
    /// Returns when the engine enters `Draining` and the queue is empty.
    ///
    /// In the current no-op form this drains the ingress receiver so it is
    /// dropped promptly on return (failing any subsequent API `send()`), but
    /// does **not** process requests. The CLI's supervisor task is what moves
    /// the engine to `Stopped` when this task ends — `run()` intentionally
    /// does not touch engine state on its own exit, so a panic here still
    /// leaves the supervisor (observing the `JoinHandle`) to shut the engine
    /// down rather than relying on a post-`run()` line that a panic skips.
    /// Main scheduling loop. TODO: repeatedly pick the next prefill or decode
    /// batch, call the backend, and feed logits back to waiting requests.
    ///
    /// `cancel` is the cooperative shutdown signal from the CLI. The real
    /// loop will check `cancel.is_cancelled()` between device calls — *not*
    /// mid-call — so a CUDA operation already in flight finishes before the
    /// task exits, avoiding the use-after-free that `abort()` would cause on
    /// the device memory the scheduler is touching. Closing the ingress
    /// receiver (below) fails any in-flight `send()` on the API side.
    ///
    /// In the current no-op form the loop drains the ingress receiver so it
    /// is dropped promptly on return (failing any subsequent API `send()`),
    /// but does **not** process requests. The CLI's supervisor task is what
    /// moves the engine to `Stopped` when this task ends — `run()` does not
    /// touch engine state on its own exit, so a panic here still leaves the
    /// supervisor (observing the `JoinHandle`) to shut the engine down
    /// rather than relying on a post-`run()` line that a panic skips.
    pub async fn run(mut self, cancel: tokio_util::sync::CancellationToken) -> anyhow::Result<()> {
        if let Some(plan_path) = self.plan_path.take() {
            return self.run_plans(plan_path, cancel).await;
        }
        // One request at a time: each is handed to the backend's `generate`, which runs on a
        // blocking thread (a device call must not stall the async runtime) and streams tokens
        // back as they are chosen. Cancellation is checked between requests, never mid-call,
        // so a CUDA operation already in flight finishes before the task exits. Batching more
        // than one sequence per step is the later scheduling work; this loop makes the single
        // request path real.
        loop {
            tokio::select! {
                biased; // cancellation takes precedence over more ingress.
                _ = cancel.cancelled() => break,
                request = self.ingress.recv() => match request {
                    Some(request) => self.serve(request).await,
                    // Every sender is gone: nothing can arrive any more.
                    None => break,
                },
            }
        }
        // Close explicitly to make the fail-fast contract obvious to
        // readers; the field would drop on return anyway, but a closed
        // receiver is what makes `send()` on the API side return
        // `SendError` immediately (rather than blocking until full drop).
        self.ingress.close();
        Ok(())
    }

    /// Run one request on the backend and stream its events: `Tokens` as they are chosen,
    /// then exactly one terminal `Done` or `Error`. A failed or panicking backend call becomes
    /// an `Error` event; it never takes the scheduler down.
    async fn serve(&self, request: InferenceRequest) {
        let InferenceRequest { request_id, token_ids, sampling_params, response_tx } = request;
        let max_new = sampling_params.max_tokens as usize;
        // The request id seeds the random generator: a retry is a new request, and a replay of
        // the same id (tests, debugging) reproduces the same text.
        let sampling = helios_hal::SamplingConfig {
            temperature: sampling_params.temperature,
            top_p: sampling_params.top_p,
            seed: request_id,
        };
        let backend = self.backend.clone();
        let sink = response_tx.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            let mut generated = 0u32;
            let stop = backend.generate(&token_ids, max_new, &sampling, &mut |token| {
                generated += 1;
                // The receiver may be gone (client disconnected); finishing the generation
                // is harmless, so a failed send is ignored.
                let _ = sink.send(StreamEvent::Tokens(vec![token]));
            })?;
            Ok::<_, anyhow::Error>((stop, generated))
        })
        .await;

        let terminal = match outcome {
            Ok(Ok((stop, generated))) => StreamEvent::Done {
                finish_reason: match stop {
                    helios_hal::StopReason::EndOfSequence => FinishReason::Stop,
                    helios_hal::StopReason::MaxTokens => FinishReason::Length,
                },
                generated_tokens: generated,
            },
            Ok(Err(error)) => StreamEvent::Error(format!("{error:#}")),
            Err(join_error) => StreamEvent::Error(format!("generation task failed: {join_error}")),
        };
        tracing::debug!(request_id, "request finished");
        let _ = response_tx.send(terminal);
    }
}
