//! The plan loop of the scheduler: requests come in, the state machine plans one step, the backend runs it on a blocking thread, the output goes
//! back to the state machine and its events to the streams of the requests. A request whose client has gone away is cancelled between steps and
//! its pages are freed at once. The ingress is read only while the waiting queue has room, so a full queue backs up into the bounded channel and
//! the API answers 429.

use std::collections::HashMap;

use helios_hal::{SamplingInit, StepOutput};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::kv::{KvConfig, KvManager};
use crate::load_gauge::LoadGauge;
use crate::scheduler::{InferenceRequest, PlanPath, Scheduler, StreamEvent};
use crate::step_scheduler::{Request, SeqEvent, StepScheduler};

/// Steps between two `scheduler steps` log lines.
const STEP_LOG_EVERY: u32 = 512;

type Sinks = HashMap<u64, mpsc::UnboundedSender<StreamEvent>>;

impl Scheduler {
    pub(crate) async fn run_plans(mut self, plan_path: PlanPath, cancel: CancellationToken) -> anyhow::Result<()> {
        let info = plan_path.pool.info().clone();
        let kv = KvManager::new(KvConfig { pages: info.pages, page_tokens: info.page_tokens, hash_block_tokens: plan_path.hash_block_tokens, identity: plan_path.identity })?;
        let mut machine = StepScheduler::new(plan_path.step, kv);
        let mut sinks: Sinks = HashMap::new();
        // Prompt tokens of the requests that have not produced their first token: what the gauge counts as waiting for prefill.
        let gauge = plan_path.gauge.clone();
        let mut waiting: HashMap<u64, usize> = HashMap::new();
        // The most recent decode step since the previous prefill step: the decode the policy interleaves, counted in the time per prefilled token.
        let mut decode_between: Option<std::time::Duration> = None;
        // Wall time in the backend (submit, device, take) against the whole loop, logged every STEP_LOG_EVERY steps: the host share of a step.
        let (mut in_backend, mut steps, mut window_start, mut decode_rows) = (std::time::Duration::ZERO, 0u32, std::time::Instant::now(), 0usize);
        loop {
            // Take in what has arrived; when nothing is running, wait for the next request (or the end).
            while machine.has_room() {
                let next = if machine.is_idle() {
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => None,
                        request = self.ingress.recv() => match request {
                            Some(request) => Some(request),
                            None => break,
                        },
                    }
                } else {
                    self.ingress.try_recv().ok()
                };
                let Some(request) = next else { break };
                let InferenceRequest { request_id, token_ids, sampling_params, response_tx } = request;
                let sampling = SamplingInit {
                    temperature: sampling_params.temperature.unwrap_or(0.0),
                    top_p: sampling_params.top_p.unwrap_or(1.0),
                    top_k: sampling_params.top_k.unwrap_or(0),
                    repetition_penalty: sampling_params.repetition_penalty.unwrap_or(1.0),
                    seed: sampling_params.seed.unwrap_or_else(|| os_seed(request_id)),
                };
                if sampling.temperature > 0.0 {
                    tracing::info!(request = request_id, seed = sampling.seed, given = sampling_params.seed.is_some(), temperature = sampling.temperature, top_p = sampling.top_p, top_k = sampling.top_k, repetition_penalty = sampling.repetition_penalty, "request sampling");
                }
                let prompt_tokens = token_ids.len();
                let request = Request { id: request_id, tokens: token_ids, max_new: sampling_params.max_tokens, sampling };
                match machine.enqueue(request) {
                    Ok(()) => {
                        sinks.insert(request_id, response_tx);
                        waiting.insert(request_id, prompt_tokens);
                    }
                    Err(_) => {
                        gauge.settle(prompt_tokens);
                        let _ = response_tx.send(StreamEvent::Error("the scheduler queue is full".into()));
                    }
                }
            }
            if cancel.is_cancelled() || (machine.is_idle() && self.ingress.is_closed()) {
                break;
            }
            // A client that went away stops costing pages and steps.
            let gone: Vec<u64> = sinks.iter().filter(|(_, tx)| tx.is_closed()).map(|(id, _)| *id).collect();
            for id in gone {
                machine.cancel(id);
                sinks.remove(&id);
                if let Some(tokens) = waiting.remove(&id) {
                    gauge.settle(tokens);
                }
                tracing::info!(request = id, running = machine.running(), pages_used = machine.kv().pages_used(), "client went away: sequence cancelled, its pages freed");
            }
            let Some(plan) = machine.next_plan() else {
                deliver(&mut sinks, &mut waiting, &gauge, machine.take_events());
                continue;
            };
            let prefill_tokens = match plan.capacity {
                helios_hal::Capacity::Prefill { .. } => plan.rows.iter().map(|r| r.n_tokens as usize).sum(),
                helios_hal::Capacity::Decode { .. } => 0,
            };
            decode_rows += if matches!(plan.capacity, helios_hal::Capacity::Decode { .. }) { plan.rows.len() } else { 0 };
            let backend_started = std::time::Instant::now();
            let (backend, pool) = (self.backend.clone(), plan_path.pool.clone());
            let step = tokio::task::spawn_blocking(move || -> anyhow::Result<StepOutput> {
                let ticket = backend.submit_step(&pool, &plan)?;
                backend.take_output(ticket)
            })
            .await;
            let step_time = backend_started.elapsed();
            in_backend += step_time;
            if matches!(step, Ok(Ok(_))) {
                if prefill_tokens > 0 {
                    gauge.record_prefill(prefill_tokens, step_time, decode_between.take().unwrap_or_default());
                } else {
                    decode_between = Some(step_time);
                }
            }
            steps += 1;
            if steps == STEP_LOG_EVERY {
                let wall = window_start.elapsed();
                tracing::info!(steps, backend_ms_per_step = in_backend.as_secs_f64() * 1e3 / steps as f64, loop_ms_per_step = wall.as_secs_f64() * 1e3 / steps as f64, mean_rows_per_step = decode_rows as f64 / steps as f64, "scheduler steps");
                (in_backend, steps, window_start, decode_rows) = (std::time::Duration::ZERO, 0, std::time::Instant::now(), 0);
            }
            let events = match step {
                Ok(Ok(output)) => machine.complete(&output),
                Ok(Err(error)) => machine.fail_pending(&format!("{error:#}")),
                Err(join_error) => machine.fail_pending(&format!("step task failed: {join_error}")),
            };
            deliver(&mut sinks, &mut waiting, &gauge, events);
            deliver(&mut sinks, &mut waiting, &gauge, machine.take_events());
        }
        self.ingress.close();
        Ok(())
    }
}

/// A seed for a request that gave none: the per-process random keys of the standard library (drawn from the operating system) hashed with the request
/// id and the clock, so two requests never share one and it cannot be predicted from the id.
fn os_seed(request_id: u64) -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u64(request_id);
    hasher.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
    hasher.finish()
}

/// [`route`] the events; the first event of a request means its prompt is done, so it no longer waits for prefill in the gauge.
fn deliver(sinks: &mut Sinks, waiting: &mut HashMap<u64, usize>, gauge: &LoadGauge, events: Vec<SeqEvent>) {
    for event in &events {
        let (SeqEvent::Tokens { id, .. } | SeqEvent::Finished { id, .. } | SeqEvent::Failed { id, .. }) = event;
        if let Some(tokens) = waiting.remove(id) {
            gauge.settle(tokens);
        }
    }
    route(sinks, events);
}

/// Send each event to the stream of its request; a terminal event ends the stream.
fn route(sinks: &mut Sinks, events: Vec<SeqEvent>) {
    for event in events {
        match event {
            SeqEvent::Tokens { id, tokens } => {
                if let Some(tx) = sinks.get(&id) {
                    let _ = tx.send(StreamEvent::Tokens(tokens));
                }
            }
            SeqEvent::Finished { id, reason, generated } => {
                if let Some(tx) = sinks.remove(&id) {
                    let _ = tx.send(StreamEvent::Done { finish_reason: reason, generated_tokens: generated });
                }
            }
            SeqEvent::Failed { id, message } => {
                if let Some(tx) = sinks.remove(&id) {
                    let _ = tx.send(StreamEvent::Error(message));
                }
            }
        }
    }
}
