//! The plan loop of the scheduler: requests come in, the state machine plans one step, the backend runs it on a blocking thread, the output goes
//! back to the state machine and its events to the streams of the requests. A request whose client has gone away is cancelled between steps and
//! its pages are freed at once. The ingress is read only while the waiting queue has room, so a full queue backs up into the bounded channel and
//! the API answers 429.

use std::collections::HashMap;

use helios_hal::{SamplingInit, StepOutput};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::kv::{KvConfig, KvManager};
use crate::scheduler::{InferenceRequest, PlanPath, Scheduler, StreamEvent};
use crate::step_scheduler::{Request, SeqEvent, StepScheduler};

type Sinks = HashMap<u64, mpsc::UnboundedSender<StreamEvent>>;

impl Scheduler {
    pub(crate) async fn run_plans(mut self, plan_path: PlanPath, cancel: CancellationToken) -> anyhow::Result<()> {
        let info = plan_path.pool.info().clone();
        let kv = KvManager::new(KvConfig { pages: info.pages, page_tokens: info.page_tokens, hash_block_tokens: plan_path.hash_block_tokens, identity: plan_path.identity })?;
        let mut machine = StepScheduler::new(plan_path.step, kv);
        let mut sinks: Sinks = HashMap::new();
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
                    top_k: 0,
                    repetition_penalty: 1.0,
                    seed: request_id,
                };
                let request = Request { id: request_id, tokens: token_ids, max_new: sampling_params.max_tokens, sampling };
                match machine.enqueue(request) {
                    Ok(()) => {
                        sinks.insert(request_id, response_tx);
                    }
                    Err(_) => {
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
                tracing::info!(request = id, running = machine.running(), pages_used = machine.kv().pages_used(), "client went away: sequence cancelled, its pages freed");
            }
            let Some(plan) = machine.next_plan() else {
                route(&mut sinks, machine.take_events());
                continue;
            };
            let (backend, pool) = (self.backend.clone(), plan_path.pool.clone());
            let step = tokio::task::spawn_blocking(move || -> anyhow::Result<StepOutput> {
                let ticket = backend.submit_step(&pool, &plan)?;
                backend.take_output(ticket)
            })
            .await;
            let events = match step {
                Ok(Ok(output)) => machine.complete(&output),
                Ok(Err(error)) => machine.fail_pending(&format!("{error:#}")),
                Err(join_error) => machine.fail_pending(&format!("step task failed: {join_error}")),
            };
            route(&mut sinks, events);
            route(&mut sinks, machine.take_events());
        }
        self.ingress.close();
        Ok(())
    }
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
