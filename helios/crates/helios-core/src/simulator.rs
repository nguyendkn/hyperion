//! A deterministic simulator of the scheduler (phase 07): replay a trace of requests (arrival tick, prompt, length, optional cancellation)
//! against the plan state machine and any `ComputeBackend` that implements `submit_step` (the single-rank CPU stub in tests). Nothing in it
//! reads a clock or a random source, so the same trace gives the same plans, byte for byte; the report holds the encoded plans, the prefix hit
//! rate, the page usage and the KV events. One tick is one scheduler step; idle time between arrivals is skipped.

use std::collections::BTreeMap;

use anyhow::Result;
use helios_hal::step::SamplingInit;
use helios_hal::{ComputeBackend, KvPool};

use crate::kv::{KvConfig, KvEvent, KvManager, SeqId};
use crate::scheduler::FinishReason;
use crate::step_scheduler::{Request, SeqEvent, StepConfig, StepScheduler};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceRequest {
    /// The tick at which the request reaches the scheduler.
    pub arrive: u64,
    pub tokens: Vec<u32>,
    pub max_new: u32,
    /// The tick at which the client goes away, if it does.
    pub cancel_at: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Trace {
    pub requests: Vec<TraceRequest>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    /// Every plan, in its binary encoding, in order.
    pub plans: Vec<Vec<u8>>,
    /// The tokens each finished request received, by request id (its index in the trace plus one).
    pub generated: BTreeMap<SeqId, Vec<u32>>,
    pub finished: usize,
    pub cancelled: usize,
    /// Refused because the waiting queue was full (the API answers 429).
    pub rejected: usize,
    pub failed: usize,
    pub hit_rate: f64,
    pub peak_pages: usize,
    pub kv_events: Vec<KvEvent>,
    pub ticks: u64,
    /// Pages in use and registered blocks when the trace ended.
    pub final_pages_used: usize,
    pub final_registered_blocks: usize,
    /// Decode steps by the number of live rows they carried (batch size -> steps).
    pub decode_rows: BTreeMap<usize, usize>,
    /// Most sequences running (holding a slot) at once.
    pub peak_running: usize,
    /// Ticks from a request's arrival to its first token, by request id (the queueing delay plus its prefill).
    pub first_token_ticks: BTreeMap<SeqId, u64>,
}

/// Replay `trace`. `pool` is the pool the backend committed; the page geometry comes from it.
pub fn replay(trace: &Trace, cfg: StepConfig, identity: crate::kv::BlockHash, hash_block_tokens: usize, backend: &dyn ComputeBackend, pool: &KvPool) -> Result<Report> {
    let info = pool.info();
    let kv = KvManager::new(KvConfig { pages: info.pages, page_tokens: info.page_tokens, hash_block_tokens, identity })?;
    let mut scheduler = StepScheduler::new(cfg, kv);
    let events = scheduler.kv_mut().events().subscribe();
    let mut report = Report { plans: Vec::new(), generated: BTreeMap::new(), finished: 0, cancelled: 0, rejected: 0, failed: 0, hit_rate: 0.0, peak_pages: 0, kv_events: Vec::new(), ticks: 0, final_pages_used: 0, final_registered_blocks: 0, decode_rows: BTreeMap::new(), peak_running: 0, first_token_ticks: BTreeMap::new() };
    let mut tick = 0u64;
    let mut next = 0usize;
    let mut order: Vec<usize> = (0..trace.requests.len()).collect();
    order.sort_by_key(|&i| (trace.requests[i].arrive, i));
    let guard = 1_000_000u64;
    loop {
        while next < order.len() && trace.requests[order[next]].arrive <= tick {
            let index = order[next];
            next += 1;
            let r = &trace.requests[index];
            let request = Request { id: index as u64 + 1, tokens: r.tokens.clone(), max_new: r.max_new, sampling: SamplingInit::GREEDY };
            if scheduler.enqueue(request).is_err() {
                report.rejected += 1;
            }
        }
        for (index, r) in trace.requests.iter().enumerate() {
            if r.cancel_at == Some(tick) && scheduler.cancel(index as u64 + 1) {
                report.cancelled += 1;
            }
        }
        match scheduler.next_plan() {
            Some(plan) => {
                report.plans.push(plan.encode());
                if matches!(plan.capacity, helios_hal::step::Capacity::Decode { .. }) {
                    *report.decode_rows.entry(plan.rows.len()).or_default() += 1;
                }
                let output = backend.take_output(backend.submit_step(pool, &plan)?)?;
                let produced = scheduler.complete(&output);
                record(&mut report, produced, tick, trace);
            }
            None => {
                if next >= order.len() && scheduler.is_idle() {
                    record(&mut report, scheduler.take_events(), tick, trace);
                    break;
                }
                // Idle until the next arrival.
                if let Some(&i) = order.get(next) {
                    tick = tick.max(trace.requests[i].arrive.saturating_sub(1));
                }
            }
        }
        record(&mut report, scheduler.take_events(), tick, trace);
        report.peak_running = report.peak_running.max(scheduler.running());
        report.peak_pages = report.peak_pages.max(scheduler.kv().pages_used());
        tick += 1;
        anyhow::ensure!(tick < guard, "the simulation did not finish within {guard} ticks");
    }
    report.ticks = tick;
    report.hit_rate = scheduler.kv().stats().hit_rate();
    report.final_pages_used = scheduler.kv().pages_used();
    report.final_registered_blocks = scheduler.kv().registered_blocks();
    report.kv_events = events.try_iter().collect();
    Ok(report)
}

fn record(report: &mut Report, events: Vec<SeqEvent>, tick: u64, trace: &Trace) {
    for event in events {
        match event {
            SeqEvent::Tokens { id, tokens } => {
                report.first_token_ticks.entry(id).or_insert_with(|| tick.saturating_sub(trace.requests[id as usize - 1].arrive));
                report.generated.entry(id).or_default().extend(tokens);
            }
            SeqEvent::Finished { reason, .. } => {
                debug_assert!(matches!(reason, FinishReason::Length | FinishReason::Stop));
                report.finished += 1;
            }
            SeqEvent::Failed { .. } => report.failed += 1,
        }
    }
}

/// A deterministic multi-turn agent workload: `sessions` sessions that share one system prompt, each making `turns` requests; every turn
/// appends the previous reply (zeros, what the CPU stub returns) and a fresh user message and tool result to the whole earlier prompt. Sessions
/// start `stride` ticks apart and a session's turns `gap` ticks apart.
pub fn agent_sessions(sessions: usize, turns: usize, system_tokens: usize, message_tokens: usize, max_new: u32, stride: u64, gap: u64) -> Trace {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next_token = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        1000 + ((state >> 33) % 50_000) as u32
    };
    let system: Vec<u32> = (0..system_tokens).map(|_| next_token()).collect();
    let mut trace = Trace::default();
    for session in 0..sessions {
        let mut prompt = system.clone();
        for turn in 0..turns {
            prompt.extend((0..message_tokens).map(|_| next_token()));
            trace.requests.push(TraceRequest { arrive: session as u64 * stride + turn as u64 * gap, tokens: prompt.clone(), max_new, cancel_at: None });
            prompt.extend(std::iter::repeat_n(0u32, max_new as usize));
        }
    }
    trace
}
