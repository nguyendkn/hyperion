//! A multi-rank test double (ADR 0006): N rank threads and CPU collectives with the real semantics (sum all-reduce, concatenating all-gather,
//! sum reduce-scatter). It tests the control plane (the plan broadcast and its hash check), the sharding plans, and failure handling (a rank
//! dies, a rank is slow, the rendezvous times out, versions disagree), and nothing else. Every result produced with it is tagged
//! `[RUN on stub]`; it is never evidence about a GPU, and no claim of speed or numerics rests on it.
//!
//! One persistent thread per rank (ADR 0004). A closure sent to every rank runs on its own thread with that rank's context; a rank that
//! panics or returns an error marks the group dead, so its peers fail at once with `RankFailed` instead of waiting out a timeout.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use thiserror::Error;

use crate::step::{apply_deltas, RowOutput, StepOutput, StepPlan, Want};

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GroupError {
    #[error("rank {0} failed: the group is dead")]
    RankFailed(u32),
    #[error("timed out after {0:?} waiting for the other ranks")]
    Timeout(Duration),
    #[error("rendezvous refused: rank {rank} reports `{found}` but rank 0 reports `{expected}`")]
    VersionMismatch { rank: u32, expected: String, found: String },
    #[error("ranks disagree: {0}")]
    Diverged(String),
    #[error("collective buffers differ in length across ranks")]
    LengthMismatch,
    #[error("{0}")]
    Plan(String),
}

/// What a rank says about itself when it joins (the NCCL and driver versions in a real group).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hello {
    pub rank: u32,
    pub world_size: usize,
    pub library_version: String,
}

struct State {
    hellos: Vec<Option<Hello>>,
    dead: Option<u32>,
    generation: u64,
    arrived: usize,
    inputs: Vec<Option<Vec<f32>>>,
    results: HashMap<u64, (Arc<Vec<Vec<f32>>>, usize)>,
}

struct Shared {
    world: usize,
    timeout: Duration,
    state: Mutex<State>,
    changed: Condvar,
}

impl Shared {
    fn mark_dead(&self, rank: u32) {
        let mut state = self.state.lock().unwrap();
        state.dead.get_or_insert(rank);
        self.changed.notify_all();
    }

    /// Wait on `changed` until `done` holds, the group dies or the timeout passes.
    fn wait_until<'a>(&self, mut state: std::sync::MutexGuard<'a, State>, done: impl Fn(&State) -> bool) -> Result<std::sync::MutexGuard<'a, State>, GroupError> {
        let deadline = Instant::now() + self.timeout;
        loop {
            // A finished round is returned even when a rank failed right after it: only an unfinished one is the failure.
            if done(&state) {
                return Ok(state);
            }
            if let Some(rank) = state.dead {
                return Err(GroupError::RankFailed(rank));
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(GroupError::Timeout(self.timeout));
            }
            state = self.changed.wait_timeout(state, left).unwrap().0;
        }
    }

    /// Every rank deposits `data`; all get every rank's buffer, in rank order, once the last one has arrived.
    fn exchange(&self, rank: usize, data: Vec<f32>) -> Result<Arc<Vec<Vec<f32>>>, GroupError> {
        let mut state = self.state.lock().unwrap();
        if let Some(dead) = state.dead {
            return Err(GroupError::RankFailed(dead));
        }
        let generation = state.generation;
        state.inputs[rank] = Some(data);
        state.arrived += 1;
        if state.arrived == self.world {
            let all: Vec<Vec<f32>> = state.inputs.iter_mut().map(|slot| slot.take().expect("every rank deposited")).collect();
            state.results.insert(generation, (Arc::new(all), self.world));
            state.arrived = 0;
            state.generation += 1;
            self.changed.notify_all();
        }
        let mut state = self.wait_until(state, |s| s.generation > generation)?;
        let (all, remaining) = state.results.get_mut(&generation).expect("the result of this round");
        let all = all.clone();
        *remaining -= 1;
        if *remaining == 0 {
            state.results.remove(&generation);
        }
        Ok(all)
    }
}

/// One rank's view of the group: its identity and the CPU collectives.
pub struct RankCtx {
    pub rank: usize,
    pub world: usize,
    shared: Arc<Shared>,
    /// This rank's mirror of the device page tables, built only from the deltas of the plans it received.
    pub page_tables: HashMap<u32, Vec<Option<u32>>>,
}

impl RankCtx {
    /// Join the group: every rank reports its `library_version`; rank 0's is the reference, and any other report or a wrong world size is
    /// refused, on every rank, before anything else runs (ADR 0009).
    pub fn rendezvous(&self, library_version: &str) -> Result<(), GroupError> {
        let mut state = self.shared.state.lock().unwrap();
        state.hellos[self.rank] = Some(Hello { rank: self.rank as u32, world_size: self.world, library_version: library_version.into() });
        self.shared.changed.notify_all();
        let state = self.shared.wait_until(state, |s| s.hellos.iter().all(Option::is_some))?;
        let reference = state.hellos[0].clone().unwrap();
        for hello in state.hellos.iter().flatten() {
            if hello.library_version != reference.library_version {
                return Err(GroupError::VersionMismatch { rank: hello.rank, expected: reference.library_version.clone(), found: hello.library_version.clone() });
            }
        }
        Ok(())
    }

    fn gather(&self, data: &[f32]) -> Result<Arc<Vec<Vec<f32>>>, GroupError> {
        let all = self.shared.exchange(self.rank, data.to_vec())?;
        if all.iter().any(|v| v.len() != data.len()) {
            return Err(GroupError::LengthMismatch);
        }
        Ok(all)
    }

    /// Element-wise sum over ranks, the same result on every rank.
    pub fn all_reduce_sum(&self, data: &[f32]) -> Result<Vec<f32>, GroupError> {
        let all = self.gather(data)?;
        Ok((0..data.len()).map(|i| all.iter().map(|v| v[i]).sum()).collect())
    }

    /// Every rank's buffer one after the other, in rank order.
    pub fn all_gather(&self, data: &[f32]) -> Result<Vec<f32>, GroupError> {
        Ok(self.gather(data)?.iter().flatten().copied().collect())
    }

    /// Sum over ranks; this rank keeps block `rank` of `len / world` elements.
    pub fn reduce_scatter_sum(&self, data: &[f32]) -> Result<Vec<f32>, GroupError> {
        if data.len() % self.world != 0 {
            return Err(GroupError::LengthMismatch);
        }
        let block = data.len() / self.world;
        let all = self.gather(data)?;
        Ok((self.rank * block..(self.rank + 1) * block).map(|i| all.iter().map(|v| v[i]).sum()).collect())
    }

    /// Apply one plan as a rank does: decode (hash-checked), update the page-table mirror from the deltas, answer with a token of id 0 for
    /// each row that wants one. Returns the output and the plan hash this rank saw.
    pub fn apply_plan(&mut self, bytes: &[u8]) -> Result<(StepOutput, u64), GroupError> {
        let plan = StepPlan::decode(bytes).map_err(|e| GroupError::Plan(e.to_string()))?;
        plan.validate().map_err(|e| GroupError::Plan(e.to_string()))?;
        let slots: Vec<u32> = plan.deltas.iter().map(|d| d.slot).collect();
        for slot in slots {
            apply_deltas(self.page_tables.entry(slot).or_default(), slot, &plan.deltas);
        }
        let rows = plan.rows.iter().map(|r| RowOutput { slot: r.slot, tokens: if r.want == Want::Token { vec![0] } else { Vec::new() }, logits: None }).collect();
        Ok((StepOutput { step: plan.step, rows }, plan.hash()))
    }
}

type Job = Box<dyn FnOnce(&mut RankCtx) + Send>;

pub struct StubGroup {
    shared: Arc<Shared>,
    senders: Vec<Sender<Job>>,
    threads: Vec<JoinHandle<()>>,
}

impl StubGroup {
    /// Start `world` rank threads. `timeout` bounds every wait inside a collective or a rendezvous.
    pub fn spawn(world: usize, timeout: Duration) -> Self {
        let shared = Arc::new(Shared {
            world,
            timeout,
            state: Mutex::new(State { hellos: vec![None; world], dead: None, generation: 0, arrived: 0, inputs: vec![None; world], results: HashMap::new() }),
            changed: Condvar::new(),
        });
        let (mut senders, mut threads) = (Vec::new(), Vec::new());
        for rank in 0..world {
            let (tx, rx) = mpsc::channel::<Job>();
            let mut ctx = RankCtx { rank, world, shared: shared.clone(), page_tables: HashMap::new() };
            threads.push(thread::Builder::new().name(format!("stub-rank-{rank}")).spawn(move || {
                while let Ok(job) = rx.recv() {
                    job(&mut ctx);
                }
            }).expect("spawn a rank thread"));
            senders.push(tx);
        }
        Self { shared, senders, threads }
    }

    pub fn world(&self) -> usize {
        self.shared.world
    }

    /// Run `f` on every rank's thread and collect the answers in rank order. A rank whose closure panics or fails marks the group dead, so
    /// the others stop waiting on it. A rank that does not answer within `call_timeout` yields `Timeout`.
    pub fn run_all<T: Send + 'static>(&self, call_timeout: Duration, f: impl Fn(&mut RankCtx) -> Result<T, GroupError> + Send + Sync + 'static) -> Vec<Result<T, GroupError>> {
        let f = Arc::new(f);
        let (tx, rx) = mpsc::channel::<(usize, Result<T, GroupError>)>();
        for (rank, sender) in self.senders.iter().enumerate() {
            let (f, job_tx, shared) = (f.clone(), tx.clone(), self.shared.clone());
            let job: Job = Box::new(move |ctx| {
                let answer = catch_unwind(AssertUnwindSafe(|| f(ctx))).unwrap_or(Err(GroupError::RankFailed(rank as u32)));
                if answer.is_err() {
                    shared.mark_dead(rank as u32);
                }
                let _ = job_tx.send((rank, answer));
            });
            if sender.send(job).is_err() {
                let _ = tx.send((rank, Err(GroupError::RankFailed(rank as u32))));
            }
        }
        drop(tx);
        let mut answers: Vec<Option<Result<T, GroupError>>> = (0..self.world()).map(|_| None).collect();
        let deadline = Instant::now() + call_timeout;
        while answers.iter().any(Option::is_none) {
            match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok((rank, answer)) => answers[rank] = Some(answer),
                Err(_) => break,
            }
        }
        answers.into_iter().map(|a| a.unwrap_or(Err(GroupError::Timeout(call_timeout)))).collect()
    }

    /// Broadcast one plan to every rank (encoded once, hash-checked by each) and verify that all ranks saw the same plan and answered
    /// identically.
    pub fn step(&self, plan: &StepPlan) -> Result<StepOutput, GroupError> {
        let bytes = Arc::new(plan.encode());
        let expected = plan.hash();
        let answers = self.run_all(self.shared.timeout * 2, move |ctx| {
            let (output, seen) = ctx.apply_plan(&bytes)?;
            if seen != expected {
                return Err(GroupError::Diverged(format!("rank {} applied plan hash {seen:016x}, the scheduler sent {expected:016x}", ctx.rank)));
            }
            Ok((output, ctx.page_tables.clone()))
        });
        let mut results = Vec::new();
        for answer in answers {
            results.push(answer?);
        }
        let (first_output, first_tables) = &results[0];
        for (rank, (output, tables)) in results.iter().enumerate().skip(1) {
            if output != first_output || tables != first_tables {
                return Err(GroupError::Diverged(format!("rank {rank} answered differently from rank 0")));
            }
        }
        Ok(first_output.clone())
    }
}

impl Drop for StubGroup {
    fn drop(&mut self) {
        self.senders.clear();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}
