//! The tensor-parallel group as one backend of the `ComputeBackend` contract (ADR 0004 asks for one instance per rank; a rank here is a thread of the
//! group, and the scheduler talks to the group through the same trait it uses for one GPU): the load order runs on every rank at once, a `StepPlan` is
//! broadcast, and the pool the scheduler sees holds the pages every rank holds.

use std::sync::Arc;

use anyhow::Result;

use super::forward_device::DecodeMode;
use super::tp_group::{TpBackend, TpError};
use crate::contract::{BackendConfig, Capabilities, Deployment, GraphSet, HalError, KvPool, KvPoolInfo, KvReservation, KvSpec, LoadedModel, MemoryPolicy, RankInfo, StepTicket, WeightSource};
use crate::sharding::rank_slices;
use crate::step::{StepOutput, StepPlan};
use crate::traits::{ComputeBackend, HardwareInfo};

const TP_NAME: &str = "sm_90_tp";

/// The pool token of a tensor-parallel group: every rank holds the same number of pages.
struct TpPool {
    pages: usize,
}

struct TpCompleted {
    ticket: u64,
    output: StepOutput,
}

impl ComputeBackend for TpBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities::known("sm_90").expect("the sm_90 arch is known")
    }

    fn initialize_device(&self, _rank: &RankInfo) -> Result<HardwareInfo> {
        let answers = self.run_all(|rank| {
            let dev = rank.backend.device.lock().expect("device mutex poisoned");
            Ok(HardwareInfo { compute_capability: dev.compute_capability_string(), total_vram_bytes: dev.total_vram_bytes(), free_vram_bytes: dev.memory_info()?.0, max_cuda_streams: dev.max_streams() })
        });
        Ok(TpBackend::all_ok(answers)?.remove(0))
    }

    fn load_model(&self, config: &BackendConfig, weights: &WeightSource) -> Result<LoadedModel> {
        let size = self.size() as u32;
        if !config.supported_tp_sizes.contains(&size) {
            return Err(HalError::UnsupportedTp { tp: size, supported: config.supported_tp_sizes.clone() }.into());
        }
        let (config, files) = (Arc::new(config.clone()), Arc::new(weights.files.clone()));
        let answers = self.run_all(move |rank| {
            let mut config = (*config).clone();
            config.deployment = Deployment { tp_size: size, pp_size: 1, tp_rank: rank.rank as u32, pp_stage: 0 };
            let weights = WeightSource { files: (*files).clone(), slices: rank_slices(&config)? };
            let model = rank.backend.load_model(&config, &weights)?;
            rank.set_model(model);
            Ok(())
        });
        TpBackend::all_ok(answers)?;
        Ok(LoadedModel::new(TP_NAME, ()))
    }

    fn reserve_kv(&self, model: &LoadedModel, spec: &KvSpec) -> Result<KvReservation> {
        model.payload::<()>(TP_NAME)?;
        let spec = spec.clone();
        let answers = self.run_all(move |rank| {
            let reservation = rank.backend.reserve_kv(rank.model_ref()?, &spec)?;
            rank.set_reservation(reservation);
            Ok(())
        });
        TpBackend::all_ok(answers)?;
        Ok(KvReservation::new(TP_NAME, ()))
    }

    fn capture_graphs(&self, model: &LoadedModel, kv: &KvReservation) -> Result<GraphSet> {
        model.payload::<()>(TP_NAME)?;
        kv.payload::<()>(TP_NAME)?;
        // Every rank captures at once (the collectives are recorded, not run), then every rank runs the collectives once so the library has made its buffers.
        let answers = self.run_all(|rank| {
            let graphs = rank.backend.capture_graphs(rank.model_ref()?, rank.reservation_ref()?)?;
            rank.set_graphs(graphs);
            rank.backend.warm_collectives()
        });
        TpBackend::all_ok(answers)?;
        Ok(GraphSet::new(TP_NAME, ()))
    }

    fn commit_kv(&self, kv: KvReservation, graphs: &GraphSet, policy: &MemoryPolicy) -> Result<KvPool> {
        kv.payload::<()>(TP_NAME)?;
        graphs.payload::<()>(TP_NAME)?;
        let policy = policy.clone();
        let answers = self.run_all(move |rank| {
            let reservation = rank.take_reservation()?;
            let pool = rank.backend.commit_kv(reservation, rank.graphs_ref()?, &policy)?;
            let info = pool.info().clone();
            rank.set_pool(pool);
            Ok(info)
        });
        let infos = TpBackend::all_ok(answers)?;
        let pages: Vec<usize> = infos.iter().map(|i| i.pages).collect();
        if pages.iter().any(|p| *p != pages[0]) {
            return Err(TpError::UnequalPools { pages }.into());
        }
        let info = KvPoolInfo { page_tokens: infos[0].page_tokens, pages: pages[0], bytes_per_page: infos.iter().map(|i| i.bytes_per_page).sum(), free_before: infos[0].free_before, free_after: infos[0].free_after };
        Ok(KvPool::new(TP_NAME, info, TpPool { pages: pages[0] }))
    }

    fn submit_step(&self, pool: &KvPool, plan: &StepPlan) -> Result<StepTicket> {
        let handle = pool.payload::<TpPool>(TP_NAME)?;
        plan.validate()?;
        // Page indices are the same on every rank: a delta beyond the pool is refused here, before any rank sees it.
        if let Some(bad) = plan.deltas.iter().find(|d| d.page != crate::step::RELEASED && d.page as usize >= handle.pages) {
            return Err(HalError::Invalid(format!("page {} is outside the pool of {} pages", bad.page, handle.pages)).into());
        }
        let ticket = self.ledger.lock().expect("ledger mutex poisoned").submit()?;
        match self.submit_step_on_ranks(plan, DecodeMode::Graph) {
            Ok(output) => Ok(StepTicket::new(TP_NAME, TpCompleted { ticket, output })),
            Err(error) => {
                let _ = self.ledger.lock().expect("ledger mutex poisoned").take(ticket);
                Err(error)
            }
        }
    }

    fn take_output(&self, ticket: StepTicket) -> Result<StepOutput> {
        let done = ticket.payload::<TpCompleted>(TP_NAME)?;
        self.ledger.lock().expect("ledger mutex poisoned").take(done.ticket)?;
        Ok(done.output.clone())
    }

    fn shutdown(&self) -> Result<()> {
        let answers = self.run_all(|rank| {
            *rank.pool = None;
            rank.backend.shutdown()
        });
        TpBackend::all_ok(answers).map(|_| ())
    }
}
