//! Stub backend — implements [`ComputeBackend`] without touching a GPU.
//! Lets the workspace typecheck and run core unit tests on hosts without CUDA.
//! Reports `sm_0` so triage against a real package's `target_arch` fails
//! loudly (`ArchMismatch`) rather than silently pretending to work.
//!
//! The stub is a named test double (results tagged `[RUN on stub]`): it follows the load order and the step
//! contract (it refuses what a real backend refuses: a step with a plan that does not validate, a page the pool
//! does not have) but computes nothing: every sampled token is id 0, there is no device memory, and a pool is
//! as many pages as the memory policy asks for.

use anyhow::Result;

use crate::contract::{
    BackendConfig, Capabilities, GraphSet, HalError, KvPool, KvPoolInfo, KvReservation, KvSpec, LoadedModel, MemoryPolicy, RankInfo, StepTicket,
    WeightSource,
};
use crate::step::{RowOutput, StepOutput, StepPlan, Want, RELEASED};
use crate::traits::{ComputeBackend, HardwareInfo};

const NAME: &str = "stub";
/// Pages of a pool whose policy names no maximum: enough for a few thousand tokens.
const DEFAULT_PAGES: usize = 64;
/// Vocabulary size the stub pretends to have when a `Logits` row is asked for.
const STUB_LOGITS: usize = 8;

struct Loaded {
    layers: usize,
}

struct Reservation {
    page_tokens: usize,
    min_tokens: usize,
}

struct Pool {
    pages: usize,
}

pub struct StubBackend {
    capabilities: Capabilities,
}

impl Default for StubBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl StubBackend {
    pub fn new() -> Self {
        Self { capabilities: Capabilities::known("sm_0").expect("the stub arch is known") }
    }
}

impl ComputeBackend for StubBackend {
    fn capabilities(&self) -> Capabilities {
        self.capabilities.clone()
    }

    fn initialize_device(&self, _rank: &RankInfo) -> Result<HardwareInfo> {
        Ok(HardwareInfo { compute_capability: self.capabilities.arch.into(), total_vram_bytes: 0, free_vram_bytes: 0, max_cuda_streams: 0 })
    }

    fn load_model(&self, config: &BackendConfig, _weights: &WeightSource) -> Result<LoadedModel> {
        Ok(LoadedModel::new(NAME, Loaded { layers: config.model.num_layers as usize }))
    }

    fn reserve_kv(&self, model: &LoadedModel, spec: &KvSpec) -> Result<KvReservation> {
        model.payload::<Loaded>(NAME)?;
        if spec.page_tokens == 0 || !spec.page_tokens.is_power_of_two() {
            return Err(HalError::Invalid(format!("page_tokens {} is not a power of two", spec.page_tokens)).into());
        }
        Ok(KvReservation::new(NAME, Reservation { page_tokens: spec.page_tokens, min_tokens: spec.min_tokens }))
    }

    fn capture_graphs(&self, model: &LoadedModel, kv: &KvReservation) -> Result<GraphSet> {
        let loaded = model.payload::<Loaded>(NAME)?;
        kv.payload::<Reservation>(NAME)?;
        Ok(GraphSet::new(NAME, loaded.layers))
    }

    fn commit_kv(&self, kv: KvReservation, graphs: &GraphSet, policy: &MemoryPolicy) -> Result<KvPool> {
        let reservation = kv.payload::<Reservation>(NAME)?;
        graphs.payload::<usize>(NAME)?;
        let wanted = if policy.max_tokens > 0 { policy.max_tokens } else { reservation.page_tokens * DEFAULT_PAGES };
        let pages = wanted.max(reservation.min_tokens).div_ceil(reservation.page_tokens).max(1);
        let info = KvPoolInfo { page_tokens: reservation.page_tokens, pages, bytes_per_page: 0, free_before: 0, free_after: 0 };
        Ok(KvPool::new(NAME, info, Pool { pages }))
    }

    fn submit_step(&self, pool: &KvPool, plan: &StepPlan) -> Result<StepTicket> {
        let pool = pool.payload::<Pool>(NAME)?;
        plan.validate()?;
        for delta in plan.deltas.iter().filter(|d| d.page != RELEASED) {
            if delta.page as usize >= pool.pages {
                return Err(HalError::Invalid(format!("page {} is outside the pool of {} pages", delta.page, pool.pages)).into());
            }
        }
        let rows = plan
            .rows
            .iter()
            .map(|row| RowOutput {
                slot: row.slot,
                tokens: if row.want == Want::Token { vec![0] } else { Vec::new() },
                logits: (row.want == Want::Logits).then(|| vec![0.0; STUB_LOGITS]),
            })
            .collect();
        // Nothing to wait for: the step completes here.
        Ok(StepTicket::new(NAME, StepOutput { step: plan.step, rows }))
    }

    fn take_output(&self, ticket: StepTicket) -> Result<StepOutput> {
        Ok(ticket.payload::<StepOutput>(NAME)?.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::{Capacity, Input, PageDelta, Row};

    fn config() -> crate::contract::BackendConfig {
        crate::contract::BackendConfig {
            model_id: "m".into(),
            model: hyperion_specification::v1::ModelConfig {
                num_layers: 2,
                attention: hyperion_specification::v1::AttentionKind::Mha,
                num_attention_heads: 1,
                num_kv_heads: 1,
                head_dim: 64,
                vocab_size: 8,
                max_seq_len: 1024,
                hidden_size: 64,
                intermediate_size: 128,
                rope_theta: 1e4,
                rope_style: hyperion_specification::v1::RopeStyle::RotateHalf,
                rms_norm_eps: 1e-6,
                tie_word_embeddings: false,
                qkv_bias: false,
                qk_norm: false,
                activation: hyperion_specification::v1::Activation::Silu,
                attention_scale: None,
            },
            tensor_format: hyperion_specification::v1::TensorFormat::Bf16,
            target_arch: "sm_0".into(),
            eos_token_ids: vec![0],
            payload_files: vec!["model.safetensors".into()],
            payload_lengths: vec![None],
            tensors: Vec::new(),
            deployment: crate::contract::Deployment::SINGLE,
        }
    }

    fn pool(backend: &StubBackend) -> KvPool {
        backend.initialize_device(&RankInfo::single()).unwrap();
        let model = backend.load_model(&config(), &WeightSource { files: Vec::new(), slices: Vec::new() }).unwrap();
        let kv = backend.reserve_kv(&model, &KvSpec::default()).unwrap();
        let graphs = backend.capture_graphs(&model, &kv).unwrap();
        backend.commit_kv(kv, &graphs, &MemoryPolicy { reserve_bytes: 0, max_tokens: 640 }).unwrap()
    }

    fn decode_plan(page: u32) -> StepPlan {
        StepPlan {
            step: 1,
            capacity: Capacity::Decode { bucket: 1 },
            rows: vec![Row { slot: 0, input: Input::Tokens(vec![5]), n_tokens: 1, position: 0, table_len: 1, want: Want::Token }],
            deltas: vec![PageDelta { slot: 0, index: 0, page }],
            inits: Vec::new(),
        }
    }

    #[test]
    fn the_load_order_yields_a_pool_the_policy_sized_and_a_step_returns_a_token_per_wanted_row() {
        let backend = StubBackend::new();
        let pool = pool(&backend);
        assert_eq!(pool.info().pages, 10, "640 tokens in pages of 64");
        let output = backend.take_output(backend.submit_step(&pool, &decode_plan(3)).unwrap()).unwrap();
        assert_eq!(output.step, 1);
        assert_eq!(output.rows[0].tokens, vec![0]);
    }

    #[test]
    fn a_page_outside_the_pool_or_an_invalid_plan_is_refused() {
        let backend = StubBackend::new();
        let pool = pool(&backend);
        assert!(backend.submit_step(&pool, &decode_plan(10)).is_err());
        let mut bad = decode_plan(0);
        bad.rows[0].n_tokens = 3;
        assert!(backend.submit_step(&pool, &bad).is_err());
    }

    #[test]
    fn a_token_from_another_backend_is_refused() {
        let backend = StubBackend::new();
        let foreign = KvPool::new("other", KvPoolInfo { page_tokens: 64, pages: 1, bytes_per_page: 0, free_before: 0, free_after: 0 }, ());
        assert!(backend.submit_step(&foreign, &decode_plan(0)).is_err());
    }

    #[test]
    fn a_page_size_that_is_not_a_power_of_two_is_refused() {
        let backend = StubBackend::new();
        let model = backend.load_model(&config(), &WeightSource { files: Vec::new(), slices: Vec::new() }).unwrap();
        assert!(backend.reserve_kv(&model, &KvSpec { page_tokens: 48, min_tokens: 0 }).is_err());
    }
}
