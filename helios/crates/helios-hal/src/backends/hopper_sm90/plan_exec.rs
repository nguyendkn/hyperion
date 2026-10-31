//! A `StepPlan` on the device (ADR 0006, 0007): apply the page-table deltas, then run the plan's rows: one prefill chunk of one slot, or one
//! decode step over up to a bucket of slots. The step completes before `submit_step` returns (the scheduler of this run takes its output at
//! once); the ticket and the ledger still follow the contract, so a pipelined caller keeps working. Sampling is greedy on the device.

use anyhow::{anyhow, Result};

use super::forward_device::DecodeMode;
use super::session::Session;
use crate::contract::HalError;
use crate::cuda::{Cublas, CudaDevice};
use crate::pool_plan::PageTableLayout;
use crate::step::{Capacity, Input, RowOutput, SamplingInit, StepOutput, StepPlan, Want};

impl Session {
    pub(super) fn run_plan(&self, dev: &CudaDevice, cublas: &Cublas, plan: &StepPlan, layout: &PageTableLayout, pool_pages: usize, mode: DecodeMode) -> Result<StepOutput> {
        for init in &plan.inits {
            if init.sampling != SamplingInit::GREEDY {
                return Err(HalError::Unsupported("sampling other than greedy on sm_90 (the device chooses the argmax)").into());
            }
        }
        for row in &plan.rows {
            if row.slot as usize >= self.max_batch {
                return Err(HalError::Invalid(format!("slot {} is outside the {} slots of this session", row.slot, self.max_batch)).into());
            }
            if row.position as usize + row.n_tokens as usize > self.capacity {
                return Err(HalError::Invalid(format!("a row reaches position {} but the model holds {} tokens", row.position as usize + row.n_tokens as usize, self.capacity)).into());
            }
        }
        // Nothing is applied from a plan with a bad delta: the encoding refuses it whole.
        let pairs = layout.encode(&plan.deltas, pool_pages)?;
        self.apply_deltas(dev, &pairs)?;
        let mut rows = Vec::with_capacity(plan.rows.len());
        match plan.capacity {
            Capacity::Decode { bucket } => {
                let bucket = bucket as usize;
                if !self.buckets().contains(&bucket) {
                    return Err(HalError::Invalid(format!("no decode bucket of {bucket} rows: the buckets are {:?}", self.buckets())).into());
                }
                let mut staged = Vec::with_capacity(plan.rows.len());
                for row in &plan.rows {
                    if staged.iter().any(|&(slot, _)| slot == row.slot as i32) {
                        return Err(HalError::Invalid(format!("slot {} appears twice in one decode step", row.slot)).into());
                    }
                    match &row.input {
                        Input::Tokens(ids) => self.set_slot_token(dev, row.slot as usize, ids[0])?,
                        Input::FromDevice => {}
                    }
                    staged.push((row.slot as i32, row.position));
                }
                self.stage_rows(dev, &staged, bucket)?;
                self.decode_step(dev, cublas, bucket, mode, None)?;
                dev.synchronize()?;
                for (r, row) in plan.rows.iter().enumerate() {
                    rows.push(RowOutput {
                        slot: row.slot,
                        tokens: if row.want == Want::Token { vec![self.read_token(dev, r, row.position as usize)?] } else { Vec::new() },
                        logits: if row.want == Want::Logits { Some(self.read_logits(dev, r)?) } else { None },
                    });
                }
            }
            Capacity::Prefill { rows: capacity } => {
                let [row] = plan.rows.as_slice() else {
                    return Err(HalError::Invalid(format!("a prefill step carries one row, this one has {}", plan.rows.len())).into());
                };
                let Input::Tokens(ids) = &row.input else {
                    return Err(HalError::Invalid("a prefill row takes its tokens from the host".into()).into());
                };
                if capacity as usize > self.chunk_rows || ids.len() > capacity as usize {
                    return Err(HalError::Invalid(format!("a chunk of {} tokens does not fit the {capacity}-row prefill graph (largest {})", ids.len(), self.chunk_rows)).into());
                }
                self.prefill(dev, cublas, row.slot as usize, ids, row.position as usize, mode, row.want == Want::Token)?;
                let last = row.position as usize + ids.len() - 1;
                rows.push(RowOutput {
                    slot: row.slot,
                    tokens: if row.want == Want::Token { vec![self.read_token(dev, 0, last)?] } else { Vec::new() },
                    logits: if row.want == Want::Logits { Some(self.read_logits(dev, 0)?) } else { None },
                });
            }
        }
        if rows.len() != plan.rows.len() {
            return Err(anyhow!("internal: {} outputs for {} rows", rows.len(), plan.rows.len()));
        }
        Ok(StepOutput { step: plan.step, rows })
    }
}
