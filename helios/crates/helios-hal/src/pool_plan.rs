//! How big the KV pool is and how its page tables are laid out (ADR 0002). Pure arithmetic, so the granule math and the refusals are tested on any
//! host. The pool is one virtual range per rank, reserved before the graphs are captured and mapped completely once, after the free memory is
//! measured: `pages = floor((free - reserve) / page_bytes)`, with the mapped bytes (a whole number of granules) inside that budget.

use crate::contract::HalError;
use crate::step::PageDelta;

/// What one page holds: every layer's K and V for `page_tokens` tokens (page-major, ADR 0002 decision 17), bf16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolGeometry {
    pub page_tokens: usize,
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl PoolGeometry {
    pub fn page_bytes(&self) -> u64 {
        (self.page_tokens * self.layers * 2 * self.kv_heads * self.head_dim * 2) as u64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolPlan {
    pub pages: usize,
    pub page_bytes: u64,
    /// The bytes to map: a whole number of granules, at least `pages * page_bytes`.
    pub mapped_bytes: u64,
}

fn align_up(bytes: u64, unit: u64) -> u64 {
    bytes.div_ceil(unit) * unit
}

/// The memory kept free for what the pool must not take: the growth of the driver, cuBLAS and NCCL measured while the graphs were captured,
/// plus a fixed minimum. Logged by the caller, never silently zero.
pub fn reserve_bytes(measured_growth: u64, fixed_minimum: u64) -> u64 {
    measured_growth.saturating_add(fixed_minimum)
}

/// Size the pool from the memory measured free after the graphs exist. `max_pages` caps it (the model maximum needs no more). A request that
/// cannot hold one page is refused with a typed error, never a pool of zero pages.
pub fn plan_pool(free: u64, reserve: u64, page_bytes: u64, granule: u64, max_pages: Option<usize>) -> Result<PoolPlan, HalError> {
    if page_bytes == 0 || granule == 0 {
        return Err(HalError::Invalid("page size and granule must be positive".into()));
    }
    let budget = free.saturating_sub(reserve);
    let mut pages = (budget / page_bytes) as usize;
    if let Some(cap) = max_pages {
        pages = pages.min(cap);
    }
    // Whole granules are mapped, so `pages * page_bytes` rounded up must still fit the budget.
    while pages > 0 && align_up(pages as u64 * page_bytes, granule) > budget {
        pages -= 1;
    }
    if pages == 0 {
        return Err(HalError::Invalid(format!("{free} bytes free with {reserve} kept in reserve leave no room for one page of {page_bytes} bytes")));
    }
    Ok(PoolPlan { pages, page_bytes, mapped_bytes: align_up(pages as u64 * page_bytes, granule) })
}

/// The device page tables: `[slot][max_pages_per_slot]` of `u32` page indices, one table per slot. Updated by the deltas of a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageTableLayout {
    pub slots: usize,
    pub max_pages_per_slot: usize,
}

/// Entry value of a table index that points at no page.
pub const NO_PAGE: u32 = u32::MAX;

impl PageTableLayout {
    pub fn for_context(slots: usize, max_seq_len: usize, page_tokens: usize) -> Self {
        Self { slots, max_pages_per_slot: max_seq_len.div_ceil(page_tokens) }
    }

    pub fn bytes(&self) -> usize {
        self.slots * self.max_pages_per_slot * 4
    }

    /// The word of `(slot, index)` in the table.
    pub fn word(&self, slot: u32, index: u32) -> Option<usize> {
        ((slot as usize) < self.slots && (index as usize) < self.max_pages_per_slot).then(|| slot as usize * self.max_pages_per_slot + index as usize)
    }

    /// The deltas as the `(word, page)` pairs a device kernel applies, flattened to `u32`s. A delta outside the table or naming a page the pool
    /// does not have (a `RELEASED` delta writes [`NO_PAGE`]) is refused whole: nothing is applied from a bad plan.
    pub fn encode(&self, deltas: &[PageDelta], pool_pages: usize) -> Result<Vec<u32>, HalError> {
        let mut words = Vec::with_capacity(deltas.len() * 2);
        for d in deltas {
            let word = self.word(d.slot, d.index).ok_or_else(|| HalError::Invalid(format!("page-table delta (slot {}, index {}) is outside the table of {} slots x {} pages", d.slot, d.index, self.slots, self.max_pages_per_slot)))?;
            let page = if d.page == crate::step::RELEASED {
                NO_PAGE
            } else if (d.page as usize) < pool_pages {
                d.page
            } else {
                return Err(HalError::Invalid(format!("page {} is outside the pool of {pool_pages} pages", d.page)));
            };
            words.extend([word as u32, page]);
        }
        Ok(words)
    }

    /// Apply encoded deltas to a host copy of the table (what the device kernel does to its own).
    pub fn apply(&self, table: &mut [u32], encoded: &[u32]) {
        for pair in encoded.chunks_exact(2) {
            table[pair[0] as usize] = pair[1];
        }
    }

    pub fn empty(&self) -> Vec<u32> {
        vec![NO_PAGE; self.slots * self.max_pages_per_slot]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::RELEASED;

    const GRANULE: u64 = 2 << 20;

    #[test]
    fn a_page_of_the_7b_is_3_5_mib_and_of_the_half_b_is_0_75_mib() {
        // 28 layers, 4 KV heads, head_dim 128; 24 layers, 2 KV heads, head_dim 64.
        assert_eq!(PoolGeometry { page_tokens: 64, layers: 28, kv_heads: 4, head_dim: 128 }.page_bytes(), 3_670_016);
        assert_eq!(PoolGeometry { page_tokens: 64, layers: 24, kv_heads: 2, head_dim: 64 }.page_bytes(), 786_432);
    }

    #[test]
    fn the_pool_takes_the_free_memory_minus_the_reserve_in_whole_granules() {
        let page = PoolGeometry { page_tokens: 64, layers: 28, kv_heads: 4, head_dim: 128 }.page_bytes();
        let free = 52 * (1u64 << 30);
        let reserve = reserve_bytes(300 << 20, 1 << 30);
        let plan = plan_pool(free, reserve, page, GRANULE, None).unwrap();
        assert_eq!(plan.pages as u64, (free - reserve) / page);
        assert_eq!(plan.mapped_bytes % GRANULE, 0);
        assert!(plan.mapped_bytes >= plan.pages as u64 * page && plan.mapped_bytes <= free - reserve, "the mapping stays inside the budget");
        // One page more would not have fitted.
        assert!(align_up((plan.pages as u64 + 1) * page, GRANULE) > free - reserve);
    }

    #[test]
    fn the_model_maximum_caps_the_pool_and_no_memory_for_one_page_is_a_typed_refusal() {
        let page = 3_670_016;
        assert_eq!(plan_pool(1 << 40, 0, page, GRANULE, Some(2048)).unwrap().pages, 2048);
        assert!(matches!(plan_pool(10 << 20, 8 << 20, page, GRANULE, None), Err(HalError::Invalid(_))));
        assert!(matches!(plan_pool(1 << 30, 2 << 30, page, GRANULE, None), Err(HalError::Invalid(_))), "reserve above free");
        assert!(plan_pool(1 << 30, 0, 0, GRANULE, None).is_err());
    }

    #[test]
    fn a_page_size_that_is_not_a_granule_multiple_still_maps_whole_granules() {
        // 3.5 MiB pages: 4 pages are 14 MiB = 7 granules exactly; 3 pages are 10.5 MiB, mapped as 6 granules.
        let plan = plan_pool(15 << 20, 0, 3_670_016, GRANULE, None).unwrap();
        assert_eq!((plan.pages, plan.mapped_bytes), (4, 14 << 20));
        let tight = plan_pool(13 << 20, 0, 3_670_016, GRANULE, None).unwrap();
        assert_eq!((tight.pages, tight.mapped_bytes), (3, 12 << 20), "4 pages need 14 MiB of granules, which do not fit 13");
    }

    #[test]
    fn page_table_deltas_encode_apply_and_refuse_bad_entries_whole() {
        let layout = PageTableLayout::for_context(4, 1000, 64);
        assert_eq!(layout.max_pages_per_slot, 16);
        assert_eq!(layout.bytes(), 4 * 16 * 4);
        let mut table = layout.empty();
        let deltas = [PageDelta { slot: 0, index: 0, page: 7 }, PageDelta { slot: 3, index: 15, page: 99 }, PageDelta { slot: 0, index: 1, page: 8 }];
        layout.apply(&mut table, &layout.encode(&deltas, 100).unwrap());
        assert_eq!((table[0], table[1], table[3 * 16 + 15], table[2]), (7, 8, 99, NO_PAGE));
        layout.apply(&mut table, &layout.encode(&[PageDelta { slot: 0, index: 1, page: RELEASED }], 100).unwrap());
        assert_eq!(table[1], NO_PAGE);
        // A bad delta anywhere in a plan refuses the whole batch.
        for bad in [PageDelta { slot: 4, index: 0, page: 1 }, PageDelta { slot: 0, index: 16, page: 1 }, PageDelta { slot: 0, index: 0, page: 100 }] {
            assert!(layout.encode(&[PageDelta { slot: 0, index: 2, page: 1 }, bad], 100).is_err(), "{bad:?}");
        }
    }
}
