//! The decode buckets of a loaded model (A6): one captured graph per bucket size, 1, 2, 4, ... up to `--max-batch`. Pure arithmetic shared by the
//! backends (which capture the graphs) and the scheduler (which picks the smallest bucket that holds the running rows). A bucket larger than the
//! active row count runs with its extra rows masked: they write no KV, produce no output and exit early in attention.

use crate::contract::HalError;

/// Largest bucket accepted: the batch the 7B decode step is tuned for (activation buffers grow with it).
pub const MAX_BUCKET: usize = 32;

/// The buckets 1, 2, 4, ... up to `largest` (a power of two from 1 to [`MAX_BUCKET`]); anything else is a typed refusal, never rounded.
pub fn buckets(largest: usize) -> Result<Vec<usize>, HalError> {
    if largest == 0 || largest > MAX_BUCKET || !largest.is_power_of_two() {
        return Err(HalError::Invalid(format!("the largest decode bucket must be a power of two from 1 to {MAX_BUCKET}, not {largest}")));
    }
    Ok(std::iter::successors(Some(1), |b| (*b < largest).then_some(b * 2)).collect())
}

/// The smallest bucket that holds `rows`, or `None` when no bucket does (or there are no rows).
pub fn bucket_for(buckets: &[usize], rows: usize) -> Option<usize> {
    if rows == 0 {
        return None;
    }
    buckets.iter().copied().find(|&b| b >= rows)
}

/// The row mask of a step of `bucket` rows with `active` rows in use: `mask[r]` is 1 for a live row and 0 for a masked one. The live rows are
/// always the first `active`, so a masked row can be recognised by its index alone.
pub fn row_mask(bucket: usize, active: usize) -> Vec<u8> {
    (0..bucket).map(|r| u8::from(r < active)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_double_from_one_up_to_the_largest() {
        assert_eq!(buckets(1).unwrap(), vec![1]);
        assert_eq!(buckets(32).unwrap(), vec![1, 2, 4, 8, 16, 32]);
        for bad in [0, 3, 6, 33, 64] {
            assert!(buckets(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_smallest_bucket_that_holds_the_rows_is_chosen() {
        let b = buckets(32).unwrap();
        let picks: Vec<_> = [0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33].iter().map(|&n| bucket_for(&b, n)).collect();
        assert_eq!(picks, vec![None, Some(1), Some(2), Some(4), Some(4), Some(8), Some(8), Some(16), Some(16), Some(32), Some(32), None]);
    }

    #[test]
    fn the_mask_marks_the_first_active_rows_and_nothing_else() {
        assert_eq!(row_mask(8, 3), vec![1, 1, 1, 0, 0, 0, 0, 0]);
        assert_eq!(row_mask(4, 4), vec![1; 4]);
        assert_eq!(row_mask(1, 0), vec![0]);
    }
}
