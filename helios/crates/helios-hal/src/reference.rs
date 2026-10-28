//! CPU reference math for checking GPU results: bf16 conversion and a linear layer
//! computed in f64. Pure host code, no CUDA, unit-tested everywhere. Hardware tests
//! compare device output against these functions, so each GPU op has an independent
//! implementation to be checked against.

/// bf16 is the top 16 bits of an f32.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// f32 -> bf16 with round-to-nearest-even; NaN stays a NaN.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounding = 0x7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding)) >> 16) as u16
}

/// `y[t][o] = sum_i x[t][i] * w[o][i]`: a linear layer without bias, with `w` stored
/// row-major `[out, in]` and `x` row-major `[tokens, in]` (both bf16 bits), summed in
/// f64 so the reference is more precise than any device accumulation order.
pub fn linear(w: &[u16], x: &[u16], tokens: usize, out_features: usize, in_features: usize) -> Vec<f32> {
    assert_eq!(w.len(), out_features * in_features, "weight must be [out, in]");
    assert_eq!(x.len(), tokens * in_features, "input must be [tokens, in]");
    let mut y = Vec::with_capacity(tokens * out_features);
    for t in 0..tokens {
        let row = &x[t * in_features..(t + 1) * in_features];
        for o in 0..out_features {
            let weights = &w[o * in_features..(o + 1) * in_features];
            let sum: f64 = row
                .iter()
                .zip(weights)
                .map(|(&a, &b)| f64::from(bf16_to_f32(a)) * f64::from(bf16_to_f32(b)))
                .sum();
            y.push(sum as f32);
        }
    }
    y
}

/// RMSNorm as Hugging Face's Qwen2 does it: normalise in f32, round to bf16, then
/// multiply by the bf16 weight and round to bf16 again. `x` is `[tokens, hidden]`.
pub fn rmsnorm(x: &[u16], weight: &[u16], tokens: usize, hidden: usize, eps: f32) -> Vec<u16> {
    assert_eq!(weight.len(), hidden, "weight must have `hidden` elements");
    assert_eq!(x.len(), tokens * hidden, "input must be [tokens, hidden]");
    let mut y = Vec::with_capacity(x.len());
    for t in 0..tokens {
        let row = &x[t * hidden..(t + 1) * hidden];
        let mean: f64 = row.iter().map(|&b| f64::from(bf16_to_f32(b)).powi(2)).sum::<f64>() / hidden as f64;
        let inv = (1.0 / (mean + f64::from(eps)).sqrt()) as f32;
        for (i, &b) in row.iter().enumerate() {
            let normalised = bf16_to_f32(f32_to_bf16(bf16_to_f32(b) * inv));
            y.push(f32_to_bf16(bf16_to_f32(weight[i]) * normalised));
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_trips_values_that_fit_in_eight_bits_of_mantissa() {
        for v in [0.0f32, 1.0, -1.0, 0.5, -0.75, 3.0, 255.0, -128.0, 1.0 / 64.0] {
            assert_eq!(bf16_to_f32(f32_to_bf16(v)), v, "{v}");
        }
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        // 1 + 2^-8 sits exactly between two bf16 values; ties go to the even one (1.0).
        assert_eq!(f32_to_bf16(1.0 + f32::powi(2.0, -8)), f32_to_bf16(1.0));
        // Just above the tie rounds up to 1 + 2^-7.
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0 + f32::powi(2.0, -8) + f32::powi(2.0, -12))), 1.0 + f32::powi(2.0, -7));
    }

    #[test]
    fn bf16_keeps_nan_and_infinity() {
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
        assert_eq!(bf16_to_f32(f32_to_bf16(f32::INFINITY)), f32::INFINITY);
    }

    fn bf(values: &[f32]) -> Vec<u16> {
        values.iter().map(|&v| f32_to_bf16(v)).collect()
    }

    #[test]
    fn linear_matches_a_hand_computed_result() {
        // w = [[1, 2], [3, 4], [0, -1]] (out 3, in 2); x = [[1, 1], [2, 0]] (tokens 2)
        let y = linear(&bf(&[1.0, 2.0, 3.0, 4.0, 0.0, -1.0]), &bf(&[1.0, 1.0, 2.0, 0.0]), 2, 3, 2);
        assert_eq!(y, vec![3.0, 7.0, -1.0, 2.0, 6.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "weight must be")]
    fn linear_rejects_a_mis_shaped_weight() {
        linear(&bf(&[1.0, 2.0, 3.0]), &bf(&[1.0, 1.0]), 1, 2, 2);
    }

    #[test]
    fn rmsnorm_of_constant_rows_scales_by_the_weight() {
        // x = 2 everywhere: mean(x^2) = 4, rms = 2, normalised = 1, so y == weight.
        let y = rmsnorm(&bf(&[2.0; 4]), &bf(&[0.5, 1.0, 1.5, 2.0]), 1, 4, 0.0);
        assert_eq!(y, bf(&[0.5, 1.0, 1.5, 2.0]));
    }

    #[test]
    fn rmsnorm_normalises_each_row_independently() {
        // Row 0 is scaled by 1/rms = 1/5 * sqrt(2); row 1 is 100x bigger and must give the same result.
        let a = rmsnorm(&bf(&[3.0, 4.0]), &bf(&[1.0, 1.0]), 1, 2, 0.0);
        let both = rmsnorm(&bf(&[3.0, 4.0, 300.0, 400.0]), &bf(&[1.0, 1.0]), 2, 2, 0.0);
        assert_eq!(&both[..2], &a[..]);
        assert_eq!(&both[2..], &a[..]);
        let v = bf16_to_f32(a[0]);
        assert!((v - 3.0 / (12.5f32).sqrt()).abs() < 0.01, "{v}");
    }

    #[test]
    fn rmsnorm_eps_keeps_an_all_zero_row_finite() {
        let y = rmsnorm(&bf(&[0.0; 4]), &bf(&[1.0; 4]), 1, 4, 1e-6);
        assert!(y.iter().all(|&b| bf16_to_f32(b) == 0.0));
    }
}
