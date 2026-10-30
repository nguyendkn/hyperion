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

/// Round an f32 to bf16 and back: what torch does after every bf16 tensor operation.
fn round_bf16(v: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(v))
}

/// `y = a + b` elementwise, rounded to bf16.
pub fn add(a: &[u16], b: &[u16]) -> Vec<u16> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| f32_to_bf16(bf16_to_f32(x) + bf16_to_f32(y))).collect()
}

/// SwiGLU gate as in Qwen2's MLP: `bf16(silu(gate)) * up`, with silu computed in f32 and
/// rounded to bf16 before the product (torch rounds after each bf16 op).
pub fn swiglu(gate: &[u16], up: &[u16]) -> Vec<u16> {
    assert_eq!(gate.len(), up.len());
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| {
            let g = bf16_to_f32(g);
            let silu = round_bf16(g / (1.0 + (-g).exp()));
            f32_to_bf16(silu * bf16_to_f32(u))
        })
        .collect()
}

/// Rotary embedding tables `[max_pos, head_dim]` (bf16 bits) in the Hugging Face layout: the
/// `head_dim / 2` frequencies are repeated for both halves. Computed in f64 then rounded
/// to bf16, as HF computes in float then casts to the model dtype.
pub fn rope_tables(max_pos: usize, head_dim: usize, theta: f64) -> (Vec<u16>, Vec<u16>) {
    assert!(head_dim % 2 == 0, "head_dim must be even");
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(max_pos * head_dim);
    let mut sin = Vec::with_capacity(max_pos * head_dim);
    for pos in 0..max_pos {
        for half_pass in 0..2 {
            let _ = half_pass;
            for i in 0..half {
                let angle = pos as f64 * theta.powf(-2.0 * i as f64 / head_dim as f64);
                cos.push(f32_to_bf16(angle.cos() as f32));
                sin.push(f32_to_bf16(angle.sin() as f32));
            }
        }
    }
    (cos, sin)
}

/// Apply rotary embedding (HF `rotate_half` convention) to `x` laid out `[tokens, heads, head_dim]`
/// at the given positions. Every multiply and the add are rounded to bf16, like torch does.
pub fn rope(x: &[u16], positions: &[u32], heads: usize, head_dim: usize, cos: &[u16], sin: &[u16]) -> Vec<u16> {
    let half = head_dim / 2;
    assert_eq!(x.len(), positions.len() * heads * head_dim);
    let mut out = vec![0u16; x.len()];
    for (t, &pos) in positions.iter().enumerate() {
        let table = pos as usize * head_dim;
        for h in 0..heads {
            let base = (t * heads + h) * head_dim;
            for i in 0..half {
                let (x1, x2) = (bf16_to_f32(x[base + i]), bf16_to_f32(x[base + i + half]));
                let (c, s) = (bf16_to_f32(cos[table + i]), bf16_to_f32(sin[table + i]));
                // first half: x1*cos + (-x2)*sin ; second half: x2*cos + x1*sin
                out[base + i] = f32_to_bf16(round_bf16(x1 * c) + round_bf16(-x2 * s));
                out[base + i + half] = f32_to_bf16(round_bf16(x2 * c) + round_bf16(x1 * s));
            }
        }
    }
    out
}

/// Causal grouped-query attention, flash-style numerics: scores, softmax and the weighted sum
/// of values are accumulated in f64 and the output is rounded to bf16 once.
///
/// `q` is `[tokens, heads, head_dim]`; `k` and `v` are the cache `[seq, kv_heads, head_dim]`.
/// Query `t` sits at absolute position `q_start + t` and attends to keys `0..=q_start + t`.
/// `heads` must be a multiple of `kv_heads`; query head `h` uses KV head `h / (heads / kv_heads)`.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[u16],
    k: &[u16],
    v: &[u16],
    tokens: usize,
    seq: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    q_start: usize,
) -> Vec<u16> {
    assert_eq!(heads % kv_heads, 0, "heads must be a multiple of kv_heads");
    assert_eq!(q.len(), tokens * heads * head_dim);
    assert_eq!(k.len(), seq * kv_heads * head_dim);
    assert_eq!(v.len(), k.len());
    assert!(q_start + tokens <= seq, "queries must lie inside the cache");
    let group = heads / kv_heads;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let at = |x: &[u16], pos: usize, head: usize, d: usize| f64::from(bf16_to_f32(x[(pos * kv_heads + head) * head_dim + d]));
    let mut out = vec![0u16; q.len()];
    for t in 0..tokens {
        let last = q_start + t; // inclusive
        for h in 0..heads {
            let kv = h / group;
            let qrow = &q[(t * heads + h) * head_dim..(t * heads + h + 1) * head_dim];
            let scores: Vec<f64> = (0..=last)
                .map(|j| (0..head_dim).map(|d| f64::from(bf16_to_f32(qrow[d])) * at(k, j, kv, d)).sum::<f64>() * scale)
                .collect();
            let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = weights.iter().sum();
            for d in 0..head_dim {
                let acc: f64 = weights.iter().enumerate().map(|(j, w)| w * at(v, j, kv, d)).sum();
                out[(t * heads + h) * head_dim + d] = f32_to_bf16((acc / sum) as f32);
            }
        }
    }
    out
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

/// Qwen3's QK-norm: RMSNorm over each head of `x` (`[tokens, heads, head_dim]`) with one weight vector of `head_dim`
/// elements shared by every head and token; the same arithmetic as [`rmsnorm`] with one row per (token, head).
pub fn qk_norm(x: &[u16], weight: &[u16], tokens: usize, heads: usize, head_dim: usize, eps: f32) -> Vec<u16> {
    assert_eq!(x.len(), tokens * heads * head_dim, "input must be [tokens, heads, head_dim]");
    rmsnorm(x, weight, tokens * heads, head_dim, eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qk_norm_normalises_each_head_on_its_own() {
        // Two tokens, two heads of 4 dimensions: scaling one head must not change the others, and each head's
        // output has unit mean square (before the weight).
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0, 0.25, 0.5, 0.75, 1.0, 8.0, -8.0, 4.0, -4.0];
        let x: Vec<u16> = values.iter().map(|v| f32_to_bf16(*v)).collect();
        let ones = vec![f32_to_bf16(1.0); 4];
        let y = qk_norm(&x, &ones, 2, 2, 4, 1e-6);
        for head in 0..4 {
            let row: Vec<f32> = y[head * 4..head * 4 + 4].iter().map(|b| bf16_to_f32(*b)).collect();
            let mean_square = row.iter().map(|v| v * v).sum::<f32>() / 4.0;
            assert!((mean_square - 1.0).abs() < 0.02, "head {head}: {mean_square}");
        }
        let mut scaled = x.clone();
        for v in &mut scaled[..4] {
            *v = f32_to_bf16(bf16_to_f32(*v) * 4.0);
        }
        let y2 = qk_norm(&scaled, &ones, 2, 2, 4, 1e-6);
        assert_eq!(y[4..], y2[4..], "the other heads are untouched");
        assert_eq!(y[..4], y2[..4], "RMSNorm is invariant to the scale of its own head (a power of two here)");
        // The weight multiplies every head the same way.
        let twos = vec![f32_to_bf16(2.0); 4];
        let doubled = qk_norm(&x, &twos, 2, 2, 4, 1e-6);
        for (a, b) in y.iter().zip(&doubled) {
            assert_eq!(f32_to_bf16(bf16_to_f32(*a) * 2.0), *b);
        }
    }

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

    #[test]
    fn add_rounds_to_bf16() {
        assert_eq!(add(&bf(&[1.0, 2.0]), &bf(&[0.5, -2.0])), bf(&[1.5, 0.0]));
    }

    #[test]
    fn swiglu_known_values() {
        // silu(0) = 0; silu(large) ~ large; silu(-large) ~ 0.
        let y = swiglu(&bf(&[0.0, 8.0, -8.0]), &bf(&[3.0, 1.0, 1.0]));
        assert_eq!(bf16_to_f32(y[0]), 0.0);
        assert!((bf16_to_f32(y[1]) - 7.9973).abs() < 0.05, "{}", bf16_to_f32(y[1]));
        assert!(bf16_to_f32(y[2]).abs() < 0.01);
    }

    #[test]
    fn rope_tables_start_with_cos_one_sin_zero_at_position_zero() {
        let (cos, sin) = rope_tables(2, 4, 10000.0);
        assert!(cos[..4].iter().all(|&c| bf16_to_f32(c) == 1.0));
        assert!(sin[..4].iter().all(|&s| bf16_to_f32(s) == 0.0));
        // halves repeat: entries i and i + head_dim/2 are equal
        assert_eq!(cos[4], cos[6]);
        assert_eq!(sin[5], sin[7]);
    }

    #[test]
    fn rope_at_position_zero_is_the_identity() {
        let (cos, sin) = rope_tables(1, 4, 10000.0);
        let x = bf(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(rope(&x, &[0], 1, 4, &cos, &sin), x);
    }

    #[test]
    fn rope_rotates_by_the_expected_angle() {
        // head_dim 2: one frequency, angle = pos. Rotating (1, 0) by 1 rad gives (cos 1, sin 1).
        let (cos, sin) = rope_tables(2, 2, 10000.0);
        let y = rope(&bf(&[1.0, 0.0]), &[1], 1, 2, &cos, &sin);
        assert!((bf16_to_f32(y[0]) - 1f32.cos()).abs() < 0.01 && (bf16_to_f32(y[1]) - 1f32.sin()).abs() < 0.01);
    }

    fn attn(q: &[f32], k: &[f32], v: &[f32], tokens: usize, seq: usize, heads: usize, kv_heads: usize, d: usize, q_start: usize) -> Vec<f32> {
        attention(&bf(q), &bf(k), &bf(v), tokens, seq, heads, kv_heads, d, q_start)
            .iter()
            .map(|&b| bf16_to_f32(b))
            .collect()
    }

    #[test]
    fn attention_over_one_key_returns_that_value() {
        // A single key: softmax weight is 1, so the output is exactly v.
        let out = attn(&[1.0, 0.0], &[3.0, 4.0], &[5.0, 6.0], 1, 1, 1, 1, 2, 0);
        assert_eq!(out, vec![5.0, 6.0]);
    }

    #[test]
    fn attention_is_causal() {
        // Two keys with very different values. Query 0 (position 0) may only see key 0.
        let (q, k, v) = ([0.0f32, 0.0, 0.0, 0.0], [0.0f32, 0.0, 0.0, 0.0], [1.0f32, 1.0, 9.0, 9.0]);
        let out = attn(&q, &k, &v, 2, 2, 1, 1, 2, 0);
        assert_eq!(&out[..2], &[1.0, 1.0], "position 0 must ignore the later key");
        // Query 1 sees both keys with equal scores: the mean of the values.
        assert_eq!(&out[2..], &[5.0, 5.0]);
    }

    #[test]
    fn attention_decode_equals_the_matching_prefill_row() {
        let (heads, kv, d, seq) = (4, 2, 4, 5);
        let mk = |n: usize, m: u64| -> Vec<f32> { (0..n).map(|i| (((i as u64 * m + 7) % 13) as f32 - 6.0) / 4.0).collect() };
        let (q, k, v) = (mk(seq * heads * d, 5), mk(seq * kv * d, 3), mk(seq * kv * d, 11));
        let full = attn(&q, &k, &v, seq, seq, heads, kv, d, 0);
        // Decode of the last token: only its own query row, q_start = seq - 1.
        let last_q = &q[(seq - 1) * heads * d..];
        let one = attn(last_q, &k, &v, 1, seq, heads, kv, d, seq - 1);
        assert_eq!(&full[(seq - 1) * heads * d..], &one[..]);
    }

    #[test]
    fn attention_groups_query_heads_onto_kv_heads() {
        // 2 query heads share 1 kv head: identical queries give identical outputs.
        let q = [1.0f32, 2.0, 1.0, 2.0]; // 1 token, 2 heads, d = 2
        let out = attn(&q, &[0.5, 0.5, 1.0, -1.0], &[1.0, 2.0, 3.0, 4.0], 1, 2, 2, 1, 2, 1);
        assert_eq!(&out[..2], &out[2..]);
    }
}
