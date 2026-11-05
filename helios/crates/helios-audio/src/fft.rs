//! Complex FFT in f64: radix-2 for powers of two, Bluestein (chirp-z) for any other length (Whisper-style front ends use n_fft = 400).
//! Deterministic: fixed operation order, twiddles from `sin_cos` of exact fractions.
use std::f64::consts::PI;

type C = (f64, f64);

fn mul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn fft_pow2(x: &mut [C], inverse: bool) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            x.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let step = sign * 2.0 * PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = (step * k as f64).sin_cos();
                let (u, v) = (x[start + k], mul(x[start + k + len / 2], (c, s)));
                x[start + k] = (u.0 + v.0, u.1 + v.1);
                x[start + k + len / 2] = (u.0 - v.0, u.1 - v.1);
            }
        }
        len <<= 1;
    }
}

/// Forward DFT of any length `n >= 1` (unnormalised: `X[k] = sum x[t] e^{-2 pi i k t / n}`).
pub fn fft(input: &[C]) -> Vec<C> {
    let n = input.len();
    if n <= 1 {
        return input.to_vec();
    }
    if n.is_power_of_two() {
        let mut v = input.to_vec();
        fft_pow2(&mut v, false);
        return v;
    }
    // Bluestein: X[k] = w[k]^* sum (x[t] w[t]^*)... with w[t] = e^{i pi t^2 / n}, as a convolution of length m >= 2n-1.
    let m = (2 * n - 1).next_power_of_two();
    let chirp: Vec<C> = (0..n).map(|t| { let ang = PI * ((t * t) % (2 * n)) as f64 / n as f64; (ang.cos(), -ang.sin()) }).collect();
    let mut a = vec![(0.0, 0.0); m];
    let mut b = vec![(0.0, 0.0); m];
    for t in 0..n {
        a[t] = mul(input[t], chirp[t]);
    }
    b[0] = (chirp[0].0, -chirp[0].1);
    for t in 1..n {
        let c = (chirp[t].0, -chirp[t].1);
        b[t] = c;
        b[m - t] = c;
    }
    fft_pow2(&mut a, false);
    fft_pow2(&mut b, false);
    for i in 0..m {
        a[i] = mul(a[i], b[i]);
    }
    fft_pow2(&mut a, true);
    (0..n).map(|k| { let v = mul((a[k].0 / m as f64, a[k].1 / m as f64), chirp[k]); v }).collect()
}

/// Power spectrum `|X[k]|^2` for `k in 0..=n/2` of a real frame.
pub fn power_spectrum(frame: &[f64]) -> Vec<f64> {
    let c: Vec<C> = frame.iter().map(|&v| (v, 0.0)).collect();
    let x = fft(&c);
    (0..=frame.len() / 2).map(|k| x[k].0 * x[k].0 + x[k].1 * x[k].1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(x: &[C]) -> Vec<C> {
        let n = x.len();
        (0..n).map(|k| (0..n).fold((0.0, 0.0), |acc, t| { let (s, c) = (-2.0 * PI * ((k * t) % n) as f64 / n as f64).sin_cos(); let p = mul(x[t], (c, s)); (acc.0 + p.0, acc.1 + p.1) })).collect()
    }
    #[test]
    fn matches_naive_dft_for_pow2_and_other_lengths() {
        for n in [1usize, 2, 8, 64, 3, 5, 12, 50, 400] {
            let x: Vec<C> = (0..n).map(|i| (((i * 7 + 3) % 11) as f64 - 5.0, ((i * 5) % 13) as f64 * 0.1)).collect();
            let (a, b) = (fft(&x), naive(&x));
            let worst = a.iter().zip(&b).map(|(p, q)| (p.0 - q.0).abs().max((p.1 - q.1).abs())).fold(0.0, f64::max);
            assert!(worst < 1e-9, "n={n} worst={worst}");
        }
    }
    #[test]
    fn pure_tone_lands_in_one_bin() {
        let n = 400;
        let f: Vec<f64> = (0..n).map(|t| (2.0 * PI * 25.0 * t as f64 / n as f64).cos()).collect();
        let p = power_spectrum(&f);
        let peak = p.iter().cloned().enumerate().fold((0, 0.0), |a, (i, v)| if v > a.1 { (i, v) } else { a });
        assert_eq!(peak.0, 25);
        assert!(p.iter().enumerate().all(|(i, &v)| i == 25 || v < 1e-12 * peak.1));
    }
}
