//! Deterministic band-limited resampler: Kaiser-windowed sinc, exact rational positions (integer arithmetic), f64 accumulation.
//! Not bit-identical to librosa/torchaudio (different window and edge rules); the reference tests compare features, not samples.
use crate::error::AudioError;
use std::f64::consts::PI;

/// Zero crossings of the sinc on each side at the lower of the two rates (quality knob; 32 gives about -90 dB stop band with beta 9).
const HALF_ZEROS: f64 = 32.0;
const KAISER_BETA: f64 = 9.0;

fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, q) = (1.0, 1.0, x * x / 4.0);
    for k in 1..60 {
        term *= q / (k * k) as f64;
        sum += term;
        if term < 1e-18 * sum {
            break;
        }
    }
    sum
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Resample mono `x` from `from` Hz to `to` Hz. Output length is `floor(len * to / from)`.
pub fn resample(x: &[f32], from: u32, to: u32) -> Result<Vec<f32>, AudioError> {
    if from == 0 || to == 0 {
        return Err(AudioError::Invalid("sample rate must be positive".into()));
    }
    if from == to {
        return Ok(x.to_vec());
    }
    let g = gcd(from as u64, to as u64);
    let (down, up) = ((from as u64 / g) as usize, (to as u64 / g) as usize); // output j reads input position j * down / up
    let cutoff = (to as f64 / from as f64).min(1.0);
    let half = HALF_ZEROS / cutoff;
    let norm = 1.0 / bessel_i0(KAISER_BETA);
    let out_len = (x.len() as u64 * up as u64 / down as u64) as usize;
    // The weights depend only on the fractional phase (j * down) % up: compute each phase once.
    let taps = half.ceil() as i64;
    let mut phases: Vec<Vec<f64>> = Vec::with_capacity(up);
    for ph in 0..up {
        let frac = (ph * down % up) as f64 / up as f64; // position = floor + frac
        let mut w: Vec<f64> = (-taps + 1..=taps)
            .map(|k| {
                let d = k as f64 - frac;
                let arg = d / half;
                if arg.abs() >= 1.0 { return 0.0; }
                let s = if d == 0.0 { 1.0 } else { (PI * cutoff * d).sin() / (PI * cutoff * d) };
                s * bessel_i0(KAISER_BETA * (1.0 - arg * arg).sqrt()) * norm
            })
            .collect();
        let sum: f64 = w.iter().sum();
        w.iter_mut().for_each(|v| *v /= sum);
        phases.push(w);
    }
    let mut out = Vec::with_capacity(out_len);
    for j in 0..out_len {
        let base = (j * down / up) as i64;
        let w = &phases[j * down % up % up];
        let mut acc = 0.0f64;
        for (i, &wk) in w.iter().enumerate() {
            let idx = base + (-taps + 1 + i as i64);
            if idx >= 0 && (idx as usize) < x.len() {
                acc += wk * x[idx as usize] as f64;
            }
        }
        out.push(acc as f32);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(f: f64, rate: u32, n: usize) -> Vec<f32> {
        (0..n).map(|t| (2.0 * PI * f * t as f64 / rate as f64).sin() as f32).collect()
    }
    fn rms(x: &[f32]) -> f64 {
        (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
    }
    #[test]
    fn downsample_keeps_inband_tone_and_length() {
        let y = resample(&tone(440.0, 48000, 48000), 48000, 16000).unwrap();
        assert_eq!(y.len(), 16000);
        let want = tone(440.0, 16000, 16000);
        let err: Vec<f32> = y.iter().zip(&want).map(|(a, b)| a - b).skip(1000).take(14000).collect();
        assert!(rms(&err) < 1e-3, "rms err {}", rms(&err));
    }
    #[test]
    fn aliasing_tone_is_removed() {
        // 12 kHz is above the new Nyquist (8 kHz): it must be rejected, not folded to 4 kHz.
        let y = resample(&tone(12000.0, 48000, 48000), 48000, 16000).unwrap();
        assert!(rms(&y[1000..15000]) < 1e-3, "{}", rms(&y[1000..15000]));
    }
    #[test]
    fn upsample_22050_to_24000_and_identity() {
        let y = resample(&tone(1000.0, 22050, 22050), 22050, 24000).unwrap();
        assert_eq!(y.len(), 24000);
        assert!(resample(&[0.5, 0.25], 16000, 16000).unwrap() == vec![0.5, 0.25]);
        assert!(resample(&[], 0, 16000).is_err());
    }
    #[test]
    fn deterministic() {
        let x = tone(300.0, 44100, 5000);
        assert_eq!(resample(&x, 44100, 16000).unwrap(), resample(&x, 44100, 16000).unwrap());
    }
}
