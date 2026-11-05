//! Log-mel features in the Whisper/librosa convention (Slaney mel scale and norm), parameterised so the exact variant a model needs is a
//! config value pinned against its reference implementation (ADR 0013): n_fft, hop, n_mels, sample rate, centre padding, normalisation.
use crate::error::AudioError;
use crate::fft::power_spectrum;
use std::f64::consts::PI;

#[derive(Debug, Clone, PartialEq)]
pub struct MelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop: usize,
    pub n_mels: usize,
    pub f_min: f64,
    pub f_max: f64,
    /// Whisper: floor at (max - 8) in log10 units, then (x + 4) / 4.
    pub whisper_norm: bool,
}

impl MelConfig {
    /// Whisper-style front end (n_fft 400, hop 160 at 16 kHz); `n_mels` 80 or 128.
    pub fn whisper(n_mels: usize) -> Self {
        Self { sample_rate: 16000, n_fft: 400, hop: 160, n_mels, f_min: 0.0, f_max: 8000.0, whisper_norm: true }
    }
}

fn hz_to_mel(hz: f64) -> f64 {
    // Slaney: linear below 1 kHz (3 mels per 200 Hz), logarithmic above.
    if hz < 1000.0 { hz * 3.0 / 200.0 } else { 15.0 + (hz / 1000.0).ln() / (6.4f64.ln() / 27.0) }
}
fn mel_to_hz(mel: f64) -> f64 {
    if mel < 15.0 { mel * 200.0 / 3.0 } else { 1000.0 * ((mel - 15.0) * (6.4f64.ln() / 27.0)).exp() }
}

/// Triangular filters with Slaney area normalisation: `n_mels` rows of `n_fft / 2 + 1` weights.
pub fn mel_filterbank(c: &MelConfig) -> Vec<Vec<f64>> {
    let bins = c.n_fft / 2 + 1;
    let (lo, hi) = (hz_to_mel(c.f_min), hz_to_mel(c.f_max));
    let pts: Vec<f64> = (0..c.n_mels + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (c.n_mels + 1) as f64)).collect();
    let freq = |k: usize| k as f64 * c.sample_rate as f64 / c.n_fft as f64;
    (0..c.n_mels)
        .map(|m| {
            let (l, ctr, r) = (pts[m], pts[m + 1], pts[m + 2]);
            let norm = 2.0 / (r - l);
            (0..bins).map(|k| (((freq(k) - l) / (ctr - l)).min((r - freq(k)) / (r - ctr))).max(0.0) * norm).collect()
        })
        .collect()
}

fn hann_periodic(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()).collect()
}

/// Log-mel spectrogram of mono audio already at `c.sample_rate`: `[n_mels][frames]`.
/// Centre padding by reflection (`n_fft / 2` each side); the last STFT frame is dropped, as Whisper does, so `frames = len / hop`.
pub fn log_mel(x: &[f32], c: &MelConfig) -> Result<Vec<Vec<f32>>, AudioError> {
    if c.n_fft < 2 || c.hop == 0 || c.n_mels == 0 || x.len() <= c.n_fft / 2 {
        return Err(AudioError::Invalid("audio shorter than half an FFT window, or bad mel configuration".into()));
    }
    let pad = c.n_fft / 2;
    let mut p: Vec<f64> = Vec::with_capacity(x.len() + 2 * pad);
    p.extend((1..=pad).rev().map(|i| x[i] as f64));
    p.extend(x.iter().map(|&v| v as f64));
    p.extend((1..=pad).map(|i| x[x.len() - 1 - i] as f64));
    let (win, fb) = (hann_periodic(c.n_fft), mel_filterbank(c));
    let frames = (p.len() - c.n_fft) / c.hop + 1;
    let frames = if c.whisper_norm { frames - 1 } else { frames };
    let mut out = vec![vec![0f32; frames]; c.n_mels];
    let mut log10 = vec![vec![0f64; frames]; c.n_mels];
    let mut buf = vec![0f64; c.n_fft];
    for f in 0..frames {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = p[f * c.hop + i] * win[i];
        }
        let ps = power_spectrum(&buf);
        for m in 0..c.n_mels {
            let e: f64 = fb[m].iter().zip(&ps).map(|(w, s)| w * s).sum();
            log10[m][f] = e.max(1e-10).log10();
        }
    }
    let top = log10.iter().flatten().cloned().fold(f64::MIN, f64::max);
    for m in 0..c.n_mels {
        for f in 0..frames {
            let v = if c.whisper_norm { (log10[m][f].max(top - 8.0) + 4.0) / 4.0 } else { log10[m][f] };
            out[m][f] = v as f32;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slaney_scale_anchor_points() {
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-12 && (mel_to_hz(15.0) - 1000.0).abs() < 1e-9);
        assert!((mel_to_hz(hz_to_mel(5000.0)) - 5000.0).abs() < 1e-8);
    }
    #[test]
    fn filterbank_shape_and_area_norm() {
        let c = MelConfig::whisper(128);
        let fb = mel_filterbank(&c);
        assert_eq!((fb.len(), fb[0].len()), (128, 201));
        assert!(fb.iter().all(|r| r.iter().all(|&w| w >= 0.0) && r.iter().sum::<f64>() > 0.0));
    }
    #[test]
    fn tone_peaks_in_the_expected_mel_band_and_frame_count() {
        let c = MelConfig::whisper(80);
        let x: Vec<f32> = (0..16000).map(|t| (2.0 * PI * 1000.0 * t as f64 / 16000.0).sin() as f32).collect();
        let m = log_mel(&x, &c).unwrap();
        assert_eq!((m.len(), m[0].len()), (80, 100)); // 1 s at hop 160, last frame dropped
        let mid = 50;
        let peak = (0..80).max_by(|&a, &b| m[a][mid].total_cmp(&m[b][mid])).unwrap();
        let want = (hz_to_mel(1000.0) / hz_to_mel(8000.0) * 81.0 - 1.0).round() as i64; // filter whose centre is nearest 1 kHz
        assert!((peak as i64 - want).abs() <= 1, "peak {peak} want {want}");
        let top = m.iter().flatten().cloned().fold(f32::MIN, f32::max);
        assert!(m.iter().flatten().all(|&v| v >= top - 2.0 - 1e-6)); // Whisper floor: (max - 8) / 4 below the peak
    }
    #[test]
    fn rejects_too_short_audio() {
        assert!(log_mel(&[0.0; 100], &MelConfig::whisper(80)).is_err());
    }
}
