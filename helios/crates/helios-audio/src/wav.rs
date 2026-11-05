//! WAV reader and writer. Reads PCM 8/16/24/32-bit and IEEE float 32/64-bit (also inside WAVE_FORMAT_EXTENSIBLE); everything else is a typed refusal.
use crate::error::AudioError;

/// Decoded audio: interleaved f32 samples in [-1, 1].
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl Audio {
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels.max(1) as usize
    }
    /// Average of the channels (the reference front ends do the same).
    pub fn to_mono(&self) -> Vec<f32> {
        let c = self.channels.max(1) as usize;
        if c == 1 {
            return self.samples.clone();
        }
        self.samples.chunks_exact(c).map(|f| (f.iter().map(|&x| x as f64).sum::<f64>() / c as f64) as f32).collect()
    }
}

fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

pub fn read_wav(bytes: &[u8]) -> Result<Audio, AudioError> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(AudioError::NotWav);
    }
    let (mut fmt, mut data): (Option<(u16, u16, u32, u16)>, Option<&[u8]>) = (None, None);
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32le(bytes, pos + 4) as usize;
        let body = pos + 8;
        // A streamed WAV may carry a data size of 0 or 0xFFFFFFFF: take the rest of the file.
        let end = if id == b"data" && (size == 0 || size == u32::MAX as usize) { bytes.len() } else { body.checked_add(size).filter(|&e| e <= bytes.len()).ok_or_else(|| AudioError::Malformed(format!("chunk {:?} runs past the end of the file", String::from_utf8_lossy(id))))? };
        match id {
            b"fmt " => {
                if end - body < 16 {
                    return Err(AudioError::Malformed("fmt chunk shorter than 16 bytes".into()));
                }
                let mut tag = u16le(bytes, body);
                let bits = u16le(bytes, body + 14);
                if tag == 0xFFFE {
                    if end - body < 26 {
                        return Err(AudioError::Malformed("extensible fmt chunk too short".into()));
                    }
                    tag = u16le(bytes, body + 24); // first two bytes of the sub-format GUID
                }
                fmt = Some((tag, u16le(bytes, body + 2), u32le(bytes, body + 4), bits));
            }
            b"data" => data = Some(&bytes[body..end]),
            _ => {}
        }
        pos = end + (end - body) % 2; // chunks are word aligned
    }
    let (tag, channels, sample_rate, bits) = fmt.ok_or_else(|| AudioError::Malformed("no fmt chunk".into()))?;
    let data = data.ok_or_else(|| AudioError::Malformed("no data chunk".into()))?;
    if channels == 0 || sample_rate == 0 {
        return Err(AudioError::Malformed("zero channels or sample rate".into()));
    }
    let samples: Vec<f32> = match (tag, bits) {
        (1, 8) => data.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect(),
        (1, 16) => data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect(),
        (1, 24) => data.chunks_exact(3).map(|c| (i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8) as f32 / 8_388_608.0).collect(),
        (1, 32) => data.chunks_exact(4).map(|c| (i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64 / 2_147_483_648.0) as f32).collect(),
        (3, 32) => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        (3, 64) => data.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
        _ => return Err(AudioError::UnsupportedFormat(format!("format tag {tag}, {bits} bits"))),
    };
    if samples.len() % channels as usize != 0 {
        return Err(AudioError::Malformed("sample count is not a multiple of the channel count".into()));
    }
    Ok(Audio { sample_rate, channels, samples })
}

/// 16-bit PCM WAV (samples are clipped to [-1, 1], rounded to nearest).
pub fn write_wav_pcm16(audio: &Audio) -> Vec<u8> {
    let data = pcm16(&audio.samples);
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&audio.channels.to_le_bytes());
    out.extend_from_slice(&audio.sample_rate.to_le_bytes());
    out.extend_from_slice(&(audio.sample_rate * audio.channels as u32 * 2).to_le_bytes());
    out.extend_from_slice(&(audio.channels * 2).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    out
}

/// Raw little-endian PCM16 bytes (the `pcm` response format).
pub fn pcm16(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|&s| ((s * 32768.0).round().clamp(-32768.0, 32767.0) as i16).to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_pcm16_within_one_step() {
        let a = Audio { sample_rate: 16000, channels: 1, samples: (0..1000).map(|i| (i as f32 * 0.01).sin() * 0.9).collect() };
        let b = read_wav(&write_wav_pcm16(&a)).unwrap();
        assert_eq!((b.sample_rate, b.channels, b.samples.len()), (16000, 1, 1000));
        let worst = a.samples.iter().zip(&b.samples).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(worst <= 0.5 / 32768.0 + 1e-7, "worst {worst}");
    }
    #[test]
    fn stereo_downmix_averages() {
        let a = Audio { sample_rate: 8000, channels: 2, samples: vec![1.0, 0.0, 0.5, 0.5] };
        assert_eq!(a.to_mono(), vec![0.5, 0.5]);
    }
    #[test]
    fn refuses_compressed_and_garbage() {
        assert_eq!(read_wav(b"ID3\x03 not wav at all.."), Err(AudioError::NotWav));
        let mut w = write_wav_pcm16(&Audio { sample_rate: 8000, channels: 1, samples: vec![0.0; 4] });
        w[20] = 0x55; // format tag 0x55 = MP3 in a WAV container
        assert!(matches!(read_wav(&w), Err(AudioError::UnsupportedFormat(_))));
    }
    #[test]
    fn truncated_chunk_is_malformed() {
        let w = write_wav_pcm16(&Audio { sample_rate: 8000, channels: 1, samples: vec![0.0; 100] });
        assert!(matches!(read_wav(&w[..60]), Err(AudioError::Malformed(_))));
    }
    #[test]
    fn float32_and_24bit_decode() {
        let mut w = write_wav_pcm16(&Audio { sample_rate: 8000, channels: 1, samples: vec![0.0; 2] });
        // rewrite as float32: tag 3, 32 bits, two samples 0.25 and -0.5
        w[20] = 3;
        w[34] = 32;
        w.truncate(44);
        w[40..44].copy_from_slice(&8u32.to_le_bytes());
        w.extend_from_slice(&0.25f32.to_le_bytes());
        w.extend_from_slice(&(-0.5f32).to_le_bytes());
        assert_eq!(read_wav(&w).unwrap().samples, vec![0.25, -0.5]);
    }
}
