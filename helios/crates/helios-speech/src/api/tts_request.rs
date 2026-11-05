//! `POST /v1/audio/speech` request validation (JSON). Unknown fields and unsupported values are a 400; compressed formats are refused until
//! the owner approves an encoder (ADR 0012).
use super::error::ApiError;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsFormat {
    Wav,
    Pcm,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    model: Option<String>,
    input: Option<String>,
    voice: Option<String>,
    response_format: Option<String>,
    speed: Option<f64>,
    /// Chunked streaming of the audio body (extension, off by default).
    stream: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TtsRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    pub format: TtsFormat,
    pub speed: f64,
    pub stream: bool,
}

/// OpenAI's own limit on `input`.
pub const MAX_INPUT_CHARS: usize = 4096;

/// `voices`: the preset voices of the addressed model (a voice outside them is a 400, there is no cloning).
pub fn parse(body: &[u8], voices: &[String]) -> Result<TtsRequest, ApiError> {
    let raw: Raw = serde_json::from_slice(body).map_err(|e| {
        let m = e.to_string();
        match m.split('`').nth(1).filter(|_| m.starts_with("unknown field")) {
            Some(f) => ApiError::unsupported_field(f),
            None => ApiError::bad_request("invalid_json", None, format!("invalid request body: {m}")),
        }
    })?;
    let req = |f: &str, v: Option<String>| v.filter(|s| !s.trim().is_empty()).ok_or_else(|| ApiError::bad_request("missing_parameter", Some(f), format!("`{f}` is required")));
    let (model, input) = (req("model", raw.model)?, req("input", raw.input)?);
    if input.chars().count() > MAX_INPUT_CHARS {
        return Err(ApiError::bad_request("input_too_long", Some("input"), format!("`input` is longer than {MAX_INPUT_CHARS} characters")));
    }
    let voice = match raw.voice {
        Some(v) if voices.contains(&v) => v,
        Some(v) => return Err(ApiError::bad_request("unknown_voice", Some("voice"), format!("voice `{v}` is not one of: {}", voices.join(", ")))),
        None => voices.first().cloned().ok_or_else(|| ApiError::bad_request("unknown_voice", Some("voice"), "the model has no preset voice"))?,
    };
    let format = match raw.response_format.as_deref() {
        None | Some("wav") => TtsFormat::Wav,
        Some("pcm") => TtsFormat::Pcm,
        Some(f @ ("mp3" | "opus" | "aac" | "flac")) => return Err(ApiError::bad_request("unsupported_format", Some("response_format"), format!("`{f}` needs an audio encoder this build does not have; use wav or pcm"))),
        Some(f) => return Err(ApiError::bad_request("invalid_parameter", Some("response_format"), format!("response_format `{f}` is not one of wav, pcm"))),
    };
    let speed = raw.speed.unwrap_or(1.0);
    if !(0.25..=4.0).contains(&speed) {
        return Err(ApiError::bad_request("invalid_parameter", Some("speed"), "`speed` must be from 0.25 to 4.0"));
    }
    Ok(TtsRequest { model, input, voice, format, speed, stream: raw.stream.unwrap_or(false) })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn v() -> Vec<String> {
        vec!["a".into(), "b".into()]
    }
    #[test]
    fn defaults_and_full() {
        let r = parse(br#"{"model":"m","input":"hi"}"#, &v()).unwrap();
        assert_eq!((r.voice.as_str(), r.format, r.speed, r.stream), ("a", TtsFormat::Wav, 1.0, false));
        let r = parse(br#"{"model":"m","input":"hi","voice":"b","response_format":"pcm","speed":1.5,"stream":true}"#, &v()).unwrap();
        assert_eq!((r.voice.as_str(), r.format, r.stream), ("b", TtsFormat::Pcm, true));
    }
    #[test]
    fn refusals() {
        let cases: [(&str, &str); 7] = [
            (r#"{"model":"m","input":"hi","instructions":"x"}"#, "instructions"),
            (r#"{"model":"m","input":"hi","voice":"zzz"}"#, "voice"),
            (r#"{"model":"m","input":"hi","response_format":"mp3"}"#, "response_format"),
            (r#"{"model":"m","input":"hi","speed":9}"#, "speed"),
            (r#"{"model":"m","input":"  "}"#, "input"),
            (r#"{"input":"hi"}"#, "model"),
            (r#"{"model":"m","input":"hi","voice":7}"#, ""),
        ];
        for (body, param) in cases {
            let e = parse(body.as_bytes(), &v()).unwrap_err();
            assert_eq!(e.status, 400, "{body}");
            if !param.is_empty() {
                assert_eq!(e.param.as_deref(), Some(param), "{body}");
            }
        }
        let long = format!(r#"{{"model":"m","input":"{}"}}"#, "a".repeat(MAX_INPUT_CHARS + 1));
        assert_eq!(parse(long.as_bytes(), &v()).unwrap_err().code, "input_too_long");
        assert!(parse(b"not json", &v()).is_err());
    }
}
