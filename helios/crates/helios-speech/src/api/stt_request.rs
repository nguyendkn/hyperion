//! `POST /v1/audio/transcriptions` request validation. Fields outside the supported set are a 400, never ignored (ADR 0012).
use super::error::ApiError;
use super::multipart::Part;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SttFormat {
    Json,
    Text,
    VerboseJson,
    Srt,
    Vtt,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SttRequest {
    pub file: Vec<u8>,
    pub filename: Option<String>,
    pub model: String,
    pub language: Option<String>,
    pub format: SttFormat,
    pub temperature: f32,
}

fn text(p: &Part) -> Result<&str, ApiError> {
    std::str::from_utf8(&p.data).map(str::trim).map_err(|_| ApiError::bad_request("invalid_parameter", Some(&p.name), format!("field `{}` is not UTF-8 text", p.name)))
}

pub fn from_parts(parts: &[Part]) -> Result<SttRequest, ApiError> {
    let (mut file, mut filename, mut model, mut language) = (None, None, None, None);
    let (mut format, mut temperature) = (SttFormat::Json, 0.0f32);
    for p in parts {
        match p.name.as_str() {
            "file" => {
                if file.is_some() {
                    return Err(ApiError::bad_request("invalid_parameter", Some("file"), "`file` given twice"));
                }
                file = Some(p.data.clone());
                filename = p.filename.clone();
            }
            "model" => model = Some(text(p)?.to_string()),
            "language" => {
                let l = text(p)?;
                if l.is_empty() || l.len() > 8 || !l.bytes().all(|b| b.is_ascii_alphabetic() || b == b'-') {
                    return Err(ApiError::bad_request("invalid_parameter", Some("language"), "`language` must be an ISO code such as `en` or `vi`"));
                }
                language = Some(l.to_ascii_lowercase());
            }
            "response_format" => {
                format = match text(p)? {
                    "json" => SttFormat::Json,
                    "text" => SttFormat::Text,
                    "verbose_json" => SttFormat::VerboseJson,
                    "srt" => SttFormat::Srt,
                    "vtt" => SttFormat::Vtt,
                    other => return Err(ApiError::bad_request("invalid_parameter", Some("response_format"), format!("response_format `{other}` is not one of json, text, verbose_json, srt, vtt"))),
                }
            }
            "temperature" => {
                temperature = text(p)?.parse().ok().filter(|t: &f32| (0.0..=1.0).contains(t)).ok_or_else(|| ApiError::bad_request("invalid_parameter", Some("temperature"), "`temperature` must be a number from 0 to 1"))?
            }
            other => return Err(ApiError::unsupported_field(other)), // prompt, timestamp_granularities[], stream, ... are not supported
        }
    }
    let file = file.ok_or_else(|| ApiError::bad_request("missing_parameter", Some("file"), "`file` is required"))?;
    if file.is_empty() {
        return Err(ApiError::bad_request("invalid_parameter", Some("file"), "`file` is empty"));
    }
    let model = model.filter(|m| !m.is_empty()).ok_or_else(|| ApiError::bad_request("missing_parameter", Some("model"), "`model` is required"))?;
    Ok(SttRequest { file, filename, model, language, format, temperature })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn p(name: &str, data: &str) -> Part {
        Part { name: name.into(), filename: None, data: data.as_bytes().to_vec() }
    }
    fn base() -> Vec<Part> {
        vec![Part { name: "file".into(), filename: Some("a.wav".into()), data: vec![1, 2] }, p("model", "m")]
    }
    #[test]
    fn minimal_and_full() {
        let r = from_parts(&base()).unwrap();
        assert_eq!((r.format, r.temperature, r.language), (SttFormat::Json, 0.0, None));
        let mut v = base();
        v.extend([p("language", "VI"), p("response_format", "srt"), p("temperature", "0.2")]);
        let r = from_parts(&v).unwrap();
        assert_eq!((r.format, r.language.as_deref()), (SttFormat::Srt, Some("vi")));
    }
    #[test]
    fn refusals_are_400_naming_the_field() {
        for (name, val, param) in [("prompt", "x", "prompt"), ("response_format", "mp3", "response_format"), ("temperature", "2", "temperature"), ("language", "1!", "language"), ("timestamp_granularities[]", "word", "timestamp_granularities[]")] {
            let mut v = base();
            v.push(p(name, val));
            let e = from_parts(&v).unwrap_err();
            assert_eq!((e.status, e.param.as_deref()), (400, Some(param)), "{name}");
        }
        assert!(from_parts(&base()[..1]).is_err(), "model missing");
        assert!(from_parts(&base()[1..]).is_err(), "file missing");
    }
}
