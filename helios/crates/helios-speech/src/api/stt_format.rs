//! Response bodies of the transcription endpoint. Segments come from the model's own timestamps where it emits them; a model without
//! timestamps gives one segment spanning the clip and `verbose_json`/`srt`/`vtt` say so in the data they return, they never invent word times.
use super::stt_request::SttFormat;
use serde_json::json;

#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

fn stamp(t: f64, comma: bool) -> String {
    let ms = (t.max(0.0) * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02}{}{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, if comma { ',' } else { '.' }, ms % 1000)
}

/// `(content_type, body)`.
pub fn render(format: SttFormat, language: &str, duration: f64, segments: &[Segment]) -> (&'static str, String) {
    let full = segments.iter().map(|s| s.text.trim()).filter(|t| !t.is_empty()).collect::<Vec<_>>().join(" ");
    match format {
        SttFormat::Json => ("application/json", json!({ "text": full }).to_string()),
        SttFormat::Text => ("text/plain; charset=utf-8", format!("{full}\n")),
        SttFormat::VerboseJson => {
            let segs: Vec<_> = segments.iter().enumerate().map(|(i, s)| json!({ "id": i, "start": s.start, "end": s.end, "text": s.text })).collect();
            ("application/json", json!({ "task": "transcribe", "language": language, "duration": duration, "text": full, "segments": segs }).to_string())
        }
        SttFormat::Srt => ("application/x-subrip", segments.iter().enumerate().map(|(i, s)| format!("{}\n{} --> {}\n{}\n\n", i + 1, stamp(s.start, true), stamp(s.end, true), s.text.trim())).collect()),
        SttFormat::Vtt => ("text/vtt; charset=utf-8", format!("WEBVTT\n\n{}", segments.iter().map(|s| format!("{} --> {}\n{}\n\n", stamp(s.start, false), stamp(s.end, false), s.text.trim())).collect::<String>())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn segs() -> Vec<Segment> {
        vec![Segment { start: 0.0, end: 1.5, text: " xin chào".into() }, Segment { start: 3661.25, end: 3662.0, text: "thế giới".into() }]
    }
    #[test]
    fn json_text_and_verbose() {
        assert_eq!(render(SttFormat::Json, "vi", 4.0, &segs()).1, r#"{"text":"xin chào thế giới"}"#);
        assert_eq!(render(SttFormat::Text, "vi", 4.0, &segs()).1, "xin chào thế giới\n");
        let v: serde_json::Value = serde_json::from_str(&render(SttFormat::VerboseJson, "vi", 4.0, &segs()).1).unwrap();
        assert_eq!((v["language"].as_str(), v["segments"][1]["id"].as_u64()), (Some("vi"), Some(1)));
    }
    #[test]
    fn srt_uses_comma_and_vtt_uses_dot_with_hours() {
        assert_eq!(render(SttFormat::Srt, "vi", 4.0, &segs()).1, "1\n00:00:00,000 --> 00:00:01,500\nxin chào\n\n2\n01:01:01,250 --> 01:01:02,000\nthế giới\n\n");
        let v = render(SttFormat::Vtt, "vi", 4.0, &segs()).1;
        assert!(v.starts_with("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\n") && v.contains("01:01:01.250 --> 01:01:02.000"));
    }
    #[test]
    fn empty_transcript_is_valid() {
        assert_eq!(render(SttFormat::Json, "en", 1.0, &[]).1, r#"{"text":""}"#);
    }
}
