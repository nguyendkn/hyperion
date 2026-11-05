/// Every refusal of the audio front end is typed; nothing panics on bad input.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum AudioError {
    #[error("not a RIFF/WAVE file")]
    NotWav,
    #[error("compressed or unsupported audio format ({0}); only WAV with PCM 8/16/24/32-bit or IEEE float 32/64-bit is accepted until a decoder dependency is approved")]
    UnsupportedFormat(String),
    #[error("malformed WAV: {0}")]
    Malformed(String),
    #[error("invalid parameter: {0}")]
    Invalid(String),
}
