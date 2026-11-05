/// Model families served by Helios (ADR 0012). Adding one is a variant here plus the checklist of the ADR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFamily {
    DecoderLm,
    Stt,
    Tts,
}

/// Scheduling class, highest first (derive order): interactive audio > interactive text > batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PriorityClass {
    InteractiveAudio,
    InteractiveText,
    Batch,
}

impl ModelFamily {
    /// Default class of a family's requests; a request may be demoted to `Batch`, never promoted.
    pub fn default_class(self) -> PriorityClass {
        match self {
            ModelFamily::Stt | ModelFamily::Tts => PriorityClass::InteractiveAudio,
            ModelFamily::DecoderLm => PriorityClass::InteractiveText,
        }
    }
}
