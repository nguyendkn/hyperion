use crate::family::{ModelFamily, PriorityClass};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEntry {
    pub family: ModelFamily,
    pub class: PriorityClass,
    /// Bound of this model's arbiter queue; a full queue is a 429 naming the model.
    pub queue_bound: usize,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RegistryError {
    #[error("model '{0}' is already registered")]
    Duplicate(String),
    #[error("unknown model '{0}'")]
    Unknown(String),
    #[error("model '{name}' is a {actual:?} model, this endpoint needs {wanted:?}")]
    WrongFamily { name: String, actual: ModelFamily, wanted: ModelFamily },
    #[error("queue bound of '{0}' must be at least 1")]
    ZeroQueue(String),
}

/// name -> entry. The runtime handle of each family is held next to it by the server; the registry holds only what routing and scheduling need.
#[derive(Debug, Default)]
pub struct ModelRegistry {
    models: BTreeMap<String, ModelEntry>,
}

impl ModelRegistry {
    pub fn register(&mut self, name: &str, entry: ModelEntry) -> Result<(), RegistryError> {
        if entry.queue_bound == 0 {
            return Err(RegistryError::ZeroQueue(name.into()));
        }
        if self.models.contains_key(name) {
            return Err(RegistryError::Duplicate(name.into()));
        }
        self.models.insert(name.into(), entry);
        Ok(())
    }
    /// Look a model up for an endpoint that serves one family (a wrong family is a 400, an unknown name a 404).
    pub fn resolve(&self, name: &str, wanted: ModelFamily) -> Result<&ModelEntry, RegistryError> {
        let e = self.models.get(name).ok_or_else(|| RegistryError::Unknown(name.into()))?;
        if e.family != wanted {
            return Err(RegistryError::WrongFamily { name: name.into(), actual: e.family, wanted });
        }
        Ok(e)
    }
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.models.keys().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn e(f: ModelFamily) -> ModelEntry {
        ModelEntry { family: f, class: f.default_class(), queue_bound: 4 }
    }

    #[test]
    fn resolve_checks_name_and_family() {
        let mut r = ModelRegistry::default();
        r.register("whisper", e(ModelFamily::Stt)).unwrap();
        assert!(r.resolve("whisper", ModelFamily::Stt).is_ok());
        assert_eq!(r.resolve("nope", ModelFamily::Stt), Err(RegistryError::Unknown("nope".into())));
        assert!(matches!(r.resolve("whisper", ModelFamily::Tts), Err(RegistryError::WrongFamily { .. })));
        assert!(r.register("whisper", e(ModelFamily::Stt)).is_err());
        assert!(r.register("z", ModelEntry { queue_bound: 0, ..e(ModelFamily::Tts) }).is_err());
    }
}
