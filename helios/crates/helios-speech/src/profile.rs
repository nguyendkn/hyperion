//! Production gate (E3): the default profile refuses a model whose manifest has `production_approved = false`; the `dev` profile, declared
//! explicitly at startup, loads it with a warning, and every response of that model carries `X-Hyperion-Model-Status: provisional`.
//! Only the owner flips the manifest field (an ADR edit with evidence); nothing here can.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Profile {
    #[default]
    Default,
    Dev,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus {
    Approved,
    Provisional,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("model `{model}` is not approved for production (manifest production_approved = false); start the server with the explicit `dev` profile to load it, or have the owner approve it")]
pub struct NotApproved {
    pub model: String,
}

pub const STATUS_HEADER: &str = "X-Hyperion-Model-Status";

impl Profile {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "default" => Some(Self::Default),
            "dev" => Some(Self::Dev),
            _ => None,
        }
    }

    /// Decide at load time. A provisional model in the `dev` profile is allowed; the caller logs the startup warning.
    pub fn admit(self, model: &str, production_approved: bool) -> Result<ModelStatus, NotApproved> {
        match (production_approved, self) {
            (true, _) => Ok(ModelStatus::Approved),
            (false, Profile::Dev) => Ok(ModelStatus::Provisional),
            (false, Profile::Default) => Err(NotApproved { model: model.to_string() }),
        }
    }
}

impl ModelStatus {
    /// The header every response of the model carries, or none for an approved model.
    pub fn header(self) -> Option<(&'static str, &'static str)> {
        (self == ModelStatus::Provisional).then_some((STATUS_HEADER, "provisional"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_refuses_unapproved_and_names_the_model() {
        let e = Profile::Default.admit("vieneu", false).unwrap_err();
        assert_eq!(e.model, "vieneu");
        assert!(e.to_string().contains("`vieneu`") && e.to_string().contains("dev"));
        assert_eq!(Profile::Default.admit("qwen3-asr", true), Ok(ModelStatus::Approved));
    }
    #[test]
    fn dev_loads_unapproved_as_provisional_with_the_header() {
        let s = Profile::Dev.admit("vieneu", false).unwrap();
        assert_eq!((s, s.header()), (ModelStatus::Provisional, Some(("X-Hyperion-Model-Status", "provisional"))));
        assert_eq!(Profile::Dev.admit("x", true).unwrap().header(), None);
    }
    #[test]
    fn profile_is_declared_explicitly() {
        assert_eq!(Profile::default(), Profile::Default);
        assert_eq!((Profile::parse("dev"), Profile::parse("Dev"), Profile::parse("prod")), (Some(Profile::Dev), None, None));
    }
}
