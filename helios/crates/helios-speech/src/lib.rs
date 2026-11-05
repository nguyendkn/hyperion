//! Multi-model serving pieces of ADR 0012: the family enum, the registry, the memory partition planner and the arbiter.
//! Nothing here touches a GPU or the LLM scheduler; an LLM-only configuration never constructs any of it.
pub mod api;
pub mod arbiter;
pub mod family;
pub mod partition;
pub mod registry;
