//! JSON Schema for the manifest, generated from the Rust structs (the schema
//! source of truth). Only built with the `schema` feature.

use crate::Manifest;

/// Pretty-printed JSON Schema (draft 2020-12) for [`Manifest`], with a
/// trailing newline so the committed file is diff-friendly.
pub fn manifest_schema_json() -> String {
    let schema = schemars::schema_for!(Manifest);
    let mut out = serde_json::to_string_pretty(&schema).expect("schema serializes to JSON");
    out.push('\n');
    out
}
