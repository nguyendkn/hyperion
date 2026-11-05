//! JSON Schema of the family envelope, generated from the Rust structs (committed as `schema/family-manifest.v1.json`). `schema` feature only.
use crate::family::FamilyManifest;

pub fn family_schema_json() -> String {
    let schema = schemars::schema_for!(FamilyManifest);
    let mut out = serde_json::to_string_pretty(&schema).expect("schema serializes to JSON");
    out.push('\n');
    out
}
