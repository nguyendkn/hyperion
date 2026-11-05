//! Writes the family manifest JSON Schema. Usage:
//! `cargo run -p hyperion-specification --features schema --bin gen-family-schema [OUT]`
//! (default `specification/schema/family-manifest.v1.json`).

use std::path::PathBuf;

fn main() -> std::io::Result<()> {
    let out = std::env::args_os().nth(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("schema/family-manifest.v1.json"));
    std::fs::write(&out, hyperion_specification::family_schema::family_schema_json())?;
    eprintln!("wrote {}", out.display());
    Ok(())
}
