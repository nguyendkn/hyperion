//! Writes the manifest JSON Schema. Usage:
//! `cargo run -p hyperion-specification --features schema --bin gen-schema [OUT]`
//! (default `specification/schema/manifest.v1.json`).

use std::path::PathBuf;

fn main() -> std::io::Result<()> {
    let out = std::env::args_os().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("schema/manifest.v1.json")
    });
    std::fs::write(&out, hyperion_specification::schema::manifest_schema_json())?;
    eprintln!("wrote {}", out.display());
    Ok(())
}
