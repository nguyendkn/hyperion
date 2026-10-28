//! The arch -> tensor_format table has one Rust definition. This test keeps the
//! committed JSON dump equal to it; `ignition/tests/test_format_table_crosscheck.py`
//! compares that dump with the Python hardware recipes.
//! Set `UPDATE_FIXTURES=1` to rewrite the dump.

use std::path::PathBuf;

use serde_json::{json, Value};

fn dump_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/format-table.json")
}

#[test]
fn committed_format_table_matches_the_rust_table() {
    let mut table = serde_json::Map::new();
    for (arch, formats) in helios_hal::format_table() {
        table.insert(arch.into(), json!(formats));
    }
    let actual = Value::Object(table);
    if std::env::var_os("UPDATE_FIXTURES").is_some() {
        std::fs::write(dump_path(), serde_json::to_string_pretty(&actual).unwrap() + "\n").unwrap();
    }
    let committed: Value =
        serde_json::from_slice(&std::fs::read(dump_path()).expect("dump exists: run with UPDATE_FIXTURES=1")).unwrap();
    assert_eq!(actual, committed, "run with UPDATE_FIXTURES=1 to refresh {}", dump_path().display());
}

#[test]
fn hopper_never_lists_fp4() {
    let table: Vec<_> = helios_hal::format_table();
    let (_, sm90) = table.iter().find(|(arch, _)| *arch == "sm_90").unwrap();
    assert!(!sm90.contains(&"fp4"));
}
