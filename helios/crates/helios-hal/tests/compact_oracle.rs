//! The compact oracle file the Python script writes must be readable by the Rust side that compares
//! against it. The file here comes from `scripts/gen-oracle-large.py` run on the tiny sharded fixture
//! (float32 and bfloat16 Hugging Face runs on the CPU); no GPU is involved.

use std::path::PathBuf;

use helios_hal::oracle::{compare, CompactOracle};

fn oracle_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../testdata/oracle/qwen2-tiny-sharded.compact.json")
}

#[test]
fn the_script_output_parses_with_the_declared_shape() {
    let oracle = CompactOracle::from_json(&std::fs::read(oracle_path()).unwrap()).expect("the script's output is accepted");
    assert_eq!(oracle.model, "qwen2-tiny-sharded");
    assert!(oracle.prompts.len() >= 3);
    for prompt in &oracle.prompts {
        assert_eq!(prompt.reference.len(), oracle.generated_tokens, "{}", prompt.label);
        assert_eq!(prompt.generated_ids.len(), oracle.generated_tokens);
        for (i, record) in prompt.reference.iter().enumerate() {
            assert_eq!(record.position, prompt.ids.len() - 1 + i, "records start at the last prompt position");
            assert_eq!(record.argmax, record.top_ids[0], "the argmax is the best of the top-k");
            assert!(record.top_logits.windows(2).all(|w| w[0] >= w[1]), "top-k is sorted");
        }
        // The tokens the reference generated are its own argmax at the position before each of them.
        for (token, record) in prompt.generated_ids.iter().zip(&prompt.reference) {
            assert_eq!(*token, record.argmax, "greedy generation follows the argmax");
        }
    }
    assert!(oracle.tolerance.max_abs_top_logits > 0.0 && oracle.tolerance.max_abs_logsumexp > 0.0);
}

#[test]
fn a_reference_compared_with_a_row_rebuilt_from_its_own_top_k_agrees_on_those_ids() {
    let oracle = CompactOracle::from_json(&std::fs::read(oracle_path()).unwrap()).unwrap();
    let record = &oracle.prompts[0].reference[0];
    // The fixture's vocabulary is 128: a row that carries the reference's top logits at their ids and
    // a floor elsewhere reproduces the top-k exactly, so only the normaliser can differ.
    let mut row = vec![-100.0f32; 128];
    for (&id, &logit) in record.top_ids.iter().zip(&record.top_logits) {
        row[id as usize] = logit;
    }
    let c = compare(&row, record, &oracle.tolerance).unwrap();
    assert_eq!(c.max_abs_top_logits, 0.0);
    assert_eq!(c.argmax, record.argmax);
    assert!(c.argmax_ok);
}
