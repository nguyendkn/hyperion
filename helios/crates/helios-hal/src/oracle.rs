//! The compact logit oracle written by `scripts/gen-oracle-large.py` and the rule a Hyperion forward is
//! compared with it. Pure host logic with no CUDA dependency, so it is unit-tested everywhere and the
//! hardware test only supplies the logits.
//!
//! The file keeps, for a few prompts and each position that predicts a token, the float32 reference's argmax,
//! top-k logits with their ids and the logsumexp of the whole vocabulary row, plus the tolerances declared
//! before any Hyperion run (the measured bfloat16-vs-float32 drift times a margin).

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

use crate::sampling::argmax;

const FORMAT: &str = "compact-oracle-v1";

/// What the reference said about one position.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub position: usize,
    pub argmax: u32,
    pub top_ids: Vec<u32>,
    pub top_logits: Vec<f32>,
    pub logsumexp: f64,
}

/// The limits declared in the oracle file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    pub max_abs_top_logits: f32,
    pub max_abs_logsumexp: f64,
}

#[derive(Debug, Clone)]
pub struct OraclePrompt {
    pub label: String,
    pub ids: Vec<u32>,
    /// The float32 reference's greedy continuation.
    pub generated_ids: Vec<u32>,
    /// One record per generated token, the first at the last prompt position.
    pub reference: Vec<Record>,
    /// Per generated token, how far the Hugging Face bfloat16 run was from float32 at that position
    /// (`(max |top-k logit diff|, |logsumexp diff|)`). Absent in files written before it was recorded.
    pub bf16_deviation: Option<Vec<(f32, f64)>>,
}

/// How far the Hugging Face bfloat16 run was from float32 (the numbers the tolerance was derived from), and
/// which term of `max(margin * drift, floor)` set the top-k limit.
#[derive(Debug, Clone, PartialEq)]
pub struct Bf16Drift {
    pub max_abs_top_logits: f32,
    pub max_abs_logsumexp: f64,
    pub binding_top_logits: String,
}

#[derive(Debug, Clone)]
pub struct CompactOracle {
    pub model: String,
    pub generated_tokens: usize,
    pub tolerance: Tolerance,
    /// Absent in files written before the drift was recorded.
    pub bf16_drift: Option<Bf16Drift>,
    pub prompts: Vec<OraclePrompt>,
}

fn ids_of(value: &Value, what: &str) -> Result<Vec<u32>> {
    value
        .as_array()
        .with_context(|| format!("{what} must be an array"))?
        .iter()
        .map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok()).with_context(|| format!("{what} must hold token ids")))
        .collect()
}

fn record_of(value: &Value) -> Result<Record> {
    let top_logits: Vec<f32> = value["top_logits"]
        .as_array()
        .context("top_logits must be an array")?
        .iter()
        .map(|v| v.as_f64().map(|x| x as f32).context("top_logits must hold numbers"))
        .collect::<Result<_>>()?;
    let top_ids = ids_of(&value["top_ids"], "top_ids")?;
    if top_ids.len() != top_logits.len() || top_ids.len() < 2 {
        bail!("a record needs at least two top ids with one logit each");
    }
    Ok(Record {
        position: value["position"].as_u64().context("position")? as usize,
        argmax: u32::try_from(value["argmax"].as_u64().context("argmax")?)?,
        top_ids,
        top_logits,
        logsumexp: value["logsumexp"].as_f64().context("logsumexp")?,
    })
}

impl CompactOracle {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let doc: Value = serde_json::from_slice(bytes).context("oracle is not JSON")?;
        if doc["format"] != FORMAT {
            bail!("unsupported oracle format {:?}, expected {FORMAT}", doc["format"]);
        }
        let tolerance = Tolerance {
            max_abs_top_logits: doc["tolerance"]["max_abs_top_logits"].as_f64().context("tolerance.max_abs_top_logits")? as f32,
            max_abs_logsumexp: doc["tolerance"]["max_abs_logsumexp"].as_f64().context("tolerance.max_abs_logsumexp")?,
        };
        let generated_tokens = doc["generated_tokens"].as_u64().context("generated_tokens")? as usize;
        let mut prompts = Vec::new();
        for prompt in doc["prompts"].as_array().context("prompts")? {
            let reference: Vec<Record> = prompt["reference_fp32"].as_array().context("reference_fp32")?.iter().map(record_of).collect::<Result<_>>()?;
            let generated_ids = ids_of(&prompt["generated_ids"], "generated_ids")?;
            if reference.len() != generated_tokens || generated_ids.len() != generated_tokens {
                bail!("a prompt must have {generated_tokens} generated ids and records");
            }
            let bf16_deviation = match prompt["bf16_deviation"].as_array() {
                None => None,
                Some(rows) => {
                    let parsed: Vec<(f32, f64)> = rows
                        .iter()
                        .map(|r| Ok((r["top_logits"].as_f64().context("bf16_deviation.top_logits")? as f32, r["logsumexp"].as_f64().context("bf16_deviation.logsumexp")?)))
                        .collect::<Result<_>>()?;
                    if parsed.len() != generated_tokens {
                        bail!("bf16_deviation must have {generated_tokens} entries");
                    }
                    Some(parsed)
                }
            };
            prompts.push(OraclePrompt {
                label: prompt["label"].as_str().context("label")?.to_string(),
                ids: ids_of(&prompt["ids"], "ids")?,
                generated_ids,
                reference,
                bf16_deviation,
            });
        }
        if prompts.is_empty() {
            bail!("the oracle has no prompts");
        }
        let bf16_drift = match (doc["bf16_vs_fp32"]["top_logits"].as_f64(), doc["bf16_vs_fp32"]["logsumexp"].as_f64()) {
            (Some(top), Some(lse)) => Some(Bf16Drift {
                max_abs_top_logits: top as f32,
                max_abs_logsumexp: lse,
                binding_top_logits: doc["tolerance"]["binding_top_logits"].as_str().unwrap_or("unrecorded").to_string(),
            }),
            _ => None,
        };
        Ok(Self { model: doc["model"].as_str().context("model")?.to_string(), generated_tokens, tolerance, bf16_drift, prompts })
    }
}

/// `log(sum(exp(row)))` in float64, shifted by the maximum.
pub fn logsumexp(row: &[f32]) -> f64 {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    max + row.iter().map(|&v| (f64::from(v) - max).exp()).sum::<f64>().ln()
}

/// How one logits row differs from the reference record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Comparison {
    /// Largest `|ours - reference|` over the reference's top-k ids.
    pub max_abs_top_logits: f32,
    /// `|our logsumexp - reference logsumexp|`.
    pub abs_logsumexp: f64,
    pub argmax: u32,
    /// Our argmax equals the reference's, or the reference's two best logits are within the tolerance
    /// of each other and ours is one of them.
    pub argmax_ok: bool,
}

impl Comparison {
    pub fn within(&self, tolerance: &Tolerance) -> bool {
        self.max_abs_top_logits <= tolerance.max_abs_top_logits && self.abs_logsumexp <= tolerance.max_abs_logsumexp && self.argmax_ok
    }
}

/// Compare the full logits row `ours` with one reference record under `tolerance`.
pub fn compare(ours: &[f32], reference: &Record, tolerance: &Tolerance) -> Result<Comparison> {
    let mut worst = 0.0f32;
    for (&id, &want) in reference.top_ids.iter().zip(&reference.top_logits) {
        let got = *ours.get(id as usize).ok_or_else(|| anyhow!("reference id {id} is outside our {} logits", ours.len()))?;
        worst = worst.max((got - want).abs());
    }
    let picked = argmax(ours) as u32;
    let gap = reference.top_logits[0] - reference.top_logits[1];
    let near_tie = gap <= tolerance.max_abs_top_logits && (picked == reference.top_ids[0] || picked == reference.top_ids[1]);
    Ok(Comparison {
        max_abs_top_logits: worst,
        abs_logsumexp: (logsumexp(ours) - reference.logsumexp).abs(),
        argmax: picked,
        argmax_ok: picked == reference.argmax || near_tie,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row with distinct, well-separated values, and the record a reference would keep from it.
    fn row() -> Vec<f32> {
        (0..64).map(|i| ((i * 37) % 64) as f32 * 0.25 - 8.0).collect()
    }

    fn record_of_row(row: &[f32], k: usize) -> Record {
        let mut order: Vec<usize> = (0..row.len()).collect();
        order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
        Record {
            position: 0,
            argmax: order[0] as u32,
            top_ids: order[..k].iter().map(|&i| i as u32).collect(),
            top_logits: order[..k].iter().map(|&i| row[i]).collect(),
            logsumexp: logsumexp(row),
        }
    }

    const TOL: Tolerance = Tolerance { max_abs_top_logits: 0.1, max_abs_logsumexp: 0.05 };

    #[test]
    fn identical_logits_have_no_difference() {
        let r = row();
        let c = compare(&r, &record_of_row(&r, 8), &TOL).unwrap();
        assert_eq!(c.max_abs_top_logits, 0.0);
        assert!(c.abs_logsumexp < 1e-9 && c.argmax_ok && c.within(&TOL));
    }

    #[test]
    fn a_top_logit_off_by_more_than_the_tolerance_fails() {
        let r = row();
        let reference = record_of_row(&r, 8);
        let mut ours = r.clone();
        ours[reference.top_ids[3] as usize] += 0.2;
        let c = compare(&ours, &reference, &TOL).unwrap();
        assert!((c.max_abs_top_logits - 0.2).abs() < 1e-6);
        assert!(!c.within(&TOL));
        ours[reference.top_ids[3] as usize] = r[reference.top_ids[3] as usize] + 0.05;
        assert!(compare(&ours, &reference, &TOL).unwrap().within(&TOL), "0.05 is inside 0.1");
    }

    #[test]
    fn a_shifted_logsumexp_fails_even_when_the_top_logits_agree() {
        let r = row();
        let reference = record_of_row(&r, 4);
        // Raise every logit outside the top 4: the top ids are untouched, the normaliser moves.
        let mut ours = r.clone();
        for (i, v) in ours.iter_mut().enumerate() {
            if !reference.top_ids.contains(&(i as u32)) {
                *v += 1.5;
            }
        }
        let c = compare(&ours, &reference, &TOL).unwrap();
        assert_eq!(c.max_abs_top_logits, 0.0);
        assert!(c.abs_logsumexp > TOL.max_abs_logsumexp && !c.within(&TOL));
    }

    #[test]
    fn a_near_tie_may_flip_but_a_clear_winner_may_not() {
        let mut r = vec![0.0f32; 16];
        r[3] = 5.0;
        r[7] = 4.95; // within the 0.1 tolerance of the best
        r[1] = 1.0;
        let reference = record_of_row(&r, 4);
        assert_eq!(reference.argmax, 3);
        let mut flipped = r.clone();
        flipped[7] = 5.02; // ours prefers token 7
        let c = compare(&flipped, &reference, &Tolerance { max_abs_top_logits: 0.2, ..TOL }).unwrap();
        assert_eq!(c.argmax, 7);
        assert!(c.argmax_ok, "the reference's two best were within the tolerance");

        let mut clear = r.clone();
        clear[3] = 1.0; // ours now picks 7 although the reference had it 0.05 behind: allowed only for a near tie
        let strict = compare(&clear, &reference, &Tolerance { max_abs_top_logits: 0.01, ..TOL }).unwrap();
        assert!(!strict.argmax_ok, "with a tolerance smaller than the gap a flip is an error");
        // A third token winning is never a near-tie flip.
        let mut third = r.clone();
        third[1] = 9.0;
        assert!(!compare(&third, &reference, &Tolerance { max_abs_top_logits: 10.0, ..TOL }).unwrap().argmax_ok);
    }

    #[test]
    fn a_reference_id_outside_the_row_is_an_error() {
        let reference = Record { position: 0, argmax: 0, top_ids: vec![0, 99], top_logits: vec![1.0, 0.5], logsumexp: 1.0 };
        assert!(compare(&[0.0; 4], &reference, &TOL).unwrap_err().to_string().contains("outside"));
    }

    #[test]
    fn logsumexp_matches_a_direct_computation() {
        let v = [0.5f32, -1.0, 2.0];
        let direct = v.iter().map(|&x| f64::from(x).exp()).sum::<f64>().ln();
        assert!((logsumexp(&v) - direct).abs() < 1e-12);
        assert!(logsumexp(&[1000.0, 1000.0]).is_finite(), "shifted by the maximum, so no overflow");
    }

    fn document(format: &str) -> String {
        let rec = r#"{"position":4,"argmax":2,"top_ids":[2,1],"top_logits":[3.0,2.5],"logsumexp":3.6}"#;
        format!(
            r#"{{"format":"{format}","model":"m","generated_tokens":1,"tolerance":{{"max_abs_top_logits":0.1,"max_abs_logsumexp":0.05}},
            "prompts":[{{"label":"p","ids":[1,2,3,4,5],"generated_ids":[2],"reference_fp32":[{rec}]}}]}}"#
        )
    }

    #[test]
    fn a_document_parses_and_a_wrong_format_or_shape_is_refused() {
        let oracle = CompactOracle::from_json(document(FORMAT).as_bytes()).unwrap();
        assert_eq!(oracle.prompts[0].reference[0].argmax, 2);
        assert_eq!(oracle.tolerance, Tolerance { max_abs_top_logits: 0.1, max_abs_logsumexp: 0.05 });
        assert!(CompactOracle::from_json(document("compact-oracle-v0").as_bytes()).unwrap_err().to_string().contains("unsupported"));
        let short = document(FORMAT).replace("\"generated_tokens\":1", "\"generated_tokens\":2");
        assert!(CompactOracle::from_json(short.as_bytes()).is_err());
        assert!(CompactOracle::from_json(b"not json").is_err());
    }

    #[test]
    fn the_per_position_bf16_deviation_is_read_and_must_match_the_generated_count() {
        let with = document(FORMAT).replace("\"reference_fp32\"", "\"bf16_deviation\":[{\"top_logits\":0.5,\"logsumexp\":0.25}],\"reference_fp32\"");
        assert_eq!(CompactOracle::from_json(with.as_bytes()).unwrap().prompts[0].bf16_deviation, Some(vec![(0.5, 0.25)]));
        assert_eq!(CompactOracle::from_json(document(FORMAT).as_bytes()).unwrap().prompts[0].bf16_deviation, None);
        let wrong = with.replace("[{\"top_logits\":0.5,\"logsumexp\":0.25}]", "[]");
        assert!(CompactOracle::from_json(wrong.as_bytes()).is_err());
    }

    #[test]
    fn the_recorded_bf16_drift_and_its_binding_term_are_read_when_present() {
        assert_eq!(CompactOracle::from_json(document(FORMAT).as_bytes()).unwrap().bf16_drift, None);
        let with = document(FORMAT)
            .replace("\"max_abs_logsumexp\":0.05}", "\"max_abs_logsumexp\":0.05,\"binding_top_logits\":\"floor\"},\"bf16_vs_fp32\":{\"top_logits\":0.25,\"logsumexp\":0.125}");
        let drift = CompactOracle::from_json(with.as_bytes()).unwrap().bf16_drift.unwrap();
        assert_eq!(drift, Bf16Drift { max_abs_top_logits: 0.25, max_abs_logsumexp: 0.125, binding_top_logits: "floor".into() });
    }
}
