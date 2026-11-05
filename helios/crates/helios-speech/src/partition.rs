//! Memory partition planning before Ready (ADR 0012 section 4). Speech models have a fixed footprint declared in bytes; the LLM is loaded last
//! and gets what is left, which `commit_kv` measures itself. This planner only decides, before anything is loaded, whether the sum can fit.

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PartitionError {
    #[error("model name '{0}' is declared twice")]
    DuplicateName(String),
    #[error("partitions do not fit: {needed} bytes needed ({detail}) but only {available} are available after the {reserve}-byte reserve of a {total}-byte device")]
    DoesNotFit { needed: u64, available: u64, reserve: u64, total: u64, detail: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fixed {
    pub name: String,
    /// Weights + activations + graph memory + cache, in bytes, from the model's spec.
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionPlan {
    pub fixed: Vec<Fixed>,
    /// Bytes left for the LLM (weights, graphs and KV together), 0 when there is no LLM.
    pub llm_bytes: u64,
}

/// `llm_min_bytes`: the smallest the LLM needs (weights + graphs + a minimum pool); `None` when no LLM is served.
pub fn plan(total: u64, reserve: u64, speech: &[Fixed], llm_min_bytes: Option<u64>) -> Result<PartitionPlan, PartitionError> {
    for (i, a) in speech.iter().enumerate() {
        if speech[..i].iter().any(|b| b.name == a.name) {
            return Err(PartitionError::DuplicateName(a.name.clone()));
        }
    }
    let available = total.saturating_sub(reserve);
    let fixed_sum: u64 = speech.iter().map(|f| f.bytes).sum();
    let needed = fixed_sum + llm_min_bytes.unwrap_or(0);
    if needed > available {
        let detail = speech
            .iter()
            .map(|f| format!("{}={}", f.name, f.bytes))
            .chain(llm_min_bytes.map(|b| format!("llm_min={b}")))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(PartitionError::DoesNotFit { needed, available, reserve, total, detail });
    }
    Ok(PartitionPlan { fixed: speech.to_vec(), llm_bytes: if llm_min_bytes.is_some() { available - fixed_sum } else { 0 } })
}

#[cfg(test)]
mod tests {
    use super::*;
    const GB: u64 = 1 << 30;
    fn f(n: &str, g: u64) -> Fixed {
        Fixed { name: n.into(), bytes: g * GB }
    }

    #[test]
    fn llm_gets_what_is_left() {
        let p = plan(80 * GB, 2 * GB, &[f("stt", 6), f("tts", 2)], Some(20 * GB)).unwrap();
        assert_eq!(p.llm_bytes, 70 * GB); // 80 - 2 reserve - 6 - 2
    }
    #[test]
    fn refuses_before_ready_when_the_sum_does_not_fit() {
        let e = plan(24 * GB, 2 * GB, &[f("stt", 6), f("tts", 2)], Some(20 * GB)).unwrap_err();
        assert!(matches!(e, PartitionError::DoesNotFit { needed, available, .. } if needed == 28 * GB && available == 22 * GB));
        assert!(e.to_string().contains("stt=") && e.to_string().contains("llm_min="));
    }
    #[test]
    fn duplicate_names_and_no_llm() {
        assert_eq!(plan(80 * GB, 0, &[f("a", 1), f("a", 1)], None), Err(PartitionError::DuplicateName("a".into())));
        assert_eq!(plan(80 * GB, 0, &[f("a", 1)], None).unwrap().llm_bytes, 0);
    }
    #[test]
    fn exact_fit_is_accepted() {
        assert!(plan(10 * GB, 2 * GB, &[f("a", 8)], None).is_ok());
        assert!(plan(10 * GB, 2 * GB, &[f("a", 9)], None).is_err());
    }
}
