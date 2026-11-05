//! Lint (owner decision D4, 2026-11-05): no synchronous pageable host-device copy on the step path. `CudaDevice::htod` / `dtoh` (`cuMemcpyHtoD` /
//! `cuMemcpyDtoH`) are not ordered with the step stream: the step path uses `htod_staged` / `dtoh_staged` / `memset_staged` (pinned ring, `cuda/staging.rs`).
//! The only allowed synchronous copies in these modules are load-time uploads, marked `// allowed: load-time` on the line.

use std::path::PathBuf;

const STEP_PATH_MODULES: &[&str] = &["session.rs", "plan_exec.rs", "forward_device.rs", "prefill_plan.rs", "decode_buckets.rs"];
const FORBIDDEN: &[&str] = &[".htod(", ".dtoh(", "cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2"];

#[test]
fn the_step_path_modules_contain_no_synchronous_pageable_copy_except_marked_load_time_uploads() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/backends/hopper_sm90");
    let mut offenders = Vec::new();
    let mut scanned = 0;
    for module in STEP_PATH_MODULES {
        let Ok(text) = std::fs::read_to_string(dir.join(module)) else { continue };
        scanned += 1;
        for (number, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            if FORBIDDEN.iter().any(|f| code.contains(f)) && !line.contains("allowed: load-time") {
                offenders.push(format!("{module}:{}: {}", number + 1, line.trim()));
            }
        }
    }
    assert!(scanned >= 3, "the lint found too few step-path modules ({scanned}): the list is stale");
    assert!(offenders.is_empty(), "synchronous pageable copies on the step path (use htod_staged/dtoh_staged):\n{}", offenders.join("\n"));
}

#[test]
fn the_lint_sees_a_violation() {
    // The scanner itself: a line with a forbidden call and no marker is an offender, a marked one is not.
    let bad = "        dev.htod(x, &y)?;";
    let ok = "        dev.htod(x, &y)?; // allowed: load-time";
    let flagged = |line: &str| FORBIDDEN.iter().any(|f| line.split("//").next().unwrap().contains(f)) && !line.contains("allowed: load-time");
    assert!(flagged(bad) && !flagged(ok));
}
