# Sampler v2: exact top-k / top-p (no candidate cap)

Source of truth: the module docs of `helios/crates/helios-hal/src/sampler_reference.rs` (the host reference the device must equal token for token).
Device: `sample_rows_bf16` and `sample_wide` in `helios/crates/helios-hal/kernels/elementwise.cu`.

## What changed from v1

v1 capped the candidates at 1024: a `top_k` above 1024 and a nucleus wider than 1024 tokens were silently cut. v2 has no cap. The result is the
mathematical top-k then top-p draw over the whole vocabulary, in one fixed order of f32 operations.

## Two device paths, chosen per row inside the same graph

| Path | Used when | Cost (7B, one row) |
|---|---|---|
| Whole vocabulary (mode A) | `top_k` 0 or >= vocab, and `top_p >= 1` | one pass, ~+0.4-0.6% of a step |
| **Bounded** (sort of the 1024 best in shared memory) | `top_k` in 1..1024; or `top_k` 0 / >= vocab with `top_p < 1` **and** the nucleus lies inside the 1024 best | ~110-130 us (budget 150 us, owner 2026-11-04) |
| **Wide** (full per-row radix sort, segmented prefix sums) | `top_k` in 1025..vocab-1; or the nucleus is wider than the 1024 best | reported, no budget (`[NEEDS-GPU]`, first number from the pod) |

The branch is on device data (`shared_wide` in the kernel), so no graph is recaptured; the wide path adds only its own work to the rows that need it.

## Why the bounded path is exact

The reference defines the prefix sums `H_j` over the sorted tokens in segments of 1024: sequential inside a segment, and a sequential sum of the
earlier segment totals as base. For the first 1024 sorted tokens that is the plain sequential sum, which the bounded path computes too. The nucleus
threshold is `top_p * H_{kk-1}` for a set of at most 1024 tokens and `top_p * T` otherwise, where `T` is the chunk-interleaved total of the weights of
the set, computable without sorting. The nucleus is the first `j` with `H_j >= threshold`; if `H_1023 >= threshold` that `j` is below 1024 and equals
what the full computation finds, and the draw `target = u * H_{keep-1}` sees the same prefix. Otherwise the bounded path says "not provable" and the row
takes the wide path. `sampler_reference::sample_bounded_shortcut` is the bounded decision on the host; its test requires it to equal `sample`
whenever it answers (it answered and declined in thousands of cases including ties at the cut, flat rows and every `top_k` boundary).

## Tests

* L0 `[RUN]`: `cargo test -p helios-hal --lib sampler_reference` (reference, wide nucleus, `top_k` {1, 50, 1024, 1025, vocab}, `top_p` {tiny .. 1.0}, ties, temperature
  extremes, shortcut agreement). Penalties: `the_repetition_penalty_moves_a_seen_token_down_whatever_its_sign`.
* H100 `[NEEDS-GPU]`: `sampling_hw.rs` modes 0-9 (flat nucleus at temperature 60 and 200, `top_k` 1025 / 2000 / 1024 / larger than the vocabulary) in every bucket against the host
  reference, plus chi-square and the cost table.
