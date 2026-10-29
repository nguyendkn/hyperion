# Researcher-01: upstream architecture and model coverage (ideas only)

Source: NVIDIA local-inference clone (read-only). Labels: [V] read in source, [I] inferred. Paths relative to upstream repo root. No code copied.

## 1) How upstream gets broad model coverage

- Layout: Python `ignition/` (build-time) + Rust `helios/` runtime (+ `flare/` CLI, `heliopause/` native exporter, `qualify/` scripts). Same two-stage split as Hyperion. [V] AGENTS.md, docs/core-design.md
- Recipe vs component split [V] docs/model-development.md, docs/core-design.md:
  - **Recipe** (`ignition/recipes/<model>/`) owns model composition, checkpoint-to-tensor mapping, state layouts, memory policy, graph construction, frontend assets (chat template), sampling defaults.
  - **Component** (`ignition/components/`, ~100 files) = shared execution mechanism with its own contract: linear/blockscaled projections, MoE, MLA/sparse/paged/window attention, GDN/KDA/mamba recurrence, vocabulary shard, collectives, sampling, token state, state transfer, speculation. Rule: model/recipe identity must not leak into shared policy.
- Coverage is NOT "little per-model code". Each recipe is a directory of 15-30 files (glm53: attention, MLA, MoE, KDA, MTP, vision, weights, state, pool, contract...). Breadth comes from reusing components + Transformers as semantic source, not from a config-only model description. 11 recipes: deepseek4/41, gemma4, glm53, kimi3, mimo_v2, minimax_m3, nemotron_h, qwen38 (+dense), step5. [V listing]; [I] cost per new model is still large.
- Transformers is canonical for structure: reuse its forward where tensor layouts fit; replace the smallest coherent module (whole attention or MoE) only when needed for dataflow control; instance-local replacement, no global monkey patches. [V]
- Declaration split: `ignition/recipes/declaration.py` holds frozen dataclasses (Checkpoint, SamplingDefaults, Speculation, per-rank costs) and imports no model code or torch, so planning/listing/memory-fit works without PyTorch. [V]
- A `Checkpoint` record pins alias, repo, revision, architecture, stored/device/host bytes, draft bytes, per-rank cost, decode weight bytes. Recipes declare supported GPU counts + checkpoint revisions; only qualified checkpoints are served. [V]
- Per-recipe `contract.py` = "prepared data" handed to the shared native runtime contract (`helios/model-contract.md`); Helios has no per-model branches, only consumes the contract (typed storage, graphs, frontend metadata). [V core-design.md table of recipes]
- Quantization read from checkpoint metadata; formats (NVFP4, MXFP4, FP8 block) handled in loading/prep, not in recipe `if` branches. [V]
- Prompt rendering is part of model integration: annotated Jinja (mark directives) or native Rust encoder, recording per-message endpoints for prefix-cache cuts; verified vs pinned authoritative template + checkpoint tokenizer. [V]
- Sampling defaults declared per recipe from the model card (manually verified, cited), never from generation_config; Helios holds no per-model sampling policy. [V]

## 2) Model-addition contract and test/qualification method

Contract (synthesised from model-development.md, preparation-development.md, core-design.md): [V]
1. Read model card; declare sampling defaults + ranges with citation.
2. Pin checkpoint revision; declare in recipe (bytes, ranks, GPU counts) without importing model code.
3. Account for every checkpoint tensor via explicit mapping.
4. Reuse components; operators declared to the tuner at construction, bound after tuning; scratch sized after tuning; concurrent branches registered in a scratch arena.
5. Compile whole forward per capture capacity, capture CUDA graphs, export runtime package (storage, metadata, frontend assets).
6. Ship frontend (template/encoder) with tests vs reference renderer; prefix-cache endpoints.
7. Validation must show real-service behaviour; report skipped/unrun hardware paths explicitly (a plausible answer is not evidence).

Testing/qualification: [V]
- `tests/recipes/` is per-model, per-feature (very many files like test_glm53_*, test_deepseek41_*); `_cuda` suffix = needs GPU, others run on CPU (declarations, catalogs, frontends, pools, scratch, hosts). `tests/components/` holds oracles/support helpers.
- `ignition/reference/` holds in-tree reference implementations (qwen38 decoder, quant) used as oracle. [V listing]
- `qualify/` = standalone measurement/analysis scripts, not unit tests: text_oracle (compare full-vocab logits on identical teacher-forced token contexts; versioned artifact with prompt tokens, teacher tokens, logits, model id, revision), layer_comparison (per-layer residual outputs vs reference, measurement only, no tolerance), gpu_replay / compiled_replay, graph_nodes, kernel_timeline, performance, speculative, host_cache, multihost_cache, provenance. [V]
- Method = teacher-forced logit + per-layer residual comparison against canonical HF forward; tolerances derived per component (atomic reductions looser). [V partly]

## 3) Ideas Hyperion should adopt

| # | Idea | Tag | Reason | Module |
|---|------|-----|--------|--------|
| 1 | Declaration/execution split: list models, check memory fit and capability without torch | adopt now | Cheap; Ignition cli can validate fit pre-load; supports "refuse, never panic" | ignition/architectures |
| 2 | Qualified-checkpoint record (repo, revision, bytes, target_arch, formats) per architecture | adopt now | Provenance; manifest can embed repo+revision; refuse unqualified combos | ignition + specification |
| 3 | Explicit "every tensor accounted" weight mapping (unmapped and missing both fail) | adopt now | Catches silent mis-loads in new archs | ignition/architectures |
| 4 | Component layer separate from architectures (shared attention/MoE/norm/rope blocks; arch = composition) | adopt later | Needed at 3rd+ arch (deepseek MLA, MoE); premature with 2 archs (YAGNI). Keep registry-by-import | ignition |
| 5 | Teacher-forced logit-trace artifact (versioned: prompt+teacher tokens+logits) as cross-implementation oracle HF -> Ignition -> Helios | adopt now | Portable way to test Rust runtime vs HF without Python at runtime; fits `specification/fixtures` | specification, helios-core tests |
| 6 | Per-layer residual comparison to localise divergence | adopt later | Needs working backend kernels first | helios-hal |
| 7 | Split tests: CPU-runnable per-arch vs hardware-gated (`_cuda`-style) | adopt now | Our host has no GPU; matches cfg-gated stance | ignition/tests, helios-hal |
| 8 | Quant format read from checkpoint metadata, handled in loader not arch branches; recipe declares supported_formats per target | adopt now | Already started; extend to read quant declarations from config | ignition/hardware recipes |
| 9 | Manifest carries frontend needs: chat template, EOS/special tokens, author-verified sampling defaults + clamp ranges | adopt now | Helios must not interpret HF configs or hold per-model policy | specification, helios-core |
| 10 | Prompt-boundary endpoints (message-end marks) for RadixCache cuts | adopt later | Valuable once chat serving + prefix cache work; needs template engine in Rust | helios-core tokenizer/cache |
| 11 | Model-agnostic runtime contract; grow ModelConfig via tagged variants (MLA, window, recurrent state), never model names | adopt now | Same as our rule 1; keeps runtime free of per-model branches | specification, helios-core |
| 12 | Whole-forward graph capture per capacity bucket, exported in package | adopt later | Ignition job after kernels exist; reserve schema section | specification |
| 13 | Scratch arena with parallel-live registration | skip (for now) | Optimization detail; revisit with overlap work | helios-hal |
| 14 | Speculation/MTP/draft, Engram, mHC, vision per recipe | skip | Out of current scope; ask before adding | - |
| 15 | HF Transformers forward as canonical semantics (build-time only) | adopt now | Matches our ecosystem rationale | ignition |

## 4) Where Hyperion can be better

- Multi-hardware by design: upstream targets NVIDIA SM103/120/121 only and AGENTS.md rejects non-NVIDIA/interchangeable-backend scope [V]. Hyperion's ComputeBackend trait + exact `target_arch` triage is a real differentiator (incl. Hopper SM90 FP8/BF16).
- Declarative model description: upstream recipes are imperative Python (15-30 files each) [V/I]. Hyperion could make ModelConfig a typed architecture descriptor (layer pattern, attention kind, MoE spec, recurrent spec) so a generic runtime decoder covers a family from descriptor + weight map, lowering marginal cost per model. [I]
- Schema-first contract: Rust-owned JSON schema + Python round-trip is stricter than upstream's Python-side contract. [I]
- Signed bundle + explicit refusal on arch mismatch (upstream has docs/signing.md; not read).
- Tests runnable on a no-GPU host via stub backend + CPU oracle; upstream's kernel tests are mostly `_cuda`. [I]
- A CPU reference backend as correctness oracle: upstream rejects CPU execution as scope [V]; for Hyperion it is a scope decision for the user.

## 5) Unresolved questions

- Real recipe size in LOC and how much is shared (counted files only).
- Per-component tolerances used in oracle comparisons: not read.
- How `qualify/` artifacts gate CI (if at all): not read.
- kimi3/step5/minimax recipe sizes vs glm53: only glm53/qwen38/nemotron_h listed.
- Descriptor-driven generic decoder (sec. 4) vs per-arch Python like upstream: needs user decision (direction).
- CPU reference backend in scope? User decision.
- NOTICE/LICENSE not read; ideas-only rule applies.
