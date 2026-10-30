# Model-dependent constants in the runtime (input to phase 03)

Question: which model-specific values does Helios use that the manifest's `ModelConfig` does not carry? Anything the runtime reads must be in the signed manifest before v1 freezes (adding later means v2). Audit of `helios-hal` at commit c06873f plus the oracle commits; read by grep and code reading, not exhaustive proof.

## Read at run time, but from `config.json` next to the package, not from the manifest
`Qwen2Dims::from_config_json` (`hopper_sm90/forward.rs`) takes these from an unsigned file; the manifest is only cross-checked for the ones it has (layers, heads, KV heads, head_dim, vocab).

| Value | Where used | In manifest today |
|---|---|---|
| `hidden_size` | every GEMM shape, session buffers | no |
| `intermediate_size` | MLP shapes | no |
| `rope_theta` | `rope_tables` (host-built cos/sin) | no |
| `rms_norm_eps` | every rmsnorm launch | no |
| `tie_word_embeddings` | output head choice (`head_weight()`) | no |

## Hard-coded in kernels or code (a model that differs needs a field or a new variant)
| Behaviour | Where | Refused or silent if different |
|---|---|---|
| Q, K and V projections have a bias | `session.rs` (`q_proj.bias` etc. required), `forward.rs` | load fails on a Llama-style model with no bias (loud) |
| No QK-norm | attention path | a Qwen3 checkpoint would load its tensors as unknown and fail the strict mapper (loud), but the runtime has no way to apply them |
| MLP activation SiLU, gate/up/down | `swiglu_bf16` | silent for a model with another activation (e.g. GELU) |
| RoPE layout `rotate_half`, full head_dim rotated, theta only | `elementwise.cu` | silent for partial rotary or interleaved layouts |
| RoPE scaling (YaRN, linear, llama3) and sliding window | `Qwen2Dims::from_config_json` | refused (loud) |
| Attention scale `1/sqrt(head_dim)` | attention kernels | silent if a model uses another scale |
| Untiled prefill limit 8192 | `ops.rs`, `session.rs` | capacity error (loud); C1 lifts it |
| sm120 skeleton: `HEAD_DIM 128`, `NUM_HEADS 32`, `VOCAB_SIZE 128_256` | `blackwell_sm120/*` | not a real backend yet |

## Proposed for phase 03 (needs the owner's sign-off on shapes)
Add to the signed manifest: `hidden_size`, `intermediate_size`, `rope_theta` (and a tagged `rope` variant so scaling is a new variant, never inferred), `rms_norm_eps`, `tied_embeddings`, `qkv_bias: bool`, `qk_norm` (none or per-head rmsnorm), `activation` (enum, `silu` first), `sliding_window` (none or tokens). `cache_identity` is computed from payload hashes, layout, `target_arch` and `tensor_format`, never stored; `chat_template` text goes into the signed body. After this, the runtime stops reading `config.json`.
