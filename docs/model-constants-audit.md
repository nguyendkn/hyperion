# Model-dependent constants in the runtime

Question: which model-specific values does Helios use that the manifest does not carry? Anything the runtime reads must be in the signed manifest before v1 freezes. First audit at commit c06873f found nine; this version records the result after spec group 1 (ADR 0010).

## Result
**The sm90 runtime (the only backend with a forward pass) reads every model constant from the manifest and nothing from `config.json`.** `Qwen2Dims::from_manifest` is the single place the dimensions are built; `backend.rs` no longer opens any file beside the package except the weight payloads and the tokenizer asset. Checked by grep (`config.json`, `rope_theta`, `rms_norm_eps`, `tie_word_embeddings`, `eos_token_id`, and the literals 151643, 151645, 152064, 128256, 1e-6, 1e6, 10000.0 in `helios-hal/src`, `helios-core/src`, `helios-cli/src`): what remains is test data, doc comments and the exceptions below.

| Value | Where it lives now |
|---|---|
| hidden, intermediate, rope_theta, rms_norm_eps, tied head | `model.*` in the manifest |
| QKV bias, QK-norm, activation, RoPE style, attention scale | `model.qkv_bias`, `qk_norm`, `activation`, `rope_style`, `attention_scale`; `from_manifest` refuses any combination the kernels do not implement with a typed `UnsupportedModel` |
| end-of-sequence tokens | `generation.eos_token_ids` (list; the backend ends on any of them) |
| chat template and its bos/eos texts | `chat_template` in the manifest (signed); the CLI no longer reads `tokenizer_config.json` |
| sampling recommendations, provenance | `generation.sampling_defaults`, `qualified_checkpoint` |

## Variants the kernels fix (declared in the manifest, only the supported value exists)
`activation` has one variant (`silu`), `rope_style` one (`rotate_half`, whole head rotated), the attention scale defaults to `1/sqrt(head_dim)` and an override that differs is refused, `qkv_bias=false` and `qk_norm=true` are refused. A new variant needs a kernel and a spec change together. RoPE scaling, sliding window and partial rotary are refused by Ignition at export.

## Exceptions (not model constants, or not a runtime path yet)
| What | Why it stays |
|---|---|
| `MAX_CONTEXT 8192`, `MAX_ATTENTION_SEQ 8192` (`session.rs`, `ops.rs`) | The untiled prefill limit of this implementation, not a property of a model; lifted by tiled prefill (phase 12 step 7) |
| `DEFAULT_MAX_LENGTH 4096` (`tokenizer.rs`) | Fallback when the CLI passes no limit; the CLI always passes `--max-seq-len` |
| sm120 skeleton: `HEAD_DIM 128`, `NUM_HEADS 32`, `VOCAB_SIZE 128_256` (`blackwell_sm120/backend.rs`, `kernels.rs`) | The Blackwell backend has no forward pass and never ran on hardware; these sit in its placeholder KV sizing and logits buffer. They must come from the manifest before it serves anything. |
