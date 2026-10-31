# Gap analysis: vLLM / SGLang surface vs Hyperion today (2026-10-31)

Engine facts read today: CLI = `--model <pkg> --port --bind --max-seq-len`; HTTP = `POST /v1/chat/completions` (typed: model, messages[content: String only], max_tokens, temperature, top_p, stream) + `GET /health`; `SamplingParams` = temperature, top_p, max_tokens; hal `SamplingConfig` = temperature, top_p, seed; scheduler serves via StepPlan when backend has a KV pool; 429 on full queue; prefix cache exists (core); sm90: forward + oracle done on H100, paged-KV/`submit_step`/TP not implemented (CLAUDE.md, git log f0731e5); signing/`helios-pkg` not done (old 06); tensor_format written = bf16 only; no metrics, no auth, no `/v1/models`. Host-side facts found in code: EOS stop is a host check (`step_scheduler.rs:249`, `cfg.eos_token_ids`), request seed is today `request_id` (`scheduler.rs:225`, `scheduler_plans.rs:46`), so ignore_eos/seed/stop need no trait change (ph03). Upstream facts are `[U]` until phase 01 verifies them.

Status key: S supported by this plan, W accept-and-warn, R reject-with-error, P planned (phase), - not in scope.

| Family (examples) | Engine today | Status after plan | Depends on |
|---|---|---|---|
| Positional model / `--model` / `--model-path` / `--tokenizer` / `--revision` / `--download-dir` / `--hf-token` | `--model` = bundle path only | S (ph02, 05, 07); `--tokenizer` differing from model: R | ph05-07; old 06 for signing |
| `--host --port --api-key --served-model-name --uvicorn-log-level --log-level --max-log-len` | bind+port, no auth | S (ph02+03); log flags S/W | ph03 |
| `--max-model-len` / `--context-length` (k/M suffix, -1, auto) | `--max-seq-len`, checked vs manifest | S (ph02); `auto` = manifest max | - |
| `--gpu-memory-utilization` / `--mem-fraction-static` / `--kv-cache-memory-bytes` / `--num-gpu-blocks-override` | KV pool sizing in hal (pool sizing host-side, f0731e5) | all P(old 12): parsed, rejected at the 02 report stage (exit 2, before any download) with a typed error naming the phase until the sm90 pool lands (`--kv-cache-memory-bytes`, `--num-gpu-blocks-override`: R) | CD10, ADR 0002; sm90 KV (old 12/13) |
| `--tensor-parallel-size -tp --tp --tp-size` | TP design-only (ADR 0004-0009), stub multi-rank | 1: S. >1: P(old 13 M1), R with typed error until then | old 13 |
| `--pipeline-parallel-size`, `--data-parallel-size`, `--nnodes`, `--node-rank`, `--enable-expert-parallel`, `--distributed-executor-backend` | none | R (P: old 13 PP/DP, Dynamo router) ; `--distributed-executor-backend mp/uni` W | old 13 |
| `--dtype` | manifest tensor_format bf16 | auto/bfloat16 S; float16/float32 R | no fp16 path |
| `--quantization -q` / `--kv-cache-dtype` | no quantizer; sm90 can do fp8 kernels (decode design) | R with message (P: later quantizer phase, old 09 backlog) | quantizer (not planned) |
| `--max-num-seqs` / `--max-running-requests`, `--max-num-batched-tokens`, `--chunked-prefill-size`, `--enable-chunked-prefill`, `--scheduling-policy`, `--schedule-policy` | step scheduler has capacities, queue limit | max-num-seqs S (maps to slots/queue, ph02+07 check scheduler knobs); others W if equals engine default else R | old 07 scheduler |
| `--enable-prefix-caching` / `--no-...` / `--disable-radix-cache` / `--prefix-caching-hash-algo` | prefix cache chained hashes (core), always on | enable: W (value equals current behaviour); disable: R (no core toggle, core untouched); hash algo: R | core toggle (not planned) |
| `--enforce-eager`, `--disable-cuda-graph` | graphs captured at load | W (cannot change output; graphs stay on; research-01: must not error) | - |
| `--trust-remote-code` | never honored | W for both values (true: warned; a model that needs remote code fails typed at build, Ignition exit 12) | - |
| `--seed` / `--random-seed` | per-request seed in hal (today = request_id) | S default seed; request `seed` S ph03 (host-side `SamplingInit.seed`) | ph03 |
| `--generation-config`, `--override-generation-config` | manifest `generation` (eos, sampling defaults) | S for auto (uses manifest defaults); vllm default behaviour parity checked ph01 | ADR 0010 |
| `--chat-template`, `--chat-template-content-format` | template in signed manifest | file/inline override: S (ph05 cache key, ph06 `--chat-template-file`, ph07); content-format S ph03 | ph03/07 |
| `--load-format`, `--model-impl`, `--hf-overrides`, `--hf-config-path`, `--tokenizer-mode`, `--skip-tokenizer-init`, `--attention-backend` | safetensors only | auto S; others W/R per value | - |
| LoRA, multimodal, spec decode, reasoning/tool parsers, structured outputs, `--enable-auto-tool-choice`, kv offload/lmcache | none | R (P: tools/guided = ph08 item 5, separate owner gate; kv offload: old 11) | old 09/11 |
| SSL, CORS, `--root-path`, `--uds`, `--middleware`, `--allowed-*` | none | ssl/cors: P (not in plan) -> R; docs flags W | CD |
| Obs flags (`--otlp-traces-endpoint`, `--enable-metrics`, `--disable-log-stats`) | none | enable-metrics S/W; otlp R | ph04 |
| Env: `VLLM_API_KEY`, `VLLM_LOGGING_LEVEL`, `HF_*`, `CUDA_VISIBLE_DEVICES`, `SGLANG_*` | none read | S for first set; `SGLANG_*`/other `VLLM_*`: W when set (logged), never silent | ph02 |
| `/v1/chat/completions` full fields | 6 fields | S ph03 (content parts, stop, n, stream_options, max_completion_tokens, ignore_eos, seed, unknown extras tolerated+logged); top_k/min_p/penalties/logit_bias/min_tokens/stop_token_ids/logprobs: P(08), 400 naming phase 08 until CD7 approved | ph08, CD7, CD8 |
| `/v1/completions` (str/list/token-id prompts, echo, logprobs) | none | S basic ph03; echo+prompt logprobs P ph08 (lm-eval loglikelihood) | ph08 |
| `/v1/models`, `/version` (+unsigned-bundle extras), `/ping`, `/tokenize`, `/detokenize`, error JSON (per style/route), Bearer auth | none | S ph03; during Building the pre-engine router (ph07) answers /health, /ping, /v1/* with 503 | tokenizer in core |
| `/metrics` vllm:* (+gpu_cache_usage_perc alias), sglang:* | none | S ph04 | scheduler hooks |
| response_format json_schema / tools / guided decoding | none | P(08 item 5) (needs logits masking in backend) | CD7 |
| `/v1/embeddings`, `/v1/responses`, `/v1/messages`, rerank, score, audio | none | out of scope (404 JSON error in vLLM shape) | - |
| SGLang native `/generate /get_model_info /server_info /health_generate /ready /flush_cache` | none | S ph09 (optional); ph07 Building router also answers /health_generate, /ready when `enable_sglang_api` | ph03/04 |
| `--config` YAML | none | S ph02 | YAML dep CD12 |
