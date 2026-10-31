# SGLang server surface for a drop-in Rust engine (researcher-02)

Date 2026-10-31. Ideas only, no code copied. Tags: [V] = verified this session from docs.sglang.io (docs moved from docs.sglang.ai, 301); [U] = unverified, from memory of upstream, check against `--help` of the target release. Research budget was small: server_arguments page read only to char 100000 of 148747; sampling-params, OpenAI-API and bench pages NOT read.

## 1. launch_server flags

Entry: `python -m sglang.launch_server --model-path X`. [V] `--help` is the full list. A newer `sglang serve` CLI entry exists [U: not checked; treat as alias of launch_server with same flags].
Config file: `--config <yaml>`; CLI flags override file values [V]. Keys presumably = flag names without `--` [U].
Unknown flags: argparse errors out (exit 2, usage text) [U, standard argparse; sglang does not use parse_known_args as far as I know].

| SGLang flag (aliases) | Default | vLLM equivalent |
|---|---|---|
| `--model-path` (`--model`) [V] | None (required) | `--model` / positional `vllm serve <model>` |
| `--tokenizer-path` [V] | None = model path | `--tokenizer` |
| `--host` [V] | 127.0.0.1 | `--host` (default 0.0.0.0 [U]) |
| `--port` [V] | 30000 | `--port` (8000) |
| `--tensor-parallel-size` (`--tp-size`, `--tp`) [V] | 1 | `--tensor-parallel-size` (`-tp`) |
| `--data-parallel-size` (`--dp-size`) [V] | 1 | `--data-parallel-size` (`-dp`) |
| `--pipeline-parallel-size` (`--pp-size`) [U] | 1 | `--pipeline-parallel-size` |
| `--mem-fraction-static` [V] | auto, fallback 0.88 | `--gpu-memory-utilization` (0.9). Not identical: sglang fraction = weights+KV pool only, activations/graphs outside |
| `--context-length` [V] | from config.json | `--max-model-len` |
| `--max-running-requests` [V] | None | `--max-num-seqs` |
| `--max-total-tokens` [V] | None (auto) | no direct; `--num-gpu-blocks-override` approx |
| `--chunked-prefill-size` [V] | None (auto by GPU), -1 disables | `--max-num-batched-tokens` + `--enable-chunked-prefill` |
| `--max-prefill-tokens` [U] | 16384 | `--max-num-batched-tokens` |
| `--dtype` [V] auto,half,float16,bfloat16,float,float32 | auto | `--dtype` (same choices) |
| `--quantization` [V] awq,fp8,gptq,bitsandbytes... | None | `--quantization` (`-q`) |
| `--kv-cache-dtype` [V] auto,bf16,fp8_e5m2,fp8_e4m3,nvfp4 | auto | `--kv-cache-dtype` (auto,fp8,fp8_e4m3,fp8_e5m2) |
| `--served-model-name` [V] | None (= model path) | `--served-model-name` |
| `--api-key` [V] | None | `--api-key` |
| `--trust-remote-code` [V] | False | `--trust-remote-code` |
| `--attention-backend` [V] triton,fa3,flashinfer,flashmla,trtllm_mla | None (auto) | env `VLLM_ATTENTION_BACKEND` / `--attention-backend` in new versions [U] |
| `--disable-radix-cache` [U; page truncated] | False (radix on) | `--no-enable-prefix-caching` |
| `--schedule-policy` [V] lpm,random,fcfs,dfs-weight,lof,priority,routing-key | fcfs | `--scheduling-policy` (fcfs,priority) |
| `--log-level` [V] | info | env `VLLM_LOGGING_LEVEL`; `--uvicorn-log-level` |
| `--enable-metrics` [V] | False | on by default in vLLM (`--disable-log-stats` off) |
| `--chat-template` [V] name or file path | None | `--chat-template` (path or inline) |
| `--random-seed` [V] | None | `--seed` (0) |
| `--revision` [U] | None | `--revision` |
| `--load-format` [U] | auto | `--load-format` |
| `--download-dir` [U] | None | `--download-dir` |
| `--device` [U] | cuda | `--device` |
| `--cuda-graph-max-bs` / `--cuda-graph-bs` [U] | auto | `--max-cudagraph-capture-size` / compilation config |
| `--disable-cuda-graph` [U] | False | `--enforce-eager` |
| `--enable-torch-compile` [U] | False | `--compilation-config` |
| `--speculative-algorithm`, `--speculative-draft-model-path`, `--speculative-num-steps` [U] | None | `--speculative-config` JSON |
| `--tool-call-parser`, `--reasoning-parser` [U] | None | `--tool-call-parser` + `--enable-auto-tool-choice`, `--reasoning-parser` |
| `--json-model-override-args` [U] | `{}` | `--hf-overrides` |
| `--nnodes`, `--node-rank`, `--dist-init-addr` [U] | 1,0,None | ray / `--nnodes`, `--node-rank`, `--master-addr` |
| `--enable-lora`, `--lora-paths` [U] | off | `--enable-lora`, `--lora-modules` |
| `--grammar-backend` [U] xgrammar/outlines/llguidance | xgrammar | `--structured-outputs-config` / `--guided-decoding-backend` |
| `--allow-auto-truncate` [U] | False | n/a (vLLM rejects too-long with 400) |
| `--skip-server-warmup` [U] | False | none |
| `--watchdog-timeout`, `--dist-timeout` [U] | 300 / none | none |
| `--enable-mfu-metrics` [V] | False | none |
| `--tokenizer-mode`, `--skip-tokenizer-init` [U] | auto / False | `--tokenizer-mode` |

Minimum viable set for drop-in: model-path, tokenizer-path, host, port, tp, dp, mem-fraction-static, context-length, max-running-requests, max-total-tokens, chunked-prefill-size, dtype, quantization, kv-cache-dtype, served-model-name, api-key, trust-remote-code, attention-backend, disable-radix-cache, schedule-policy, log-level, enable-metrics, chat-template, random-seed, config, plus accept-and-ignore (with a logged warning) for CUDA-graph/torch-compile/spec-decode/parser flags. Recommendation (inference): accept-and-warn on known-but-unsupported flags, hard-fail only on truly unknown.

## 2. Environment variables

[V] on the args page: `TORCHINDUCTOR_CACHE_DIR`, `SGLANG_DISABLE_STATIC_WATERFILL`, `SGLANG_SANITIZE_NAN_LOGITS`, `SGLANG_DSA_FUSE_TOPK`, `SGLANG_MOE_NVFP4_DISPATCH`. All model-specific/internal; none needed for drop-in.
[U] commonly seen in launch scripts: `SGLANG_USE_MODELSCOPE`, `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN`, `SGLANG_TORCH_PROFILER_DIR`, `SGLANG_ENABLE_JIT_DEEPGEMM`, `SGLANG_LOGGING_CONFIG_PATH`, `SGLANG_HEALTH_CHECK_TIMEOUT`... Policy suggestion: read none by default; honor `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` only if a target needs it.
HF: `HF_HOME`, `HF_TOKEN`, `HF_ENDPOINT`, `HF_HUB_OFFLINE`, `HUGGING_FACE_HUB_TOKEN`, `TRANSFORMERS_OFFLINE` [U, standard HF; sglang resolves repo ids through huggingface_hub]. Also `CUDA_VISIBLE_DEVICES`, `NCCL_*`.

## 3. HTTP surface

Verified native routes [V]:
- `POST /generate`: body `{"text": "..."}` minimal. Full shape [U]: `text` or `input_ids`, `sampling_params` object, `stream` bool, `return_logprob`, `logprob_start_len`, `top_logprobs_num`, `lora_path`, `rid`, batch via list `text`.
  - `sampling_params` names [U]: `max_new_tokens` (default 128), `min_new_tokens`, `temperature` (1.0), `top_p`, `top_k` (-1), `min_p`, `stop` (str/list), `stop_token_ids`, `repetition_penalty`, `frequency_penalty`, `presence_penalty`, `ignore_eos`, `n`, `regex`, `json_schema`, `ebnf`, `skip_special_tokens`, `no_stop_trim`, `sampling_seed`.
  - Response [U]: `{"text": "...", "meta_info": {"id", "finish_reason": {"type": "length|stop", "length"|"matched"}, "prompt_tokens", "completion_tokens", "cached_tokens", "e2e_latency", ...}}`; list if batch.
  - Streaming [U]: SSE `data: {json}\n\n`, each chunk carries cumulative `text` (not delta) plus `meta_info`; terminator `data: [DONE]`. Verify cumulative vs delta; this is a classic compat trap.
- `GET /get_model_info` [V]: `model_path`, `is_generation`, `tokenizer_path`, `preferred_sampling_params`, `weight_version`, `has_image_understanding`, `has_audio_understanding`, `model_type`, `architectures`, optional `embedding`. (Older: also `/model_info`, [U]).
- `GET /server_info` [V] (page lists this name; the older `/get_server_info` likely still routed [U]): CLI args, token limits, memory pool sizes. Replaces deprecated `get_server_args`, `get_memory_pool_size`, `get_max_total_num_tokens`. Fields unlisted; clients (bench) read `max_total_num_tokens`, `internal_states` [U].
- `GET /health` [V], `GET /health_generate` [V, generates one token], `GET /ready` [V: 200 accepting, 503 starting/paused/unhealthy/shutting down], `GET /ping` [V SageMaker].
- `POST /flush_cache?timeout=<float>` [V]: 0 = fail fast if not idle (non-200 then [U]).
- `POST /tokenize`, `/detokenize` [V] (vLLM-style bodies, `tokens`/`count`/`max_model_len`/`text`), `/encode`, `/classify`, `/v1/rerank`, `/v1/score`, `/v1/decisions`, `/update_weights_from_disk`, expert-distribution record routes [V].
- OpenAI-compatible [U, not read this session]: `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `GET /v1/models`, plus `/v1/responses`, `/v1/batches`, `/v1/files` in recent versions. `/v1/models` lists `served-model-name` (or model path), with `max_model_len` [U].
- `GET /metrics` [V], needs `--enable-metrics`; same port (30000 default).
  Metric names [V]: `sglang:prompt_tokens_total`, `generation_tokens_total` (counters); gauges `token_usage` (0..1), `cache_hit_rate`, `num_running_reqs`, `num_used_tokens`, `gen_throughput`, `num_queue_reqs`, `spec_num_steps`, `spec_num_draft_tokens`; histograms `time_to_first_token_seconds`, `e2e_request_latency_seconds`, `func_latency_seconds`, `request_time_per_output_token_seconds`, `inter_token_latency_seconds` (type unclear). MFU: `estimated_flops_per_gpu_total`, `estimated_read_bytes_per_gpu_total`, `estimated_write_bytes_per_gpu_total` (need `--enable-mfu-metrics`). Labels (`model_name`, `engine_type`, `tp_rank`, `pp_rank`) [U]. Names change between releases (SigNoz cites v0.5.19).

Differences vs vLLM [U unless noted]:
- Prefix `sglang:` vs `vllm:`; vLLM names `vllm:num_requests_running`, `num_requests_waiting`, `gpu_cache_usage_perc` (now `kv_cache_usage_perc`), `time_to_first_token_seconds`, `e2e_request_latency_seconds`, `prompt_tokens_total`, `generation_tokens_total`. Dashboards need a mapping; could emit both prefixes.
- vLLM health: `GET /health` 200 empty body; sglang `/health` similar. vLLM has no `/health_generate`, `/generate` (native `/generate` exists only in some entrypoints with different body: `prompt`, `sampling_params`), `/get_model_info`.
- vLLM extra request fields on chat/completions: `top_k`, `min_p`, `repetition_penalty`, `guided_json`/`structured_outputs`, `ignore_eos`, `use_beam_search`, `add_generation_prompt`, `chat_template_kwargs`, `return_token_ids`. SGLang: `regex`, `json_schema` via `response_format`/`extra_body`, `top_k`, `min_p`, `separate_reasoning`, `chat_template_kwargs`, `stream_options`, `rid`, `lora_path`. Both ignore unknown extras mostly (sglang is lenient [U]).
- Error shape: both OpenAI-style `{"object":"error","message","type","param","code"}` in vLLM; sglang returns `{"object":"error","message","type","param","code"}` for OpenAI routes [U] but native routes may give plain `{"error": ...}` or FastAPI `{"detail": ...}` [U]. Too-long prompt: sglang 400 (unless `--allow-auto-truncate`) [U].

## 4. How benchmarks/tests drive it

All [U] (bench/lm-eval pages not fetched):
- `python -m sglang.bench_serving --backend sglang|sglang-oai|sglang-oai-chat|vllm|... --base-url / --host --port --dataset-name random|sharegpt|... --num-prompts --random-input-len --random-output-len --request-rate --max-concurrency --model --tokenizer --output-file`. Backend `sglang` hits native `/generate` with `sampling_params{max_new_tokens,temperature,top_p...}`, `stream:true`, parses cumulative `text`, uses `meta_info`; `sglang-oai` -> `/v1/completions`, `sglang-oai-chat` -> `/v1/chat/completions`. It also calls `GET /get_model_info` and sometimes `/server_info` (flush cache via `POST /flush_cache` before runs). Safest bench target for Hyperion: `--backend sglang-oai-chat` (OpenAI only), then native.
- lm-eval: `--model local-completions|local-chat-completions --model_args base_url=http://host:30000/v1/completions,model=<name>,tokenized_requests=False` (needs `logprobs`/`echo` for loglikelihood tasks: `/v1/completions` with `echo`, `logprobs`, `max_tokens=0` [U]). Also sglang has its own `sglang.test.few_shot_gsm8k`, `run_eval` using openai client.
- openai client: `OpenAI(base_url="http://host:30000/v1", api_key="EMPTY")`.
- Readiness: sglang's own test helpers (`popen_launch_server`) poll `GET /health_generate` (or `/v1/models` with Authorization bearer when api_key set) until 200 [U, my recollection; verify in `sglang/test/test_utils.py`]. Docker/K8s examples use `/health` for liveness, `/health_generate` or `/ready` for readiness. Launch scripts/notebooks use `wait_for_server(url)` polling `/v1/models` [U]. Server also does a startup warmup (a 1-token generate) before announcing "The server is fired up and ready to roll!" [U log line; scripts grep it]. Drop-in should: return 503 on `/health_generate`/`/ready`/`/v1/models` until loaded, print that exact line [U], honor Bearer `--api-key` on all but health routes [U].

## Priorities (inference)
1. Flags table above + accept-and-warn. 2. `/v1/*` chat/completions/models with sglang extras. 3. `/health`, `/health_generate`, `/ready`, `/get_model_info`, `/server_info`, `/flush_cache`. 4. `/generate` with cumulative-text streaming. 5. `/metrics` under `sglang:` names (cheap: ~8 metrics cover dashboards).

## Unresolved questions
- Is `sglang serve` a real, flag-compatible entry in the target version?
- `/generate` streaming: cumulative or delta text in current release? Exact `meta_info` and `finish_reason` shape?
- Exact `/server_info` fields consumed by bench_serving (`max_total_num_tokens`, `internal_states`?).
- Which endpoint do current `popen_launch_server` / `wait_for_server` poll (`/health_generate` vs `/v1/models`)?
- Does `--disable-radix-cache` still exist under that name (page truncated)?
- Does `--config` YAML use flag names with dashes or underscores?
- Native error JSON shape vs OpenAI shape on non-OpenAI routes.
- Which sglang version to pin as target (metric/flag names drift per release).

Sources: docs.sglang.io/advanced_features/server_arguments.html, docs.sglang.io/basic_usage/native_api.html, docs.sglang.io/references/production_metrics.html
