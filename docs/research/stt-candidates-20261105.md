# STT candidates for Hyperion/Helios (EN must-have + VI), LICENSE_POLICY=commercial

Date 2026-11-05. All URLs accessed 2026-11-05. Tags: **[V]** verified on primary source this session, **[T]** third-party number, **[I]** inferred/from memory, not verified.
Primary-source = HF API/model card, GitHub license API, paper PDF. HF repo shas are the revisions to pin.

## 1. Recommendation

| Role | Winner | Why |
|---|---|---|
| Overall | **Qwen3-ASR-1.7B** (Apache-2.0) | best EN+VI quality among compliant models, decoder is a stock Qwen3-1.7B (qk_norm GQA, already in ModelConfig), safetensors bf16, streaming-capable encoder |
| Encoder + autoregressive decoder family | **Qwen3-ASR-1.7B** (small variant: 0.6B). Classical cross-attn alternative: Cohere Transcribe (best EN, 2B) or Whisper-large-v3-turbo (MIT, simplest) | cross-attn models need a new KV class (static encoder K/V), Qwen3-ASR does not |
| Transducer/CTC family | **Nemotron-3.5-ASR-Streaming-0.6B** (OpenMDW-1.1, **needs owner OK: not on the licence list**) | only transducer with EN+VI and cache-aware streaming. Strict-policy fallback: no compliant transducer has VI; use Parakeet-TDT-0.6B-v2 (EN, CC-BY-4.0) and a separate VI model |

Suggested path: implement Qwen3-ASR (encoder = new, decoder = existing path), test sets in section 6. Transducer family only if streaming latency (<=320 ms) becomes a hard requirement.

## 2. Hard-constraint results (license of weights / code / text components)

| Model | Weights license [V] | Ref code | Tokenizer/text | Pass? |
|---|---|---|---|---|
| Qwen3-ASR-1.7B / 0.6B | Apache-2.0 (HF API) | QwenLM/Qwen3-ASR Apache-2.0 (GitHub API); transformers Apache-2.0 | Qwen BPE vocab.json+merges.txt (Apache, same repo) | PASS |
| Whisper-large-v3 | **Apache-2.0** (HF API; not MIT) | openai/whisper MIT (GitHub API) | GPT-2-style BPE json in repo | PASS |
| Whisper-large-v3-turbo | MIT (HF API) | MIT | same | PASS |
| Whisper small/medium/base | Apache-2.0 per third-party list [T] (handy); MIT per openai repo [I] | MIT | same | PASS |
| Cohere Transcribe 03-2026 | Apache-2.0, auto-gated (HF API) | `custom_code` in repo (Apache per card); transformers >=5.4 native | 16k SentencePiece BPE, byte fallback | PASS (gate = click-through, record it) |
| Nemotron-3.5-ASR-Streaming-0.6B | OpenMDW-1.1 ("other") | NeMo Apache-2.0 (GitHub API) | in repo | CONDITIONAL: permissive (use/modify/commercial, keep notice) but not in policy list; patent+copyright suit terminates rights. Owner decision |
| Parakeet-TDT-0.6B-v3 / v2 | CC-BY-4.0 (HF API) | NeMo Apache-2.0 | SentencePiece 8192 | license PASS, **VI FAIL** (v3 = 25 EU langs, v2 = EN) |
| Canary-1b-v2 | CC-BY-4.0 | NeMo | | license PASS, **VI FAIL** (25 EU langs) |
| Canary-Qwen-2.5B | CC-BY-4.0 | NeMo | Qwen3 | license PASS, **VI FAIL** (EN only) |
| Parakeet-CTC-0.6B-Vietnamese | NVIDIA Open Model License, gated (HF API) | NeMo | | **FAIL license** (not in list; also .nemo only, no safetensors) |
| PhoWhisper (large/medium) | BSD-3-Clause (HF API) | transformers | Whisper BPE | license PASS; **EN FAIL** [I] (vi-only fine-tune, no EN numbers), pickle `.bin` only |
| coflar-ai/viet-asr (Zipformer2, 86M) | Apache-2.0 (card) | icefall/sherpa Apache [I] | 8192 BPE | license PASS; **EN FAIL** (vi only), no FLEURS/CV numbers |
| Granite-speech-3.3-8b | Apache-2.0 | | | **VI FAIL** (en fr de es pt) |
| Voxtral-Mini-3B-2507 | Apache-2.0 | | | **VI FAIL** (en fr de es it pt nl hi tags) |
| Omnilingual-ASR (CTC/LLM 300M-7B) | Apache-2.0 code+models, dataset CC-BY-4.0 (GitHub README) | fairseq2 | | license PASS, vi_Latn supported (lang_ids.py) [V]; **weak fit**: `.pt` only (pickle), <40 s input, no VI number verified |
| Moonshine-base-vi (61.5M) | HF "other", no text/link on card; third party says MIT [T] | | | **FAIL: license unverifiable** |
| Fun-ASR-MLT-Nano | FunASR Model Open Source License v1.1 [T] | | | FAIL (not on list) |
| MMS, SeamlessM4T | CC-BY-NC [I, not re-checked] | | | FAIL (NC) |

## 3. Architecture and op inventory

"New op" = not in a standard decoder-only transformer path (RMSNorm, RoPE, GQA/MHA causal attention, SwiGLU, paged KV, sampler).

| Model | Params / bf16 | Encoder | Decoder | Front-end | Tokenizer |
|---|---|---|---|---|---|
| Qwen3-ASR-1.7B [V config] | 2.349B / 4.70 GB | AuT: 24 L, d=1024, 16 heads, FFN 4096 GELU, conv 8x subsample (hidden 480) -> 12.5 Hz, 128-mel, dynamic flash-attn window 1-8 s (paper); 300M | Qwen3-1.7B: 28 L, h=2048, 16 q / 8 kv heads, hd=128, FFN 6144 SiLU, vocab 151936, rope 1e6, tied emb; mrope_interleaved [24,20,20] | Whisper-style: 16 kHz [I], n_fft 400, hop 160, 128 mels, chunk 30 s, dither 0 | Qwen BPE 151k |
| Qwen3-ASR-0.6B | 0.938B / 1.88 GB | AuT 180M, h=896 (paper) | Qwen3-0.6B [I dims] | same | same |
| Whisper-large-v3 | 1.543B / 3.09 GB | 32 L, d=1280, 20 heads, 2 conv1d (k3, s1 + s2, GELU), 1500 pos | 32 L (turbo: 4 L), cross-attn, learned pos, 448 ctx | 16 kHz, n_fft 400, hop 160, 128 mels (30 s pad), log-mel clamp [I] | 51866 BPE |
| Whisper-large-v3-turbo [V config] | 0.809B / 1.62 GB | 32 L, d=1280 | 4 L | same | 51866 |
| Cohere Transcribe [V HF doc/API] | 2.066B / 4.13 GB | Fast-Conformer, >90% of params (layers/dims: gated config, not read) | 8 L, h=1024, 8 heads, FFN 4096, ReLU, bias, cross-attn, vocab 16384, ctx 1024 | 16 kHz, n_fft 512, win 400, hop 160, 128 mels, preemph 0.97, dither 1e-5, chunk 35 s | 16k BPE |
| Parakeet-TDT-0.6B-v3 [V config] | 0.627B / 1.25 GB | FastConformer 24 L, d=1024, 8 heads, FFN 4096 SiLU, conv k9, 8x dw-striding (256 ch), 128 mel | 2-layer predictor h=640, TDT durations 0-4, vocab 8193 (blank 8192) | 16 kHz mono, 128 mel | SP 8192 |
| Nemotron-3.5-ASR-Streaming [V card] | 0.638B / 1.28 GB | cache-aware FastConformer 24 L, 80 ms frames, att_context [56, 0..13], LangID one-hot 128-d concat | RNN-T | via processor | 35 langs |

| Model | Distinct new ops (count) | Notes |
|---|---|---|
| Qwen3-ASR | **6**: log-mel front-end (STFT+mel+log, can stay on CPU/Rust), conv2d stride-2 x3, non-causal windowed MHSA, LayerNorm+GELU-FFN with bias, sinusoidal pos add, audio-embedding splice+projector | decoder unchanged. Needs "input-embeds prefill" entry and mrope check: with 1-D text/audio positions all 3 mrope sections should equal plain RoPE [I, verify against HF modeling] vs manifest's refusal of rope_scaling |
| Whisper | **8**: mel front-end, conv1d x2, sinusoidal pos, learned pos, LayerNorm, bidirectional full MHA (1500), GELU FFN+bias, **cross-attention with static encoder KV (new KV class in pager)** | 30 s fixed window |
| Cohere | **~11** [I from FastConformer]: mel+preemph+dither, conv2d subsample, rel-pos MHSA, depthwise conv1d, GLU, BatchNorm/LN, SiLU, macaron half-FFN, plus cross-attn KV, ReLU decoder | |
| Parakeet / Nemotron | **~11-12**: same FastConformer set + LSTM predictor [I for Nemotron], joint net, TDT/RNNT greedy loop with durations (control flow, not GEMM), cache-aware conv/attn state | no paged KV/sampler reuse |

## 4. Published quality (EN LibriSpeech test-clean, VI FLEURS vi / Common Voice vi)

| Model | EN LS clean / other | VI FLEURS | VI Common Voice | Source |
|---|---|---|---|---|
| Qwen3-ASR-1.7B | 1.63 / 3.38 [V README]; streaming 1.95 / 4.51 | **5.55** [V paper Table A.2b, parsed from PDF, column order inferred]; 6.15 [T handy Q8_0 ggml] | no published number | arXiv 2601.21337; github QwenLM/Qwen3-ASR |
| Qwen3-ASR-0.6B | 2.11 / 4.55 | 8.52 (paper) ; 9.32 [T] | none | same |
| Whisper-large-v3 | 1.51 / 3.97 (as run in Qwen README, [T-ish]) | 8.74 [T handy]; large-v2: 10.3 [V paper Table 13, parsed, order inferred] | large-v2 CV9 vi 18.2 [V paper Table 11, parsed, order inferred]; large-v3 none | arXiv 2212.04356 |
| Whisper-large-v3-turbo | no published number (leaderboard only) | 9.48 [T] | none | |
| Whisper medium / small | | 13.74 / 22.47 [T] | none | |
| Cohere Transcribe | **1.25 / 2.37** [V blog] | 7.39 [T handy Q8_0]; official = per-language chart averaged over FLEURS+CV17+MLS+Wenet, no number in text | none (chart only) | HF blog 2026-03 |
| Nemotron-3.5-ASR-Streaming | no published number | 11.18 (1.12 s chunk) to 13.41 (80 ms), LangID mode [V card; normalisation inflates] | none | card |
| Parakeet-TDT-0.6B-v2 / v3 | 1.69 / 3.19 ; 1.93 / 3.59 [V] | not supported | not supported | cards |
| Parakeet-CTC-0.6B-vi (excluded) | n/a | 6.86 (FLEURS in training set, in-domain) | MCV20 8.58 in-domain; blind avg 9.30 | card |
| coflar viet-asr | none | none (VIVOS 20.92, VLSP 15.97-27.90) | none | card |
| PhoWhisper | none | none | none on card | card |

Caveat: Whisper/Qwen/Cohere WERs use different normalisers; compare only after running one shared normaliser on our own harness.

## 5. Weights, streaming, pins

| Model | Format | Streaming | Pin | No pickle execution? |
|---|---|---|---|---|
| Qwen3-ASR-1.7B | 2 safetensors shards bf16; sha 7278e1e70fe206f11671096ffdd38061171dd6e5 | encoder window 1-8 s; official streaming only via vLLM (no batch, no timestamps); LLM re-decodes | config says transformers 4.57.6 via `qwen-asr` pkg; native HF support noted 2026-06-26 (version not stated). 0.6B sha 5eb144179a02acc5e5ba31e748d22b0cf3e303b0 | yes |
| Whisper-large-v3 | safetensors f16 (+ .bin, flax); sha 06f233fe06e710322aca913c1bc4249a0d71fce1 | no (30 s window; chunked hacks) | transformers (no min stated) | yes (use safetensors) |
| Whisper-large-v3-turbo | safetensors f16; sha 41f01f3fe87f28c78e2fbf8b568835947dd65ed9 | no | same | yes |
| Cohere Transcribe | safetensors bf16; sha d552dadc2922ba6f6c1f994f8db904d52dcaf942 | no (offline; 35 s chunks with energy split; vLLM server) | transformers>=5.4.0; doc example uses `refs/pr/6` revision; repo has custom code (trust_remote_code), prefer native class | yes if native class used |
| Nemotron-3.5 | safetensors f32 + .nemo + gguf; sha ea30d66debe3740a08b573244286791d423d6b3e | **native cache-aware, 80 ms to 1.12 s** | NeMo 26.06; transformers>=5.13.0 | yes (safetensors) |
| Parakeet-TDT v3 | safetensors f32 + .nemo + gguf; sha 541d1f99c6b0c3cd0b11a95167540bb8edefd82b | chunked, 2 s chunks; long-form local attn | NeMo 2.4 | yes |
| Parakeet-TDT v2 | **.nemo only** (tar with pickle ckpt); sha ae9ad07059c7c739ffaf932226a8fe64ae2620b0 | chunked | NeMo 2.2 | NO (needs torch.load in a sandbox, or take v3 safetensors) |
| PhoWhisper-large | **pytorch_model.bin only**; sha b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb | no | transformers | NO |
| Omnilingual | `.pt` (fairseq2) | no, <40 s | fairseq2 | NO |

## 6. Scored matrix (weights: quality 40, reuse 25, streaming 15, new-op count 10, size/speed 10)

Scores are my judgement from the evidence above, not measured. Quality = EN+VI, VI weighted more because EN is near-saturated (all <=2.1 LS-clean).

| Candidate | Quality /40 | Reuse /25 | Stream /15 | Ops /10 | Size /10 | **Total /100** |
|---|---|---|---|---|---|---|
| **Qwen3-ASR-1.7B** | 36 | 22 | 11 | 5 | 6 | **80** |
| Qwen3-ASR-0.6B | 30 | 22 | 11 | 5 | 9 | 77 |
| Whisper-large-v3-turbo | 29 | 12 | 4 | 4 | 9 | 58 |
| Cohere Transcribe | 35 | 12 | 3 | 2 | 6 | 58 |
| Whisper-large-v3 | 31 | 12 | 4 | 4 | 6 | 57 |
| Nemotron-3.5-ASR-Streaming (license conditional) | 25 | 4 | 15 | 2 | 9 | 55 |
| Whisper-medium | 22 | 12 | 4 | 4 | 8 | 50 |
| Whisper-small | 15 | 12 | 4 | 4 | 10 | 45 |
| Omnilingual LLM-ASR (small variants) | 18 | 15 | 3 | 3 | 3 | 42 |

Reuse rationale: Qwen3-ASR decoder = existing Qwen3 path (22); Whisper/Cohere need encoder-decoder cross-attn KV (12); transducers reuse nothing of decoder/KV/sampler (4).

## 7. Excluded (failing hard constraint)

Parakeet-TDT v2/v3, Canary-1b-v2, Canary-Qwen-2.5B, Granite-speech, Voxtral-Mini: no Vietnamese. Parakeet-CTC-0.6B-vi: NVIDIA OML (not listed), gated, .nemo. PhoWhisper, viet-asr: no English must-have (PhoWhisper also pickle-only). Moonshine-vi: license unverifiable. Fun-ASR-MLT-Nano: FunASR license. MMS/Seamless: NC [I]. Omnilingual: passes license, kept only as low-score option (pickle .pt, no vi number).

## 8. Public test sets

| Set | HF id | Revision (sha) | License [V] | Notes |
|---|---|---|---|---|
| LibriSpeech test-clean | `openslr/librispeech_asr`, config `clean`, split `test` | 71cacbfb7e2354c4226d01e70d77d5fca3d04ba1 | CC-BY-4.0 (HF tags), not gated | parquet `clean/test/0000.parquet`, 2,620 utts |
| FLEURS vi | `google/fleurs`, config `vi_vn` [config name I], split `test` | 70bb2e84b976b7e960aa89f1c648e09c59f894dd | CC-BY-4.0 (tags+cardData), not gated | tags say parquet; file listing and vi_vn test size not read, check before pinning |
| Common Voice vi | `mozilla-foundation/common_voice_17_0` | 11dc88355e899d1bf2df74f01b904a8544a17b33 | **no data on HF**: repo has only README/.gitattributes; "since Oct 2025 exclusively on Mozilla Data Collective". CC0 not shown on the HF page [I, from memory] | must download manually from Mozilla Data Collective and record version+sha256; license to be confirmed there |

Attribution: CC-BY-4.0 weights (Parakeet, Canary) and datasets need attribution only; Qwen/Cohere/Whisper-v3 Apache needs NOTICE, Whisper-turbo MIT needs notice.

## 9. Unresolved

1. Is OpenMDW-1.1 "equivalent" under LICENSE_POLICY? (blocks Nemotron).
2. No Vietnamese Common Voice WER published for any compliant model; run our own on the Mozilla Data Collective vi split.
3. Qwen3-ASR VI 5.55 comes from a parsed PDF table (column order inferred) and agrees with third-party 6.15; confirm by running it. Same for Whisper large-v2 10.3 / 18.2.
4. Qwen3-ASR mrope: confirm text-only positions equal plain RoPE; manifest currently refuses rope scaling.
5. Cohere encoder layer count/dims not read (config gated, 401 on raw file); Cohere VI number only in an unreadable chart.
6. Qwen3-ASR minimum transformers version for native support not stated; pick one at pin time.
7. Qwen3-ASR streaming is only documented via vLLM; our engine would need its own chunked re-prefill, effort unestimated.
8. Whisper small/medium licence tags checked only via third-party list.

## Sources

- https://huggingface.co/Qwen/Qwen3-ASR-1.7B , https://huggingface.co/api/models/Qwen/Qwen3-ASR-1.7B , .../Qwen3-ASR-0.6B , https://github.com/QwenLM/Qwen3-ASR , https://arxiv.org/abs/2601.21337
- https://huggingface.co/openai/whisper-large-v3 , https://huggingface.co/api/models/openai/whisper-large-v3 , .../whisper-large-v3-turbo , https://arxiv.org/abs/2212.04356 , https://api.github.com/repos/openai/whisper/license
- https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 , https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release , https://huggingface.co/docs/transformers/main/en/model_doc/cohere_asr
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 , .../parakeet-tdt-0.6b-v2 , .../parakeet-ctc-0.6b-Vietnamese , .../nemotron-3.5-asr-streaming-0.6b , .../canary-1b-v2 , .../canary-qwen-2.5b , https://api.github.com/repos/NVIDIA-NeMo/NeMo/license
- https://openmdw.ai/license/1-1/ , https://huggingface.co/vinai/PhoWhisper-large , https://huggingface.co/coflar-ai/viet-asr , https://huggingface.co/UsefulSensors/moonshine-base-vi
- https://github.com/facebookresearch/omnilingual-asr , https://huggingface.co/ibm-granite/granite-speech-3.3-8b , https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
- https://models.handy.computer/languages/vi (third-party FLEURS vi, Q8_0 ggml)
- https://huggingface.co/api/datasets/openslr/librispeech_asr , https://huggingface.co/api/datasets/google/fleurs , https://huggingface.co/api/datasets/mozilla-foundation/common_voice_17_0
