# 0013: STT family

Status: **proposed** (speech run, 2026-11-05). Selection by S-A3 (weights 40/25/15/10/10). Evidence and sources: `docs/research/stt-candidates-20261105.md` (accessed 2026-11-05; most facts from HF cards/APIs, GitHub licence API and two papers, each tagged verified / third-party / inferred there). Related: 0012, 0014.

## Decision matrix
| Candidate | Quality 40 | Reuse 25 | Stream 15 | Impl 10 | Size 10 | Total |
|---|---|---|---|---|---|---|
| **Qwen3-ASR-1.7B** (Apache-2.0) | 36 | 22 | 11 | 5 | 6 | **80** |
| Qwen3-ASR-0.6B | 30 | 22 | 11 | 5 | 9 | 77 |
| Whisper-large-v3-turbo (MIT) | 29 | 12 | 4 | 4 | 9 | 58 |
| Cohere Transcribe | 35 | 12 | 3 | 2 | 6 | 58 |
| Whisper-large-v3 (Apache-2.0 on HF) | 31 | 12 | 4 | 4 | 6 | 57 |
| Nemotron-3.5-ASR-Streaming-0.6B (OpenMDW-1.1, licence conditional) | 25 | 4 | 15 | 2 | 9 | 55 |

Both families required by S-A3 were examined: encoder + autoregressive decoder (Qwen3-ASR, Whisper, Cohere) and transducer/CTC (Nemotron; Parakeet-TDT v2/v3 are CC-BY-4.0 but have no Vietnamese).

## Excluded (failing a hard constraint)
No Vietnamese: Parakeet-TDT v2/v3, Canary-1b-v2, Canary-Qwen, Granite-speech, Voxtral. Licence: Parakeet-CTC-0.6B-Vietnamese (NVIDIA Open Model License, gated), Fun-ASR (own licence), Moonshine-vi (unverifiable). No English: PhoWhisper (BSD-3, pickle `.bin` only), viet-asr Zipformer. Parakeet-TDT-0.6B-v2 ships `.nemo` only (pickle). Nemotron is **not excluded but conditional**: OpenMDW-1.1 is permissive but not on the policy list; if the owner accepts it, it is the only transducer with both languages (FLEURS vi 11.2 to 13.4, no LibriSpeech number published).

## Chosen model: Qwen3-ASR-1.7B (0.6B as the small variant)
- Licence: Apache-2.0 weights and code (HF API, QwenLM/Qwen3-ASR); Qwen BPE vocabulary in the same repo. Weights: 2 bf16 safetensors shards, 4.7 GB, revision `7278e1e70fe206f11671096ffdd38061171dd6e5` (0.6B: `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`). Fits one H100 beside a 7B LLM (about 14 GB + 4.7 GB + activations).
- Published quality: LibriSpeech test-clean WER 1.63; FLEURS vi 5.55 (paper table parsed from a PDF, column order inferred) and 6.15 (third-party quantised run). **Both Vietnamese numbers are to be confirmed by our own run** on the fixed subset.
- Architecture [config verified, rest from the paper]: audio encoder AuT, 24 layers, d=1024, 16 heads, FFN 4096 (GELU, LayerNorm, biases), three stride-2 conv2d (8x), 12.5 Hz output, non-causal attention in a dynamic window of 1 to 8 s, sinusoidal positions; front end 16 kHz, n_fft 400, hop 160, 128 mels, 30 s chunks; the encoder output is projected and **spliced into the decoder input embeddings**. Decoder: stock Qwen3-1.7B (28 layers, hidden 2048, 16 query / 8 KV heads, head_dim 128, qk-norm, FFN 6144 SiLU, vocab 151936, rope 1e6, tied embeddings), config `mrope_interleaved [24,20,20]`.

## Why it reuses Helios and what it still needs
- The decoder has no cross-attention: it is a `decoder_lm` instance driven through the unchanged `ComputeBackend`; KV, paging, prefix reuse and the seeded sampler apply as they are. The ADR 0012 rule for transformer decoders is satisfied by composition.
- **Hook needed from the LLM track (request R1 in `docs/requests-to-llm-track.md`, to ride the StepPlan v2 bump; includes the prefix-cache identity rule):** a prefill whose input is a sequence of embeddings (or token ids with spans replaced by supplied embedding rows) instead of token ids. The existing path takes token ids only (to be confirmed by reading `StepPlan`). The same hook serves the TTS backbone (0014). Until the owner or the LLM track provides it, the family builds and tests the encoder and everything else, and the end-to-end decode is `[NEEDS-GPU]`/blocked; I will not edit LLM-owned files to get around it.
- **MRoPE:** with one-dimensional text and audio positions the three sections should reduce to plain RoPE [I]. Verified against the Hugging Face modelling code at the pin; the manifest refuses `rope_scaling` today, so the STT section records `rope: plain` with an Ignition check that the reduction holds, instead of loosening that refusal.
- **New ops (about 6, encoder only):** log-mel front end (Rust CPU in `helios-audio`; GPU only if profiling says so), conv2d stride 2 x3 (as im2col GEMM), non-causal windowed MHSA, LayerNorm + GELU FFN with bias, sinusoidal position add, embedding splice and projector.

## Execution design
`SttBackend` of 0012: `reserve` fixes the audio-length buckets (for example 10 s, 30 s, 120 s as 1 to 8 s windows) and concurrency; `capture` records one encoder graph per bucket before Ready; `submit` takes mel features; the encoder output is handed to the LLM path as the embedding prefill (hook above); `take` returns tokens, with segment timestamps only if the model emits them (srt/vtt otherwise built from chunk boundaries and documented as coarse, or refused with 400 for `verbose_json` timing fields). Long audio: fixed 30 s chunking with the previous text as prompt is **not** assumed; chunk handling is decided from the reference implementation's behaviour at the pin.

## Streaming plan (design only in this run)
Encoder window 1 to 8 s with a cache of finished windows; the decoder re-prefills the unstable tail (the reference streaming mode is only documented through vLLM, so this is our own design, effort not estimated); partial and stable transcripts; WebSocket protocol at `/v1/realtime/transcription` carrying PCM16 frames in and `partial`/`final` events out. Implementation only if everything else is green.

## Reference pin and oracles
Pinned at S1.2: `QwenLM/Qwen3-ASR` commit, `transformers` version with native support (not stated on the card; chosen and recorded at pin time), torch. Oracles: encoder output summaries per layer (fp32 and bf16 drift gives the tolerance, declared before any Helios run), decoder top-k and logsumexp per step, decoded text. Test sets: `openslr/librispeech_asr` test-clean (CC-BY-4.0, 200 utterances), `google/fleurs` vi (CC-BY-4.0, 200 utterances), revisions and sha256 recorded when fetched. Common Voice is no longer on Hugging Face (Mozilla Data Collective since 2025-10); not used.

## Budget (S-A6)
WER on the fixed subsets within +0.5 absolute of the reference on the same audio. Speed: real-time factor for 10 s, 30 s, 120 s clips at batch 1 and concurrency 8.

## Owner decisions requested
Accept Qwen3-ASR as the first STT; whether OpenMDW-1.1 counts as acceptable (it would add a streaming transducer family); approval of the embedding-prefill hook request to the LLM track; streaming STT implementation.
