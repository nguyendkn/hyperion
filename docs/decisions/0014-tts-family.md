# 0014: TTS family

Status: **proposed** (speech run, 2026-11-05). **VieNeu is `provisional, not for production`** (owner direction 2026-11-05): engineering proceeds, because the host, codec and API work is reusable for any same-shape model; the manifest field `production_approved` stays false until the author confirms the training-data provenance and commercial rights in writing and the `sea-g2p` table licences are audited. A licence-clean fine-tune is evaluated in parallel; its hard part is enough clean Vietnamese data (possibly purchased or recorded). Selection by the rule of S-A3 (weights 40/25/15/10/10). Evidence and sources: `docs/research/tts-candidates-20261105.md` (accessed 2026-11-05). Related: 0012 (families), 0013 (STT, used for round-trip evaluation).

## Evidence quality (read first)
The research read Hugging Face cards, GitHub pages and `config.json` through summarising fetches, not the raw LICENSE files or the modelling code. Licences below are **claims from model cards** until the owner or a legal pass confirms them. Items marked [I] are inference. The first task of S1.2 is to pin each repo to a commit and read the modelling code; any difference from this ADR is recorded as an amendment.

## Decision matrix (family A: LLM backbone emitting codec tokens)
| Candidate | Vi 40 | Reuse 25 | Stream 15 | Impl 10 | Size 10 | Total |
|---|---|---|---|---|---|---|
| **VieNeu-TTS v3 Turbo** (Apache-2.0) | 28 | 21 | 14 | 8 | 10 | **81** |
| Qwen3-TTS-12Hz-1.7B + community Vi fine-tune (Apache-2.0 chain) | 30 | 22 | 14 | 5 | 5 | 76 |
| VieNeu-TTS 0.5B v1 | 20 | 22 | 10 | 4 | 6 | 62 |

VieNeu Vietnamese score is capped: **no published CER/WER/MOS exists**; the 28 rests on a dedicated ~10 000 h en+vi training claim. Qwen3-TTS-Vi has a self-reported CER 0.065 (24 sentences, two speakers, Northern accent), base model without the fine-tune CER 0.25.

## Family B (non-autoregressive acoustic model + vocoder): none selected
Every Vietnamese F5/VITS/XTTS checkpoint found is CC-BY-NC, CPML, or fine-tuned from the NC (Emilia) F5 weights (a CC-BY tag on such a derivative is not trusted); Kokoro and Piper fail Vietnamese or the GPL-frontend rule; ZeroTTS (MIT, WER 1.03 percent on its authors' own benchmark) ships ONNX only, with undisclosed architecture and data, so it cannot be classified or ported without graph inspection. By S-A3 **the second TTS family line is stopped** (hard constraint: commercial licence plus Vietnamese). Owner options: accept the AR family only (recommended), fine-tune on licence-clean Vietnamese data, or later revisit ZeroTTS after its architecture is documented.

## Excluded (failing hard constraint)
Orpheus (no Vietnamese; Llama licence), Llasa, Spark-TTS, OuteTTS (NC), CSM, Dia, Chatterbox, MOSS-TTS-Nano (no Vietnamese or licence unclear), Zonos (espeak GPL, no Vietnamese), VieNeu 0.3B (NC), VieNeu v4 (proprietary), viXTTS (CPML), MMS-TTS vie and VietTTS (NC), OmniVoice (NC). Not evaluated for lack of time: Fish Speech, Higgs, Kyutai TTS, VibeVoice, IndexTTS2, NeuTTS Air.

## Chosen model: VieNeu-TTS v3 Turbo (provisional, gated)
- 0.1B, one `model.safetensors` (262 MB, fp32 file, cast to bf16). Backbone: 12 layers, hidden 768, 12 heads / 4 KV heads, head_dim 64, FFN 3072, RMSNorm 1e-6, untied embeddings, max position 1024; a 2-layer acoustic (depth) decoder over 16 codebook slots [I: exact delay and depth scheme to be read from the code]. Text vocabulary 419 (output of the sea-g2p frontend [I]); 16 codebooks of 1024 entries plus pad.
- Codec: MOSS-Audio-Tokenizer-Nano (Apache-2.0, 22M parameters, causal transformer blocks, no convolutions, 12.5 Hz, 16 RVQ, 48 kHz). The card is inconsistent on the downsample rate (7680 or 3840 samples); 3840 is what the streaming example uses; resolved by reading the code.
- Memory: about 0.3 GB of weights; fits next to a 7B LLM by a wide margin.

## Gates before more than skeleton work (owner may override)
1. **Quality gate, measured by us:** synthesise the 100 fixed Vietnamese sentences with the PyTorch reference, transcribe with the STT of 0013, report CER/WER, and listen to a sample. If round-trip WER is above 15 percent absolute, stop and report (threshold declared now; it separates "works" from "unusable", it is not a quality claim).
2. **Provenance gate:** the training data is undisclosed. This is reported to the owner as a legal risk, not resolved here. Only bundled preset voices are served; voice cloning (speaker encoder, ONNX) is not implemented and a `voice` outside the presets is a 400.
3. **Frontend:** `sea-g2p` is a Rust crate (Apache-2.0) but its embedded Vietnamese tables have no stated provenance. Vendoring is blocked until audited; fallback is an in-repo rule-based normaliser (numbers, dates, abbreviations) plus the model's own text vocabulary, as S3 requires.
4. **Fallback model:** if gate 1 or the code review fails, Qwen3-TTS-Vi (about 2B, convolutional codec, MRoPE variant, about 7 new op types [I]) is the next candidate; its fine-tune data licence is also unstated.

## Execution design (family A on ADR 0012)
- Backbone: a `decoder_lm` instance through the unchanged `ComputeBackend`, extended vocabulary, seeded device sampler. Needs an input the LLM path may not have: **summed multi-codebook embeddings per frame** and **16 output heads per step**. If the page-index path cannot express them, this is a request for the LLM track (a hook for per-step embedding input and extra heads), and the family stays on its own small kernels until then. Decided after reading the model code.
- Own kernels (`kernels/speech/`): multi-codebook embedding gather-sum, depth transformer (15 sequential inner steps per frame, scratch KV, CUDA-graph friendly), RVQ dequantise-sum, codec decoder (patch/unpatch linear, causal windowed attention, waveform head). About 4 new op types [I].
- Reproducibility: per-draw key (seed, frame, codebook) on the Philox sampler; exact equality of discrete codes is claimed only against our own host reference; against PyTorch only under greedy (declared before the run).
- Streaming (stretch): 12.5 Hz frames (80 ms), causal codec, chunked decode with a codec state; time to first audio is measured at concurrency 1 and 8.
- Reference pin: VieNeu repo commit, `transformers` 4.57.6, torch 2.8 (from the card), codec repo commit; custom modelling code is read, never executed with `trust_remote_code` at Helios load time.

## Op inventory (new vs existing Helios kernels, presets only)
Gather-sum of 16 embedding tables; depth transformer layers (reuse of layer kernels with scratch KV); 16 output heads with per-head sampling (reuses sampler); RVQ lookup-sum; codec decoder blocks with patch up/down projection; final waveform head. Existing: embedding, RMSNorm, RoPE, GQA attention, SwiGLU, bf16 GEMM, sampler.

## Budgets (S-A6)
Round-trip WER within +1.0 absolute of the reference synthesis transcribed by the same STT; speed reported as real-time factor and time to first chunk, no budget.

## Owner decisions requested
Accept AR-family-only TTS; accept the provenance risk or require written author statements; voices policy (presets only, cloning off); whether to fund a licence-clean fine-tune; whether the missing manifest `ModelConfig` variant for codec output is acceptable as the `tts_<arch>` section of ADR 0012.
