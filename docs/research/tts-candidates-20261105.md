# TTS candidates for Hyperion/Helios (Vietnamese + English, commercial licence)

Date 2026-11-05. All URLs accessed 2026-11-05. Tags: **[V]** verified on primary page (HF card / GitHub / config.json); **[S]** secondary source only (search snippet, blog); **[I]** inference, not verified.
Policy: weights, code, codec, text frontend all Apache-2.0/MIT/BSD/CC-BY-4.0/CC0. No GPL/espeak, no NC/CPML.
Method limit: WebFetch summaries via a small model; quoted numbers come from those summaries. LICENSE files for weights repos were read from HF card metadata, not from a signed legal review.

## 1. Bottom line

- **One candidate meets Vietnamese + commercial licence on paper: VieNeu-TTS v3 Turbo** (Apache-2.0 on every shipped artifact, vi+en, 0.1B, MOSS codec Apache-2.0, sea-g2p Rust Apache-2.0). Residual risks: training data undisclosed (gated, pipeline not public), **no published CER/WER/MOS** for it, sea-g2p data-table provenance unstated.
- Second: **ZeroTTS** (MIT code+weights, Vietnamese only, self-reported WER 1.03%) but architecture undisclosed, ONNX-only weights, voice encoder unpublished (no cloning), data undisclosed. Shares the same codec as VieNeu.
- Third (fallback, bigger): **Qwen3-TTS-12Hz-1.7B + community Vietnamese fine-tune (beyoru)**: Apache-2.0 chain, CER 0.065 on 24 sentences (self-reported). Base model has no Vietnamese (CER 0.25). Fine-tune data licence unknown.
- **Family B (NAR acoustic model + vocoder): no candidate meets Vietnamese + commercial.** Every Vietnamese F5/VITS/XTTS checkpoint is NC/CPML or derived from NC Emilia-trained F5 weights. Closest: ZeroTTS (if it is NAR; unknown) then Kokoro (licence ok, no Vietnamese).
- **Overall winner: VieNeu-TTS v3 Turbo (family A).** Gate before commit: independent Vietnamese eval (PhoWhisper CER + listening) and written data-provenance answer from the author.

## 2. Hard-constraint screen

| Candidate | Fam | Weights licence | Vietnamese | Verdict | Source |
|---|---|---|---|---|---|
| VieNeu-TTS v3 Turbo | A | Apache-2.0 [V] | yes (vi+en) [V] | PASS (data risk) | [HF](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo) |
| VieNeu-TTS 0.5B (v1) | A | Apache-2.0 [V]; fine-tune of NeuTTS Air; codec NeuCodec Apache-2.0 [V], codec repo gated | yes | PASS-ish, older, 0.5B, datasets 140h/1000h licence unstated [V] | [HF](https://huggingface.co/pnnbao-ump/VieNeu-TTS) |
| VieNeu-TTS-0.3B | A | CC-BY-NC-4.0 [S] (PyPI) | yes | EXCLUDED NC | [PyPI](https://pypi.org/project/vieneu/1.2.0/) |
| VieNeu v4 | A | proprietary [V] | yes | EXCLUDED not released | [GitHub](https://github.com/pnnbao97/VieNeu-TTS) |
| Qwen3-TTS 1.7B Base | A | Apache-2.0 [V] | NO (10 langs, none vi); fine-tune card: base vi CER 0.25 [V] | EXCLUDED as-is (vi quality) | [HF](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) |
| beyoru Qwen3-TTS-1.7B-Vi | A | Apache-2.0 inherited [V]; finetune data licence not stated [V] | yes, CER 0.065 (24 sent., Whisper-large-v3-turbo, self-report) [V] | PASS (data risk) | [HF](https://huggingface.co/beyoru/Qwen3-TTS-12Hz-1.7B-Vi) |
| Orpheus 3B ft | A | Apache-2.0 tag, base Llama-3.2 (Llama community licence, not on the allow list) [V] | no (English tag) | EXCLUDED no vi; Llama licence | [HF](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) |
| Llasa-3B | A | CC-BY-NC-4.0 [V] | zh/en | EXCLUDED NC, no vi | [HF](https://huggingface.co/HKUSTAudio/Llasa-3B) |
| Spark-TTS 0.5B | A | CC-BY-NC-SA-4.0 [V] | zh/en | EXCLUDED NC, no vi | [HF](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) |
| OuteTTS 1.0 1B | A | CC-BY-NC-SA + Llama licence [V] | 23 langs, no vi | EXCLUDED NC, no vi | [HF](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B) |
| Sesame CSM-1B | A | Apache-2.0 [V] | English (weak non-en) [V] | EXCLUDED no vi | [HF](https://huggingface.co/sesame/csm-1b) |
| Zonos v0.1 | A | Apache-2.0 [V] | en/ja/zh/fr/de; needs espeak-ng [V] | EXCLUDED espeak (GPL), no vi | [HF](https://huggingface.co/Zyphra/Zonos-v0.1-transformer) |
| Dia 1.6B | A | Apache-2.0 [V] | English only [V] | EXCLUDED no vi | [HF](https://huggingface.co/nari-labs/Dia-1.6B-0626) |
| Chatterbox multilingual | A | MIT [V], mandatory PerTh watermark [V] | 23 langs, no vi [V] | EXCLUDED no vi | [HF](https://huggingface.co/ResembleAI/chatterbox) |
| MOSS-TTS-Nano 100M | A | Apache tag but README says treat as unlicensed until LICENSE published [V] | 19 listed langs, no vi [V] | EXCLUDED no vi, licence unclear | [HF](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M) |
| Fish Speech, Higgs, Kyutai TTS, VibeVoice, IndexTTS2, NeuTTS Air | A | not examined | not examined | NOT EVALUATED (no time; NeuTTS Air is English [I]) | |
| ZeroTTS 202M | B? | MIT code+weights [V] | Vietnamese only [V] | PASS (opaque) | [GitHub](https://github.com/zeroweight-ai/ZeroTTS), [HF](https://huggingface.co/zeroweight-ai/ZeroTTS) |
| Kokoro-82M | B | Apache-2.0 [V] | no vi (8 langs) [V]; misaki + espeak in usage [V] | EXCLUDED no vi | [HF](https://huggingface.co/hexgrad/Kokoro-82M) |
| zalopay/vietnamese-tts (F5 FT) | B | CC-BY-4.0 tag, but base F5-TTS weights are CC-BY-NC (Emilia) [V] | yes, no eval [V] | EXCLUDED derivative of NC weights; relicensing invalid [I] | [HF](https://huggingface.co/zalopay/vietnamese-tts), [F5](https://huggingface.co/SWivid/F5-TTS) |
| Vi-F5-TTS (danhtran2mind), EraX F5 | B | MIT code; same Emilia/NC base warning [S] | yes | EXCLUDED same reason | [GitHub](https://github.com/danhtran2mind/Vi-F5-TTS) |
| hynt F5-TTS-Vietnamese-ViVoice/100h | B | CC-BY-NC-SA-4.0 [V] | yes, no eval | EXCLUDED NC | [HF](https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice) |
| viXTTS | B/A | CPML [V] | yes | EXCLUDED CPML | [HF](https://huggingface.co/capleaf/viXTTS) |
| MMS-TTS vie (VITS 36.3M) | B | CC-BY-NC-4.0 [V] | yes | EXCLUDED NC | [HF](https://huggingface.co/facebook/mms-tts-vie) |
| VietTTS (dangvansam) | B | code Apache, pretrained models CC-BY-NC [S] | yes | EXCLUDED NC | [HF](https://huggingface.co/dangvansam/viet-tts) |
| OmniVoice (k2-fsa, 0.6B masked-diffusion LM) | B | code Apache, model CC-BY-NC (Emilia) [V] | 646 langs, vi not stated [V] | EXCLUDED NC | [HF](https://huggingface.co/k2-fsa/OmniVoice) |
| Piper | B | repo MIT, archived 2025-10-06, successor repo named `piper1-gpl` [V]; espeak-ng frontend [I] | vi voices exist [I] | EXCLUDED GPL frontend/successor [I] | [GitHub](https://github.com/rhasspy/piper) |
| F5/E2/Matcha/StyleTTS2/VITS2 code, no commercial-clean vi checkpoint found | B | n/a | n/a | no checkpoint (searches 2026-11-05; not exhaustive) | |

## 3. Shortlist walk-through

### 3.1 VieNeu-TTS v3 Turbo [V unless tagged]
- Licence: Apache-2.0 on model.safetensors, ONNX, configs, tokenizer, 23 preset voice assets; preset voices and output declared commercially usable. Repo LICENSE is stock Apache-2.0, copyright holder line unfilled. Attribution/notice retention required.
- Weights: `model.safetensors` 262 MB, no pickle at repo root (subfolders not inspected), plus `speaker_encoder.onnx` 28 MB, `denoiser.onnx` 43 MB (ONNX, optional; reference-clip cloning path). Config says float32; card says BF16 0.1B.
- Params 0.1B (~131M if 262 MB is bf16 [I]). bf16 weights ~0.26 GB; with codec decoder ~22M: negligible.
- Backbone (config.json): 12 layers, hidden 768, 12 heads / 4 KV heads (GQA), head_dim 64, FFN 3072, RMSNorm eps 1e-6, max pos 1024, untied embeddings. Card mentions a Qwen3 backbone (qk-norm likely [I]). Plus **acoustic decoder: 2 layers, 8 heads, FFN 2048** with learned slot-position embedding (a per-frame depth/local transformer across n_vq=16 slots [I]; exact delay/depth scheme not documented; must read modeling code).
- Tokens: 16 codebooks x 1024 entries (+ pad 1024); text vocab 419 (char/phoneme-level, sea-g2p output [I]); specials: BOS/EOS, text start/end, speech start/end, audio-reference slot, 5 emotion ids, 30 reserved.
- Codec: MOSS-Audio-Tokenizer-Nano (Apache-2.0, [HF](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano)): 21.97M params (enc 10.45M, dec 10.45M, quantizer 1.07M), **causal Transformer blocks, no CNN** ("Cat" design), 48 kHz stereo I/O (VieNeu outputs 48 kHz mono [I]), 12.5 Hz, 16 RVQ x 1024 (0.125 kbps each, max 2 kbps), safetensors fp32. Card inconsistent on downsample rate (7680 vs 3840 samples; 3840 = 12.5 Hz used in streaming example). Layer count/patching of decoder not on card: read code.
- Vietnamese: dedicated, ~10,000 h en+vi, code-switching; 3 emotion tags. **Evidence: none** (no CER/WER/MOS; only latency/throughput self-reports: first audio ~115 ms GPU, 16 streams RTX 3060 RTF 0.5-0.6). Third-party comparison from ZeroTTS repo omits VieNeu. Quality is unproven.
- Frontend: sea-g2p ([GitHub](https://github.com/pnnbao97/sea-g2p)), Rust, Apache-2.0, dictionaries+rules+17-stage Vietnamese normaliser, no espeak, zero Python deps; usable in-repo as a Rust crate (C ABI) [V]. Provenance/licence of embedded tables (`vi_top_syllables.rs`, `vi_bigrams.rs`, `resources.rs`) not stated [unresolved]; Indonesian side uses WikiPron/KBBI data (irrelevant for vi, but Wiktionary data is CC-BY-SA: audit if the crate is vendored whole).
- Reference impl: PyTorch + transformers pinned 4.57.6, torch 2.8, CUDA >= 12.8 [V]. trust_remote_code not mentioned; model type `vieneu_v3_turbo` / class `VieNeuV3TurboForTTS` implies custom modeling code: pin the GitHub commit and read it. Helios loads safetensors only, so no pickle at load time [V for safetensors; I for no hidden pickle in subfolders].
- Streaming: frame-by-frame generation, 12.5 Hz frames -> 80 ms/frame; causal codec decoder => chunked decode natively.
- Seeded reproducibility: AR with 16 samplings per frame (1 backbone + depth loop [I]); the existing Philox device sampler fits; per-slot RNG stream key = (seed, frame, codebook) is feasible. Bit-exact vs PyTorch not expected, only vs own host reference (as for the LLM path).
- Voices/cloning: 23 preset voices (consent covers bundled presets only). Cloning from a 3-8 s clip is supported by the model (needs speaker_encoder.onnx or an equivalent port: extra op set). Policy: ship presets only; cloning behind explicit config + consent flag; card warns on impersonation.

### 3.2 Qwen3-TTS 12Hz 1.7B + Vi fine-tune [V]
- Base: Apache-2.0, safetensors BF16, 10 languages (zh, en, ja, ko, de, fr, ru, pt, es, it), no Vietnamese; 5M h training; model paper [arXiv 2601.15621](https://arxiv.org/abs/2601.15621). Code/tokenizer Apache-2.0 per paper abstract.
- Talker: 28 layers, hidden 2048, 16 heads / 8 KV, FFN 6144, rope_theta 1e6, **interleaved MRoPE sections [24,20,20]**; code predictor: 5 layers, hidden 1024, 16/8 heads, FFN 3072; 16 code groups, predictor vocab 2048, talker vocab 3072; speaker encoder out 2048, 24 kHz.
- Codec 12 Hz tokenizer: 12.5 Hz, 16 codebooks, **lightweight causal ConvNet** decoder (details beyond that not on card). First packet 97 ms (self-report). Dual-track text/audio streaming.
- Vi fine-tune ([HF](https://huggingface.co/beyoru/Qwen3-TTS-12Hz-1.7B-Vi)): CER 0.065 vs 0.253 base (24 sentences, Whisper-large-v3-turbo, 2 training speakers, Northern accent only, no vi language token, "Misa" mispronunciation, proper nouns need lexicon). Fine-tune dataset not named. 2B params bf16 ~4 GB.
- Text frontend: raw text through Qwen tokenizer, no phonemizer [I].
- Voice cloning from 3 s reference (+transcript) via speaker encoder: extra op set.

### 3.3 ZeroTTS 202M [V from README]
- MIT code+weights; bundled MOSS decoder Apache-2.0 (ONNX decoder graphs); benchmark set CC-BY-NC (only the dataset). 48 kHz, streaming, first audio ~53-89 ms CPU.
- Self-benchmark (ZeroBench-TTS, PhoWhisper ASR): WER 1.03% vs OmniVoice 4.13, XTTS-v2-vi 16.42, viXTTS 18.40; UTMOS 2.91 (predicted, not listener MOS). Benchmark authored by the model's own team; VieNeu not compared.
- Acoustic model type/layers undisclosed; weights **fp32 ONNX ~900 MB** (size inconsistent with 202M params, so graphs probably include codec and more [I]); no safetensors; voice encoder unpublished (voices as `.npz` latents, shape (1, n_voice_queries, 768)); training data undisclosed. Hosted platform sells voice creation. Needs a Vietnamese normaliser (separate, optional).
- Cannot classify AR vs NAR or estimate op inventory without ONNX graph inspection. Scored as a NAR-leaning unknown.

## 4. Op inventory vs standard decoder-only transformer (Helios already has: embedding, RMSNorm, RoPE, GQA paged attention, SwiGLU, bf16 GEMM, sampler)

| New op / feature | VieNeu v3 Turbo | Qwen3-TTS-Vi | Notes |
|---|---|---|---|
| Multi-codebook input embedding sum (16 tables) + audio/text token mux | yes | yes | 1 gather-sum kernel |
| Depth/local transformer over codebooks per frame (small, non-paged, 2 or 5 layers) | yes (2 layers) | yes (5 layers) | reuse layer kernels with fresh scratch KV; sequential 15 inner steps/frame; CUDA-graph friendly |
| Per-codebook output heads (16 heads) + per-head sampling | yes | yes | reuses sampler |
| Interleaved MRoPE | no [I] | yes | variant of RoPE kernel |
| Codec decoder: causal transformer w/ patch/unpatch upsampling, RVQ dequant (sum of 16 lookups), final waveform head | yes (Cat, no conv) | no | few new ops: RVQ lookup-sum, linear patch up/down; sliding-window/causal attention (existing kernel w/ window, [I]) |
| Codec decoder: causal Conv1d, ConvTranspose1d upsample, snake/other activation, RVQ dequant | no | yes | 3-4 new op types ([I] exact set unknown) |
| Speaker encoder (cloning) | ONNX (CNN/other, unknown) | transformer/other, 24 kHz mel (unknown) | optional; skip for presets-only |
| Text frontend | sea-g2p Rust crate | none (BPE) | no GPU op |
| Total distinct new GPU op types (inference, presets only) | ~4 [I] | ~7 [I] | |

## 5. Scored matrices (weights: Vietnamese 40, reuse 25, streaming 15, impl size 10, size/speed 10)

Scores are my judgement from the evidence above; Vietnamese score is capped by lack of independent eval.

### Family A (LLM backbone + codec tokens)

| Candidate | Vi /40 | Reuse /25 | Stream /15 | Impl /10 | Size-speed /10 | Total |
|---|---|---|---|---|---|---|
| **VieNeu-TTS v3 Turbo** | 28 (dedicated 10k h, no eval) | 21 (GQA decoder + small depth tf; custom heads) | 14 (80 ms frames, causal codec, 115 ms TTFA) | 8 (transformer codec, no conv) | 10 (0.1B) | **81** |
| Qwen3-TTS 1.7B + Vi FT | 30 (CER 0.065 measured, 2 speakers, N accent) | 22 (28L GQA talker, MRoPE tweak) | 14 (97 ms) | 5 (conv codec, MRoPE) | 5 (2B, 28 layers, 16-codebook depth model) | **76** |
| VieNeu-TTS 0.5B (v1) | 20 (no eval, 140h/1000h data) | 22 (Qwen-style AR + single-codebook NeuCodec) | 10 (NeuCodec 50 Hz 0.6B decoder, upsampling decoder) | 4 (0.6B conv/transformer codec, w2v-bert-like path [I]) | 6 | 62 |
| Qwen3-TTS base (no FT) | 12 | 22 | 14 | 5 | 5 | 58 (EXCLUDED, vi quality) |
| Orpheus / CSM / Dia / Zonos / Chatterbox / MOSS-Nano | 0 | ~20 | ~10 | ~5 | ~5 | not ranked: EXCLUDED, no Vietnamese and/or licence |
| Llasa / Spark / OuteTTS | 0 | n/a | n/a | n/a | n/a | EXCLUDED NC |

Winner A: **VieNeu-TTS v3 Turbo** (81), conditional on eval + data provenance. Runner-up Qwen3-TTS-Vi (76) is the safer bet on measured Vietnamese quality and a documented architecture lineage, at 20x the params; it also has an official English path (English CER 0.000 in the card test).

### Family B (NAR acoustic model + vocoder)

| Candidate | Vi /40 | Reuse /25 | Stream /15 | Impl /10 | Size-speed /10 | Total |
|---|---|---|---|---|---|---|
| ZeroTTS (MIT) | 30 (WER 1.03% self-bench, 2.91 UTMOS) | 8 (unknown; ONNX-only) | 12 (~70 ms CPU) | 3 (unknown, ONNX import) | 9 (202M) | **62** (opaque) |
| Kokoro-82M | 0 (no vi) | 5 (StyleTTS2: BiLSTM, AdaIN, ISTFTNet; not transformer-decoder) | 6 | 2 (many new ops) | 10 | 23 (EXCLUDED, no vi) |
| F5-TTS Vi fine-tunes (zalopay, hynt, EraX, Vi-F5) | 20-25 | 8 (DiT, flow-matching ODE loop, Vocos) | 4 | 4 | 6 | EXCLUDED: NC base / NC-SA |
| VITS (MMS vie, VietTTS) | 22 | 3 | 8 | 2 | 10 | EXCLUDED NC |

Winner B: none passes all hard constraints with verified facts. **ZeroTTS is the only licence-clean Vietnamese candidate**, but its class (AR vs NAR) and op set are unknown; it may in fact be an LLM-style model sharing the MOSS codec with VieNeu, so it may belong in family A.

## 6. If neither candidate survives (what would be needed)

Trigger cases: eval shows VieNeu quality below bar; author cannot document training data; legal rejects undisclosed data.
1. **Fine-tune Qwen3-TTS-1.7B (or the 0.1B VieNeu-like architecture trained from scratch) on licence-clean Vietnamese data.** Candidate datasets [I, not verified this session, check each licence before use]: Mozilla Common Voice vi (CC0; crowd mic noise, short read sentences), Google FLEURS vi (CC-BY-4.0, ~10 h), Emilia-YODAS vi subset (CC-BY-4.0 per NeuCodec card's claim that YODAS part is CC-BY; vi share unknown), VIVOS (CC-BY-SA 4.0: not on allow list), viVoice / VLSP / phoaudiobook (licences unknown or NC: do not use). Realistic CC0/CC-BY volume is small (tens of hours), enough for fine-tuning an already-multilingual base, not for training from scratch.
2. Alternatively commission recorded studio data under a commercial contract (own the licence).
3. Do not use F5/XTTS/VITS routes: the NC base weights and datasets contaminate every derivative checkpoint found.

## 7. Integration notes for Helios

- Codec decode (MOSS Nano, 22M) is the shared critical new component for VieNeu and ZeroTTS: implement once. Its encoder is only needed for cloning/reference prompts.
- Backbone path maps onto the existing paged-KV decoder (12 layers, 768 dim, GQA 12/4, head_dim 64; page size and head_dim 64 should be supported already [I]). Depth transformer needs non-paged tiny KV.
- Manifest: needs a new `ModelConfig` variant (architecture/contract revision, ADR territory per project rules) for audio-codec output; owner decision.
- bf16: both fp32 checkpoints must be cast; validate against fp32 PyTorch reference token-by-token under greedy, then codec output by waveform SNR/mel distance.
- No GPL: sea-g2p vendoring needs a data-table provenance audit first.

## 8. Unresolved questions

1. Does VieNeu v3 Turbo reach acceptable Vietnamese CER/MOS on an independent set (PhoWhisper + human listening incl. tones, numbers, English code-switch)? No published numbers.
2. Training data for VieNeu v3 Turbo and ZeroTTS (undisclosed): any NC or consent-less sources inside? Ask authors in writing; the Apache/MIT tag alone is a claim, not proof.
3. Provenance and licence of sea-g2p embedded Vietnamese tables.
4. Exact VieNeu modeling code: depth-decoder scheme, delay pattern, qk-norm, whether trust_remote_code or pickle is involved; codec decoder layer structure (read after pinning commits).
5. ZeroTTS architecture (AR vs NAR), ONNX graph contents, and whether the voice encoder will ever be published.
6. MOSS-TTS-Nano repo LICENSE state vs the tokenizer repo's Apache tag (tokenizer card says Apache-2.0; TTS-Nano README hedges).
7. Not evaluated: Fish Speech, Higgs Audio, Kyutai TTS, VibeVoice, IndexTTS2, NeuTTS Air, Confucius4-TTS (arXiv 2608.11650), 2026 Vietnamese releases beyond those found.
8. Is Llama-3.2-derived (Orpheus) licence acceptable as "equivalent"? Moot since no Vietnamese.

## Sources (accessed 2026-11-05)
- https://github.com/pnnbao97/VieNeu-TTS ; https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo (+ raw/main/config.json) ; https://huggingface.co/pnnbao-ump/VieNeu-TTS
- https://github.com/pnnbao97/sea-g2p
- https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano ; https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M
- https://github.com/zeroweight-ai/ZeroTTS ; https://huggingface.co/zeroweight-ai/ZeroTTS
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base (+ raw/main/config.json) ; https://arxiv.org/abs/2601.15621 ; https://huggingface.co/beyoru/Qwen3-TTS-12Hz-1.7B-Vi
- https://huggingface.co/neuphonic/neucodec
- Model cards listed in section 2 table.
