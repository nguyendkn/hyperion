# 0012: model families and several models in one process

Status: **proposed** (speech run, 2026-11-05; the owner reviews it at the end of the run). Implemented provisionally under owner authorisation S-A2. Related: 0006 (ComputeBackend, unchanged), 0007 (StepPlan, unchanged), 0010 (manifest constants), 0013 (STT), 0014 (TTS).

## Context
Helios serves one decoder-only LLM. Speech models (STT, TTS) and later others (embeddings, rerankers, vision encoders) have different inputs, outputs and execution shapes: audio in and text out, text in and audio out, one forward pass with no KV. They must share what is expensive to build (driver and cuBLAS loaders, VMM arenas, graph capture, kernel binaries, packaging, signing, the HTTP layer) without bending the LLM contracts, and several of them must fit one GPU at once.

## Decision

### 1. Families
- `ModelFamily` is a closed enum: `decoder_lm` (today's LLM, the implicit default), `stt_<arch>`, `tts_<arch>`; a new family is a new variant plus the checklist in section 6.
- **The LLM `Manifest` struct is not edited.** A new `FamilyManifest` (new file `specification/src/v1/family.rs`) carries the common header (`model_id`, `target_arch`, `tensor_format`, `payloads`, `contract_revision`, `family`, signature fields later) and one family section (`stt`, `tts`) with that family's constants. `parse_any(bytes)` peeks at the `family` key: absent or `"decoder_lm"` goes to `Manifest::from_json_bytes` unchanged, anything else to `FamilyManifest`. Existing manifests, goldens and the committed schema stay byte-identical (tested). A family whose backbone is a decoder LM (LLM-backbone TTS) embeds a complete `ModelConfig` in its section, so the backbone is described by the same type as any LLM.
- Weights, `payloads` table and `cache_identity()` rules of ADR 0003/0010 apply to every family; the identity domain tag includes the family.

### 2. Execution contracts: new traits, same lifecycle
`ComputeBackend` stays the `decoder_lm` contract. New traits live in a new crate `helios-speech` (depends on `helios-hal`'s public `cuda::*` modules; no `helios-hal` file is edited):
```rust
pub trait SttBackend: Send + Sync {            // one instance per model
    fn load(&self, cfg: &SttConfig, w: &WeightSource) -> Result<SttLoaded>;
    fn reserve(&self, m: &SttLoaded, spec: &SttSpec) -> Result<SttReservation>;   // VMM range + length buckets, concurrency
    fn capture(&self, m: &SttLoaded, r: &SttReservation) -> Result<SttGraphs>;    // per audio-length bucket, before Ready
    fn commit(&self, r: SttReservation, g: &SttGraphs, p: &MemoryPolicy) -> Result<SttReady>;
    fn submit(&self, r: &SttReady, job: SttJob) -> Result<SttTicket>;             // features in, returns at once
    fn take(&self, t: SttTicket) -> Result<SttOutput>;                            // tokens, timestamps, logprobs
}
pub trait TtsBackend: Send + Sync { /* same chain; TtsJob = text ids + voice + seed; TtsOutput = PCM chunks (streaming-capable) */ }
```
The type-state chain is the one of ADR 0006: no call can serve before `commit` made the ready token, and nothing allocates, compiles or captures after it (invariant for every family, tested with the allocation counter). Family-specific job and output types are plain data.

### 3. Transformer decoders inside a family: composition, no edits
- **LLM-backbone TTS** (text and codec tokens in one vocabulary): the backbone is a real `decoder_lm` instance driven through the unchanged `ComputeBackend` (`submit_step` / `take_output`, StepPlan with the extended vocabulary and the seeded device sampler). The TTS family adds only the codec decoder and the glue. A model that needs an input the LLM path has no slot for (cross-attention, audio embeddings in place of token embeddings, multi-codebook heads per step) needs a hook in LLM-owned files: it goes in the run report as a request, and the family is built on its own kernels meanwhile.
- **Encoder-decoder STT**: cross-attention is not in the LLM path. The decoder is implemented inside `helios-speech` with its own small self-attention cache and a static cross-attention cache per request, reusing the public device helpers (cuBLAS wrappers, VMM, graph builder) and, where the signature allows, the existing elementwise and RMSNorm kernels through their fatbin. Paged attention is **not** reused unless 0013 shows that a public API fits.
- A family never reaches into `helios-core`: the LLM scheduler, StepPlan and sampler reference are untouched.

### 4. Several models in one process
- **Registry:** `ModelRegistry`: name to `{family, runtime handle, partition, priority class, queue bound}`. Flags (additive, new subcommand `serve-multi` or repeated `--model name=path,fraction=...`; the existing `serve` flags do not change) declare models and partitions.
- **Memory partitions:** each model owns its own VMM arena, streams and graphs, all reserved and captured before Ready. The planner computes `fixed bytes = weights + activations + graph memory + KV/cache` per speech model from its spec, in bytes, and refuses with a typed `PartitionError` before Ready when the sum exceeds the device total minus the reserve. **The LLM is loaded last**: its `commit_kv` already measures the free memory and maps all of it, so its partition is "what is left", bounded by the existing `MemoryPolicy.max_tokens` and `reserve_bytes` (no field added); a declared LLM budget becomes a `max_tokens` the planner derives from `KvSpec`. LLM-only configuration takes the old code path with no registry and no arbiter.
- **Scheduling:** one scheduler per model (the LLM's is as is) plus a **GPU arbiter** ordering `submit` calls between models by class and deadline: interactive audio > interactive text > batch, earliest deadline first inside a class, bounded queues, `429` per model (the response names the model). The arbiter never reorders inside a model and never touches StepPlan; it only decides when a model's scheduler thread may call `submit`. Each model runs on its own stream, so the arbiter is cooperative ordering at submit time, not preemption: an in-flight LLM step is not interrupted (stated limit; measured in S4).

### 5. API and audio front end
- `POST /v1/audio/transcriptions` (multipart: `file`, `model`, `language`, `response_format` = `json|text|verbose_json|srt|vtt`, `temperature`); `POST /v1/audio/speech` (json: `model`, `input`, `voice`, `response_format` = `wav|pcm`, `speed`; chunked streaming as an option). Errors use the existing OpenAI-shaped body; unsupported fields and values are refused with 400, never ignored.
- **Front end in-repo** (new crate `helios-audio`, no dependency): WAV/PCM read and write, deterministic windowed-sinc resampler, STFT/iSTFT and mel filterbank exactly as the reference model defines them. Compressed formats (mp3, flac, ogg, m4a) are refused with a clear 400 until the owner approves a decoder dependency (S-A5).
- Routes live in a new module registered by one line (`speech: hook route`); the model field selects the registry entry; a model of the wrong family is a 400.

### 6. Checklist for a new family
Manifest section and parser branch, Ignition compiler (strict tensor map, payload table, shard support), backend trait and implementation, oracle generator with tolerances written first, API route, budgets, ledger section, memory spec for the planner, priority class for the arbiter. Embeddings, rerankers and vision encoders follow it as "one pass, no KV" families (a shared `OnePassBackend` shape is likely and is decided at the second such family, not now).

## Why over the alternatives
- **Extend `ComputeBackend`:** forces speech into KV-and-token shapes and breaks the accepted trait; owner rule: one change, it happened.
- **One process per model:** duplicates CUDA contexts (~0.5 GB each), cannot order work across models, and loses shared weights and arena planning.
- **Edit `Manifest` for families:** touches LLM-owned fields and every golden; the envelope costs one dispatcher.
- **Preemptive arbitration:** needs kernel-level preemption or splitting every model into small steps; unneeded until measurements show the deadlines fail.

## Consequences and limits
- New crates `helios-audio`, `helios-speech`; new schema file; the committed LLM schema is unchanged.
- The arbiter orders submissions only; head-of-line blocking behind a long LLM prefill chunk is possible and is measured in S4 (mixed-load ITL and audio deadlines).
- Quantized speech weights are out of scope (bf16 first, as for the LLM).

## Verification plan
LLM manifests and goldens parse unchanged; partitions that do not fit are refused before Ready; arbiter ordering, deadlines and bounded queues on the stub (`[RUN]`); LLM-only path has no arbiter in a profile; then H100 runs of 0013/0014 and the S4 co-hosting budgets (S-A6).
