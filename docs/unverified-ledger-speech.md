# Verification ledger, speech section

Separate file from `docs/unverified-ledger.md` on purpose: both tracks append, and two appends at the end of one file conflict on every rebase. Same tags as `docs/testing-layers.md`. Rows are append-only; a row is closed by a new row, not by editing.

## Speech run S0 and S1 (2026-11-05)

| Item | Tag | Evidence |
|---|---|---|
| Model selection (STT Qwen3-ASR-1.7B 80 points, TTS VieNeu-TTS v3 Turbo 81 points, no NAR TTS candidate passes) | decision, not a test | ADR 0013, 0014; `docs/research/{stt,tts}-candidates-20261105.md`; owner may override |
| Licences of Qwen3-ASR, VieNeu, MOSS codec | VERIFIED from raw files at pinned commits for the items listed in `verification-20261105.md`; **NOT VERIFIED**: Viet74K, `speaker_encoder.onnx`/`denoiser.onnx`, MOSS GitHub LICENSE text, `.npz` loading mode | `docs/research/verification-20261105.md` |
| `sea-g2p` data tables fail the licence policy; nothing from it is used | VERIFIED (absence of provenance, CC-BY-SA lineage of lists) | same file; ADR 0014 amendment |
| VieNeu training data and commercial rights | **OPEN**, written statement from the author requested; status `provisional, not for production`, manifest `production_approved = false` | ADR 0014 |
| Qwen3-ASR MRoPE reduces to plain RoPE for audio + text | VERIFIED by reading the reference code at the pin; the numerical equality is still to be checked by the encoder/decoder oracle `[NEEDS-GPU]` | ADR 0013 amendment; request R2 closed |
| Qwen3-ASR Vietnamese WER 5.55 (paper) / 6.15 (third party) | third-party figures, **not measured by us** | ADR 0013 |
| VieNeu Vietnamese quality | no published figure; our round-trip measurement is gate 1, `[NEEDS-GPU]` | ADR 0014 |
| Embedding-prefill hook (StepPlan `Input::Embeddings`) and its prefix-cache identity rule | request R1 sent to the LLM track; blocks decode end to end for both families | `docs/requests-to-llm-track.md` |
| `helios-audio`: WAV round trip, FFT against naive DFT (lengths 1 to 400 incl. Bluestein), resampler (in-band tone kept, 12 kHz tone rejected when going to 16 kHz, deterministic), Slaney mel scale, filter-bank shape, tone lands in the expected mel band, frame count | `[RUN]` 15 tests | `cargo test -p helios-audio` |
| `helios-audio` features equal the reference features on fixed clips | `[NEEDS-GPU]` no, **not run**: needs the pinned Python reference and a clip (S1.2) | open |
| `helios-speech`: partition planner refuses before Ready, arbiter orders by class then deadline then arrival, per-model bounded queue (429 naming the model), registry family checks | `[RUN]` 7 tests; the arbiter is not wired to any server | `cargo test -p helios-speech` |
| Audio API logic: multipart parser (binary data with delimiter look-alikes), transcription and speech request validation (unsupported field = 400, compressed format refused), srt/vtt/verbose_json rendering, OpenAI-shaped errors | `[RUN]` 17 tests incl. the above; no HTTP handler exists yet | `cargo test -p helios-speech` |
| Family manifest: LLM fixtures parse unchanged through `parse_any`, speech manifests strict, schema committed and drift-tested | `[RUN]` 6 new + 27 existing tests | `cargo test -p hyperion-specification --features schema` |
| Whole workspace, default `stub` | `[RUN]` 320 tests passed, 0 failing suites, at `speech/main` (Windows). pytest not run | `cargo test --workspace` |
| Everything on a GPU (encoder, codec, kernels, graphs, WER, round-trip WER, RTF, co-hosting regression, arbiter under load) | `[NEEDS-GPU]` not started | parts S2 to S4 |
