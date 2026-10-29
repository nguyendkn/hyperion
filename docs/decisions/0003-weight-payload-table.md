# 0003: weights live in safetensors payload files listed by the manifest

Status: accepted (user, 2026-10-29). Plan: phase 03 (spec before freeze), open decision 6 ("weights as safetensors payload files vs custom framing", recommendation: safetensors), phase 06 (signing) and phase 13 (sharding at load).

## Context
A checkpoint is often split over several files (Qwen2.5-7B: four shards plus `model.safetensors.index.json`). The manifest recorded byte offsets inside one `model.safetensors`, and the runtime read that one file, so a sharded model could not be described. The first idea, a side file (`index.json`) beside the package, would let a file outside the signed manifest decide which bytes are read from which file: a trust hole once the manifest is signed. Adding only a `file` string per layer would have to be replaced when phase 03 adds payload identity (sha256 per file), changing the spec twice before it freezes.

## Decision
- Weights stay in safetensors files (open decision 6: no custom framing).
- The manifest has a payload table, `payloads: [{ file, byte_length, sha256 }]`, in the signed body. Each layer carries `payload`, an index into the table (0 when absent), and keeps `byte_offset` and `byte_length`, which count from the start of the data section of that payload's file.
- Compatibility: a manifest without a table has one payload, `model.safetensors`, with nothing recorded about it (every older manifest, and a header-only compile). An empty table and payload 0 are not written, so existing golden manifests do not change.
- `file` is a bare file name that opens an ordinary file on Linux and on Windows: not empty, at most 255 bytes, not `.` or `..`, no path separator, drive or stream colon or NUL, not ending in a dot or a space, and not a Windows device name (`CON`, `PRN`, `AUX`, `NUL`, `COM0`-`COM9`, `LPT0`-`LPT9`, `CONIN$`, `CONOUT$`, in any case and with any extension, so `NUL.safetensors` is refused). The rule is `hyperion_specification::v1::is_bare_file_name`, mirrored in Ignition, and both test against the same example list (`specification/fixtures/bare-file-names.json`).
- Ignition computes size and sha256 of each real weights file when it compiles. A sharded checkpoint is read through `model.safetensors.index.json` and each shard's own header; the index and the headers must agree, and shard names must be bare file names. Ignition refuses a source that has both an index and a single file, a missing shard, or an index that disagrees with a header. A header-only compile of a sharded model is refused, because the table needs each shard's hash.
- The runtime checks that each payload file exists and that its size equals the table; it does not hash the files yet. Verifying the sha256 (with a cache of "already verified" so 15 GB is not hashed on every start) belongs to phase 06. No flag to skip that check will be added later.

## Why over the alternatives
- A side `index.json` (what Hugging Face writes): not covered by the signature, so a modified index would make the runtime read the wrong bytes without detection.
- A `file` string per layer only: replaced soon by a payload table; two spec changes before the freeze.
- A custom weights container: more parser surface for no gain, and Hugging Face tooling already writes and reads safetensors.

## Consequences
- Phase 13: when weights are sharded at load, a rank reads slices of tensors that lie across several files; the table says where each lives.
- Phase 06 can sign the manifest and rely on the hashes for the weights; the location of a payload on a remote store is a separate matter and stays outside the signed body.
- The JSON schema gained the `payloads` property, `payload` on layers, and patterns for `file` and `sha256`; `manifest.v1.json` was regenerated.
- Tested with a tiny random Qwen2 saved by Hugging Face's `save_pretrained(max_shard_size=...)` (5 shards, `testdata/synthetic/qwen2-tiny-sharded`), the real Qwen2.5-7B index and shard headers (339 tensors over 4 shards), and a Rust test that reads every tensor of the fixture from its own shard. Not run on a GPU: the device upload of a sharded model.
