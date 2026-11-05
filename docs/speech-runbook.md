# Speech track runbook (data, environments, oracles)

## Where things live (outside the repo, survive scratch cleanup)
- `%LOCALAPPDATA%\hyperion\data\speech\` : `librispeech_test_clean.parquet` (sha256 `7113aa4c...17df0`, openslr/librispeech_asr rev `71cacbfb`, CC-BY-4.0) and `fleurs_vi_test.parquet` (sha256 `65325be1...c77`, google/fleurs rev `70bb2e84`, CC-BY-4.0). Download URLs: the `resolve/<rev>/clean/test/0000.parquet` and `resolve/<rev>/parquet-data/vi_vn/test-00000-of-00001.parquet` paths of those datasets; verify the sha256 (also printed in `testdata/oracle/*.stt-clips.json`).
- `...\data\speech\clips\<lang>\<id>.{flac,wav}` : the 20 + 20 oracle clips, extracted byte-exact; `python scripts/gen-oracle-stt-clips.py` recreates them from the parquets (rule declared in the script, ids and per-file sha256 committed in `testdata/oracle/qwen3-asr-1.7b.stt-clips.json`).
- Weights: `python scripts/fetch-speech-fixtures.py qwen3-asr-1.7b --weights` (sha256-pinned, `testdata/models/`, gitignored).
- Reference venv: `%LOCALAPPDATA%\hyperion\venvs\oracle-qwen3asr`, recreated from `scripts/oracle-env/qwen3-asr.requirements.lock.txt`. The older `venvs\oracle` (transformers 5.x) cannot load the original checkpoint format and is not used for this model.
- Pods: copy the two parquets and the clips directory (or `PACKAGE_STORE` once set); the clips are verified by sha256 before use.

## Oracle
`PYTHONIOENCODING=utf-8 <venv python> -I scripts/gen-oracle-stt.py` writes `testdata/oracle/qwen3-asr-1.7b.stt.json` with the tolerances inside (method in the script header). Run it on an otherwise idle machine: the fp32 model takes about 9.4 GB of RAM, and a concurrent large job was killed for memory once (2026-11-05).
