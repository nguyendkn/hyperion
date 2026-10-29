"""`scripts/fetch-model-fixtures.py --weights`: the checks that guard a 15 GB download, run against a local
server that serves the tiny sharded checkpoint, so no network and no large file is involved."""

from __future__ import annotations

import hashlib
import importlib.util
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "testdata" / "synthetic" / "qwen2-tiny-sharded"


def _load_script():
    spec = importlib.util.spec_from_file_location("fetch_model_fixtures", ROOT / "scripts" / "fetch-model-fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = _load_script()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the test output quiet
        pass

    def do_GET(self):
        name = self.path.rsplit("/", 1)[-1]
        path = FIXTURE / name
        self.server.requests.append((name, self.headers.get("Range")))
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        if name in self.server.corrupt:
            data = data[:-1] + bytes([data[-1] ^ 0xFF])
        start = 0
        if self.headers.get("Range"):
            start = int(self.headers["Range"].removeprefix("bytes=").rstrip("-"))
        body = data[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.requests, httpd.corrupt = [], set()
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()


def _spec():
    shards = sorted(FIXTURE.glob("model-*-of-*.safetensors"))
    return {
        "repo": "test/tiny",
        "revision": "r1",
        "weights": {s.name: {"sha256": hashlib.sha256(s.read_bytes()).hexdigest(), "bytes": s.stat().st_size} for s in shards},
    }


def _endpoint(server) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def test_the_files_are_downloaded_verified_and_announced(server, tmp_path, capsys):
    spec = _spec()
    fetch.fetch_weights("tiny", spec, tmp_path, _endpoint(server))
    for name in spec["weights"]:
        assert (tmp_path / "tiny" / name).read_bytes() == (FIXTURE / name).read_bytes()
        assert not (tmp_path / "tiny" / (name + ".part")).exists()
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("FETCH git_revision=") and "script_sha256=" in out.splitlines()[0]
    assert "FETCH model=tiny repo=test/tiny revision=r1" in out
    for name, info in spec["weights"].items():
        assert f"FETCH expect {name} bytes={info['bytes']} sha256={info['sha256']}" in out


def test_a_second_run_downloads_nothing(server, tmp_path):
    spec = _spec()
    fetch.fetch_weights("tiny", spec, tmp_path, _endpoint(server))
    server.requests.clear()
    fetch.fetch_weights("tiny", spec, tmp_path, _endpoint(server))
    assert server.requests == []


def test_a_file_that_does_not_match_its_pinned_hash_is_deleted_and_reported(server, tmp_path):
    spec = _spec()
    victim = next(iter(spec["weights"]))
    server.corrupt.add(victim)
    with pytest.raises(SystemExit, match="sha256"):
        fetch.fetch_weights("tiny", spec, tmp_path, _endpoint(server))
    assert not (tmp_path / "tiny" / victim).exists() and not (tmp_path / "tiny" / (victim + ".part")).exists()


def test_not_enough_disk_space_stops_before_any_request(server, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "free_bytes", lambda path: 1024)
    with pytest.raises(SystemExit, match="not enough disk space"):
        fetch.fetch_weights("tiny", _spec(), tmp_path, _endpoint(server))
    assert server.requests == []


def test_an_interrupted_download_resumes_from_its_part_file(server, tmp_path):
    spec = _spec()
    name = next(iter(spec["weights"]))
    target = tmp_path / "tiny"
    target.mkdir()
    half = (FIXTURE / name).read_bytes()[: spec["weights"][name]["bytes"] // 2]
    (target / (name + ".part")).write_bytes(half)
    fetch.fetch_weights("tiny", spec, tmp_path, _endpoint(server))
    assert (target / name).read_bytes() == (FIXTURE / name).read_bytes()
    assert (name, f"bytes={len(half)}-") in server.requests, "the request must ask only for the missing tail"


def test_a_model_without_listed_weights_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no `weights`"):
        fetch.fetch_weights("tiny", {"repo": "x", "revision": "y"}, tmp_path, "http://127.0.0.1:1")
