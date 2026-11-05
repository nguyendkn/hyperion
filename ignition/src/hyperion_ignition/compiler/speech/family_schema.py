"""Validation of a family manifest against the committed schema generated from the Rust structs."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_NAME = "family-manifest.v1.json"


def schema_path() -> Path:
    return Path(__file__).resolve().parents[5] / "specification" / "schema" / SCHEMA_NAME


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = json.loads(schema_path().read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_family_manifest(manifest: dict[str, Any]) -> None:
    errors = sorted(_validator().iter_errors(manifest), key=lambda e: list(map(str, e.absolute_path)))
    if errors:
        lines = [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in errors]
        raise ValueError("family manifest violates schema:\n  " + "\n  ".join(lines))
