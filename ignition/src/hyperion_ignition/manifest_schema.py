"""Validate manifests against the JSON Schema generated from the Rust structs.

The schema (`specification/schema/manifest.v1.json`) is the contract; this
module only loads and applies it. Ignition validates every manifest before it
is written, so an invalid package never exists.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_NAME = "manifest.v1.json"


class ManifestValidationError(ValueError):
    """The manifest does not satisfy the schema."""


def schema_path() -> Path:
    """Packaged copy (wheel install) first, then the repository checkout."""
    packaged = Path(__file__).resolve().parent / "schema" / SCHEMA_NAME
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parents[3] / "specification" / "schema" / SCHEMA_NAME


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = json.loads(schema_path().read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_manifest(manifest: dict[str, Any]) -> None:
    errors = sorted(_validator().iter_errors(manifest), key=lambda e: list(map(str, e.absolute_path)))
    if errors:
        lines = [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in errors]
        raise ManifestValidationError("manifest violates schema:\n  " + "\n  ".join(lines))
