"""The Rust arch -> tensor_format table (helios-hal) and the Python hardware
recipes' `supported_formats` must agree. The Rust side owns the table and
commits a JSON dump (`helios/crates/helios-hal/tests/data/format-table.json`,
checked by a helios-hal test); this test compares that dump with the recipes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperion_ignition.hardware.registry import build_recipe

DUMP = Path(__file__).resolve().parents[2] / "helios/crates/helios-hal/tests/data/format-table.json"
TABLE = json.loads(DUMP.read_text(encoding="utf-8"))


@pytest.mark.parametrize("arch", ["sm_90", "sm_120"])
def test_recipe_formats_equal_the_rust_table(arch):
    assert list(build_recipe(arch).supported_formats) == TABLE[arch]


def test_hopper_has_no_fp4_on_either_side():
    assert "fp4" not in TABLE["sm_90"]
    assert "fp4" not in build_recipe("sm_90").supported_formats


def test_every_recipe_arch_exists_in_the_rust_table():
    # sm_0 is the stub, which has no Ignition recipe by design.
    assert {"sm_90", "sm_120"} <= set(TABLE)
