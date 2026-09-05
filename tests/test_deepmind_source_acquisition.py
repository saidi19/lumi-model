from __future__ import annotations

from pathlib import Path

import pytest

from lumi.posttraining_deepmind_source_acquisition import (
    SOURCE_BYTES,
    SOURCE_OBJECTS,
    git_blob_sha1,
    load_acquisition_plan,
    raw_url,
    validate_expected_inventory,
)
from lumi.posttraining_math_replacement_preflight import DEEPMIND_OBJECTS

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/lumi-4b-sft-deepmind-source-acquisition-v0.1.json"


def test_exact_inventory_is_still_valid() -> None:
    validate_expected_inventory(DEEPMIND_OBJECTS)
    assert len(DEEPMIND_OBJECTS) == SOURCE_OBJECTS
    assert sum(item["size_bytes"] for item in DEEPMIND_OBJECTS) == SOURCE_BYTES


def test_git_blob_hash_uses_git_framing() -> None:
    assert git_blob_sha1(b"test content\n") == "d670460b4b4aece5915caf5c68d12f560a9fe3e4"


def test_raw_url_is_revision_pinned() -> None:
    url = raw_url("mathematics_dataset/modules/algebra.py")
    assert "/427f45075f84b8b9774950196ad63867ca20ffb3/" in url
    assert url.endswith("mathematics_dataset/modules/algebra.py")


def test_unsafe_inventory_path_is_rejected() -> None:
    changed = [dict(item) for item in DEEPMIND_OBJECTS]
    changed[0]["path"] = "../LICENSE"
    with pytest.raises(ValueError, match="inventory changed"):
        validate_expected_inventory(changed)


def test_exact_plan_loads_when_materialized() -> None:
    if not PLAN_PATH.exists():
        pytest.skip("DeepMind source-acquisition plan is not materialized")
    plan = load_acquisition_plan(PLAN_PATH, repository_root=ROOT)
    assert plan["limits"]["exact_source_objects"] == 30
    assert plan["authorization"]["canary_generation_authorized"] is False
