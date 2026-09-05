from __future__ import annotations

from lumi.posttraining_prosocial_dialog_source_promotion_v19 import (
    EXPECTED_SUCCESSOR_REGISTRY_BYTES,
    EXPECTED_SUCCESSOR_REGISTRY_SHA256,
    json_file_bytes,
    project_successor_registry,
    source_metadata,
)
from lumi.posttraining_prosocial_dialog_safety_curation_v1 import file_sha256
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_registry_projection_is_exact_append_only() -> None:
    predecessor_path = ROOT / "configs/lumi-4b-sft-source-registry-v0.18.json"
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    successor = project_successor_registry(predecessor)
    assert successor["sources"][:-1] == predecessor["sources"]
    assert successor["sources"][-1] == source_metadata()
    encoded = json_file_bytes(successor)
    assert len(encoded) == EXPECTED_SUCCESSOR_REGISTRY_BYTES
    assert hashlib.sha256(encoded).hexdigest() == EXPECTED_SUCCESSOR_REGISTRY_SHA256


def test_predecessor_registry_binding_is_current() -> None:
    assert file_sha256(ROOT / "configs/lumi-4b-sft-source-registry-v0.18.json") == (
        "b9ba6a6c551fd7e4da76bc6e0df5e347d0fcdf47f31e12599e4ddc4665dab01d"
    )
