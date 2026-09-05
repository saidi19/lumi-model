from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_existing_candidate_registry_v18 import ADDITIONS, build_registry


def test_registry_v18_is_append_only() -> None:
    root = Path(__file__).resolve().parents[1]
    predecessor_path = root / "configs/lumi-4b-sft-source-registry-v0.17.json"
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    successor = build_registry(predecessor_path)
    assert successor["sources"][:18] == predecessor["sources"]
    assert successor["sources"][18:] == list(ADDITIONS)
    assert len({row["source_id"] for row in successor["sources"]}) == 22
    assert not any(row["production_eligible"] for row in successor["sources"])

