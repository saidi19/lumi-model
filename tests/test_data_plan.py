from __future__ import annotations

import json
import math
from pathlib import Path


def test_candidate_data_mix_is_complete_and_sums_to_80b() -> None:
    path = Path(__file__).parents[1] / "configs" / "data-mix-v0.1.json"
    with path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    mixture = plan["mixture"]
    assert math.isclose(sum(item["weight"] for item in mixture), 1.0)
    assert sum(item["target_tokens_at_80b"] for item in mixture) == 80_000_000_000
    for item in mixture:
        assert item["source_components"]
        assert item["target_tokens_at_80b"] == round(item["weight"] * 80_000_000_000)
