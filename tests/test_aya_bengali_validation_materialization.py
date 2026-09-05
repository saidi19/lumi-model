from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from lumi.posttraining_aya_validation import EXPECTED_PROJECTION
from lumi.posttraining_aya_validation_materialization import (
    MAXIMUM_PRIVATE_WRITE_BYTES,
    OUTPUT_FILENAMES,
    load_exact_materialization_inputs,
    resolve_private_root,
    validate_materialization_receipt,
    verify_materialized_output,
)

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = resolve_private_root(ROOT)
IMPLEMENTATION = ROOT / "src/lumi/posttraining_aya_validation_materialization.py"
RUNNER = ROOT / "scripts/materialize_aya_validation_split.py"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_exact_private_aya_split_rebuilds_and_verifies() -> None:
    verification = verify_materialized_output(
        PRIVATE_ROOT,
        repository_root=ROOT,
        implementation_path=IMPLEMENTATION,
        runner_path=RUNNER,
    )
    assert verification["total_private_bytes"] == 5_480_313
    assert verification["total_private_bytes"] < MAXIMUM_PRIVATE_WRITE_BYTES
    assert verification["receipt_sha256"] == (
        "a4507bd0dbe7b1d335f4602205f4d62b1368efe9c0f8ce8f6f31606e667f8932"
    )


def test_materialized_manifest_is_raw_text_free_and_exact() -> None:
    manifest = _read_jsonl(PRIVATE_ROOT / OUTPUT_FILENAMES["manifest"])
    assert len(manifest) == 1_478
    assert Counter(row["split"] for row in manifest) == {"train": 1330, "validation": 148}
    assert sum(row["supervised_tokens"] for row in manifest) == 577_453
    assert set().union(*(row.keys() for row in manifest)) == {
        "attribution_row_sha256",
        "masked_tokens",
        "record_id",
        "schema_version",
        "serialized_tokens",
        "split",
        "supervised_tokens",
    }


def test_receipt_preserves_nonproduction_boundary() -> None:
    receipt = _read_json(PRIVATE_ROOT / OUTPUT_FILENAMES["receipt"])
    validate_materialization_receipt(receipt)
    assert receipt["projection"] == EXPECTED_PROJECTION
    assert receipt["production"] == {
        "dataset_sealed": False,
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "production_sft_authorized": False,
        "source_registry_changed": False,
    }
    assert receipt["raw_training_text_in_receipt"] is False


def test_materialization_inputs_remain_exactly_hash_bound() -> None:
    plan, report, rows, attribution, _template, _tokenizer = (
        load_exact_materialization_inputs(ROOT)
    )
    assert report["report_sha256"] == (
        "48b21e9f5b1ce07290beab67c3f77a33572a0ab887b871e19fcebc5b99fc8628"
    )
    assert len(rows) == len(attribution) == plan["limits"]["candidate_records"] == 1_478
