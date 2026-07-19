from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from lumi.aws_s3_production import (
    _data_lifecycle,
    _log_lifecycle,
    _production_tags,
    build_production_acquisition_plan,
    production_bucket_names,
)

ROOT = Path(__file__).parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_plan(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    fixture_root = tmp_path / "repo"
    (fixture_root / "configs").mkdir(parents=True)
    (fixture_root / "src" / "lumi").mkdir(parents=True)
    contract = json.loads(
        (ROOT / "configs" / "production-acquisition-v0.1.json").read_text(encoding="utf-8")
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_bytes(b'{"fixture":"tokenizer"}\n')
    tokenizer_sha256 = _sha256(tokenizer_path)
    ledger_path = tmp_path / "training-exclusion-ledger.jsonl"
    ledger_path.write_text('{"content_sha256":"a"}\n{"content_sha256":"b"}\n', encoding="utf-8")

    for name, pin in contract["inputs"].items():
        source = ROOT / pin["path"]
        destination = fixture_root / pin["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name == "tokenizer_decision":
            value = json.loads(source.read_text(encoding="utf-8"))
            value["baseline"]["tokenizer_json_sha256"] = tokenizer_sha256
            destination.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        else:
            shutil.copyfile(source, destination)
        pin["sha256"] = _sha256(destination)

    for relative in (
        "src/lumi/aws_s3_production.py",
        "src/lumi/aws_s3_canary.py",
    ):
        shutil.copyfile(ROOT / relative, fixture_root / relative)
    contract["external_artifacts"]["frozen_tokenizer"].update(
        {"bytes": tokenizer_path.stat().st_size, "sha256": tokenizer_sha256}
    )
    contract["external_artifacts"]["training_exclusion_ledger"].update(
        {
            "bytes": ledger_path.stat().st_size,
            "entries": 2,
            "sha256": _sha256(ledger_path),
        }
    )
    contract_path = fixture_root / "configs" / "production-acquisition-v0.1.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return fixture_root, contract_path, tokenizer_path, ledger_path


def test_production_plan_pins_authorization_sources_and_limits(tmp_path: Path) -> None:
    root, contract, tokenizer, ledger = _fixture_plan(tmp_path)
    plan = build_production_acquisition_plan(
        contract,
        repository_root=root,
        tokenizer_path=tokenizer,
        training_exclusion_ledger=ledger,
    )
    assert plan["decision"] == {
        "full_4b_training": "not_authorized",
        "gpu_compute": "not_authorized",
        "production_s3": "authorized",
        "source_payload_transfer": "authorized",
        "status": "ready_for_s3_provisioning",
    }
    assert sum(row["candidate_floor_tokens"] for row in plan["categories"]) == 120_000_000_000
    assert plan["source_inventory"] == {
        "compressed_source_bytes": 499_817_285_384,
        "data_files": 758,
        "repositories": 23,
    }


def test_production_plan_rejects_training_authorization(tmp_path: Path) -> None:
    root, contract_path, tokenizer, ledger = _fixture_plan(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["authorization"]["model_training"] = True
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="must not authorize GPU or model training"):
        build_production_acquisition_plan(
            contract_path,
            repository_root=root,
            tokenizer_path=tokenizer,
            training_exclusion_ledger=ledger,
        )


def test_production_bucket_names_are_stable_and_hide_account_id() -> None:
    fingerprint = "a" * 64
    first = production_bucket_names("123456789012", fingerprint)
    second = production_bucket_names("123456789012", fingerprint)
    assert first == second
    assert first[0].startswith("lumi-corpus-prod-")
    assert first[1].startswith("lumi-corpus-prod-audit-")
    assert all("123456789012" not in name and len(name) <= 63 for name in first)


def test_production_lifecycle_and_tags_are_not_canary_cleanup() -> None:
    data = _data_lifecycle("lumi-corpus/v0.1")
    logs = _log_lifecycle()
    tags = {
        row["Key"]: row["Value"]
        for row in _production_tags("AdaptiveCorpusAcquisition", "b" * 64)["TagSet"]
    }
    assert any(
        row.get("AbortIncompleteMultipartUpload", {}).get("DaysAfterInitiation") == 1
        for row in data["Rules"]
    )
    assert logs["Rules"][0]["Expiration"]["Days"] == 400
    assert tags["Environment"] == "Production"
    assert tags["DeleteAfterCanary"] == "false"
