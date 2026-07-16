from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from lumi.scaling_pilot import (
    load_scaling_pilot_plan,
    plan_scaling_pilot,
    validate_scaling_pilot_inputs,
    write_scaling_report,
)

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "configs" / "scaling-pilot-1b-v0.1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _copy_plan(tmp_path: Path) -> tuple[Path, dict]:
    configs = tmp_path / "configs"
    configs.mkdir()
    shutil.copy2(ROOT / "configs" / "lumi-1b.json", configs / "lumi-1b.json")
    shutil.copy2(ROOT / "configs" / "lumi-4b.json", configs / "lumi-4b.json")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan_path = configs / PLAN.name
    return plan_path, plan


def test_canonical_1b_plan_freezes_architecture_and_stages() -> None:
    plan = load_scaling_pilot_plan(PLAN)

    assert plan["model"]["parameters"] == 1_050_981_888
    assert plan["training"]["tokens_per_step"] == 32_768
    assert plan["stages"][-1] == {
        "name": "unique_corpus",
        "processed_tokens": 1_400_012_800,
        "repeated_tokens": 16_384,
        "steps": 42_725,
        "target_tokens": 1_399_996_416,
    }
    assert plan["approval"] == {"full_4b_training": False, "paid_gpu_launch": False}


def test_cost_projection_uses_measured_throughput_and_explicit_rate() -> None:
    report = plan_scaling_pilot(
        plan_path=PLAN,
        tokens_per_second=10_000,
        hourly_rate=4.29,
        fixed_overhead_hours=0.25,
        credit_balance=7_419.56,
    )
    final = report["stages"][-1]
    expected_hours = final["processed_tokens"] / 10_000 / 3_600
    expected_cost = (expected_hours + 0.25) * 4.29

    assert report["status"] == "costed"
    assert final["projection"]["compute_hours"] == pytest.approx(expected_hours)
    assert final["projection"]["estimated_cost"] == pytest.approx(expected_cost)
    assert final["projection"]["credit_remaining"] == pytest.approx(7_419.56 - expected_cost)
    assert report["pricing"]["projected_cost_per_billion_tokens"] == pytest.approx(
        1e9 / 10_000 / 3_600 * 4.29
    )


def test_plan_rejects_tampered_stage_arithmetic(tmp_path: Path) -> None:
    plan_path, plan = _copy_plan(tmp_path)
    plan["stages"][-1]["repeated_tokens"] += 1
    _write_json(plan_path, plan)

    with pytest.raises(ValueError, match="repeated_tokens"):
        load_scaling_pilot_plan(plan_path)


def test_local_input_validation_pins_all_source_and_baseline_evidence(tmp_path: Path) -> None:
    plan_path, plan = _copy_plan(tmp_path)
    source = {
        "output": {"sha256": "a" * 64, "text_utf8_bytes": 100},
        "statistics": {"documents": 10},
        "status": "complete",
    }
    heldout = {
        "build_fingerprint": "b" * 64,
        "leakage": {"validation_documents": 2},
        "selection": {"strata": [{"id": "one"}]},
        "statistics": {"tokens": 4096},
    }
    token_count = {
        "tokenizers": {
            "65k": {
                "artifact": {"vocab_size": 65_536},
                "overall": {"tokens": 1_500_000_000},
            }
        }
    }
    baseline = {
        "artifacts": {"heldout_build_fingerprint": "b" * 64},
        "loss": {"mean": 2.0, "perplexity": 7.38905609893065},
        "runtime": {"model_parameters": 150_625_536},
    }
    source_path = tmp_path / "source.json"
    heldout_path = tmp_path / "heldout.json"
    token_count_path = tmp_path / "tokens.json"
    baseline_path = tmp_path / "baseline.json"
    ledger_path = tmp_path / "selection.jsonl"
    _write_json(source_path, source)
    _write_json(heldout_path, heldout)
    _write_json(token_count_path, token_count)
    _write_json(baseline_path, baseline)
    ledger_path.write_text('{"id":"one"}\n{"id":"two"}\n', encoding="utf-8")

    plan["data"]["source_sample"].update(
        {
            "accepted_text_bytes": 100,
            "documents": 10,
            "manifest_sha256": _sha256(source_path),
            "observed_tokens": 1_500_000_000,
            "sample_sha256": "a" * 64,
            "token_count_report_sha256": _sha256(token_count_path),
        }
    )
    plan["data"]["heldout"].update(
        {
            "build_fingerprint": "b" * 64,
            "documents": 2,
            "manifest_sha256": _sha256(heldout_path),
            "selection_ledger_sha256": _sha256(ledger_path),
            "tokens": 4096,
        }
    )
    plan["data"]["selection"]["source_strata"] = 1
    plan["evaluation"]["baseline_150m"].update(
        {
            "heldout_loss": 2.0,
            "heldout_perplexity": 7.38905609893065,
            "report_sha256": _sha256(baseline_path),
        }
    )
    _write_json(plan_path, plan)

    loaded = load_scaling_pilot_plan(plan_path)
    result = validate_scaling_pilot_inputs(
        loaded,
        source_manifest_path=source_path,
        heldout_manifest_path=heldout_path,
        heldout_selection_ledger_path=ledger_path,
        token_count_report_path=token_count_path,
        baseline_evaluation_report_path=baseline_path,
    )

    assert result["status"] == "pass"
    assert result["heldout_documents"] == 2
    assert result["baseline_heldout_loss"] == 2.0


def test_scaling_report_write_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    report = plan_scaling_pilot(plan_path=PLAN)
    output = tmp_path / "report.json"

    assert write_scaling_report(output, report) == output
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError):
        write_scaling_report(output, report)
