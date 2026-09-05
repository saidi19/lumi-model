from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lumi.posttraining_safety_tool_tokens import (
    load_token_audit_plan,
    sha256_value,
    verify_candidate_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "lumi-4b-safety-tool-sft-token-audit-v0.1.json"
REPORT = ROOT / "evidence" / "lumi-4b-safety-tool-sft-token-audit-v0.1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_token_audit_plan_is_exact_local_and_non_promoting() -> None:
    plan = load_token_audit_plan(PLAN, repository_root=ROOT)
    evidence_path = ROOT / plan["bindings"]["candidate_evidence"]["path"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    verify_candidate_evidence(evidence, plan=plan)
    assert plan["limits"]["context_window"] == 4096
    assert plan["limits"]["expected_records"] == 320
    assert plan["authorization"]["network_access_authorized"] is False
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False


def test_token_audit_report_has_exact_counts_and_valid_internal_seal() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    seal = report.pop("report_sha256")
    assert seal == sha256_value(report)
    metrics = report["metrics"]
    assert metrics["accepted_records"] == 320
    assert metrics["rejected_records"] == 0
    assert metrics["overlength_records"] == 0
    assert metrics["serialized_tokens"] == 39_325
    assert metrics["supervised_tokens"] == 25_024
    assert metrics["by_split"]["train"]["supervised_tokens"] == 18_636
    assert metrics["by_split"]["validation"]["supervised_tokens"] == 6_388
    assert metrics["serialized_length"]["maximum"] == 256
    assert metrics["context_window"] == 4096
    assert report["owner_review_complete"] is False
    assert report["production_eligible"] is False
    assert report["production_promotions"] == 0
    assert report["raw_training_text_retained"] is False


def test_token_audit_report_and_bound_files_are_raw_free_and_hash_bound() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    report_text = REPORT.read_text(encoding="utf-8")
    assert "stop a frozen local test process" not in report_text
    assert "messages" not in json.loads(report_text)
    for binding in plan["bindings"].values():
        assert _sha256((ROOT / binding["path"]).resolve()) == binding["sha256"]

