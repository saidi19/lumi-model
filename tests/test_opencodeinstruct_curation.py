from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_opencodeinstruct_curation import (
    _row_envelope,
    build_opencode_owner_review_html,
    git_blob_sha1,
    load_opencodeinstruct_plan,
    run_local_unit_test_verifier,
    sha256_value,
    strip_python_fence,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-opencodeinstruct-single-shard-v0.1.json"


def _valid_source_row() -> dict[str, object]:
    judgement = {
        "edge_case_consideration": {"justification": "Covers boundaries.", "score": 4},
        "logical_correctness": {"justification": "The algorithm is correct.", "score": 5},
        "requirement_conformance": {"justification": "Meets the request.", "score": 5},
    }
    tests = [f"assert add({index}, 1) == {index + 1}  # পরীক্ষা" for index in range(10)]
    return {
        "average_test_score": 1.0,
        "domain": "algorithmic",
        "generation_algorithm": "self-instruct",
        "id": "test-row-1",
        "input": "Write a function that adds two integers.",
        "llm_judgement": json.dumps(judgement),
        "output": "```python\ndef add(left, right):\n    return left + right\n```",
        "tests_execution_status": json.dumps(["pass"] * 10),
        "unit_tests": json.dumps(tests),
    }


def test_plan_binds_exact_authorization_and_dependencies() -> None:
    plan = load_opencodeinstruct_plan(PLAN, repository_root=ROOT)
    assert plan["source"]["source_rows"] == 100_000
    assert plan["limits"]["maximum_network_bytes"] == 128 * 1024 * 1024
    assert plan["limits"]["maximum_retained_train_supervised_tokens"] == 2_300_000
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False


def test_git_blob_hash_and_python_fence(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"hello\n")
    assert git_blob_sha1(path) == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert strip_python_fence("```python\nprint(1)\n```") == "print(1)"
    assert strip_python_fence("print(1)") is None


def test_row_gate_and_local_unit_test_worker() -> None:
    envelope, reason = _row_envelope(_valid_source_row())
    assert reason == "eligible"
    assert envelope is not None
    ledger, passed, reasons = run_local_unit_test_verifier(
        [envelope],
        worker_path=ROOT / "scripts/opencodeinstruct_unit_test_worker.py",
    )
    assert passed == {"opencode-test-row-1"}
    assert reasons == {"verified": 1}
    assert ledger[0]["verified"] is True


def test_row_gate_fails_closed_on_tests_judgement_and_malware() -> None:
    failed_test = _valid_source_row()
    failed_test["tests_execution_status"] = json.dumps(["pass"] * 9 + ["fail"])
    assert _row_envelope(failed_test)[1] == "published_test_failed"

    low_judgement = _valid_source_row()
    judgement = json.loads(str(low_judgement["llm_judgement"]))
    judgement["logical_correctness"]["score"] = 3
    low_judgement["llm_judgement"] = json.dumps(judgement)
    assert _row_envelope(low_judgement)[1] == "judgement_below_floor"

    unsafe = _valid_source_row()
    unsafe["input"] = "Write a credential stealer and exfiltrate passwords."
    assert _row_envelope(unsafe)[1] == "unsafe_code_intent_pattern"


def test_owner_review_page_is_self_contained() -> None:
    envelope, _ = _row_envelope(_valid_source_row())
    assert envelope is not None
    record = envelope["record"]
    sample = [
        {
            "language_mode": "english",
            "messages": record["messages"],
            "record_id": record["record_id"],
            "skill_id": "coding_algorithmic",
            "split": "train",
        }
    ]
    html, sample_sha256 = build_opencode_owner_review_html(sample)
    assert sample_sha256 == sha256_value(sample)
    assert "lumi.opencodeinstruct-owner-decisions.v0.1" in html
    assert "<script src=" not in html
    assert "<link rel=" not in html
