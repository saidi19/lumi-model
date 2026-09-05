from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_first_party_teacher_preflight import (
    CANARY_CONTRACT,
    MODEL_FILE_BYTES,
    MODEL_FILE_SHA256,
    NEXT_AUTHORIZATION_STATEMENT,
    build_first_party_teacher_preflight_report,
    build_first_party_teacher_preflight_verification,
    file_sha256,
    load_first_party_teacher_preflight_plan,
    validate_first_party_teacher_preflight_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-first-party-teacher-canary-preflight-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-first-party-teacher-canary-preflight-v0.1.json"
VERIFICATION = ROOT / (
    "evidence/lumi-4b-sft-first-party-teacher-canary-preflight-verification-v0.1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_plan_is_metadata_only_and_binds_exact_teacher() -> None:
    plan = load_first_party_teacher_preflight_plan(PLAN, repository_root=ROOT)
    assert plan["authorization"]["model_weight_download_authorized"] is False
    assert plan["authorization"]["external_api_authorized"] is False
    assert plan["model_metadata"]["model_file"]["size_bytes"] == MODEL_FILE_BYTES
    assert plan["model_metadata"]["model_file"]["sha256"] == MODEL_FILE_SHA256
    assert plan["next_gate"]["exact_authorization_statement"] == NEXT_AUTHORIZATION_STATEMENT


def test_canary_cross_product_is_exact_and_bounded() -> None:
    assert len(CANARY_CONTRACT["task_categories"]) == 5
    assert len(CANARY_CONTRACT["language_modes"]) == 4
    assert CANARY_CONTRACT["records_per_task_language_cell"] == 4
    assert CANARY_CONTRACT["candidate_records"] == 80
    assert CANARY_CONTRACT["split"] == {"train_records": 60, "validation_records": 20}
    assert CANARY_CONTRACT["maximum_retained_supervised_tokens"] == 24_000


def test_report_reconstructs_and_keeps_api_providers_fail_closed() -> None:
    plan = load_first_party_teacher_preflight_plan(PLAN, repository_root=ROOT)
    observed = _load(REPORT)
    rebuilt = build_first_party_teacher_preflight_report(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
    )
    assert rebuilt == observed
    validate_first_party_teacher_preflight_report(observed)
    providers = {row["provider"]: row for row in observed["provider_decisions"]}
    assert providers["openai_direct_api"]["decision"].startswith("ineligible_fail_closed")
    assert providers["azure_openai"]["decision"].startswith("ineligible_fail_closed")
    assert providers["self_hosted_qwen3_8b_gguf"]["decision"] == (
        "provisional_private_local_canary_only"
    )


def test_verification_reconstructs_and_outputs_are_raw_text_free() -> None:
    plan = load_first_party_teacher_preflight_plan(PLAN, repository_root=ROOT)
    report = _load(REPORT)
    expected = build_first_party_teacher_preflight_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(REPORT),
    )
    assert expected == _load(VERIFICATION)
    for path in (REPORT, VERIFICATION):
        serialized = path.read_text(encoding="utf-8")
        for forbidden in ('"messages"', '"assistant_response"', '"raw_text"'):
            assert forbidden not in serialized

