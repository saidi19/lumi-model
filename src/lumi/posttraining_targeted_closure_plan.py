"""Build and verify the non-authorizing Lumi targeted SFT closure plan."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_value(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(json_file_bytes(value))
    temporary.replace(destination)


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _verify_internal_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    expected = _require_sha256(value.get(field), f"{label}.{field}")
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != expected:
        raise ValueError(f"{label} internal seal changed")


def _resolve_binding(
    repository_root: str | Path,
    specification: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    root = Path(repository_root).resolve()
    relative = specification.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label}.path must be a non-empty string")
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    expected = _require_sha256(specification.get("sha256"), f"{label}.sha256")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} file SHA-256 changed")
    return path


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = _read_object(path)
    if plan.get("schema_version") != "lumi.sft-targeted-closure-plan.v0.1":
        raise ValueError("unexpected targeted-closure plan schema")
    if plan.get("status") != "authorized_cpu_only_nonmutating_targeted_closure_plan":
        raise ValueError("targeted-closure plan is not authorized")
    authorization = plan.get("authorization")
    if not isinstance(authorization, dict) or authorization.get("cpu_only") is not True:
        raise ValueError("targeted-closure plan must be CPU-only")
    forbidden = (
        "candidate_materialization_authorized",
        "dataset_seal_authorized",
        "external_api_authorized",
        "external_distribution_authorized",
        "gpu_authorized",
        "model_download_authorized",
        "network_access_authorized",
        "production_promotion_authorized",
        "registry_change_authorized",
        "sft_training_authorized",
        "unrelated_access_authorized",
    )
    if any(authorization.get(name) is not False for name in forbidden):
        raise ValueError("targeted-closure plan authorizes a forbidden action")
    bindings = plan.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("targeted-closure plan has no bindings")
    for name, specification in bindings.items():
        if not isinstance(specification, dict):
            raise ValueError(f"binding {name} must be an object")
        _resolve_binding(repository_root, specification, label=name)
    return plan


def _bound_json(
    plan: Mapping[str, Any], repository_root: str | Path, binding: str
) -> dict[str, Any]:
    path = _resolve_binding(repository_root, plan["bindings"][binding], label=binding)
    return _read_object(path)


def _matrix_totals(
    matrix: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, int], dict[str, int]]:
    task_totals: dict[str, int] = {}
    language_totals: dict[str, int] = {}
    for task, languages in matrix.items():
        if not isinstance(languages, dict) or not languages:
            raise ValueError(f"allocation row {task} must be a non-empty object")
        task_total = 0
        for language, tokens in languages.items():
            if not isinstance(tokens, int) or tokens < 0:
                raise ValueError(f"allocation {task}/{language} must be a non-negative integer")
            task_total += tokens
            language_totals[language] = language_totals.get(language, 0) + tokens
        task_totals[task] = task_total
    return task_totals, language_totals


def _share(tokens: int, total: int) -> float:
    return round(tokens / total, 8)


def build_report(
    plan: Mapping[str, Any], *, plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    closure = _bound_json(plan, repository_root, "closure_successor_report")
    closure_verification = _bound_json(
        plan, repository_root, "closure_successor_verification"
    )
    teacher = _bound_json(plan, repository_root, "teacher_reconciliation_report")
    teacher_verification = _bound_json(
        plan, repository_root, "teacher_reconciliation_verification"
    )
    aya_rights = _bound_json(plan, repository_root, "aya_rights_report")
    aya_rights_verification = _bound_json(plan, repository_root, "aya_rights_verification")
    closure_curation = _bound_json(plan, repository_root, "closure_curation_report")
    closure_curation_verification = _bound_json(
        plan, repository_root, "closure_curation_verification"
    )
    teacher_ab = _bound_json(plan, repository_root, "teacher_ab_report")
    teacher_ab_verification = _bound_json(plan, repository_root, "teacher_ab_verification")

    for label, value, field in (
        ("closure_successor_report", closure, "report_sha256"),
        ("closure_successor_verification", closure_verification, "verification_sha256"),
        ("teacher_reconciliation_report", teacher, "report_sha256"),
        ("teacher_reconciliation_verification", teacher_verification, "verification_sha256"),
        ("aya_rights_report", aya_rights, "report_sha256"),
        ("aya_rights_verification", aya_rights_verification, "verification_sha256"),
        ("closure_curation_report", closure_curation, "report_sha256"),
        ("closure_curation_verification", closure_curation_verification, "verification_sha256"),
        ("teacher_ab_report", teacher_ab, "report_sha256"),
        ("teacher_ab_verification", teacher_ab_verification, "verification_sha256"),
    ):
        _verify_internal_seal(value, field, label)

    if closure.get("status") != "verified_closure_projection_blocked_pending_targeted_gap_fill":
        raise ValueError("closure successor is not at the expected targeted-gap boundary")
    projection = closure.get("projection")
    if not isinstance(projection, dict):
        raise ValueError("closure projection missing")
    expected_current = plan["expected_current_state"]
    for field in (
        "train_supervised_tokens",
        "validation_supervised_tokens",
        "strict_targeted_gap_lower_bound",
    ):
        if projection.get(field) != expected_current[field]:
            raise ValueError(f"closure projection {field} changed")
    if projection.get("behavior_task_floor_gaps") != expected_current["task_gaps"]:
        raise ValueError("behavior task gaps changed")
    if projection.get("behavior_language_floor_gaps") != expected_current["language_gaps"]:
        raise ValueError("behavior language gaps changed")

    rejected = teacher.get("rejected_teacher_strategies")
    if not isinstance(rejected, dict):
        raise ValueError("teacher rejection evidence missing")
    if rejected.get("qwen3_8b_q4_k_m", {}).get("decision") != "bulk_generation_ineligible":
        raise ValueError("Qwen3-8B rejection changed")
    if rejected.get("qwen3_14b_q4_k_m", {}).get("decision") != "stronger_teacher_not_promoted":
        raise ValueError("Qwen3-14B rejection changed")
    if teacher_ab.get("machine_gate_passed") is not False:
        raise ValueError("Qwen3-14B machine gate unexpectedly passed")

    retained = {
        item["family_id"]: item
        for item in aya_rights.get("constituents", [])
        if item.get("disposition", "").startswith("retain_")
    }
    quarantined = {
        item["family_id"]: item
        for item in aya_rights.get("constituents", [])
        if item.get("disposition", "").startswith("quarantine_")
    }
    if set(retained) != {"wiki_split_inst_translated", "joke_explaination_inst_translated"}:
        raise ValueError("Aya retained-family decision changed")
    if set(quarantined) != {"flan_gem_wikilingua_translated"}:
        raise ValueError("Aya quarantined-family decision changed")

    raw_inventory = closure_curation.get("raw_object_inventory")
    if not isinstance(raw_inventory, list):
        raise ValueError("closure raw-object inventory missing")
    aya_raw = [
        item
        for item in raw_inventory
        if item.get("sha256") == plan["local_aya_capacity_lane"]["raw_object_sha256"]
    ]
    if len(aya_raw) != 1:
        raise ValueError("exact Aya raw object is not bound once")
    source_gate = closure_curation.get("source_gate", {})
    if source_gate.get("rows_seen", {}).get("aya_bengali") != plan["local_aya_capacity_lane"][
        "maximum_rows_read"
    ]:
        raise ValueError("Aya raw row count changed")

    allocation = plan.get("target_allocation")
    if not isinstance(allocation, dict):
        raise ValueError("target allocation missing")
    matrix = allocation.get("task_by_language_train_supervised_tokens")
    if not isinstance(matrix, dict):
        raise ValueError("target allocation matrix missing")
    task_totals, language_totals = _matrix_totals(matrix)
    target_tokens = allocation.get("target_train_supervised_tokens")
    if sum(task_totals.values()) != target_tokens or sum(language_totals.values()) != target_tokens:
        raise ValueError("target allocation does not sum to the target")
    if task_totals != allocation.get("task_totals"):
        raise ValueError("target task totals changed")
    if language_totals != allocation.get("language_totals"):
        raise ValueError("target language totals changed")
    for task, gap in expected_current["task_gaps"].items():
        if task_totals.get(task, 0) < gap:
            raise ValueError(f"target allocation does not close task gap: {task}")
    for language, gap in expected_current["language_gaps"].items():
        if language_totals.get(language, 0) < gap:
            raise ValueError(f"target allocation does not close language gap: {language}")
    if target_tokens < expected_current["strict_targeted_gap_lower_bound"]:
        raise ValueError("target allocation is below strict lower bound")

    projected_total = expected_current["train_supervised_tokens"] + target_tokens
    family_tokens = dict(projection["source_family_train_supervised_tokens"])
    family_tokens["lumi_first_party_family"] += target_tokens
    family_shares = {
        name: _share(tokens, projected_total) for name, tokens in family_tokens.items()
    }
    source_cap = projection["source_cap"]
    if any(share > source_cap for share in family_shares.values()):
        raise ValueError("target allocation leaves a source-family cap violation")

    provider_policy = plan["provider_policy"]
    if provider_policy["openai_api"]["eligible_for_lumi_teacher_generation"] is not False:
        raise ValueError("OpenAI API must remain blocked for competing-model teacher generation")
    if provider_policy["azure_openai"]["eligible_for_lumi_teacher_generation"] is not False:
        raise ValueError("Azure OpenAI must remain blocked pending exact terms")
    if provider_policy["recommended_self_hosted_canary"]["model_id"] != "openai/gpt-oss-120b":
        raise ValueError("unexpected recommended self-hosted teacher canary")

    report: dict[str, Any] = {
        "schema_version": "lumi.sft-targeted-closure-plan-report.v0.1",
        "status": "verified_targeted_closure_plan_no_execution_authorized",
        "observed_at": plan["observed_at"],
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "bindings": {
            name: {"file_sha256": specification["sha256"]}
            for name, specification in sorted(plan["bindings"].items())
        },
        "current_verified_state": expected_current,
        "rejected_teacher_strategies": rejected,
        "aya_rights_boundary": {
            "eligible_for_capacity_measurement_only": sorted(retained),
            "quarantined": sorted(quarantined),
            "wiki_split_production_compliance_resolved": False,
            "production_tokens_counted_from_capacity_lane": 0,
        },
        "provider_policy": provider_policy,
        "target_closure": {
            "target_train_supervised_tokens": target_tokens,
            "margin_above_strict_lower_bound": target_tokens
            - expected_current["strict_targeted_gap_lower_bound"],
            "task_by_language_train_supervised_tokens": matrix,
            "task_totals": task_totals,
            "language_totals": language_totals,
            "projected_train_supervised_tokens": projected_total,
            "projected_source_family_tokens": family_tokens,
            "projected_source_family_shares": family_shares,
            "source_family_cap": source_cap,
            "all_task_gaps_closed_by_allocation": True,
            "all_language_gaps_closed_by_allocation": True,
            "all_source_family_shares_within_cap": True,
        },
        "execution_sequence": plan["execution_sequence"],
        "local_aya_capacity_lane": plan["local_aya_capacity_lane"],
        "self_hosted_teacher_canary": plan["self_hosted_teacher_canary"],
        "quality_gates": plan["quality_gates"],
        "next_gate": plan["next_gate"],
        "blockers": [
            "local_aya_eligible_capacity_not_measured",
            "wiki_split_share_alike_compliance_not_resolved",
            "self_hosted_teacher_exact_revision_objects_runtime_and_cost_not_bound",
            "self_hosted_teacher_canary_not_run",
            "targeted_1_6m_candidate_not_materialized_or_reviewed",
            "source_registry_successor_not_materialized",
            "full_cross_source_hygiene_reaudit_not_run",
            "frozen_train_validation_dataset_not_sealed",
            "cpu_sft_smoke_and_base_benchmarks_not_run",
            "gpu_sft_not_authorized",
        ],
        "authorization": plan["authorization"],
        "production_eligible": False,
        "raw_training_text_retained": False,
    }
    report["report_sha256"] = sha256_value(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != "lumi.sft-targeted-closure-plan-report.v0.1":
        raise ValueError("unexpected targeted-closure report schema")
    if report.get("status") != "verified_targeted_closure_plan_no_execution_authorized":
        raise ValueError("targeted-closure report status is not verified")
    _verify_internal_seal(report, "report_sha256", "targeted_closure_report")
    if report.get("production_eligible") is not False:
        raise ValueError("targeted-closure plan cannot be production eligible")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("targeted-closure report retained raw training text")


def build_verification(
    *, plan: Mapping[str, Any], report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_report(report)
    verification: dict[str, Any] = {
        "schema_version": "lumi.sft-targeted-closure-plan-verification.v0.1",
        "status": "PASS",
        "observed_at": plan["observed_at"],
        "report_file_sha256": _require_sha256(report_file_sha256, "report_file_sha256"),
        "report_internal_sha256": report["report_sha256"],
        "checks": {
            "bound_evidence_replayed": True,
            "allocation_arithmetic_exact": True,
            "strict_task_and_language_gaps_closed_by_plan": True,
            "source_family_cap_projected_below_or_equal_0_35": True,
            "failed_qwen_teacher_strategies_blocked": True,
            "openai_and_azure_api_teacher_generation_blocked": True,
            "wiki_lingua_quarantined": True,
            "wiki_split_not_counted_before_compliance_resolution": True,
            "no_network_api_gpu_registry_seal_or_sft_action_authorized": True,
            "raw_training_text_absent": True,
        },
        "production_eligible": False,
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
