"""Fail-closed exact boundary preflight for the self-hosted gpt-oss teacher."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_candidate_mix_preflight import file_sha256, sha256_value

PLAN_SCHEMA = "lumi.sft-gpt-oss-teacher-preflight-plan.v0.1"
REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-preflight-report.v0.1"
VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-teacher-preflight-verification.v0.1"
MODEL_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
MODEL_BYTES = 65_276_857_840
RUNTIME_BYTES = 11_763_193_277
TOTAL_NETWORK_BYTES = MODEL_BYTES + RUNTIME_BYTES
RUNTIME_DIGEST = (
    "sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8"
)
EXPECTED_BINDINGS = frozenset(
    {
        "aya_capacity_plan",
        "aya_capacity_report",
        "aya_capacity_verification",
        "benchmark_quality_contract",
        "source_registry",
        "targeted_closure_plan",
        "targeted_closure_report",
        "targeted_closure_verification",
    }
)
TOOLING_PATHS = {
    "implementation": "src/lumi/posttraining_gpt_oss_teacher_preflight.py",
    "runner": "scripts/plan_sft_gpt_oss_teacher_preflight.py",
    "test": "tests/test_gpt_oss_teacher_preflight.py",
    "verifier": "scripts/verify_sft_gpt_oss_teacher_preflight.py",
}


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve_bindings(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("gpt-oss preflight binding set changed")
    root = Path(repository_root).resolve()
    resolved: dict[str, Path] = {}
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping) or not isinstance(binding.get("path"), str):
            raise ValueError(f"gpt-oss binding malformed: {name}")
        raw = Path(binding["path"]).expanduser()
        path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        if not path.is_file() or file_sha256(path) != binding.get("sha256"):
            raise ValueError(f"gpt-oss binding missing or changed: {name}")
        resolved[name] = path
    return resolved


def _verify_predecessors(resolved: Mapping[str, Path]) -> None:
    closure = _load_json(resolved["targeted_closure_report"])
    if (
        closure.get("status")
        != "verified_targeted_closure_plan_no_execution_authorized"
        or closure.get("self_hosted_teacher_canary", {}).get("candidate_model")
        != "openai/gpt-oss-120b"
        or closure.get("self_hosted_teacher_canary", {}).get("exact_final_outputs")
        != 160
    ):
        raise ValueError("targeted closure teacher boundary changed")
    if not all(
        _load_json(resolved["targeted_closure_verification"])
        .get("checks", {})
        .values()
    ):
        raise ValueError("targeted closure verification is not clean")
    capacity = _load_json(resolved["aya_capacity_report"])
    decision = capacity.get("capacity_decision", {})
    if (
        capacity.get("status")
        != "verified_capacity_only_no_candidate_or_production_action"
        or decision.get("wikisplit_production_token_credit") != 0
        or not decision.get("wikisplit_capacity_lower_bound_reaches_numeric_target")
    ):
        raise ValueError("Aya capacity decision changed")
    verification = _load_json(resolved["aya_capacity_verification"])
    if verification.get("status") != (
        "verified_independent_full_replay_capacity_only"
    ) or not all(
        verification.get("checks", {}).values()
    ):
        raise ValueError("Aya capacity verification is not clean")


def _verify_model_objects(plan: Mapping[str, Any]) -> None:
    model = plan.get("model", {})
    objects = plan.get("model_objects")
    if (
        model.get("revision") != MODEL_REVISION
        or model.get("runtime_object_count") != 25
        or model.get("runtime_object_bytes") != MODEL_BYTES
        or not isinstance(objects, list)
        or len(objects) != 25
    ):
        raise ValueError("gpt-oss model inventory boundary changed")
    paths = [row.get("path") for row in objects if isinstance(row, Mapping)]
    if len(paths) != 25 or len(set(paths)) != 25:
        raise ValueError("gpt-oss model paths are not exactly unique")
    if sum(int(row.get("bytes", -1)) for row in objects) != MODEL_BYTES:
        raise ValueError("gpt-oss model byte arithmetic changed")
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", "")))
        for row in objects
    ):
        raise ValueError("gpt-oss model object SHA-256 malformed")
    weight_paths = [
        path
        for path in paths
        if isinstance(path, str) and path.endswith(".safetensors")
    ]
    if len(weight_paths) != 15:
        raise ValueError("gpt-oss converted weight shard count changed")


def _verify_execution_boundary(plan: Mapping[str, Any]) -> None:
    boundary = plan.get("execution_boundary", {})
    hardware = plan.get("hardware_and_cost", {})
    runtime = plan.get("runtime", {})
    if (
        boundary.get("model_download_bytes") != MODEL_BYTES
        or boundary.get("runtime_image_compressed_bytes") != RUNTIME_BYTES
        or boundary.get("maximum_network_bytes") != TOTAL_NETWORK_BYTES
        or runtime.get("container_image_digest") != RUNTIME_DIGEST
        or runtime.get("container_size_bytes") != RUNTIME_BYTES
    ):
        raise ValueError("gpt-oss runtime/network boundary changed")
    expected_cost = round(
        hardware.get("observed_price_usd_per_hour", -1)
        * boundary.get("maximum_instance_minutes", -1)
        / 60,
        2,
    )
    if expected_cost != boundary.get("maximum_usd_before_tax"):
        raise ValueError("gpt-oss maximum cost arithmetic changed")
    if hardware.get("exact_host_bound") is not False:
        raise ValueError("gpt-oss preflight unexpectedly claims an exact host")


def load_gpt_oss_teacher_preflight_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(path)
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status")
        != "authorized_metadata_only_exact_teacher_boundary_preflight"
        or plan.get("observed_at") != "2026-08-10"
    ):
        raise ValueError("gpt-oss preflight plan identity changed")
    authorization = plan.get("authorization", {})
    for field in (
        "candidate_generation_authorized",
        "dataset_seal_authorized",
        "external_api_authorized",
        "model_or_runtime_download_authorized",
        "paid_or_cloud_gpu_authorized",
        "production_sft_authorized",
    ):
        if authorization.get(field) is not False:
            raise ValueError(f"gpt-oss preflight authorization widened: {field}")
    resolved = _resolve_bindings(plan, repository_root)
    _verify_predecessors(resolved)
    _verify_model_objects(plan)
    _verify_execution_boundary(plan)
    contract = plan.get("canary_contract", {})
    if (
        contract.get("exact_prompt_cells") != 16
        or contract.get("prompts_per_cell") != 10
        or contract.get("exact_final_outputs") != 160
        or contract.get("maximum_inference_calls") != 200
        or contract.get("maximum_runtime_retries") != 40
        or contract.get("final_channel_only") is not True
        or contract.get("reasoning_or_analysis_channel_retained") is not False
    ):
        raise ValueError("gpt-oss canary contract changed")
    if plan.get("next_gate", {}).get("exact_host_bound") is not False:
        raise ValueError("gpt-oss next gate host boundary changed")
    return plan


def build_gpt_oss_teacher_preflight_report(
    *, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    resolved = _resolve_bindings(plan, repository_root)
    _verify_predecessors(resolved)
    report: dict[str, Any] = {
        "authorization": plan["authorization"],
        "bindings": {
            name: {"path": plan["bindings"][name]["path"], "sha256": file_sha256(path)}
            for name, path in sorted(resolved.items())
        },
        "blockers": [
            "exact_lambda_instance_ip_and_ed25519_host_key_not_bound",
            "exact_h100_gpu_uuid_and_observed_hbm_not_bound",
            "model_and_runtime_download_not_authorized",
            "teacher_generation_not_authorized",
            "targeted_closure_candidate_not_materialized_or_reviewed",
            "production_dataset_not_sealed",
            "gpu_sft_not_authorized",
        ],
        "canary_contract": plan["canary_contract"],
        "execution_boundary": plan["execution_boundary"],
        "hardware_and_cost": plan["hardware_and_cost"],
        "model": plan["model"],
        "model_inventory": {
            "object_count": len(plan["model_objects"]),
            "object_manifest_sha256": sha256_value(plan["model_objects"]),
            "total_bytes": sum(row["bytes"] for row in plan["model_objects"]),
            "weight_shard_count": sum(
                row["path"].endswith(".safetensors")
                for row in plan["model_objects"]
            ),
        },
        "next_gate": plan["next_gate"],
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "rights_and_provenance": plan["rights_and_provenance"],
        "runtime": plan["runtime"],
        "schema_version": REPORT_SCHEMA,
        "stage_progress": {
            "stage_2_aya_capacity_scan": "complete",
            "stage_3_exact_teacher_model_runtime_gpu_cost_boundary": (
                "complete_except_exact_host_and_gpu_identity"
            ),
            "stage_4_teacher_canary": "not_authorized",
        },
        "status": "verified_exact_teacher_boundary_host_unbound_no_execution_authorized",
        "tooling": {
            name: {"path": path, "sha256": file_sha256(Path(repository_root) / path)}
            for name, path in sorted(TOOLING_PATHS.items())
        },
        "wikisplit_production_token_credit": 0,
    }
    report["report_sha256"] = sha256_value(report)
    return report


def validate_gpt_oss_teacher_preflight_report(report: Mapping[str, Any]) -> None:
    unsigned = dict(report)
    observed = unsigned.pop("report_sha256", None)
    if observed != sha256_value(unsigned):
        raise ValueError("gpt-oss preflight report hash mismatch")
    if report.get("status") != (
        "verified_exact_teacher_boundary_host_unbound_no_execution_authorized"
    ):
        raise ValueError("gpt-oss preflight report status changed")
    if report.get("wikisplit_production_token_credit") != 0:
        raise ValueError("gpt-oss preflight credited WikiSplit unexpectedly")


def build_gpt_oss_teacher_preflight_verification(
    *, report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_gpt_oss_teacher_preflight_report(report)
    checks = {
        "authorization_remains_nonexecuting": not any(
            report["authorization"][field]
            for field in (
                "candidate_generation_authorized",
                "model_or_runtime_download_authorized",
                "paid_or_cloud_gpu_authorized",
                "production_sft_authorized",
            )
        ),
        "canary_is_exactly_160_final_outputs": (
            report["canary_contract"]["exact_final_outputs"] == 160
        ),
        "exact_host_remains_unbound": report["hardware_and_cost"]["exact_host_bound"]
        is False,
        "model_inventory_is_exact": (
            report["model_inventory"]["object_count"] == 25
            and report["model_inventory"]["total_bytes"] == MODEL_BYTES
            and report["model_inventory"]["weight_shard_count"] == 15
        ),
        "runtime_digest_is_exact": (
            report["runtime"]["container_image_digest"] == RUNTIME_DIGEST
        ),
        "wikisplit_credit_remains_zero": report["wikisplit_production_token_credit"]
        == 0,
    }
    if not all(checks.values()):
        raise ValueError("gpt-oss preflight independent verification failed")
    verification: dict[str, Any] = {
        "checks": checks,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
