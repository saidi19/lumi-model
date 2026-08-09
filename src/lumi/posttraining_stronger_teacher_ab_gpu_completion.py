"""Correct the bounded A10 stronger-teacher A/B after 256-token truncation."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import posttraining_first_party_scale_canary as scale
from . import posttraining_stronger_teacher_ab as paired
from . import posttraining_stronger_teacher_ab_gpu as gpu
from . import posttraining_stronger_teacher_ab_preflight as preflight

PLAN_SCHEMA_VERSION = "lumi.sft-stronger-teacher-ab-gpu-completion-plan.v0.1"
AUTHORIZATION_SCHEMA_VERSION = (
    "lumi.sft-stronger-teacher-ab-gpu-completion-authorization.v0.1"
)
REPORT_SCHEMA_VERSION = "lumi.sft-stronger-teacher-ab-gpu-completion-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-stronger-teacher-ab-gpu-completion-verification.v0.1"
)

FAILURE_RECEIPT_SHA256 = (
    "2799d76bc9611abda5070258e8c073f69370cc2e687b7764834d7fb597f52c3d"
)
FAILED_RECORD_ID = "lumi-scale-v01-train-multilingual_bengali-bengali_script-0001"
FAILURE_BOUNDARY = {
    "baseline_journal_bytes": 228_796,
    "baseline_journal_sha256": (
        "09ab1f3a14237ba4923a482f9a18908f09119739da7921110246ddb56f9dc587"
    ),
    "candidate_accepted_valid_finals": 8,
    "candidate_incomplete_finals": 1,
    "candidate_journal_bytes": 104_744,
    "candidate_journal_sha256": (
        "28aae4dd64345c1cc107f0084533a43475862495a5afd29b1e84588d7c438c1d"
    ),
    "candidate_starts": 61,
    "candidate_valid_finals": 21,
    "cumulative_generated_output_tokens": 30_884,
    "cumulative_inference_attempts": 182,
    "cumulative_runtime_retries": 60,
}
CORRECTION = {
    "failed_record_id": FAILED_RECORD_ID,
    "maximum_response_tokens_after": 512,
    "maximum_response_tokens_before": 256,
    "reopen_only_incomplete_retryable_final": True,
    "unchanged_generation_paths": [
        "baseline_journal",
        "candidate_complete_finals",
        "chat_template",
        "language_contracts",
        "paired_prompt_schedule",
        "quality_validators",
        "safety_and_tool_contracts",
        "sampling_options_except_num_predict",
    ],
}
LIMITS = {
    "cumulative_generated_output_tokens": 56_320,
    "cumulative_inference_attempts": 282,
    "cumulative_runtime_retries": 81,
    "maximum_additional_inference_attempts": 100,
    "maximum_additional_instance_minutes": 120,
    "maximum_additional_runtime_retries": 21,
    "maximum_cost_usd": 2.58,
    "maximum_network_bytes": 0,
    "maximum_private_remote_write_bytes": 1024**3,
    "remaining_valid_candidate_outputs": 79,
}
PROHIBITIONS = {
    "baseline_download_or_inference": True,
    "candidate_registry_change": True,
    "dataset_seal": True,
    "external_inference_api": True,
    "external_distribution": True,
    "human_authorship_claim": True,
    "model_or_runtime_download": True,
    "new_instance": True,
    "production_promotion": True,
    "public_ollama_port": True,
    "sft_training": True,
    "termination": True,
    "unrelated_access": True,
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def artifact_root(repository_root: Path) -> Path:
    return gpu.gpu_artifact_root(repository_root).with_name("attempt-002")


def _journal_summary(
    path: Path, schedule: Sequence[Any], *, model_key: str
) -> dict[str, Any]:
    journal = paired._read_jsonl(path)  # noqa: SLF001
    starts, finishes = paired._journal_state(  # noqa: SLF001
        journal, schedule, model_key=model_key
    )
    current_finals = [
        paired._final_event(finishes.get(slot.record_id, []))  # noqa: SLF001
        for slot in schedule
    ]
    finals = [event for event in current_finals if event is not None]
    valid = [event for event in finals if event.get("runtime_complete") is True]
    incomplete = [event for event in finals if event.get("runtime_complete") is not True]
    return {
        "accepted_valid_finals": sum(event.get("accepted") is True for event in valid),
        "bytes": path.stat().st_size,
        "current_final_outputs": len(finals),
        "incomplete_final_record_ids": sorted(
            str(event["record_id"]) for event in incomplete
        ),
        "incomplete_finals": len(incomplete),
        "journal_sha256": scale.file_sha256(path),
        "starts": len(starts),
        "valid_finals": len(valid),
    }


def current_boundary(
    *, artifact: Path, exact_failure_boundary: bool
) -> dict[str, Any]:
    schedule = preflight.selected_schedule()
    baseline_path = artifact / "private" / "baseline-call-journal.jsonl"
    candidate_path = artifact / "private" / "candidate-call-journal.jsonl"
    baseline = _journal_summary(baseline_path, schedule, model_key="baseline")
    candidate = _journal_summary(candidate_path, schedule, model_key="candidate")
    prefix_bytes = int(gpu.BOUNDARY["candidate_journal_prefix_bytes"])
    candidate_payload = candidate_path.read_bytes()
    if hashlib.sha256(candidate_payload[:prefix_bytes]).hexdigest() != gpu.BOUNDARY[
        "candidate_journal_prefix_sha256"
    ]:
        raise ValueError("candidate journal original prefix changed")
    if baseline["journal_sha256"] != FAILURE_BOUNDARY["baseline_journal_sha256"]:
        raise ValueError("completed baseline journal changed")
    if baseline["bytes"] != FAILURE_BOUNDARY["baseline_journal_bytes"]:
        raise ValueError("completed baseline journal size changed")
    if baseline["valid_finals"] != paired.PAIRED_PROMPTS:
        raise ValueError("completed baseline valid-final count changed")
    boundary = {
        "baseline": baseline,
        "candidate": candidate,
        "candidate_original_prefix_preserved": True,
        "cumulative_generated_output_tokens": paired._generated_output_tokens(  # noqa: SLF001
            artifact
        ),
        "cumulative_inference_attempts": paired._all_journal_attempts(artifact),  # noqa: SLF001
        "cumulative_runtime_retries": paired._runtime_retries_used(artifact),  # noqa: SLF001
    }
    if exact_failure_boundary:
        observed = {
            "baseline_journal_bytes": baseline["bytes"],
            "baseline_journal_sha256": baseline["journal_sha256"],
            "candidate_accepted_valid_finals": candidate["accepted_valid_finals"],
            "candidate_incomplete_finals": candidate["incomplete_finals"],
            "candidate_journal_bytes": candidate["bytes"],
            "candidate_journal_sha256": candidate["journal_sha256"],
            "candidate_starts": candidate["starts"],
            "candidate_valid_finals": candidate["valid_finals"],
            "cumulative_generated_output_tokens": boundary[
                "cumulative_generated_output_tokens"
            ],
            "cumulative_inference_attempts": boundary["cumulative_inference_attempts"],
            "cumulative_runtime_retries": boundary["cumulative_runtime_retries"],
        }
        if observed != FAILURE_BOUNDARY:
            raise ValueError("corrected continuation failure boundary changed")
        if candidate["incomplete_final_record_ids"] != [FAILED_RECORD_ID]:
            raise ValueError("corrected continuation incomplete record changed")
    else:
        if candidate["valid_finals"] != paired.PAIRED_PROMPTS:
            raise ValueError("corrected continuation candidate output count is incomplete")
        if candidate["incomplete_finals"] != 0:
            raise ValueError("corrected continuation retained an incomplete current final")
        if boundary["cumulative_inference_attempts"] > LIMITS[
            "cumulative_inference_attempts"
        ]:
            raise ValueError("corrected continuation inference-attempt ceiling exceeded")
        if boundary["cumulative_runtime_retries"] > LIMITS[
            "cumulative_runtime_retries"
        ]:
            raise ValueError("corrected continuation runtime-retry ceiling exceeded")
        if boundary["cumulative_generated_output_tokens"] > LIMITS[
            "cumulative_generated_output_tokens"
        ]:
            raise ValueError("corrected continuation output-token ceiling exceeded")
    return boundary


def _tooling_hashes(repository_root: Path) -> dict[str, str]:
    paths = {
        "base_implementation": "src/lumi/posttraining_stronger_teacher_ab.py",
        "completion_implementation": (
            "src/lumi/posttraining_stronger_teacher_ab_gpu_completion.py"
        ),
        "completion_preflight": (
            "scripts/preflight_stronger_teacher_ab_gpu_completion.py"
        ),
        "completion_runner": "scripts/run_stronger_teacher_ab_gpu_completion.py",
        "completion_test": "tests/test_stronger_teacher_ab_gpu_completion.py",
        "completion_verifier": (
            "scripts/verify_stronger_teacher_ab_gpu_completion.py"
        ),
    }
    return {
        name: scale.file_sha256(repository_root / relative_path)
        for name, relative_path in paths.items()
    }


def build_preflight(
    *,
    base_plan_path: Path,
    failure_receipt_path: Path,
    remote_receipt_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = paired.load_plan(base_plan_path.resolve(), repository_root=root)
    source_artifact = paired.artifact_root(plan, root)
    before = current_boundary(artifact=source_artifact, exact_failure_boundary=True)
    if scale.file_sha256(failure_receipt_path) != FAILURE_RECEIPT_SHA256:
        raise ValueError("A10 failure receipt changed")
    gpu._load_remote_receipt(remote_receipt_path.resolve())  # noqa: SLF001
    checks = {
        "baseline_byte_exact": before["baseline"]["journal_sha256"]
        == FAILURE_BOUNDARY["baseline_journal_sha256"],
        "candidate_failure_boundary_exact": before["candidate"]["journal_sha256"]
        == FAILURE_BOUNDARY["candidate_journal_sha256"],
        "candidate_prefix_byte_exact": before["candidate_original_prefix_preserved"],
        "exact_one_incomplete_record": before["candidate"][
            "incomplete_final_record_ids"
        ]
        == [FAILED_RECORD_ID],
        "remaining_outputs_fit_call_budget": LIMITS[
            "remaining_valid_candidate_outputs"
        ]
        <= LIMITS["maximum_additional_inference_attempts"],
        "response_ceiling_corrected": CORRECTION["maximum_response_tokens_after"]
        == 2 * CORRECTION["maximum_response_tokens_before"],
        "zero_additional_network": LIMITS["maximum_network_bytes"] == 0,
    }
    if not all(checks.values()):
        raise ValueError("corrected A10 continuation preflight failed")
    output_root = artifact_root(root)
    proposed = {
        "base_plan_file_sha256": scale.file_sha256(base_plan_path),
        "checks": checks,
        "correction": CORRECTION,
        "failure_boundary": before,
        "failure_receipt_file_sha256": scale.file_sha256(failure_receipt_path),
        "limits": LIMITS,
        "prohibitions": PROHIBITIONS,
        "remote_receipt_file_sha256": scale.file_sha256(remote_receipt_path),
        "schema_version": PLAN_SCHEMA_VERSION,
        "server": gpu.SERVER,
        "status": "ready_for_explicit_corrected_continuation_authorization",
        "tooling_file_sha256": _tooling_hashes(root),
    }
    proposed["plan_sha256"] = gpu.sha256_value(proposed)
    plan_path = output_root / "corrected-continuation-plan.json"
    _write_json(plan_path, proposed)
    verification = {
        "checks": checks,
        "plan_file_sha256": scale.file_sha256(plan_path),
        "plan_internal_sha256": proposed["plan_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS",
    }
    verification["verification_sha256"] = gpu.sha256_value(verification)
    _write_json(output_root / "preflight-verification.json", verification)
    return {"plan": proposed, "verification": verification}


def load_authorization(path: Path, *, plan_path: Path) -> dict[str, Any]:
    authorization = _load_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported corrected A10 continuation authorization")
    if authorization.get("status") != "authorized_corrected_a10_completion":
        raise ValueError("corrected A10 continuation authorization status changed")
    if authorization.get("plan_file_sha256") != scale.file_sha256(plan_path):
        raise ValueError("corrected A10 continuation plan binding changed")
    if authorization.get("failure_boundary") != FAILURE_BOUNDARY:
        raise ValueError("corrected A10 continuation authorization boundary changed")
    if authorization.get("correction") != CORRECTION:
        raise ValueError("corrected A10 continuation correction changed")
    if authorization.get("limits") != LIMITS:
        raise ValueError("corrected A10 continuation limits changed")
    if authorization.get("prohibitions") != PROHIBITIONS:
        raise ValueError("corrected A10 continuation prohibitions changed")
    if authorization.get("server") != gpu.SERVER:
        raise ValueError("corrected A10 continuation server binding changed")
    return authorization


def _machine_audit(artifact: Path, schedule: list[Any]) -> dict[str, Any]:
    baseline = paired._model_audit(artifact, schedule, model_key="baseline")  # noqa: SLF001
    candidate = paired._model_audit(artifact, schedule, model_key="candidate")  # noqa: SLF001
    advantage = round(candidate["acceptance_rate"] - baseline["acceptance_rate"], 6)
    checks = {
        "candidate_advantage": advantage
        >= paired.EXPECTED_GATES["candidate_minimum_advantage_over_baseline"],
        "candidate_machine_acceptance": candidate["acceptance_rate"]
        >= paired.EXPECTED_GATES["candidate_minimum_machine_acceptance_rate"],
        "candidate_per_cell_acceptance": min(candidate["cell_acceptance_rates"].values())
        >= paired.EXPECTED_GATES["candidate_minimum_per_cell_acceptance_rate"],
        "candidate_supervised_token_ceiling": sum(
            candidate["retained_supervised_tokens"].values()
        )
        <= paired.MAXIMUM_RETAINED_CANDIDATE_SUPERVISED_TOKENS,
        "critical_factual_safety_tool_failures": candidate[
            "critical_factual_safety_or_tool_failures"
        ]
        == 0,
        "exact_final_output_count": baseline["final_outputs"]
        + candidate["final_outputs"]
        == paired.EXPECTED_FINAL_OUTPUTS,
        "inference_attempt_ceiling": paired._all_journal_attempts(artifact)  # noqa: SLF001
        <= LIMITS["cumulative_inference_attempts"],
        "output_token_ceiling": paired._generated_output_tokens(artifact)  # noqa: SLF001
        <= LIMITS["cumulative_generated_output_tokens"],
        "runtime_retry_ceiling": paired._runtime_retries_used(artifact)  # noqa: SLF001
        <= LIMITS["cumulative_runtime_retries"],
    }
    return {
        "acceptance_advantage": advantage,
        "baseline": baseline,
        "candidate": candidate,
        "machine_checks": checks,
    }


def run_completion(
    *,
    base_plan_path: Path,
    authorization_path: Path,
    completion_plan_path: Path,
    remote_receipt_path: Path,
    repository_root: Path,
    ssh_key: Path,
    known_hosts: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    base_plan = paired.load_plan(base_plan_path.resolve(), repository_root=root)
    source_artifact = paired.artifact_root(base_plan, root)
    completion_plan = _load_json(completion_plan_path.resolve())
    if completion_plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("corrected A10 continuation plan schema changed")
    unsigned_plan = dict(completion_plan)
    observed_plan_sha = unsigned_plan.pop("plan_sha256", None)
    if observed_plan_sha != gpu.sha256_value(unsigned_plan):
        raise ValueError("corrected A10 continuation plan internal hash changed")
    authorization = load_authorization(
        authorization_path.resolve(), plan_path=completion_plan_path.resolve()
    )
    gpu._load_remote_receipt(remote_receipt_path.resolve())  # noqa: SLF001
    before = current_boundary(artifact=source_artifact, exact_failure_boundary=True)
    paired.verify_candidate_objects(base_plan, repository_root=root)
    resolved = paired._resolve_bindings(base_plan, root)  # noqa: SLF001
    paired.verify_ollama_manifest(resolved["baseline_ollama_manifest"], model_key="baseline")
    paired.verify_ollama_manifest(  # noqa: SLF001
        paired._candidate_manifest_path(), model_key="candidate"
    )
    paired._ollama_version()  # noqa: SLF001
    authorized_start_unix = int(authorization["authorized_start_unix"])
    if time.time() - authorized_start_unix > LIMITS[
        "maximum_additional_instance_minutes"
    ] * 60:
        raise RuntimeError("corrected A10 continuation runtime authorization expired")
    state = _load_json(source_artifact / "private" / "execution-state.json")
    original_started_unix = int(state["started_unix"])
    if time.time() - original_started_unix > paired.MAXIMUM_RUNTIME_MINUTES * 60:
        raise RuntimeError("original paired A/B cumulative runtime authorization expired")

    schedule = preflight.selected_schedule()
    quality, benchmark, tokenizer, template, exemplars = (  # noqa: SLF001
        paired._load_runtime_resources(base_plan, repository_root=root)
    )
    gpu_before = gpu._gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())  # noqa: SLF001
    phase = paired._run_model_phase(  # noqa: SLF001
        model_key="candidate",
        schedule=schedule,
        artifact=source_artifact,
        quality=quality,
        benchmark=benchmark,
        tokenizer=tokenizer,
        template=template,
        exemplars=exemplars,
        thermal_cache=Path(base_plan["execution"]["thermal_probe_cache_path"]),
        started_unix=original_started_unix,
        maximum_attempts=LIMITS["cumulative_inference_attempts"],
        maximum_retries=LIMITS["cumulative_runtime_retries"],
        continuation_authorized=True,
        maximum_response_tokens=CORRECTION["maximum_response_tokens_after"],
    )
    gpu_after = gpu._gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())  # noqa: SLF001
    after = current_boundary(artifact=source_artifact, exact_failure_boundary=False)
    audit = _machine_audit(source_artifact, schedule)
    paired._replay_model_rows(  # noqa: SLF001
        artifact=source_artifact,
        schedule=schedule,
        model_key="baseline",
        quality=quality,
        benchmark=benchmark,
        tokenizer=tokenizer,
        template=template,
    )
    paired._replay_model_rows(  # noqa: SLF001
        artifact=source_artifact,
        schedule=schedule,
        model_key="candidate",
        quality=quality,
        benchmark=benchmark,
        tokenizer=tokenizer,
        template=template,
    )
    review = paired._materialize_review(source_artifact, schedule)  # noqa: SLF001
    machine_gate_passed = all(audit["machine_checks"].values())
    report = {
        "authorization_file_sha256": scale.file_sha256(authorization_path),
        "boundary_after": after,
        "boundary_before": before,
        "comparative_audit": audit,
        "completion_plan_file_sha256": scale.file_sha256(completion_plan_path),
        "controller_safety_snapshots": phase["safety_snapshots"],
        "correction": CORRECTION,
        "gpu": {
            "after": gpu_after,
            "before": gpu_before,
            "cloud_or_paid_gpu_used": True,
            "maximum_additional_cost_usd": LIMITS["maximum_cost_usd"],
            "rate_usd_per_hour": 1.29,
            "server": gpu.SERVER,
        },
        "machine_gate_passed": machine_gate_passed,
        "owner_review": review,
        "production": {
            "candidate_registry_changed": False,
            "dataset_sealed": False,
            "production_eligible": False,
            "sft_training_started": False,
        },
        "provenance": {
            "ai_assisted": True,
            "external_inference_api_calls": 0,
            "human_authorship_claimed": False,
            "self_hosted_over_pinned_ssh_tunnel": True,
        },
        "remote_receipt_file_sha256": scale.file_sha256(remote_receipt_path),
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": (
            "machine_verified_candidate_pending_owner_review"
            if machine_gate_passed
            else "verified_candidate_machine_gate_not_met"
        ),
        "tooling_file_sha256": _tooling_hashes(root),
    }
    report["report_sha256"] = gpu.sha256_value(report)
    output = artifact_root(root) / "evidence" / "completion-report.json"
    _write_json(output, report)
    return report


def verify_completion(
    *,
    base_plan_path: Path,
    authorization_path: Path,
    completion_plan_path: Path,
    remote_receipt_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    base_plan = paired.load_plan(base_plan_path.resolve(), repository_root=root)
    source_artifact = paired.artifact_root(base_plan, root)
    load_authorization(authorization_path.resolve(), plan_path=completion_plan_path.resolve())
    gpu._load_remote_receipt(remote_receipt_path.resolve())  # noqa: SLF001
    boundary = current_boundary(artifact=source_artifact, exact_failure_boundary=False)
    schedule = preflight.selected_schedule()
    quality, benchmark, tokenizer, template, _ = paired._load_runtime_resources(  # noqa: SLF001
        base_plan, repository_root=root
    )
    for model_key in paired.MODEL_KEYS:
        paired._replay_model_rows(  # noqa: SLF001
            artifact=source_artifact,
            schedule=schedule,
            model_key=model_key,
            quality=quality,
            benchmark=benchmark,
            tokenizer=tokenizer,
            template=template,
        )
    expected_audit = _machine_audit(source_artifact, schedule)
    report_path = artifact_root(root) / "evidence" / "completion-report.json"
    report = _load_json(report_path)
    unsigned = dict(report)
    observed_internal_sha = unsigned.pop("report_sha256", None)
    checks = {
        "authorization_bound": report.get("authorization_file_sha256")
        == scale.file_sha256(authorization_path),
        "boundary_reproduced": report.get("boundary_after") == boundary,
        "comparative_audit_reproduced": report.get("comparative_audit")
        == expected_audit,
        "completion_plan_bound": report.get("completion_plan_file_sha256")
        == scale.file_sha256(completion_plan_path),
        "report_internal_hash": observed_internal_sha == gpu.sha256_value(unsigned),
        "remote_receipt_bound": report.get("remote_receipt_file_sha256")
        == scale.file_sha256(remote_receipt_path),
        "status_matches_machine_gate": report.get("machine_gate_passed")
        == all(expected_audit["machine_checks"].values()),
        "tooling_bound": report.get("tooling_file_sha256") == _tooling_hashes(root),
    }
    verification = {
        "checks": checks,
        "completion_report_file_sha256": scale.file_sha256(report_path),
        "completion_report_internal_sha256": observed_internal_sha,
        "machine_gate_passed": all(expected_audit["machine_checks"].values()),
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    verification["verification_sha256"] = gpu.sha256_value(verification)
    output = artifact_root(root) / "evidence" / "completion-verification.json"
    _write_json(output, verification)
    if verification["status"] != "PASS":
        raise ValueError("corrected A10 completion independent verification failed")
    return verification
