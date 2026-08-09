"""Result-preserving A10 continuation for the Lumi stronger-teacher A/B gate."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from . import posttraining_first_party_scale_canary as scale
from . import posttraining_stronger_teacher_ab as paired
from . import posttraining_stronger_teacher_ab_preflight as preflight

AUTHORIZATION_SCHEMA_VERSION = "lumi.sft-stronger-teacher-ab-gpu-authorization.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-stronger-teacher-ab-gpu-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-stronger-teacher-ab-gpu-verification.v0.1"

REMOTE_ROOT = "/home/ubuntu/lumi-stronger-teacher-ab-gpu-c52325bacdb8"
SERVER = {
    "gpu_memory_mib": 23028,
    "gpu_name": "NVIDIA A10",
    "gpu_uuid": "GPU-959da0a2-b29e-601d-f97b-e245ed0997c1",
    "host": "147.224.14.222",
    "host_key_fingerprint": "SHA256:XB6mn0C7Kpmjh6NHayN04uYeNQWdDMucdngXCcGwstY",
    "ssh_user": "ubuntu",
}
DOWNLOADS = [
    {
        "bytes": 1_422_353_729,
        "path": f"{REMOTE_ROOT}/downloads/ollama-linux-amd64.tar.zst",
        "sha256": "f7d6bdbcf71b83aa8670c4e7dc4b6936c0952fcf8b114eaf6a11cbadb9684214",
        "url": (
            "https://github.com/ollama/ollama/releases/download/v0.32.5/"
            "ollama-linux-amd64.tar.zst"
        ),
    },
    {
        "bytes": 9_001_752_960,
        "path": f"{REMOTE_ROOT}/model/Qwen3-14B-Q4_K_M.gguf",
        "sha256": "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0",
        "url": (
            "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/"
            "c75e7b2d0234068f674a1bacf548ea32e27ccd29/Qwen3-14B-Q4_K_M.gguf"
        ),
    },
    {
        "bytes": 11_544,
        "path": f"{REMOTE_ROOT}/model/LICENSE",
        "sha256": "5de36594c10839788a8c589443a8ef9d8b8d17c65a1b5807206ae037fc36c6bd",
        "url": (
            "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/"
            "c75e7b2d0234068f674a1bacf548ea32e27ccd29/LICENSE"
        ),
    },
    {
        "bytes": 7_493,
        "path": f"{REMOTE_ROOT}/model/README.md",
        "sha256": "45989a28a5ba6525cbbce906d8bc165ca612d8856a68ca27f568ce62056405a3",
        "url": (
            "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/"
            "c75e7b2d0234068f674a1bacf548ea32e27ccd29/README.md"
        ),
    },
]
LIMITS = {
    "cumulative_generated_output_tokens": 56_320,
    "cumulative_inference_attempts": 260,
    "cumulative_runtime_retries": 60,
    "maximum_additional_inference_attempts": 135,
    "maximum_additional_instance_minutes": 240,
    "maximum_additional_runtime_retries": 39,
    "maximum_cost_usd": 5.16,
    "maximum_network_bytes": 10_424_125_726,
    "maximum_private_remote_write_bytes": 32 * 1024**3,
    "remaining_candidate_outputs": 96,
}
BOUNDARY = {
    "baseline_final_outputs": 100,
    "baseline_journal_bytes": 228_796,
    "baseline_journal_sha256": "09ab1f3a14237ba4923a482f9a18908f09119739da7921110246ddb56f9dc587",
    "candidate_final_outputs": 4,
    "candidate_journal_prefix_bytes": 7_638,
    "candidate_journal_prefix_sha256": (
        "734a3b24ed4627ec820934ae97c368294803a530785e24c0cde66cfa05475df3"
    ),
    "cumulative_generated_output_tokens": 18_001,
    "cumulative_inference_attempts": 125,
    "cumulative_runtime_retries": 21,
}
PROHIBITIONS = {
    "baseline_download_or_inference": True,
    "candidate_registry_change": True,
    "dataset_seal": True,
    "external_inference_api": True,
    "external_distribution": True,
    "human_authorship_claim": True,
    "new_instance": True,
    "production_promotion": True,
    "public_ollama_port": True,
    "sft_training": True,
    "termination": True,
    "unrelated_access": True,
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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


def gpu_artifact_root(repository_root: Path) -> Path:
    return (
        repository_root.resolve().parent
        / "lumi-artifacts"
        / "4b-sft-stronger-teacher-ab-gpu-continuation-v0.1"
        / "c52325bacdb8"
        / "attempt-001"
    )


def load_gpu_authorization(path: Path) -> dict[str, Any]:
    authorization = _load_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported stronger-teacher GPU authorization")
    if authorization.get("status") != "authorized_result_preserving_a10_continuation":
        raise ValueError("stronger-teacher GPU authorization status changed")
    if authorization.get("server") != SERVER:
        raise ValueError("stronger-teacher GPU server binding changed")
    if authorization.get("downloads") != DOWNLOADS:
        raise ValueError("stronger-teacher GPU download inventory changed")
    if authorization.get("limits") != LIMITS:
        raise ValueError("stronger-teacher GPU limits changed")
    if authorization.get("boundary") != BOUNDARY:
        raise ValueError("stronger-teacher GPU boundary changed")
    if authorization.get("prohibitions") != PROHIBITIONS:
        raise ValueError("stronger-teacher GPU prohibitions changed")
    return authorization


def _journal_finals(path: Path, schedule: list[Any], model_key: str) -> int:
    _, finishes = paired._journal_state(  # noqa: SLF001
        paired._read_jsonl(path), schedule, model_key=model_key  # noqa: SLF001
    )
    return sum(
        paired._final_event(finishes.get(slot.record_id, [])) is not None  # noqa: SLF001
        for slot in schedule
    )


def verify_preserved_boundary(artifact: Path) -> dict[str, Any]:
    schedule = preflight.selected_schedule()
    baseline = artifact / "private" / "baseline-call-journal.jsonl"
    candidate = artifact / "private" / "candidate-call-journal.jsonl"
    baseline_bytes = baseline.read_bytes()
    candidate_bytes = candidate.read_bytes()
    if len(baseline_bytes) != BOUNDARY["baseline_journal_bytes"]:
        raise ValueError("completed baseline journal size changed")
    if hashlib.sha256(baseline_bytes).hexdigest() != BOUNDARY["baseline_journal_sha256"]:
        raise ValueError("completed baseline journal changed")
    prefix_bytes = int(BOUNDARY["candidate_journal_prefix_bytes"])
    if len(candidate_bytes) < prefix_bytes:
        raise ValueError("candidate journal prefix was truncated")
    if hashlib.sha256(candidate_bytes[:prefix_bytes]).hexdigest() != BOUNDARY[
        "candidate_journal_prefix_sha256"
    ]:
        raise ValueError("candidate journal prefix changed")
    baseline_finals = _journal_finals(baseline, schedule, "baseline")
    candidate_finals = _journal_finals(candidate, schedule, "candidate")
    if baseline_finals != BOUNDARY["baseline_final_outputs"]:
        raise ValueError("completed baseline output count changed")
    if candidate_finals < BOUNDARY["candidate_final_outputs"]:
        raise ValueError("candidate output boundary regressed")
    if paired._all_journal_attempts(artifact) > LIMITS[  # noqa: SLF001
        "cumulative_inference_attempts"
    ]:
        raise ValueError("cumulative inference boundary exceeded")
    if paired._runtime_retries_used(artifact) > LIMITS[  # noqa: SLF001
        "cumulative_runtime_retries"
    ]:
        raise ValueError("cumulative retry boundary exceeded")
    if paired._generated_output_tokens(artifact) > LIMITS[  # noqa: SLF001
        "cumulative_generated_output_tokens"
    ]:
        raise ValueError("cumulative output-token boundary exceeded")
    return {
        "baseline_final_outputs": baseline_finals,
        "baseline_journal_sha256": scale.file_sha256(baseline),
        "candidate_final_outputs": candidate_finals,
        "candidate_journal_file_sha256": scale.file_sha256(candidate),
        "candidate_prefix_preserved": True,
    }


def _load_remote_receipt(path: Path) -> dict[str, Any]:
    receipt = _load_json(path)
    if receipt.get("status") != "PASS_exact_localhost_a10_runtime_ready":
        raise ValueError("A10 runtime receipt is not PASS")
    if receipt.get("server") != SERVER:
        raise ValueError("A10 runtime receipt server changed")
    if receipt.get("downloads") != [
        {"bytes": item["bytes"], "path": item["path"], "sha256": item["sha256"]}
        for item in DOWNLOADS
    ]:
        raise ValueError("A10 runtime receipt download inventory changed")
    if receipt.get("ollama") != {
        "alias": paired.MODEL_ALIASES["candidate"],
        "endpoint": "127.0.0.1:11434",
        "models_root": f"{REMOTE_ROOT}/ollama-models",
        "public_listener": False,
        "version": "0.32.5",
    }:
        raise ValueError("A10 runtime receipt Ollama binding changed")
    if not all(receipt.get("checks", {}).values()):
        raise ValueError("A10 runtime receipt has a failed check")
    return receipt


def _gpu_snapshot(ssh_key: Path, known_hosts: Path) -> dict[str, Any]:
    query = (
        "name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,"
        "power.draw,ecc.errors.uncorrected.volatile.total"
    )
    command = [
        "ssh",
        "-i",
        str(ssh_key),
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=15",
        f"{SERVER['ssh_user']}@{SERVER['host']}",
        f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("expected exactly one GPU from A10 continuation server")
    fields = [item.strip() for item in rows[0].split(",")]
    if len(fields) != 8:
        raise RuntimeError("unexpected A10 telemetry field count")
    snapshot = {
        "ecc_uncorrected_volatile": int(fields[7]),
        "gpu_name": fields[0],
        "gpu_uuid": fields[1],
        "memory_total_mib": int(fields[3]),
        "memory_used_mib": int(fields[2]),
        "power_watts": float(fields[6]),
        "temperature_c": int(fields[5]),
        "utilization_percent": int(fields[4]),
    }
    if snapshot["gpu_name"] != SERVER["gpu_name"]:
        raise RuntimeError("A10 GPU name changed")
    if snapshot["gpu_uuid"] != SERVER["gpu_uuid"]:
        raise RuntimeError("A10 GPU UUID changed")
    if snapshot["memory_total_mib"] != SERVER["gpu_memory_mib"]:
        raise RuntimeError("A10 GPU memory binding changed")
    if snapshot["ecc_uncorrected_volatile"] != 0:
        raise RuntimeError("A10 reported an uncorrected ECC error")
    return snapshot


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


def _tooling_hashes(repository_root: Path) -> dict[str, str]:
    paths = {
        "base_implementation": "src/lumi/posttraining_stronger_teacher_ab.py",
        "gpu_implementation": "src/lumi/posttraining_stronger_teacher_ab_gpu.py",
        "gpu_runner": "scripts/run_stronger_teacher_ab_gpu.py",
        "gpu_test": "tests/test_stronger_teacher_ab_gpu.py",
        "gpu_verifier": "scripts/verify_stronger_teacher_ab_gpu.py",
    }
    return {
        name: scale.file_sha256(repository_root / relative_path)
        for name, relative_path in paths.items()
    }


def run_gpu_continuation(
    *,
    base_plan_path: Path,
    continuation_authorization_path: Path,
    gpu_authorization_path: Path,
    remote_receipt_path: Path,
    repository_root: Path,
    ssh_key: Path,
    known_hosts: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = paired.load_plan(base_plan_path.resolve(), repository_root=root)
    artifact = paired.artifact_root(plan, root)
    paired.load_continuation_authorization(
        continuation_authorization_path.resolve(), artifact=artifact, repository_root=root
    )
    gpu_authorization = load_gpu_authorization(gpu_authorization_path.resolve())
    before = verify_preserved_boundary(artifact)
    if before["candidate_final_outputs"] != BOUNDARY["candidate_final_outputs"]:
        raise ValueError("fresh GPU continuation must begin at exactly four candidate outputs")
    remote_receipt = _load_remote_receipt(remote_receipt_path.resolve())
    paired.verify_candidate_objects(plan, repository_root=root)
    resolved = paired._resolve_bindings(plan, root)  # noqa: SLF001
    paired.verify_ollama_manifest(resolved["baseline_ollama_manifest"], model_key="baseline")
    paired.verify_ollama_manifest(  # noqa: SLF001
        paired._candidate_manifest_path(), model_key="candidate"
    )
    paired._ollama_version()  # noqa: SLF001 -- checks tunneled remote server and local client

    state = _load_json(artifact / "private" / "execution-state.json")
    started_unix = int(state["started_unix"])
    if time.time() - started_unix > paired.MAXIMUM_RUNTIME_MINUTES * 60:
        raise RuntimeError("original paired A/B cumulative runtime authorization expired")
    gpu_started_unix = int(gpu_authorization["authorized_start_unix"])
    if time.time() - gpu_started_unix > LIMITS["maximum_additional_instance_minutes"] * 60:
        raise RuntimeError("A10 continuation runtime authorization expired")

    schedule = preflight.selected_schedule()
    quality, benchmark, tokenizer, template, exemplars = (  # noqa: SLF001
        paired._load_runtime_resources(plan, repository_root=root)
    )
    gpu_before = _gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    phase = paired._run_model_phase(  # noqa: SLF001
        model_key="candidate",
        schedule=schedule,
        artifact=artifact,
        quality=quality,
        benchmark=benchmark,
        tokenizer=tokenizer,
        template=template,
        exemplars=exemplars,
        thermal_cache=Path(plan["execution"]["thermal_probe_cache_path"]),
        started_unix=started_unix,
        maximum_attempts=LIMITS["cumulative_inference_attempts"],
        maximum_retries=LIMITS["cumulative_runtime_retries"],
        continuation_authorized=True,
    )
    gpu_after = _gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    after = verify_preserved_boundary(artifact)
    if after["candidate_final_outputs"] != paired.PAIRED_PROMPTS:
        raise RuntimeError("A10 continuation did not complete all candidate outputs")

    audit = _machine_audit(artifact, schedule)
    paired._replay_model_rows(  # noqa: SLF001
        artifact=artifact,
        schedule=schedule,
        model_key="baseline",
        quality=quality,
        benchmark=benchmark,
        tokenizer=tokenizer,
        template=template,
    )
    paired._replay_model_rows(  # noqa: SLF001
        artifact=artifact,
        schedule=schedule,
        model_key="candidate",
        quality=quality,
        benchmark=benchmark,
        tokenizer=tokenizer,
        template=template,
    )
    review = paired._materialize_review(artifact, schedule)  # noqa: SLF001
    report = {
        "authorization_file_sha256": scale.file_sha256(gpu_authorization_path),
        "base_continuation_authorization_file_sha256": scale.file_sha256(
            continuation_authorization_path
        ),
        "base_plan_file_sha256": scale.file_sha256(base_plan_path),
        "boundary_after": after,
        "boundary_before": before,
        "comparative_audit": audit,
        "controller_safety_snapshots": phase["safety_snapshots"],
        "generated_output_tokens": paired._generated_output_tokens(artifact),  # noqa: SLF001
        "gpu": {
            "after": gpu_after,
            "before": gpu_before,
            "cloud_or_paid_gpu_used": True,
            "maximum_cost_usd": LIMITS["maximum_cost_usd"],
            "rate_usd_per_hour": 1.29,
            "server": SERVER,
        },
        "inference_attempts": paired._all_journal_attempts(artifact),  # noqa: SLF001
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
        "runtime_retries": paired._runtime_retries_used(artifact),  # noqa: SLF001
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": (
            "machine_verified_candidate_pending_owner_review"
            if all(audit["machine_checks"].values())
            else "failed_closed_candidate_machine_gate_not_met"
        ),
        "tooling_file_sha256": _tooling_hashes(root),
    }
    report["report_sha256"] = sha256_value(report)
    output = gpu_artifact_root(root) / "evidence" / "gpu-run-report.json"
    _write_json(output, report)
    return report


def verify_gpu_continuation(
    *,
    base_plan_path: Path,
    continuation_authorization_path: Path,
    gpu_authorization_path: Path,
    remote_receipt_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = paired.load_plan(base_plan_path.resolve(), repository_root=root)
    artifact = paired.artifact_root(plan, root)
    paired.load_continuation_authorization(
        continuation_authorization_path.resolve(), artifact=artifact, repository_root=root
    )
    load_gpu_authorization(gpu_authorization_path.resolve())
    _load_remote_receipt(remote_receipt_path.resolve())
    boundary = verify_preserved_boundary(artifact)
    if boundary["candidate_final_outputs"] != paired.PAIRED_PROMPTS:
        raise ValueError("verified A10 continuation is incomplete")
    schedule = preflight.selected_schedule()
    quality, benchmark, tokenizer, template, _ = paired._load_runtime_resources(  # noqa: SLF001
        plan, repository_root=root
    )
    for model_key in paired.MODEL_KEYS:
        paired._replay_model_rows(  # noqa: SLF001
            artifact=artifact,
            schedule=schedule,
            model_key=model_key,
            quality=quality,
            benchmark=benchmark,
            tokenizer=tokenizer,
            template=template,
        )
    expected_audit = _machine_audit(artifact, schedule)
    report_path = gpu_artifact_root(root) / "evidence" / "gpu-run-report.json"
    report = _load_json(report_path)
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("A10 continuation report schema changed")
    unsigned = dict(report)
    observed_internal_sha = unsigned.pop("report_sha256", None)
    if observed_internal_sha != sha256_value(unsigned):
        raise ValueError("A10 continuation report internal hash changed")
    if report.get("comparative_audit") != expected_audit:
        raise ValueError("A10 continuation comparative audit changed")
    if report.get("boundary_after") != boundary:
        raise ValueError("A10 continuation final boundary changed")
    if report.get("authorization_file_sha256") != scale.file_sha256(
        gpu_authorization_path
    ):
        raise ValueError("A10 continuation authorization binding changed")
    if report.get("remote_receipt_file_sha256") != scale.file_sha256(
        remote_receipt_path
    ):
        raise ValueError("A10 continuation remote receipt binding changed")
    if report.get("tooling_file_sha256") != _tooling_hashes(root):
        raise ValueError("A10 continuation tooling changed")
    checks = {
        "all_machine_checks_reproduced": all(expected_audit["machine_checks"].values()),
        "baseline_journal_preserved": boundary["baseline_journal_sha256"]
        == BOUNDARY["baseline_journal_sha256"],
        "candidate_outputs_complete": boundary["candidate_final_outputs"]
        == paired.PAIRED_PROMPTS,
        "candidate_prefix_preserved": boundary["candidate_prefix_preserved"] is True,
        "generated_output_token_ceiling": paired._generated_output_tokens(artifact)  # noqa: SLF001
        <= LIMITS["cumulative_generated_output_tokens"],
        "inference_attempt_ceiling": paired._all_journal_attempts(artifact)  # noqa: SLF001
        <= LIMITS["cumulative_inference_attempts"],
        "runtime_retry_ceiling": paired._runtime_retries_used(artifact)  # noqa: SLF001
        <= LIMITS["cumulative_runtime_retries"],
    }
    verification = {
        "checks": checks,
        "gpu_run_report_file_sha256": scale.file_sha256(report_path),
        "gpu_run_report_internal_sha256": observed_internal_sha,
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    verification["verification_sha256"] = sha256_value(verification)
    output = gpu_artifact_root(root) / "evidence" / "gpu-run-verification.json"
    _write_json(output, verification)
    if verification["status"] != "PASS":
        raise ValueError("A10 continuation independent verification failed")
    return verification
