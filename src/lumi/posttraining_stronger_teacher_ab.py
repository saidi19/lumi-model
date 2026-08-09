"""Bounded Mac-local paired A/B for the Lumi stronger-teacher gate."""

from __future__ import annotations

import hashlib
import html
import http.client
import json
import os
import statistics
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_first_party_scale_canary as scale
from . import posttraining_first_party_teacher_canary as teacher
from . import posttraining_stronger_teacher_ab_preflight as preflight
from .data_quality import BenchmarkRegistry
from .posttraining_format import load_chat_template

PLAN_SCHEMA_VERSION = "lumi.sft-stronger-teacher-ab-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-stronger-teacher-ab-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-stronger-teacher-ab-verification.v0.1"
ROW_SOURCE_ID = "lumi_stronger_teacher_ab_v0_1"
LOCAL_ENDPOINT = "http://127.0.0.1:11434/api/generate"
OLLAMA_SERVER_VERSION_ENDPOINT = "http://127.0.0.1:11434/api/version"
OLLAMA_RUNTIME_ROOT = Path("/private/tmp/lumi-ollama-v0.32.5")
OLLAMA_ARCHIVE = OLLAMA_RUNTIME_ROOT / "ollama-darwin.tgz"
OLLAMA_BINARY = OLLAMA_RUNTIME_ROOT / "runtime" / "ollama"
OLLAMA_ARCHIVE_BYTES = 145_747_028
OLLAMA_ARCHIVE_SHA256 = (
    "5789dd037a86adb328c72c11fc45e6c558452d07e5b50814a8bdb7b0fbdbcd81"
)
OLLAMA_BINARY_BYTES = 68_349_456
OLLAMA_BINARY_SHA256 = (
    "a3a5498d6220759ac8b2c5cea451f30107990ad486604d28ae65d911f722ac8e"
)
OLLAMA_RESTORATION_PRIVATE_BYTES = 632_307_460
SYSTEM_OLLAMA_BINARY = Path("/Applications/Ollama.app/Contents/Resources/ollama")
SYSTEM_OLLAMA_BINARY_SHA256 = (
    "acecd6ca61310c1329fed95d38ccfe05cf8d5fddbc28304fa46629e96d03c318"
)
RUNTIME_RESTORATION_AUTHORIZATION_FILE_SHA256 = (
    "df796926d9e45b73735a0e66f90bb39a987d37b9ef0a4994c328f8772cf20533"
)
RUNTIME_RESTORATION_RECEIPT_FILE_SHA256 = (
    "baf18c9a0edb6203ce3095470c002e7ac3e45cb74515f0db690012f41dc43115"
)
RUNTIME_RESTORATION_VERIFICATION_FILE_SHA256 = (
    "c4b68b6397fed6564314460e204cddb8fafb4e393dc798d293d29b0478d2a7d1"
)
EXECUTION_AMENDMENT_004_FILE_SHA256 = (
    "ce0a3a6c7b76233afadcf6a1851e750228c3ff91dfe7f1ddb12c2d26c0fd7171"
)
EXECUTION_AMENDMENT_004_TOOLING = {
    "implementation": "9c5ff77e243a37d478dd37f82ca95f6da206fa0a239bd3ee7013a36be69941fe",
    "runner": "19d1585a568ca623e57680deb28aced49a68ac41e3caaf6a9872781c1160516a",
    "test": "028461404cad53aad959dd34a73b8d2c05a3fca53eaff96bd3c96cf31bef2b48",
    "verifier": "973706ce392293c94e766a7361f4dd9776a53536dc3707e370757b2afc475dc3",
}

PAIRED_PROMPTS = 100
EXPECTED_FINAL_OUTPUTS = 200
MAXIMUM_ATTEMPTS = 220
MAXIMUM_RETRIES = 20
MAXIMUM_RESPONSE_TOKENS = 256
MAXIMUM_GENERATED_OUTPUT_TOKENS = MAXIMUM_ATTEMPTS * MAXIMUM_RESPONSE_TOKENS
CONTINUATION_MAXIMUM_ATTEMPTS = 260
CONTINUATION_MAXIMUM_RETRIES = 60
MAXIMUM_RETAINED_CANDIDATE_SUPERVISED_TOKENS = 30_000
MAXIMUM_RUNTIME_MINUTES = 360
MAXIMUM_PRIVATE_WRITE_BYTES = 24 * 1024**3
OWNER_REVIEW_ROWS = 40

CONTINUATION_AUTHORIZATION_SCHEMA_VERSION = (
    "lumi.sft-stronger-teacher-ab-continuation-authorization.v0.1"
)
CONTINUATION_AUTHORIZATION_STATEMENT = (
    "I approve the exact result-preserving continuation proposed above, raising the "
    "cumulative inference ceiling to 260 and runtime-retry ceiling to 60. Run Mac-local "
    "first and stop fail-closed on memory or thermal pressure. Do not migrate to a GPU "
    "server without binding the exact server, GPU, files, runtime, and cost in a separate "
    "authorization."
)
CONTINUATION_LIMITS = {
    "cumulative_maximum_generated_output_tokens": MAXIMUM_GENERATED_OUTPUT_TOKENS,
    "cumulative_maximum_inference_attempts": CONTINUATION_MAXIMUM_ATTEMPTS,
    "cumulative_maximum_runtime_minutes": MAXIMUM_RUNTIME_MINUTES,
    "cumulative_maximum_runtime_retries": CONTINUATION_MAXIMUM_RETRIES,
    "maximum_additional_inference_attempts_from_boundary": 144,
    "maximum_additional_runtime_retries_from_boundary": 40,
    "maximum_private_write_bytes": MAXIMUM_PRIVATE_WRITE_BYTES,
    "maximum_response_tokens": MAXIMUM_RESPONSE_TOKENS,
}

MODEL_KEYS = ("baseline", "candidate")
MODEL_ALIASES = {
    "baseline": "lumi-qwen3-8b-teacher-canary:7c41481-q4km",
    "candidate": "lumi-qwen3-14b-teacher-ab:c75e7b2-q4km",
}

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": False,
    "cloud_or_paid_gpu_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "human_authorship_claim_authorized": False,
    "local_ollama_inference_authorized": True,
    "model_download_authorized": True,
    "package_installation_authorized": False,
    "production_promotion_authorized": False,
    "sft_training_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "expected_final_outputs": EXPECTED_FINAL_OUTPUTS,
    "maximum_generated_output_tokens": MAXIMUM_GENERATED_OUTPUT_TOKENS,
    "maximum_inference_attempts": MAXIMUM_ATTEMPTS,
    "maximum_network_bytes": preflight.REMOTE_OBJECT_BYTES,
    "maximum_private_write_bytes": MAXIMUM_PRIVATE_WRITE_BYTES,
    "maximum_response_tokens": MAXIMUM_RESPONSE_TOKENS,
    "maximum_retained_candidate_supervised_tokens": (
        MAXIMUM_RETAINED_CANDIDATE_SUPERVISED_TOKENS
    ),
    "maximum_runtime_minutes": MAXIMUM_RUNTIME_MINUTES,
    "owner_review_paired_rows": OWNER_REVIEW_ROWS,
    "paired_prompts": PAIRED_PROMPTS,
    "runtime_or_incomplete_retries": MAXIMUM_RETRIES,
}
EXPECTED_GATES = {
    "candidate_minimum_advantage_over_baseline": 0.15,
    "candidate_minimum_machine_acceptance_rate": 0.90,
    "candidate_minimum_owner_acceptance_rate": 0.90,
    "candidate_minimum_owner_preference_rate": 0.70,
    "candidate_minimum_per_cell_acceptance_rate": 0.80,
    "critical_factual_safety_or_tool_failures": 0,
}
EXPECTED_BINDINGS = frozenset(
    {
        "authorization",
        "baseline_model_file",
        "baseline_ollama_manifest",
        "base_execution_plan",
        "base_schedule_source",
        "benchmark_signatures",
        "chat_template",
        "data_quality",
        "owner_reviewed_quality_exemplars",
        "preflight_plan",
        "preflight_report",
        "preflight_source",
        "preflight_verification",
        "quality_exemplar_owner_review",
        "tokenizer_config",
        "tokenizer_json",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Any, *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _resolve(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_bindings(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("stronger-teacher execution binding set changed")
    root = Path(repository_root).resolve()
    resolved: dict[str, Path] = {}
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping) or not isinstance(binding.get("path"), str):
            raise ValueError(f"invalid stronger-teacher execution binding: {name}")
        path = _resolve(root, str(binding["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"stronger-teacher execution binding missing: {name}")
        if scale.file_sha256(path) != binding.get("sha256"):
            raise ValueError(f"stronger-teacher execution binding changed: {name}")
        resolved[name] = path
    return resolved


def _candidate_objects_from_plan(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    objects = plan.get("candidate_objects")
    if not isinstance(objects, list) or len(objects) != 3:
        raise ValueError("candidate object inventory changed")
    expected = []
    for source in preflight.EXPECTED_REMOTE_OBJECTS:
        expected.append(
            {
                "bytes": source["bytes"],
                "path": f"model/{source['path']}",
                "sha256": source["sha256"],
                "url": source["url"],
            }
        )
    if objects != expected:
        raise ValueError("candidate object inventory differs from the sealed preflight")
    return [dict(item) for item in objects]


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported stronger-teacher paired A/B plan")
    if plan.get("status") != "authorized_private_mac_local_stronger_teacher_paired_ab":
        raise ValueError("stronger-teacher paired A/B status changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("stronger-teacher paired A/B authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("stronger-teacher paired A/B limits changed")
    if plan.get("gates") != EXPECTED_GATES:
        raise ValueError("stronger-teacher paired A/B gates changed")
    execution = plan.get("execution", {})
    expected_execution = {
        "baseline_alias": MODEL_ALIASES["baseline"],
        "candidate_alias": MODEL_ALIASES["candidate"],
        "endpoint": LOCAL_ENDPOINT,
        "maximum_loaded_models": 1,
        "model_order": ["baseline", "candidate"],
        "ollama_version": "0.32.5",
        "selected_record_ids_sha256": preflight.SELECTED_IDS_SHA256,
        "selected_schedule_internal_sha256": preflight.SELECTED_SCHEDULE_SHA256,
        "sequential_single_model": True,
        "thermal_probe_cache_path": "/private/tmp/lumi-stronger-teacher-ab-swift-cache",
    }
    if execution != expected_execution:
        raise ValueError("stronger-teacher paired A/B execution contract changed")
    if plan.get("models", {}).get("baseline") != {
        "file_bytes": preflight.BASELINE_MODEL["file_bytes"],
        "file_sha256": preflight.BASELINE_MODEL["file_sha256"],
        "id": preflight.BASELINE_MODEL["model_id"],
        "quantization": "Q4_K_M",
        "revision": preflight.BASELINE_MODEL["revision"],
    }:
        raise ValueError("baseline model binding changed")
    if plan.get("models", {}).get("candidate") != {
        "file_bytes": preflight.CANDIDATE_MODEL["file_bytes"],
        "file_sha256": preflight.CANDIDATE_MODEL["file_sha256"],
        "id": preflight.CANDIDATE_MODEL["model_id"],
        "quantization": "Q4_K_M",
        "revision": preflight.CANDIDATE_MODEL["revision"],
        "weights_license": "apache-2.0",
    }:
        raise ValueError("candidate model binding changed")
    _candidate_objects_from_plan(plan)
    resolved = _resolve_bindings(plan, repository_root)
    preflight_plan = preflight.load_stronger_teacher_ab_preflight_plan(
        resolved["preflight_plan"], repository_root=repository_root
    )
    preflight_report = _load_json(resolved["preflight_report"])
    if (
        preflight_report.get("status")
        != "verified_rights_safe_qwen14b_paired_ab_preflight_no_model_download_or_inference"
    ):
        raise ValueError("stronger-teacher preflight report status changed")
    preflight_verification = _load_json(resolved["preflight_verification"])
    if not all(preflight_verification.get("checks", {}).values()):
        raise ValueError("stronger-teacher preflight verification is not clean")
    authorization = _load_json(resolved["authorization"])
    if (
        authorization.get("authorization_statement")
        != preflight_plan.get("next_gate", {}).get("exact_authorization_statement")
    ):
        raise ValueError("owner authorization statement changed")
    expected_authorization_limits = dict(EXPECTED_LIMITS)
    expected_authorization_limits.pop("maximum_response_tokens")
    expected_authorization_limits.pop("runtime_or_incomplete_retries")
    if authorization.get("limits") != expected_authorization_limits:
        raise ValueError("owner authorization limits changed")
    scale.load_plan(resolved["base_execution_plan"], repository_root=repository_root)
    schedule = preflight.selected_schedule()
    if len(schedule) != PAIRED_PROMPTS:
        raise ValueError("paired schedule count changed")
    return plan


def artifact_root(plan: Mapping[str, Any], repository_root: str | Path) -> Path:
    raw = plan.get("output", {}).get("artifact_root")
    if not isinstance(raw, str):
        raise ValueError("stronger-teacher artifact root missing")
    root = _resolve(Path(repository_root).resolve(), raw)
    expected = (
        Path(repository_root).resolve().parent
        / "lumi-artifacts"
        / "4b-sft-stronger-teacher-ab-v0.1"
        / "c52325bacdb8"
        / "attempt-001"
    ).resolve()
    if root != expected:
        raise ValueError("stronger-teacher artifact root changed")
    return root


def load_continuation_authorization(
    path: str | Path, *, artifact: Path, repository_root: str | Path
) -> dict[str, Any]:
    authorization_path = Path(path).resolve()
    expected_path = (artifact / "continuation-authorization.json").resolve()
    if authorization_path != expected_path:
        raise ValueError("paired A/B continuation authorization path changed")
    authorization = _load_json(authorization_path)
    if (
        authorization.get("schema_version")
        != CONTINUATION_AUTHORIZATION_SCHEMA_VERSION
    ):
        raise ValueError("paired A/B continuation authorization schema changed")
    if authorization.get("status") != "authorized_result_preserving_mac_local_continuation":
        raise ValueError("paired A/B continuation authorization status changed")
    if authorization.get("authorization_statement") != CONTINUATION_AUTHORIZATION_STATEMENT:
        raise ValueError("paired A/B continuation authorization statement changed")
    if authorization.get("limits") != CONTINUATION_LIMITS:
        raise ValueError("paired A/B continuation limits changed")
    if authorization.get("preservation") != {
        "all_95_completed_baseline_outputs_byte_exact": True,
        "candidate_uses_baseline_final_prompt_variant": True,
        "quality_failures_retryable": False,
        "remaining_baseline_outputs": 5,
        "remaining_candidate_outputs": 100,
    }:
        raise ValueError("paired A/B continuation preservation contract changed")
    if authorization.get("prohibitions") != {
        "cloud_or_paid_gpu": True,
        "dataset_seal": True,
        "external_api": True,
        "external_distribution": True,
        "gpu_server_migration_without_separate_exact_authorization": True,
        "human_authorship_claim": True,
        "model_or_data_download": True,
        "package_installation": True,
        "production_promotion": True,
        "registry_change": True,
        "sft_training": True,
        "unrelated_access": True,
    }:
        raise ValueError("paired A/B continuation prohibition contract changed")
    if authorization.get("retry_classifier_scope") != {
        "local_endpoint_only": LOCAL_ENDPOINT,
        "no_response_transport_failures_only": True,
        "permitted_classes": [
            "RemoteDisconnected",
            "ConnectionResetError",
            "ConnectionAbortedError",
            "BrokenPipeError",
            "TimeoutError",
        ],
    }:
        raise ValueError("paired A/B continuation retry scope changed")

    boundary = authorization.get("boundary_bindings")
    if boundary != {
        "baseline_journal_file_bytes": 220342,
        "baseline_journal_file_sha256": (
            "27fb0abd7ad48e17f8c0d8afa23ceb987319dc18b24cd2208823c87d009f4200"
        ),
        "boundary_stop_report_file_sha256": (
            "a5775c4a27641ae1c2dc11509d5ba5daf9538cac00f54c5cadc6063176332135"
        ),
        "boundary_stop_verification_file_sha256": (
            "82d968a90a5ffa275246f3fc37d0263afc3a278001a446e58c3e35d8ac87259b"
        ),
        "completed_baseline_outputs": 95,
        "inference_starts": 116,
        "runtime_retries_used": 20,
    }:
        raise ValueError("paired A/B continuation boundary binding changed")
    boundary_report_path = artifact / "evidence" / "boundary-stop-report.json"
    boundary_verification_path = (
        artifact / "evidence" / "boundary-stop-verification.json"
    )
    if scale.file_sha256(boundary_report_path) != boundary[
        "boundary_stop_report_file_sha256"
    ]:
        raise ValueError("paired A/B boundary-stop report changed")
    if scale.file_sha256(boundary_verification_path) != boundary[
        "boundary_stop_verification_file_sha256"
    ]:
        raise ValueError("paired A/B boundary-stop verification changed")
    if _load_json(boundary_verification_path).get("status") != "PASS":
        raise ValueError("paired A/B boundary-stop verification is not PASS")
    journal_path = artifact / "private" / "baseline-call-journal.jsonl"
    prefix_bytes = int(boundary["baseline_journal_file_bytes"])
    journal_bytes = journal_path.read_bytes()
    if len(journal_bytes) < prefix_bytes:
        raise ValueError("paired A/B baseline journal predecessor was truncated")
    if hashlib.sha256(journal_bytes[:prefix_bytes]).hexdigest() != boundary[
        "baseline_journal_file_sha256"
    ]:
        raise ValueError("paired A/B baseline journal predecessor prefix changed")

    root = Path(repository_root).resolve()
    amendment_path = artifact / "execution-package-amendment-004.json"
    if scale.file_sha256(amendment_path) != EXECUTION_AMENDMENT_004_FILE_SHA256:
        raise ValueError("paired A/B continuation amendment 004 changed")
    amendment = _load_json(amendment_path)
    if amendment.get("schema_version") != "lumi.sft-stronger-teacher-ab-execution-amendment.v0.1":
        raise ValueError("paired A/B continuation amendment schema changed")
    if amendment.get("status") != "sealed_result_preserving_authorized_continuation":
        raise ValueError("paired A/B continuation amendment status changed")
    if amendment.get("continuation_authorization_file_sha256") != scale.file_sha256(
        authorization_path
    ):
        raise ValueError("paired A/B continuation amendment authorization binding changed")
    if amendment.get("tooling_file_sha256") != EXECUTION_AMENDMENT_004_TOOLING:
        raise ValueError("paired A/B historical continuation tooling binding changed")
    if amendment.get("limits") != CONTINUATION_LIMITS:
        raise ValueError("paired A/B continuation amendment limits changed")

    restoration_authorization_path = artifact / "runtime-restoration-authorization.json"
    if (
        scale.file_sha256(restoration_authorization_path)
        != RUNTIME_RESTORATION_AUTHORIZATION_FILE_SHA256
    ):
        raise ValueError("paired A/B runtime-restoration authorization changed")
    restoration_authorization = _load_json(restoration_authorization_path)
    if restoration_authorization.get("schema_version") != (
        "lumi.sft-stronger-teacher-ab-runtime-restoration-authorization.v0.1"
    ):
        raise ValueError("paired A/B runtime-restoration authorization schema changed")
    if restoration_authorization.get("status") != (
        "authorized_exact_private_ollama_runtime_restoration"
    ):
        raise ValueError("paired A/B runtime-restoration authorization status changed")
    if restoration_authorization.get("archive") != {
        "maximum_network_bytes": 160 * 1024**2,
        "sha256": OLLAMA_ARCHIVE_SHA256,
        "url": (
            "https://github.com/ollama/ollama/releases/download/v0.32.5/"
            "ollama-darwin.tgz"
        ),
    }:
        raise ValueError("paired A/B runtime-restoration archive binding changed")
    if restoration_authorization.get("extraction") != {
        "maximum_private_write_bytes": 1024**3,
        "maximum_runtime_minutes": 30,
        "root": str(OLLAMA_RUNTIME_ROOT),
        "system_app_modification_authorized": False,
    }:
        raise ValueError("paired A/B runtime-restoration extraction binding changed")
    if restoration_authorization.get("resume") != {
        "cumulative_maximum_generated_output_tokens": MAXIMUM_GENERATED_OUTPUT_TOKENS,
        "cumulative_maximum_inference_attempts": CONTINUATION_MAXIMUM_ATTEMPTS,
        "cumulative_maximum_runtime_minutes": MAXIMUM_RUNTIME_MINUTES,
        "cumulative_maximum_runtime_retries": CONTINUATION_MAXIMUM_RETRIES,
        "required_ollama_version": "0.32.5",
    }:
        raise ValueError("paired A/B runtime-restoration resume binding changed")
    restoration_bindings = restoration_authorization.get("bindings")
    if restoration_bindings != {
        "runtime_drift_report_file_sha256": (
            "192f74e64c322d01f21aaaea82279e0683c6dd246cacfba459acb51667ced7a5"
        ),
        "runtime_drift_verification_file_sha256": (
            "a2bfc92f9878c4476ef09676f03b0ba7e02104751d5cf4bafe9fd1e414b56a49"
        ),
    }:
        raise ValueError("paired A/B runtime-drift authorization binding changed")
    drift_report = artifact / "evidence" / "runtime-drift-stop-report.json"
    drift_verification = artifact / "evidence" / "runtime-drift-stop-verification.json"
    if scale.file_sha256(drift_report) != restoration_bindings[
        "runtime_drift_report_file_sha256"
    ]:
        raise ValueError("paired A/B runtime-drift report changed")
    if scale.file_sha256(drift_verification) != restoration_bindings[
        "runtime_drift_verification_file_sha256"
    ]:
        raise ValueError("paired A/B runtime-drift verification changed")

    restoration_receipt_path = artifact / "evidence" / "runtime-restoration-receipt.json"
    restoration_verification_path = (
        artifact / "evidence" / "runtime-restoration-verification.json"
    )
    if (
        scale.file_sha256(restoration_receipt_path)
        != RUNTIME_RESTORATION_RECEIPT_FILE_SHA256
    ):
        raise ValueError("paired A/B runtime-restoration receipt changed")
    if (
        scale.file_sha256(restoration_verification_path)
        != RUNTIME_RESTORATION_VERIFICATION_FILE_SHA256
    ):
        raise ValueError("paired A/B runtime-restoration verification changed")
    restoration_receipt = _load_json(restoration_receipt_path)
    restoration_verification = _load_json(restoration_verification_path)
    if restoration_receipt.get("status") != (
        "PASS_exact_private_ollama_v0_32_5_restored_and_smoked"
    ):
        raise ValueError("paired A/B runtime-restoration receipt is not PASS")
    if restoration_verification.get("status") != "PASS":
        raise ValueError("paired A/B runtime-restoration verification is not PASS")
    if restoration_verification.get("runtime_restoration_receipt_file_sha256") != (
        RUNTIME_RESTORATION_RECEIPT_FILE_SHA256
    ):
        raise ValueError("paired A/B runtime-restoration verification binding changed")

    if set(path.name for path in OLLAMA_RUNTIME_ROOT.iterdir()) != {
        "ollama-darwin.tgz",
        "runtime",
    }:
        raise ValueError("paired A/B private Ollama runtime root inventory changed")
    if OLLAMA_ARCHIVE.stat().st_size != OLLAMA_ARCHIVE_BYTES:
        raise ValueError("paired A/B private Ollama archive size changed")
    if scale.file_sha256(OLLAMA_ARCHIVE) != OLLAMA_ARCHIVE_SHA256:
        raise ValueError("paired A/B private Ollama archive hash changed")
    if OLLAMA_BINARY.stat().st_size != OLLAMA_BINARY_BYTES:
        raise ValueError("paired A/B private Ollama binary size changed")
    if scale.file_sha256(OLLAMA_BINARY) != OLLAMA_BINARY_SHA256:
        raise ValueError("paired A/B private Ollama binary hash changed")
    regular_files = [
        path
        for path in OLLAMA_RUNTIME_ROOT.rglob("*")
        if path.is_file() and not path.is_symlink()
    ]
    symlinks = [path for path in OLLAMA_RUNTIME_ROOT.rglob("*") if path.is_symlink()]
    if len(regular_files) != 33 or len(symlinks) != 10:
        raise ValueError("paired A/B private Ollama extraction inventory changed")
    if sum(path.stat().st_size for path in regular_files) != (
        OLLAMA_RESTORATION_PRIVATE_BYTES
    ):
        raise ValueError("paired A/B private Ollama extraction bytes changed")
    if scale.file_sha256(SYSTEM_OLLAMA_BINARY) != SYSTEM_OLLAMA_BINARY_SHA256:
        raise ValueError("installed Ollama app changed after private restoration")

    amendment_005_path = artifact / "execution-package-amendment-005.json"
    amendment_005 = _load_json(amendment_005_path)
    if amendment_005.get("schema_version") != (
        "lumi.sft-stronger-teacher-ab-execution-amendment.v0.1"
    ):
        raise ValueError("paired A/B private-runtime amendment schema changed")
    if amendment_005.get("status") != (
        "sealed_exact_private_ollama_runtime_authorized_continuation"
    ):
        raise ValueError("paired A/B private-runtime amendment status changed")
    if amendment_005.get("predecessor") != {
        "continuation_authorization_file_sha256": scale.file_sha256(
            authorization_path
        ),
        "execution_amendment_004_file_sha256": EXECUTION_AMENDMENT_004_FILE_SHA256,
        "runtime_drift_report_file_sha256": restoration_bindings[
            "runtime_drift_report_file_sha256"
        ],
        "runtime_drift_verification_file_sha256": restoration_bindings[
            "runtime_drift_verification_file_sha256"
        ],
    }:
        raise ValueError("paired A/B private-runtime predecessor binding changed")
    if amendment_005.get("runtime_restoration") != {
        "archive_file_sha256": OLLAMA_ARCHIVE_SHA256,
        "authorization_file_sha256": RUNTIME_RESTORATION_AUTHORIZATION_FILE_SHA256,
        "binary_file_sha256": OLLAMA_BINARY_SHA256,
        "binary_path": str(OLLAMA_BINARY),
        "ollama_version": "0.32.5",
        "receipt_file_sha256": RUNTIME_RESTORATION_RECEIPT_FILE_SHA256,
        "verification_file_sha256": RUNTIME_RESTORATION_VERIFICATION_FILE_SHA256,
    }:
        raise ValueError("paired A/B private-runtime restoration binding changed")
    expected_tooling = {
        "implementation": root / "src" / "lumi" / "posttraining_stronger_teacher_ab.py",
        "runner": root / "scripts" / "run_stronger_teacher_ab.py",
        "test": root / "tests" / "test_stronger_teacher_ab.py",
        "verifier": root / "scripts" / "verify_stronger_teacher_ab.py",
    }
    if amendment_005.get("tooling_file_sha256") != {
        name: scale.file_sha256(tool_path) for name, tool_path in expected_tooling.items()
    }:
        raise ValueError("paired A/B private-runtime tooling binding changed")
    if amendment_005.get("limits") != CONTINUATION_LIMITS:
        raise ValueError("paired A/B private-runtime amendment limits changed")
    if amendment_005.get("prohibitions_unchanged") is not True:
        raise ValueError("paired A/B private-runtime prohibitions changed")
    return authorization


def verify_candidate_objects(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> list[dict[str, Any]]:
    artifact = artifact_root(plan, repository_root)
    verified: list[dict[str, Any]] = []
    for item in _candidate_objects_from_plan(plan):
        path = (artifact / item["path"]).resolve()
        if not path.is_relative_to((artifact / "model").resolve()):
            raise ValueError("candidate object escaped model directory")
        if not path.is_file():
            raise FileNotFoundError(f"candidate object missing: {path}")
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"candidate object size changed: {path.name}")
        if scale.file_sha256(path) != item["sha256"]:
            raise ValueError(f"candidate object SHA-256 changed: {path.name}")
        verified.append(
            {
                "bytes": item["bytes"],
                "file_sha256": item["sha256"],
                "path": str(path),
            }
        )
    return verified


def _candidate_manifest_path() -> Path:
    return (
        Path.home()
        / ".ollama/models/manifests/registry.ollama.ai/library"
        / "lumi-qwen3-14b-teacher-ab"
        / "c75e7b2-q4km"
    )


def verify_ollama_manifest(path: Path, *, model_key: str) -> dict[str, Any]:
    manifest = _load_json(path)
    model = preflight.BASELINE_MODEL if model_key == "baseline" else preflight.CANDIDATE_MODEL
    expected_layer = {
        "mediaType": "application/vnd.ollama.image.model",
        "digest": f"sha256:{model['file_sha256']}",
        "size": model["file_bytes"],
    }
    layers = manifest.get("layers")
    if not isinstance(layers, list) or expected_layer not in layers:
        raise ValueError(f"{model_key} Ollama manifest model layer changed")
    return manifest


def _ollama_client_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "NO_PROXY": "127.0.0.1,localhost",
            "OLLAMA_HOST": "127.0.0.1:11434",
            "OLLAMA_MODELS": str(Path.home() / ".ollama" / "models"),
            "OLLAMA_NO_CLOUD": "true",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    return env


def _ollama_version() -> str:
    if OLLAMA_BINARY.stat().st_size != OLLAMA_BINARY_BYTES:
        raise RuntimeError("private Ollama binary size changed")
    if scale.file_sha256(OLLAMA_BINARY) != OLLAMA_BINARY_SHA256:
        raise RuntimeError("private Ollama binary SHA-256 changed")
    result = subprocess.run(
        [str(OLLAMA_BINARY), "--version"],
        check=False,
        capture_output=True,
        cwd=OLLAMA_BINARY.parent,
        env=_ollama_client_env(),
        text=True,
        timeout=30,
    )
    text = f"{result.stdout}\n{result.stderr}"
    if "0.32.5" not in text:
        raise RuntimeError("Ollama 0.32.5 is not available")
    request = urllib.request.Request(
        OLLAMA_SERVER_VERSION_ENDPOINT,
        headers={"User-Agent": "lumi-local-runtime-verifier/0.1"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            server_version = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"private Ollama localhost smoke failed: {exc}") from exc
    if server_version != {"version": "0.32.5"}:
        raise RuntimeError("private Ollama server version changed")
    return "0.32.5"


def prepare_candidate_alias(
    *, plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Any]:
    artifact = artifact_root(plan, repository_root)
    verified_objects = verify_candidate_objects(plan, repository_root=repository_root)
    _ollama_version()
    model_path = artifact / "model" / preflight.CANDIDATE_MODEL["file_name"]
    modelfile = artifact / "model" / "Modelfile"
    expected_modelfile = (
        f"FROM {model_path}\n"
        "PARAMETER num_ctx 4096\n"
        "PARAMETER temperature 0.2\n"
        "PARAMETER top_k 40\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER repeat_penalty 1.05\n"
    )
    if modelfile.exists() and modelfile.read_text(encoding="utf-8") != expected_modelfile:
        raise ValueError("candidate Modelfile changed")
    if not modelfile.exists():
        modelfile.write_text(expected_modelfile, encoding="utf-8")
    manifest_path = _candidate_manifest_path()
    if not manifest_path.is_file():
        result = subprocess.run(
            [
                str(OLLAMA_BINARY),
                "create",
                MODEL_ALIASES["candidate"],
                "--file",
                str(modelfile),
            ],
            check=False,
            capture_output=True,
            cwd=OLLAMA_BINARY.parent,
            env=_ollama_client_env(),
            text=True,
            timeout=1800,
        )
        if result.returncode != 0:
            raise RuntimeError(f"candidate Ollama import failed: {result.stderr.strip()}")
    manifest = verify_ollama_manifest(manifest_path, model_key="candidate")
    receipt = {
        "candidate_alias": MODEL_ALIASES["candidate"],
        "candidate_manifest_file_sha256": scale.file_sha256(manifest_path),
        "candidate_manifest_internal_sha256": sha256_value(manifest),
        "modelfile_file_sha256": scale.file_sha256(modelfile),
        "objects": verified_objects,
        "ollama_version": "0.32.5",
        "schema_version": "lumi.sft-stronger-teacher-ab-model-import.v0.1",
        "status": "exact_candidate_model_imported_and_verified",
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    _write_json(artifact / "evidence" / "model-import-receipt.json", receipt)
    return receipt


def _iter_unique_files(paths: Sequence[Path]) -> list[Path]:
    seen: set[tuple[int, int]] = set()
    files: list[Path] = []
    for root in paths:
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = [path for path in root.rglob("*") if path.is_file()]
        else:
            candidates = []
        for path in candidates:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            files.append(path)
    return files


def _private_write_bytes(artifact: Path, thermal_cache: Path) -> int:
    paths = [artifact, thermal_cache, _candidate_manifest_path()]
    manifest_path = _candidate_manifest_path()
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        descriptors = [manifest.get("config", {})]
        layers = manifest.get("layers", [])
        if isinstance(layers, list):
            descriptors.extend(layers)
        for descriptor in descriptors:
            if not isinstance(descriptor, Mapping):
                continue
            digest = descriptor.get("digest")
            if isinstance(digest, str) and digest.startswith("sha256:"):
                paths.append(
                    Path.home()
                    / ".ollama/models/blobs"
                    / f"sha256-{digest.removeprefix('sha256:')}"
                )
    return sum(path.stat().st_size for path in _iter_unique_files(paths))


def runtime_safety_snapshot(artifact: Path, thermal_cache: Path) -> dict[str, int]:
    memory = teacher.memory_free_percent()
    thermal = scale._thermal_state(thermal_cache)  # noqa: SLF001
    private_bytes = _private_write_bytes(artifact, thermal_cache)
    if memory < 15:
        raise RuntimeError("memory pressure became unsafe")
    if thermal >= 2:
        raise RuntimeError("macOS thermal state became serious or critical")
    if private_bytes > MAXIMUM_PRIVATE_WRITE_BYTES:
        raise RuntimeError("paired A/B exceeded the private-write authorization")
    return {
        "memory_free_percent": memory,
        "private_write_bytes": private_bytes,
        "thermal_state": thermal,
    }


def _load_runtime_resources(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> tuple[dict[str, Any], BenchmarkRegistry, Any, Any, dict[str, tuple[str, str]]]:
    resolved = _resolve_bindings(plan, repository_root)
    quality = _load_json(resolved["data_quality"])
    benchmark = BenchmarkRegistry(
        quality["benchmark_decontamination"], Path(repository_root).resolve()
    )
    tokenizer = scale._load_direct_tokenizer(  # noqa: SLF001
        resolved["tokenizer_json"].parent
    )
    template = load_chat_template(resolved["chat_template"])
    exemplars = scale._load_quality_exemplars(  # noqa: SLF001
        resolved["owner_reviewed_quality_exemplars"]
    )
    return quality, benchmark, tokenizer, template, exemplars


def _payload(
    slot: scale.Slot,
    *,
    model_key: str,
    exemplars: Mapping[str, tuple[str, str]],
    prompt_variant: str,
    maximum_response_tokens: int = MAXIMUM_RESPONSE_TOKENS,
) -> dict[str, Any]:
    if maximum_response_tokens < MAXIMUM_RESPONSE_TOKENS:
        raise ValueError("paired A/B response-token ceiling cannot regress")
    payload = scale.ollama_payload(slot, exemplars=exemplars)
    payload["model"] = MODEL_ALIASES[model_key]
    payload["keep_alive"] = -1
    payload["options"]["num_predict"] = maximum_response_tokens
    if prompt_variant == "runtime_retry_v1":
        payload["prompt"] += (
            "\nRUNTIME_COMPLETION_RETRY: The previous response reached the fixed output "
            "ceiling. Preserve every schema, language, safety, tool, factual, and literal "
            "requirement, but keep assistant_response between 55 and 70 words so the JSON "
            f"object completes within {maximum_response_tokens} tokens."
        )
    elif prompt_variant == "runtime_retry_v2":
        marker = "<|im_end|>\n<|im_start|>assistant\n"
        if not payload["prompt"].endswith(marker):
            raise ValueError("paired A/B prompt assistant marker changed")
        instruction = (
            "\nRUNTIME_COMPLETION_RETRY: The previous response reached the fixed output "
            "ceiling. Preserve every schema, language, safety, tool, factual, and literal "
            "requirement. Write exactly 2 concise sentences and keep assistant_response "
            "between 22 and 32 words so the complete JSON object fits within "
            f"{maximum_response_tokens} tokens.\n"
        )
        payload["prompt"] = payload["prompt"][: -len(marker)] + instruction + marker
    elif prompt_variant != "initial":
        raise ValueError("paired A/B prompt variant changed")
    return payload


def _patch_provenance(row: dict[str, Any], *, model_key: str) -> dict[str, Any]:
    model = preflight.BASELINE_MODEL if model_key == "baseline" else preflight.CANDIDATE_MODEL
    provenance = dict(row["provenance"])
    provenance.update(
        {
            "ab_model_key": model_key,
            "model_file_sha256": model["file_sha256"],
            "model_id": model["model_id"],
            "model_revision": model["revision"],
            "ollama_model": MODEL_ALIASES[model_key],
            "paired_ab": True,
        }
    )
    row = dict(row)
    row["source_id"] = ROW_SOURCE_ID
    row["provenance"] = provenance
    return row


def _is_runtime_retryable(
    *, response: Mapping[str, Any] | None, error: Exception | None
) -> bool:
    if error is None:
        return False
    if isinstance(
        error,
        (
            http.client.RemoteDisconnected,
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
            TimeoutError,
        ),
    ):
        return response is None
    if isinstance(error, RuntimeError) and "local Ollama request failed" in str(error):
        return True
    if response is not None and (
        response.get("done") is not True or response.get("done_reason") != "stop"
    ):
        return True
    return False


def _continuation_reopens_local_transport_failure(
    event: Mapping[str, Any] | None,
) -> bool:
    if event is None or event.get("runtime_complete") is True:
        return False
    if event.get("raw_response_text") not in {"", None}:
        return False
    reason = str(event.get("rejection_reason") or "")
    return reason.startswith(
        (
            "RemoteDisconnected:",
            "ConnectionResetError:",
            "ConnectionAbortedError:",
            "BrokenPipeError:",
            "TimeoutError:",
        )
    )


def _critical_failure(slot: scale.Slot, rejection_reason: str | None) -> bool:
    reason = rejection_reason or ""
    if reason.startswith("ValueError:tool:"):
        return True
    if reason.startswith("ValueError:safety:") and "hidden" not in reason:
        return True
    if reason.startswith("ValueError:grounding:") and slot.task in {
        "factual_qa",
        "safety_helpfulness",
        "tool_api",
    }:
        return True
    return False


def _journal_state(
    journal: Sequence[Mapping[str, Any]], schedule: Sequence[scale.Slot], *, model_key: str
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    starts: list[dict[str, Any]] = []
    finishes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expected_ids = {slot.record_id for slot in schedule}
    for event in journal:
        if event.get("model_key") != model_key:
            raise ValueError("paired A/B journal model key changed")
        record_id = event.get("record_id")
        if record_id not in expected_ids:
            raise ValueError("paired A/B journal record ID changed")
        if event.get("event") == "call_started":
            starts.append(dict(event))
        elif event.get("event") in {"call_finished", "call_abandoned"}:
            finishes[str(record_id)].append(dict(event))
        else:
            raise ValueError("paired A/B journal event changed")
    if len(starts) - sum(len(items) for items in finishes.values()) > 1:
        raise ValueError("paired A/B journal has multiple unresolved calls")
    return starts, dict(finishes)


def _final_event(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not events:
        return None
    event = events[-1]
    return event if event.get("final_for_slot") is True else None


def _all_journal_attempts(artifact: Path) -> int:
    return sum(
        len(
            [
                row
                for row in _read_jsonl(artifact / "private" / f"{model_key}-call-journal.jsonl")
                if row.get("event") == "call_started"
            ]
        )
        for model_key in MODEL_KEYS
    )


def _runtime_retries_used(artifact: Path) -> int:
    starts: list[tuple[str, str]] = []
    for model_key in MODEL_KEYS:
        starts.extend(
            (model_key, str(row["record_id"]))
            for row in _read_jsonl(
                artifact / "private" / f"{model_key}-call-journal.jsonl"
            )
            if row.get("event") == "call_started"
        )
    return len(starts) - len(set(starts))


def _baseline_prompt_variant(artifact: Path, record_id: str) -> str:
    journal = _read_jsonl(artifact / "private" / "baseline-call-journal.jsonl")
    _, finishes = _journal_state(
        journal, preflight.selected_schedule(), model_key="baseline"
    )
    final = _final_event(finishes.get(record_id, []))
    if final is None or final.get("runtime_complete") is not True:
        raise ValueError("candidate phase lacks a complete paired baseline output")
    attempt_index = int(final["attempt_index"])
    starts = [
        row
        for row in journal
        if row.get("event") == "call_started"
        and row.get("record_id") == record_id
        and row.get("attempt_index") == attempt_index
    ]
    if len(starts) != 1:
        raise ValueError("paired baseline prompt variant start is ambiguous")
    explicit = starts[0].get("prompt_variant")
    if explicit in {"initial", "runtime_retry_v1", "runtime_retry_v2"}:
        return str(explicit)
    return "initial" if attempt_index == 1 else "runtime_retry_v1"


def _generated_output_tokens(artifact: Path) -> int:
    total = 0
    for model_key in MODEL_KEYS:
        for event in _read_jsonl(
            artifact / "private" / f"{model_key}-call-journal.jsonl"
        ):
            if event.get("event") == "call_finished":
                count = event.get("telemetry", {}).get("eval_count")
                if isinstance(count, int):
                    total += count
    return total


def _unload_model(model_key: str) -> None:
    response = teacher._post_local_json(  # noqa: SLF001
        LOCAL_ENDPOINT,
        {"model": MODEL_ALIASES[model_key], "keep_alive": 0},
        timeout=120,
    )
    if response.get("done") is not True:
        raise RuntimeError(f"failed to unload {model_key} model")


def _latency_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    values = sorted(
        float(event["telemetry"]["elapsed_seconds"])
        for event in events
        if event.get("event") == "call_finished"
        and isinstance(event.get("telemetry"), Mapping)
        and isinstance(event["telemetry"].get("elapsed_seconds"), (int, float))
    )
    if not values:
        return {"measured_calls": 0, "median_seconds": 0.0, "p90_seconds": 0.0}
    p90_index = min(len(values) - 1, max(0, int(0.9 * len(values)) - 1))
    return {
        "measured_calls": len(values),
        "median_seconds": round(statistics.median(values), 6),
        "p90_seconds": round(values[p90_index], 6),
    }


def _run_model_phase(
    *,
    model_key: str,
    schedule: Sequence[scale.Slot],
    artifact: Path,
    quality: Mapping[str, Any],
    benchmark: BenchmarkRegistry,
    tokenizer: Any,
    template: Any,
    exemplars: Mapping[str, tuple[str, str]],
    thermal_cache: Path,
    started_unix: int,
    maximum_attempts: int,
    maximum_retries: int,
    continuation_authorized: bool,
    maximum_response_tokens: int = MAXIMUM_RESPONSE_TOKENS,
) -> dict[str, Any]:
    journal_path = artifact / "private" / f"{model_key}-call-journal.jsonl"
    journal = _read_jsonl(journal_path)
    starts, finishes = _journal_state(journal, schedule, model_key=model_key)
    unfinished = len(starts) - sum(len(items) for items in finishes.values())
    if unfinished == 1:
        unresolved = starts[-1]
        _append_jsonl(
            journal_path,
            {
                "attempt_index": unresolved["attempt_index"],
                "event": "call_abandoned",
                "final_for_slot": False,
                "model_key": model_key,
                "record_id": unresolved["record_id"],
                "retryable_runtime_failure": True,
            },
        )
        journal = _read_jsonl(journal_path)
        starts, finishes = _journal_state(journal, schedule, model_key=model_key)

    accepted_rows: list[dict[str, Any]] = []
    for slot in schedule:
        final = _final_event(finishes.get(slot.record_id, []))
        if final is not None and final.get("accepted") is True:
            row = final.get("row")
            if not isinstance(row, dict):
                raise ValueError("accepted paired A/B event lacks row")
            accepted_rows.append(dict(row))
    exact, near, tokens = scale._screening_state(  # noqa: SLF001
        accepted_rows,
        quality=quality,
        benchmark=benchmark,
        tokenizer=tokenizer,
        template=template,
    )
    snapshots: list[dict[str, Any]] = []
    snapshots.append(
        {"after_attempts": _all_journal_attempts(artifact), **runtime_safety_snapshot(artifact, thermal_cache)}
    )

    for slot in schedule:
        existing_final = _final_event(finishes.get(slot.record_id, []))
        reopened_transport_failure = (
            continuation_authorized
            and _continuation_reopens_local_transport_failure(existing_final)
        )
        if existing_final is not None and (
            existing_final.get("runtime_complete") is True
            or (
                existing_final.get("retryable_runtime_failure") is not True
                and not reopened_transport_failure
            )
        ):
            continue
        while True:
            if time.time() - started_unix > MAXIMUM_RUNTIME_MINUTES * 60:
                raise RuntimeError("paired A/B reached its runtime boundary")
            global_attempt = _all_journal_attempts(artifact) + 1
            if global_attempt > maximum_attempts:
                raise RuntimeError("paired A/B reached its inference-attempt boundary")
            attempt_index = len(finishes.get(slot.record_id, [])) + 1
            runtime_retry = attempt_index > 1
            if runtime_retry and _runtime_retries_used(artifact) >= maximum_retries:
                raise RuntimeError("paired A/B reached its runtime-retry boundary")
            if runtime_retry:
                prompt_variant = "runtime_retry_v2"
            elif model_key == "candidate":
                prompt_variant = _baseline_prompt_variant(artifact, slot.record_id)
            else:
                prompt_variant = "initial"
            payload = _payload(
                slot,
                model_key=model_key,
                exemplars=exemplars,
                prompt_variant=prompt_variant,
                maximum_response_tokens=maximum_response_tokens,
            )
            started_event = {
                "attempt_index": attempt_index,
                "event": "call_started",
                "global_attempt": global_attempt,
                "model_key": model_key,
                "prompt_variant": prompt_variant,
                "record_id": slot.record_id,
                "request_sha256": sha256_value(payload),
                "started_unix": int(time.time()),
            }
            _append_jsonl(journal_path, started_event)
            call_started = time.monotonic()
            response: dict[str, Any] | None = None
            raw_response = ""
            row: dict[str, Any] | None = None
            token_counts: dict[str, int] | None = None
            accepted = False
            error: Exception | None = None
            try:
                response = teacher._post_local_json(  # noqa: SLF001
                    LOCAL_ENDPOINT, payload, timeout=300
                )
                if isinstance(response.get("response"), str):
                    raw_response = str(response["response"])
                row, raw_response = scale._generation_result(slot, response)  # noqa: SLF001
                row = _patch_provenance(row, model_key=model_key)
                token_counts = scale._screen_row(  # noqa: SLF001
                    row,
                    exact=exact,
                    near=near,
                    tokens=tokens,
                    benchmark=benchmark,
                    tokenizer=tokenizer,
                    template=template,
                )
                accepted = True
            except Exception as caught:  # every rejection is recorded, never hidden
                error = caught
            runtime_complete = bool(
                response is not None
                and response.get("done") is True
                and response.get("done_reason") == "stop"
                and isinstance(response.get("response"), str)
            )
            retryable = _is_runtime_retryable(response=response, error=error)
            retry_boundary_reached = _runtime_retries_used(artifact) >= maximum_retries
            final_for_slot = (
                runtime_complete
                or not retryable
                or retry_boundary_reached
                or global_attempt >= maximum_attempts
            )
            rejection_reason = None if error is None else f"{type(error).__name__}:{error}"
            elapsed = time.monotonic() - call_started
            terminal = {
                "accepted": accepted,
                "attempt_index": attempt_index,
                "critical_failure": _critical_failure(slot, rejection_reason),
                "event": "call_finished",
                "final_for_slot": final_for_slot,
                "model_key": model_key,
                "raw_response_text": raw_response,
                "record_id": slot.record_id,
                "rejection_reason": rejection_reason,
                "retryable_runtime_failure": retryable,
                "row": row,
                "runtime_complete": runtime_complete,
                "telemetry": {
                    "elapsed_seconds": round(elapsed, 6),
                    "eval_count": None if response is None else response.get("eval_count"),
                    "eval_duration_ns": None if response is None else response.get("eval_duration"),
                    "load_duration_ns": None if response is None else response.get("load_duration"),
                    "prompt_eval_count": None if response is None else response.get("prompt_eval_count"),
                    "prompt_eval_duration_ns": None if response is None else response.get("prompt_eval_duration"),
                    "response_envelope_sha256": None if response is None else sha256_value(response),
                },
                "token_counts": token_counts,
            }
            _append_jsonl(journal_path, terminal)
            finishes.setdefault(slot.record_id, []).append(terminal)
            if accepted and row is not None:
                accepted_rows.append(row)
            generated = _generated_output_tokens(artifact)
            if generated > MAXIMUM_GENERATED_OUTPUT_TOKENS:
                raise RuntimeError("paired A/B exceeded generated-output-token authorization")
            attempts = _all_journal_attempts(artifact)
            current_safety = runtime_safety_snapshot(artifact, thermal_cache)
            if attempts % 10 == 0:
                completed = sum(
                    _final_event(finishes.get(item.record_id, [])) is not None
                    for item in schedule
                )
                print(
                    f"model={model_key} final={completed}/{PAIRED_PROMPTS} "
                    f"accepted={len(accepted_rows)} attempts={attempts}/{maximum_attempts} "
                    f"generated_tokens={generated}/{MAXIMUM_GENERATED_OUTPUT_TOKENS}",
                    flush=True,
                )
                snapshots.append(
                    {"after_attempts": attempts, **current_safety}
                )
            if final_for_slot:
                if not runtime_complete:
                    raise RuntimeError("paired A/B output-count shortfall after runtime retry")
                break

    _unload_model(model_key)
    return {
        "accepted_rows": accepted_rows,
        "safety_snapshots": snapshots,
        "supervised_tokens": dict(tokens),
    }


def _final_events(
    artifact: Path, schedule: Sequence[scale.Slot], *, model_key: str
) -> dict[str, dict[str, Any]]:
    journal = _read_jsonl(artifact / "private" / f"{model_key}-call-journal.jsonl")
    _, finishes = _journal_state(journal, schedule, model_key=model_key)
    result: dict[str, dict[str, Any]] = {}
    for slot in schedule:
        final = _final_event(finishes.get(slot.record_id, []))
        if final is None:
            raise ValueError(f"paired A/B lacks final output: {model_key}:{slot.record_id}")
        result[slot.record_id] = dict(final)
    return result


def _model_audit(
    artifact: Path, schedule: Sequence[scale.Slot], *, model_key: str
) -> dict[str, Any]:
    finals = _final_events(artifact, schedule, model_key=model_key)
    accepted = [event for event in finals.values() if event.get("accepted") is True]
    slot_by_id = {slot.record_id: slot for slot in schedule}
    cell_totals = Counter((slot.task, slot.language_mode) for slot in schedule)
    cell_accepted = Counter(
        (
            slot_by_id[record_id].task,
            slot_by_id[record_id].language_mode,
        )
        for record_id, event in finals.items()
        if event.get("accepted") is True
    )
    cell_rates = {
        f"{task}/{language}": round(
            cell_accepted[(task, language)] / cell_totals[(task, language)], 6
        )
        for task in scale.TASKS
        for language in scale.LANGUAGE_MODES
    }
    critical = sum(event.get("critical_failure") is True for event in finals.values())
    supervised = Counter()
    for event in accepted:
        row = event["row"]
        supervised[str(row["split"])] += int(event["token_counts"]["supervised_tokens"])
    attempts = [
        event
        for event in _read_jsonl(
            artifact / "private" / f"{model_key}-call-journal.jsonl"
        )
        if event.get("event") == "call_started"
    ]
    finished = [
        event
        for event in _read_jsonl(
            artifact / "private" / f"{model_key}-call-journal.jsonl"
        )
        if event.get("event") == "call_finished"
    ]
    rejection_counts = Counter(
        str(event.get("rejection_reason") or "accepted")
        for event in finals.values()
        if event.get("accepted") is not True
    )
    return {
        "acceptance_rate": round(len(accepted) / PAIRED_PROMPTS, 6),
        "accepted_outputs": len(accepted),
        "attempts": len(attempts),
        "cell_acceptance_rates": cell_rates,
        "critical_factual_safety_or_tool_failures": critical,
        "final_outputs": len(finals),
        "latency": _latency_summary(finished),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "retained_supervised_tokens": dict(supervised),
    }


def _replay_model_rows(
    *,
    artifact: Path,
    schedule: Sequence[scale.Slot],
    model_key: str,
    quality: Mapping[str, Any],
    benchmark: BenchmarkRegistry,
    tokenizer: Any,
    template: Any,
) -> None:
    finals = _final_events(artifact, schedule, model_key=model_key)
    rows: list[dict[str, Any]] = []
    for slot in schedule:
        event = finals[slot.record_id]
        if event.get("runtime_complete") is not True:
            raise ValueError("paired A/B final output is not runtime-complete")
        if event.get("accepted") is not True:
            continue
        row = event.get("row")
        if not isinstance(row, dict):
            raise ValueError("accepted paired A/B output lacks a row")
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("accepted paired A/B row lacks messages")
        response = messages[-1].get("content")
        output = {
            "assistant_response": response,
            "safety_label": slot.safety_label,
            "scenario_id": slot.scenario_id,
            "tool_call": (
                None
                if slot.tool_name is None
                else {"name": slot.tool_name, "arguments": slot.tool_arguments}
            ),
            "tool_result": slot.tool_result,
            "user_prompt": slot.user_prompt,
        }
        rebuilt = scale._validate_generated_output(output, slot)  # noqa: SLF001
        rebuilt = _patch_provenance(rebuilt, model_key=model_key)
        if rebuilt != row:
            raise ValueError("accepted paired A/B row does not reproduce")
        rows.append(dict(row))
    _, _, supervised = scale._screening_state(  # noqa: SLF001
        rows,
        quality=quality,
        benchmark=benchmark,
        tokenizer=tokenizer,
        template=template,
    )
    expected = _model_audit(artifact, schedule, model_key=model_key)[
        "retained_supervised_tokens"
    ]
    if dict(supervised) != {"train": expected.get("train", 0), "validation": expected.get("validation", 0)}:
        raise ValueError("paired A/B supervised-token replay changed")


def _review_slots(schedule: Sequence[scale.Slot]) -> list[scale.Slot]:
    by_cell: dict[tuple[str, str], list[scale.Slot]] = defaultdict(list)
    for slot in schedule:
        by_cell[(slot.task, slot.language_mode)].append(slot)
    selected: list[scale.Slot] = []
    for task in scale.TASKS:
        for language in scale.LANGUAGE_MODES:
            choices = sorted(
                by_cell[(task, language)],
                key=lambda slot: hashlib.sha256(
                    f"lumi-stronger-teacher-owner-v0.1:{slot.record_id}".encode()
                ).hexdigest(),
            )
            selected.extend(choices[:2])
    if len(selected) != OWNER_REVIEW_ROWS:
        raise ValueError("paired owner-review sample is not exactly 40 rows")
    return sorted(selected, key=lambda slot: slot.record_id)


def _display_response(event: Mapping[str, Any]) -> str:
    row = event.get("row")
    if isinstance(row, Mapping):
        messages = row.get("messages")
        if isinstance(messages, list) and messages:
            return str(messages[-1].get("content", ""))
    return str(event.get("raw_response_text") or "[No complete response]")


def _paired_review_rows(
    artifact: Path, schedule: Sequence[scale.Slot]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline = _final_events(artifact, schedule, model_key="baseline")
    candidate = _final_events(artifact, schedule, model_key="candidate")
    rows: list[dict[str, Any]] = []
    key: dict[str, Any] = {}
    for slot in _review_slots(schedule):
        candidate_is_a = int(
            hashlib.sha256(f"lumi-ab-side-v0.1:{slot.record_id}".encode()).hexdigest(), 16
        ) % 2 == 0
        models = {
            "A": "candidate" if candidate_is_a else "baseline",
            "B": "baseline" if candidate_is_a else "candidate",
        }
        events = {"baseline": baseline[slot.record_id], "candidate": candidate[slot.record_id]}
        rows.append(
            {
                "language_mode": slot.language_mode,
                "record_id": slot.record_id,
                "response_a": _display_response(events[models["A"]]),
                "response_a_machine_accepted": bool(events[models["A"]].get("accepted")),
                "response_b": _display_response(events[models["B"]]),
                "response_b_machine_accepted": bool(events[models["B"]].get("accepted")),
                "split": slot.split,
                "task": slot.task,
                "user_prompt": slot.user_prompt,
            }
        )
        key[slot.record_id] = models
    return rows, key


def _review_html(rows: Sequence[Mapping[str, Any]], sample_sha256: str) -> str:
    payload = json.dumps(list(rows), ensure_ascii=False).replace("</", "<\\/")
    escaped_sha = html.escape(sample_sha256)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lumi stronger-teacher paired review</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:auto;padding:20px;background:#f6f7f9}}.card{{background:white;padding:18px;margin:16px 0;border-radius:12px;border:1px solid #ddd}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}pre{{white-space:pre-wrap;font:inherit;background:#f8f8f8;padding:12px;border-radius:8px}}label{{display:block;margin:8px 0}}button{{padding:12px 18px;margin:8px 8px 8px 0}}@media(max-width:700px){{.pair{{grid-template-columns:1fr}}}}</style></head>
<body><h1>Lumi stronger-teacher blind paired review</h1><p>Review all 40 pairs. Models are blinded. Sample SHA-256: <code>{escaped_sha}</code></p><div id="app"></div>
<button onclick="download()">Download decisions JSON</button>
<script>const rows={payload};const decisions={{}};const app=document.getElementById('app');
for(const [i,r] of rows.entries()){{const c=document.createElement('section');c.className='card';c.innerHTML=`<h2>${{i+1}}/40 — ${{r.task}} · ${{r.language_mode}}</h2><h3>Prompt</h3><pre>${{esc(r.user_prompt)}}</pre><div class="pair"><div><h3>Response A</h3><pre>${{esc(r.response_a)}}</pre><label><input type="checkbox" data-id="${{r.record_id}}" data-field="a_acceptable"> A acceptable</label></div><div><h3>Response B</h3><pre>${{esc(r.response_b)}}</pre><label><input type="checkbox" data-id="${{r.record_id}}" data-field="b_acceptable"> B acceptable</label></div></div><label>Preference: <select data-id="${{r.record_id}}" data-field="preference"><option value="">Choose</option><option value="A">A</option><option value="B">B</option><option value="tie">Tie</option><option value="neither">Neither</option></select></label>`;app.appendChild(c);decisions[r.record_id]={{record_id:r.record_id,a_acceptable:false,b_acceptable:false,preference:""}};}}
document.addEventListener('change',e=>{{const id=e.target.dataset.id,field=e.target.dataset.field;if(!id||!field)return;decisions[id][field]=e.target.type==='checkbox'?e.target.checked:e.target.value;}});
function esc(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function download(){{const values=Object.values(decisions);if(values.some(x=>!x.preference)){{alert('Review all 40 preference fields first.');return;}}const out={{schema_version:'lumi.sft-stronger-teacher-ab-owner-decisions.v0.1',sample_sha256:'{escaped_sha}',decisions:values}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}}));a.download='lumi-stronger-teacher-ab-owner-decisions-v0.1.json';a.click();}}
</script></body></html>"""


def _materialize_review(
    artifact: Path, schedule: Sequence[scale.Slot]
) -> dict[str, Any]:
    rows, blind_key = _paired_review_rows(artifact, schedule)
    sample_sha = sha256_value(rows)
    sample_path = artifact / "private" / "owner-review-pairs-40.jsonl"
    key_path = artifact / "private" / "owner-review-blind-key.json"
    html_path = artifact / "review" / "owner-review-pairs-40.html"
    _write_jsonl(sample_path, rows)
    _write_json(key_path, blind_key)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_review_html(rows, sample_sha), encoding="utf-8")
    return {
        "blind_key_file_sha256": scale.file_sha256(key_path),
        "html_file_sha256": scale.file_sha256(html_path),
        "records": len(rows),
        "sample_file_sha256": scale.file_sha256(sample_path),
        "sample_internal_sha256": sample_sha,
        "status": "pending_owner_review",
    }


def _tooling_hashes(repository_root: Path) -> dict[str, str]:
    paths = {
        "downloader": "scripts/download_stronger_teacher_ab_objects.py",
        "implementation": "src/lumi/posttraining_stronger_teacher_ab.py",
        "model_preparer": "scripts/prepare_stronger_teacher_ab_model.py",
        "runner": "scripts/run_stronger_teacher_ab.py",
        "test": "tests/test_stronger_teacher_ab.py",
        "verifier": "scripts/verify_stronger_teacher_ab.py",
    }
    return {name: scale.file_sha256(repository_root / path) for name, path in paths.items()}


def run_paired_ab(
    *,
    plan_path: str | Path,
    continuation_authorization_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_file = Path(plan_path).resolve()
    plan = load_plan(plan_file, repository_root=root)
    artifact = artifact_root(plan, root)
    continuation_file = Path(continuation_authorization_path).resolve()
    load_continuation_authorization(
        continuation_file, artifact=artifact, repository_root=root
    )
    maximum_attempts = CONTINUATION_MAXIMUM_ATTEMPTS
    maximum_retries = CONTINUATION_MAXIMUM_RETRIES
    verify_candidate_objects(plan, repository_root=root)
    resolved = _resolve_bindings(plan, root)
    verify_ollama_manifest(resolved["baseline_ollama_manifest"], model_key="baseline")
    verify_ollama_manifest(_candidate_manifest_path(), model_key="candidate")
    _ollama_version()
    state_path = artifact / "private" / "execution-state.json"
    if state_path.is_file():
        state = _load_json(state_path)
    else:
        state = {
            "plan_file_sha256": scale.file_sha256(plan_file),
            "schema_version": "lumi.sft-stronger-teacher-ab-state.v0.1",
            "selected_schedule_internal_sha256": preflight.SELECTED_SCHEDULE_SHA256,
            "started_unix": int(time.time()),
        }
        _write_json(state_path, state)
    if state.get("plan_file_sha256") != scale.file_sha256(plan_file):
        raise ValueError("paired A/B execution state plan changed")
    started_unix = int(state["started_unix"])
    if time.time() - started_unix > MAXIMUM_RUNTIME_MINUTES * 60:
        raise RuntimeError("paired A/B runtime authorization expired")
    schedule = preflight.selected_schedule()
    quality, benchmark, tokenizer, template, exemplars = _load_runtime_resources(
        plan, repository_root=root
    )
    thermal_cache = Path(plan["execution"]["thermal_probe_cache_path"])
    safety_snapshots: list[dict[str, Any]] = []
    for model_key in MODEL_KEYS:
        result = _run_model_phase(
            model_key=model_key,
            schedule=schedule,
            artifact=artifact,
            quality=quality,
            benchmark=benchmark,
            tokenizer=tokenizer,
            template=template,
            exemplars=exemplars,
            thermal_cache=thermal_cache,
            started_unix=started_unix,
            maximum_attempts=maximum_attempts,
            maximum_retries=maximum_retries,
            continuation_authorized=True,
        )
        safety_snapshots.extend(
            {"model_key": model_key, **snapshot}
            for snapshot in result["safety_snapshots"]
        )
    baseline_audit = _model_audit(artifact, schedule, model_key="baseline")
    candidate_audit = _model_audit(artifact, schedule, model_key="candidate")
    advantage = round(
        candidate_audit["acceptance_rate"] - baseline_audit["acceptance_rate"], 6
    )
    machine_checks = {
        "candidate_advantage": advantage >= EXPECTED_GATES[
            "candidate_minimum_advantage_over_baseline"
        ],
        "candidate_machine_acceptance": candidate_audit["acceptance_rate"]
        >= EXPECTED_GATES["candidate_minimum_machine_acceptance_rate"],
        "candidate_per_cell_acceptance": min(
            candidate_audit["cell_acceptance_rates"].values()
        )
        >= EXPECTED_GATES["candidate_minimum_per_cell_acceptance_rate"],
        "candidate_supervised_token_ceiling": sum(
            candidate_audit["retained_supervised_tokens"].values()
        )
        <= MAXIMUM_RETAINED_CANDIDATE_SUPERVISED_TOKENS,
        "critical_factual_safety_tool_failures": candidate_audit[
            "critical_factual_safety_or_tool_failures"
        ]
        == 0,
        "exact_final_output_count": (
            baseline_audit["final_outputs"] + candidate_audit["final_outputs"]
            == EXPECTED_FINAL_OUTPUTS
        ),
        "inference_attempt_ceiling": _all_journal_attempts(artifact)
        <= maximum_attempts,
        "output_token_ceiling": _generated_output_tokens(artifact)
        <= MAXIMUM_GENERATED_OUTPUT_TOKENS,
    }
    review = _materialize_review(artifact, schedule)
    report: dict[str, Any] = {
        "authorization_file_sha256": scale.file_sha256(resolved["authorization"]),
        "continuation": {
            "authorization_file_sha256": scale.file_sha256(continuation_file),
            "boundary_stop_report_file_sha256": scale.file_sha256(
                artifact / "evidence" / "boundary-stop-report.json"
            ),
            "boundary_stop_verification_file_sha256": scale.file_sha256(
                artifact / "evidence" / "boundary-stop-verification.json"
            ),
            "cumulative_maximum_inference_attempts": maximum_attempts,
            "cumulative_maximum_runtime_retries": maximum_retries,
            "execution_amendment_004_file_sha256": scale.file_sha256(
                artifact / "execution-package-amendment-004.json"
            ),
            "execution_amendment_005_file_sha256": scale.file_sha256(
                artifact / "execution-package-amendment-005.json"
            ),
            "result_preserving": True,
            "runtime_restoration_authorization_file_sha256": scale.file_sha256(
                artifact / "runtime-restoration-authorization.json"
            ),
            "runtime_restoration_receipt_file_sha256": scale.file_sha256(
                artifact / "evidence" / "runtime-restoration-receipt.json"
            ),
            "runtime_restoration_verification_file_sha256": scale.file_sha256(
                artifact / "evidence" / "runtime-restoration-verification.json"
            ),
        },
        "comparative_audit": {
            "acceptance_advantage": advantage,
            "baseline": baseline_audit,
            "candidate": candidate_audit,
            "machine_checks": machine_checks,
        },
        "generated_output_tokens": _generated_output_tokens(artifact),
        "inference_attempts": _all_journal_attempts(artifact),
        "model_import_receipt_file_sha256": scale.file_sha256(
            artifact / "evidence" / "model-import-receipt.json"
        ),
        "network_and_compute": {
            "cloud_or_paid_gpu_used": False,
            "external_api_calls": 0,
            "model_network_payload_bytes": preflight.REMOTE_OBJECT_BYTES,
            "package_installations": 0,
            "runtime_restoration_network_bytes": OLLAMA_ARCHIVE_BYTES,
            "sequential_single_model": True,
            "total_network_payload_bytes": (
                preflight.REMOTE_OBJECT_BYTES + OLLAMA_ARCHIVE_BYTES
            ),
        },
        "owner_review": review,
        "plan_file_sha256": scale.file_sha256(plan_file),
        "production": {
            "candidate_registry_changed": False,
            "dataset_sealed": False,
            "production_eligible": False,
            "sft_training_started": False,
        },
        "provenance": {
            "ai_assisted": True,
            "human_authorship_claimed": False,
            "self_hosted": True,
        },
        "runtime": {
            "elapsed_wall_seconds": int(time.time()) - started_unix,
            "finished_unix": int(time.time()),
            "ollama_binary_file_sha256": OLLAMA_BINARY_SHA256,
            "ollama_binary_path": str(OLLAMA_BINARY),
            "ollama_version": "0.32.5",
            "private_write_bytes": _private_write_bytes(artifact, thermal_cache),
            "runtime_restoration_private_write_bytes": (
                OLLAMA_RESTORATION_PRIVATE_BYTES
            ),
            "safety_snapshots_internal_sha256": sha256_value(safety_snapshots),
            "started_unix": started_unix,
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "selected_record_ids_sha256": preflight.SELECTED_IDS_SHA256,
        "selected_schedule_internal_sha256": preflight.SELECTED_SCHEDULE_SHA256,
        "status": (
            "machine_verified_candidate_pending_owner_review"
            if all(machine_checks.values())
            else "failed_closed_candidate_machine_gate_not_met"
        ),
        "tooling_file_sha256": _tooling_hashes(root),
    }
    report["report_sha256"] = sha256_value(report)
    _write_json(artifact / "evidence" / "run-report.json", report)
    return report


def verify_paired_ab(
    *,
    plan_path: str | Path,
    continuation_authorization_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_file = Path(plan_path).resolve()
    plan = load_plan(plan_file, repository_root=root)
    artifact = artifact_root(plan, root)
    continuation_file = Path(continuation_authorization_path).resolve()
    load_continuation_authorization(
        continuation_file, artifact=artifact, repository_root=root
    )
    _ollama_version()
    report = _load_json(artifact / "evidence" / "run-report.json")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("paired A/B report schema changed")
    unsigned = dict(report)
    observed_internal_sha = unsigned.pop("report_sha256", None)
    if observed_internal_sha != sha256_value(unsigned):
        raise ValueError("paired A/B report internal SHA-256 changed")
    if report.get("tooling_file_sha256") != _tooling_hashes(root):
        raise ValueError("paired A/B tooling changed")
    expected_continuation = {
        "authorization_file_sha256": scale.file_sha256(continuation_file),
        "boundary_stop_report_file_sha256": scale.file_sha256(
            artifact / "evidence" / "boundary-stop-report.json"
        ),
        "boundary_stop_verification_file_sha256": scale.file_sha256(
            artifact / "evidence" / "boundary-stop-verification.json"
        ),
        "cumulative_maximum_inference_attempts": CONTINUATION_MAXIMUM_ATTEMPTS,
        "cumulative_maximum_runtime_retries": CONTINUATION_MAXIMUM_RETRIES,
        "execution_amendment_004_file_sha256": scale.file_sha256(
            artifact / "execution-package-amendment-004.json"
        ),
        "execution_amendment_005_file_sha256": scale.file_sha256(
            artifact / "execution-package-amendment-005.json"
        ),
        "result_preserving": True,
        "runtime_restoration_authorization_file_sha256": scale.file_sha256(
            artifact / "runtime-restoration-authorization.json"
        ),
        "runtime_restoration_receipt_file_sha256": scale.file_sha256(
            artifact / "evidence" / "runtime-restoration-receipt.json"
        ),
        "runtime_restoration_verification_file_sha256": scale.file_sha256(
            artifact / "evidence" / "runtime-restoration-verification.json"
        ),
    }
    if report.get("continuation") != expected_continuation:
        raise ValueError("paired A/B continuation report binding changed")
    verify_candidate_objects(plan, repository_root=root)
    schedule = preflight.selected_schedule()
    quality, benchmark, tokenizer, template, _ = _load_runtime_resources(
        plan, repository_root=root
    )
    for model_key in MODEL_KEYS:
        _replay_model_rows(
            artifact=artifact,
            schedule=schedule,
            model_key=model_key,
            quality=quality,
            benchmark=benchmark,
            tokenizer=tokenizer,
            template=template,
        )
    baseline_audit = _model_audit(artifact, schedule, model_key="baseline")
    candidate_audit = _model_audit(artifact, schedule, model_key="candidate")
    if baseline_audit != report["comparative_audit"]["baseline"]:
        raise ValueError("baseline paired A/B audit changed")
    if candidate_audit != report["comparative_audit"]["candidate"]:
        raise ValueError("candidate paired A/B audit changed")
    rows, blind_key = _paired_review_rows(artifact, schedule)
    sample_sha = sha256_value(rows)
    if sample_sha != report["owner_review"]["sample_internal_sha256"]:
        raise ValueError("paired owner-review sample changed")
    if blind_key != _load_json(artifact / "private" / "owner-review-blind-key.json"):
        raise ValueError("paired owner-review blind key changed")
    paths = {
        "blind_key_file_sha256": artifact / "private" / "owner-review-blind-key.json",
        "html_file_sha256": artifact / "review" / "owner-review-pairs-40.html",
        "sample_file_sha256": artifact / "private" / "owner-review-pairs-40.jsonl",
    }
    for field, path in paths.items():
        if scale.file_sha256(path) != report["owner_review"][field]:
            raise ValueError(f"paired owner-review artifact changed: {field}")
    checks = {
        "all_200_final_outputs_accounted": (
            baseline_audit["final_outputs"] + candidate_audit["final_outputs"]
            == EXPECTED_FINAL_OUTPUTS
        ),
        "candidate_and_baseline_exact_models_bound": True,
        "candidate_objects_and_ollama_manifest_verified": True,
        "comparative_machine_audits_reproduced": True,
        "inference_and_generated_token_bounds_preserved": (
            report["inference_attempts"] <= CONTINUATION_MAXIMUM_ATTEMPTS
            and report["generated_output_tokens"] <= MAXIMUM_GENERATED_OUTPUT_TOKENS
        ),
        "exact_private_ollama_runtime_bound": report["runtime"].get(
            "ollama_binary_path"
        )
        == str(OLLAMA_BINARY)
        and report["runtime"].get("ollama_binary_file_sha256")
        == OLLAMA_BINARY_SHA256
        and report["runtime"].get("ollama_version") == "0.32.5",
        "network_accounting_exact": report["network_and_compute"]
        == {
            "cloud_or_paid_gpu_used": False,
            "external_api_calls": 0,
            "model_network_payload_bytes": preflight.REMOTE_OBJECT_BYTES,
            "package_installations": 0,
            "runtime_restoration_network_bytes": OLLAMA_ARCHIVE_BYTES,
            "sequential_single_model": True,
            "total_network_payload_bytes": (
                preflight.REMOTE_OBJECT_BYTES + OLLAMA_ARCHIVE_BYTES
            ),
        },
        "no_api_gpu_package_install_registry_seal_or_sft": not any(
            report["production"].values()
        )
        and report["network_and_compute"]["external_api_calls"] == 0
        and report["network_and_compute"]["cloud_or_paid_gpu_used"] is False
        and report["network_and_compute"]["package_installations"] == 0,
        "paired_owner_review_packet_reproduced": True,
        "private_write_boundary_preserved": report["runtime"]["private_write_bytes"]
        <= MAXIMUM_PRIVATE_WRITE_BYTES,
        "runtime_restoration_private_write_boundary_preserved": report["runtime"].get(
            "runtime_restoration_private_write_bytes"
        )
        == OLLAMA_RESTORATION_PRIVATE_BYTES
        and OLLAMA_RESTORATION_PRIVATE_BYTES <= 1024**3,
        "schedule_and_split_identity_reproduced": True,
    }
    verification: dict[str, Any] = {
        "checks": checks,
        "machine_gate_passed": all(
            report["comparative_audit"]["machine_checks"].values()
        ),
        "next_gate": (
            "owner_review_required_before_any_scale_generation"
            if all(report["comparative_audit"]["machine_checks"].values())
            else "candidate_ineligible_new_teacher_strategy_required"
        ),
        "owner_review_status": report["owner_review"]["status"],
        "production_eligible": False,
        "report_file_sha256": scale.file_sha256(
            artifact / "evidence" / "run-report.json"
        ),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_private_paired_ab_pending_owner_decision",
    }
    verification["verification_sha256"] = sha256_value(verification)
    _write_json(artifact / "evidence" / "verification-report.json", verification)
    return verification
