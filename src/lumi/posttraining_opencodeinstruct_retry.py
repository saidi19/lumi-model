"""Deterministic no-network retry support for OpenCodeInstruct curation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_opencodeinstruct_curation import (
    EXPECTED_POLICY,
    OWNER_AUTHORIZATION_STATEMENT,
    SHARD_BYTES,
    SHARD_PATH,
    SHARD_SHA256,
    SOURCE_DATASET_ID,
    SOURCE_ID,
    SOURCE_LICENSE,
    SOURCE_REVISION,
    SOURCE_ROWS,
    file_sha256,
)

RETRY_PLAN_SCHEMA_VERSION = "lumi.sft-opencodeinstruct-deterministic-retry.v0.1"
ORIGINAL_PLAN_FILE_SHA256 = (
    "ba977d954a1c630ea2befed5f42db6c144db2aed51b389f61d7adc460ed3d866"
)
ORIGINAL_REPORT_INTERNAL_SHA256 = (
    "2a5bf1c6f9eb724e317d6dbd93a7e0b139f998e8e238e54beb63992a2b9f75a0"
)
ORIGINAL_VERIFIER_SHA256 = (
    "a3800f58f3c3055c58946a036848ad79c16c18cba91a6cfd8d3ca363bc85b070"
)
EXPECTED_BINDINGS = {
    "base_curation_implementation",
    "benchmark_registry",
    "bulk_source_shortlist",
    "chat_template",
    "data_contract",
    "data_quality_implementation",
    "filtering_implementation",
    "original_runner",
    "original_verifier",
    "owner_review_implementation",
    "predecessor_plan",
    "predecessor_report",
    "quality_config",
    "retry_implementation",
    "retry_runner",
    "retry_verifier",
    "tokenizer_config",
    "tokenizer_json",
    "unit_test_worker",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_retry_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(Path(path))
    if plan.get("schema_version") != RETRY_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported OpenCodeInstruct retry plan")
    if plan.get("status") != "authorized_result_preserving_deterministic_retry":
        raise ValueError("OpenCodeInstruct retry status changed")
    if plan.get("observed_at") != "2026-08-07":
        raise ValueError("OpenCodeInstruct retry observation date changed")
    if plan.get("owner_authorization_statement") != OWNER_AUTHORIZATION_STATEMENT:
        raise ValueError("OpenCodeInstruct retry owner authorization changed")
    if plan.get("retry_reason") != (
        "attempt_001_independent_replay_found_python_hash_seed_nondeterminism"
    ):
        raise ValueError("OpenCodeInstruct retry reason changed")
    authorization = plan.get("authorization")
    if authorization != {
        "candidate_registry_promotion_authorized": False,
        "cpu_only_curation_authorized": True,
        "dataset_seal_authorized": False,
        "external_api_authorized": False,
        "external_distribution_authorized": False,
        "network_access_authorized": False,
        "paid_gpu_authorized": False,
        "production_sft_authorized": False,
        "unrelated_dataset_access_authorized": False,
    }:
        raise ValueError("OpenCodeInstruct retry authorization changed")
    limits = plan.get("limits")
    if limits != {
        "maximum_decoded_bytes": 536_870_912,
        "maximum_network_bytes": 0,
        "maximum_retained_train_supervised_tokens": 2_300_000,
        "maximum_runtime_minutes": 180,
        "maximum_source_objects": 2,
        "maximum_source_rows": SOURCE_ROWS,
        "owner_sample_records": 32,
        "target_train_supervised_tokens": 2_100_000,
        "validation_fraction": 0.05,
    }:
        raise ValueError("OpenCodeInstruct retry limits changed")
    source = plan.get("source")
    if not isinstance(source, Mapping) or source != {
        "dataset_id": SOURCE_DATASET_ID,
        "declared_license_id": SOURCE_LICENSE,
        "objects": plan["source"]["objects"],
        "repository_url": "https://huggingface.co/datasets/nvidia/OpenCodeInstruct",
        "revision": SOURCE_REVISION,
        "source_id": SOURCE_ID,
        "source_rows": SOURCE_ROWS,
        "source_split": "train",
    }:
        raise ValueError("OpenCodeInstruct retry source changed")
    if source["objects"] != [
        {
            "git_blob_sha1": "8226aea38e6986a3a243ea161739ed3ad45ac600",
            "path": "README.md",
            "size_bytes": 4_377,
        },
        {"path": SHARD_PATH, "sha256": SHARD_SHA256, "size_bytes": SHARD_BYTES},
    ]:
        raise ValueError("OpenCodeInstruct retry object set changed")
    output = plan.get("output")
    if output != {
        "private_root": (
            "../lumi-artifacts/4b-sft-opencodeinstruct-single-shard-v0.1/"
            "8f3ba5bafe4d/attempt-002"
        ),
        "raw_text_free_report_path": (
            "evidence/lumi-4b-sft-opencodeinstruct-single-shard-retry-v0.1.json"
        ),
        "raw_text_free_verification_path": (
            "evidence/lumi-4b-sft-opencodeinstruct-single-shard-retry-verification-v0.1.json"
        ),
        "raw_training_text_in_git": False,
    }:
        raise ValueError("OpenCodeInstruct retry output changed")
    expected_policy = {
        **EXPECTED_POLICY,
        "unit_test_replay": {
            "interpreter_flags": ["-S", "-B"],
            "python_hash_seed": "0",
            "repeat_in_independent_verifier": True,
        },
    }
    if plan.get("policy") != expected_policy:
        raise ValueError("OpenCodeInstruct retry policy changed")
    if plan["policy"]["unit_test_replay"] != {
        "interpreter_flags": ["-S", "-B"],
        "python_hash_seed": "0",
        "repeat_in_independent_verifier": True,
    }:
        raise ValueError("OpenCodeInstruct retry determinism policy changed")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("OpenCodeInstruct retry bindings missing")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid OpenCodeInstruct retry binding: {name}")
        relative = binding.get("path")
        expected = binding.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"invalid OpenCodeInstruct retry binding fields: {name}")
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root.parent.resolve()) or not resolved.is_file():
            raise ValueError(f"OpenCodeInstruct retry binding missing: {name}")
        if file_sha256(resolved) != expected:
            raise ValueError(f"OpenCodeInstruct retry binding changed: {name}")
    predecessor_plan = bindings.get("predecessor_plan")
    predecessor_report = bindings.get("predecessor_report")
    original_verifier = bindings.get("original_verifier")
    if not all(
        isinstance(item, Mapping)
        for item in (predecessor_plan, predecessor_report, original_verifier)
    ):
        raise ValueError("OpenCodeInstruct retry predecessor bindings missing")
    if predecessor_plan.get("sha256") != ORIGINAL_PLAN_FILE_SHA256:
        raise ValueError("OpenCodeInstruct predecessor plan binding changed")
    if original_verifier.get("sha256") != ORIGINAL_VERIFIER_SHA256:
        raise ValueError("OpenCodeInstruct original verifier binding changed")
    report = _load_json((root / predecessor_report["path"]).resolve())
    if report.get("report_sha256") != ORIGINAL_REPORT_INTERNAL_SHA256:
        raise ValueError("OpenCodeInstruct predecessor report seal changed")
    if report.get("production", {}).get("production_eligible") is not False:
        raise ValueError("OpenCodeInstruct predecessor unexpectedly became production eligible")
    return plan


def run_deterministic_unit_test_verifier(
    envelopes: Sequence[Mapping[str, Any]],
    *,
    worker_path: str | Path,
    python_executable: str = sys.executable,
    timeout_seconds: int = 1_200,
) -> tuple[list[dict[str, Any]], set[str], dict[str, int]]:
    payload = []
    expected: dict[str, tuple[str, str]] = {}
    for envelope in envelopes:
        attribution = envelope["attribution"]
        record_id = str(envelope["record"]["record_id"])
        expected[record_id] = (
            str(attribution["code_sha256"]),
            str(attribution["tests_sha256"]),
        )
        payload.append(
            json.dumps(
                {
                    "code": envelope["code"],
                    "record_id": record_id,
                    "unit_tests": envelope["unit_tests"],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
    }
    completed = subprocess.run(
        [python_executable, "-S", "-B", str(Path(worker_path).resolve())],
        check=False,
        capture_output=True,
        cwd="/",
        env=environment,
        input=("\n".join(payload) + "\n").encode("utf-8"),
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "deterministic OpenCodeInstruct worker failed raw-text-free: "
            f"returncode={completed.returncode} stderr_sha256="
            f"{hashlib.sha256(completed.stderr).hexdigest()}"
        )
    results = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    if len(results) != len(envelopes):
        raise ValueError("deterministic OpenCodeInstruct result count changed")
    ids = [result.get("record_id") for result in results]
    if set(ids) != set(expected) or len(ids) != len(set(ids)):
        raise ValueError("deterministic OpenCodeInstruct verifier IDs changed")
    passed: set[str] = set()
    reasons: Counter[str] = Counter()
    ledger: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: str(item["record_id"])):
        record_id = str(result["record_id"])
        code_sha256, tests_sha256 = expected[record_id]
        if result.get("code_sha256") != code_sha256 or result.get("tests_sha256") != tests_sha256:
            raise ValueError("deterministic OpenCodeInstruct content identity changed")
        verified = result.get("verified") is True
        reason = result.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError("deterministic OpenCodeInstruct reason changed")
        if verified:
            if reason != "verified":
                raise ValueError("deterministic OpenCodeInstruct pass reason changed")
            passed.add(record_id)
        reasons[reason] += 1
        ledger.append(
            {
                "code_sha256": code_sha256,
                "reason": reason,
                "record_id": record_id,
                "tests_sha256": tests_sha256,
                "verified": verified,
            }
        )
    return ledger, passed, dict(sorted(reasons.items()))


__all__ = [
    "ORIGINAL_VERIFIER_SHA256",
    "RETRY_PLAN_SCHEMA_VERSION",
    "load_retry_plan",
    "run_deterministic_unit_test_verifier",
]
