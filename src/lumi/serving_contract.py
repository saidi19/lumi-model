"""Fail-closed serving readiness contract for the final Lumi-4B assistant."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SERVING_CONTRACT_SCHEMA_VERSION = "lumi.serving.v0.1"
FINAL_PRETRAINING_STEP = 76_294
FINAL_PRETRAINING_TOKENS = 80_000_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def validate_serving_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate contract structure without pretending blocked gates have passed."""

    if value.get("schema_version") != SERVING_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported serving contract schema")
    if value.get("model_id") != "lumi-4b-instruct":
        raise ValueError("serving contract must target lumi-4b-instruct")

    base = value.get("base_checkpoint")
    if not isinstance(base, Mapping):
        raise ValueError("base_checkpoint must be an object")
    if base.get("required_step") != FINAL_PRETRAINING_STEP:
        raise ValueError("base checkpoint step boundary changed")
    if base.get("required_training_tokens") != FINAL_PRETRAINING_TOKENS:
        raise ValueError("base checkpoint token boundary changed")

    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifacts must be an object")
    required_artifacts = {
        "base_checkpoint_manifest_sha256",
        "benchmark_report_sha256",
        "safety_report_sha256",
        "sft_checkpoint_manifest_sha256",
        "sft_dataset_manifest_sha256",
        "tokenizer_manifest_sha256",
    }
    if set(artifacts) != required_artifacts:
        raise ValueError("serving artifact set changed")
    for name, digest in artifacts.items():
        if digest is not None and not _valid_sha256(digest):
            raise ValueError(f"{name} must be null or a SHA-256")

    gates = value.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        raise ValueError("gates must be a non-empty object")
    for name, gate in gates.items():
        if not isinstance(name, str) or not isinstance(gate, Mapping):
            raise ValueError("invalid gate")
        if gate.get("status") not in {"blocked", "failed", "passed", "pending"}:
            raise ValueError(f"{name}: invalid gate status")
        if not isinstance(gate.get("evidence"), list):
            raise ValueError(f"{name}: evidence must be an array")

    api = value.get("api")
    if not isinstance(api, Mapping):
        raise ValueError("api must be an object")
    if api.get("chat_completions_path") != "/v1/chat/completions":
        raise ValueError("chat completions path changed")
    if api.get("models_path") != "/v1/models":
        raise ValueError("models path changed")
    if api.get("health_path") != "/health":
        raise ValueError("health path changed")
    if api.get("readiness_path") != "/ready":
        raise ValueError("readiness path changed")
    if api.get("streaming_protocol") != "server_sent_events":
        raise ValueError("streaming protocol changed")

    deployment = value.get("deployment")
    if not isinstance(deployment, Mapping):
        raise ValueError("deployment must be an object")
    if not isinstance(deployment.get("public_enabled"), bool):
        raise ValueError("public_enabled must be boolean")
    if deployment.get("require_authentication") is not True:
        raise ValueError("production serving must require authentication")
    return json.loads(json.dumps(value))


def serving_readiness(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic blockers and a contract fingerprint."""

    contract = validate_serving_contract(value)
    blockers = [
        f"gate:{name}"
        for name, gate in sorted(contract["gates"].items())
        if gate["status"] != "passed"
    ]
    blockers.extend(
        f"gate-evidence:{name}"
        for name, gate in sorted(contract["gates"].items())
        if gate["status"] == "passed" and not gate["evidence"]
    )
    blockers.extend(
        f"artifact:{name}"
        for name, digest in sorted(contract["artifacts"].items())
        if digest is None
    )
    if contract["deployment"]["public_enabled"] and blockers:
        blockers.append("deployment:premature_public_enable")
    report = {
        "blockers": blockers,
        "contract_sha256": hashlib.sha256(_canonical_json(contract)).hexdigest(),
        "model_id": contract["model_id"],
        "ready": not blockers,
        "schema_version": SERVING_CONTRACT_SCHEMA_VERSION,
    }
    report["report_sha256"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report


def load_serving_contract(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("serving contract must be a JSON object")
    return validate_serving_contract(value)
