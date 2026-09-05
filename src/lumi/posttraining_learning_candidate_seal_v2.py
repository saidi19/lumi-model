"""One-pass token-ledger recovery for the registry-v0.17 learning seal."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_learning_candidate_seal_v1 as v1
from .posttraining_candidate_mix_preflight import (
    _effective_split,
    file_sha256,
    sha256_value,
)
from .posttraining_candidate_mix_reaudit_v12 import (
    _effective_plan,
    load_candidate_mix_reaudit_v12_plan,
    validate_candidate_mix_reaudit_v12_report,
)

PLAN_SCHEMA_VERSION = "lumi.sft-learning-candidate-seal-runtime-recovery-plan.v0.2"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-learning-candidate-seal-runtime-recovery-verification.v0.2"
)
EXPECTED_FAILURE_SHA256 = (
    "b59d06d81c5d8d91b4779beb31d219fa9a7de17799fdccfb707621567d326a6f"
)
EXPECTED_FAILURE_VERIFICATION_SHA256 = (
    "fd576f03612dfbd311019d1ab18700a8e6ebd3f0eff815a4df5eda8bcc1fdeba"
)
EXPECTED_LIMITS = {
    "maximum_additional_full_passes": 1,
    "maximum_additional_gpu_use": 0,
    "maximum_additional_network_bytes": 0,
    "maximum_additional_private_write_bytes": 1_073_741_824,
    "maximum_additional_row_reads": 264_981,
    "maximum_additional_runtime_minutes": 60,
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_learning_candidate_seal_recovery_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(Path(path))
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported learning-seal recovery plan")
    if plan.get("status") != "authorized_one_pass_token_ledger_recovery":
        raise ValueError("learning-seal recovery plan is not authorized")
    if plan.get("authorization") != v1.EXPECTED_AUTHORIZATION:
        raise ValueError("learning-seal recovery authorization changed")
    if plan.get("input_inventory") != v1.EXPECTED_INPUT:
        raise ValueError("learning-seal recovery inventory changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("learning-seal recovery limits changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("learning-seal recovery bindings missing")
    resolved = {
        name: v1._resolve_binding(root, binding, name=name)
        for name, binding in bindings.items()
    }
    required = {
        "candidate_mix_plan",
        "candidate_mix_report",
        "candidate_mix_verification",
        "chat_template",
        "closure_candidate",
        "data_contract",
        "failure_report",
        "failure_verification",
        "final_base_binding",
        "implementation",
        "predecessor_execution_plan",
        "runner",
        "source_registry",
        "test",
        "tokenizer",
        "verifier",
    }
    if set(bindings) != required:
        raise ValueError("learning-seal recovery binding set changed")
    failure = _load_json(resolved["failure_report"])
    failure_verification = _load_json(resolved["failure_verification"])
    if (
        bindings["failure_report"]["sha256"] != EXPECTED_FAILURE_SHA256
        or bindings["failure_verification"]["sha256"]
        != EXPECTED_FAILURE_VERIFICATION_SHA256
        or failure.get("status")
        != "FAIL_CLOSED_token_ledger_consistency_no_final_outputs"
        or failure.get("diagnosis", {}).get(
            "full_candidate_mix_gate_replay_completed_before_failure"
        )
        is not True
        or failure.get("authorization_consumption", {}).get("candidate_row_reads")
        != 529_962
        or failure_verification.get("status")
        != "PASS_verified_fail_closed_no_final_outputs"
        or not all(failure_verification.get("checks", {}).values())
    ):
        raise ValueError("learning-seal predecessor failure evidence changed")
    mix = _load_json(resolved["candidate_mix_report"])
    mix_verification = _load_json(resolved["candidate_mix_verification"])
    validate_candidate_mix_reaudit_v12_report(mix)
    if (
        bindings["candidate_mix_report"]["sha256"]
        != v1.EXPECTED_MIX_REPORT_SHA256
        or bindings["candidate_mix_verification"]["sha256"]
        != v1.EXPECTED_MIX_VERIFICATION_SHA256
        or not all(mix_verification.get("checks", {}).values())
    ):
        raise ValueError("learning-seal recovery mix evidence changed")
    return plan


def _materialize_sources_with_token_ledger(
    *,
    source_inputs: Sequence[tuple[str, Path]],
    expected_source_metrics: Mapping[str, Mapping[str, Mapping[str, int]]],
    expected_input: Mapping[str, Any],
    train_path: Path,
    validation_path: Path,
) -> dict[str, Any]:
    if list(source_inputs) != sorted(source_inputs, key=lambda item: item[0]):
        raise ValueError("learning-seal recovery sources are not in lexical order")
    if set(expected_source_metrics) != {source_id for source_id, _ in source_inputs}:
        raise ValueError("learning-seal recovery source ledger changed")
    for output in (train_path, validation_path):
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite learning-seal split: {output}")
        temporary = output.with_name(f".{output.name}.tmp")
        if temporary.exists():
            raise FileExistsError(f"stale learning-seal temporary exists: {temporary}")

    temporary_train = train_path.with_name(f".{train_path.name}.tmp")
    temporary_validation = validation_path.with_name(f".{validation_path.name}.tmp")
    split_counts = Counter()
    split_hashers = {"train": hashlib.sha256(), "validation": hashlib.sha256()}
    observed_source_records: dict[str, Counter[str]] = {}
    record_id_hasher = hashlib.sha256()
    try:
        with temporary_train.open("xb") as train_handle, temporary_validation.open(
            "xb"
        ) as validation_handle:
            handles = {"train": train_handle, "validation": validation_handle}
            for source_id, source_path in source_inputs:
                observed_source_records[source_id] = Counter()
                with source_path.open(encoding="utf-8") as source_handle:
                    for line_number, line in enumerate(source_handle, start=1):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError(
                                f"JSON object required at {source_path}:{line_number}"
                            )
                        if row.get("source", {}).get("source_id") != source_id:
                            raise ValueError("learning-seal recovery source changed")
                        split, normalized = _effective_split(row, "top_level")
                        if normalized:
                            raise ValueError("learning-seal recovery split normalization")
                        payload = v1._canonical_row_bytes(row) + b"\n"
                        handles[split].write(payload)
                        split_hashers[split].update(payload)
                        split_counts[split] += 1
                        observed_source_records[source_id][split] += 1
                        record_id = row.get("record_id")
                        if not isinstance(record_id, str):
                            raise ValueError("learning-seal recovery record ID changed")
                        record_id_hasher.update(
                            f"{source_id}\0{record_id}\n".encode("utf-8")
                        )
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())

        frozen_source_metrics: dict[str, dict[str, dict[str, int]]] = {}
        split_tokens = Counter()
        for source_id, by_split in expected_source_metrics.items():
            frozen_source_metrics[source_id] = {}
            for split in ("train", "validation"):
                expected = by_split[split]
                if observed_source_records[source_id][split] != expected["records"]:
                    raise ValueError(
                        f"learning-seal recovery row ledger changed: {source_id}:{split}"
                    )
                frozen_source_metrics[source_id][split] = {
                    "records": expected["records"],
                    "supervised_tokens": expected["supervised_tokens"],
                }
                split_tokens[split] += expected["supervised_tokens"]
        if (
            split_counts["train"] != expected_input["train_records"]
            or split_counts["validation"] != expected_input["validation_records"]
            or split_tokens["train"] != expected_input["train_supervised_tokens"]
            or split_tokens["validation"]
            != expected_input["validation_supervised_tokens"]
        ):
            raise ValueError("learning-seal recovery frozen ledger totals changed")
        temporary_train.replace(train_path)
        temporary_validation.replace(validation_path)
    except BaseException:
        temporary_train.unlink(missing_ok=True)
        temporary_validation.unlink(missing_ok=True)
        raise
    return {
        "record_id_sequence_sha256": record_id_hasher.hexdigest(),
        "source_metrics": frozen_source_metrics,
        "splits": {
            split: {
                "bytes": (train_path if split == "train" else validation_path).stat().st_size,
                "records": split_counts[split],
                "sha256": split_hashers[split].hexdigest(),
                "supervised_tokens": split_tokens[split],
            }
            for split in ("train", "validation")
        },
    }


def build_learning_candidate_seal_recovery(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved = {
        name: v1._resolve_binding(root, binding, name=name)
        for name, binding in plan["bindings"].items()
    }
    mix_plan = load_candidate_mix_reaudit_v12_plan(
        resolved["candidate_mix_plan"], repository_root=root
    )
    existing_report = _load_json(resolved["candidate_mix_report"])
    validate_candidate_mix_reaudit_v12_report(existing_report)
    effective = _effective_plan(mix_plan, repository_root=root)
    source_inputs: list[tuple[str, Path]] = []
    candidate_input_bytes = 0
    for source in sorted(effective["candidate_sources"], key=lambda item: item["source_id"]):
        binding = effective["bindings"][source["candidate_binding"]]
        candidate_path = (root / binding["path"]).resolve()
        if file_sha256(candidate_path) != source["candidate_file_sha256"]:
            raise ValueError(f"learning-seal recovery candidate changed: {source['source_id']}")
        source_inputs.append((source["source_id"], candidate_path))
        candidate_input_bytes += candidate_path.stat().st_size
    if (
        len(source_inputs) != v1.EXPECTED_INPUT["sources"]
        or candidate_input_bytes != v1.EXPECTED_INPUT["candidate_input_bytes"]
        or resolved["closure_candidate"] in {candidate for _, candidate in source_inputs}
    ):
        raise ValueError("learning-seal recovery source inventory changed")

    train_path = (root / plan["output"]["train_path"]).resolve()
    validation_path = (root / plan["output"]["validation_path"]).resolve()
    private_root = (
        root / "artifacts/lumi-sft-learning-candidate-seal-runtime-recovery-v0.2"
    ).resolve()
    if not train_path.is_relative_to(private_root) or not validation_path.is_relative_to(
        private_root
    ):
        raise ValueError("learning-seal recovery output escaped private root")
    expected_source_metrics = {
        source_id: metrics["by_split"]
        for source_id, metrics in existing_report["source_metrics"].items()
    }
    materialized = _materialize_sources_with_token_ledger(
        source_inputs=source_inputs,
        expected_source_metrics=expected_source_metrics,
        expected_input=v1.EXPECTED_INPUT,
        train_path=train_path,
        validation_path=validation_path,
    )
    write_bytes = sum(value["bytes"] for value in materialized["splits"].values())
    if write_bytes > plan["limits"]["maximum_additional_private_write_bytes"]:
        raise ValueError("learning-seal recovery write ceiling exceeded")

    manifest: dict[str, Any] = {
        "authorization_boundary": {
            "active_data_contract_changed": False,
            "closure_supplement_included": False,
            "network_or_gpu_used": False,
            "production_dataset_seal": False,
            "production_sft_started": False,
            "registry_changed": False,
            "smoke_only": True,
        },
        "bindings": {
            "candidate_mix_report_sha256": v1.EXPECTED_MIX_REPORT_SHA256,
            "candidate_mix_verification_sha256": v1.EXPECTED_MIX_VERIFICATION_SHA256,
            "failure_report_sha256": EXPECTED_FAILURE_SHA256,
            "source_registry_sha256": v1.EXPECTED_REGISTRY_SHA256,
            "tokenizer_sha256": plan["bindings"]["tokenizer"]["sha256"],
        },
        "closure_exclusion": {
            "candidate_sha256": v1.EXPECTED_CLOSURE_CANDIDATE_SHA256,
            "excluded_rows": 1463,
            "included": False,
        },
        "gate_replay": {
            "bound_predecessor_gate_replay_passed_before_failure": True,
            "candidate_mix_internal_report_sha256": existing_report["report_sha256"],
            "exact_duplicate_groups": existing_report["exact_duplicate_audit"][
                "duplicate_groups"
            ],
            "generic_sft_rejected_records": existing_report[
                "generic_sft_validation"
            ]["rejected"],
            "near_duplicate_multi_record_components": existing_report[
                "near_duplicate_audit"
            ]["multi_record_components"],
            "train_validation_overlap_groups": existing_report[
                "exact_duplicate_audit"
            ]["train_validation_overlap_groups"],
        },
        "input": {
            "candidate_input_bytes": candidate_input_bytes,
            "records": v1.EXPECTED_INPUT["records"],
            "sources": len(source_inputs),
        },
        "ordering_contract": "source_id_lexical_then_existing_file_order_v0.1",
        "output": materialized,
        "plan_file_sha256": plan_file_sha256,
        "production_blockers_preserved": existing_report["closure"]["blockers"],
        "raw_training_text_retained_in_manifest": False,
        "recovery_contract": "bind_frozen_v0_12_per_source_token_ledger_no_row_retokenization_v0.2",
        "schema_version": v1.MANIFEST_SCHEMA_VERSION,
        "status": "PASS_nonproduction_learning_candidate_sealed_smoke_only",
    }
    manifest["seal_sha256"] = sha256_value(manifest)
    v1.validate_learning_candidate_manifest(manifest)
    return manifest


def build_learning_candidate_recovery_verification(
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    verification = v1.build_learning_candidate_verification(
        plan=plan,
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
        repository_root=repository_root,
    )
    verification.pop("verification_sha256", None)
    verification["checks"]["frozen_v0_12_per_source_token_ledger_bound"] = True
    verification["checks"]["one_additional_materialization_pass_only"] = True
    verification["predecessor_failure_sha256"] = EXPECTED_FAILURE_SHA256
    verification["schema_version"] = VERIFICATION_SCHEMA_VERSION
    verification["status"] = (
        "PASS_verified_nonproduction_learning_candidate_seal_runtime_recovery"
    )
    verification["verification_sha256"] = sha256_value(verification)
    return verification


json_file_bytes = v1.json_file_bytes
_write_atomic = v1._write_atomic

__all__ = [
    "build_learning_candidate_recovery_verification",
    "build_learning_candidate_seal_recovery",
    "json_file_bytes",
    "load_learning_candidate_seal_recovery_plan",
]
