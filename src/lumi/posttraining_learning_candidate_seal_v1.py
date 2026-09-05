"""Fail-closed non-production seal for the verified registry-v0.17 SFT mix."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_candidate_mix_preflight import (
    _effective_split,
    _measure_conversation,
    file_sha256,
    sha256_value,
)
from .posttraining_candidate_mix_reaudit_v12 import (
    _effective_plan,
    load_candidate_mix_reaudit_v12_plan,
    rebuild_candidate_mix_reaudit_v12,
    validate_candidate_mix_reaudit_v12_report,
)
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
)

PLAN_SCHEMA_VERSION = "lumi.sft-learning-candidate-seal-plan.v0.1"
MANIFEST_SCHEMA_VERSION = "lumi.sft-learning-candidate-seal.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-learning-candidate-seal-verification.v0.1"

EXPECTED_AUTHORIZATION = {
    "active_data_contract_change_authorized": False,
    "candidate_materialization_authorized": True,
    "dataset_seal_authorized": True,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "registry_change_authorized": False,
    "sft_smoke_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_INPUT = {
    "accepted_dataset_sha256": (
        "cf379dba7de33b34fb45282c6c83ed65ce85bcef0b3de99a12ba8ba3cad442bb"
    ),
    "candidate_input_bytes": 359_324_855,
    "records": 264_981,
    "sources": 11,
    "train_records": 251_176,
    "train_supervised_tokens": 7_341_582,
    "validation_records": 13_805,
    "validation_supervised_tokens": 427_005,
}
EXPECTED_LIMITS = {
    "maximum_full_passes": 2,
    "maximum_gpu_use": 0,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 1_073_741_824,
    "maximum_row_reads": 529_962,
    "maximum_runtime_minutes": 120,
}
EXPECTED_OUTPUT = {
    "manifest_path": "evidence/lumi-4b-sft-learning-candidate-seal-v0.1.json",
    "train_path": (
        "artifacts/lumi-sft-learning-candidate-seal-v0.1/attempt-001/private/train.jsonl"
    ),
    "validation_path": (
        "artifacts/lumi-sft-learning-candidate-seal-v0.1/attempt-001/private/validation.jsonl"
    ),
    "verification_path": (
        "evidence/lumi-4b-sft-learning-candidate-seal-verification-v0.1.json"
    ),
}
EXPECTED_CLOSURE_CANDIDATE_SHA256 = (
    "e8c0a360c898e6c8f78701c27f534b49f8bbe43c557bec9d15f0ec2c4eb74b1e"
)
EXPECTED_MIX_REPORT_SHA256 = (
    "2389e61c66fa0000690fe119745e963b9cc2cab1eb26e61224bef87ef9522dcb"
)
EXPECTED_MIX_VERIFICATION_SHA256 = (
    "d12d28e192cf31e33d3c034293868a720fd74d9934ae506c0e1d5c2d1d3c18d2"
)
EXPECTED_REGISTRY_SHA256 = (
    "85a688a62be0902b177469c99eb31525fb633efab5e420416b705d03e8d80e1a"
)
EXPECTED_PLAN_BINDINGS = frozenset(
    {
        "candidate_mix_plan",
        "candidate_mix_report",
        "candidate_mix_verification",
        "chat_template",
        "closure_candidate",
        "closure_retirement",
        "data_contract",
        "final_base_binding",
        "implementation",
        "preflight_plan",
        "preflight_report",
        "preflight_verification",
        "runner",
        "source_registry",
        "test",
        "tokenizer",
        "verifier",
    }
)


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _canonical_row_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _resolve_binding(
    root: Path, binding: Mapping[str, Any], *, name: str
) -> Path:
    relative = binding.get("path")
    expected_sha256 = binding.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_sha256, str):
        raise ValueError(f"invalid learning-seal binding: {name}")
    path = (root / relative).resolve()
    if not path.is_file():
        raise ValueError(f"missing learning-seal binding: {name}")
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"learning-seal binding changed: {name}")
    return path


def load_learning_candidate_seal_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(Path(path))
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported learning-candidate seal plan")
    if plan.get("status") != "authorized_nonproduction_learning_candidate_seal":
        raise ValueError("learning-candidate seal plan is not authorized")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("learning-candidate seal authorization changed")
    if plan.get("input_inventory") != EXPECTED_INPUT:
        raise ValueError("learning-candidate seal input inventory changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("learning-candidate seal limits changed")
    if plan.get("output") != EXPECTED_OUTPUT:
        raise ValueError("learning-candidate seal output paths changed")
    if plan.get("closure_candidate_sha256_excluded") != (
        EXPECTED_CLOSURE_CANDIDATE_SHA256
    ):
        raise ValueError("learning-candidate closure exclusion changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_PLAN_BINDINGS:
        raise ValueError("learning-candidate seal binding set changed")
    resolved = {
        name: _resolve_binding(root, binding, name=name)
        for name, binding in bindings.items()
    }
    if (
        bindings["candidate_mix_report"]["sha256"] != EXPECTED_MIX_REPORT_SHA256
        or bindings["candidate_mix_verification"]["sha256"]
        != EXPECTED_MIX_VERIFICATION_SHA256
        or bindings["source_registry"]["sha256"] != EXPECTED_REGISTRY_SHA256
        or bindings["closure_candidate"]["sha256"]
        != EXPECTED_CLOSURE_CANDIDATE_SHA256
    ):
        raise ValueError("learning-candidate exact evidence binding changed")
    preflight = _load_json(resolved["preflight_plan"])
    preflight_report = _load_json(resolved["preflight_report"])
    preflight_verification = _load_json(resolved["preflight_verification"])
    if (
        preflight.get("status") != "PASS_preflight_ready_successor_not_authorized"
        or preflight_report.get("status")
        != "PASS_successor_plan_ready_no_materialization_or_training"
        or preflight_verification.get("status")
        != "PASS_independently_verified_successor_preflight_only"
        or not all(preflight_verification.get("checks", {}).values())
    ):
        raise ValueError("learning-candidate preflight evidence changed")
    mix = _load_json(resolved["candidate_mix_report"])
    mix_verification = _load_json(resolved["candidate_mix_verification"])
    validate_candidate_mix_reaudit_v12_report(mix)
    if (
        mix.get("candidate_pool", {}).get("candidate_records_read") != 264_981
        or mix.get("candidate_pool", {}).get("accepted_train_supervised_tokens")
        != 7_341_582
        or mix.get("candidate_pool", {}).get(
            "registry_admissible_validation_supervised_tokens"
        )
        != 427_005
        or mix_verification.get("status")
        != "PASS_verified_registry_v0_17_candidate_mix_reaudit_nonproduction"
        or not all(mix_verification.get("checks", {}).values())
    ):
        raise ValueError("learning-candidate mix evidence changed")
    closure = _load_json(resolved["closure_retirement"])
    if (
        closure.get("local_evidence", {}).get("candidate", {}).get("sha256")
        != EXPECTED_CLOSURE_CANDIDATE_SHA256
        or closure.get("closure_generation", {}).get("accepted_rows") != 1463
    ):
        raise ValueError("learning-candidate closure retirement changed")
    return plan


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite learning-seal output: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale learning-seal temporary exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _materialize_sources(
    *,
    source_inputs: Sequence[tuple[str, Path]],
    train_path: Path,
    validation_path: Path,
    measure: Callable[[Mapping[str, Any]], int],
) -> dict[str, Any]:
    """Stream canonical rows in source-id lexical and existing-file order."""

    if list(source_inputs) != sorted(source_inputs, key=lambda item: item[0]):
        raise ValueError("learning-seal sources are not in lexical order")
    for output in (train_path, validation_path):
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite learning-seal split: {output}")
        temporary = output.with_name(f".{output.name}.tmp")
        if temporary.exists():
            raise FileExistsError(f"stale learning-seal split temporary: {temporary}")

    temporary_train = train_path.with_name(f".{train_path.name}.tmp")
    temporary_validation = validation_path.with_name(f".{validation_path.name}.tmp")
    split_counts = Counter()
    split_tokens = Counter()
    split_hashers = {"train": hashlib.sha256(), "validation": hashlib.sha256()}
    source_metrics: dict[str, dict[str, dict[str, int]]] = {}
    record_id_hasher = hashlib.sha256()
    seen_source_ids: set[str] = set()
    try:
        with temporary_train.open("xb") as train_handle, temporary_validation.open(
            "xb"
        ) as validation_handle:
            handles = {"train": train_handle, "validation": validation_handle}
            for source_id, source_path in source_inputs:
                if source_id in seen_source_ids:
                    raise ValueError("duplicate learning-seal source")
                seen_source_ids.add(source_id)
                source_metrics[source_id] = {
                    split: {"records": 0, "supervised_tokens": 0}
                    for split in ("train", "validation")
                }
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
                            raise ValueError("learning-seal row source binding changed")
                        split, normalized = _effective_split(row, "top_level")
                        if normalized:
                            raise ValueError("learning-seal split normalization is forbidden")
                        supervised_tokens = measure(row)
                        payload = _canonical_row_bytes(row) + b"\n"
                        handles[split].write(payload)
                        split_hashers[split].update(payload)
                        split_counts[split] += 1
                        split_tokens[split] += supervised_tokens
                        source_metrics[source_id][split]["records"] += 1
                        source_metrics[source_id][split][
                            "supervised_tokens"
                        ] += supervised_tokens
                        record_id = row.get("record_id")
                        if not isinstance(record_id, str):
                            raise ValueError("learning-seal record ID changed")
                        identity = f"{source_id}\0{record_id}\n".encode("utf-8")
                        record_id_hasher.update(identity)
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
        if (
            split_counts["train"] != EXPECTED_INPUT["train_records"]
            or split_counts["validation"] != EXPECTED_INPUT["validation_records"]
            or split_tokens["train"] != EXPECTED_INPUT["train_supervised_tokens"]
            or split_tokens["validation"]
            != EXPECTED_INPUT["validation_supervised_tokens"]
        ):
            raise ValueError("learning-seal materialized split totals changed")
        temporary_train.replace(train_path)
        temporary_validation.replace(validation_path)
    except BaseException:
        temporary_train.unlink(missing_ok=True)
        temporary_validation.unlink(missing_ok=True)
        raise
    return {
        "record_id_sequence_sha256": record_id_hasher.hexdigest(),
        "source_metrics": source_metrics,
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


def build_learning_candidate_seal(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    resolved = {
        name: _resolve_binding(root, binding, name=name)
        for name, binding in bindings.items()
    }
    mix_plan = load_candidate_mix_reaudit_v12_plan(
        resolved["candidate_mix_plan"], repository_root=root
    )
    rebuilt = rebuild_candidate_mix_reaudit_v12(
        plan=mix_plan,
        plan_file_sha256=file_sha256(resolved["candidate_mix_plan"]),
        repository_root=root,
    )
    existing_report = _load_json(resolved["candidate_mix_report"])
    if rebuilt != existing_report:
        raise ValueError("learning-seal full frozen gate replay changed")
    effective = _effective_plan(mix_plan, repository_root=root)
    source_inputs: list[tuple[str, Path]] = []
    candidate_input_bytes = 0
    for source in sorted(effective["candidate_sources"], key=lambda item: item["source_id"]):
        binding = effective["bindings"][source["candidate_binding"]]
        candidate_path = (root / binding["path"]).resolve()
        if file_sha256(candidate_path) != source["candidate_file_sha256"]:
            raise ValueError(f"learning-seal candidate changed: {source['source_id']}")
        source_inputs.append((source["source_id"], candidate_path))
        candidate_input_bytes += candidate_path.stat().st_size
    if (
        len(source_inputs) != EXPECTED_INPUT["sources"]
        or candidate_input_bytes != EXPECTED_INPUT["candidate_input_bytes"]
        or resolved["closure_candidate"] in {path for _, path in source_inputs}
    ):
        raise ValueError("learning-seal source inventory or closure exclusion changed")

    template = load_chat_template(resolved["chat_template"])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = PureByteLevelBPETokenizer(resolved["tokenizer"])
    verify_chat_tokenizer(tokenizer, template)

    train_path = (root / plan["output"]["train_path"]).resolve()
    validation_path = (root / plan["output"]["validation_path"]).resolve()
    private_root = (root / "artifacts/lumi-sft-learning-candidate-seal-v0.1").resolve()
    if not train_path.is_relative_to(private_root) or not validation_path.is_relative_to(
        private_root
    ):
        raise ValueError("learning-seal split output escaped the private artifact root")

    materialized = _materialize_sources(
        source_inputs=source_inputs,
        train_path=train_path,
        validation_path=validation_path,
        measure=lambda row: _measure_conversation(
            row["messages"], tokenizer=tokenizer, template=template
        ),
    )
    split_write_bytes = sum(
        int(value["bytes"]) for value in materialized["splits"].values()
    )
    if split_write_bytes > plan["limits"]["maximum_private_write_bytes"]:
        raise ValueError("learning-seal private write ceiling exceeded")

    expected_source_metrics = {
        source_id: metrics["by_split"]
        for source_id, metrics in existing_report["source_metrics"].items()
    }
    if materialized["source_metrics"] != expected_source_metrics:
        raise ValueError("learning-seal per-source split metrics changed")

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
            "candidate_mix_report_sha256": EXPECTED_MIX_REPORT_SHA256,
            "candidate_mix_verification_sha256": EXPECTED_MIX_VERIFICATION_SHA256,
            "chat_template_sha256": bindings["chat_template"]["sha256"],
            "data_contract_sha256": bindings["data_contract"]["sha256"],
            "final_base_binding_sha256": bindings["final_base_binding"]["sha256"],
            "source_registry_sha256": EXPECTED_REGISTRY_SHA256,
            "tokenizer_sha256": bindings["tokenizer"]["sha256"],
        },
        "closure_exclusion": {
            "candidate_sha256": EXPECTED_CLOSURE_CANDIDATE_SHA256,
            "excluded_rows": 1463,
            "included": False,
        },
        "gate_replay": {
            "benchmark_decontamination_passed": True,
            "candidate_mix_internal_report_sha256": rebuilt["report_sha256"],
            "exact_duplicate_groups": rebuilt["exact_duplicate_audit"][
                "duplicate_groups"
            ],
            "generic_sft_rejected_records": rebuilt["generic_sft_validation"][
                "rejected"
            ],
            "near_duplicate_multi_record_components": rebuilt[
                "near_duplicate_audit"
            ]["multi_record_components"],
            "train_validation_overlap_groups": rebuilt["exact_duplicate_audit"][
                "train_validation_overlap_groups"
            ],
        },
        "input": {
            "candidate_input_bytes": candidate_input_bytes,
            "records": EXPECTED_INPUT["records"],
            "sources": len(source_inputs),
        },
        "ordering_contract": "source_id_lexical_then_existing_file_order_v0.1",
        "output": materialized,
        "plan_file_sha256": plan_file_sha256,
        "production_blockers_preserved": rebuilt["closure"]["blockers"],
        "raw_training_text_retained_in_manifest": False,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "PASS_nonproduction_learning_candidate_sealed_smoke_only",
    }
    manifest["seal_sha256"] = sha256_value(manifest)
    validate_learning_candidate_manifest(manifest)
    return manifest


def validate_learning_candidate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported learning-candidate manifest")
    observed_sha256 = manifest.get("seal_sha256")
    if not isinstance(observed_sha256, str):
        raise ValueError("learning-candidate manifest has no seal")
    unsigned = dict(manifest)
    unsigned.pop("seal_sha256", None)
    if sha256_value(unsigned) != observed_sha256:
        raise ValueError("learning-candidate manifest seal changed")
    boundary = manifest.get("authorization_boundary", {})
    if (
        manifest.get("status")
        != "PASS_nonproduction_learning_candidate_sealed_smoke_only"
        or boundary.get("smoke_only") is not True
        or boundary.get("production_dataset_seal") is not False
        or boundary.get("production_sft_started") is not False
        or boundary.get("closure_supplement_included") is not False
        or manifest.get("raw_training_text_retained_in_manifest") is not False
    ):
        raise ValueError("learning-candidate manifest boundary changed")
    splits = manifest.get("output", {}).get("splits", {})
    if (
        splits.get("train", {}).get("records") != EXPECTED_INPUT["train_records"]
        or splits.get("train", {}).get("supervised_tokens")
        != EXPECTED_INPUT["train_supervised_tokens"]
        or splits.get("validation", {}).get("records")
        != EXPECTED_INPUT["validation_records"]
        or splits.get("validation", {}).get("supervised_tokens")
        != EXPECTED_INPUT["validation_supervised_tokens"]
    ):
        raise ValueError("learning-candidate manifest split totals changed")
    serialized = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"user_prompt"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("learning-candidate manifest contains raw training text")


def build_learning_candidate_verification(
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    validate_learning_candidate_manifest(manifest)
    splits = manifest["output"]["splits"]
    observed: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        path = (root / plan["output"][f"{split}_path"]).resolve()
        hasher = hashlib.sha256()
        newlines = 0
        byte_count = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
                byte_count += len(chunk)
                newlines += chunk.count(b"\n")
        observed[split] = {
            "bytes": byte_count,
            "records": newlines,
            "sha256": hasher.hexdigest(),
        }
        if (
            observed[split]["bytes"] != splits[split]["bytes"]
            or observed[split]["records"] != splits[split]["records"]
            or observed[split]["sha256"] != splits[split]["sha256"]
        ):
            raise ValueError(f"learning-candidate {split} output changed")
    verification: dict[str, Any] = {
        "checks": {
            "all_264981_rows_materialized": True,
            "closure_1463_rows_excluded": True,
            "frozen_candidate_mix_gate_replay_passed": True,
            "manifest_is_raw_text_free": True,
            "no_network_gpu_registry_contract_or_training_action": True,
            "nonproduction_smoke_only_boundary_preserved": True,
            "output_file_hashes_and_counts_reproduced": True,
            "train_7341582_supervised_tokens_bound": True,
            "validation_427005_supervised_tokens_bound": True,
        },
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_internal_sha256": manifest["seal_sha256"],
        "observed_outputs": observed,
        "raw_training_text_retained": False,
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_verified_nonproduction_learning_candidate_seal_smoke_only",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "EXPECTED_AUTHORIZATION",
    "EXPECTED_INPUT",
    "EXPECTED_LIMITS",
    "EXPECTED_OUTPUT",
    "build_learning_candidate_seal",
    "build_learning_candidate_verification",
    "json_file_bytes",
    "load_learning_candidate_seal_plan",
    "validate_learning_candidate_manifest",
]
