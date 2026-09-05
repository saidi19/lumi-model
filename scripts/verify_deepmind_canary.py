#!/usr/bin/env python3
"""Independently verify the bounded private DeepMind adapter canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from lumi.posttraining_data import filter_sft_records
from lumi.posttraining_deepmind_canary_run import (
    DETERMINISM_ADAPTER,
    EXPECTED_GENERATED_RECORDS,
    PRIVATE_ROOT,
    SOURCE_BYTES,
    VERIFICATION_SCHEMA_VERSION,
    apply_prompt_near_dedup,
    audit_source_runtime,
    build_candidate_records,
    file_sha256,
    generation_schedule,
    jsonl_file_bytes,
    load_canary_plan,
    load_runtime_components,
    sha256_value,
    validate_generated_records,
    validate_raw_text_free_report,
    write_json_atomic,
)
from lumi.posttraining_oasst1_full_run import measure_token_fit


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_json_object_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"JSON root must be an object list: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_canary_plan(plan_path, repository_root=root)
    private_root = Path(plan["output"]["private_root"]).resolve()
    if private_root != Path(PRIVATE_ROOT).resolve():
        raise ValueError("DeepMind private canary root changed")
    report_path = (root / plan["output"]["raw_text_free_report_path"]).resolve()
    verification_path = (
        root / plan["output"]["raw_text_free_verification_path"]
    ).resolve()
    if verification_path.exists():
        raise FileExistsError(
            f"refusing to overwrite DeepMind canary verification: {verification_path}"
        )
    report = _load_json(report_path)
    validate_raw_text_free_report(report)
    if report["plan_file_sha256"] != file_sha256(plan_path):
        raise ValueError("DeepMind canary report plan binding changed")
    private_report = _load_json(private_root / "raw-text-free-canary-report.json")
    if private_report != report:
        raise ValueError("DeepMind private and Git reports differ")

    artifact_paths = {
        "accepted_records": private_root / "curated/accepted-records.jsonl",
        "attribution_manifest": private_root / "curated/attribution-manifest.jsonl",
        "filter_decisions": private_root / "curated/filter-decisions.json",
        "generated_adapter_records": private_root / "generated/adapter-records.jsonl",
        "near_dedup_ledger": private_root / "curated/near-dedup-ledger.jsonl",
        "shard_ledger": private_root / "audit/generation-shards.json",
        "source_runtime_audit": private_root / "audit/source-runtime-audit.json",
        "token_metrics": private_root / "curated/token-metrics.json",
    }
    reported_artifacts = {item["name"]: item for item in report["artifacts"]["objects"]}
    if set(reported_artifacts) != set(artifact_paths):
        raise ValueError("DeepMind private artifact inventory changed")
    for name, path in artifact_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"DeepMind private artifact is missing: {name}")
        expected = reported_artifacts[name]
        if expected["bytes"] != path.stat().st_size:
            raise ValueError(f"DeepMind private artifact size changed: {name}")
        if expected["sha256"] != file_sha256(path):
            raise ValueError(f"DeepMind private artifact hash changed: {name}")

    generated = _load_jsonl(artifact_paths["generated_adapter_records"])
    generation_metrics = validate_generated_records(generated)
    if generation_metrics != {
        key: value
        for key, value in report["canary"]["generation"].items()
        if key
        in {
            "conflicting_answer_prompt_groups",
            "different_seed_prompt_groups",
            "different_seed_prompt_overlap",
            "flat_modules_observed",
            "generated_records",
            "module_counts_sha256",
            "prompt_inventory_sha256",
            "seed_inventory_sha256",
            "split_counts",
            "train_validation_prompt_overlap",
            "train_validation_seed_overlap",
        }
    }:
        raise ValueError("DeepMind generated-record metrics changed")
    if len(generated) != EXPECTED_GENERATED_RECORDS:
        raise ValueError("DeepMind persisted identity count changed")

    shard_ledger = _load_json_object_list(artifact_paths["shard_ledger"])
    if len(shard_ledger) != 16:
        raise ValueError("DeepMind shard ledger changed")
    cursor = 0
    for shard, expected in zip(shard_ledger, generation_schedule(), strict=True):
        if shard.get("split") != expected["split"]:
            raise ValueError("DeepMind shard split changed")
        if shard.get("top_module") != expected["top_module"]:
            raise ValueError("DeepMind shard top module changed")
        if shard.get("count") != expected["count"]:
            raise ValueError("DeepMind shard count changed")
        shard_rows = generated[cursor : cursor + expected["count"]]
        cursor += expected["count"]
        payload = jsonl_file_bytes(shard_rows)
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        if shard.get("output_bytes") != len(payload):
            raise ValueError("DeepMind shard output byte count changed")
        if shard.get("output_sha256") != payload_sha256:
            raise ValueError("DeepMind shard output hash changed")
        if shard.get("replay_sha256") != payload_sha256 or shard.get("replay_match") is not True:
            raise ValueError("DeepMind shard replay evidence changed")
    if cursor != len(generated):
        raise AssertionError("DeepMind shard cursor changed")

    source_audit = _load_json(artifact_paths["source_runtime_audit"])
    replayed_source_audit = audit_source_runtime(plan["source"]["local_source_root"])
    runtime_only_fields = {
        "runtime_determinism_adapter",
        "runtime_loaded_source_modules",
        "runtime_loaded_source_modules_sha256",
        "runtime_official_test_calls",
        "runtime_train_entrypoint",
    }
    static_source_audit = {
        key: value for key, value in source_audit.items() if key not in runtime_only_fields
    }
    if static_source_audit != replayed_source_audit:
        raise ValueError("DeepMind static source audit changed")
    loaded_modules = source_audit.get("runtime_loaded_source_modules")
    if not isinstance(loaded_modules, list):
        raise ValueError("DeepMind runtime module closure is missing")
    if source_audit.get("runtime_loaded_source_modules_sha256") != sha256_value(
        loaded_modules
    ):
        raise ValueError("DeepMind runtime module closure hash changed")
    forbidden_runtime = {
        "mathematics_dataset.generate",
        "mathematics_dataset.generate_settings",
        "mathematics_dataset.generate_to_file",
    }
    if forbidden_runtime.intersection(loaded_modules):
        raise ValueError("DeepMind runtime imported an excluded generator module")
    if source_audit.get("runtime_official_test_calls") != 0:
        raise ValueError("DeepMind official test/evaluation generator was called")
    if source_audit.get("runtime_determinism_adapter") != DETERMINISM_ADAPTER:
        raise ValueError("DeepMind runtime determinism adapter changed")
    if source_audit.get("runtime_train_entrypoint") != (
        "mathematics_dataset.modules.modules.train"
    ):
        raise ValueError("DeepMind train-only entrypoint changed")
    if source_audit != report["canary"]["source_runtime_audit"]:
        raise ValueError("DeepMind report source audit changed")

    records = _load_jsonl(artifact_paths["accepted_records"])
    attributions = _load_jsonl(artifact_paths["attribution_manifest"])
    filter_decisions = _load_json(artifact_paths["filter_decisions"])
    near_ledger = _load_jsonl(artifact_paths["near_dedup_ledger"])
    token_metrics = _load_json(artifact_paths["token_metrics"])
    if len(records) != report["canary"]["retained_records"]:
        raise ValueError("DeepMind retained record count changed")
    expected_fields = {
        "language",
        "messages",
        "provenance",
        "quality",
        "record_id",
        "safety",
        "schema_version",
        "source",
        "split",
        "task_category",
    }
    for record in records:
        if set(record) != expected_fields:
            raise ValueError("DeepMind candidate record schema changed")
        messages = record.get("messages")
        if not isinstance(messages, list) or [item.get("role") for item in messages] != [
            "user",
            "assistant",
        ]:
            raise ValueError("DeepMind candidate message sequence changed")
        if not str(messages[1].get("content", "")).startswith("The answer is "):
            raise ValueError("DeepMind answer adapter template changed")
        source = record.get("source")
        if not isinstance(source, dict) or source.get("source_id") != (
            "deepmind_mathematics_dataset"
        ):
            raise ValueError("DeepMind candidate source binding changed")
        if record.get("split") not in {"train", "validation"}:
            raise ValueError("DeepMind candidate split changed")

    projected_records, projected_attributions = build_candidate_records(generated)
    policy, benchmark, template, tokenizer = load_runtime_components(
        plan,
        repository_root=root,
    )
    structurally_accepted, replayed_filter = filter_sft_records(
        projected_records,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=None,
    )
    if replayed_filter != filter_decisions:
        raise ValueError("DeepMind schema/PII/secret/benchmark filtering changed")
    deduped, replayed_near_ledger, replayed_near_metrics = apply_prompt_near_dedup(
        structurally_accepted
    )
    if replayed_near_ledger != near_ledger:
        raise ValueError("DeepMind global near-dedup ledger changed")
    if replayed_near_metrics != report["canary"]["near_duplicate"]:
        raise ValueError("DeepMind near-dedup metrics changed")
    measured, replayed_token_metrics = measure_token_fit(
        deduped,
        template=template,
        tokenizer=tokenizer,
    )
    reconstructed_records = sorted(
        (dict(record) for record in measured),
        key=lambda row: (str(row["split"]), str(row["record_id"])),
    )
    if reconstructed_records != records:
        raise ValueError("DeepMind final candidate reconstruction changed")
    if replayed_token_metrics != token_metrics:
        raise ValueError("DeepMind frozen-tokenizer measurement changed")
    if token_metrics != report["canary"]["token_measurement"]:
        raise ValueError("DeepMind report token metrics changed")
    final_ids = {str(record["record_id"]) for record in records}
    reconstructed_attributions = sorted(
        (row for row in projected_attributions if str(row["record_id"]) in final_ids),
        key=lambda row: str(row["record_id"]),
    )
    if reconstructed_attributions != attributions:
        raise ValueError("DeepMind per-record attribution reconstruction changed")
    rededuped, _ledger, final_dedup_metrics = apply_prompt_near_dedup(records)
    if len(rededuped) != len(records) or final_dedup_metrics["rejected"] != 0:
        raise ValueError("DeepMind final candidate contains a near duplicate")

    generated_item = reported_artifacts["generated_adapter_records"]
    decoded_bytes = (
        SOURCE_BYTES
        + int(report["artifacts"]["total_bytes"])
        + int(generated_item["bytes"])
    )
    if decoded_bytes != report["canary"]["decoded_bytes_accounted"]:
        raise ValueError("DeepMind decoded-byte accounting changed")
    if decoded_bytes > plan["limits"]["maximum_decoded_bytes"]:
        raise ValueError("DeepMind decoded-byte ceiling exceeded")
    if report["execution"] != {
        "additional_network_bytes": 0,
        "external_api_calls": 0,
        "gpu_used": False,
        "package_installations": 0,
        "runtime_seconds": report["execution"]["runtime_seconds"],
        "source_acquisition_reused_verified_exact_objects": True,
    }:
        raise ValueError("DeepMind execution boundary changed")

    verification: dict[str, Any] = {
        "checks": {
            "assistant_only_loss_and_4k_fit_replayed": True,
            "benchmark_decontamination_replayed": True,
            "cross_process_same_seed_replay_verified": True,
            "decoded_byte_ceiling_verified": decoded_bytes,
            "exact_30_object_source_reverified": True,
            "generated_record_identities": len(generated),
            "global_exact_and_near_duplicates_in_final_candidate": 0,
            "official_test_or_evaluation_records": 0,
            "package_installations": 0,
            "per_record_attribution_verified": True,
            "pii_and_secret_filter_replayed": True,
            "runtime_network_guard_verified": True,
            "source_registry_check_deferred_until_separate_promotion": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": report["report_sha256"],
        "retained_records": len(records),
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_cpu_only_nonproduction_deepmind_adapter_canary",
        "supervised_tokens": token_metrics["supervised_tokens"],
    }
    verification["verification_sha256"] = sha256_value(verification)
    write_json_atomic(verification_path, verification)
    print(
        "PASS: independently verified DeepMind CPU canary; "
        f"identities={len(generated)} retained={len(records)} "
        f"supervised_tokens={token_metrics['supervised_tokens']} "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
