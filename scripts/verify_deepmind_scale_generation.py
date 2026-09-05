#!/usr/bin/env python3
"""Independently verify the private DeepMind Mathematics scale candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumi.posttraining_data import filter_sft_records
from lumi.posttraining_deepmind_scale_generation import (
    DETERMINISM_ADAPTER,
    EXPECTED_GENERATED_RECORDS,
    MAXIMUM_PRIVATE_BYTES,
    MAXIMUM_RUNTIME_SECONDS,
    MAXIMUM_TRAIN_SUPERVISED_TOKENS,
    MINIMUM_TRAIN_SUPERVISED_TOKENS,
    PRIVATE_ROOT,
    VERIFICATION_SCHEMA_VERSION,
    apply_prompt_near_dedup,
    audit_source_runtime,
    build_candidate_records,
    build_scale_owner_review_html,
    derive_identity_seed,
    derive_rng_seed,
    deterministic_owner_sample,
    file_sha256,
    generation_schedule,
    load_runtime_components,
    load_scale_plan,
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


def _load_json_list(path: Path) -> list[dict[str, Any]]:
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
                raise ValueError(f"JSONL row is invalid: {path}:{line_number}")
            rows.append(value)
    return rows


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("DeepMind scale timestamp is not UTC")
    return parsed.astimezone(UTC)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_scale_plan(plan_path, repository_root=root)
    private_root = Path(plan["output"]["private_root"]).resolve()
    if private_root != Path(PRIVATE_ROOT).resolve():
        raise ValueError("DeepMind scale private root changed")
    report_path = (root / plan["output"]["raw_text_free_report_path"]).resolve()
    verification_path = (
        root / plan["output"]["raw_text_free_verification_path"]
    ).resolve()
    if verification_path.exists():
        raise FileExistsError(
            f"refusing to overwrite DeepMind scale verification: {verification_path}"
        )
    report = _load_json(report_path)
    validate_raw_text_free_report(report)
    if report["plan_file_sha256"] != file_sha256(plan_path):
        raise ValueError("DeepMind scale report plan binding changed")
    private_report = _load_json(private_root / "raw-text-free-scale-report.json")
    if private_report != report:
        raise ValueError("DeepMind private and repository reports differ")

    artifact_paths = {
        "accepted_records": private_root / "curated/accepted-records.jsonl",
        "attribution_manifest": private_root / "curated/attribution-manifest.jsonl",
        "filter_decisions": private_root / "curated/filter-decisions.json",
        "generated_adapter_records": private_root / "generated/adapter-records.jsonl",
        "near_dedup_ledger": private_root / "curated/near-dedup-ledger.jsonl",
        "owner_review_html": private_root / "review/owner-review-32.html",
        "owner_review_sample": private_root / "review/owner-review-32.jsonl",
        "shard_ledger": private_root / "audit/generation-shards.json",
        "source_runtime_audit": private_root / "audit/source-runtime-audit.json",
        "token_metrics": private_root / "curated/token-metrics.json",
    }
    reported_artifacts = {
        str(item["name"]): item for item in report["artifacts"]["objects"]
    }
    if set(reported_artifacts) != set(artifact_paths):
        raise ValueError("DeepMind scale artifact inventory changed")
    for name, path in artifact_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"DeepMind scale artifact is missing: {name}")
        expected = reported_artifacts[name]
        if expected["bytes"] != path.stat().st_size:
            raise ValueError(f"DeepMind scale artifact size changed: {name}")
        if expected["sha256"] != file_sha256(path):
            raise ValueError(f"DeepMind scale artifact hash changed: {name}")
    if int(report["artifacts"]["total_bytes"]) > MAXIMUM_PRIVATE_BYTES:
        raise ValueError("DeepMind scale private-byte ceiling changed")

    print("VERIFY: generated identities, shard hashes, and replay ledger", flush=True)
    generated_payload = artifact_paths["generated_adapter_records"].read_bytes()
    generated = _load_jsonl(artifact_paths["generated_adapter_records"])
    generation_metrics = validate_generated_records(generated)
    reported_generation = report["scale"]["generation"]
    if generation_metrics != {
        key: reported_generation[key] for key in generation_metrics
    }:
        raise ValueError("DeepMind scale generation metrics changed")
    if len(generated) != EXPECTED_GENERATED_RECORDS:
        raise ValueError("DeepMind scale persisted identity count changed")
    shard_ledger = _load_json_list(artifact_paths["shard_ledger"])
    if len(shard_ledger) != 16:
        raise ValueError("DeepMind scale shard ledger changed")
    cursor = 0
    for shard, expected in zip(shard_ledger, generation_schedule(), strict=True):
        for field in (
            "count",
            "global_offset",
            "namespace",
            "split",
            "top_module",
        ):
            if shard.get(field) != expected[field]:
                raise ValueError(f"DeepMind scale shard changed: {field}")
        shard_rows = generated[cursor : cursor + expected["count"]]
        cursor += expected["count"]
        payload = _jsonl_bytes(shard_rows)
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        if shard.get("output_bytes") != len(payload):
            raise ValueError("DeepMind scale shard byte count changed")
        if shard.get("output_sha256") != payload_sha256:
            raise ValueError("DeepMind scale shard output hash changed")
        if (
            shard.get("replay_sha256") != payload_sha256
            or shard.get("replay_match") is not True
        ):
            raise ValueError("DeepMind scale full replay evidence changed")
        first = shard_rows[0]
        last = shard_rows[-1]
        if first["seed"] != derive_identity_seed(
            expected["namespace"], expected["module_file"], 0
        ) or last["seed"] != derive_identity_seed(
            expected["namespace"],
            expected["module_file"],
            expected["count"] - 1,
        ):
            raise ValueError("DeepMind scale shard identity endpoints changed")
        if first["rng_seed"] != derive_rng_seed(expected["global_offset"]) or last[
            "rng_seed"
        ] != derive_rng_seed(expected["global_offset"] + expected["count"] - 1):
            raise ValueError("DeepMind scale shard RNG endpoints changed")
    if cursor != len(generated):
        raise AssertionError("DeepMind scale shard cursor changed")
    if hashlib.sha256(generated_payload).hexdigest() != reported_generation[
        "replay_payload_sha256"
    ]:
        raise ValueError("DeepMind scale generated payload hash changed")

    source_audit = _load_json(artifact_paths["source_runtime_audit"])
    replayed_source_audit = audit_source_runtime(plan["source"]["local_source_root"])
    runtime_only = {
        "runtime_determinism_adapter",
        "runtime_loaded_source_modules",
        "runtime_loaded_source_modules_sha256",
        "runtime_official_test_calls",
        "runtime_train_entrypoint",
    }
    if {
        key: value for key, value in source_audit.items() if key not in runtime_only
    } != replayed_source_audit:
        raise ValueError("DeepMind scale static source audit changed")
    loaded_modules = source_audit.get("runtime_loaded_source_modules")
    if not isinstance(loaded_modules, list) or source_audit.get(
        "runtime_loaded_source_modules_sha256"
    ) != sha256_value(loaded_modules):
        raise ValueError("DeepMind scale runtime closure changed")
    if source_audit.get("runtime_determinism_adapter") != DETERMINISM_ADAPTER:
        raise ValueError("DeepMind scale determinism adapter changed")
    if source_audit.get("runtime_official_test_calls") != 0:
        raise ValueError("DeepMind scale official split call count changed")
    if source_audit != report["scale"]["source_runtime_audit"]:
        raise ValueError("DeepMind scale report source audit changed")

    print("VERIFY: filter, global dedup, 4K fit, and assistant-only tokens", flush=True)
    records = _load_jsonl(artifact_paths["accepted_records"])
    attributions = _load_jsonl(artifact_paths["attribution_manifest"])
    filter_decisions = _load_json(artifact_paths["filter_decisions"])
    near_ledger = _load_jsonl(artifact_paths["near_dedup_ledger"])
    token_metrics = _load_json(artifact_paths["token_metrics"])
    expected_record_fields = {
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
        if set(record) != expected_record_fields:
            raise ValueError("DeepMind scale candidate schema changed")
        if [message.get("role") for message in record["messages"]] != [
            "user",
            "assistant",
        ]:
            raise ValueError("DeepMind scale message sequence changed")
        if not str(record["messages"][1].get("content", "")).startswith(
            "The answer is "
        ):
            raise ValueError("DeepMind scale answer template changed")
        if record["source"].get("source_id") != "deepmind_mathematics_dataset":
            raise ValueError("DeepMind scale source binding changed")
    projected_records, projected_attributions = build_candidate_records(generated)
    projected_records.sort(
        key=lambda row: (
            0 if row["split"] == "validation" else 1,
            str(row["record_id"]),
        )
    )
    policy, benchmark, template, tokenizer = load_runtime_components(
        plan, repository_root=root
    )
    structurally_accepted, replayed_filter = filter_sft_records(
        projected_records,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=None,
    )
    if replayed_filter != filter_decisions:
        raise ValueError("DeepMind scale schema/PII/secret/benchmark filter changed")
    deduped, replayed_near_ledger, replayed_near_metrics = apply_prompt_near_dedup(
        structurally_accepted
    )
    if replayed_near_ledger != near_ledger:
        raise ValueError("DeepMind scale near-dedup ledger changed")
    if replayed_near_metrics != report["scale"]["near_duplicate"]:
        raise ValueError("DeepMind scale near-dedup metrics changed")
    measured, replayed_token_metrics = measure_token_fit(
        deduped, template=template, tokenizer=tokenizer
    )
    reconstructed_records = sorted(
        (dict(record) for record in measured),
        key=lambda row: (str(row["split"]), str(row["record_id"])),
    )
    if reconstructed_records != records:
        raise ValueError("DeepMind scale final candidate reconstruction changed")
    if replayed_token_metrics != token_metrics:
        raise ValueError("DeepMind scale frozen-token measurement changed")
    if token_metrics != report["scale"]["token_measurement"]:
        raise ValueError("DeepMind scale report token metrics changed")
    train_tokens = int(token_metrics["by_split"]["train"]["supervised_tokens"])
    if not MINIMUM_TRAIN_SUPERVISED_TOKENS <= train_tokens <= (
        MAXIMUM_TRAIN_SUPERVISED_TOKENS
    ):
        raise ValueError("DeepMind scale train-token target changed")
    final_ids = {str(record["record_id"]) for record in records}
    reconstructed_attributions = sorted(
        (
            dict(row)
            for row in projected_attributions
            if str(row["record_id"]) in final_ids
        ),
        key=lambda row: str(row["record_id"]),
    )
    if reconstructed_attributions != attributions:
        raise ValueError("DeepMind scale attribution reconstruction changed")
    rededuped, _final_ledger, final_dedup_metrics = apply_prompt_near_dedup(records)
    if len(rededuped) != len(records) or final_dedup_metrics["rejected"] != 0:
        raise ValueError("DeepMind scale final candidate contains a near duplicate")

    print("VERIFY: deterministic 32-row owner packet and boundaries", flush=True)
    owner_sample = _load_jsonl(artifact_paths["owner_review_sample"])
    reconstructed_sample = deterministic_owner_sample(records, attributions)
    if owner_sample != reconstructed_sample:
        raise ValueError("DeepMind scale owner sample changed")
    if sha256_value(owner_sample) != report["owner_review"]["sample_sha256"]:
        raise ValueError("DeepMind scale owner sample hash changed")
    expected_html = build_scale_owner_review_html(owner_sample).encode("utf-8")
    if expected_html != artifact_paths["owner_review_html"].read_bytes():
        raise ValueError("DeepMind scale owner-review HTML changed")
    if report["owner_review"]["complete"] is not False:
        raise ValueError("DeepMind scale falsely claims completed owner review")
    execution = report["execution"]
    if execution["additional_network_bytes"] != 0 or execution["gpu_used"] is not False:
        raise ValueError("DeepMind scale execution boundary changed")
    started_at = _utc(str(execution["started_at_utc"]))
    total_elapsed = (datetime.now(UTC) - started_at).total_seconds()
    if total_elapsed > MAXIMUM_RUNTIME_SECONDS:
        raise TimeoutError("DeepMind scale total authorization window exceeded")

    verification: dict[str, Any] = {
        "checks": {
            "assistant_only_loss_and_4k_fit_replayed": True,
            "benchmark_decontamination_replayed": True,
            "deterministic_owner_review_packet_reconstructed": True,
            "exact_250000_identity_generation_verified": True,
            "exact_30_object_source_reverified": True,
            "full_cross_process_replay_ledger_verified": True,
            "global_exact_and_near_duplicates_in_final_candidate": 0,
            "official_test_or_test_extra_records": 0,
            "per_record_attribution_verified": True,
            "pii_and_secret_filter_replayed": True,
            "private_bytes_verified": report["artifacts"]["total_bytes"],
            "runtime_network_guard_verified": True,
            "train_validation_identity_and_rng_overlap": 0,
        },
        "generated_records": len(generated),
        "observed_at": plan["observed_at"],
        "owner_review_status": "pending_owner_review",
        "production_eligible": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": report["report_sha256"],
        "retained_records": len(records),
        "retained_train_supervised_tokens": train_tokens,
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_cpu_only_nonproduction_deepmind_scale_candidate",
        "total_elapsed_seconds_from_generation_start": round(total_elapsed, 3),
    }
    verification["verification_sha256"] = sha256_value(verification)
    write_json_atomic(verification_path, verification)
    print(
        "PASS: independently verified DeepMind CPU scale candidate; "
        f"generated={len(generated)} retained={len(records)} "
        f"train_supervised_tokens={train_tokens} "
        f"verification={verification['verification_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
