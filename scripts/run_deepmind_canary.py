#!/usr/bin/env python3
"""Run the exact bounded DeepMind Mathematics CPU adapter canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from lumi.posttraining_data import filter_sft_records
from lumi.posttraining_deepmind_canary_run import (
    DETERMINISM_ADAPTER,
    EXPECTED_FLAT_MODULE_INVENTORY_SHA256,
    EXPECTED_FLAT_MODULES,
    EXPECTED_GENERATED_RECORDS,
    PRIVATE_ROOT,
    REPORT_SCHEMA_VERSION,
    SOURCE_BYTES,
    SOURCE_ROOT,
    apply_prompt_near_dedup,
    artifact_inventory,
    audit_source_runtime,
    build_candidate_records,
    count_by_split,
    file_sha256,
    generation_schedule,
    json_file_bytes,
    jsonl_file_bytes,
    load_canary_plan,
    load_runtime_components,
    sha256_value,
    validate_generated_records,
    validate_raw_text_free_report,
    write_bytes_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)
from lumi.posttraining_oasst1_full_run import measure_token_fit

STATS_PREFIX = "LUMI_DEEPMIND_STATS="


def _parse_jsonl(payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"DeepMind worker row is not an object: {line_number}")
        rows.append(value)
    return rows


def _parse_worker_stats(stderr: bytes) -> dict[str, Any]:
    matches = [
        line[len(STATS_PREFIX) :]
        for line in stderr.decode("utf-8", errors="replace").splitlines()
        if line.startswith(STATS_PREFIX)
    ]
    if len(matches) != 1:
        raise ValueError(
            "DeepMind worker stats marker changed: "
            f"stderr_sha256={hashlib.sha256(stderr).hexdigest()}"
        )
    value = json.loads(matches[0])
    if not isinstance(value, dict):
        raise ValueError("DeepMind worker stats root changed")
    return value


def _run_worker(
    *,
    root: Path,
    worker: Path,
    shard: dict[str, Any],
    source_root: Path,
    overlay_root: Path,
    python_executable: Path,
) -> tuple[bytes, dict[str, Any]]:
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "NO_PROXY": "*",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": os.pathsep.join((str(source_root), str(overlay_root))),
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    completed = subprocess.run(
        [
            str(python_executable),
            str(worker),
            "--source-root",
            str(source_root),
            "--top-module",
            str(shard["top_module"]),
            "--namespace",
            str(shard["namespace"]),
            "--count",
            str(shard["count"]),
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        timeout=900,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "DeepMind generation worker failed without exposing source text: "
            f"split={shard['split']} module={shard['top_module']} "
            f"returncode={completed.returncode} "
            f"stdout_sha256={hashlib.sha256(completed.stdout).hexdigest()} "
            f"stderr_sha256={hashlib.sha256(completed.stderr).hexdigest()}"
        )
    stats = _parse_worker_stats(completed.stderr)
    if stats.get("records") != shard["count"]:
        raise ValueError("DeepMind worker record count changed")
    if stats.get("top_module") != shard["top_module"]:
        raise ValueError("DeepMind worker top-module binding changed")
    if stats.get("namespace") != shard["namespace"]:
        raise ValueError("DeepMind worker namespace binding changed")
    if stats.get("flat_module_count") != EXPECTED_FLAT_MODULES:
        raise ValueError("DeepMind worker flat-module count changed")
    if stats.get("flat_module_inventory_sha256") != (
        EXPECTED_FLAT_MODULE_INVENTORY_SHA256
    ):
        raise ValueError("DeepMind worker flat-module inventory changed")
    if stats.get("determinism_adapter") != DETERMINISM_ADAPTER:
        raise ValueError("DeepMind worker determinism adapter changed")
    if stats.get("official_test_calls") != 0:
        raise ValueError("DeepMind worker called an official test generator")
    if stats.get("network_guard_active") is not True:
        raise ValueError("DeepMind worker network guard was not active")
    if stats.get("upstream_generate_cli_imported") is not False:
        raise ValueError("DeepMind worker imported upstream generate CLI")
    return completed.stdout, stats


def _artifact_payload_sizes(
    *,
    generated_payload: bytes,
    candidate_records: list[dict[str, Any]],
    attributions: list[dict[str, Any]],
    filter_report: dict[str, Any],
    near_ledger: list[dict[str, Any]],
    token_metrics: dict[str, Any],
    source_audit: dict[str, Any],
    shard_ledger: list[dict[str, Any]],
) -> int:
    return sum(
        (
            SOURCE_BYTES,
            len(generated_payload),
            len(generated_payload),  # byte-exact cross-process replay payload
            len(jsonl_file_bytes(candidate_records)),
            len(jsonl_file_bytes(attributions)),
            len(json_file_bytes(filter_report)),
            len(jsonl_file_bytes(near_ledger)),
            len(json_file_bytes(token_metrics)),
            len(json_file_bytes(source_audit)),
            len(json_file_bytes(shard_ledger)),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    started = time.monotonic()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_canary_plan(plan_path, repository_root=root)
    private_root = Path(plan["output"]["private_root"]).resolve()
    if private_root != Path(PRIVATE_ROOT).resolve():
        raise ValueError("DeepMind private canary root changed")
    artifact_root = (root.parent / "lumi-artifacts").resolve()
    if not private_root.is_relative_to(artifact_root):
        raise ValueError("DeepMind private canary root escaped lumi-artifacts")
    if private_root.exists() and any(private_root.rglob("*")):
        raise FileExistsError(f"refusing nonempty DeepMind canary root: {private_root}")
    report_path = (root / plan["output"]["raw_text_free_report_path"]).resolve()
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite DeepMind canary report: {report_path}")

    source_root = Path(plan["source"]["local_source_root"]).resolve()
    if source_root != Path(SOURCE_ROOT).resolve():
        raise ValueError("DeepMind source root changed")
    overlay_root = Path(plan["runtime"]["dependency_overlay"]).resolve()
    # Keep the virtualenv launcher path intact. Resolving its symlink would execute the
    # base interpreter without the pinned NumPy/SymPy environment.
    python_executable = Path(plan["runtime"]["python_executable"])
    if not overlay_root.is_dir() or not python_executable.is_file():
        raise FileNotFoundError("DeepMind pinned CPU runtime is missing")

    print("AUDIT: exact 30-object source, imports, RNG, and global state", flush=True)
    source_audit = audit_source_runtime(source_root)
    worker = (root / plan["bindings"]["canary_worker"]["path"]).resolve()
    generated_chunks: list[bytes] = []
    shard_ledger: list[dict[str, Any]] = []
    runtime_loaded_modules: list[str] | None = None
    schedule = generation_schedule()
    for shard_number, shard in enumerate(schedule, start=1):
        print(
            f"GENERATE: shard {shard_number:02d}/{len(schedule)} "
            f"{shard['split']}:{shard['top_module']} identities={shard['count']}",
            flush=True,
        )
        payload, stats = _run_worker(
            root=root,
            worker=worker,
            shard=shard,
            source_root=source_root,
            overlay_root=overlay_root,
            python_executable=python_executable,
        )
        replay_payload, replay_stats = _run_worker(
            root=root,
            worker=worker,
            shard=shard,
            source_root=source_root,
            overlay_root=overlay_root,
            python_executable=python_executable,
        )
        if payload != replay_payload or stats != replay_stats:
            raise ValueError(
                "DeepMind same-seed cross-process replay diverged: "
                f"split={shard['split']} module={shard['top_module']}"
            )
        rows = _parse_jsonl(payload)
        if len(rows) != shard["count"]:
            raise ValueError("DeepMind worker JSONL row count changed")
        loaded_modules = stats.get("loaded_source_modules")
        if not isinstance(loaded_modules, list) or not all(
            isinstance(item, str) for item in loaded_modules
        ):
            raise ValueError("DeepMind runtime source-module audit changed")
        if runtime_loaded_modules is None:
            runtime_loaded_modules = loaded_modules
        elif runtime_loaded_modules != loaded_modules:
            raise ValueError("DeepMind runtime import closure changed across shards")
        generated_chunks.append(payload)
        shard_ledger.append(
            {
                "count": shard["count"],
                "dropped_length": stats["dropped_length"],
                "exception_counts": stats["exception_counts"],
                "module_sample_counts": stats["module_sample_counts"],
                "namespace": shard["namespace"],
                "output_bytes": len(payload),
                "output_sha256": hashlib.sha256(payload).hexdigest(),
                "replay_match": True,
                "replay_sha256": hashlib.sha256(replay_payload).hexdigest(),
                "split": shard["split"],
                "top_module": shard["top_module"],
            }
        )

    generated_payload = b"".join(generated_chunks)
    generated_records = _parse_jsonl(generated_payload)
    generation_metrics = validate_generated_records(generated_records)
    if generation_metrics["generated_records"] != EXPECTED_GENERATED_RECORDS:
        raise ValueError("DeepMind canary identity ceiling changed")
    if runtime_loaded_modules is None:
        raise AssertionError("DeepMind runtime import closure was not observed")
    excluded_runtime_modules = {
        "mathematics_dataset.generate",
        "mathematics_dataset.generate_settings",
        "mathematics_dataset.generate_to_file",
    }
    if excluded_runtime_modules.intersection(runtime_loaded_modules):
        raise ValueError("DeepMind runtime imported an excluded generator module")
    source_audit["runtime_loaded_source_modules"] = runtime_loaded_modules
    source_audit["runtime_loaded_source_modules_sha256"] = sha256_value(
        runtime_loaded_modules
    )
    source_audit["runtime_official_test_calls"] = 0
    source_audit["runtime_determinism_adapter"] = DETERMINISM_ADAPTER
    source_audit["runtime_train_entrypoint"] = "mathematics_dataset.modules.modules.train"

    print("FILTER: schema, PII/secret, benchmark, global exact/near dedup", flush=True)
    candidate_records, attributions = build_candidate_records(generated_records)
    policy, benchmark, template, tokenizer = load_runtime_components(
        plan,
        repository_root=root,
    )
    structurally_accepted, filter_report = filter_sft_records(
        candidate_records,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=None,
    )
    near_accepted, near_ledger, near_metrics = apply_prompt_near_dedup(
        structurally_accepted
    )
    token_accepted, token_metrics = measure_token_fit(
        near_accepted,
        template=template,
        tokenizer=tokenizer,
    )
    final_records = sorted(
        (dict(record) for record in token_accepted),
        key=lambda row: (str(row["split"]), str(row["record_id"])),
    )
    final_ids = {str(record["record_id"]) for record in final_records}
    final_attributions = sorted(
        (row for row in attributions if str(row["record_id"]) in final_ids),
        key=lambda row: str(row["record_id"]),
    )
    if len(final_attributions) != len(final_records):
        raise ValueError("DeepMind per-record attribution coverage changed")
    rededuped, _replay_ledger, replay_dedup_metrics = apply_prompt_near_dedup(
        final_records
    )
    if len(rededuped) != len(final_records) or replay_dedup_metrics["rejected"] != 0:
        raise ValueError("DeepMind final candidate retained a prompt near-duplicate")

    decoded_bytes = _artifact_payload_sizes(
        generated_payload=generated_payload,
        candidate_records=final_records,
        attributions=final_attributions,
        filter_report=filter_report,
        near_ledger=near_ledger,
        token_metrics=token_metrics,
        source_audit=source_audit,
        shard_ledger=shard_ledger,
    )
    if decoded_bytes > plan["limits"]["maximum_decoded_bytes"]:
        raise ValueError("DeepMind canary exceeded decoded byte ceiling")

    private_paths = {
        "accepted_records": private_root / "curated/accepted-records.jsonl",
        "attribution_manifest": private_root / "curated/attribution-manifest.jsonl",
        "filter_decisions": private_root / "curated/filter-decisions.json",
        "generated_adapter_records": private_root / "generated/adapter-records.jsonl",
        "near_dedup_ledger": private_root / "curated/near-dedup-ledger.jsonl",
        "shard_ledger": private_root / "audit/generation-shards.json",
        "source_runtime_audit": private_root / "audit/source-runtime-audit.json",
        "token_metrics": private_root / "curated/token-metrics.json",
    }
    write_bytes_atomic(private_paths["generated_adapter_records"], generated_payload)
    write_jsonl_atomic(private_paths["accepted_records"], final_records)
    write_jsonl_atomic(private_paths["attribution_manifest"], final_attributions)
    write_json_atomic(private_paths["filter_decisions"], filter_report)
    write_jsonl_atomic(private_paths["near_dedup_ledger"], near_ledger)
    write_json_atomic(private_paths["shard_ledger"], shard_ledger)
    write_json_atomic(private_paths["source_runtime_audit"], source_audit)
    write_json_atomic(private_paths["token_metrics"], token_metrics)
    artifacts = artifact_inventory(
        private_paths,
        raw_text_artifacts={"accepted_records", "generated_adapter_records"},
    )

    exception_counts: Counter[str] = Counter()
    dropped_length = 0
    for shard in shard_ledger:
        exception_counts.update(shard["exception_counts"])
        dropped_length += int(shard["dropped_length"])
    report: dict[str, Any] = {
        "artifacts": artifacts,
        "authorization": dict(plan["authorization"]),
        "bindings": {
            "dependency_overlay_receipt_internal_sha256": (
                "192fbac2bf5cf1049ff5bcd33e84507bf55d686e296b9dfb2d67cd7b5c0bce72"
            ),
            "dependency_overlay_verification_internal_sha256": (
                "10ce6707e13472fcdf8641a4f9c5b5a8cb120865e0ad5a1e0b4ff51ceffabd59"
            ),
            "source_acquisition_receipt_internal_sha256": (
                "4d68a934d2d970ef6396c4786fe696f28b4b406e69e1dd7f404515237e3433a9"
            ),
            "source_acquisition_verification_internal_sha256": (
                "e4bf0ad734e8c718e7b0c9e0cbb490a69e7d233d917bee2d77812024e81c00ee"
            ),
        },
        "canary": {
            "decoded_bytes_accounted": decoded_bytes,
            "filter": {
                "accepted": filter_report["accepted"],
                "accepted_dataset_sha256": filter_report["accepted_dataset_sha256"],
                "raw_text_retained": False,
                "rejected": filter_report["rejected"],
                "rejection_counts": filter_report["rejection_counts"],
                "report_sha256": filter_report["report_sha256"],
                "source_registry_check_deferred_until_promotion": True,
            },
            "generation": {
                **generation_metrics,
                "cross_process_replay_identities": EXPECTED_GENERATED_RECORDS,
                "dropped_length_attempts": dropped_length,
                "exception_counts": dict(sorted(exception_counts.items())),
                "fresh_process_shards": len(shard_ledger),
                "official_test_or_evaluation_records": 0,
                "persisted_record_identities": EXPECTED_GENERATED_RECORDS,
                "replay_byte_exact": True,
                "replay_payload_sha256": hashlib.sha256(generated_payload).hexdigest(),
            },
            "near_duplicate": near_metrics,
            "retained_records": len(final_records),
            "source_runtime_audit": source_audit,
            "split_counts": count_by_split(final_records),
            "token_measurement": token_metrics,
        },
        "execution": {
            "additional_network_bytes": 0,
            "external_api_calls": 0,
            "gpu_used": False,
            "package_installations": 0,
            "runtime_seconds": round(time.monotonic() - started, 3),
            "source_acquisition_reused_verified_exact_objects": True,
        },
        "limits": dict(plan["limits"]),
        "observed_at": plan["observed_at"],
        "plan_file_sha256": file_sha256(plan_path),
        "production": {
            "candidate_registry_promotions": 0,
            "dataset_sealed": False,
            "gpu_used": False,
            "production_eligible": False,
            "production_sft_authorized": False,
            "sft_training_started": False,
        },
        "raw_training_text_present": False,
        "remaining_gates": [
            "canary_verification_receipt",
            "separate_source_registry_promotion_if_selected",
            "production_dataset_mix_and_seal",
            "separate_gpu_sft_authorization",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": {
            "declared_license_id": plan["source"]["declared_license_id"],
            "exact_object_bytes": plan["source"]["exact_object_bytes"],
            "exact_object_count": plan["source"]["exact_object_count"],
            "local_source_inventory_sha256": source_audit[
                "local_source_inventory_sha256"
            ],
            "revision": plan["source"]["revision"],
            "source_id": plan["source"]["source_id"],
        },
        "status": "verified_cpu_only_nonproduction_adapter_canary",
    }
    report["report_sha256"] = sha256_value(report)
    validate_raw_text_free_report(report)
    private_report_path = private_root / "raw-text-free-canary-report.json"
    write_json_atomic(private_report_path, report)
    write_bytes_atomic(report_path, json_file_bytes(report))
    print(
        "PASS: DeepMind CPU canary generated and filtered; "
        f"identities={EXPECTED_GENERATED_RECORDS} retained={len(final_records)} "
        f"supervised_tokens={token_metrics['supervised_tokens']} "
        f"report={report['report_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
