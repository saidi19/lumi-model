#!/usr/bin/env python3
"""Independently verify the private OpenCodeInstruct candidate and receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lumi.posttraining_data import filter_sft_records
from lumi.posttraining_opencodeinstruct_curation import (
    REQUIRED_CODING_BENCHMARKS,
    SHARD_PATH,
    VERIFICATION_SCHEMA_VERSION,
    apply_conversation_near_dedup,
    file_sha256,
    load_opencodeinstruct_plan,
    load_runtime_components,
    load_source_envelopes,
    measure_and_select,
    run_local_unit_test_verifier,
    sha256_value,
    validate_raw_text_free_report,
    verify_raw_objects,
    write_json_atomic,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL object required: {path}:{line_number}")
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_opencodeinstruct_plan(plan_path, repository_root=root)
    private_root = (root / plan["output"]["private_root"]).resolve()
    report_path = (root / plan["output"]["raw_text_free_report_path"]).resolve()
    verification_path = (
        root / plan["output"]["raw_text_free_verification_path"]
    ).resolve()
    if verification_path.exists():
        raise FileExistsError(
            f"refusing to overwrite OpenCodeInstruct verification: {verification_path}"
        )

    verify_raw_objects(private_root / "raw")
    report = _load_json(report_path)
    validate_raw_text_free_report(report)
    private_report = _load_json(private_root / "raw-text-free-curation-report.json")
    if private_report != report:
        raise ValueError("private and Git OpenCodeInstruct reports differ")

    paths = {
        "accepted_records": private_root / "curated" / "accepted-records.jsonl",
        "attribution_manifest": private_root / "curated" / "attribution-manifest.jsonl",
        "filter_decisions": private_root / "curated" / "filter-decisions.json",
        "near_dedup_ledger": private_root / "curated" / "near-dedup-ledger.jsonl",
        "owner_review_html": private_root / "review" / "owner-review-32.html",
        "owner_sample": private_root / "review" / "owner-sample.jsonl",
        "token_ledger": private_root / "curated" / "token-ledger.jsonl",
        "token_metrics": private_root / "curated" / "token-metrics.json",
        "verifier_ledger": (
            private_root / "curated" / "unit-test-verifier-ledger.jsonl"
        ),
    }
    inventory = {item["name"]: item for item in report["artifacts"]["objects"]}
    if set(inventory) != set(paths):
        raise ValueError("OpenCodeInstruct artifact inventory names changed")
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"OpenCodeInstruct private artifact missing: {name}")
        item = inventory[name]
        if item["bytes"] != path.stat().st_size or item["sha256"] != file_sha256(path):
            raise ValueError(f"OpenCodeInstruct artifact changed: {name}")

    records = _load_jsonl(paths["accepted_records"])
    attributions = _load_jsonl(paths["attribution_manifest"])
    owner_sample = _load_jsonl(paths["owner_sample"])
    near_ledger = _load_jsonl(paths["near_dedup_ledger"])
    token_ledger = _load_jsonl(paths["token_ledger"])
    token_metrics = _load_json(paths["token_metrics"])
    filter_decisions = _load_json(paths["filter_decisions"])
    verifier_ledger = _load_jsonl(paths["verifier_ledger"])

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
    record_ids = [str(record.get("record_id")) for record in records]
    if len(record_ids) != len(set(record_ids)) or "None" in record_ids:
        raise ValueError("OpenCodeInstruct record IDs are invalid or duplicated")
    for record in records:
        if set(record) != expected_fields:
            raise ValueError("OpenCodeInstruct accepted record schema changed")
        if [message.get("role") for message in record.get("messages", [])] != [
            "user",
            "assistant",
        ]:
            raise ValueError("OpenCodeInstruct message sequence changed")
        if record.get("language") != "en" or record.get("task_category") != "coding":
            raise ValueError("OpenCodeInstruct task or language changed")
        source = record.get("source", {})
        if (
            source.get("source_id") != "nvidia_opencodeinstruct"
            or source.get("revision") != plan["source"]["revision"]
            or record.get("split") not in {"train", "validation"}
        ):
            raise ValueError("OpenCodeInstruct source or split binding changed")
    if len(records) != report["curation"]["retained_records"]:
        raise ValueError("OpenCodeInstruct retained count changed")

    attribution_ids = [str(row.get("record_id")) for row in attributions]
    if attribution_ids != sorted(record_ids):
        raise ValueError("OpenCodeInstruct attribution coverage changed")
    forbidden = {"input", "output", "unit_tests", "tests_execution_status"}
    if any(forbidden.intersection(row) for row in attributions):
        raise ValueError("OpenCodeInstruct attribution leaked source text")
    if {str(row.get("record_id")) for row in token_ledger} != set(record_ids):
        raise ValueError("OpenCodeInstruct token ledger coverage changed")

    ledger_passed = {
        str(row["record_id"])
        for row in verifier_ledger
        if row.get("verified") is True and row.get("reason") == "verified"
    }
    if not set(record_ids).issubset(ledger_passed):
        raise ValueError("OpenCodeInstruct candidate contains a verifier-failed row")
    if filter_decisions.get("report_sha256") != report["curation"]["filter"][
        "report_sha256"
    ]:
        raise ValueError("OpenCodeInstruct filter receipt changed")
    near_accepted = {
        str(row["record_id"]) for row in near_ledger if row.get("accepted") is True
    }
    if not set(record_ids).issubset(near_accepted):
        raise ValueError("OpenCodeInstruct retained row failed near dedup")

    policy, benchmark, near_config, template, tokenizer = load_runtime_components(
        plan,
        repository_root=root,
    )
    refiltered, replay_filter = filter_sft_records(
        records,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=None,
    )
    if len(refiltered) != len(records) or replay_filter["rejected"] != 0:
        raise ValueError("OpenCodeInstruct candidate no longer passes global filters")
    rededuped, _replay_ledger, replay_near = apply_conversation_near_dedup(
        records,
        near_duplicate_config=near_config,
    )
    if len(rededuped) != len(records) or replay_near["rejected"] != 0:
        raise ValueError("OpenCodeInstruct candidate contains a near duplicate")
    reselected, replay_tokens, replay_token_ledger = measure_and_select(
        records,
        maximum_train_supervised_tokens=plan["limits"][
            "maximum_retained_train_supervised_tokens"
        ],
        target_train_supervised_tokens=plan["limits"][
            "target_train_supervised_tokens"
        ],
        validation_fraction=plan["limits"]["validation_fraction"],
        template=template,
        tokenizer=tokenizer,
    )
    if {str(row["record_id"]) for row in reselected} != set(record_ids):
        raise ValueError("OpenCodeInstruct bounded token selection changed")
    if replay_tokens != token_metrics or replay_token_ledger != token_ledger:
        raise ValueError("OpenCodeInstruct frozen-tokenizer measurement changed")

    source_envelopes, source_metrics = load_source_envelopes(
        private_root / "raw" / SHARD_PATH,
        maximum_decoded_bytes=plan["limits"]["maximum_decoded_bytes"],
    )
    selected_envelopes = [
        envelope
        for envelope in source_envelopes
        if str(envelope["record"]["record_id"]) in set(record_ids)
    ]
    replay_verifier, replay_passed, _replay_reasons = run_local_unit_test_verifier(
        selected_envelopes,
        worker_path=root / plan["bindings"]["unit_test_worker"]["path"],
    )
    if replay_passed != set(record_ids) or any(
        row.get("verified") is not True for row in replay_verifier
    ):
        raise ValueError("OpenCodeInstruct local unit-test replay failed")
    if source_metrics["rows"] != report["source"]["source_rows"]:
        raise ValueError("OpenCodeInstruct source coverage changed")

    if len(owner_sample) != plan["limits"]["owner_sample_records"]:
        raise ValueError("OpenCodeInstruct owner sample count changed")
    if sha256_value(owner_sample) != report["owner_review"]["sample_sha256"]:
        raise ValueError("OpenCodeInstruct owner sample hash changed")
    if not {str(row["record_id"]) for row in owner_sample}.issubset(set(record_ids)):
        raise ValueError("OpenCodeInstruct owner sample escaped candidate")
    owner_html = paths["owner_review_html"].read_text(encoding="utf-8")
    if "<script src=" in owner_html or "<link rel=" in owner_html:
        raise ValueError("OpenCodeInstruct owner review gained external dependencies")
    if "lumi.opencodeinstruct-owner-decisions.v0.1" not in owner_html:
        raise ValueError("OpenCodeInstruct owner decision schema is missing")

    present = sorted(REQUIRED_CODING_BENCHMARKS & benchmark.benchmarks)
    missing = sorted(REQUIRED_CODING_BENCHMARKS - benchmark.benchmarks)
    if present != report["benchmark_decontamination"]["present_required_benchmarks"]:
        raise ValueError("OpenCodeInstruct benchmark coverage changed")
    if missing != report["benchmark_decontamination"]["missing_required_benchmarks"]:
        raise ValueError("OpenCodeInstruct missing benchmark coverage changed")
    if not missing:
        raise ValueError("expected unresolved LiveCodeBench/BigCodeBench coverage disappeared")

    train_tokens = token_metrics["by_split"]["train"]["supervised_tokens"]
    if not (
        plan["limits"]["target_train_supervised_tokens"]
        <= train_tokens
        <= plan["limits"]["maximum_retained_train_supervised_tokens"]
    ):
        raise ValueError("OpenCodeInstruct train token boundary changed")
    verification = {
        "checks": {
            "artifact_hashes_verified": True,
            "assistant_only_loss_and_4k_fit_replayed": True,
            "benchmark_registry_replayed": True,
            "candidate_records_locally_reexecuted": len(records),
            "exact_authorized_raw_inventory_verified": True,
            "global_filters_replayed": True,
            "near_duplicate_replay_rejected": replay_near["rejected"],
            "owner_sample_records": len(owner_sample),
            "source_registry_check_deferred_until_promotion": True,
        },
        "missing_required_benchmark_registries": missing,
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": report["report_sha256"],
        "retained_records": len(records),
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_nonproduction_owner_review_and_benchmark_coverage_pending",
        "train_supervised_tokens": train_tokens,
        "validation_supervised_tokens": token_metrics["by_split"]["validation"][
            "supervised_tokens"
        ],
    }
    verification["verification_sha256"] = sha256_value(verification)
    write_json_atomic(verification_path, verification)
    print(
        "PASS: independently verified OpenCodeInstruct curation "
        f"records={len(records)} train_supervised_tokens={train_tokens} "
        f"owner_sample={len(owner_sample)} production_eligible=False "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
