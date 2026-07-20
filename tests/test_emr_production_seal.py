from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lumi.emr_production_dedup import dedup_job_token
from lumi.emr_production_seal import (
    build_final_corpus_seal,
    final_corpus_seal_fingerprint,
    write_final_corpus_seal,
)
from lumi.emr_production_shard_writer import (
    build_shard_writer_job_request,
    dedup_result_fingerprint,
    shard_writer_job_token,
    validate_global_dedup_result,
)
from lumi.emr_serverless_production import (
    PRODUCTION_RUN_APPROVAL_TEXT,
    build_emr_production_plan,
    initial_production_checkpoint,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "configs/emr-production-resume-v0.1.json"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _plan() -> dict:
    return build_emr_production_plan(
        CONTRACT,
        repository_root=ROOT,
        source_commit="9" * 40,
    )


def _authorization(plan: dict, checkpoint: dict) -> dict:
    return {
        "approval_text": PRODUCTION_RUN_APPROVAL_TEXT,
        "authorization": {
            "full_4b_training": False,
            "model_training": False,
            "paid_cpu_stager": True,
            "paid_emr_production": True,
            "paid_gpu_compute": False,
            "production_adaptive_acquisition": True,
        },
        "bounds": {
            "maximum_compute_usd": "1200.00",
            "maximum_cumulative_emr_hours": "96",
            "maximum_emr_vcpu": 128,
            "maximum_job_runtime_hours": "24",
            "maximum_source_files": len(plan["source_schedule"]),
            "maximum_total_usd": "2000.00",
        },
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "gate": "lumi_emr_production_authorization_v0.1",
        "owner": "Afnan Technologies Inc.",
        "plan_fingerprint": plan["plan_fingerprint"],
        "run_token": "3" * 32,
        "schema_version": 1,
    }


def _dedup_result(plan: dict, authorization: dict) -> dict:
    categories = []
    total_documents = 0
    total_tokens = 0
    for row in plan["categories"]:
        selected_tokens = int(row["floor_tokens"]) - int(row["accepted_tokens"])
        carried = (
            int(plan["resume"]["partial_prefix"]["accepted_tokens"])
            if row["category"] == "science_and_education"
            else 0
        )
        selected_documents = (selected_tokens + carried + 99_999_999) // 100_000_000
        categories.append(
            {
                "available_documents": selected_documents + 10,
                "available_tokens": selected_tokens + 500_000_000,
                "category": row["category"],
                "deficit_tokens": 0,
                "final_documents": int(row["accepted_documents"])
                + selected_documents,
                "final_tokens": int(row["accepted_tokens"]) + selected_tokens,
                "floor_tokens": int(row["floor_tokens"]),
                "selected_documents": selected_documents,
                "selected_tokens": selected_tokens,
            }
        )
        total_documents += selected_documents
        total_tokens += selected_tokens
    map_sha = "a" * 64
    token = dedup_job_token(authorization, {"checkpoint_sha256": map_sha})
    namespace = plan["storage"]["namespace"]
    bucket = plan["storage"]["bucket"]
    return {
        "accepted_candidates": {
            "documents": total_documents,
            "object_bytes": 400_000_000_000,
            "object_count": 512,
            "objects_sha256": "b" * 64,
            "s3_uri": f"s3://{bucket}/{namespace}/deduplicated/{token}/accepted",
            "tokens": total_tokens,
        },
        "authorization_run_token": authorization["run_token"],
        "categories": categories,
        "deduplication": {"accepted_documents": total_documents},
        "gate": "lumi_emr_production_global_dedup_result_v0.1",
        "job_token": token,
        "map_checkpoint_sha256": map_sha,
        "plan_fingerprint": plan["plan_fingerprint"],
        "rejected_ids": {
            "object_bytes": 1_000_000,
            "object_count": 2,
            "objects_sha256": "c" * 64,
            "s3_uri": f"s3://{bucket}/{namespace}/deduplicated/{token}/rejected",
        },
        "runtime": {"python": "3.12"},
        "schema_version": 1,
        "status": "pass_category_floors",
        "total_deficit_tokens": 0,
    }


def _writer_results(
    plan: dict, checkpoint: dict, authorization: dict, dedup: dict
) -> list[dict]:
    rows = []
    namespace = plan["storage"]["namespace"]
    partial = checkpoint["partial_shard"]
    for category_row in dedup["categories"]:
        category = category_row["category"]
        carried_tokens = int(partial["tokens"]) if partial["category"] == category else 0
        carried_documents = (
            int(partial["documents"]) if partial["category"] == category else 0
        )
        physical_tokens = int(category_row["selected_tokens"]) + carried_tokens
        shard_tokens = []
        while physical_tokens:
            amount = min(100_000_000, physical_tokens)
            shard_tokens.append(amount)
            physical_tokens -= amount
        start_sequence = int(partial["sequence"]) if carried_tokens else 0
        shards = []
        for offset, tokens in enumerate(shard_tokens):
            sequence = start_sequence + offset
            label = f"{category}:{sequence}"
            documents = 1 + (carried_documents if offset == 0 else 0)
            clean_sha = _digest(f"clean:{label}")
            token_sha = _digest(f"token:{label}")
            provenance_sha = _digest(f"provenance:{label}")
            manifest_sha = _digest(f"manifest:{label}")
            shards.append(
                {
                    "accepted_text_utf8_bytes": tokens * 4,
                    "category": category,
                    "documents": documents,
                    "manifest_object_key": (
                        f"{namespace}/control/manifests/{manifest_sha}.json"
                    ),
                    "manifest_sha256": manifest_sha,
                    "object_bytes": tokens,
                    "object_key": f"{namespace}/staging/clean/{category}/{clean_sha}.zst",
                    "object_sha256": clean_sha,
                    "provenance_index_bytes": 128,
                    "provenance_index_key": (
                        f"{namespace}/control/provenance/{provenance_sha}.parquet"
                    ),
                    "provenance_index_sha256": provenance_sha,
                    "sequence": sequence,
                    "shard_id": f"{category}-{sequence:06d}",
                    "source_manifest_sha256": (
                        plan["inputs"]["source_file_manifest"]["sha256"]
                    ),
                    "token_object_bytes": tokens * 2,
                    "token_object_key": (
                        f"{namespace}/staging/tokenized/{category}/{token_sha}.bin"
                    ),
                    "token_object_sha256": token_sha,
                    "tokens": tokens,
                }
            )
        rows.append(
            {
                "authorization_run_token": authorization["run_token"],
                "carried_partial_documents": carried_documents,
                "carried_partial_tokens": carried_tokens,
                "category": category,
                "dedup_result_sha256": dedup_result_fingerprint(dedup),
                "gate": "lumi_emr_production_shard_writer_result_v0.1",
                "job_token": shard_writer_job_token(authorization, dedup, category),
                "next_sequence": start_sequence + len(shards),
                "plan_fingerprint": plan["plan_fingerprint"],
                "schema_version": 1,
                "selected_documents": len(shards),
                "selected_tokens": category_row["selected_tokens"],
                "shards": shards,
                "status": "complete",
            }
        )
    return rows


def _fixtures() -> tuple[dict, dict, dict, dict, list[dict]]:
    plan = _plan()
    checkpoint = initial_production_checkpoint(plan)
    authorization = _authorization(plan, checkpoint)
    dedup = _dedup_result(plan, authorization)
    writers = _writer_results(plan, checkpoint, authorization, dedup)
    return plan, checkpoint, authorization, dedup, writers


def test_shard_writer_request_uses_seven_bounded_16_vcpu_slots() -> None:
    plan, checkpoint, authorization, dedup, _ = _fixtures()
    category = plan["categories"][0]["category"]
    bucket = plan["storage"]["bucket"]
    prefix = f"s3://{bucket}/{plan['storage']['namespace']}/control/production"
    request = build_shard_writer_job_request(
        plan,
        checkpoint,
        authorization,
        dedup,
        category=category,
        application_id="00fixture",
        execution_role_arn="arn:aws:iam::123456789012:role/lumi-production",
        plan_s3_uri=f"{prefix}/plan.json",
        accepted_checkpoint_s3_uri=f"{prefix}/accepted.json",
        authorization_s3_uri=f"{prefix}/authorization.json",
        dedup_result_s3_uri=f"{prefix}/dedup.json",
        code_archive_s3_uri=f"{prefix}/code.zip",
        job_script_s3_uri=f"{prefix}/writer.py",
    )
    parameters = request["jobDriver"]["sparkSubmit"]["sparkSubmitParameters"]
    assert "spark.executor.instances=3" in parameters
    assert "spark.emr-serverless.driver.disk=20G" in parameters
    assert request["executionTimeoutMinutes"] == 720
    assert (
        plan["pipeline"]["maximum_concurrent_shard_writers"]
        * plan["pipeline"]["shard_writer_job_vcpu"]
        == 112
    )


def test_final_corpus_seal_conserves_120b_and_writes_hash_manifest(
    tmp_path: Path,
) -> None:
    plan, checkpoint, authorization, dedup, writers = _fixtures()
    assert validate_global_dedup_result(dedup, plan, authorization) == dedup
    seal = build_final_corpus_seal(
        plan, checkpoint, authorization, dedup, writers
    )
    assert seal["status"] == "sealed_ready_for_80b_sampling"
    assert seal["total_accepted_tokens"] == 120_000_000_000
    assert seal["candidate_hard_maximum_tokens"] == 150_000_000_000
    assert seal["final_training_target_tokens"] == 80_000_000_000
    assert seal["seal_sha256"] == final_corpus_seal_fingerprint(seal)
    written = write_final_corpus_seal(seal, tmp_path / "sealed")
    seal_path = Path(written["seal_path"])
    expected = hashlib.sha256(seal_path.read_bytes()).hexdigest()
    assert Path(written["manifest_path"]).read_text() == (
        f"{expected}  final-candidate-corpus.json\n"
    )


def test_final_corpus_seal_rejects_duplicate_manifest_and_dedup_token_drift() -> None:
    plan, checkpoint, authorization, dedup, writers = _fixtures()
    writers[1]["shards"][0]["manifest_sha256"] = writers[0]["shards"][0][
        "manifest_sha256"
    ]
    with pytest.raises(ValueError, match="repeats a shard manifest"):
        build_final_corpus_seal(plan, checkpoint, authorization, dedup, writers)
    changed = json.loads(json.dumps(dedup))
    changed["job_token"] = "f" * 32
    with pytest.raises(ValueError, match="job token changed"):
        validate_global_dedup_result(changed, plan, authorization)
