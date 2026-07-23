from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lumi.final_corpus import (
    assign_hash_ranked_token_shards,
    build_final_corpus_plan,
    final_selection_rank,
    final_selection_result_fingerprint,
    select_hash_ranked_token_prefix,
    validate_final_selection_result,
    validate_final_shard_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "final-corpus-80b-v0.1.json"
SOURCE_COMMIT = "a" * 40


def _plan() -> dict:
    return build_final_corpus_plan(
        CONTRACT,
        repository_root=ROOT,
        source_commit=SOURCE_COMMIT,
    )


def _selection_result(plan: dict) -> dict:
    categories = []
    for index, row in enumerate(plan["categories"], start=1):
        selected_tokens = int(row["requested_candidate_tokens"]) + 1
        selected_documents = min(1000, int(row["available_candidate_documents"]))
        categories.append(
            {
                "available_candidate_documents": row["available_candidate_documents"],
                "available_candidate_content_tokens": row[
                    "available_candidate_content_tokens"
                ],
                "available_candidate_tokens": row["available_candidate_tokens"],
                "category": row["category"],
                "cutoff": {
                    "_lumi_selection_rank": f"{index + 20:064x}",
                    "content_sha256": f"{index + 30:064x}",
                    "document_id": f"document-{index}",
                    "segment_index": 0,
                    "source_line": index,
                    "source_ordinal": index - 1,
                },
                "final_documents": int(row["baseline_documents"]) + selected_documents,
                "final_content_tokens": selected_tokens - selected_documents,
                "final_tokens": int(row["baseline_tokens"]) + selected_tokens,
                "selected_candidate_documents": selected_documents,
                "selected_candidate_content_tokens": selected_tokens
                - selected_documents,
                "selected_candidate_tokens": selected_tokens,
                "selected_inventory": {
                    "object_bytes": 1000 + index,
                    "object_count": 1,
                    "objects_sha256": f"{index:064x}",
                    "s3_uri": (
                        f"s3://{plan['output']['bucket']}/{plan['output']['namespace']}/"
                        f"selected/{row['category']}"
                    ),
                },
            }
        )
    result = {
        "categories": categories,
        "dedup_result_sha256": plan["inputs"]["dedup_result"]["canonical_sha256"],
        "final_documents": sum(int(row["final_documents"]) for row in categories),
        "final_content_tokens": sum(
            int(row["final_content_tokens"]) for row in categories
        ),
        "final_tokens": sum(int(row["final_tokens"]) for row in categories),
        "gate": "lumi_final_corpus_80b_selection_result_v0.1",
        "plan_fingerprint": plan["plan_fingerprint"],
        "schema_version": 1,
        "selection_seed_sha256": plan["selection"]["seed_sha256"],
        "status": "complete",
        "training_token_limit": plan["limits"]["training_token_limit"],
    }
    result["result_sha256"] = final_selection_result_fingerprint(result)
    return result


def _split_total(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def _shard_manifest(plan: dict, result: dict) -> dict:
    result_by_category = {row["category"]: row for row in result["categories"]}
    shards = []
    sequence = 0
    for category_plan in plan["categories"]:
        category = category_plan["category"]
        shard_count = int(category_plan["new_packed_shards"])
        if int(category_plan["baseline_tokens"]):
            shard_count += int(plan["output"]["preserved_baseline_shards"])
        category_result = result_by_category[category]
        token_parts = _split_total(int(category_result["final_tokens"]), shard_count)
        document_parts = _split_total(
            int(category_result["final_documents"]), shard_count
        )
        for tokens, documents in zip(token_parts, document_parts, strict=True):
            shards.append(
                {
                    "category": category,
                    "content_tokens": tokens - documents,
                    "document_markers": documents,
                    "documents": documents,
                    "object_bytes": tokens * 2,
                    "object_sha256": f"{sequence + 1:064x}",
                    "s3_uri": (
                        f"s3://{plan['output']['bucket']}/{plan['output']['namespace']}/"
                        f"packed/shard-{sequence:06d}.bin"
                    ),
                    "sequence": sequence,
                    "tokens": tokens,
                }
            )
            sequence += 1
    return {
        "content_tokens": result["final_content_tokens"],
        "document_markers": result["final_documents"],
        "documents": result["final_documents"],
        "gate": "lumi_final_corpus_80b_shard_manifest_v0.1",
        "packing": {
            "doc_token": "<|doc|>",
            "doc_token_id": 3,
            "markers_per_document": 1,
            "token_accounting": "content_plus_one_document_marker",
        },
        "plan_fingerprint": plan["plan_fingerprint"],
        "schema_version": 1,
        "selection_result_sha256": result["result_sha256"],
        "shards": shards,
        "status": "sealed",
        "tokens": result["final_tokens"],
        "training_token_limit": plan["limits"]["training_token_limit"],
    }


def test_verified_plan_binds_120b_pool_to_exact_80b_mix() -> None:
    plan = _plan()
    assert plan["status"] == "ready_for_bounded_glue_canary_approval"
    assert plan["limits"]["final_training_tokens"] == 80_000_000_000
    assert plan["limits"]["maximum_final_selected_tokens"] == 80_000_917_504
    assert plan["inputs"]["dedup_result"]["canonical_sha256"] == (
        "dbfbf9fe258ec97d5bd844c165157d41d328d0571cbbada5fd4d67750c46b742"
    )
    assert sum(row["target_final_tokens"] for row in plan["categories"]) == 80_000_000_000
    assert sum(row["baseline_tokens"] for row in plan["categories"]) == 0
    science = plan["categories"][0]
    assert science["requested_candidate_tokens"] == 24_000_000_000
    assert plan["output"]["expected_new_packed_shards"] == 160
    assert plan["output"]["expected_total_packed_shards"] == 160
    assert plan["packing"]["doc_token_id"] == 3
    assert all(value is False for value in plan["authorization"].values())


def test_plan_fails_closed_when_authorization_or_input_pin_changes(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text())
    contract["authorization"]["paid_glue_selection"] = True
    changed = tmp_path / "authorization.json"
    changed.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="must not authorize"):
        build_final_corpus_plan(
            changed,
            repository_root=ROOT,
            source_commit=SOURCE_COMMIT,
        )

    contract = json.loads(CONTRACT.read_text())
    contract["inputs"]["dedup_result"]["canonical_sha256"] = "f" * 64
    changed = tmp_path / "dedup.json"
    changed.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="canonical content changed"):
        build_final_corpus_plan(
            changed,
            repository_root=ROOT,
            source_commit=SOURCE_COMMIT,
        )


def test_cross_runtime_selection_rank_is_stable() -> None:
    seed = "a84e427156a2f88998524ff746e3ae31d6d150cbc5aad1ab019f8bf56da4227a"
    rank = final_selection_rank(
        seed,
        category="code",
        component="stackv2_edu",
        document_id="doc-1",
        content_sha256="b" * 64,
    )
    assert rank == "002ebef8385bbbc5a725dd92e1d4cf392a5466d2ab101e28f44727f3eb9cc73f"


def test_selection_result_and_shard_manifest_fail_closed() -> None:
    plan = _plan()
    result = _selection_result(plan)
    assert validate_final_selection_result(result, plan)["final_tokens"] == 80_000_000_007

    too_many = copy.deepcopy(result)
    too_many["categories"][0]["selected_candidate_tokens"] = (
        plan["categories"][0]["maximum_selected_candidate_tokens"] + 1
    )
    too_many["categories"][0]["selected_candidate_content_tokens"] = (
        too_many["categories"][0]["selected_candidate_tokens"]
        - too_many["categories"][0]["selected_candidate_documents"]
    )
    too_many["result_sha256"] = final_selection_result_fingerprint(too_many)
    with pytest.raises(ValueError, match="overshoot cap"):
        validate_final_selection_result(too_many, plan)

    manifest = _shard_manifest(plan, result)
    assert len(manifest["shards"]) == 160
    assert validate_final_shard_manifest(manifest, plan, result)["tokens"] == (
        result["final_tokens"]
    )
    bad_bytes = copy.deepcopy(manifest)
    bad_bytes["shards"][0]["object_bytes"] += 1
    with pytest.raises(ValueError, match="uint16 byte conservation"):
        validate_final_shard_manifest(bad_bytes, plan, result)
    duplicate = copy.deepcopy(manifest)
    duplicate["shards"][1]["s3_uri"] = duplicate["shards"][0]["s3_uri"]
    with pytest.raises(ValueError, match="repeats an object"):
        validate_final_shard_manifest(duplicate, plan, result)


def test_local_spark_hash_selection_matches_python_rank() -> None:
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("lumi-final-corpus-selection-test")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    seed = "a84e427156a2f88998524ff746e3ae31d6d150cbc5aad1ab019f8bf56da4227a"
    try:
        rows = []
        for index, tokens in enumerate((3, 4, 5, 2, 6, 1)):
            document_id = f"doc-{index}"
            content_sha256 = hashlib.sha256(document_id.encode()).hexdigest()
            rows.append(
                {
                    "category": "code",
                    "component": "fixture",
                    "content_sha256": content_sha256,
                    "document_id": document_id,
                    "segment_index": 0,
                    "source_line": index,
                    "source_ordinal": 0,
                    "token_count": tokens,
                }
            )
        expected = sorted(
            rows,
            key=lambda row: (
                final_selection_rank(
                    seed,
                    category=row["category"],
                    component=row["component"],
                    document_id=row["document_id"],
                    content_sha256=row["content_sha256"],
                ),
                row["content_sha256"],
                row["document_id"],
                row["source_ordinal"],
                row["source_line"],
                row["segment_index"],
            ),
        )
        expected_ids = []
        expected_tokens = 0
        for row in expected:
            if expected_tokens >= 10:
                break
            expected_ids.append(row["document_id"])
            expected_tokens += row["token_count"]
        frame = spark.createDataFrame(rows)
        observed = []
        for partitions in (2, 3):
            selection = select_hash_ranked_token_prefix(
                frame,
                seed_sha256=seed,
                target_tokens=10,
                maximum_overshoot_tokens=6,
                partitions=partitions,
            )
            selected_rows = selection["selected"].orderBy(
                "_lumi_selection_rank",
                "content_sha256",
                "document_id",
                "source_ordinal",
                "source_line",
                "segment_index",
            ).collect()
            observed.append([row["document_id"] for row in selected_rows])
            assert selection["selected_tokens"] == expected_tokens
            shards = assign_hash_ranked_token_shards(
                selection["selected"],
                target_tokens=6,
                maximum_overshoot_tokens=6,
                start_sequence=3,
                partitions=partitions,
            )
            shard_rows = shards["assigned"].orderBy(
                "_lumi_selection_rank",
                "content_sha256",
                "document_id",
                "source_ordinal",
                "source_line",
                "segment_index",
            ).collect()
            expected_sequences = []
            running = 0
            for row in selected_rows:
                expected_sequences.append(3 + running // 6)
                running += int(row["token_count"])
            assert [row["shard_sequence"] for row in shard_rows] == expected_sequences
            assert shards["tokens"] == expected_tokens
            shards["assigned"].unpersist()
            selection["selected"].unpersist()
        assert observed == [expected_ids, expected_ids]
    finally:
        spark.stop()
