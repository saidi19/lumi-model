from __future__ import annotations

import hashlib
import json
import runpy
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lumi.emr_serverless_production import build_emr_production_plan
from lumi.glue_production_dedup import (
    INPUT_INDEX_FILE,
    MAP_CHECKPOINT_FILE,
    MAP_RELEASE_FILE,
    RELEASE_FILE,
    build_glue_global_dedup_authorization,
    build_glue_global_dedup_input_index,
    build_glue_global_dedup_job_definition,
    build_glue_production_global_dedup_package,
    expected_glue_global_dedup_approval,
    glue_global_dedup_job_token,
    validate_glue_global_dedup_authorization,
    validate_glue_global_dedup_input_index,
    validate_glue_global_dedup_result,
    validate_glue_production_global_dedup_release,
)
from lumi.glue_production_mapping import (
    advance_mapping_checkpoint,
    build_initial_mapping_checkpoint,
    plan_mapping_wave,
    validate_glue_production_mapping_release,
)
from lumi.production_corpus import _canonical_json_bytes

ROOT = Path(__file__).parents[1]
MAPPING_HELPERS = runpy.run_path(
    ROOT / "tests" / "test_glue_production_mapping.py",
    run_name="lumi_glue_mapping_fixture_helpers",
)


def _refingerprint(value: dict[str, object], field: str) -> None:
    material = dict(value)
    material.pop(field, None)
    value[field] = hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _final_map_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, list[Path], dict[str, object], dict[str, object]]:
    release = MAPPING_HELPERS["_release"]()
    commit = "b" * 40
    plan = build_emr_production_plan(
        ROOT / "configs" / "emr-production-resume-v0.1.json",
        repository_root=ROOT,
        source_commit=commit,
    )
    bucket = plan["storage"]["bucket"]
    namespace = plan["storage"]["namespace"]
    release["source_commit"] = commit
    release["storage"] = {"bucket": bucket, "namespace": namespace}
    release["data_quality"] = plan["data_quality"]
    release["filters"] = plan["filters"]
    release["output"] = {
        "checkpoint_prefix": f"{namespace}/control/test-map-checkpoints",
        "mapped_prefix": f"{namespace}/transient/test-mapped",
        "result_prefix": f"{namespace}/control/test-map-runs",
        "temp_prefix": f"{namespace}/control/test-map-temp",
    }
    counts = {str(row["category"]): 0 for row in release["categories"]}
    for row in release["staged_sources"]:
        row["chunk_count"] = 20
        counts[str(row["category"])] += 20
    release["category_inventory"] = [
        {"category": row["category"], "staged_chunks": counts[str(row["category"])]}
        for row in release["categories"]
    ]
    release["stage"]["staged_chunks"] = sum(counts.values())
    _refingerprint(release, "release_fingerprint")
    validate_glue_production_mapping_release(release)

    checkpoint = build_initial_mapping_checkpoint(release)
    manifests: list[dict[str, object]] = []
    for wave_sequence in range(1, 13):
        wave = plan_mapping_wave(checkpoint, release)
        close = wave_sequence == 12
        results = []
        for row in wave["categories"]:
            name = str(row["category"])
            selected_tokens = (
                int(row["remaining_target_tokens"]) - 1 if close else 1
            )
            digest = hashlib.sha256(f"{wave_sequence}:{name}".encode()).hexdigest()
            results.append(
                {
                    "available_documents": 1,
                    "available_tokens": selected_tokens,
                    "candidate_documents": 1,
                    "candidate_tokens": selected_tokens,
                    "category": name,
                    "closed": close,
                    "cutoff": {"document_id": f"{wave_sequence}:{name}"}
                    if close
                    else None,
                    "end_chunk_index": row["end_chunk_index"],
                    "object_count": 1,
                    "objects_sha256": digest,
                    "parquet_bytes": 10,
                    "parquet_s3_uri": (
                        f"s3://{bucket}/{release['output']['mapped_prefix']}/"
                        f"{release['release_fingerprint']}/fixture/wave-{wave_sequence:06d}/"
                        f"{name}"
                    ),
                    "start_chunk_index": row["start_chunk_index"],
                }
            )
        manifest: dict[str, object] = {
            "candidate_documents": len(results),
            "candidate_tokens": sum(int(row["candidate_tokens"]) for row in results),
            "categories": results,
            "gate": "lumi_glue_production_map_wave_result_v0.1",
            "manifest_s3_key": (
                f"{release['output']['result_prefix']}/{release['release_fingerprint']}/"
                f"fixture/waves/wave-{wave_sequence:06d}/manifest.json"
            ),
            "parent_checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "parquet_bytes": sum(int(row["parquet_bytes"]) for row in results),
            "release_fingerprint": release["release_fingerprint"],
            "schema_version": 1,
            "wave_plan_sha256": wave["wave_sha256"],
            "wave_sequence": wave_sequence,
        }
        _refingerprint(manifest, "manifest_sha256")
        checkpoint = advance_mapping_checkpoint(checkpoint, release, manifest)
        manifests.append(manifest)
    assert checkpoint["status"] == "ready_for_global_dedup"
    assert checkpoint["checkpoint_sequence"] == 12

    release_path = tmp_path / "map-release.json"
    checkpoint_path = tmp_path / "map-checkpoint.json"
    release_path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n"
    )
    manifest_paths = []
    for manifest in manifests:
        path = tmp_path / f"wave-{int(manifest['wave_sequence']):02d}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifest_paths.append(path)
    return release_path, checkpoint_path, manifest_paths, release, checkpoint


def _package_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], Path, dict[str, object], dict[str, object]]:
    release_path, checkpoint_path, manifests, map_release, checkpoint = (
        _final_map_fixture(tmp_path)
    )
    package = tmp_path / "dedup-package"
    release = build_glue_production_global_dedup_package(
        contract_path=ROOT / "configs" / "glue-production-global-dedup-v0.1.json",
        repository_root=ROOT,
        source_commit="b" * 40,
        map_release_path=release_path,
        map_checkpoint_path=checkpoint_path,
        wave_manifest_paths=manifests,
        output_directory=package,
    )
    return release, package, map_release, checkpoint


def test_input_index_binds_all_waves_and_conserves_the_150b_map(tmp_path: Path) -> None:
    release_path, checkpoint_path, manifests, map_release, checkpoint = (
        _final_map_fixture(tmp_path)
    )
    index = build_glue_global_dedup_input_index(
        map_release_path=release_path,
        map_checkpoint_path=checkpoint_path,
        wave_manifest_paths=manifests,
    )
    assert len(index["waves"]) == 12
    assert len(index["inputs"]) > 12
    assert index["mapped_tokens"] == checkpoint["total_mapped_tokens"]
    assert index["mapped_documents"] == checkpoint["total_mapped_documents"]
    assert index["mapped_parquet_bytes"] == checkpoint["mapped_parquet_bytes"]
    assert index["total_candidate_tokens"] == 150_000_000_000 - 7
    assert validate_glue_global_dedup_input_index(
        index, map_release, checkpoint
    ) == index


def test_input_index_refuses_missing_reordered_and_corrupt_manifests(
    tmp_path: Path,
) -> None:
    release_path, checkpoint_path, manifests, _, _ = _final_map_fixture(tmp_path)
    with pytest.raises(ValueError, match="count changed"):
        build_glue_global_dedup_input_index(
            map_release_path=release_path,
            map_checkpoint_path=checkpoint_path,
            wave_manifest_paths=manifests[:-1],
        )
    with pytest.raises(ValueError, match="order changed"):
        build_glue_global_dedup_input_index(
            map_release_path=release_path,
            map_checkpoint_path=checkpoint_path,
            wave_manifest_paths=list(reversed(manifests)),
        )
    changed = json.loads(manifests[0].read_text())
    changed["candidate_tokens"] += 1
    corrupt = tmp_path / "corrupt-wave.json"
    corrupt.write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="fingerprint changed"):
        build_glue_global_dedup_input_index(
            map_release_path=release_path,
            map_checkpoint_path=checkpoint_path,
            wave_manifest_paths=[corrupt, *manifests[1:]],
        )


def test_input_index_refuses_namespace_escape_and_duplicate_objects(
    tmp_path: Path,
) -> None:
    release_path, checkpoint_path, manifests, map_release, checkpoint = (
        _final_map_fixture(tmp_path)
    )
    index = build_glue_global_dedup_input_index(
        map_release_path=release_path,
        map_checkpoint_path=checkpoint_path,
        wave_manifest_paths=manifests,
    )
    escaped = deepcopy(index)
    escaped["inputs"][0]["parquet_s3_uri"] = "s3://attacker/outside.parquet"
    _refingerprint(escaped, "input_index_sha256")
    with pytest.raises(ValueError, match="mapped namespace"):
        validate_glue_global_dedup_input_index(escaped, map_release, checkpoint)

    duplicated = deepcopy(index)
    duplicated["inputs"][1]["parquet_s3_uri"] = duplicated["inputs"][0][
        "parquet_s3_uri"
    ]
    _refingerprint(duplicated, "input_index_sha256")
    with pytest.raises(ValueError, match="repeats a Parquet input"):
        validate_glue_global_dedup_input_index(duplicated, map_release, checkpoint)

    repeated_inventory = deepcopy(index)
    repeated_inventory["inputs"][1]["objects_sha256"] = repeated_inventory["inputs"][
        0
    ]["objects_sha256"]
    _refingerprint(repeated_inventory, "input_index_sha256")
    with pytest.raises(ValueError, match="object inventory"):
        validate_glue_global_dedup_input_index(
            repeated_inventory, map_release, checkpoint
        )


def test_package_is_hash_bound_and_does_not_authorize_paid_work(tmp_path: Path) -> None:
    release, package, _, _ = _package_fixture(tmp_path)
    verified = validate_glue_production_global_dedup_release(release)
    assert verified["status"] == "verified_not_authorized"
    assert release["authorization"]["paid_glue_dedup"] is False
    assert release["decision"]["full_4b_training"] == "not_authorized"
    assert release["map"]["wave_count"] == 12
    assert release["jobs"]["canary"]["number_of_workers"] == 8
    assert release["jobs"]["production"]["number_of_workers"] == 100
    assert release["jobs"]["production"]["maximum_compute_usd"] == "1056.000000"
    assert (package / RELEASE_FILE).is_file()
    assert (package / INPUT_INDEX_FILE).is_file()
    assert (package / MAP_RELEASE_FILE).is_file()
    assert (package / MAP_CHECKPOINT_FILE).is_file()
    lines = (package / "manifest.sha256").read_text().splitlines()
    assert len(lines) == 18
    assert any("wave-manifests/wave-000012.json" in line for line in lines)


def test_canary_and_production_authorizations_are_separate_and_bounded(
    tmp_path: Path,
) -> None:
    release, _, _, _ = _package_fixture(tmp_path)
    now = datetime.now(UTC)
    with pytest.raises(ValueError, match="does not match"):
        build_glue_global_dedup_authorization(
            release,
            mode="canary",
            approval_text="approve it",
            now=now,
        )
    canary = build_glue_global_dedup_authorization(
        release,
        mode="canary",
        approval_text=expected_glue_global_dedup_approval(release, mode="canary"),
        now=now,
    )
    production = build_glue_global_dedup_authorization(
        release,
        mode="production",
        approval_text=expected_glue_global_dedup_approval(
            release, mode="production"
        ),
        now=now,
    )
    assert canary["authorization_sha256"] != production["authorization_sha256"]
    assert canary["bounds"]["hard_maximum_dpu"] == 8
    assert canary["bounds"]["maximum_usd"] == "1.760000"
    assert canary["permissions"]["production_global_dedup"] is False
    assert production["bounds"]["hard_maximum_dpu"] == 100
    assert production["bounds"]["maximum_usd"] == "1056.000000"
    assert production["permissions"]["production_global_dedup"] is True
    assert production["permissions"]["model_training"] is False
    assert validate_glue_global_dedup_authorization(
        canary, release, now=now
    )["status"] == "authorized_one_canary_global_dedup_run"
    with pytest.raises(ValueError, match="not currently valid"):
        validate_glue_global_dedup_authorization(
            canary, release, now=now + timedelta(hours=25)
        )

    bucket = release["storage"]["bucket"]
    namespace = release["storage"]["namespace"]
    canary_job = build_glue_global_dedup_job_definition(
        release,
        canary,
        role_arn="arn:aws:iam::123456789012:role/lumi-glue-dedup",
        release_s3_uri=f"s3://{bucket}/{namespace}/control/dedup/release.json",
        authorization_s3_uri=f"s3://{bucket}/{namespace}/control/dedup/canary-auth.json",
    )
    production_job = build_glue_global_dedup_job_definition(
        release,
        production,
        role_arn="arn:aws:iam::123456789012:role/lumi-glue-dedup",
        release_s3_uri=f"s3://{bucket}/{namespace}/control/dedup/release.json",
        authorization_s3_uri=(
            f"s3://{bucket}/{namespace}/control/dedup/production-auth.json"
        ),
    )
    assert (canary_job["NumberOfWorkers"], canary_job["Timeout"]) == (8, 30)
    assert (production_job["NumberOfWorkers"], production_job["Timeout"]) == (
        100,
        1_440,
    )
    assert canary_job["MaxRetries"] == production_job["MaxRetries"] == 0
    assert "model" not in canary_job["Description"].lower().split(";", 1)[0]


def test_runtime_reuses_the_proven_dedup_core_and_initializes_spark_first() -> None:
    source = (ROOT / "scripts" / "run_glue_production_global_dedup.py").read_text()
    assert "spark_deduplicate_candidates" in source
    assert "maximum_candidate_pairs" in source
    assert "errorifexists" in source
    context = source.index("context = SparkContext.getOrCreate()")
    release = source.index('SparkFiles.get(args["LUMI_RELEASE_FILE"])')
    assert context < release
    namespace = runpy.run_path(
        ROOT / "scripts" / "run_glue_production_global_dedup.py",
        run_name="lumi_glue_global_dedup_script_test",
    )
    assert namespace["GLUE_REQUESTED_OPTIONS"] == (
        "JOB_NAME",
        "LUMI_AUTHORIZATION_FILE",
        "LUMI_INPUT_INDEX_FILE",
        "LUMI_MAP_CHECKPOINT_FILE",
        "LUMI_MAP_RELEASE_FILE",
        "LUMI_RELEASE_FILE",
    )


def test_runtime_accepts_the_pinned_aws_spark_build_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = runpy.run_path(
        ROOT / "scripts" / "run_glue_production_global_dedup.py",
        run_name="lumi_glue_global_dedup_runtime_test",
    )
    monkeypatch.setattr(
        namespace["platform"], "python_version_tuple", lambda: ("3", "11", "9")
    )
    monkeypatch.setattr(namespace["platform"], "python_version", lambda: "3.11.9")
    monkeypatch.setattr(
        namespace["importlib"].metadata,
        "version",
        lambda package: "2.0.0" if package == "datasketch" else "unexpected",
    )

    class Spark:
        def __init__(self, version: str) -> None:
            self.version = version

    runtime = namespace["_runtime"](
        Spark("3.5.6-amzn-1"), expected_spark_version="3.5.6"
    )
    assert runtime["spark"] == "3.5.6-amzn-1"
    assert runtime["spark_base"] == "3.5.6"

    with pytest.raises(RuntimeError, match="Spark changed"):
        namespace["_runtime"](
            Spark("3.5.7-amzn-1"), expected_spark_version="3.5.6"
        )
    with pytest.raises(RuntimeError, match="Spark changed"):
        namespace["_runtime"](
            Spark("3.5.6-other-1"), expected_spark_version="3.5.6"
        )


def test_result_contract_separates_canary_from_production(tmp_path: Path) -> None:
    release, _, _, _ = _package_fixture(tmp_path)
    now = datetime.now(UTC)
    canary_auth = build_glue_global_dedup_authorization(
        release,
        mode="canary",
        approval_text=expected_glue_global_dedup_approval(release, mode="canary"),
        now=now,
    )
    canary_token = glue_global_dedup_job_token(canary_auth)
    canary_prefix = f"{release['output']['canary_prefix']}/{canary_token}"
    inventory = {
        "object_bytes": 10,
        "object_count": 1,
        "objects_sha256": "a" * 64,
    }
    canary = {
        "accepted_ids": {
            **inventory,
            "s3_uri": (
                f"s3://{release['storage']['bucket']}/{canary_prefix}/accepted-ids"
            ),
        },
        "authorization_sha256": canary_auth["authorization_sha256"],
        "deduplication": {
            "accepted_documents": 100,
            "candidate_documents": 100,
            "exact_rejected_documents": 0,
            "near_rejected_documents": 0,
        },
        "gate": "lumi_glue_production_global_dedup_canary_result_v0.1",
        "input_documents": 100,
        "input_index_sha256": release["input_index_sha256"],
        "input_tokens": 1_000,
        "job_token": canary_token,
        "map_checkpoint_sha256": release["map"]["checkpoint_sha256"],
        "mode": "canary",
        "rejected_ids": {
            **{**inventory, "objects_sha256": "b" * 64},
            "s3_uri": (
                f"s3://{release['storage']['bucket']}/{canary_prefix}/rejected-ids"
            ),
        },
        "release_fingerprint": release["release_fingerprint"],
        "runtime": {},
        "schema_version": 1,
        "status": "pass_structural_canary",
    }
    assert validate_glue_global_dedup_result(
        canary, release, canary_auth
    ) == canary

    production_auth = build_glue_global_dedup_authorization(
        release,
        mode="production",
        approval_text=expected_glue_global_dedup_approval(
            release, mode="production"
        ),
        now=now,
    )
    production_token = glue_global_dedup_job_token(production_auth)
    production_prefix = (
        f"{release['output']['production_prefix']}/{production_token}"
    )
    categories = []
    selected_tokens = 0
    for row in release["categories"]:
        selected = int(row["floor_tokens"]) - int(row["accepted_tokens"])
        selected_tokens += selected
        categories.append(
            {
                "available_documents": 10,
                "available_tokens": selected,
                "category": row["category"],
                "deficit_tokens": 0,
                "final_documents": int(row["accepted_documents"]) + 1,
                "final_tokens": int(row["floor_tokens"]),
                "floor_tokens": int(row["floor_tokens"]),
                "selected_documents": 1,
                "selected_tokens": selected,
            }
        )
    production = {
        "accepted_candidates": {
            **inventory,
            "documents": 7,
            "s3_uri": (
                f"s3://{release['storage']['bucket']}/{production_prefix}/"
                "accepted-candidates"
            ),
            "tokens": selected_tokens,
        },
        "authorization_sha256": production_auth["authorization_sha256"],
        "categories": categories,
        "deduplication": {
            "accepted_documents": release["map"]["input_documents"],
            "candidate_documents": release["map"]["input_documents"],
            "exact_rejected_documents": 0,
            "near_rejected_documents": 0,
        },
        "gate": "lumi_glue_production_global_dedup_result_v0.1",
        "input_index_sha256": release["input_index_sha256"],
        "job_token": production_token,
        "map_checkpoint_sha256": release["map"]["checkpoint_sha256"],
        "mode": "production",
        "rejected_ids": {
            **{**inventory, "objects_sha256": "b" * 64},
            "s3_uri": (
                f"s3://{release['storage']['bucket']}/{production_prefix}/rejected-ids"
            ),
        },
        "release_fingerprint": release["release_fingerprint"],
        "runtime": {},
        "schema_version": 1,
        "status": "pass_category_floors",
        "total_deficit_tokens": 0,
    }
    assert validate_glue_global_dedup_result(
        production, release, production_auth
    ) == production
    changed = deepcopy(production)
    changed["categories"][0]["final_tokens"] -= 1
    with pytest.raises(ValueError, match="token conservation"):
        validate_glue_global_dedup_result(changed, release, production_auth)
