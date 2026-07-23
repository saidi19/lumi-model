from __future__ import annotations

import copy
import hashlib
import io
import json
import runpy
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lumi.final_corpus import (
    build_final_corpus_plan,
    final_selection_result_fingerprint,
)
from lumi.glue_final_corpus_production import (
    CATEGORY_CHECKPOINT_GATE,
    RESULT_GATE,
    _validate_canary_report,
    build_glue_final_corpus_production_authorization,
    build_glue_final_corpus_production_job_definition,
    build_glue_final_corpus_production_package,
    category_checkpoint_fingerprint,
    expected_glue_final_corpus_production_approval,
    glue_final_corpus_production_job_token,
    production_result_fingerprint,
    validate_glue_final_corpus_category_checkpoint,
    validate_glue_final_corpus_production_authorization,
    validate_glue_final_corpus_production_release,
    validate_glue_final_corpus_production_result,
)
from lumi.glue_runtime import FINAL_CORPUS_SPARK_CONFIG, glue_conf_argument

ROOT = Path(__file__).resolve().parents[1]
CANARY_SOURCE_COMMIT = "fad0b9d9340f58586ae9c5320fd66a2a5fe0c9e1"
SOURCE_COMMIT = "c" * 40


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, dict, Path]:
    directory = tmp_path_factory.mktemp("final-corpus-production")
    plan = build_final_corpus_plan(
        ROOT / "configs" / "final-corpus-80b-v0.1.json",
        repository_root=ROOT,
        source_commit=CANARY_SOURCE_COMMIT,
    )
    verified_canary = json.loads(
        (ROOT / "reports" / "glue-final-corpus-canary-result-v0.1.json").read_text()
    )
    if verified_canary.get("plan_fingerprint") != plan["plan_fingerprint"]:
        pytest.skip("marker-aware production packaging awaits a new verified canary")
    plan_path = directory / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    output = directory / "package"
    release = build_glue_final_corpus_production_package(
        contract_path=ROOT / "configs" / "glue-final-corpus-production-v0.1.json",
        final_plan_path=plan_path,
        repository_root=ROOT,
        source_commit=SOURCE_COMMIT,
        output_directory=output,
    )
    return release, plan, output


def _authorization(release: dict) -> dict:
    return build_glue_final_corpus_production_authorization(
        release,
        approval_text=expected_glue_final_corpus_production_approval(release),
        now=datetime.now(UTC),
    )


def _split_total(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def _checkpoint(release: dict, authorization: dict, category_index: int) -> dict:
    planned = release["categories"][category_index]
    category = str(planned["category"])
    shard_count = int(planned["new_packed_shards"])
    selected_tokens = int(planned["requested_candidate_tokens"]) + 1
    selected_documents = shard_count * 10
    selected_content_tokens = selected_tokens - selected_documents
    token_parts = _split_total(selected_tokens, shard_count)
    document_parts = _split_total(selected_documents, shard_count)
    token = glue_final_corpus_production_job_token(authorization)
    prefix = f"{release['output']['run_prefix']}/{token}"
    packed = []
    for offset, (tokens, documents) in enumerate(
        zip(token_parts, document_parts, strict=True)
    ):
        sequence = int(planned["first_new_shard_sequence"]) + offset
        digest = hashlib.sha256(f"{category}:{sequence}".encode()).hexdigest()
        packed.append(
            {
                "category": category,
                "content_tokens": tokens - documents,
                "document_markers": documents,
                "documents": documents,
                "object_bytes": tokens * 2,
                "object_sha256": digest,
                "s3_uri": (
                    f"s3://{release['storage']['bucket']}/{prefix}/packed/"
                    f"{category}/{digest}.uint16.bin"
                ),
                "sequence": sequence,
                "tokens": tokens,
            }
        )
    selected_bytes = 1_000_000 + category_index
    checkpoint = {
        "authorization_sha256": authorization["authorization_sha256"],
        "available_candidate_content_tokens": planned[
            "available_candidate_content_tokens"
        ],
        "available_candidate_documents": planned["available_candidate_documents"],
        "available_candidate_tokens": planned["available_candidate_tokens"],
        "category": category,
        "cutoff": {
            "_lumi_selection_rank": f"{category_index + 1:064x}",
            "content_sha256": f"{category_index + 11:064x}",
            "document_id": f"document-{category_index}",
            "segment_index": 0,
            "source_line": category_index + 1,
            "source_ordinal": category_index,
        },
        "gate": CATEGORY_CHECKPOINT_GATE,
        "job_token": token,
        "output_bytes": selected_bytes + selected_tokens * 2,
        "packed_shards": packed,
        "plan_fingerprint": release["input"]["plan_fingerprint"],
        "release_fingerprint": release["release_fingerprint"],
        "schema_version": 1,
        "selected_candidate_content_tokens": selected_content_tokens,
        "selected_candidate_documents": selected_documents,
        "selected_candidate_tokens": selected_tokens,
        "selected_inventory": {
            "object_bytes": selected_bytes,
            "object_count": 1,
            "objects_sha256": hashlib.sha256(
                f"inventory:{category}".encode()
            ).hexdigest(),
            "s3_uri": (
                f"s3://{release['storage']['bucket']}/{prefix}/selected/{category}"
            ),
        },
        "selection_sha256": hashlib.sha256(f"selection:{category}".encode()).hexdigest(),
        "status": "complete",
    }
    checkpoint["checkpoint_sha256"] = category_checkpoint_fingerprint(checkpoint)
    return checkpoint


def _production_result(release: dict, plan: dict, authorization: dict) -> dict:
    checkpoints = [
        _checkpoint(release, authorization, index)
        for index in range(len(release["categories"]))
    ]
    selected_categories = []
    for planned, checkpoint in zip(
        release["categories"], checkpoints, strict=True
    ):
        selected_categories.append(
            {
                "available_candidate_documents": planned[
                    "available_candidate_documents"
                ],
                "available_candidate_content_tokens": planned[
                    "available_candidate_content_tokens"
                ],
                "available_candidate_tokens": planned["available_candidate_tokens"],
                "category": planned["category"],
                "cutoff": checkpoint["cutoff"],
                "final_documents": int(planned["baseline_documents"])
                + int(checkpoint["selected_candidate_documents"]),
                "final_content_tokens": checkpoint[
                    "selected_candidate_content_tokens"
                ],
                "final_tokens": int(planned["baseline_tokens"])
                + int(checkpoint["selected_candidate_tokens"]),
                "selected_candidate_content_tokens": checkpoint[
                    "selected_candidate_content_tokens"
                ],
                "selected_candidate_documents": checkpoint[
                    "selected_candidate_documents"
                ],
                "selected_candidate_tokens": checkpoint["selected_candidate_tokens"],
                "selected_inventory": checkpoint["selected_inventory"],
                "selection_sha256": checkpoint["selection_sha256"],
            }
        )
    selection = {
        "categories": selected_categories,
        "dedup_result_sha256": release["input"]["dedup_result_sha256"],
        "final_documents": sum(
            int(row["final_documents"]) for row in selected_categories
        ),
        "final_content_tokens": sum(
            int(row["final_content_tokens"]) for row in selected_categories
        ),
        "final_tokens": sum(int(row["final_tokens"]) for row in selected_categories),
        "gate": "lumi_final_corpus_80b_selection_result_v0.1",
        "plan_fingerprint": release["input"]["plan_fingerprint"],
        "schema_version": 1,
        "selection_seed_sha256": release["input"]["selection_seed_sha256"],
        "status": "complete",
        "training_token_limit": release["limits"]["training_token_limit"],
    }
    selection["result_sha256"] = final_selection_result_fingerprint(selection)

    token = glue_final_corpus_production_job_token(authorization)
    shards = []
    for checkpoint in checkpoints:
        shards.extend(checkpoint["packed_shards"])
    shards.sort(key=lambda row: int(row["sequence"]))
    manifest = {
        "content_tokens": sum(int(row["content_tokens"]) for row in shards),
        "document_markers": sum(int(row["document_markers"]) for row in shards),
        "documents": sum(int(row["documents"]) for row in shards),
        "gate": "lumi_final_corpus_80b_shard_manifest_v0.1",
        "packing": {
            "doc_token": "<|doc|>",
            "doc_token_id": 3,
            "markers_per_document": 1,
            "token_accounting": "content_plus_one_document_marker",
        },
        "plan_fingerprint": release["input"]["plan_fingerprint"],
        "schema_version": 1,
        "selection_result_sha256": selection["result_sha256"],
        "shards": shards,
        "status": "sealed",
        "tokens": sum(int(row["tokens"]) for row in shards),
        "training_token_limit": release["limits"]["training_token_limit"],
    }
    result = {
        "authorization_sha256": authorization["authorization_sha256"],
        "category_checkpoints": checkpoints,
        "data_output_bytes": sum(int(row["output_bytes"]) for row in checkpoints),
        "dedup_result_sha256": release["input"]["dedup_result_sha256"],
        "gate": RESULT_GATE,
        "job_token": token,
        "plan_fingerprint": release["input"]["plan_fingerprint"],
        "release_fingerprint": release["release_fingerprint"],
        "runtime": {"python": "3.11.15", "spark": "3.5.6-amzn-1"},
        "schema_version": 1,
        "selection_result": selection,
        "shard_manifest": manifest,
        "status": "sealed",
    }
    result["result_sha256"] = production_result_fingerprint(result)
    return result


def test_package_is_non_authorizing_manifested_and_calibrated(
    package: tuple[dict, dict, Path],
) -> None:
    release, _, directory = package
    validated = validate_glue_final_corpus_production_release(release)
    assert validated == {
        "conservative_runtime_minutes": 458,
        "conservative_usd": "335.478978",
        "maximum_usd": "528.000000",
        "release_fingerprint": release["release_fingerprint"],
        "status": "verified_not_authorized",
    }
    assert all(value is False for value in release["authorization"].values())
    assert release["cost_model"]["ideal_linear_runtime_minutes_at_100_dpu"] == 153
    assert release["cost_model"]["ideal_linear_usd"] == "111.826367"
    assert release["cost_model"]["estimated_maximum_output_bytes"] == 513_342_209_894
    assert release["categories"][0]["first_new_shard_sequence"] == 0
    assert release["categories"][-1]["last_new_shard_sequence"] == 159
    rows = (directory / "manifest.sha256").read_text().splitlines()
    assert len(rows) == 5
    assert {row.split("  ", 1)[1] for row in rows} == {
        "final-corpus-80b-plan-v0.1.json",
        "glue-final-corpus-production-release-v0.1.json",
        "lumi-glue-final-corpus-production-code-v0.1.zip",
        "run_glue_final_corpus_production.py",
        "verified-final-corpus-canary-result-v0.1.json",
    }


def test_canary_runtime_evidence_is_bounded_and_cost_consistent() -> None:
    report = json.loads(
        (ROOT / "reports" / "glue-final-corpus-canary-result-v0.1.json").read_text()
    )
    assert _validate_canary_report(report)["job"]["dpu_seconds"] == 2303.0

    changed = copy.deepcopy(report)
    changed["job"]["dpu_seconds"] = 8 * 30 * 60 + 1
    with pytest.raises(ValueError, match="compute evidence"):
        _validate_canary_report(changed)

    changed = copy.deepcopy(report)
    changed["job"]["estimated_compute_usd_at_0_44_per_dpu_hour"] = "0.000001"
    with pytest.raises(ValueError, match="cost evidence"):
        _validate_canary_report(changed)


def test_exact_approval_and_job_definition_are_fail_closed(
    package: tuple[dict, dict, Path],
) -> None:
    release, _, _ = package
    expected = expected_glue_final_corpus_production_approval(release)
    with pytest.raises(ValueError, match="does not match"):
        build_glue_final_corpus_production_authorization(
            release, approval_text=expected + " "
        )
    authorization = _authorization(release)
    assert validate_glue_final_corpus_production_authorization(
        authorization, release
    )["status"] == "authorized_one_production_final_corpus_run"
    namespace = release["storage"]["namespace"]
    bucket = release["storage"]["bucket"]
    definition = build_glue_final_corpus_production_job_definition(
        release,
        authorization,
        role_arn="arn:aws:iam::123456789012:role/lumi-final-production",
        release_s3_uri=f"s3://{bucket}/{namespace}/control/release.json",
        authorization_s3_uri=f"s3://{bucket}/{namespace}/control/authorization.json",
    )
    assert definition["NumberOfWorkers"] == 100
    assert definition["Timeout"] == 720
    assert definition["MaxRetries"] == 0
    assert definition["ExecutionProperty"] == {"MaxConcurrentRuns": 1}
    immutable = definition["NonOverridableArguments"]
    assert immutable["--enable-auto-scaling"] == "false"
    assert immutable["--conf"] == glue_conf_argument(FINAL_CORPUS_SPARK_CONFIG)
    assert "spark.speculation=false" in immutable["--conf"]
    assert immutable["--additional-python-modules"] == "zstandard==0.25.0"
    assert "--LUMI_AUTHORIZATION_FILE" not in definition["DefaultArguments"]


def test_category_checkpoints_and_final_result_conserve_all_tokens(
    package: tuple[dict, dict, Path],
) -> None:
    release, plan, _ = package
    authorization = _authorization(release)
    checkpoint = _checkpoint(release, authorization, 0)
    assert validate_glue_final_corpus_category_checkpoint(
        checkpoint, release, authorization
    )["category"] == "science_and_education"
    changed = copy.deepcopy(checkpoint)
    changed["packed_shards"][0]["tokens"] += 1
    changed["packed_shards"][0]["object_bytes"] += 2
    changed["checkpoint_sha256"] = category_checkpoint_fingerprint(changed)
    with pytest.raises(ValueError, match="marker accounting"):
        validate_glue_final_corpus_category_checkpoint(
            changed, release, authorization
        )
    changed = copy.deepcopy(checkpoint)
    changed["packed_shards"][0]["document_markers"] += 1
    changed["checkpoint_sha256"] = category_checkpoint_fingerprint(changed)
    with pytest.raises(ValueError, match="marker accounting"):
        validate_glue_final_corpus_category_checkpoint(
            changed, release, authorization
        )

    result = _production_result(release, plan, authorization)
    validated = validate_glue_final_corpus_production_result(
        result, release, authorization, plan
    )
    assert validated["selection_result"]["final_tokens"] == 80_000_000_007
    assert len(validated["shard_manifest"]["shards"]) == 160
    assert validated["shard_manifest"]["document_markers"] == (
        validated["selection_result"]["final_documents"]
    )

    changed_result = copy.deepcopy(result)
    changed_result["shard_manifest"]["packing"]["doc_token_id"] = 4
    changed_result["result_sha256"] = production_result_fingerprint(changed_result)
    with pytest.raises(ValueError, match="document-marker packing"):
        validate_glue_final_corpus_production_result(
            changed_result, release, authorization, plan
        )


def test_launcher_policy_and_launch_claim_are_narrow_and_single_use() -> None:
    namespace = runpy.run_path(
        ROOT / "scripts" / "launch_glue_final_corpus_production.py",
        run_name="lumi_glue_final_corpus_production_launcher_test",
    )
    policy = namespace["_policy"](
        account="123456789012",
        region="us-east-1",
        bucket="lumi-test",
        accepted_prefix="lumi/input/accepted",
        exact_read_keys=["lumi/bootstrap/release.json", "lumi/baseline/tokens.bin"],
        run_prefix="lumi/final/run/token",
        result_prefix="lumi/final/result/token",
        temp_prefix="lumi/final/temp/token",
    )
    statements = {row["Sid"]: row for row in policy["Statement"]}
    assert statements["ReadExactInputs"]["Resource"] == [
        "arn:aws:s3:::lumi-test/lumi/input/accepted/*",
        "arn:aws:s3:::lumi-test/lumi/bootstrap/release.json",
        "arn:aws:s3:::lumi-test/lumi/baseline/tokens.bin",
    ]
    assert statements["ReadWriteExactRunAndTemp"]["Resource"] == [
        "arn:aws:s3:::lumi-test/lumi/final/run/token/*",
        "arn:aws:s3:::lumi-test/lumi/final/temp/token/*",
    ]
    assert statements["WriteImmutableResultOnly"]["Resource"] == (
        "arn:aws:s3:::lumi-test/lumi/final/result/token/*"
    )

    class ClaimExists(Exception):
        response = {"Error": {"Code": "PreconditionFailed"}}

    class FakeS3:
        claimed = False

        def put_object(self, **request: object) -> dict:
            assert request["IfNoneMatch"] == "*"
            if self.claimed:
                raise ClaimExists
            self.claimed = True
            return {"VersionId": "claim-version"}

    s3 = FakeS3()
    claim = namespace["_claim_once"](
        s3,
        bucket="lumi-test",
        key="lumi/final/claims/auth.json",
        claim={"authorization_sha256": "a" * 64},
    )
    assert claim["version_id"] == "claim-version"
    with pytest.raises(RuntimeError, match="already has a launch claim"):
        namespace["_claim_once"](
            s3,
            bucket="lumi-test",
            key="lumi/final/claims/auth.json",
            claim={"authorization_sha256": "a" * 64},
        )


def test_runtime_materializes_verified_partial_baseline(tmp_path: Path) -> None:
    zstandard = pytest.importorskip("zstandard")
    namespace = runpy.run_path(
        ROOT / "scripts" / "run_glue_final_corpus_production.py",
        run_name="lumi_glue_final_corpus_production_runtime_test",
    )
    token_bytes = b"\x01\x00\x02\x00\x03\x00\x04\x00"
    token_sha = hashlib.sha256(token_bytes).hexdigest()
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:") as handle:
        info = tarfile.TarInfo("./partial-current-shard/tokens.uint16.bin")
        info.size = len(token_bytes)
        handle.addfile(info, io.BytesIO(token_bytes))
    archive_bytes = zstandard.ZstdCompressor().compress(tar_buffer.getvalue())
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()

    class FakeS3:
        def __init__(self) -> None:
            self.objects = {("source", "baseline.tar.zst"): archive_bytes}
            self.uploaded: dict[tuple[str, str], bytes] = {}

        def head_object(self, *, Bucket: str, Key: str, **_: object) -> dict:
            body = self.objects.get((Bucket, Key), self.uploaded.get((Bucket, Key)))
            assert body is not None
            return {
                "ContentLength": len(body),
                "Metadata": {"sha256": hashlib.sha256(body).hexdigest()},
            }

        def download_file(self, bucket: str, key: str, destination: str) -> None:
            Path(destination).write_bytes(self.objects[(bucket, key)])

        def put_object(self, *, Bucket: str, Key: str, Body: object, **_: object) -> dict:
            assert hasattr(Body, "read")
            self.uploaded[(Bucket, Key)] = Body.read()
            return {"VersionId": "version-1"}

    s3 = FakeS3()
    result = namespace["_materialize_partial_baseline"](
        s3,
        bucket="destination",
        archive={
            "bytes": len(archive_bytes),
            "s3_uri": "s3://source/baseline.tar.zst",
            "sha256": archive_sha,
        },
        partial={
            "bytes": len(token_bytes),
            "documents": 1,
            "member": "partial-current-shard/tokens.uint16.bin",
            "sha256": token_sha,
            "tokens": 4,
        },
        output_prefix="lumi/final/packed",
    )
    assert result["sequence"] == 2
    assert result["tokens"] == 4
    assert s3.uploaded[
        ("destination", f"lumi/final/packed/science_and_education/{token_sha}.uint16.bin")
    ] == token_bytes


def test_runtime_rejects_unsafe_tar_member_names() -> None:
    namespace = runpy.run_path(
        ROOT / "scripts" / "run_glue_final_corpus_production.py",
        run_name="lumi_glue_final_corpus_production_tar_name_test",
    )
    normalize = namespace["_normalized_tar_member_name"]
    assert normalize("partial-current-shard/tokens.uint16.bin") == (
        "partial-current-shard/tokens.uint16.bin"
    )
    assert normalize("./partial-current-shard/tokens.uint16.bin") == (
        "partial-current-shard/tokens.uint16.bin"
    )
    with pytest.raises(ValueError, match="unsafe"):
        normalize("../partial-current-shard/tokens.uint16.bin")
    with pytest.raises(ValueError, match="unsafe"):
        normalize("/partial-current-shard/tokens.uint16.bin")


def test_failed_production_attempt_report_is_fail_closed() -> None:
    report = json.loads(
        (
            ROOT
            / "reports"
            / "glue-final-corpus-production-attempt-v0.1.json"
        ).read_text()
    )
    assert report["status"] == "failed_closed_before_candidate_selection"
    assert report["compute"] == {
        "dpu_seconds": 8380,
        "estimated_usd_at_0_44_per_dpu_hour": "1.024222",
        "execution_seconds": 83,
    }
    assert report["output"]["result_prefix_empty"] is True
    assert report["output"]["category_checkpoints"] == 0
    assert report["output"]["corrupt_objects_detected"] is False
    assert report["decision"]["replacement_run"] == "not_authorized"
    assert report["cleanup"] == {
        "billing_resource_active": False,
        "job": "deleted_and_live_absence_verified",
        "role": "deleted_and_live_absence_verified",
    }
