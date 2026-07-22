from __future__ import annotations

import copy
import hashlib
import json
import runpy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lumi.final_corpus import build_final_corpus_plan
from lumi.glue_final_corpus import (
    build_glue_final_corpus_canary_authorization,
    build_glue_final_corpus_canary_job_definition,
    build_glue_final_corpus_canary_package,
    expected_glue_final_corpus_canary_approval,
    glue_final_corpus_canary_job_token,
    validate_glue_final_corpus_canary_authorization,
    validate_glue_final_corpus_canary_release,
    validate_glue_final_corpus_canary_result,
)
from lumi.glue_runtime import FINAL_CORPUS_SPARK_CONFIG, glue_conf_argument

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "b" * 40


@pytest.fixture()
def release(tmp_path: Path) -> dict:
    plan = build_final_corpus_plan(
        ROOT / "configs" / "final-corpus-80b-v0.1.json",
        repository_root=ROOT,
        source_commit=SOURCE_COMMIT,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return build_glue_final_corpus_canary_package(
        contract_path=ROOT / "configs" / "glue-final-corpus-canary-v0.1.json",
        final_plan_path=plan_path,
        repository_root=ROOT,
        source_commit=SOURCE_COMMIT,
        output_directory=tmp_path / "package",
    )


def _authorization(release: dict) -> dict:
    return build_glue_final_corpus_canary_authorization(
        release,
        approval_text=expected_glue_final_corpus_canary_approval(release),
        now=datetime.now(UTC),
    )


def _result(release: dict, authorization: dict) -> dict:
    token = glue_final_corpus_canary_job_token(authorization)
    bucket = release["storage"]["bucket"]
    prefix = release["output"]["canary_prefix"]
    categories = []
    output_bytes = 0
    total_documents = 0
    total_tokens = 0
    for category_index, category_row in enumerate(release["categories"]):
        category = category_row["category"]
        selected_tokens = 1_000_001
        selected_documents = 100
        inventory_bytes = 10_000 + category_index
        inventory_sha = hashlib.sha256(f"inventory:{category}".encode()).hexdigest()
        shards = []
        token_parts = (250_000, 250_000, 250_000, 250_001)
        for sequence, tokens in enumerate(token_parts):
            digest = hashlib.sha256(f"{category}:{sequence}".encode()).hexdigest()
            shards.append(
                {
                    "documents": 25,
                    "object_bytes": tokens * 2,
                    "object_sha256": digest,
                    "s3_uri": (
                        f"s3://{bucket}/{prefix}/{token}/packed/{category}/"
                        f"{digest}.uint16.bin"
                    ),
                    "sequence": sequence,
                    "tokens": tokens,
                }
            )
        categories.append(
            {
                "available_documents": 1000,
                "available_tokens": 10_000_000,
                "category": category,
                "cutoff": {"_lumi_selection_rank": f"{category_index + 1:064x}"},
                "packed_shards": shards,
                "partition_parity": "pass",
                "selected_documents": selected_documents,
                "selected_inventory": {
                    "object_bytes": inventory_bytes,
                    "object_count": 1,
                    "objects_sha256": inventory_sha,
                    "s3_uri": f"s3://{bucket}/{prefix}/{token}/selected/{category}",
                },
                "selected_tokens": selected_tokens,
                "selection_sha256": hashlib.sha256(category.encode()).hexdigest(),
            }
        )
        output_bytes += inventory_bytes + selected_tokens * 2
        total_documents += selected_documents
        total_tokens += selected_tokens
    return {
        "accepted_input_inventory": copy.deepcopy(
            release["input"]["accepted_candidates"]
        ),
        "authorization_sha256": authorization["authorization_sha256"],
        "baseline_verification": {
            "archive_head": "pass",
            "sealed_shard_heads": 2,
            "status": "pass",
        },
        "categories": categories,
        "dedup_result_sha256": release["input"]["dedup_result_sha256"],
        "gate": "lumi_glue_final_corpus_canary_result_v0.1",
        "job_token": token,
        "output_bytes": output_bytes,
        "plan_fingerprint": release["input"]["plan_fingerprint"],
        "release_fingerprint": release["release_fingerprint"],
        "runtime": {"python": "3.11.15", "spark": "3.5.6-amzn-1"},
        "schema_version": 1,
        "selected_documents": total_documents,
        "selected_tokens": total_tokens,
        "selection_seed_sha256": release["input"]["selection_seed_sha256"],
        "status": "pass_selection_sharding_canary",
    }


def test_canary_package_is_non_authorizing_and_manifested(
    release: dict, tmp_path: Path
) -> None:
    validated = validate_glue_final_corpus_canary_release(release)
    assert validated["status"] == "verified_not_authorized"
    assert validated["maximum_usd"] == "1.760000"
    assert all(value is False for value in release["authorization"].values())
    package = tmp_path / "package"
    manifest_rows = (package / "manifest.sha256").read_text().splitlines()
    assert len(manifest_rows) == 4
    assert release["input"]["accepted_candidates"]["tokens"] == 119_762_267_075
    assert sum(row["final_target_tokens"] for row in release["categories"]) == (
        80_000_000_000
    )


def test_exact_approval_is_required_and_job_is_bounded(release: dict) -> None:
    expected = expected_glue_final_corpus_canary_approval(release)
    with pytest.raises(ValueError, match="does not match"):
        build_glue_final_corpus_canary_authorization(
            release,
            approval_text=expected + " ",
        )
    authorization = _authorization(release)
    assert validate_glue_final_corpus_canary_authorization(
        authorization, release
    )["status"] == "authorized_one_bounded_final_corpus_canary"
    definition = build_glue_final_corpus_canary_job_definition(
        release,
        authorization,
        role_arn="arn:aws:iam::123456789012:role/lumi-final-canary",
        release_s3_uri=(
            f"s3://{release['storage']['bucket']}/{release['storage']['namespace']}/"
            "control/glue-canary/bootstrap/releases/release.json"
        ),
        authorization_s3_uri=(
            f"s3://{release['storage']['bucket']}/{release['storage']['namespace']}/"
            "control/glue-canary/bootstrap/authorizations/authorization.json"
        ),
    )
    assert definition["NumberOfWorkers"] == 8
    assert definition["Timeout"] == 30
    assert definition["MaxRetries"] == 0
    assert definition["ExecutionProperty"] == {"MaxConcurrentRuns": 1}
    assert "--LUMI_AUTHORIZATION_FILE" not in definition["DefaultArguments"]
    assert definition["NonOverridableArguments"]["--enable-auto-scaling"] == "false"
    assert definition["NonOverridableArguments"]["--conf"] == glue_conf_argument(
        FINAL_CORPUS_SPARK_CONFIG
    )
    assert definition["NonOverridableArguments"]["--TempDir"].endswith("/")


def test_canary_result_conserves_selection_and_packed_bytes(release: dict) -> None:
    authorization = _authorization(release)
    result = _result(release, authorization)
    assert validate_glue_final_corpus_canary_result(
        result, release, authorization
    )["selected_tokens"] == 7_000_007

    changed = copy.deepcopy(result)
    changed["categories"][0]["packed_shards"][0]["object_bytes"] += 1
    with pytest.raises(ValueError, match="packed-shard bounds"):
        validate_glue_final_corpus_canary_result(changed, release, authorization)

    changed = copy.deepcopy(result)
    changed["categories"][1]["packed_shards"][0]["object_sha256"] = changed[
        "categories"
    ][0]["packed_shards"][0]["object_sha256"]
    with pytest.raises(ValueError, match="repeats a packed object"):
        validate_glue_final_corpus_canary_result(changed, release, authorization)


def test_release_and_authorization_fail_closed_on_tampering(release: dict) -> None:
    changed = copy.deepcopy(release)
    changed["job"]["number_of_workers"] = 9
    with pytest.raises(ValueError, match="fingerprint changed"):
        validate_glue_final_corpus_canary_release(changed)

    authorization = _authorization(release)
    changed_authorization = copy.deepcopy(authorization)
    changed_authorization["permissions"]["paid_glue_production"] = True
    with pytest.raises(ValueError, match="fingerprint changed"):
        validate_glue_final_corpus_canary_authorization(
            changed_authorization, release
        )


def test_launcher_policy_is_bounded_to_exact_inputs_and_canary_outputs() -> None:
    namespace = runpy.run_path(
        ROOT / "scripts" / "launch_glue_final_corpus_canary.py",
        run_name="lumi_glue_final_corpus_launcher_test",
    )
    policy = namespace["_policy"](
        account="123456789012",
        region="us-east-1",
        bucket="lumi-test",
        accepted_prefix="lumi/input/accepted",
        exact_read_keys=["lumi/bootstrap/release.json", "lumi/baseline/tokens.bin"],
        canary_prefix="lumi/final/canary/token",
        result_prefix="lumi/final/results/token",
        temp_prefix="lumi/final/temp/token",
    )
    statements = {row["Sid"]: row for row in policy["Statement"]}
    read_resources = statements["ReadExactInputs"]["Resource"]
    assert "arn:aws:s3:::lumi-test/lumi/input/accepted/*" in read_resources
    assert "arn:aws:s3:::lumi-test/lumi/bootstrap/release.json" in read_resources
    write_resources = statements["ReadWriteCanaryAndTempOnly"]["Resource"]
    assert write_resources == [
        "arn:aws:s3:::lumi-test/lumi/final/canary/token/*",
        "arn:aws:s3:::lumi-test/lumi/final/temp/token/*",
    ]
    assert statements["WriteImmutableResultOnly"]["Resource"] == (
        "arn:aws:s3:::lumi-test/lumi/final/results/token/*"
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
