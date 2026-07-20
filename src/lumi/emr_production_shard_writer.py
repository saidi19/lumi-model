"""Per-category shard-writer contract for Lumi's two-pass EMR pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from decimal import Decimal
from typing import Any
from urllib.parse import urlparse

from .emr_production_dedup import dedup_job_token
from .emr_serverless_production import _bounded_s3_uri, _resource_shape


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def dedup_result_fingerprint(result: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(result)).hexdigest()


def validate_global_dedup_result(
    result: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    if result.get("schema_version") != 1:
        raise ValueError("unsupported global-dedup result schema")
    if result.get("gate") != "lumi_emr_production_global_dedup_result_v0.1":
        raise ValueError("unexpected global-dedup result gate")
    if result.get("status") != "pass_category_floors" or result.get(
        "total_deficit_tokens"
    ) != 0:
        raise ValueError("global dedup has not satisfied every category floor")
    if result.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("global-dedup result targets another plan")
    if result.get("authorization_run_token") != authorization["run_token"]:
        raise ValueError("global-dedup result authorization changed")
    map_checkpoint_sha256 = result.get("map_checkpoint_sha256")
    if (
        not isinstance(map_checkpoint_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", map_checkpoint_sha256) is None
    ):
        raise ValueError("global-dedup map checkpoint SHA-256 is invalid")
    expected_job_token = dedup_job_token(
        authorization, {"checkpoint_sha256": map_checkpoint_sha256}
    )
    if result.get("job_token") != expected_job_token:
        raise ValueError("global-dedup job token changed")
    categories = result.get("categories")
    if not isinstance(categories, list):
        raise ValueError("global-dedup category results are missing")
    expected = {row["category"]: row for row in plan["categories"]}
    observed: dict[str, Mapping[str, Any]] = {}
    selected_documents = 0
    selected_tokens = 0
    final_tokens = 0
    for row in categories:
        if not isinstance(row, Mapping) or row.get("category") not in expected:
            raise ValueError("global-dedup category result is invalid")
        name = str(row["category"])
        if name in observed:
            raise ValueError("global-dedup repeats a category")
        observed[name] = row
        if int(row.get("floor_tokens", -1)) != int(expected[name]["floor_tokens"]):
            raise ValueError("global-dedup category floor changed")
        if int(row.get("selected_tokens", -1)) < 0 or int(
            row.get("selected_documents", -1)
        ) < 0:
            raise ValueError("global-dedup selected totals are invalid")
        if int(row.get("final_tokens", -1)) != int(expected[name]["accepted_tokens"]) + int(
            row["selected_tokens"]
        ):
            raise ValueError("global-dedup category token conservation failed")
        if int(row.get("final_documents", -1)) != int(
            expected[name]["accepted_documents"]
        ) + int(row["selected_documents"]):
            raise ValueError("global-dedup category document conservation failed")
        if int(row.get("final_tokens", -1)) < int(row["floor_tokens"]):
            raise ValueError("global-dedup category remains below its floor")
        if int(row["final_tokens"]) > int(row["floor_tokens"]) + int(
            plan["limits"]["maximum_category_overshoot_tokens"]
        ):
            raise ValueError("global-dedup category exceeds its bounded overshoot")
        if int(row.get("deficit_tokens", -1)) != 0:
            raise ValueError("global-dedup category reports a deficit")
        selected_documents += int(row["selected_documents"])
        selected_tokens += int(row["selected_tokens"])
        final_tokens += int(row["final_tokens"])
    if set(observed) != set(expected):
        raise ValueError("global-dedup category set changed")
    accepted = result.get("accepted_candidates")
    if not isinstance(accepted, Mapping):
        raise ValueError("global-dedup accepted-candidate output is missing")
    if int(accepted.get("documents", -1)) != selected_documents:
        raise ValueError("global-dedup accepted document total changed")
    if int(accepted.get("tokens", -1)) != selected_tokens:
        raise ValueError("global-dedup accepted token total changed")
    parsed = urlparse(str(accepted.get("s3_uri")))
    if parsed.scheme != "s3" or parsed.netloc != plan["storage"]["bucket"]:
        raise ValueError("global-dedup accepted candidates left the production bucket")
    namespace = f"/{plan['storage']['namespace']}/"
    if not parsed.path.startswith(namespace):
        raise ValueError("global-dedup accepted candidates left the production namespace")
    for name in ("object_bytes", "object_count"):
        value = accepted.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"global-dedup accepted {name} is invalid")
    rejected = result.get("rejected_ids")
    if not isinstance(rejected, Mapping):
        raise ValueError("global-dedup rejected-ID output is missing")
    rejected_uri = urlparse(str(rejected.get("s3_uri")))
    if (
        rejected_uri.scheme != "s3"
        or rejected_uri.netloc != plan["storage"]["bucket"]
        or not rejected_uri.path.startswith(namespace)
    ):
        raise ValueError("global-dedup rejected IDs left the production namespace")
    if final_tokens > int(plan["limits"]["candidate_hard_maximum_tokens"]):
        raise ValueError("global-dedup result exceeds the 150B candidate cap")
    return dict(result)


def shard_writer_job_token(
    authorization: Mapping[str, Any],
    dedup_result: Mapping[str, Any],
    category: str,
) -> str:
    run_token = authorization.get("run_token")
    if not isinstance(run_token, str) or len(run_token) != 32:
        raise ValueError("shard-writer authorization token is invalid")
    if re.fullmatch(r"[a-z0-9_]+", category) is None:
        raise ValueError("shard-writer category is invalid")
    result_sha = dedup_result_fingerprint(dedup_result)
    return hashlib.sha256(f"shard:{run_token}:{result_sha}:{category}".encode()).hexdigest()[:32]


def build_shard_writer_job_request(
    plan: Mapping[str, Any],
    accepted_checkpoint: Mapping[str, Any],
    authorization: Mapping[str, Any],
    dedup_result: Mapping[str, Any],
    *,
    category: str,
    application_id: str,
    execution_role_arn: str,
    plan_s3_uri: str,
    accepted_checkpoint_s3_uri: str,
    authorization_s3_uri: str,
    dedup_result_s3_uri: str,
    code_archive_s3_uri: str,
    job_script_s3_uri: str,
) -> dict[str, Any]:
    """Render one of seven independent 16-vCPU shard writers."""

    validate_global_dedup_result(dedup_result, plan, authorization)
    if accepted_checkpoint.get("checkpoint_sha256") != authorization.get(
        "checkpoint_sha256"
    ):
        raise ValueError("shard writer must start from the authorized accepted prefix")
    if category not in {str(row["category"]) for row in plan["categories"]}:
        raise ValueError("shard writer category is not in the production mixture")
    if re.fullmatch(r"[0-9a-z]+", application_id) is None:
        raise ValueError("shard-writer application ID is invalid")
    if re.fullmatch(
        r"arn:aws:iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]+", execution_role_arn
    ) is None:
        raise ValueError("shard-writer execution role ARN is invalid")
    bucket = str(plan["storage"]["bucket"])
    files = [
        plan_s3_uri,
        accepted_checkpoint_s3_uri,
        authorization_s3_uri,
        dedup_result_s3_uri,
    ]
    file_names = [_bounded_s3_uri(uri, bucket=bucket) for uri in files]
    if len(file_names) != len(set(file_names)):
        raise ValueError("shard-writer SparkFiles basenames collide")
    _bounded_s3_uri(code_archive_s3_uri, bucket=bucket)
    _bounded_s3_uri(job_script_s3_uri, bucket=bucket)
    maximum_vcpu = int(plan["pipeline"]["shard_writer_job_vcpu"])
    shape = _resource_shape(plan["application"], maximum_vcpu)
    python = "/opt/lumi/python312/bin/python"
    partitions = shape["executor_instances"] * 16
    parameters = [
        "--conf", "spark.driver.cores=4",
        "--conf", "spark.driver.memory=14g",
        "--conf", "spark.executor.cores=4",
        "--conf", "spark.executor.memory=14g",
        "--conf", f"spark.executor.instances={shape['executor_instances']}",
        "--conf", "spark.dynamicAllocation.enabled=false",
        "--conf", "spark.speculation=false",
        "--conf", "spark.emr-serverless.driver.disk=20G",
        "--conf", "spark.emr-serverless.executor.disk=20G",
        "--conf", f"spark.sql.shuffle.partitions={partitions}",
        "--conf", f"spark.default.parallelism={partitions}",
        "--conf", "spark.network.crypto.enabled=true",
        "--conf", "spark.io.encryption.enabled=true",
        "--conf", f"spark.emr-serverless.driverEnv.PYSPARK_DRIVER_PYTHON={python}",
        "--conf", f"spark.emr-serverless.driverEnv.PYSPARK_PYTHON={python}",
        "--conf", f"spark.executorEnv.PYSPARK_PYTHON={python}",
        "--py-files", code_archive_s3_uri,
        "--files", ",".join(files),
    ]
    token = shard_writer_job_token(authorization, dedup_result, category)
    runtime_minutes = int(
        Decimal(str(plan["pipeline"]["shard_writer_job_timeout_hours"]))
        * Decimal(60)
    )
    return {
        "applicationId": application_id,
        "clientToken": f"lumi-production-shard-{token}",
        "configurationOverrides": {
            "monitoringConfiguration": {
                "s3MonitoringConfiguration": {
                    "logUri": (
                        f"s3://{bucket}/{plan['storage']['namespace']}/control/distributed/"
                        f"production/shard-runs/{token}/logs"
                    )
                }
            }
        },
        "executionRoleArn": execution_role_arn,
        "executionTimeoutMinutes": runtime_minutes,
        "jobDriver": {
            "sparkSubmit": {
                "entryPoint": job_script_s3_uri,
                "entryPointArguments": [
                    "--lumi-accepted-checkpoint-file", file_names[1],
                    "--lumi-authorization-file", file_names[2],
                    "--lumi-category", category,
                    "--lumi-dedup-result-file", file_names[3],
                    "--lumi-job-token", token,
                    "--lumi-plan-file", file_names[0],
                    "--lumi-run-token", str(authorization["run_token"]),
                ],
                "sparkSubmitParameters": " ".join(parameters),
            }
        },
        "mode": "BATCH",
        "name": f"lumi-production-shard-{category[:16]}-{token[:12]}",
        "retryPolicy": {"maxAttempts": 1},
        "tags": {
            "LumiPlanFingerprint": str(plan["plan_fingerprint"]),
            "LumiScope": f"bounded-production-shard-writer-{category}",
        },
    }
