"""Global-dedup job contract for Lumi's two-pass EMR corpus pipeline."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from .emr_production_map import validate_map_checkpoint
from .emr_serverless_production import _bounded_s3_uri, _resource_shape


def dedup_job_token(
    authorization: Mapping[str, Any], map_checkpoint: Mapping[str, Any]
) -> str:
    run_token = authorization.get("run_token")
    checkpoint_sha256 = map_checkpoint.get("checkpoint_sha256")
    if not isinstance(run_token, str) or len(run_token) != 32:
        raise ValueError("production dedup authorization token is invalid")
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise ValueError("production dedup map checkpoint is invalid")
    return hashlib.sha256(f"dedup:{run_token}:{checkpoint_sha256}".encode()).hexdigest()[:32]


def build_global_dedup_job_request(
    plan: Mapping[str, Any],
    map_checkpoint: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    application_id: str,
    execution_role_arn: str,
    plan_s3_uri: str,
    map_checkpoint_s3_uri: str,
    authorization_s3_uri: str,
    code_archive_s3_uri: str,
    job_script_s3_uri: str,
) -> dict[str, Any]:
    """Render one 128-vCPU global dedup job without calling AWS."""

    validate_map_checkpoint(map_checkpoint, plan, authorization)
    if map_checkpoint.get("status") != "ready_for_global_dedup":
        raise ValueError("production map checkpoint is not ready for global dedup")
    if re.fullmatch(r"[0-9a-z]+", application_id) is None:
        raise ValueError("production dedup application ID is invalid")
    if re.fullmatch(
        r"arn:aws:iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]+", execution_role_arn
    ) is None:
        raise ValueError("production dedup execution role ARN is invalid")
    bucket = str(plan["storage"]["bucket"])
    files = [
        plan_s3_uri,
        map_checkpoint_s3_uri,
        authorization_s3_uri,
        f"s3://{bucket}/{plan['storage']['artifacts']['initial_anchors']}",
    ]
    file_names = [_bounded_s3_uri(uri, bucket=bucket) for uri in files]
    if len(file_names) != len(set(file_names)):
        raise ValueError("production dedup SparkFiles basenames collide")
    _bounded_s3_uri(code_archive_s3_uri, bucket=bucket)
    _bounded_s3_uri(job_script_s3_uri, bucket=bucket)
    maximum_vcpu = int(plan["pipeline"]["global_dedup_job_vcpu"])
    shape = _resource_shape(plan["application"], maximum_vcpu)
    python = "/opt/lumi/python312/bin/python"
    partitions = shape["executor_instances"] * 32
    parameters = [
        "--conf", "spark.driver.cores=4",
        "--conf", "spark.driver.memory=14g",
        "--conf", "spark.executor.cores=4",
        "--conf", "spark.executor.memory=14g",
        "--conf", f"spark.executor.instances={shape['executor_instances']}",
        "--conf", "spark.dynamicAllocation.enabled=false",
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
    job_token = dedup_job_token(authorization, map_checkpoint)
    runtime_minutes = int(
        Decimal(str(plan["pipeline"]["global_dedup_job_timeout_hours"]))
        * Decimal(60)
    )
    return {
        "applicationId": application_id,
        "clientToken": f"lumi-production-dedup-{job_token}",
        "configurationOverrides": {
            "monitoringConfiguration": {
                "s3MonitoringConfiguration": {
                    "logUri": (
                        f"s3://{bucket}/{plan['storage']['namespace']}/control/distributed/"
                        f"production/dedup-runs/{job_token}/logs"
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
                    "--lumi-authorization-file", file_names[2],
                    "--lumi-job-token", job_token,
                    "--lumi-map-checkpoint-file", file_names[1],
                    "--lumi-plan-file", file_names[0],
                    "--lumi-run-token", str(authorization["run_token"]),
                ],
                "sparkSubmitParameters": " ".join(parameters),
            }
        },
        "mode": "BATCH",
        "name": f"lumi-production-dedup-128vcpu-{job_token[:12]}",
        "retryPolicy": {"maxAttempts": 1},
        "tags": {
            "LumiPlanFingerprint": str(plan["plan_fingerprint"]),
            "LumiScope": "bounded-production-global-dedup-only",
        },
    }
