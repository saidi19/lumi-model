"""Provision one bounded, CPU-only AWS production corpus worker for Lumi."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .aws_s3_canary import AwsCli
from .aws_s3_production import _put_immutable_object

SCHEMA_VERSION = 1
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
CANONICAL_UBUNTU_OWNER = "099720109477"
UBUNTU_AMI_PARAMETER = (
    "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
)
SSM_CORE_POLICY_ARN = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any], *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if private:
        temporary.chmod(0o600)
    temporary.replace(path)


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _commit(value: Any) -> str:
    if not isinstance(value, str) or COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError("source commit must be a lowercase 40-character Git commit")
    return value


def build_worker_launch_plan(
    worker_release_path: Path,
    *,
    production_report_path: Path,
    pricing_path: Path,
    source_bundle_path: Path,
    source_commit: str,
) -> dict[str, Any]:
    """Build the immutable launch envelope without accessing AWS."""

    worker_release_path = worker_release_path.resolve()
    production_report_path = production_report_path.resolve()
    pricing_path = pricing_path.resolve()
    source_bundle_path = source_bundle_path.resolve()
    for path in (
        worker_release_path,
        production_report_path,
        pricing_path,
        source_bundle_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    release = _read_json(worker_release_path)
    production = _read_json(production_report_path)
    pricing = _read_json(pricing_path)
    commit = _commit(source_commit)
    if release.get("schema_version") != 1 or release.get("gate") != (
        "production_corpus_worker_release_v0.1"
    ):
        raise ValueError("production worker release is invalid")
    if release.get("decision") != {
        "full_4b_training": "not_authorized",
        "gpu_compute": "not_authorized",
        "status": "ready_to_launch_acquisition_worker",
    }:
        raise ValueError("worker release does not preserve the no-training boundary")
    if production.get("decision", {}).get("status") != (
        "production_s3_ready_for_acquisition_worker"
    ):
        raise ValueError("production S3 is not ready for a worker")
    if production.get("plan_fingerprint") != release.get("production_plan_fingerprint"):
        raise ValueError("worker release and production S3 plan differ")
    if pricing.get("gate") != "aws_production_worker_pricing_snapshot_v0.1":
        raise ValueError("worker pricing snapshot is invalid")
    runtime = release.get("runtime", {})
    if runtime.get("instance_type") != pricing.get("ec2", {}).get("instance_type"):
        raise ValueError("worker runtime and pricing instance types differ")
    if runtime.get("market") != "on_demand":
        raise ValueError("production acquisition must use on-demand EC2")
    if runtime.get("instance_initiated_shutdown_behavior") != "terminate":
        raise ValueError("worker must terminate after instance-initiated shutdown")
    if runtime.get("metadata_http_tokens") != "required":
        raise ValueError("worker must require IMDSv2")
    if runtime.get("state_volume_delete_on_termination") is not True:
        raise ValueError("worker state volume must not create an idle-storage bill")
    maximum_cost = Decimal(str(release.get("cost_forecast", {}).get("maximum_72h_usd")))
    if maximum_cost >= Decimal("100"):
        raise ValueError("worker release exceeds the approved $100 initial cap")

    worker_fingerprint = _sha256(release.get("worker_release_fingerprint"), "worker release")
    namespace = str(production["provider"]["namespace"])
    source_sha256 = _sha256_file(source_bundle_path)
    release_sha256 = _sha256_file(worker_release_path)
    report_sha256 = _sha256_file(production_report_path)
    artifact_keys = {
        "frozen_tokenizer": (
            f"{namespace}/control/bootstrap/artifacts/frozen_tokenizer/"
            f"{release['external_artifacts']['frozen_tokenizer']['sha256']}.json"
        ),
        "production_report": (
            f"{namespace}/control/worker-releases/{worker_fingerprint}/"
            f"production-report-{report_sha256}.json"
        ),
        "source_bundle": (
            f"{namespace}/control/worker-releases/{worker_fingerprint}/"
            f"source-{commit}-{source_sha256}.tar.gz"
        ),
        "training_exclusion_ledger": (
            f"{namespace}/control/bootstrap/artifacts/training_exclusion_ledger/"
            f"{release['external_artifacts']['training_exclusion_ledger']['sha256']}.jsonl"
        ),
        "worker_release": (
            f"{namespace}/control/worker-releases/{worker_fingerprint}/"
            f"release-{release_sha256}.json"
        ),
    }
    material = {
        "artifact_keys": artifact_keys,
        "pricing_sha256": _sha256_file(pricing_path),
        "production_report_sha256": report_sha256,
        "source_bundle_sha256": source_sha256,
        "source_commit": commit,
        "worker_release_fingerprint": worker_fingerprint,
        "worker_release_sha256": release_sha256,
    }
    return {
        "artifact_keys": artifact_keys,
        "authorization": {
            "full_4b_training": False,
            "model_training": False,
            "paid_gpu_compute": False,
            "production_adaptive_acquisition": True,
        },
        "bucket": production["bucket_identity"]["data_bucket"],
        "cost_forecast": release["cost_forecast"],
        "created_at": _utc_now(),
        "gate": "aws_production_acquisition_worker_launch_v0.1",
        "launch_plan_fingerprint": _sha256_bytes(_canonical_json_bytes(material)),
        "namespace": namespace,
        "pricing": pricing,
        "production_account_id_sha256": production["bucket_identity"][
            "account_id_sha256"
        ],
        "production_plan_fingerprint": production["plan_fingerprint"],
        "production_report_sha256": report_sha256,
        "region": production["provider"]["region"],
        "runtime": runtime,
        "schema_version": SCHEMA_VERSION,
        "source_bundle": {
            "bytes": source_bundle_path.stat().st_size,
            "sha256": source_sha256,
        },
        "source_commit": commit,
        "worker_release_fingerprint": worker_fingerprint,
        "worker_release_sha256": release_sha256,
    }


def _iam_policy(bucket: str, namespace: str) -> dict[str, Any]:
    read_prefixes = [
        f"{namespace}/control/bootstrap/*",
        f"{namespace}/control/checkpoints/*",
        f"{namespace}/control/worker-releases/*",
    ]
    writable_prefixes = [
        f"{namespace}/staging/*",
        f"{namespace}/control/checkpoints/*",
        f"{namespace}/control/manifests/*",
        f"{namespace}/control/progress/*",
        f"{namespace}/control/provenance/*",
        f"{namespace}/control/worker-logs/*",
    ]
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ReadExactLumiControlObjects",
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket}/{prefix}" for prefix in read_prefixes],
            },
            {
                "Sid": "WriteAndVerifyLumiCorpusObjects",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject"],
                "Resource": [
                    f"arn:aws:s3:::{bucket}/{prefix}" for prefix in writable_prefixes
                ],
            },
            {
                "Sid": "ListOnlyLumiResumeCheckpoints",
                "Effect": "Allow",
                "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
                "Resource": f"arn:aws:s3:::{bucket}",
                "Condition": {
                    "StringLike": {
                        "s3:prefix": [f"{namespace}/control/checkpoints/*"]
                    }
                },
            },
        ],
    }


def _trust_policy() -> dict[str, Any]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }


def _ensure_iam(
    cli: AwsCli,
    *,
    role_name: str,
    profile_name: str,
    bucket: str,
    namespace: str,
    fingerprint: str,
) -> None:
    code, role, error = cli.call(
        "iam", "get-role", "--role-name", role_name, allow_failure=True
    )
    if code:
        if "NoSuchEntity" not in error:
            raise RuntimeError(f"unable to inspect worker IAM role: {error}")
        cli.call(
            "iam",
            "create-role",
            "--role-name",
            role_name,
            "--path",
            "/lumi/",
            "--assume-role-policy-document",
            json.dumps(_trust_policy(), separators=(",", ":")),
            "--description",
            "CPU-only Lumi production corpus acquisition worker",
            "--tags",
            "Key=Project,Value=Lumi",
            "Key=Purpose,Value=ProductionCorpusAcquisition",
            f"Key=Release,Value={fingerprint[:16]}",
        )
    else:
        trust = role.get("Role", {}).get("AssumeRolePolicyDocument")
        if trust != _trust_policy():
            raise ValueError("existing worker IAM role trust policy differs")
    cli.call(
        "iam",
        "put-role-policy",
        "--role-name",
        role_name,
        "--policy-name",
        "LumiProductionCorpusObjects",
        "--policy-document",
        json.dumps(_iam_policy(bucket, namespace), separators=(",", ":")),
    )
    cli.call(
        "iam",
        "attach-role-policy",
        "--role-name",
        role_name,
        "--policy-arn",
        SSM_CORE_POLICY_ARN,
    )
    code, profile, error = cli.call(
        "iam",
        "get-instance-profile",
        "--instance-profile-name",
        profile_name,
        allow_failure=True,
    )
    if code:
        if "NoSuchEntity" not in error:
            raise RuntimeError(f"unable to inspect worker instance profile: {error}")
        cli.call(
            "iam",
            "create-instance-profile",
            "--instance-profile-name",
            profile_name,
            "--path",
            "/lumi/",
            "--tags",
            "Key=Project,Value=Lumi",
            "Key=Purpose,Value=ProductionCorpusAcquisition",
        )
        profile = {}
    roles = profile.get("InstanceProfile", {}).get("Roles", [])
    if roles and {item.get("RoleName") for item in roles} != {role_name}:
        raise ValueError("existing worker instance profile has another role")
    if not roles:
        cli.call(
            "iam",
            "add-role-to-instance-profile",
            "--instance-profile-name",
            profile_name,
            "--role-name",
            role_name,
        )
    cli.call(
        "iam", "wait", "instance-profile-exists", "--instance-profile-name", profile_name
    )


def _choose_network(cli: AwsCli, instance_type: str) -> tuple[str, str, str]:
    _, vpcs, _ = cli.call(
        "ec2", "describe-vpcs", "--filters", "Name=is-default,Values=true"
    )
    rows = vpcs.get("Vpcs", [])
    if len(rows) != 1:
        raise ValueError("expected exactly one default VPC")
    vpc_id = str(rows[0]["VpcId"])
    _, offerings, _ = cli.call(
        "ec2",
        "describe-instance-type-offerings",
        "--location-type",
        "availability-zone",
        "--filters",
        f"Name=instance-type,Values={instance_type}",
    )
    available_zones = {str(row["Location"]) for row in offerings.get("InstanceTypeOfferings", [])}
    _, subnets, _ = cli.call(
        "ec2",
        "describe-subnets",
        "--filters",
        f"Name=vpc-id,Values={vpc_id}",
        "Name=default-for-az,Values=true",
        "Name=state,Values=available",
    )
    candidates = [
        row
        for row in subnets.get("Subnets", [])
        if row.get("MapPublicIpOnLaunch") is True
        and row.get("AvailabilityZone") in available_zones
        and int(row.get("AvailableIpAddressCount", 0)) > 0
    ]
    if not candidates:
        raise RuntimeError(f"no public default subnet offers {instance_type}")
    selected = sorted(
        candidates,
        key=lambda row: (-int(row["AvailableIpAddressCount"]), row["AvailabilityZone"]),
    )[0]
    return vpc_id, str(selected["SubnetId"]), str(selected["AvailabilityZone"])


def _ensure_security_group(
    cli: AwsCli, *, vpc_id: str, group_name: str, fingerprint: str
) -> str:
    _, existing, _ = cli.call(
        "ec2",
        "describe-security-groups",
        "--filters",
        f"Name=vpc-id,Values={vpc_id}",
        f"Name=group-name,Values={group_name}",
    )
    groups = existing.get("SecurityGroups", [])
    if groups:
        if len(groups) != 1 or groups[0].get("IpPermissions"):
            raise ValueError("existing worker security group is not inbound-closed")
        return str(groups[0]["GroupId"])
    _, created, _ = cli.call(
        "ec2",
        "create-security-group",
        "--group-name",
        group_name,
        "--description",
        "Lumi production acquisition: no inbound access",
        "--vpc-id",
        vpc_id,
        "--tag-specifications",
        json.dumps(
            [
                {
                    "ResourceType": "security-group",
                    "Tags": [
                        {"Key": "Project", "Value": "Lumi"},
                        {"Key": "Purpose", "Value": "ProductionCorpusAcquisition"},
                        {"Key": "Release", "Value": fingerprint[:16]},
                    ],
                }
            ],
            separators=(",", ":"),
        ),
    )
    return str(created["GroupId"])


def _resolve_ami(cli: AwsCli) -> tuple[str, str]:
    _, parameter, _ = cli.call(
        "ssm", "get-parameter", "--name", UBUNTU_AMI_PARAMETER
    )
    image_id = str(parameter.get("Parameter", {}).get("Value", ""))
    if not image_id.startswith("ami-"):
        raise ValueError("official Ubuntu AMI parameter is invalid")
    _, images, _ = cli.call("ec2", "describe-images", "--image-ids", image_id)
    rows = images.get("Images", [])
    if len(rows) != 1:
        raise ValueError("official Ubuntu AMI did not resolve uniquely")
    image = rows[0]
    if (
        image.get("OwnerId") != CANONICAL_UBUNTU_OWNER
        or image.get("Architecture") != "x86_64"
        or image.get("State") != "available"
        or not str(image.get("Name", "")).startswith(
            "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-"
        )
    ):
        raise ValueError("resolved AMI is not the expected Canonical Ubuntu 24.04 image")
    return image_id, str(image["RootDeviceName"])


def _verify_quota(cli: AwsCli, required_vcpus: int) -> float:
    _, quota, _ = cli.call(
        "service-quotas",
        "get-service-quota",
        "--service-code",
        "ec2",
        "--quota-code",
        "L-1216C47A",
    )
    value = float(quota.get("Quota", {}).get("Value", 0))
    if value < required_vcpus:
        raise RuntimeError(
            f"standard on-demand EC2 quota {value:g} is below required {required_vcpus} vCPU"
        )
    return value


def _verify_live_prices(cli: AwsCli, pricing: Mapping[str, Any]) -> dict[str, str]:
    """Fail closed if the official on-demand EC2 or gp3 price moved."""

    ec2 = pricing["ec2"]
    _, ec2_result, _ = cli.call(
        "pricing",
        "get-products",
        "--service-code",
        "AmazonEC2",
        "--filters",
        f"Type=TERM_MATCH,Field=instanceType,Value={ec2['instance_type']}",
        "Type=TERM_MATCH,Field=location,Value=US East (N. Virginia)",
        "Type=TERM_MATCH,Field=operatingSystem,Value=Linux",
        "Type=TERM_MATCH,Field=tenancy,Value=Shared",
        "Type=TERM_MATCH,Field=preInstalledSw,Value=NA",
        "Type=TERM_MATCH,Field=capacitystatus,Value=Used",
        "--max-results",
        "100",
    )
    ec2_products = [json.loads(item) for item in ec2_result.get("PriceList", [])]
    ec2_product = next(
        (item for item in ec2_products if item.get("product", {}).get("sku") == ec2["sku"]),
        None,
    )
    if ec2_product is None:
        raise ValueError("pinned EC2 pricing SKU is no longer present")
    ec2_dimension = (
        ec2_product["terms"]["OnDemand"][f"{ec2['sku']}.JRTCKXETXF"]["priceDimensions"][
            ec2["rate_code"]
        ]
    )
    if Decimal(ec2_dimension["pricePerUnit"]["USD"]) != Decimal(
        ec2["on_demand_usd_per_hour"]
    ):
        raise ValueError("live r7i.4xlarge on-demand price changed")

    ebs = pricing["ebs"]
    _, ebs_result, _ = cli.call(
        "pricing",
        "get-products",
        "--service-code",
        "AmazonEC2",
        "--filters",
        "Type=TERM_MATCH,Field=volumeApiName,Value=gp3",
        "Type=TERM_MATCH,Field=location,Value=US East (N. Virginia)",
        "--max-results",
        "100",
    )
    ebs_products = [json.loads(item) for item in ebs_result.get("PriceList", [])]
    ebs_product = next(
        (item for item in ebs_products if item.get("product", {}).get("sku") == ebs["sku"]),
        None,
    )
    if ebs_product is None:
        raise ValueError("pinned gp3 pricing SKU is no longer present")
    ebs_dimension = (
        ebs_product["terms"]["OnDemand"][f"{ebs['sku']}.JRTCKXETXF"]["priceDimensions"][
            ebs["rate_code"]
        ]
    )
    if Decimal(ebs_dimension["pricePerUnit"]["USD"]) != Decimal(
        ebs["storage_usd_per_gb_month"]
    ):
        raise ValueError("live gp3 storage price changed")
    return {
        "ec2_usd_per_hour": ec2_dimension["pricePerUnit"]["USD"],
        "gp3_usd_per_gb_month": ebs_dimension["pricePerUnit"]["USD"],
    }


def _worker_userdata(plan: Mapping[str, Any]) -> str:
    pricing = plan["pricing"]
    aws_cli = pricing["worker_bootstrap"]["aws_cli"]
    packages = pricing["worker_bootstrap"]["python_packages"]
    keys = plan["artifact_keys"]
    variables = {
        "AWS_REGION": plan["region"],
        "BUCKET": plan["bucket"],
        "NAMESPACE": plan["namespace"],
        "SOURCE_KEY": keys["source_bundle"],
        "SOURCE_SHA256": plan["source_bundle"]["sha256"],
        "RELEASE_KEY": keys["worker_release"],
        "RELEASE_SHA256": plan["worker_release_sha256"],
        "REPORT_KEY": keys["production_report"],
        "REPORT_SHA256": plan["production_report_sha256"],
        "TOKENIZER_KEY": keys["frozen_tokenizer"],
        "TOKENIZER_SHA256": _read_release_hash(plan, "frozen_tokenizer"),
        "EXCLUSION_KEY": keys["training_exclusion_ledger"],
        "EXCLUSION_SHA256": _read_release_hash(plan, "training_exclusion_ledger"),
        "WORKER_FINGERPRINT": plan["worker_release_fingerprint"],
        "AWSCLI_URL": aws_cli["url"],
        "AWSCLI_SHA256": aws_cli["sha256"],
    }
    assignments = "\n".join(f"{name}={shlex.quote(str(value))}" for name, value in variables.items())
    package_spec = " ".join(
        shlex.quote(f"{name}=={version}") for name, version in sorted(packages.items())
    )
    return f"""#!/bin/bash
set -Eeuo pipefail
exec > >(tee -a /var/log/lumi-bootstrap.log) 2>&1
{assignments}
export AWS_DEFAULT_REGION="$AWS_REGION" AWS_PAGER=""
bootstrap_failed=1
cleanup_bootstrap() {{
  if [ "$bootstrap_failed" -ne 0 ]; then
    echo "Lumi bootstrap failed; terminating bounded worker"
    shutdown -h now || true
  fi
}}
trap cleanup_bootstrap EXIT

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl python3-venv unzip zstd
curl --location --fail --silent --show-error --output /tmp/awscliv2.zip "$AWSCLI_URL"
echo "$AWSCLI_SHA256  /tmp/awscliv2.zip" | sha256sum --check --strict
rm -rf /tmp/aws-cli-install
unzip -q /tmp/awscliv2.zip -d /tmp/aws-cli-install
/tmp/aws-cli-install/aws/install --update
/usr/local/bin/aws --version

state_device=$(lsblk --bytes -dpno NAME,TYPE,SIZE | awk '$2 == "disk" && $3 >= 400000000000 {{print $1}}' | head -n 1)
test -n "$state_device"
if ! blkid "$state_device" >/dev/null 2>&1; then
  mkfs.ext4 -F -L lumi-acquisition-state "$state_device"
fi
mkdir -p /var/lib/lumi-acquisition
state_uuid=$(blkid -s UUID -o value "$state_device")
grep -q "$state_uuid" /etc/fstab || echo "UUID=$state_uuid /var/lib/lumi-acquisition ext4 defaults,nofail 0 2" >> /etc/fstab
mountpoint -q /var/lib/lumi-acquisition || mount /var/lib/lumi-acquisition
mkdir -p /opt/lumi/repo /opt/lumi/artifacts /var/log/lumi

/usr/local/bin/aws s3api get-object --bucket "$BUCKET" --key "$SOURCE_KEY" /opt/lumi/source.tar.gz >/dev/null
echo "$SOURCE_SHA256  /opt/lumi/source.tar.gz" | sha256sum --check --strict
tar -xzf /opt/lumi/source.tar.gz -C /opt/lumi/repo
/usr/local/bin/aws s3api get-object --bucket "$BUCKET" --key "$RELEASE_KEY" /opt/lumi/artifacts/worker-release.json >/dev/null
echo "$RELEASE_SHA256  /opt/lumi/artifacts/worker-release.json" | sha256sum --check --strict
/usr/local/bin/aws s3api get-object --bucket "$BUCKET" --key "$REPORT_KEY" /opt/lumi/artifacts/production-report.json >/dev/null
echo "$REPORT_SHA256  /opt/lumi/artifacts/production-report.json" | sha256sum --check --strict
/usr/local/bin/aws s3api get-object --bucket "$BUCKET" --key "$TOKENIZER_KEY" /opt/lumi/artifacts/tokenizer.json >/dev/null
echo "$TOKENIZER_SHA256  /opt/lumi/artifacts/tokenizer.json" | sha256sum --check --strict
/usr/local/bin/aws s3api get-object --bucket "$BUCKET" --key "$EXCLUSION_KEY" /opt/lumi/artifacts/training-exclusion-ledger.jsonl >/dev/null
echo "$EXCLUSION_SHA256  /opt/lumi/artifacts/training-exclusion-ledger.jsonl" | sha256sum --check --strict

python3 -m venv /opt/lumi/venv
/opt/lumi/venv/bin/pip install --disable-pip-version-check --no-cache-dir {package_spec}
/opt/lumi/venv/bin/python - <<'PY'
import importlib.metadata
expected = {json.dumps(packages, sort_keys=True)!r}
import json
expected = json.loads(expected)
for name, version in expected.items():
    actual = importlib.metadata.version(name)
    if actual != version:
        raise SystemExit(f"{{name}} version mismatch: {{actual}} != {{version}}")
PY

checkpoint_prefix="$NAMESPACE/control/checkpoints/"
latest_checkpoint=$(/usr/local/bin/aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$checkpoint_prefix" --query 'reverse(sort_by(Contents,&LastModified))[0].Key' --output text 2>/dev/null || true)
if [ -n "$latest_checkpoint" ] && [ "$latest_checkpoint" != "None" ]; then
  /usr/local/bin/aws s3api get-object --bucket "$BUCKET" --key "$latest_checkpoint" /tmp/state.sqlite3.zst >/dev/null
  checkpoint_name=${{latest_checkpoint##*/}}
  checkpoint_sha=${{checkpoint_name%.sqlite3.zst}}
  echo "$checkpoint_sha  /tmp/state.sqlite3.zst" | sha256sum --check --strict
  zstd -q -d -f /tmp/state.sqlite3.zst -o /var/lib/lumi-acquisition/acquisition-state.sqlite3
fi

cat >/usr/local/sbin/run-lumi-acquisition <<'RUNNER'
#!/bin/bash
set -uo pipefail
export AWS_DEFAULT_REGION="__AWS_REGION__" AWS_PAGER="" PYTHONPATH=/opt/lumi/repo/src TOKENIZERS_PARALLELISM=true
cd /opt/lumi/repo
set +e
/opt/lumi/venv/bin/python scripts/run_production_acquisition.py \
  --state-directory /var/lib/lumi-acquisition \
  --tokenizer /opt/lumi/artifacts/tokenizer.json \
  --training-exclusion-ledger /opt/lumi/artifacts/training-exclusion-ledger.jsonl \
  --production-report /opt/lumi/artifacts/production-report.json \
  --maximum-runtime-seconds 259200 \
  --batch-records 64 >>/var/log/lumi/worker.log 2>&1
worker_status=$?
set -e
if [ -f /var/log/lumi/worker.log ]; then
  log_sha=$(sha256sum /var/log/lumi/worker.log | awk '{{print $1}}')
  /usr/local/bin/aws s3api put-object --bucket "__BUCKET__" \
    --key "__NAMESPACE__/control/worker-logs/$log_sha.log" \
    --body /var/log/lumi/worker.log --if-none-match '*' --server-side-encryption AES256 >/dev/null 2>&1 || true
fi
sync
shutdown -h now
exit "$worker_status"
RUNNER
sed -i "s|__AWS_REGION__|$AWS_REGION|g; s|__BUCKET__|$BUCKET|g; s|__NAMESPACE__|$NAMESPACE|g" /usr/local/sbin/run-lumi-acquisition
chmod 0755 /usr/local/sbin/run-lumi-acquisition

cat >/etc/systemd/system/lumi-production-acquisition.service <<'UNIT'
[Unit]
Description=Lumi bounded production corpus acquisition
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/sbin/run-lumi-acquisition
Restart=no
TimeoutStartSec=infinity
TimeoutStopSec=infinity
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now lumi-production-acquisition.service
bootstrap_failed=0
echo "Lumi production acquisition service launched"
"""


def _read_release_hash(plan: Mapping[str, Any], name: str) -> str:
    key = str(plan["artifact_keys"][name])
    filename = key.rsplit("/", 1)[-1]
    digest = filename.split(".", 1)[0]
    return _sha256(digest, f"{name} hash")


def _tag_specifications(fingerprint: str, source_commit: str) -> str:
    tags = [
        {"Key": "Name", "Value": "lumi-production-acquisition-v0.1"},
        {"Key": "Project", "Value": "Lumi"},
        {"Key": "Purpose", "Value": "ProductionCorpusAcquisition"},
        {"Key": "ManagedBy", "Value": "lumi-model"},
        {"Key": "WorkerRelease", "Value": fingerprint},
        {"Key": "SourceCommit", "Value": source_commit},
        {"Key": "GPUTraining", "Value": "false"},
        {"Key": "AutoTerminate", "Value": "true"},
    ]
    return json.dumps(
        [
            {"ResourceType": "instance", "Tags": tags},
            {"ResourceType": "volume", "Tags": tags},
        ],
        separators=(",", ":"),
    )


def _find_existing_instance(cli: AwsCli, fingerprint: str) -> dict[str, Any] | None:
    _, result, _ = cli.call(
        "ec2",
        "describe-instances",
        "--filters",
        f"Name=tag:WorkerRelease,Values={fingerprint}",
        "Name=instance-state-name,Values=pending,running,stopping,stopped",
    )
    instances = [
        instance
        for reservation in result.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]
    if len(instances) > 1:
        raise RuntimeError("more than one production worker exists for this release")
    return instances[0] if instances else None


def provision_production_worker(
    plan: Mapping[str, Any],
    *,
    worker_release_path: Path,
    production_report_path: Path,
    source_bundle_path: Path,
    output_directory: Path,
    profile: str = "default",
) -> dict[str, Any]:
    """Upload the release and launch exactly one bounded worker."""

    if plan.get("authorization", {}).get("production_adaptive_acquisition") is not True:
        raise ValueError("production acquisition launch is not authorized")
    if any(
        plan.get("authorization", {}).get(name) is not False
        for name in ("full_4b_training", "model_training", "paid_gpu_compute")
    ):
        raise ValueError("worker launch must not authorize model training")
    cli = AwsCli(profile, str(plan["region"]))
    _, identity, _ = cli.call("sts", "get-caller-identity")
    account_id = str(identity["Account"])
    if _sha256_bytes(account_id.encode()) != plan["production_account_id_sha256"]:
        raise ValueError("AWS account differs from the approved production S3 account")

    observed_hashes = {
        "worker_release": _sha256_file(worker_release_path),
        "production_report": _sha256_file(production_report_path),
        "source_bundle": _sha256_file(source_bundle_path),
    }
    expected_hashes = {
        "worker_release": plan["worker_release_sha256"],
        "production_report": plan["production_report_sha256"],
        "source_bundle": plan["source_bundle"]["sha256"],
    }
    if observed_hashes != expected_hashes:
        raise ValueError("worker launch artifacts changed after the launch plan was built")

    bucket = str(plan["bucket"])
    keys = plan["artifact_keys"]
    for key, path in (
        (keys["worker_release"], worker_release_path),
        (keys["production_report"], production_report_path),
        (keys["source_bundle"], source_bundle_path),
    ):
        _put_immutable_object(
            cli,
            bucket=bucket,
            key=str(key),
            path=path,
            account_id=account_id,
        )

    fingerprint = str(plan["worker_release_fingerprint"])
    suffix = fingerprint[:12]
    role_name = f"lumi-acq-{suffix}"
    profile_name = role_name
    security_group_name = f"lumi-acq-{suffix}"
    _ensure_iam(
        cli,
        role_name=role_name,
        profile_name=profile_name,
        bucket=bucket,
        namespace=str(plan["namespace"]),
        fingerprint=fingerprint,
    )
    live_prices = _verify_live_prices(cli, plan["pricing"])
    quota = _verify_quota(cli, int(plan["runtime"]["instance_vcpus"]))
    vpc_id, subnet_id, availability_zone = _choose_network(
        cli, str(plan["runtime"]["instance_type"])
    )
    group_id = _ensure_security_group(
        cli,
        vpc_id=vpc_id,
        group_name=security_group_name,
        fingerprint=fingerprint,
    )
    image_id, root_device = _resolve_ami(cli)
    existing = _find_existing_instance(cli, fingerprint)
    if existing is None:
        output_directory.mkdir(parents=True, exist_ok=True)
        user_data_path = output_directory / "user-data.sh"
        user_data = _worker_userdata(plan)
        user_data_path.write_text(user_data, encoding="utf-8")
        user_data_path.chmod(0o600)
        runtime = plan["runtime"]
        block_devices = [
            {
                "DeviceName": root_device,
                "Ebs": {
                    "DeleteOnTermination": True,
                    "Encrypted": True,
                    "Iops": 3000,
                    "Throughput": 125,
                    "VolumeSize": int(runtime["root_volume_gib"]),
                    "VolumeType": "gp3",
                },
            },
            {
                "DeviceName": "/dev/sdf",
                "Ebs": {
                    "DeleteOnTermination": True,
                    "Encrypted": True,
                    "Iops": 3000,
                    "Throughput": 125,
                    "VolumeSize": int(runtime["state_volume_gib"]),
                    "VolumeType": "gp3",
                },
            },
        ]
        _, launched, _ = cli.call(
            "ec2",
            "run-instances",
            "--image-id",
            image_id,
            "--instance-type",
            str(runtime["instance_type"]),
            "--count",
            "1",
            "--subnet-id",
            subnet_id,
            "--security-group-ids",
            group_id,
            "--associate-public-ip-address",
            "--iam-instance-profile",
            f"Name={profile_name}",
            "--block-device-mappings",
            json.dumps(block_devices, separators=(",", ":")),
            "--metadata-options",
            "HttpEndpoint=enabled,HttpTokens=required,HttpPutResponseHopLimit=1",
            "--instance-initiated-shutdown-behavior",
            "terminate",
            "--user-data",
            f"file://{user_data_path}",
            "--tag-specifications",
            _tag_specifications(fingerprint, str(plan["source_commit"])),
            "--client-token",
            f"lumi-acq-{plan['launch_plan_fingerprint'][:48]}",
        )
        rows = launched.get("Instances", [])
        if len(rows) != 1:
            raise RuntimeError("EC2 did not return exactly one production worker")
        existing = rows[0]
    instance_id = str(existing["InstanceId"])
    cli.call("ec2", "wait", "instance-running", "--instance-ids", instance_id)
    _, described, _ = cli.call("ec2", "describe-instances", "--instance-ids", instance_id)
    instance = described["Reservations"][0]["Instances"][0]
    _, shutdown, _ = cli.call(
        "ec2",
        "describe-instance-attribute",
        "--instance-id",
        instance_id,
        "--attribute",
        "instanceInitiatedShutdownBehavior",
    )
    volume_ids = [
        mapping["Ebs"]["VolumeId"] for mapping in instance.get("BlockDeviceMappings", [])
    ]
    _, volume_result, _ = cli.call("ec2", "describe-volumes", "--volume-ids", *volume_ids)
    volumes = volume_result.get("Volumes", [])
    expected_sizes = {
        int(plan["runtime"]["root_volume_gib"]),
        int(plan["runtime"]["state_volume_gib"]),
    }
    checks = {
        "auto_terminate": shutdown.get("InstanceInitiatedShutdownBehavior", {}).get("Value")
        == "terminate",
        "encrypted_volumes": len(volumes) == 2
        and {int(volume["Size"]) for volume in volumes} == expected_sizes
        and all(volume.get("Encrypted") is True for volume in volumes)
        and all(volume.get("VolumeType") == "gp3" for volume in volumes)
        and all(
            mapping.get("Ebs", {}).get("DeleteOnTermination") is True
            and mapping.get("Ebs", {}).get("Status") == "attached"
            for mapping in instance.get("BlockDeviceMappings", [])
        ),
        "imds_v2": instance.get("MetadataOptions", {}).get("HttpTokens") == "required",
        "instance_type": instance.get("InstanceType") == plan["runtime"]["instance_type"],
        "no_gpu_instance": not str(instance.get("InstanceType", "")).startswith(("p", "g")),
        "on_demand": "InstanceLifecycle" not in instance,
        "running": instance.get("State", {}).get("Name") == "running",
    }
    if not all(checks.values()):
        raise RuntimeError(f"launched production worker failed verification: {checks}")
    report = {
        "account_id_sha256": _sha256_bytes(account_id.encode()),
        "authorization": plan["authorization"],
        "availability_zone": availability_zone,
        "checks": checks,
        "cost_forecast": plan["cost_forecast"],
        "instance_id": instance_id,
        "launch_plan_fingerprint": plan["launch_plan_fingerprint"],
        "launched_at": _utc_now(),
        "live_prices": live_prices,
        "quota_vcpus": quota,
        "region": plan["region"],
        "resources": {
            "iam_instance_profile": profile_name,
            "iam_role": role_name,
            "security_group_id": group_id,
        },
        "schema_version": SCHEMA_VERSION,
        "source_commit": plan["source_commit"],
        "status": "running_bootstrap",
        "worker_release_fingerprint": fingerprint,
    }
    _write_json_atomic(output_directory / "launch-report.json", report, private=True)
    return report


def wait_for_worker_online(
    report: Mapping[str, Any], *, profile: str = "default", timeout_seconds: int = 1200
) -> dict[str, Any]:
    """Wait until SSM can inspect the launched worker without opening an inbound port."""

    cli = AwsCli(profile, str(report["region"]))
    instance_id = str(report["instance_id"])
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        _, result, _ = cli.call(
            "ssm",
            "describe-instance-information",
            "--filters",
            f"Key=InstanceIds,Values={instance_id}",
        )
        rows = result.get("InstanceInformationList", [])
        if rows and rows[0].get("PingStatus") == "Online":
            return {
                "instance_id": instance_id,
                "last_ping": rows[0].get("LastPingDateTime"),
                "platform": rows[0].get("PlatformName"),
                "status": "ssm_online",
            }
        time.sleep(10)
    raise TimeoutError("production worker did not become SSM-online before the deadline")
