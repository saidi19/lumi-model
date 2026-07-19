"""Fail-closed provisioning for Lumi's production corpus S3 namespace."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .aws_s3_canary import AwsCli, _audit_bucket, _deny_insecure_policy, _log_bucket_policy

SCHEMA_VERSION = 1
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
EXPECTED_INPUTS = {
    "aws_provider",
    "cccc_decision",
    "corpus_storage",
    "data_mix",
    "data_quality",
    "pricing_snapshot",
    "source_file_manifest",
    "tokenizer_decision",
}
IMPLEMENTATION_FILES = (
    "src/lumi/aws_s3_production.py",
    "src/lumi/aws_s3_canary.py",
)


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
        raise ValueError(f"expected a JSON object: {path}")
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


def _resolve(root: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path is invalid")
    path = (root / raw_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes the repository")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _money(value: Any, label: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{label} is not a decimal") from error
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _validate_authorization(specification: Mapping[str, Any]) -> None:
    authorization = specification.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("production authorization is missing")
    required_true = {
        "bulk_corpus_materialization",
        "create_production_s3_resources",
        "production_adaptive_acquisition",
        "transfer_source_payload",
    }
    required_false = {"full_4b_training", "model_training", "paid_gpu_compute"}
    if any(authorization.get(name) is not True for name in required_true):
        raise ValueError("production S3 and adaptive acquisition must be explicitly authorized")
    if any(authorization.get(name) is not False for name in required_false):
        raise ValueError("production acquisition must not authorize GPU or model training")
    if authorization.get("owner") != "Afnan Technologies Inc.":
        raise ValueError("unexpected production-authorization owner")
    approved_at = authorization.get("approved_at")
    if not isinstance(approved_at, str) or not approved_at.endswith("Z"):
        raise ValueError("production authorization needs a UTC timestamp")
    if not isinstance(authorization.get("reason"), str) or not authorization["reason"].strip():
        raise ValueError("production authorization needs a reason")


def _validate_provider(specification: Mapping[str, Any]) -> None:
    provider = specification.get("provider")
    expected = {
        "abort_incomplete_multipart_upload_days": 1,
        "data_bucket_prefix": "lumi-corpus-prod",
        "default_encryption": "AES256",
        "log_bucket_prefix": "lumi-corpus-prod-audit",
        "namespace": "lumi-corpus/v0.1",
        "object_ownership": "BucketOwnerEnforced",
        "public_access_block": True,
        "region": "us-east-1",
        "server_access_logging": True,
        "service": "aws_s3",
        "storage_class": "STANDARD",
        "versioning": "Enabled",
    }
    if not isinstance(provider, Mapping):
        raise ValueError("production provider is missing")
    for name, value in expected.items():
        if provider.get(name) != value:
            raise ValueError(f"unsafe or unsupported production provider field: {name}")


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def build_production_acquisition_plan(
    contract_path: Path,
    *,
    repository_root: Path,
    tokenizer_path: Path,
    training_exclusion_ledger: Path,
) -> dict[str, Any]:
    """Validate every approved input and return the exact production plan."""

    contract_path = contract_path.resolve()
    root = repository_root.resolve()
    specification = _read_json(contract_path)
    if specification.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported production-acquisition schema")
    if specification.get("gate") != "production_s3_adaptive_acquisition_v0.1":
        raise ValueError("unexpected production-acquisition gate")
    _validate_authorization(specification)
    _validate_provider(specification)

    configured_inputs = specification.get("inputs")
    if not isinstance(configured_inputs, Mapping) or set(configured_inputs) != EXPECTED_INPUTS:
        raise ValueError("production input set changed")
    paths: dict[str, Path] = {}
    input_hashes: dict[str, dict[str, str]] = {}
    for name in sorted(EXPECTED_INPUTS):
        pin = configured_inputs[name]
        if not isinstance(pin, Mapping):
            raise ValueError(f"invalid production input pin: {name}")
        path = _resolve(root, pin.get("path"), name)
        expected_digest = _sha256(pin.get("sha256"), f"{name} SHA-256")
        observed_digest = _sha256_file(path)
        if observed_digest != expected_digest:
            raise ValueError(f"production input hash changed: {name}")
        paths[name] = path
        input_hashes[name] = {
            "path": str(path.relative_to(root)),
            "sha256": observed_digest,
        }

    external = specification.get("external_artifacts")
    if not isinstance(external, Mapping) or set(external) != {
        "frozen_tokenizer",
        "training_exclusion_ledger",
    }:
        raise ValueError("production external-artifact set changed")
    external_paths = {
        "frozen_tokenizer": tokenizer_path.resolve(),
        "training_exclusion_ledger": training_exclusion_ledger.resolve(),
    }
    external_report = {}
    for name, path in external_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing production artifact: {path}")
        pin = external[name]
        if not isinstance(pin, Mapping):
            raise ValueError(f"invalid external-artifact pin: {name}")
        observed_bytes = path.stat().st_size
        observed_sha256 = _sha256_file(path)
        if observed_bytes != _positive_int(pin.get("bytes"), f"{name} bytes"):
            raise ValueError(f"external artifact byte size changed: {name}")
        if observed_sha256 != _sha256(pin.get("sha256"), f"{name} SHA-256"):
            raise ValueError(f"external artifact hash changed: {name}")
        logical_name = pin.get("logical_name")
        if not isinstance(logical_name, str) or Path(logical_name).name != logical_name:
            raise ValueError(f"external artifact logical name is unsafe: {name}")
        external_report[name] = {
            "bytes": observed_bytes,
            "logical_name": logical_name,
            "sha256": observed_sha256,
        }
    expected_entries = _positive_int(
        external["training_exclusion_ledger"].get("entries"), "exclusion entries"
    )
    if _line_count(training_exclusion_ledger) != expected_entries:
        raise ValueError("training-exclusion ledger entry count changed")
    external_report["training_exclusion_ledger"]["entries"] = expected_entries

    storage = _read_json(paths["corpus_storage"])
    provider = _read_json(paths["aws_provider"])
    cccc = _read_json(paths["cccc_decision"])
    source_manifest = _read_json(paths["source_file_manifest"])
    mixture = _read_json(paths["data_mix"])
    tokenizer = _read_json(paths["tokenizer_decision"])
    limits = specification.get("limits")
    if not isinstance(limits, Mapping):
        raise ValueError("production limits are missing")
    expected_limits = {
        "candidate_hard_maximum_tokens": 150_000_000_000,
        "candidate_total_floor_tokens": 120_000_000_000,
        "final_training_tokens": 80_000_000_000,
        "hard_logical_storage_bytes": 2_000_000_000_000,
        "maximum_category_overshoot_tokens": 100_000_000,
    }
    if dict(limits) != expected_limits:
        raise ValueError("production token or storage limits changed")
    adaptive = storage.get("adaptive_acquisition", {})
    storage_budget = storage.get("storage_budget", {})
    shards = storage.get("shards", {})
    if any(adaptive.get(name) != expected_limits[name] for name in (
        "candidate_hard_maximum_tokens",
        "final_training_tokens",
    )):
        raise ValueError("production limits differ from the storage contract")
    if storage_budget.get("hard_logical_storage_bytes") != expected_limits[
        "hard_logical_storage_bytes"
    ]:
        raise ValueError("production storage cap differs from the storage contract")
    if shards.get("maximum_category_overshoot_tokens") != expected_limits[
        "maximum_category_overshoot_tokens"
    ]:
        raise ValueError("production overshoot differs from the storage contract")

    acquisition = specification.get("acquisition")
    if not isinstance(acquisition, Mapping):
        raise ValueError("production acquisition settings are missing")
    if acquisition.get("raw_source_payload_persistence") is not False:
        raise ValueError("raw source payload persistence must remain disabled")
    if acquisition.get("maximum_concurrent_workers") != 1:
        raise ValueError("v0.1 requires one global-dedup writer")
    if acquisition.get("worker_strategy") != "single_writer_global_dedup_v0.1":
        raise ValueError("unexpected production worker strategy")
    if acquisition.get("clean_shard_target_tokens") != shards.get(
        "clean_candidate_target_tokens"
    ):
        raise ValueError("production clean-shard target differs from the storage contract")
    if acquisition.get("worker_local_scratch_hard_cap_bytes") != shards.get(
        "worker_local_scratch_hard_cap_bytes"
    ):
        raise ValueError("production scratch cap differs from the storage contract")

    if cccc.get("decision_record", {}).get("selected_option") != "include_with_caveat":
        raise ValueError("CCCC is not approved with the required caveat")
    cccc_ack = cccc.get("decision_record", {}).get("acknowledgements", {})
    for name in (
        "accept_source_level_basis_for_lumi_personal_research",
        "maintain_removal_ledger_and_rebuild_affected_shards",
        "preserve_document_and_warc_traceability",
    ):
        if cccc_ack.get(name) is not True:
            raise ValueError(f"CCCC production acknowledgement is missing: {name}")

    tokenizer_pin = external_report["frozen_tokenizer"]
    baseline = tokenizer.get("baseline", {})
    if baseline.get("label") != "current_65k" or baseline.get("vocab_size") != 65_536:
        raise ValueError("the frozen production tokenizer decision changed")
    if baseline.get("tokenizer_json_sha256") != tokenizer_pin["sha256"]:
        raise ValueError("the production tokenizer artifact differs from its decision")

    sources = source_manifest.get("sources")
    inventory = specification.get("source_inventory")
    if not isinstance(sources, list) or not isinstance(inventory, Mapping):
        raise ValueError("production source inventory is missing")
    data_files = [data_file for source in sources for data_file in source.get("data_files", [])]
    if len(sources) != inventory.get("repositories"):
        raise ValueError("production source repository count changed")
    if len(data_files) != inventory.get("data_files"):
        raise ValueError("production source-file count changed")
    if sum(int(row.get("source_bytes", 0)) for row in data_files) != inventory.get(
        "compressed_source_bytes"
    ):
        raise ValueError("production compressed-source byte total changed")
    source_digests = set()
    for row in data_files:
        digest = row.get("source_digest", {})
        if digest.get("algorithm") != "sha256" or digest.get("basis") != "git_lfs_object":
            raise ValueError("a production source file lacks a Git LFS SHA-256 pin")
        source_digests.add(_sha256(digest.get("value"), "source-file SHA-256"))
    if len(source_digests) != len(data_files):
        raise ValueError("production source-file SHA-256 identities are not unique")

    categories = mixture.get("mixture")
    if not isinstance(categories, list) or len(categories) != 7:
        raise ValueError("production mixture must contain seven categories")
    category_rows = []
    for row in categories:
        target = _positive_int(row.get("target_tokens_at_80b"), "category target")
        floor = target * 3 // 2
        category_rows.append(
            {
                "candidate_floor_tokens": floor,
                "category": row["category"],
                "source_components": list(row["source_components"]),
                "target_tokens": target,
            }
        )
    if sum(row["target_tokens"] for row in category_rows) != limits["final_training_tokens"]:
        raise ValueError("production category targets do not total 80B tokens")
    if sum(row["candidate_floor_tokens"] for row in category_rows) != limits[
        "candidate_total_floor_tokens"
    ]:
        raise ValueError("production category floors do not total 120B tokens")

    cost = specification.get("cost_policy")
    provider_cap = provider.get("cost_policy", {}).get(
        "owner_approved_s3_monthly_cost_cap_usd"
    )
    if not isinstance(cost, Mapping):
        raise ValueError("production cost policy is missing")
    s3_cap = _money(cost.get("owner_approved_s3_monthly_cost_cap_usd"), "S3 cap")
    if s3_cap != _money(provider_cap, "provider S3 cap"):
        raise ValueError("production S3 cap differs from the approved provider cap")
    compute_cap = _money(cost.get("initial_paid_cpu_compute_cap_usd"), "CPU compute cap")
    if cost.get("require_new_approval_before_compute_cap_increase") is not True:
        raise ValueError("CPU compute cap increases must require new approval")

    implementation = {}
    for raw_path in IMPLEMENTATION_FILES:
        path = _resolve(root, raw_path, "production implementation")
        implementation[raw_path] = _sha256_file(path)
    contract_sha256 = _sha256_file(contract_path)
    fingerprint_material = {
        "contract_sha256": contract_sha256,
        "external_artifacts": external_report,
        "implementation": implementation,
        "inputs": input_hashes,
        "schema_version": SCHEMA_VERSION,
    }
    plan_fingerprint = _sha256_bytes(_canonical_json_bytes(fingerprint_material))
    return {
        "acquisition": dict(acquisition),
        "authorization": dict(specification["authorization"]),
        "categories": category_rows,
        "cost_policy": {
            "initial_paid_cpu_compute_cap_usd": format(compute_cap, ".2f"),
            "owner_approved_s3_monthly_cost_cap_usd": format(s3_cap, ".2f"),
        },
        "decision": {
            "full_4b_training": "not_authorized",
            "gpu_compute": "not_authorized",
            "production_s3": "authorized",
            "source_payload_transfer": "authorized",
            "status": "ready_for_s3_provisioning",
        },
        "external_artifacts": external_report,
        "gate": specification["gate"],
        "implementation": implementation,
        "inputs": {
            "contract": {
                "path": str(contract_path.relative_to(root)),
                "sha256": contract_sha256,
            },
            **input_hashes,
        },
        "limits": dict(limits),
        "plan_fingerprint": plan_fingerprint,
        "provider": dict(specification["provider"]),
        "schema_version": SCHEMA_VERSION,
        "source_inventory": dict(inventory),
        "version": specification["version"],
    }


def production_bucket_names(account_id: str, plan_fingerprint: str) -> tuple[str, str]:
    """Return stable bucket names without exposing the AWS account ID."""

    if not account_id.isdigit() or len(account_id) != 12:
        raise ValueError("AWS account ID is invalid")
    _sha256(plan_fingerprint, "plan fingerprint")
    suffix = _sha256_bytes(f"{account_id}:{plan_fingerprint}".encode())[:20]
    return f"lumi-corpus-prod-{suffix}", f"lumi-corpus-prod-audit-{suffix}"


def _data_lifecycle(namespace: str) -> dict[str, Any]:
    return {
        "Rules": [
            {
                "ID": "abort-incomplete-corpus-multipart-after-one-day",
                "Status": "Enabled",
                "Filter": {"Prefix": f"{namespace}/"},
                "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
            },
            {
                "ID": "expire-noncurrent-staging-versions-after-thirty-days",
                "Status": "Enabled",
                "Filter": {"Prefix": f"{namespace}/staging/"},
                "NoncurrentVersionExpiration": {"NoncurrentDays": 30},
            },
        ]
    }


def _log_lifecycle() -> dict[str, Any]:
    return {
        "Rules": [
            {
                "ID": "retain-access-logs-for-four-hundred-days",
                "Status": "Enabled",
                "Filter": {"Prefix": "access/"},
                "Expiration": {"Days": 400},
                "NoncurrentVersionExpiration": {"NoncurrentDays": 30},
                "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
            }
        ]
    }


def _production_tags(purpose: str, plan_fingerprint: str) -> dict[str, Any]:
    return {
        "TagSet": [
            {"Key": "Project", "Value": "Lumi"},
            {"Key": "Environment", "Value": "Production"},
            {"Key": "Purpose", "Value": purpose},
            {"Key": "ManagedBy", "Value": "lumi-model"},
            {"Key": "PlanFingerprint", "Value": plan_fingerprint},
            {"Key": "DeleteAfterCanary", "Value": "false"},
        ]
    }


def _configure_bucket(
    cli: AwsCli,
    *,
    bucket: str,
    account_id: str,
    lifecycle: Mapping[str, Any],
    policy: Mapping[str, Any],
    purpose: str,
    plan_fingerprint: str,
) -> None:
    cli.call(
        "s3api",
        "put-public-access-block",
        "--bucket",
        bucket,
        "--public-access-block-configuration",
        '{"BlockPublicAcls":true,"IgnorePublicAcls":true,"BlockPublicPolicy":true,"RestrictPublicBuckets":true}',
        "--expected-bucket-owner",
        account_id,
    )
    cli.call(
        "s3api",
        "put-bucket-ownership-controls",
        "--bucket",
        bucket,
        "--ownership-controls",
        '{"Rules":[{"ObjectOwnership":"BucketOwnerEnforced"}]}',
        "--expected-bucket-owner",
        account_id,
    )
    cli.call(
        "s3api",
        "put-bucket-encryption",
        "--bucket",
        bucket,
        "--server-side-encryption-configuration",
        '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"},"BucketKeyEnabled":false}]}',
        "--expected-bucket-owner",
        account_id,
    )
    cli.call(
        "s3api",
        "put-bucket-versioning",
        "--bucket",
        bucket,
        "--versioning-configuration",
        '{"Status":"Enabled"}',
        "--expected-bucket-owner",
        account_id,
    )
    cli.call(
        "s3api",
        "put-bucket-lifecycle-configuration",
        "--bucket",
        bucket,
        "--lifecycle-configuration",
        json.dumps(lifecycle, separators=(",", ":")),
        "--expected-bucket-owner",
        account_id,
    )
    cli.call(
        "s3api",
        "put-bucket-policy",
        "--bucket",
        bucket,
        "--policy",
        json.dumps(policy, separators=(",", ":")),
        "--expected-bucket-owner",
        account_id,
    )
    cli.call(
        "s3api",
        "put-bucket-tagging",
        "--bucket",
        bucket,
        "--tagging",
        json.dumps(_production_tags(purpose, plan_fingerprint), separators=(",", ":")),
        "--expected-bucket-owner",
        account_id,
    )


def _bucket_exists(cli: AwsCli, bucket: str, account_id: str) -> bool:
    code, _, error = cli.call(
        "s3api",
        "head-bucket",
        "--bucket",
        bucket,
        "--expected-bucket-owner",
        account_id,
        allow_failure=True,
    )
    if code == 0:
        return True
    if "404" in error or "Not Found" in error or "NoSuchBucket" in error:
        return False
    raise RuntimeError(f"unable to check production bucket {bucket}: {error}")


def _bucket_tags(cli: AwsCli, bucket: str, account_id: str) -> dict[str, str]:
    code, output, error = cli.call(
        "s3api",
        "get-bucket-tagging",
        "--bucket",
        bucket,
        "--expected-bucket-owner",
        account_id,
        allow_failure=True,
    )
    if code:
        if "NoSuchTagSet" in error:
            return {}
        raise RuntimeError(f"unable to inspect production bucket tags: {error}")
    return {str(row["Key"]): str(row["Value"]) for row in output.get("TagSet", [])}


def _ensure_bucket(
    cli: AwsCli,
    bucket: str,
    account_id: str,
    plan_fingerprint: str,
    *,
    created_in_state: bool,
) -> bool:
    exists = _bucket_exists(cli, bucket, account_id)
    if not exists:
        cli.call("s3api", "create-bucket", "--bucket", bucket)
        return True
    tags = _bucket_tags(cli, bucket, account_id)
    if tags.get("PlanFingerprint") == plan_fingerprint and tags.get("Project") == "Lumi":
        return created_in_state
    if created_in_state and not tags:
        return True
    raise RuntimeError("refusing to reuse an untagged, foreign, or differently planned bucket")


def _checksum_b64(path: Path) -> str:
    return base64.b64encode(bytes.fromhex(_sha256_file(path))).decode()


def _head_immutable_object(
    cli: AwsCli,
    *,
    bucket: str,
    key: str,
    path: Path,
    account_id: str,
) -> dict[str, Any]:
    _, output, _ = cli.call(
        "s3api",
        "head-object",
        "--bucket",
        bucket,
        "--key",
        key,
        "--checksum-mode",
        "ENABLED",
        "--expected-bucket-owner",
        account_id,
    )
    digest = _sha256_file(path)
    checks = {
        "bytes": output.get("ContentLength") == path.stat().st_size,
        "checksum": output.get("ChecksumSHA256") == _checksum_b64(path),
        "encryption": output.get("ServerSideEncryption") == "AES256",
        "metadata": output.get("Metadata", {}).get("sha256") == digest,
    }
    if not all(checks.values()):
        raise RuntimeError(f"immutable production control object failed verification: {checks}")
    return {"bytes": path.stat().st_size, "key": key, "sha256": digest}


def _put_immutable_object(
    cli: AwsCli,
    *,
    bucket: str,
    key: str,
    path: Path,
    account_id: str,
) -> dict[str, Any]:
    digest = _sha256_file(path)
    code, _, error = cli.call(
        "s3api",
        "put-object",
        "--bucket",
        bucket,
        "--key",
        key,
        "--body",
        str(path),
        "--if-none-match",
        "*",
        "--metadata",
        f"sha256={digest}",
        "--checksum-algorithm",
        "SHA256",
        "--checksum-sha256",
        _checksum_b64(path),
        "--server-side-encryption",
        "AES256",
        "--storage-class",
        "STANDARD",
        "--expected-bucket-owner",
        account_id,
        allow_failure=True,
    )
    if code and not ("PreconditionFailed" in error or "412" in error):
        raise RuntimeError(f"failed immutable production control upload: {error}")
    return _head_immutable_object(
        cli,
        bucket=bucket,
        key=key,
        path=path,
        account_id=account_id,
    )


def _bootstrap_files(
    plan: Mapping[str, Any],
    *,
    repository_root: Path,
    tokenizer_path: Path,
    training_exclusion_ledger: Path,
    contract_path: Path,
    plan_path: Path,
) -> list[tuple[str, Path]]:
    namespace = str(plan["provider"]["namespace"])
    files: list[tuple[str, Path]] = []
    all_inputs = {
        "contract": {
            "path": str(contract_path.relative_to(repository_root)),
            "sha256": _sha256_file(contract_path),
        },
        **plan["inputs"],
    }
    for name, pin in sorted(all_inputs.items()):
        path = _resolve(repository_root, pin["path"], name)
        suffix = path.suffix or ".json"
        key = f"{namespace}/control/bootstrap/contracts/{name}/{pin['sha256']}{suffix}"
        files.append((key, path))
    for name, path in (
        ("frozen_tokenizer", tokenizer_path),
        ("training_exclusion_ledger", training_exclusion_ledger),
    ):
        pin = plan["external_artifacts"][name]
        suffix = Path(pin["logical_name"]).suffix
        key = f"{namespace}/control/bootstrap/artifacts/{name}/{pin['sha256']}{suffix}"
        files.append((key, path))
    files.append(
        (
            f"{namespace}/control/bootstrap/plans/{_sha256_file(plan_path)}.json",
            plan_path,
        )
    )
    return files


def _seal_local(output: Path) -> None:
    entries = []
    for name in ("plan.json", "report.json"):
        path = output / name
        if not path.is_file():
            raise ValueError(f"cannot seal missing production artifact: {name}")
        entries.append(f"{_sha256_file(path)}  {name}")
    (output / "manifest.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")


def provision_production_s3(
    contract_path: Path,
    output: Path,
    *,
    repository_root: Path,
    tokenizer_path: Path,
    training_exclusion_ledger: Path,
    profile: str = "default",
) -> dict[str, Any]:
    """Create or resume the approved private production S3 namespace."""

    root = repository_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = build_production_acquisition_plan(
        contract_path,
        repository_root=root,
        tokenizer_path=tokenizer_path,
        training_exclusion_ledger=training_exclusion_ledger,
    )
    plan_path = output / "plan.json"
    _write_json_atomic(plan_path, plan)
    cli = AwsCli(profile, str(plan["provider"]["region"]))
    _, identity, _ = cli.call("sts", "get-caller-identity")
    account_id = str(identity.get("Account"))
    data_bucket, log_bucket = production_bucket_names(account_id, plan["plan_fingerprint"])
    state_path = output / "state.json"
    if state_path.exists():
        state = _read_json(state_path)
        expected = (plan["plan_fingerprint"], data_bucket, log_bucket, profile)
        observed = (
            state.get("plan_fingerprint"),
            state.get("data_bucket"),
            state.get("log_bucket"),
            state.get("profile"),
        )
        if observed != expected:
            raise ValueError("existing production S3 state belongs to another plan or account")
    else:
        state = {
            "account_id": account_id,
            "data_bucket": data_bucket,
            "data_bucket_created": False,
            "log_bucket": log_bucket,
            "log_bucket_created": False,
            "phase": "initializing",
            "plan_fingerprint": plan["plan_fingerprint"],
            "profile": profile,
            "region": plan["provider"]["region"],
            "schema_version": SCHEMA_VERSION,
            "started_at": _utc_now(),
        }
        _write_json_atomic(state_path, state, private=True)

    state["log_bucket_created"] = _ensure_bucket(
        cli,
        log_bucket,
        account_id,
        plan["plan_fingerprint"],
        created_in_state=bool(state["log_bucket_created"]),
    )
    _write_json_atomic(state_path, state, private=True)
    _configure_bucket(
        cli,
        bucket=log_bucket,
        account_id=account_id,
        lifecycle=_log_lifecycle(),
        policy=_log_bucket_policy(log_bucket, data_bucket, account_id),
        purpose="CorpusAccessAudit",
        plan_fingerprint=plan["plan_fingerprint"],
    )

    state["data_bucket_created"] = _ensure_bucket(
        cli,
        data_bucket,
        account_id,
        plan["plan_fingerprint"],
        created_in_state=bool(state["data_bucket_created"]),
    )
    _write_json_atomic(state_path, state, private=True)
    _configure_bucket(
        cli,
        bucket=data_bucket,
        account_id=account_id,
        lifecycle=_data_lifecycle(str(plan["provider"]["namespace"])),
        policy=_deny_insecure_policy(data_bucket),
        purpose="AdaptiveCorpusAcquisition",
        plan_fingerprint=plan["plan_fingerprint"],
    )
    cli.call(
        "s3api",
        "put-bucket-logging",
        "--bucket",
        data_bucket,
        "--bucket-logging-status",
        json.dumps(
            {"LoggingEnabled": {"TargetBucket": log_bucket, "TargetPrefix": "access/"}},
            separators=(",", ":"),
        ),
        "--expected-bucket-owner",
        account_id,
    )
    security = {
        "data": _audit_bucket(cli, data_bucket, account_id, expected_log_bucket=log_bucket),
        "logs": _audit_bucket(cli, log_bucket, account_id, expected_log_bucket=None),
    }

    state["phase"] = "uploading_control_plane"
    _write_json_atomic(state_path, state, private=True)
    control_objects = []
    for key, path in _bootstrap_files(
        plan,
        repository_root=root,
        tokenizer_path=tokenizer_path.resolve(),
        training_exclusion_ledger=training_exclusion_ledger.resolve(),
        contract_path=contract_path.resolve(),
        plan_path=plan_path,
    ):
        control_objects.append(
            _put_immutable_object(
                cli,
                bucket=data_bucket,
                key=key,
                path=path,
                account_id=account_id,
            )
        )
    report = {
        **plan,
        "bucket_identity": {
            "account_id_sha256": _sha256_bytes(account_id.encode()),
            "data_bucket": data_bucket,
            "log_bucket": log_bucket,
        },
        "control_objects": control_objects,
        "decision": {
            **plan["decision"],
            "status": "production_s3_ready_for_acquisition_worker",
        },
        "provisioned_at": _utc_now(),
        "security_audit": security,
    }
    report_path = output / "report.json"
    _write_json_atomic(report_path, report)
    report_key = (
        f"{plan['provider']['namespace']}/control/bootstrap/reports/"
        f"{_sha256_file(report_path)}.json"
    )
    state["report_object"] = _put_immutable_object(
        cli,
        bucket=data_bucket,
        key=report_key,
        path=report_path,
        account_id=account_id,
    )
    state["phase"] = "complete"
    state["completed_at"] = _utc_now()
    _write_json_atomic(state_path, state, private=True)
    _seal_local(output)
    return verify_production_s3(
        contract_path,
        output,
        repository_root=root,
        tokenizer_path=tokenizer_path,
        training_exclusion_ledger=training_exclusion_ledger,
        profile=profile,
    )


def verify_production_s3(
    contract_path: Path,
    output: Path,
    *,
    repository_root: Path,
    tokenizer_path: Path,
    training_exclusion_ledger: Path,
    profile: str = "default",
) -> dict[str, Any]:
    """Independently revalidate local pins and the live production namespace."""

    plan = build_production_acquisition_plan(
        contract_path,
        repository_root=repository_root,
        tokenizer_path=tokenizer_path,
        training_exclusion_ledger=training_exclusion_ledger,
    )
    manifest_lines = (output / "manifest.sha256").read_text(encoding="utf-8").splitlines()
    if len(manifest_lines) != 2:
        raise ValueError("production manifest must have exactly two entries")
    entries = {}
    for line in manifest_lines:
        digest, name = line.split("  ", 1)
        entries[name] = _sha256(digest, "production manifest digest")
    if set(entries) != {"plan.json", "report.json"}:
        raise ValueError("production manifest has an unexpected artifact set")
    for name, digest in entries.items():
        if _sha256_file(output / name) != digest:
            raise ValueError(f"production local artifact hash mismatch: {name}")
    local_plan = _read_json(output / "plan.json")
    report = _read_json(output / "report.json")
    if local_plan != plan or report.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("production report or plan differs from the approved inputs")

    state = _read_json(output / "state.json")
    cli = AwsCli(profile, str(plan["provider"]["region"]))
    _, identity, _ = cli.call("sts", "get-caller-identity")
    account_id = str(identity.get("Account"))
    expected_buckets = production_bucket_names(account_id, plan["plan_fingerprint"])
    if (state.get("data_bucket"), state.get("log_bucket")) != expected_buckets:
        raise ValueError("production state bucket identity changed")
    data_bucket, log_bucket = expected_buckets
    security = {
        "data": _audit_bucket(cli, data_bucket, account_id, expected_log_bucket=log_bucket),
        "logs": _audit_bucket(cli, log_bucket, account_id, expected_log_bucket=None),
    }
    for bucket, purpose in (
        (data_bucket, "AdaptiveCorpusAcquisition"),
        (log_bucket, "CorpusAccessAudit"),
    ):
        tags = _bucket_tags(cli, bucket, account_id)
        if tags != {
            row["Key"]: row["Value"]
            for row in _production_tags(purpose, plan["plan_fingerprint"])["TagSet"]
        }:
            raise ValueError("production bucket tags differ from the approved plan")
    for item in report.get("control_objects", []):
        key = item.get("key")
        if not isinstance(key, str):
            raise ValueError("production control-object key is invalid")
        _, head, _ = cli.call(
            "s3api",
            "head-object",
            "--bucket",
            data_bucket,
            "--key",
            key,
            "--checksum-mode",
            "ENABLED",
            "--expected-bucket-owner",
            account_id,
        )
        if (
            head.get("ContentLength") != item.get("bytes")
            or head.get("Metadata", {}).get("sha256") != item.get("sha256")
            or head.get("ServerSideEncryption") != "AES256"
        ):
            raise ValueError("production control object changed")
    return {
        "control_objects": len(report["control_objects"]),
        "data_bucket": data_bucket,
        "log_bucket": log_bucket,
        "plan_fingerprint": plan["plan_fingerprint"],
        "security_audit": security,
        "status": "verified",
    }
