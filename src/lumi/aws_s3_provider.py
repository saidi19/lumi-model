"""Fail-closed AWS S3 pricing and bounded-canary planning for Lumi."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .corpus_storage import build_corpus_storage_report

SCHEMA_VERSION = 1
DECIMAL_GB = Decimal(1_000_000_000)
IMPLEMENTATION_FILES = (
    "src/lumi/aws_s3_provider.py",
    "src/lumi/corpus_capacity.py",
    "src/lumi/corpus_storage.py",
)
FORBIDDEN_AUTHORIZATIONS = {
    "bulk_corpus_materialization",
    "full_4b_training",
    "paid_compute",
    "transfer_bulk_source_data",
}
EXPECTED_PRICES = {
    "data_transfer_in_from_internet_usd_per_gb": Decimal("0"),
    "data_transfer_out_to_internet_first_10tb_usd_per_gb": Decimal("0.09"),
    "get_and_other_requests_usd_per_request": Decimal("0.0000004"),
    "put_copy_post_list_usd_per_request": Decimal("0.000005"),
    "same_region_aws_transfer_usd_per_gb": Decimal("0"),
    "standard_storage_first_50tb_usd_per_gb_month": Decimal("0.023"),
}
EXPECTED_PRODUCTS = {
    "data_transfer_in": "RX66S3KERGV4YBS4",
    "data_transfer_out_to_internet": "HQEH3ZWJVT46JHRG",
    "request_tier_1": "E9YHNFENF4XQBZR6",
    "request_tier_2": "ZWQ6Q48CRJXX4FXE",
    "standard_storage": "WP9ANXZGBYYSGJEA",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path is invalid")
    path = (root / raw_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes the repository")
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    return path


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _money(value: Any, label: str, *, allow_zero: bool = True) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{label} is not a decimal") from error
    if not parsed.is_finite() or parsed < 0 or (not allow_zero and parsed == 0):
        raise ValueError(f"{label} is outside the allowed range")
    return parsed


def _decimal_text(value: Decimal, places: str = "0.000000") -> str:
    return format(value.quantize(Decimal(places)), "f")


def _validate_pricing_snapshot(snapshot: dict[str, Any]) -> dict[str, Decimal]:
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported AWS S3 pricing snapshot schema")
    if snapshot.get("gate") != "aws_s3_us_east_1_pricing_snapshot_v0.1":
        raise ValueError("unexpected AWS S3 pricing gate")
    if snapshot.get("region") != "us-east-1" or snapshot.get("currency") != "USD":
        raise ValueError("the pricing snapshot must be USD for us-east-1")
    for offer_name in ("s3_offer", "data_transfer_offer"):
        offer = snapshot.get(offer_name)
        if not isinstance(offer, dict):
            raise ValueError(f"missing {offer_name}")
        if not isinstance(offer.get("version"), str) or not offer["version"].isdigit():
            raise ValueError(f"{offer_name} has an invalid version")
        if not isinstance(offer.get("publication_date"), str):
            raise ValueError(f"{offer_name} has no publication date")
        digest = offer.get("index_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{offer_name} has an invalid source hash")
        if not str(offer.get("index_url", "")).startswith(
            "https://pricing.us-east-1.amazonaws.com/"
        ):
            raise ValueError(f"{offer_name} must use the official AWS Price List endpoint")
    products = snapshot.get("products")
    if not isinstance(products, dict):
        raise ValueError("pricing products are missing")
    for name, expected_sku in EXPECTED_PRODUCTS.items():
        product = products.get(name)
        if not isinstance(product, dict) or product.get("sku") != expected_sku:
            raise ValueError(f"AWS pricing SKU changed: {name}")
    configured = snapshot.get("prices")
    if not isinstance(configured, dict) or set(configured) != set(EXPECTED_PRICES):
        raise ValueError("AWS pricing fields changed")
    parsed = {name: _money(configured[name], name) for name in EXPECTED_PRICES}
    if parsed != EXPECTED_PRICES:
        raise ValueError("AWS S3 price snapshot differs from the reviewed rates")
    return parsed


def evaluate_aws_s3_provider_plan(
    specification: dict[str, Any],
    pricing_snapshot: dict[str, Any],
    base_report: dict[str, Any],
) -> dict[str, Any]:
    """Validate the selected provider and return an exact bounded cost plan."""

    if specification.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported AWS S3 provider schema")
    if specification.get("gate") != "aws_s3_provider_and_bounded_io_canary_v0.1":
        raise ValueError("unexpected AWS S3 provider gate")
    authorization = specification.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("AWS S3 authorization is missing")
    if authorization.get("local_planning") is not True:
        raise ValueError("local AWS S3 planning must be authorized")
    if authorization.get("create_bounded_canary_resources") is not True:
        raise ValueError("bounded S3 canary resource creation must be explicit")
    if any(authorization.get(name) is not False for name in FORBIDDEN_AUTHORIZATIONS):
        raise ValueError("AWS S3 canary must not authorize bulk data, compute, or training")
    max_payload = _positive_int(
        authorization.get("bounded_canary_max_payload_bytes"), "canary payload cap"
    )
    if max_payload > 64 * 1024 * 1024:
        raise ValueError("bounded S3 canary payload cap exceeds 64 MiB")

    provider = specification.get("provider")
    if not isinstance(provider, dict):
        raise ValueError("AWS S3 provider configuration is missing")
    required_provider = {
        "default_encryption": "AES256",
        "deny_insecure_transport": True,
        "expected_bucket_owner_required": True,
        "object_ownership": "BucketOwnerEnforced",
        "region": "us-east-1",
        "server_access_logging": True,
        "service": "aws_s3",
        "storage_class": "STANDARD",
        "versioning": "Enabled",
    }
    for name, expected in required_provider.items():
        if provider.get(name) != expected:
            raise ValueError(f"unsafe or unsupported AWS S3 provider field: {name}")
    public = provider.get("public_access_block")
    if not isinstance(public, dict) or set(public) != {
        "block_public_acls",
        "block_public_policy",
        "ignore_public_acls",
        "restrict_public_buckets",
    }:
        raise ValueError("AWS S3 public-access block configuration is incomplete")
    if not all(value is True for value in public.values()):
        raise ValueError("all AWS S3 public-access block settings must be enabled")
    for name in ("data_bucket_prefix", "log_bucket_prefix"):
        prefix = provider.get(name)
        if not isinstance(prefix, str) or not prefix.startswith("lumi-corpus-"):
            raise ValueError(f"unsafe AWS S3 bucket prefix: {name}")

    prices = _validate_pricing_snapshot(pricing_snapshot)
    forecast = base_report["storage_forecast"]
    if base_report["summary"]["minimum_candidate_tokens"] != 120_000_000_000:
        raise ValueError("AWS provider overlay requires the sealed 120B candidate floor plan")
    if forecast["forecast_logical_bytes"] != 1_037_500_000_000:
        raise ValueError("AWS provider overlay requires the sealed storage forecast")
    if any(
        base_report["authorization"].get(name) is not False
        for name in {
            "bulk_corpus_materialization",
            "create_cloud_resources",
            "full_4b_training",
            "paid_compute",
            "transfer_bulk_source_data",
        }
    ):
        raise ValueError("base storage plan has an unsafe authorization")

    cost_policy = specification.get("cost_policy")
    if not isinstance(cost_policy, dict):
        raise ValueError("AWS S3 cost policy is missing")
    if cost_policy.get("pricing_uses_global_free_tier") is not False:
        raise ValueError("cost projection must not depend on the global free tier")
    monthly_cap = _money(
        cost_policy.get("owner_approved_s3_monthly_cost_cap_usd"),
        "owner-approved S3 monthly cost cap",
        allow_zero=False,
    )
    tier1_requests = _positive_int(
        cost_policy.get("estimated_put_copy_post_list_requests_per_month"),
        "monthly tier-1 requests",
    )
    tier2_requests = _positive_int(
        cost_policy.get("estimated_get_and_other_requests_per_month"),
        "monthly tier-2 requests",
    )
    egress_bytes = _positive_int(
        cost_policy.get("conservative_internet_egress_bytes_per_month"),
        "monthly conservative internet egress",
    )
    peak_gb = Decimal(forecast["forecast_logical_bytes"]) / DECIMAL_GB
    persistent_gb = Decimal(forecast["persistent_logical_bytes"]) / DECIMAL_GB
    storage_rate = prices["standard_storage_first_50tb_usd_per_gb_month"]
    peak_storage_cost = peak_gb * storage_rate
    persistent_storage_cost = persistent_gb * storage_rate
    tier1_cost = Decimal(tier1_requests) * prices["put_copy_post_list_usd_per_request"]
    tier2_cost = Decimal(tier2_requests) * prices["get_and_other_requests_usd_per_request"]
    egress_cost = (
        Decimal(egress_bytes)
        / DECIMAL_GB
        * prices["data_transfer_out_to_internet_first_10tb_usd_per_gb"]
    )
    projected_monthly_cost = peak_storage_cost + tier1_cost + tier2_cost + egress_cost
    if projected_monthly_cost > monthly_cap:
        raise ValueError("conservative AWS S3 monthly projection exceeds the approved cap")

    canary = specification.get("canary")
    if not isinstance(canary, dict):
        raise ValueError("bounded S3 canary configuration is missing")
    conditional_bytes = _positive_int(canary.get("conditional_put_bytes"), "conditional PUT bytes")
    multipart_bytes = _positive_int(canary.get("multipart_payload_bytes"), "multipart bytes")
    part_bytes = _positive_int(canary.get("multipart_part_bytes"), "multipart part bytes")
    if conditional_bytes + multipart_bytes > max_payload:
        raise ValueError("configured S3 canary payload exceeds its authorization")
    if part_bytes < 5 * 1024 * 1024 or multipart_bytes <= part_bytes:
        raise ValueError("S3 multipart canary does not exercise at least two valid parts")
    range_start = _positive_int(canary.get("range_start_byte"), "range start")
    range_end = _positive_int(canary.get("range_end_byte"), "range end")
    if range_end < range_start or range_end >= multipart_bytes:
        raise ValueError("S3 canary byte range is invalid")
    if canary.get("cleanup_required") is not True:
        raise ValueError("S3 canary cleanup must be mandatory")
    if canary.get("retain_bucket_after_success") is not False:
        raise ValueError("bounded S3 canary must not retain buckets after success")
    canary_cap = _money(
        canary.get("maximum_canary_cost_usd"), "maximum S3 canary cost", allow_zero=False
    )
    canary_gb = Decimal(max_payload) / DECIMAL_GB
    canary_cost = (
        canary_gb * storage_rate / Decimal(30)
        + Decimal(100) * prices["put_copy_post_list_usd_per_request"]
        + Decimal(100) * prices["get_and_other_requests_usd_per_request"]
        + canary_gb * prices["data_transfer_out_to_internet_first_10tb_usd_per_gb"]
    )
    if canary_cost > canary_cap:
        raise ValueError("bounded S3 canary cost estimate exceeds its cap")

    full_peak_egress_cost = peak_gb * prices["data_transfer_out_to_internet_first_10tb_usd_per_gb"]
    return {
        "authorization": dict(authorization),
        "base_plan": {
            "final_training_tokens": base_report["summary"]["final_training_tokens"],
            "minimum_candidate_tokens": base_report["summary"]["minimum_candidate_tokens"],
            "plan_fingerprint": base_report["plan_fingerprint"],
            "raw_source_payload_bytes": forecast["raw_source_payload_bytes"],
        },
        "canary": {
            **canary,
            "authorized_payload_bytes": max_payload,
            "estimated_maximum_cost_usd": _decimal_text(canary_cost),
        },
        "cost_projection": {
            "approved_monthly_cost_cap_usd": _decimal_text(monthly_cap, "0.00"),
            "conservative_internet_egress_bytes": egress_bytes,
            "conservative_internet_egress_cost_usd": _decimal_text(egress_cost),
            "full_peak_internet_egress_cost_usd": _decimal_text(full_peak_egress_cost),
            "get_and_other_request_cost_usd": _decimal_text(tier2_cost),
            "peak_logical_gb": _decimal_text(peak_gb, "0.000"),
            "peak_standard_storage_cost_usd_per_month": _decimal_text(peak_storage_cost),
            "persistent_logical_gb": _decimal_text(persistent_gb, "0.000"),
            "persistent_standard_storage_cost_usd_per_month": _decimal_text(
                persistent_storage_cost
            ),
            "projected_conservative_total_usd_per_month": _decimal_text(projected_monthly_cost),
            "put_copy_post_list_request_cost_usd": _decimal_text(tier1_cost),
            "same_region_aws_transfer_cost_usd_per_gb": _decimal_text(
                prices["same_region_aws_transfer_usd_per_gb"]
            ),
            "unused_monthly_cap_usd": _decimal_text(monthly_cap - projected_monthly_cost),
        },
        "decision": {
            "bounded_s3_io_canary": "authorized_not_run",
            "bulk_corpus_materialization": "not_authorized",
            "full_4b_training": "not_authorized",
            "monthly_cost_gate": "pass",
            "next_action": "authenticate_and_run_bounded_s3_io_canary",
            "paid_compute": "not_authorized",
            "pricing_snapshot": "pass",
            "provider_selection": "pass",
            "provider_readiness": "blocked_canary_not_run",
        },
        "gate": specification["gate"],
        "pricing": {
            "currency": pricing_snapshot["currency"],
            "data_transfer_offer": dict(pricing_snapshot["data_transfer_offer"]),
            "effective_dates": dict(pricing_snapshot["effective_dates"]),
            "prices": dict(pricing_snapshot["prices"]),
            "s3_offer": dict(pricing_snapshot["s3_offer"]),
        },
        "provider": dict(provider),
        "remaining_acquisition_blockers": [
            "private_bucket_policy_and_encryption_audit",
            "full_source_file_name_size_and_revision_manifest",
            "owner_decision_for_cccc_source_level_license_basis",
            "bounded_object_store_put_get_resume_delete_canary",
        ],
        "schema_version": SCHEMA_VERSION,
        "version": specification["version"],
    }


def build_aws_s3_provider_report(
    contract_path: Path, *, repository_root: Path | None = None
) -> dict[str, Any]:
    """Build a provider report bound to the neutral plan, pricing, and code."""

    contract_path = contract_path.resolve()
    root = repository_root.resolve() if repository_root else contract_path.parents[1]
    specification = _read_json(contract_path)
    inputs = specification.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "base_storage_contract",
        "pricing_snapshot",
    }:
        raise ValueError("AWS S3 provider inputs changed")
    base_contract = _resolve(root, inputs["base_storage_contract"], "base storage contract")
    pricing_path = _resolve(root, inputs["pricing_snapshot"], "pricing snapshot")
    base_report = build_corpus_storage_report(base_contract, repository_root=root)
    report = evaluate_aws_s3_provider_plan(
        specification,
        _read_json(pricing_path),
        base_report,
    )
    report["inputs"] = {
        "base_storage_contract": {
            "path": base_contract.relative_to(root).as_posix(),
            "sha256": _sha256_file(base_contract),
        },
        "contract": {
            "path": contract_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(contract_path),
        },
        "pricing_snapshot": {
            "path": pricing_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(pricing_path),
        },
    }
    report["implementation"] = {path: _sha256_file(root / path) for path in IMPLEMENTATION_FILES}
    report["plan_fingerprint"] = _sha256_bytes(
        _canonical_json_bytes(
            {
                "authorization": report["authorization"],
                "base_plan": report["base_plan"],
                "canary": report["canary"],
                "cost_projection": report["cost_projection"],
                "implementation": report["implementation"],
                "inputs": report["inputs"],
                "pricing": report["pricing"],
                "provider": report["provider"],
            }
        )
    )
    return report


def verify_aws_s3_provider_report(
    contract_path: Path,
    report_path: Path,
    manifest_path: Path,
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Verify a sealed provider plan and independently reproduce its contents."""

    report_path = report_path.resolve()
    manifest_path = manifest_path.resolve()
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ValueError("AWS S3 provider manifest must have exactly one entry")
    parts = lines[0].split()
    if len(parts) != 2 or parts[1] != report_path.name:
        raise ValueError("AWS S3 provider manifest entry is invalid")
    if _sha256_file(report_path) != parts[0]:
        raise ValueError("AWS S3 provider report hash does not match its manifest")
    observed = _read_json(report_path)
    expected = build_aws_s3_provider_report(
        contract_path,
        repository_root=repository_root,
    )
    if observed != expected:
        raise ValueError("AWS S3 provider report does not reproduce")
    return observed


def verify_public_price_indexes(
    pricing_snapshot_path: Path,
    s3_index_path: Path,
    data_transfer_index_path: Path,
) -> dict[str, str]:
    """Verify the downloaded public AWS Price List files behind the curated snapshot."""

    snapshot = _read_json(pricing_snapshot_path.resolve())
    _validate_pricing_snapshot(snapshot)
    indexes = {
        "s3_offer": (s3_index_path.resolve(), "AmazonS3"),
        "data_transfer_offer": (data_transfer_index_path.resolve(), "AWSDataTransfer"),
    }
    result: dict[str, str] = {}
    for name, (path, offer_code) in indexes.items():
        expected = snapshot[name]
        if _sha256_file(path) != expected["index_sha256"]:
            raise ValueError(f"{name} public price-list hash changed")
        index = _read_json(path)
        if index.get("offerCode") != offer_code:
            raise ValueError(f"{name} public price-list offer changed")
        if index.get("version") != expected["version"]:
            raise ValueError(f"{name} public price-list version changed")
        if index.get("publicationDate") != expected["publication_date"]:
            raise ValueError(f"{name} public price-list publication date changed")
        result[name] = expected["index_sha256"]
    return result
