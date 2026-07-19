from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lumi.aws_s3_provider import (
    build_aws_s3_provider_report,
    evaluate_aws_s3_provider_plan,
    verify_aws_s3_provider_report,
)
from lumi.corpus_storage import build_corpus_storage_report

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "configs" / "aws-s3-provider-v0.1.json"


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict, dict]:
    return (
        _load("configs/aws-s3-provider-v0.1.json"),
        _load("configs/aws-s3-us-east-1-pricing-v0.1.json"),
        build_corpus_storage_report(
            ROOT / "configs" / "corpus-object-storage-v0.1.json",
            repository_root=ROOT,
        ),
    )


def test_s3_provider_selection_and_conservative_cost_gate_pass() -> None:
    report = build_aws_s3_provider_report(CONTRACT, repository_root=ROOT)
    assert report["provider"]["service"] == "aws_s3"
    assert report["provider"]["region"] == "us-east-1"
    assert report["provider"]["storage_class"] == "STANDARD"
    assert all(report["provider"]["public_access_block"].values())
    assert report["base_plan"]["minimum_candidate_tokens"] == 120_000_000_000
    assert report["cost_projection"] == {
        "approved_monthly_cost_cap_usd": "100.00",
        "conservative_internet_egress_bytes": 20_000_000_000,
        "conservative_internet_egress_cost_usd": "1.800000",
        "full_peak_internet_egress_cost_usd": "93.375000",
        "get_and_other_request_cost_usd": "0.040000",
        "peak_logical_gb": "1037.500",
        "peak_standard_storage_cost_usd_per_month": "23.862500",
        "persistent_logical_gb": "830.000",
        "persistent_standard_storage_cost_usd_per_month": "19.090000",
        "projected_conservative_total_usd_per_month": "25.802500",
        "put_copy_post_list_request_cost_usd": "0.100000",
        "same_region_aws_transfer_cost_usd_per_gb": "0.000000",
        "unused_monthly_cap_usd": "74.197500",
    }
    assert report["decision"] == {
        "bounded_s3_io_canary": "authorized_not_run",
        "bulk_corpus_materialization": "not_authorized",
        "full_4b_training": "not_authorized",
        "monthly_cost_gate": "pass",
        "next_action": "authenticate_and_run_bounded_s3_io_canary",
        "paid_compute": "not_authorized",
        "pricing_snapshot": "pass",
        "provider_selection": "pass",
        "provider_readiness": "blocked_canary_not_run",
    }
    assert float(report["canary"]["estimated_maximum_cost_usd"]) < 0.01


@pytest.mark.parametrize(
    "field",
    [
        "bulk_corpus_materialization",
        "full_4b_training",
        "paid_compute",
        "transfer_bulk_source_data",
    ],
)
def test_s3_canary_authorization_cannot_expand_to_bulk_or_training(field: str) -> None:
    specification, pricing, base = _inputs()
    specification["authorization"][field] = True
    with pytest.raises(ValueError, match="must not authorize"):
        evaluate_aws_s3_provider_plan(specification, pricing, base)


def test_s3_security_region_and_price_snapshot_fail_closed() -> None:
    specification, pricing, base = _inputs()
    unsafe = copy.deepcopy(specification)
    unsafe["provider"]["public_access_block"]["block_public_policy"] = False
    with pytest.raises(ValueError, match="must be enabled"):
        evaluate_aws_s3_provider_plan(unsafe, pricing, base)

    wrong_region = copy.deepcopy(specification)
    wrong_region["provider"]["region"] = "us-west-2"
    with pytest.raises(ValueError, match="provider field: region"):
        evaluate_aws_s3_provider_plan(wrong_region, pricing, base)

    changed_price = copy.deepcopy(pricing)
    changed_price["prices"]["standard_storage_first_50tb_usd_per_gb_month"] = "0.01"
    with pytest.raises(ValueError, match="differs from the reviewed rates"):
        evaluate_aws_s3_provider_plan(specification, changed_price, base)


def test_s3_cost_and_canary_caps_fail_closed() -> None:
    specification, pricing, base = _inputs()
    low_cap = copy.deepcopy(specification)
    low_cap["cost_policy"]["owner_approved_s3_monthly_cost_cap_usd"] = "25.00"
    with pytest.raises(ValueError, match="exceeds the approved cap"):
        evaluate_aws_s3_provider_plan(low_cap, pricing, base)

    oversized = copy.deepcopy(specification)
    oversized["canary"]["multipart_payload_bytes"] = 40 * 1024 * 1024
    with pytest.raises(ValueError, match="payload exceeds"):
        evaluate_aws_s3_provider_plan(oversized, pricing, base)

    retained = copy.deepcopy(specification)
    retained["canary"]["retain_bucket_after_success"] = True
    with pytest.raises(ValueError, match="must not retain"):
        evaluate_aws_s3_provider_plan(retained, pricing, base)


def test_s3_provider_report_reproduces_and_detects_tampering(tmp_path: Path) -> None:
    report = build_aws_s3_provider_report(CONTRACT, repository_root=ROOT)
    report_path = tmp_path / "report.json"
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    report_path.write_text(encoded, encoding="utf-8")
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    manifest_path = tmp_path / "manifest.sha256"
    manifest_path.write_text(f"{digest}  report.json\n", encoding="utf-8")
    assert (
        verify_aws_s3_provider_report(
            CONTRACT,
            report_path,
            manifest_path,
            repository_root=ROOT,
        )
        == report
    )
    report_path.write_text(encoded + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        verify_aws_s3_provider_report(
            CONTRACT,
            report_path,
            manifest_path,
            repository_root=ROOT,
        )
