from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from lumi.aws_production_worker import (
    _iam_policy,
    _worker_userdata,
    build_worker_launch_plan,
)


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixtures(tmp_path: Path):
    worker_fingerprint = "a" * 64
    tokenizer_sha256 = "b" * 64
    exclusion_sha256 = "c" * 64
    release = {
        "cost_forecast": {
            "attached_ebs_72h_usd": "4.1819",
            "ec2_72h_usd": "76.2048",
            "maximum_72h_usd": "80.3867",
        },
        "decision": {
            "full_4b_training": "not_authorized",
            "gpu_compute": "not_authorized",
            "status": "ready_to_launch_acquisition_worker",
        },
        "external_artifacts": {
            "frozen_tokenizer": {"sha256": tokenizer_sha256},
            "training_exclusion_ledger": {"sha256": exclusion_sha256},
        },
        "gate": "production_corpus_worker_release_v0.1",
        "production_plan_fingerprint": "d" * 64,
        "runtime": {
            "instance_initiated_shutdown_behavior": "terminate",
            "instance_memory_gib": 128,
            "instance_type": "r7i.4xlarge",
            "instance_vcpus": 16,
            "market": "on_demand",
            "maximum_runtime_hours": 72,
            "metadata_http_tokens": "required",
            "on_demand_usd_per_hour": "1.0584",
            "root_volume_gib": 30,
            "root_volume_type": "gp3",
            "state_volume_delete_on_termination": True,
            "state_volume_gib": 500,
            "state_volume_type": "gp3",
        },
        "schema_version": 1,
        "worker_release_fingerprint": worker_fingerprint,
    }
    production = {
        "bucket_identity": {
            "account_id_sha256": "e" * 64,
            "data_bucket": "fixture-production-bucket",
        },
        "decision": {"status": "production_s3_ready_for_acquisition_worker"},
        "plan_fingerprint": "d" * 64,
        "provider": {"namespace": "lumi-corpus/v0.1", "region": "us-east-1"},
    }
    pricing = {
        "ec2": {"instance_type": "r7i.4xlarge"},
        "gate": "aws_production_worker_pricing_snapshot_v0.1",
        "worker_bootstrap": {
            "aws_cli": {
                "sha256": "f" * 64,
                "url": "https://awscli.amazonaws.com/awscli-fixture.zip",
            },
            "python_packages": {
                "datasketch": "2.0.0",
                "lingua-language-detector": "2.2.0",
                "pyarrow": "25.0.0",
                "tokenizers": "0.22.2",
                "zstandard": "0.25.0",
            },
        },
    }
    release_path = _write_json(tmp_path / "release.json", release)
    production_path = _write_json(tmp_path / "production.json", production)
    pricing_path = _write_json(tmp_path / "pricing.json", pricing)
    bundle_path = tmp_path / "source.tar.gz"
    bundle_path.write_bytes(b"fixture source archive")
    return release_path, production_path, pricing_path, bundle_path


def test_launch_plan_and_userdata_preserve_bounded_cpu_worker(tmp_path: Path) -> None:
    release, production, pricing, bundle = _fixtures(tmp_path)
    plan = build_worker_launch_plan(
        release,
        production_report_path=production,
        pricing_path=pricing,
        source_bundle_path=bundle,
        source_commit="1" * 40,
    )
    assert plan["authorization"]["paid_gpu_compute"] is False
    assert plan["cost_forecast"]["maximum_72h_usd"] == "80.3867"
    assert plan["source_bundle"]["sha256"] == hashlib.sha256(bundle.read_bytes()).hexdigest()
    assert plan["runtime"]["instance_type"] == "r7i.4xlarge"
    assert plan["runtime"]["instance_initiated_shutdown_behavior"] == "terminate"

    user_data = _worker_userdata(plan)
    script = tmp_path / "user-data.sh"
    script.write_text(user_data, encoding="utf-8")
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert "--maximum-runtime-seconds 259200" in user_data
    assert "HttpTokens" not in user_data
    assert "shutdown -h now" in user_data
    assert "training-exclusion-ledger" in user_data
    assert "torch" not in user_data.lower()


def test_launch_plan_rejects_training_boundary_change(tmp_path: Path) -> None:
    release_path, production, pricing, bundle = _fixtures(tmp_path)
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["decision"]["gpu_compute"] = "authorized"
    _write_json(release_path, release)
    with pytest.raises(ValueError, match="no-training boundary"):
        build_worker_launch_plan(
            release_path,
            production_report_path=production,
            pricing_path=pricing,
            source_bundle_path=bundle,
            source_commit="1" * 40,
        )


def test_worker_iam_policy_has_no_delete_or_training_permissions() -> None:
    policy = _iam_policy("fixture-production-bucket", "lumi-corpus/v0.1")
    rendered = json.dumps(policy, sort_keys=True)
    assert "s3:Delete" not in rendered
    assert "ec2:" not in rendered
    assert "s3:PutObject" in rendered
    assert "lumi-corpus/v0.1/control/checkpoints/*" in rendered
