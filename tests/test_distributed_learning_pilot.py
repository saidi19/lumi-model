from __future__ import annotations

import copy
import hashlib
import json
import sys
from array import array
from pathlib import Path

import pytest
import torch

import lumi.distributed_learning_pilot as learning_pilot
from lumi.distributed_export import validate_distributed_hf_export
from lumi.distributed_learning_pilot import (
    _validate_contract,
    prepare_learning_pilot_fixture,
    validate_owner_authorization,
    write_owner_authorization,
)
from lumi.distributed_training import (
    PAID_LEARNING_PILOT_MAX_STEPS,
    PAID_LEARNING_PILOT_MAX_TOKENS,
    PAID_LEARNING_PILOT_OWNER_ACKNOWLEDGEMENT,
    PAID_LEARNING_PILOT_RUN_KIND,
    DistributedContext,
    _validate_paid_authorization,
)
from lumi.pilot_training import _fingerprint

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "4b-learning-pilot-v0.1.json"


def _authorization_plan(world_size: int) -> dict:
    plan = {
        "authorization_template": {
            "approved_through_step": PAID_LEARNING_PILOT_MAX_STEPS,
            "approved_token_cap": PAID_LEARNING_PILOT_MAX_TOKENS,
            "full_4b_training": False,
            "owner_acknowledgement": PAID_LEARNING_PILOT_OWNER_ACKNOWLEDGEMENT,
            "paid_gpu": False,
            "run_fingerprint": "a" * 64,
            "run_kind": PAID_LEARNING_PILOT_RUN_KIND,
            "schema_version": 1,
            "source_commit": "b" * 40,
            "status": "planning_only",
            "strategy": "fsdp_full_shard",
            "world_size": world_size,
        },
        "selected_topology": {
            "gradient_accumulation_steps": 64 if world_size == 4 else 32,
            "world_size": world_size,
        },
        "status": "planning_only",
    }
    plan["plan_sha256"] = _fingerprint(plan)
    return plan


def _write_plan(tmp_path: Path, world_size: int) -> Path:
    path = tmp_path / f"plan-{world_size}.json"
    path.write_text(json.dumps(_authorization_plan(world_size)), encoding="utf-8")
    return path


def _uint16_bytes(values: list[int]) -> bytes:
    payload = array("H", values)
    if sys.byteorder != "little":
        payload.byteswap()
    return payload.tobytes()


def test_learning_pilot_contract_keeps_equal_4_and_8_gpu_token_geometry() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    run = _validate_contract(contract)

    assert run["global_sequences_per_step"] == 256
    assert {
        topology["world_size"]
        * run["per_rank_batch_size"]
        * topology["gradient_accumulation_steps"]
        for topology in run["topologies"]
    } == {256}
    assert (
        run["global_sequences_per_step"]
        * contract["corpus"]["sequence_length"]
        * run["resume_through_step"]
        == PAID_LEARNING_PILOT_MAX_TOKENS
    )

    changed = copy.deepcopy(contract)
    changed["run"]["topologies"][0]["gradient_accumulation_steps"] = 32
    with pytest.raises(ValueError, match="topology geometry"):
        _validate_contract(changed)


def test_learning_pilot_setup_is_pinned_to_80gb_h100() -> None:
    script = (ROOT / "scripts" / "setup_4b_learning_pilot.sh").read_text(
        encoding="utf-8"
    )

    assert 'capabilities != {(9, 0)}' in script
    assert '"H100" not in name.upper()' in script
    assert "80_000_000_000 <= value <= 90_000_000_000" in script
    assert "host_memory < 128_000_000_000" in script
    assert "free_disk < 300_000_000_000" in script


def test_learning_pilot_staging_enters_the_pinned_repo() -> None:
    script = (
        ROOT / "scripts" / "stage_4b_learning_pilot_remote.sh"
    ).read_text(encoding="utf-8")

    cd_repo = script.index('cd "$repo"')
    prepare_fixture = script.index(
        '"$python_bin" scripts/prepare_4b_learning_pilot_data.py'
    )
    assert cd_repo < prepare_fixture


def test_learning_pilot_fixture_is_exact_hash_bound_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_values = [3, 11, 12, 13, 3, 21, 22, 23, 3, 31]
    source_bytes = _uint16_bytes(source_values)
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(source_bytes)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source_contract = {
        "bytes": len(source_bytes),
        "category": "fixture",
        "sequence": 0,
        "sha256": source_sha256,
        "s3_uri": "s3://fixture-bucket/source.bin",
        "tokens": len(source_values),
    }
    contract = {
        "corpus": {
            "doc_token_id": 3,
            "final_manifest_sha256": "a" * 64,
            "final_plan_fingerprint": "b" * 64,
            "final_selection_result_sha256": "c" * 64,
            "fixture_sequences": 2,
            "fixture_tokens": 8,
            "sequence_length": 4,
            "source_shard": source_contract,
            "vocab_size": 128,
        }
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    final_manifest_path = tmp_path / "final-manifest.json"
    final_manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(learning_pilot, "_validate_contract", lambda value: value)
    monkeypatch.setattr(
        learning_pilot,
        "validate_final_corpus_training_manifest",
        lambda *args, **kwargs: {"shards": [source_contract]},
    )
    output = tmp_path / "fixture"

    report = prepare_learning_pilot_fixture(
        contract_path=contract_path,
        final_manifest_path=final_manifest_path,
        source_shard_path=source_path,
        output=output,
    )

    assert (output / "shard-00000.bin").read_bytes() == _uint16_bytes(source_values[:8])
    assert report["document_markers"] == 2
    assert report["source_shard_category"] == "fixture"
    assert report["tokens"] == 8
    assert (
        prepare_learning_pilot_fixture(
            contract_path=contract_path,
            final_manifest_path=final_manifest_path,
            source_shard_path=source_path,
            output=output,
        )
        == report
    )


@pytest.mark.parametrize(
    ("world_size", "allowed_minutes", "rejected_minutes"),
    ((4, 480, 481), (8, 240, 241)),
)
def test_owner_authorization_runtime_cap_is_topology_specific(
    tmp_path: Path,
    world_size: int,
    allowed_minutes: int,
    rejected_minutes: int,
) -> None:
    plan_path = _write_plan(tmp_path, world_size)
    authorization_path = tmp_path / f"authorization-{world_size}.json"

    authorization = write_owner_authorization(
        plan_path=plan_path,
        output=authorization_path,
        provider="fixture-cloud",
        instance_type=f"{world_size}x-h100",
        hourly_rate_usd=20.0,
        max_runtime_minutes=allowed_minutes,
    )

    assert authorization["max_runtime_minutes"] == allowed_minutes
    assert authorization["full_4b_training"] is False
    assert validate_owner_authorization(
        plan_path=plan_path,
        authorization_path=authorization_path,
    ) == authorization
    with pytest.raises(ValueError, match="runtime must be between"):
        write_owner_authorization(
            plan_path=plan_path,
            output=authorization_path,
            provider="fixture-cloud",
            instance_type=f"{world_size}x-h100",
            hourly_rate_usd=20.0,
            max_runtime_minutes=rejected_minutes,
        )


@pytest.mark.parametrize(
    ("world_size", "runtime_minutes"),
    ((4, 480), (8, 240)),
)
def test_trainer_accepts_only_exact_bounded_learning_pilot(
    tmp_path: Path,
    world_size: int,
    runtime_minutes: int,
) -> None:
    authorization = {
        **_authorization_plan(world_size)["authorization_template"],
        "full_4b_training": False,
        "hourly_rate_usd": 20.0,
        "instance_type": f"{world_size}x-h100",
        "max_runtime_minutes": runtime_minutes,
        "paid_gpu": True,
        "provider": "fixture-cloud",
        "status": "owner_approved",
    }
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(authorization), encoding="utf-8")
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=world_size,
        backend="nccl",
        device=torch.device("cuda", 0),
    )

    validated = _validate_paid_authorization(
        path,
        run_fingerprint="a" * 64,
        source_commit="b" * 40,
        context=context,
        strategy="fsdp_full_shard",
        steps=PAID_LEARNING_PILOT_MAX_STEPS,
        target_tokens=PAID_LEARNING_PILOT_MAX_TOKENS,
    )

    assert validated["world_size"] == world_size
    authorization["approved_token_cap"] -= 1
    path.write_text(json.dumps(authorization), encoding="utf-8")
    with pytest.raises(ValueError, match="token boundary"):
        _validate_paid_authorization(
            path,
            run_fingerprint="a" * 64,
            source_commit="b" * 40,
            context=context,
            strategy="fsdp_full_shard",
            steps=PAID_LEARNING_PILOT_MAX_STEPS,
            target_tokens=PAID_LEARNING_PILOT_MAX_TOKENS,
        )
    authorization["approved_token_cap"] = PAID_LEARNING_PILOT_MAX_TOKENS
    authorization["max_runtime_minutes"] = runtime_minutes + 1
    path.write_text(json.dumps(authorization), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime cap"):
        _validate_paid_authorization(
            path,
            run_fingerprint="a" * 64,
            source_commit="b" * 40,
            context=context,
            strategy="fsdp_full_shard",
            steps=PAID_LEARNING_PILOT_MAX_STEPS,
            target_tokens=PAID_LEARNING_PILOT_MAX_TOKENS,
        )


def test_distributed_hf_export_inventory_is_hash_bound(tmp_path: Path) -> None:
    output = tmp_path / "export"
    output.mkdir()
    config = output / "config.json"
    weights = output / "model.safetensors"
    config.write_text('{"model_type":"lumi"}\n', encoding="utf-8")
    weights.write_bytes(b"weights")
    files = [
        {
            "bytes": path.stat().st_size,
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted((config, weights))
    ]
    report = {
        "checkpoint_manifest_sha256": "c" * 64,
        "checkpoint_step": 10,
        "files": files,
        "schema_version": 1,
        "status": "complete",
    }
    (output / "distributed-export-report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    assert validate_distributed_hf_export(
        output,
        checkpoint_manifest_sha256="c" * 64,
        checkpoint_step=10,
    ) == report
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="inventory changed"):
        validate_distributed_hf_export(
            output,
            checkpoint_manifest_sha256="c" * 64,
            checkpoint_step=10,
        )
