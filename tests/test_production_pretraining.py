from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

import lumi.distributed_training as distributed_training
from lumi.distributed_training import (
    PAID_4B_PRODUCTION_MAX_STEPS,
    PAID_4B_PRODUCTION_TOKEN_CAP,
    PAID_4B_PRODUCTION_WORLD_SIZE,
    DistributedContext,
    _validate_authorized_resume_boundary,
    _validate_paid_authorization,
)
from lumi.production_pretraining import (
    _validate_contract,
    plan_4b_80b_production,
    validate_owner_authorization,
    validate_production_plan,
    write_owner_authorization,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "4b-80b-production-run-v0.1.json"
FINAL_MANIFEST = Path(
    "/Users/afnan/Documents/Flowaistudio/lumi-artifacts/final-corpus-80b-v0.1/"
    "bdd656e14cee-marker-production-run/evidence/results/shard-manifest.json"
)


@pytest.fixture(scope="module")
def production_plan() -> dict:
    if not FINAL_MANIFEST.is_file():
        pytest.skip("verified marker-aware corpus evidence is not mounted")
    return plan_4b_80b_production(
        contract_path=CONTRACT,
        manifest_path=FINAL_MANIFEST,
        config_path=ROOT / "configs/lumi-4b-v0.2.json",
        pilot_result_path=ROOT / "reports/4b-learning-pilot-result-v0.1.json",
        source_commit="1" * 40,
    )


def test_repository_contract_is_planning_only_and_exact() -> None:
    contract = _validate_contract(json.loads(CONTRACT.read_text(encoding="utf-8")))

    assert contract["approval"] == {
        "full_4b_training": False,
        "paid_gpu": False,
    }
    assert contract["run"]["maximum_steps"] == PAID_4B_PRODUCTION_MAX_STEPS
    assert contract["corpus"]["training_token_limit"] == PAID_4B_PRODUCTION_TOKEN_CAP
    assert contract["run"]["world_size"] == PAID_4B_PRODUCTION_WORLD_SIZE
    assert contract["run"]["schedule_steps"] == PAID_4B_PRODUCTION_MAX_STEPS

    changed = copy.deepcopy(contract)
    changed["approval"]["paid_gpu"] = True
    with pytest.raises(ValueError, match="cannot authorize paid work"):
        _validate_contract(changed)


def test_production_plan_binds_exact_tail_cost_and_three_segments(
    production_plan: dict,
) -> None:
    validated = validate_production_plan(production_plan)

    assert validated["approval"]["paid_gpu"] is False
    assert validated["run"]["global_tokens_per_step"] == 1_048_576
    assert validated["run"]["exact_final_step_sequences"] == 242
    assert validated["run"]["exact_final_step_tokens"] == 991_232
    assert validated["run_identity"]["distributed"]["checkpoint_steps"] == 250
    assert validated["corpus"]["training_tokens"] == PAID_4B_PRODUCTION_TOKEN_CAP
    assert validated["corpus"]["trailing_tokens_not_consumed"] == 8_364
    assert len(validated["segments"]) == 3
    assert [
        (segment["start_step"], segment["stop_step"])
        for segment in validated["segments"]
    ] == [(0, 31_500), (31_500, 63_000), (63_000, 76_294)]
    assert all(
        segment["authorization_template"]["paid_gpu"] is False
        for segment in validated["segments"]
    )
    assert validated["budget"]["estimated_measured_training_hours"] == pytest.approx(
        308.9190090844041
    )
    assert validated["budget"]["estimated_conservative_training_hours"] == pytest.approx(
        386.14876135550514
    )

    changed = copy.deepcopy(validated)
    changed["segments"][0]["stop_step"] += 1
    with pytest.raises(ValueError, match="plan hash changed"):
        validate_production_plan(changed)


def test_owner_artifact_is_one_exact_hash_bound_segment(
    tmp_path: Path,
    production_plan: dict,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(production_plan), encoding="utf-8")
    authorization_path = tmp_path / "authorization.json"
    authorization = write_owner_authorization(
        plan_path=plan_path,
        output=authorization_path,
        segment_index=1,
        provider="Lambda Cloud",
        instance_type="gpu_8x_h100_sxm5",
        hourly_rate_usd=31.92,
        max_runtime_minutes=10_080,
        max_cost_usd=7_000,
    )

    assert authorization["paid_gpu"] is True
    assert authorization["full_4b_training"] is True
    assert authorization["approved_from_step"] == 0
    assert authorization["approved_through_step"] == 31_500
    assert validate_owner_authorization(
        plan_path=plan_path,
        authorization_path=authorization_path,
    ) == authorization

    changed = copy.deepcopy(authorization)
    changed["approved_token_cap"] -= 1
    authorization_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="changed approved_token_cap"):
        validate_owner_authorization(
            plan_path=plan_path,
            authorization_path=authorization_path,
        )


def test_old_plan_refuses_segment_two_before_throughput_gated_continuation(
    tmp_path: Path,
    production_plan: dict,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(production_plan), encoding="utf-8")

    with pytest.raises(PermissionError, match="throughput gate"):
        write_owner_authorization(
            plan_path=plan_path,
            output=tmp_path / "authorization.json",
            segment_index=2,
            provider="Lambda Cloud",
            instance_type="gpu_8x_h100_sxm5",
            hourly_rate_usd=31.92,
            max_runtime_minutes=10_080,
            max_cost_usd=7_000,
        )


def test_trainer_accepts_only_exact_production_segment(
    tmp_path: Path,
    production_plan: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(production_plan), encoding="utf-8")
    authorization_path = tmp_path / "authorization.json"
    authorization = write_owner_authorization(
        plan_path=plan_path,
        output=authorization_path,
        segment_index=1,
        provider="Lambda Cloud",
        instance_type="gpu_8x_h100_sxm5",
        hourly_rate_usd=31.92,
        max_runtime_minutes=10_080,
        max_cost_usd=7_000,
    )
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=8,
        backend="nccl",
        device=torch.device("cuda", 0),
    )
    monkeypatch.setattr(
        distributed_training,
        "_validate_production_cuda_runtime",
        lambda _context: None,
    )

    validated = _validate_paid_authorization(
        authorization_path,
        run_fingerprint=production_plan["run_fingerprint"],
        source_commit="1" * 40,
        context=context,
        strategy="fsdp_full_shard",
        steps=authorization["approved_through_step"],
        target_tokens=authorization["approved_token_cap"],
        global_tokens_per_step=1_048_576,
    )

    assert validated["segment_index"] == 1
    changed = copy.deepcopy(authorization)
    changed["full_4b_training"] = False
    authorization_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="explicitly approve"):
        _validate_paid_authorization(
            authorization_path,
            run_fingerprint=production_plan["run_fingerprint"],
            source_commit="1" * 40,
            context=context,
            strategy="fsdp_full_shard",
            steps=authorization["approved_through_step"],
            target_tokens=authorization["approved_token_cap"],
            global_tokens_per_step=1_048_576,
        )


def test_production_segment_rejects_the_wrong_resume_checkpoint(
    production_plan: dict,
) -> None:
    scope = {
        **production_plan["segments"][1]["authorization_template"],
        "paid_gpu": True,
        "run_kind": "lumi_4b_80b_pretraining_v0.1",
        "status": "owner_approved",
    }
    _validate_authorized_resume_boundary(
        scope,
        {
            "step": 31_500,
            "tokens_seen": 33_030_144_000,
        },
        global_tokens_per_step=1_048_576,
    )

    with pytest.raises(ValueError, match="outside the authorized segment"):
        _validate_authorized_resume_boundary(
            scope,
            {
                "step": 0,
                "tokens_seen": 0,
            },
            global_tokens_per_step=1_048_576,
        )


def test_production_runtime_rejects_non_h100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=8,
        backend="nccl",
        device=torch.device("cuda", 0),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 0))

    with pytest.raises(RuntimeError, match="compute capability 9.0"):
        distributed_training._validate_production_cuda_runtime(context)
