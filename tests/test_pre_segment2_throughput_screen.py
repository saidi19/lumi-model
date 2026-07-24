from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from lumi.distributed_training import (
    PAID_SEGMENT2_SCREEN_AGGREGATE_TOKEN_CAP,
    PAID_SEGMENT2_SCREEN_MAX_STEPS,
    PAID_SEGMENT2_SCREEN_OWNER_ACKNOWLEDGEMENT,
    PAID_SEGMENT2_SCREEN_RUN_KIND,
    PAID_SEGMENT2_SCREEN_TOKEN_CAP,
    DistributedContext,
    _validate_paid_authorization,
)


def _authorization(fingerprint: str) -> dict:
    return {
        "aggregate_candidate_count": 5,
        "aggregate_token_cap": PAID_SEGMENT2_SCREEN_AGGREGATE_TOKEN_CAP,
        "allowed_run_fingerprints": {
            "baseline_b1_gc": fingerprint,
            "batch2_gc": "b" * 64,
            "batch1_no_gc": "c" * 64,
            "batch2_no_gc": "d" * 64,
            "batch2_no_gc_compile": "e" * 64,
        },
        "approved_through_step": PAID_SEGMENT2_SCREEN_MAX_STEPS,
        "approved_token_cap": PAID_SEGMENT2_SCREEN_TOKEN_CAP,
        "full_4b_training": False,
        "hourly_rate_usd": 31.92,
        "instance_ip": "209.20.157.52",
        "instance_type": "gpu_8x_h100_sxm5",
        "max_cost_usd": 63.84,
        "max_runtime_minutes": 120,
        "owner_acknowledgement": PAID_SEGMENT2_SCREEN_OWNER_ACKNOWLEDGEMENT,
        "paid_gpu": True,
        "provider": "Lambda Cloud",
        "run_kind": PAID_SEGMENT2_SCREEN_RUN_KIND,
        "schema_version": 1,
        "segment_1_access": {
            "checkpoint_transfer": False,
            "modify_or_restart": False,
            "read_only_inputs": [
                "bounded_training_prefix",
                "heldout_data",
                "tokenizer",
            ],
        },
        "source_commit": "a" * 40,
        "status": "owner_approved",
        "strategy": "fsdp_full_shard",
        "world_size": 8,
    }


def _context() -> DistributedContext:
    return DistributedContext(
        rank=0,
        local_rank=0,
        world_size=8,
        backend="nccl",
        device=torch.device("cuda"),
    )


def test_pre_segment2_authorization_allows_only_sealed_fingerprints(
    tmp_path: Path,
) -> None:
    fingerprint = "a" * 64
    authorization = _authorization(fingerprint)
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(authorization), encoding="utf-8")

    observed = _validate_paid_authorization(
        path,
        run_fingerprint=fingerprint,
        source_commit="a" * 40,
        context=_context(),
        strategy="fsdp_full_shard",
        steps=33,
        target_tokens=PAID_SEGMENT2_SCREEN_TOKEN_CAP,
        global_tokens_per_step=1_048_576,
    )

    assert observed == authorization
    with pytest.raises(ValueError, match="fingerprint matrix"):
        _validate_paid_authorization(
            path,
            run_fingerprint="f" * 64,
            source_commit="a" * 40,
            context=_context(),
            strategy="fsdp_full_shard",
            steps=33,
            target_tokens=PAID_SEGMENT2_SCREEN_TOKEN_CAP,
            global_tokens_per_step=1_048_576,
        )


def test_pre_segment2_authorization_rejects_scope_expansion(tmp_path: Path) -> None:
    authorization = _authorization("a" * 64)
    changed = copy.deepcopy(authorization)
    changed["full_4b_training"] = True
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot approve full"):
        _validate_paid_authorization(
            path,
            run_fingerprint="a" * 64,
            source_commit="a" * 40,
            context=_context(),
            strategy="fsdp_full_shard",
            steps=33,
            target_tokens=PAID_SEGMENT2_SCREEN_TOKEN_CAP,
            global_tokens_per_step=1_048_576,
        )
