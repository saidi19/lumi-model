from __future__ import annotations

import hashlib
from pathlib import Path

from lumi.posttraining_stronger_teacher_ab_gpu import (
    BOUNDARY,
    DOWNLOADS,
    LIMITS,
    PROHIBITIONS,
    SERVER,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT.parent
    / "lumi-artifacts"
    / "4b-sft-stronger-teacher-ab-v0.1"
    / "c52325bacdb8"
    / "attempt-001"
)


def test_gpu_continuation_is_exactly_bounded() -> None:
    assert SERVER["gpu_name"] == "NVIDIA A10"
    assert SERVER["gpu_uuid"] == "GPU-959da0a2-b29e-601d-f97b-e245ed0997c1"
    assert sum(item["bytes"] for item in DOWNLOADS) == LIMITS["maximum_network_bytes"]
    assert LIMITS["maximum_additional_inference_attempts"] == 135
    assert LIMITS["maximum_additional_runtime_retries"] == 39
    assert LIMITS["remaining_candidate_outputs"] == 96
    assert all(PROHIBITIONS.values())


def test_gpu_continuation_preserves_existing_journal_boundary() -> None:
    baseline = (ARTIFACT / "private" / "baseline-call-journal.jsonl").read_bytes()
    candidate = (ARTIFACT / "private" / "candidate-call-journal.jsonl").read_bytes()
    assert len(baseline) == BOUNDARY["baseline_journal_bytes"]
    assert hashlib.sha256(baseline).hexdigest() == BOUNDARY[
        "baseline_journal_sha256"
    ]
    prefix_bytes = BOUNDARY["candidate_journal_prefix_bytes"]
    assert len(candidate) >= prefix_bytes
    assert hashlib.sha256(candidate[:prefix_bytes]).hexdigest() == BOUNDARY[
        "candidate_journal_prefix_sha256"
    ]
