from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from lumi import posttraining_sft_cpu_audit_recovery_v3 as recovery


def _plan_shape() -> dict:
    bindings = copy.deepcopy(recovery.EXPECTED_FIXED_BINDINGS)
    bindings.update(
        {
            name: {"path": path, "sha256": "1" * 64}
            for name, path in recovery.EXPECTED_CODE_PATHS.items()
        }
    )
    return {
        "authorization_bytes": recovery.EXPECTED_AUTHORIZATION_BYTES,
        "authorization_sha256": recovery.EXPECTED_AUTHORIZATION_SHA256,
        "authorization_text": recovery.EXPECTED_AUTHORIZATION_TEXT,
        "bindings": bindings,
        "failed_attempt": copy.deepcopy(recovery.EXPECTED_FAILED_ATTEMPT),
        "limits": copy.deepcopy(recovery.EXPECTED_LIMITS),
        "outputs": copy.deepcopy(recovery.EXPECTED_OUTPUTS),
        "production_eligible": False,
        "raw_training_text_retained": False,
        "schema_version": recovery.PLAN_SCHEMA_VERSION,
        "status": recovery.PLAN_STATUS,
    }


def _completed_state() -> dict:
    return {
        "best_loss": recovery.EXPECTED_FINAL_STATE["loss_sequence"][0],
        "last_loss": recovery.EXPECTED_FINAL_STATE["loss_sequence"][-1],
        "latest_checkpoint": "checkpoint-00000002",
        "records_seen": 4,
        "run_fingerprint": "a" * 64,
        "run_specification": {
            "data": {"read_audit": copy.deepcopy(recovery.EXPECTED_READ_AUDIT)}
        },
        "schema_version": "lumi.sft-cpu-canary-state.v0.3",
        "status": "complete",
        "step": 2,
        "supervised_tokens_seen": 153,
        "tokens_seen": 219,
    }


def test_recovery_authorization_is_exact() -> None:
    payload = recovery.EXPECTED_AUTHORIZATION_TEXT.encode("utf-8")
    assert len(payload) == 19
    assert hashlib.sha256(payload).hexdigest() == recovery.EXPECTED_AUTHORIZATION_SHA256


def test_plan_contract_is_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan_shape()
    monkeypatch.setattr(recovery, "_resolve_bound_file", lambda *args, **kwargs: None)
    recovery.validate_plan(plan, repository_root=tmp_path)
    plan["limits"]["maximum_optimizer_updates"] = 1
    with pytest.raises(ValueError, match="plan contract changed"):
        recovery.validate_plan(plan, repository_root=tmp_path)


def test_only_exact_checkpoint_schema_leak_is_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uninterrupted = _completed_state()
    resumed = {
        **copy.deepcopy(uninterrupted),
        "checkpoint_schema_version": "lumi.sft-cpu-canary-checkpoint.v0.3",
    }

    def read_state(path: Path) -> dict:
        return copy.deepcopy(resumed if "resumed" in path.parts else uninterrupted)

    monkeypatch.setattr(recovery, "_read_json", read_state)
    left, right = recovery._normalized_completed_states(tmp_path)
    assert left == right == uninterrupted

    resumed["unexpected"] = True
    with pytest.raises(ValueError, match="failure signature changed"):
        recovery._normalized_completed_states(tmp_path)


def test_private_snapshot_is_raw_text_free_and_rejects_symlink(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    payload = root / "state.json"
    payload.write_text(json.dumps({"step": 2}), encoding="utf-8")
    payload.chmod(0o600)
    snapshot = recovery._private_tree_snapshot(root)
    assert snapshot == [
        {
            "bytes": payload.stat().st_size,
            "kind": "file",
            "mode": "0600",
            "path": "state.json",
            "sha256": recovery.sha256_file(payload),
        }
    ]
    link = root / "state-link.json"
    link.symlink_to(payload)
    with pytest.raises(ValueError, match="symlink"):
        recovery._private_tree_snapshot(root)


def test_recovery_never_calls_dataset_or_training_runtime() -> None:
    source = inspect.getsource(recovery)
    assert "EncodedSFTDataset(" not in source
    assert "run_sft_cpu_canary(" not in source
    assert "optimizer.step(" not in source
    assert recovery.EXPECTED_LIMITS["maximum_optimizer_updates"] == 0
    assert recovery.EXPECTED_LIMITS[
        "maximum_additional_encoded_index_rows_read"
    ] == 0


def test_report_size_accounting_converges() -> None:
    report = {
        "recovery_resource_accounting": {},
        "report_sha256": "0" * 64,
    }
    size = recovery._finalize_report_size(report, marker_bytes=123)
    assert size == len(recovery.failed_smoke.json_file_bytes(report))
    assert report["recovery_resource_accounting"][
        "recovery_private_write_bytes_upper_bound"
    ] == (123 + size + 65_536)


def test_failed_attempt_and_recovery_outputs_are_disjoint() -> None:
    failed = Path(recovery.EXPECTED_FAILED_ATTEMPT["private_root"])
    recovered = Path(recovery.EXPECTED_OUTPUTS["private_root"])
    assert failed != recovered
    assert not recovered.is_relative_to(failed)
    assert recovery.EXPECTED_LIMITS["maximum_retries_of_failed_attempt"] == 0
