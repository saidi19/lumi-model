from __future__ import annotations

import copy
import hashlib
import inspect
import os
from pathlib import Path

import pytest
import torch

from lumi import posttraining_sft_cpu_correctness_smoke_v2 as smoke


def _plan_shape() -> dict:
    bindings = copy.deepcopy(smoke.EXPECTED_FIXED_BINDINGS)
    bindings.update(
        {
            name: {"path": path, "sha256": "1" * 64}
            for name, path in smoke.EXPECTED_CODE_PATHS.items()
        }
    )
    return {
        "authorization": copy.deepcopy(smoke.EXPECTED_AUTHORIZATION),
        "authorization_bytes": smoke.EXPECTED_AUTHORIZATION_BYTES,
        "authorization_sha256": smoke.EXPECTED_AUTHORIZATION_SHA256,
        "authorization_text": smoke.EXPECTED_AUTHORIZATION_TEXT,
        "bindings": bindings,
        "input": copy.deepcopy(smoke.EXPECTED_INPUT),
        "limits": copy.deepcopy(smoke.EXPECTED_LIMITS),
        "model": copy.deepcopy(smoke.EXPECTED_MODEL),
        "observed_at": smoke.EXPECTED_OBSERVED_AT,
        "outputs": copy.deepcopy(smoke.EXPECTED_OUTPUTS),
        "production_eligible": False,
        "raw_training_text_retained": False,
        "runtime_contract": copy.deepcopy(smoke.EXPECTED_RUNTIME_CONTRACT),
        "schema_version": smoke.PLAN_SCHEMA_VERSION,
        "source_checkpoint_contract": copy.deepcopy(
            smoke.EXPECTED_SOURCE_CHECKPOINT_CONTRACT
        ),
        "status": smoke.PLAN_STATUS,
        "training": copy.deepcopy(smoke.EXPECTED_TRAINING),
    }


def test_authorization_binding_is_exact() -> None:
    payload = smoke.EXPECTED_AUTHORIZATION_TEXT.encode("utf-8")
    assert len(payload) == 18
    assert hashlib.sha256(payload).hexdigest() == smoke.EXPECTED_AUTHORIZATION_SHA256


def test_strict_json_and_raw_text_guards_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        smoke.strict_json_bytes(b'{"a":1,"a":2}')
    with pytest.raises(ValueError, match="nonfinite"):
        smoke.strict_json_bytes(b'{"value":NaN}')
    with pytest.raises(ValueError, match="raw training text"):
        smoke.assert_raw_text_free({"nested": {"content": "forbidden"}})


def test_plan_contract_rejects_any_limit_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan_shape()
    monkeypatch.setattr(smoke, "_resolve_bound_file", lambda *_args, **_kwargs: tmp_path)
    smoke.validate_plan(plan, repository_root=tmp_path)
    plan["limits"]["maximum_encoded_index_rows_parsed"] = 13
    with pytest.raises(ValueError, match="plan contract changed"):
        smoke.validate_plan(plan, repository_root=tmp_path)


def test_runner_commands_are_three_fresh_v3_processes_with_explicit_defaults(
    tmp_path: Path,
) -> None:
    commands = smoke.runner_commands(
        plan=_plan_shape(),
        repository_root=tmp_path,
        source_manifest_sha256=smoke.EXPECTED_SOURCE_CHECKPOINT["manifest.sha256"],
    )
    assert len(commands) == 3
    assert all(command[1].endswith("run_sft_cpu_canary_v3.py") for command in commands)
    assert ["--resume" in command for command in commands] == [False, False, True]
    assert [command[command.index("--stop-after-step") + 1] for command in commands] == [
        "2",
        "1",
        "2",
    ]
    for command in commands:
        for flag in (
            "--checkpoint-steps",
            "--minimum-learning-rate-ratio",
            "--optimizer-profile",
            "--pad-to-multiple-of",
        ):
            assert flag in command


def test_static_read_caps_cover_exactly_three_four_record_loads() -> None:
    limits = smoke.EXPECTED_LIMITS
    assert limits["maximum_dataset_loads"] == 3
    assert limits["maximum_encoded_record_file_reads"] == 3 * 4
    assert limits["maximum_encoded_index_rows_parsed"] == 3 * 4
    assert limits["maximum_encoded_binary_bytes_read"] == 3 * 4 * 4095 * 8
    assert limits["maximum_encoded_index_bytes_read"] == 3 * 4 * 175


def test_nested_tensor_comparison_and_digest_are_exact() -> None:
    left = {"optimizer": {0: {"step": torch.tensor(1), "value": torch.tensor([1.0])}}}
    right = copy.deepcopy(left)
    assert smoke._nested_exact(left, right)
    assert smoke._semantic_tensor_digest(left) == smoke._semantic_tensor_digest(right)
    right["optimizer"][0]["value"][0] = 2.0
    assert not smoke._nested_exact(left, right)
    assert smoke._semantic_tensor_digest(left) != smoke._semantic_tensor_digest(right)


def test_private_evidence_writer_is_exclusive_nofollow_and_0600(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    smoke._private_write_exclusive(evidence, {"safe": True})
    assert smoke._private_evidence_size(evidence) == evidence.stat().st_size
    assert evidence.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        smoke._private_write_exclusive(evidence, {"safe": False})

    target = tmp_path / "target.json"
    target.write_text("untouched", encoding="utf-8")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises((FileExistsError, OSError)):
        smoke._private_write_exclusive(symlink, {"safe": False})
    assert target.read_text(encoding="utf-8") == "untouched"


def test_report_and_verification_fixed_point_accounting_is_exact() -> None:
    report = {"report_sha256": "0" * 64, "resource_accounting": {}}
    report_size = smoke._finalize_report_size(report, private_root_bytes=1234)
    assert report_size == len(smoke.json_file_bytes(report))
    assert report["resource_accounting"]["cumulative_private_write_bytes_upper_bound"] == (
        1234 + report_size + 65_536 + 1_048_576
    )

    verification = {"status": "synthetic"}
    verification_size = smoke._finalize_verification_size(
        verification,
        private_root_bytes=1234,
        report_file_bytes=report_size,
    )
    assert verification_size == len(smoke.json_file_bytes(verification))
    assert verification["private_write_audit"][
        "cumulative_private_write_bytes_upper_bound"
    ] == (
        1234 + report_size + verification_size + 1_048_576
    )


def test_orchestrator_never_directly_adds_a_fourth_dataset_load() -> None:
    source = inspect.getsource(smoke)
    assert "EncodedSFTDataset(" not in source
    assert '"PYTHONDONTWRITEBYTECODE": "1"' in source
    assert '"PYTHONHASHSEED": "0"' in source
    assert '"OMP_NUM_THREADS": "1"' in source
    assert "remaining_seconds = deadline - time.monotonic()" in source


def test_source_checkpoint_contract_is_exact_and_disables_cache() -> None:
    contract = smoke.EXPECTED_SOURCE_CHECKPOINT_CONTRACT
    assert contract["config_semantics"]["use_cache"] is False
    assert contract["inventory"] == sorted(smoke.EXPECTED_SOURCE_CHECKPOINT)
    assert contract["exact_file_sha256"]["model.safetensors"] == (
        "a1779301e8b0eef3e648672baf55a8e090b2f1d4e5a232ebe0be93863a2f596a"
    )
    assert not any(Path(name).is_absolute() for name in contract["inventory"])


def test_no_shell_or_network_api_is_used_by_orchestrator() -> None:
    source = inspect.getsource(smoke)
    assert "shell=True" not in source
    assert "requests." not in source
    assert "urllib" not in source
    assert "socket" not in source
    assert os.path.basename(smoke.EXPECTED_CODE_PATHS["orchestrator"]) == (
        "run_sft_cpu_correctness_smoke_v2.py"
    )
