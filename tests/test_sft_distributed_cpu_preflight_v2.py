from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import runpy
import stat
import sys
from pathlib import Path

import pytest
import torch

from lumi import posttraining_distributed_cpu_preflight_v2 as preflight


def _plan_shape() -> dict:
    bindings = copy.deepcopy(preflight.EXPECTED_FIXED_BINDINGS)
    bindings.update(
        {
            name: {"path": path, "sha256": "1" * 64}
            for name, path in preflight.EXPECTED_CODE_PATHS.items()
        }
    )
    return {
        "authorization_boundary": copy.deepcopy(
            preflight.EXPECTED_AUTHORIZATION_BOUNDARY
        ),
        "authorizations": copy.deepcopy(preflight.EXPECTED_AUTHORIZATIONS),
        "bindings": bindings,
        "execution": copy.deepcopy(preflight.EXPECTED_EXECUTION),
        "historical_v0_1": copy.deepcopy(preflight.EXPECTED_HISTORICAL_V0_1),
        "input": copy.deepcopy(preflight.EXPECTED_INPUT),
        "limits": copy.deepcopy(preflight.EXPECTED_LIMITS),
        "outputs": copy.deepcopy(preflight.EXPECTED_OUTPUTS),
        "production_eligible": False,
        "raw_training_text_retained": False,
        "schema_version": preflight.PLAN_SCHEMA_VERSION,
        "status": preflight.PLAN_STATUS,
        "training": copy.deepcopy(preflight.EXPECTED_TRAINING),
    }


def _historical_state() -> dict:
    return {key: 0 for key in preflight.EXPECTED_HISTORICAL_STATE_KEYS} | {
        "created_at": "left-created",
        "last_checkpoint_seconds": 1.0,
        "updated_at": "left-updated",
    }


def test_authorization_bindings_are_exact() -> None:
    for binding in preflight.EXPECTED_AUTHORIZATIONS:
        payload = binding["text"].encode("utf-8")
        assert len(payload) == binding["bytes"]
        assert hashlib.sha256(payload).hexdigest() == binding["sha256"]


def test_plan_contract_is_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan_shape()
    monkeypatch.setattr(preflight, "_verify_binding", lambda *args, **kwargs: None)
    preflight.validate_plan(plan, repository_root=tmp_path)
    plan["training"]["backend"] = "nccl"
    with pytest.raises(ValueError, match="plan contract changed"):
        preflight.validate_plan(plan, repository_root=tmp_path)


def test_historical_diagnostic_accepts_only_singular_schema_leak() -> None:
    reference = _historical_state()
    resumed = copy.deepcopy(reference)
    resumed.update(
        {
            "checkpoint_schema_version": 1,
            "created_at": "right-created",
            "last_checkpoint_seconds": 2.0,
            "updated_at": "right-updated",
        }
    )
    result = preflight._historical_state_delta(reference, resumed)
    assert result["resume_extra_keys"] == ["checkpoint_schema_version"]
    assert result["reference_key_count"] == 15
    assert result["resume_key_count"] == 16

    resumed["unexpected"] = True
    with pytest.raises(ValueError, match="failure signature changed"):
        preflight._historical_state_delta(reference, resumed)


def test_launch_schedule_and_commands_are_exact(tmp_path: Path) -> None:
    plan = _plan_shape()
    commands = [
        preflight._launch_command(
            python_executable=Path("/python"),
            plan=plan,
            repository_root=tmp_path,
            spec=spec,
            port=29_650 + index,
        )
        for index, spec in enumerate(preflight.LAUNCH_SPECS, 1)
    ]
    assert [spec.code for spec in preflight.LAUNCH_SPECS] == [
        "ddp_reference",
        "ddp_pause",
        "ddp_resume",
        "fsdp_reference",
        "fsdp_pause",
        "fsdp_resume",
    ]
    assert sum("--resume" in command for command in commands) == 2
    for command in commands:
        assert command[:3] == ["/python", "-m", "torch.distributed.run"]
        assert "--nproc-per-node=2" in command
        assert "--master-addr=127.0.0.1" in command
        assert "--local-addr=127.0.0.1" in command
        assert command[command.index("--device") + 1] == "cpu"
        assert command[command.index("--backend") + 1] == "gloo"
        assert command[command.index("--encoded-shards-record-limit") + 1] == "4"


def test_full_command_is_accepted_by_worker_cli(tmp_path, monkeypatch) -> None:
    worker_path = Path(__file__).resolve().parents[1] / "scripts/train_distributed_sft_v2.py"
    parser = runpy.run_path(str(worker_path))["parse_args"]
    for spec in preflight.LAUNCH_SPECS:
        command = preflight._launch_command(
            python_executable=Path(sys.executable), plan=_plan_shape(),
            repository_root=tmp_path, spec=spec, port=29651,
        )
        script_index = command.index(preflight.EXPECTED_CODE_PATHS["worker_runner"])
        monkeypatch.setattr(sys, "argv", command[script_index:])
        parsed = parser()
        assert parsed.resume is spec.resume
        assert parsed.keep_checkpoints == 2
        assert parsed.strategy == spec.strategy


def test_offline_environment_removes_proxy_and_gpu_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://example.invalid")
    environment = preflight._offline_environment(tmp_path)
    assert "HTTPS_PROXY" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["NO_PROXY"] == "localhost,127.0.0.1"


def test_attempt_marker_is_exclusive_private_and_predeclares_no_retry(
    tmp_path: Path,
) -> None:
    plan = _plan_shape()
    root = preflight._create_attempt_root(tmp_path, plan)
    marker, marker_bytes = preflight._publish_no_retry_marker(
        repository_root=tmp_path,
        plan=plan,
        plan_file_sha256="a" * 64,
    )
    assert root == marker.parent
    assert marker_bytes == marker.stat().st_size
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    assert json.loads(marker.read_text())["retries_after_this_marker"] == 0
    with pytest.raises(FileExistsError):
        preflight._publish_no_retry_marker(
            repository_root=tmp_path,
            plan=plan,
            plan_file_sha256="a" * 64,
        )


def test_aggregate_physical_reads_are_exact() -> None:
    audit = {
        "binary_bytes_read": 3_504,
        "encoded_index_rows_read": 8,
        "index_bytes_read": 1_288,
        "rank_local_loads": 2,
        "selected_prefix_sha256": "a" * 64,
    }
    result = preflight._aggregate_load_audits(
        [copy.deepcopy(audit) for _ in range(6)],
        preflight.EXPECTED_LIMITS,
    )
    assert result == {
        "encoded_binary_bytes_read": 21_024,
        "encoded_index_bytes_read": 7_728,
        "encoded_index_rows_read": 48,
        "rank_local_dataset_loads": 12,
        "selected_prefix_sha256": "a" * 64,
    }


def test_stable_metrics_drop_only_timing_fields(tmp_path: Path) -> None:
    metric = {
        "checkpoint_seconds": 1.0,
        "elapsed_seconds": 2.0,
        "loss": 3.0,
        "step": 1,
        "tokens_per_second": 4.0,
    }
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(metric) + "\n", encoding="utf-8")
    assert preflight._stable_metrics(path) == [{"loss": 3.0, "step": 1}]


def test_semantic_tensor_digest_and_nested_exact_are_strict() -> None:
    left = {
        "optimizer": {"step": torch.tensor([1], dtype=torch.int64)},
        "rng_state": torch.tensor([1, 2], dtype=torch.uint8),
    }
    right = copy.deepcopy(left)
    assert preflight._nested_exact(left, right)
    assert preflight._semantic_digest(left) == preflight._semantic_digest(right)
    right["rng_state"][0] = 2
    assert not preflight._nested_exact(left, right)


def test_child_output_is_never_captured_and_log_is_raw_text_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict = {}

    class Process:
        pid = 12_345

        def wait(self, timeout=None):
            observed["timeout"] = timeout
            return 0

    def fake_popen(command: list[str], **kwargs):
        observed.update(kwargs)
        return Process()

    monkeypatch.setattr(preflight.subprocess, "Popen", fake_popen)
    log_path = tmp_path / "launch.json"
    result = preflight._run_launch(
        command=["python", "worker.py"],
        environment={},
        repository_root=tmp_path,
        spec=preflight.LAUNCH_SPECS[0],
        log_path=log_path,
        timeout_seconds=10,
        maximum_log_bytes=4_096,
    )
    assert observed["stdout"] is preflight.subprocess.DEVNULL
    assert observed["stderr"] is preflight.subprocess.DEVNULL
    assert result["child_output_bytes_retained"] == 0
    assert os.stat(log_path).st_mode & 0o777 == 0o600
    assert "prompt" not in log_path.read_text(encoding="utf-8")


def test_source_contains_no_retry_loop_or_shell_launch() -> None:
    source = inspect.getsource(preflight.run_distributed_cpu_preflight)
    assert "shell=True" not in source
    assert "while" not in source
    assert preflight.EXPECTED_LIMITS["maximum_retries_after_started_attempt"] == 0
    assert len(preflight.LAUNCH_SPECS) == 6
