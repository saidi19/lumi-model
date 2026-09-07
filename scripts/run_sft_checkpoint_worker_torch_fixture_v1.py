#!/usr/bin/env python3
"""One isolated Linux CI fixture; never a model/data/worker launcher.

All Torch imports and tensor work happen after the Linux/synthetic-only guards.
Only generated tiny Linear/Adam state is used. This file is not run on the Mac.
The real Supervisor is shared in-process only; its frozen schema has two rank
slots while this checkpoint acceptance fixture exercises rank zero alone.
No distributed-process, OS-containment or production claim follows.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import json
import os
import stat
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_synthetic_sft_checkpoint_worker_fixture_v1"
STAGE = "guard"
CURRENT_WORKER = None
LIMITS = {
    "maximum_file_bytes": 4 * 1024 * 1024,
    "maximum_read_bytes": 64 * 1024 * 1024,
    "maximum_read_call_bytes": 4 * 1024 * 1024,
    "maximum_write_call_bytes": 4 * 1024 * 1024,
    "maximum_sync_operations": 128,
}
FLAGS = (
    "real_torch_dcp_used", "model_resume_equal", "optimizer_resume_equal",
    "rng_resume_equal", "repeated_restore_charged", "wrong_hash_rejected",
    "read_cap_rejected", "write_cap_rejected", "existing_checkpoint_preserved",
    "private_modes_verified", "borrowed_parent_fd_preserved",
    "explicit_dcp_storage_verified", "serializer_filelikes_verified",
    "stock_filesystem_io_blocked",
)
NONCLAIMS = (
    "real_model_or_data_access", "cuda_used", "network_authorized",
    "multiprocess_shared_controller_verified", "os_resource_enforcement_verified",
    "production_eligible", "real_sft_execution_permitted",
)


def require(condition):
    if not condition:
        raise AssertionError("fixture_assertion")


def module(name):
    return importlib.import_module(PACKAGE + "." + name)


def controller(*, maximum_writes=16 * 1024 * 1024):
    counts = {
        "sequential_invocations": 3, "global_optimizer_steps": 4,
        "rank_optimizer_steps": 8, "logical_sample_events": 8,
        "maximum_unique_train_records": 8, "rank_local_input_loads": 6,
        "maximum_encoded_index_rows": 48,
        "maximum_selected_binary_bytes": 48 * 32 * 8,
        "maximum_selected_index_bytes": 48 * 176,
        "includes_checkpoint_hashing_or_final_digest": False,
        "is_total_io_or_cost_budget": False,
    }
    caps = {
        "maximum_combined_runtime_seconds": 60, "maximum_cost_microusd": 1000,
        "total_host_hourly_rate_microusd": 60_000,
        "maximum_host_ram_bytes": 512 * 1024 * 1024,
        "maximum_per_gpu_memory_bytes": 1,
        "maximum_private_write_bytes": maximum_writes,
        "maximum_checkpoint_read_bytes": 64 * 1024 * 1024,
        "maximum_encoded_binary_read_bytes": counts["maximum_selected_binary_bytes"],
        "maximum_encoded_index_read_bytes": counts["maximum_selected_index_bytes"],
        "maximum_log_bytes": 1, "maximum_invocations": 3,
        "maximum_global_optimizer_steps": 4, "maximum_external_network_bytes": 0,
        "maximum_inference_calls": 0,
    }
    supervisor = module("posttraining_sft_supervisor_v1").Supervisor(
        caps, counts, world_size=2, clock_ns=lambda: 0,
    )
    supervisor.begin_invocation("reference", 0, 2, False)
    return supervisor


def worker(implementation, supervisor, **limits):
    global CURRENT_WORKER
    CURRENT_WORKER = None
    CURRENT_WORKER = implementation.CheckpointWorker(
        rank=0, world_size=1, reserve=supervisor.reserve, interrupt=supervisor.interrupt,
        limits={**LIMITS, **limits},
    )
    return CURRENT_WORKER


def equal_tree(torch, left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left, right))
    if type(left) is dict:
        return (type(right) is dict and left.keys() == right.keys()
                and all(equal_tree(torch, left[key], right[key]) for key in left))
    if type(left) in (list, tuple):
        return (type(left) is type(right) and len(left) == len(right)
                and all(equal_tree(torch, a, b) for a, b in zip(left, right, strict=True)))
    return type(left) is type(right) and left == right


def optimizer_step(torch, model, optimizer):
    # This is a 4-by-3 generated CPU fixture, not user data or the real SFT loop.
    inputs = torch.randn(4, 3, dtype=torch.float64)
    expected = torch.randn(4, 2, dtype=torch.float64)
    optimizer.zero_grad(set_to_none=True)
    ((model(inputs) - expected) ** 2).mean().backward()
    optimizer.step()


def new_model(torch):
    model = torch.nn.Linear(3, 2, dtype=torch.float64, device="cpu")
    return model, torch.optim.Adam(model.parameters(), lr=0.01)


def private_inventory(root):
    """Fixture-owned content only, never the user's source or checkpoints."""
    values = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        require(info.st_uid == os.geteuid() and not stat.S_ISLNK(info.st_mode))
        if stat.S_ISDIR(info.st_mode):
            require(stat.S_IMODE(info.st_mode) == 0o700)
        else:
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1)
            require(stat.S_IMODE(info.st_mode) == 0o600)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            values[path.relative_to(root).as_posix()] = digest
    return values


def expect_rejection(implementation, supervisor, operation):
    try:
        operation()
    except implementation.CheckpointWorkerError as error:
        require(error.code in implementation.ERROR_CODES)
        require(error.args == (error.code,))
    else:
        raise AssertionError("fixture_rejection_missing")
    require(supervisor.summary()["state"] == "failed")


@contextlib.contextmanager
def native_guards(torch, dcp, filesystem, counters):
    save, load, dcp_save, dcp_load = torch.save, torch.load, dcp.save, dcp.load

    def safe_save(obj, destination, *args, **kwargs):
        require(not isinstance(destination, (str, bytes, os.PathLike, int)))
        require(callable(getattr(destination, "write", None)))
        counters["torch_save_filelike_calls"] += 1
        return save(obj, destination, *args, **kwargs)

    def safe_load(source, *args, **kwargs):
        require(not isinstance(source, (str, bytes, os.PathLike, int)))
        require(callable(getattr(source, "read", None)))
        require(kwargs.get("weights_only") is True)
        counters["torch_load_filelike_calls"] += 1
        return load(source, *args, **kwargs)

    def explicit_save(*args, **kwargs):
        require(kwargs.get("storage_writer") is not None)
        require(kwargs.get("checkpoint_id") is None)
        counters["dcp_explicit_save_calls"] += 1
        return dcp_save(*args, **kwargs)

    def explicit_load(*args, **kwargs):
        require(kwargs.get("storage_reader") is not None)
        require(kwargs.get("checkpoint_id") is None)
        counters["dcp_explicit_load_calls"] += 1
        return dcp_load(*args, **kwargs)

    with contextlib.ExitStack() as stack:
        for owner, name, replacement in (
            (torch, "save", safe_save), (torch, "load", safe_load),
            (dcp, "save", explicit_save), (dcp, "load", explicit_load),
        ):
            stack.enter_context(mock.patch.object(owner, name, replacement))
        stack.enter_context(mock.patch.object(os, "sync", side_effect=AssertionError("os_sync")))
        for name in ("create_stream", "mkdir", "rename", "rm_file"):
            stack.enter_context(mock.patch.object(
                filesystem.FileSystem, name, side_effect=AssertionError("native_filesystem"),
            ))
        yield


def run():
    global STAGE
    require(sys.platform == "linux" and sys.argv[1:] == ["--synthetic-only"])
    require(os.environ.get("LUMI_CHECKPOINT_TORCH_CI_FIXTURE") == "1")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "")
    os.umask(0o077)
    STAGE = "dependencies"
    torch = importlib.import_module("torch")
    dcp = importlib.import_module("torch.distributed.checkpoint")
    filesystem = importlib.import_module("torch.distributed.checkpoint.filesystem")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.distributed.is_initialized())
    require(not torch.cuda.is_initialized())
    torch.manual_seed(271828)
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(ROOT / "src/lumi")]
    sys.modules[PACKAGE] = package
    implementation = module("posttraining_sft_checkpoint_worker_v1")
    counters = dict.fromkeys((
        "torch_save_filelike_calls", "torch_load_filelike_calls",
        "dcp_explicit_save_calls", "dcp_explicit_load_calls",
    ), 0)
    report = {
        "schema_version": "lumi.sft-checkpoint-worker-fixture.v0.1",
        **dict.fromkeys(FLAGS, False), **dict.fromkeys(NONCLAIMS, False),
    }
    STAGE = "generated_fixture"
    with tempfile.TemporaryDirectory(prefix="lumi-SYNTHETIC-torch-worker-v1-") as temporary:
        parent = Path(temporary)
        parent.chmod(0o700)
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            model, optimizer = new_model(torch)
            optimizer_step(torch, model, optimizer)
            saved_rng = torch.get_rng_state().clone()
            saved_model = copy.deepcopy(model.state_dict())
            saved_optimizer = copy.deepcopy(optimizer.state_dict())
            supervisor = controller()
            active = worker(implementation, supervisor)
            with native_guards(torch, dcp, filesystem, counters):
                STAGE = "save_checkpoint"
                receipt = active.save_checkpoint(
                    parent_fd, model=model, optimizer=optimizer, rng_state=saved_rng,
                    training_state={"step": 1, "schema_version": "SYNTHETIC"}, step=1,
                )
                require(receipt["checkpoint_name"] == "checkpoint-00000001")
                require(receipt["status"] == "published_checkpoint")
                require(receipt["published"] is True and receipt["production_eligible"] is False)
                checkpoint = parent / receipt["checkpoint_name"]
                frozen = private_inventory(checkpoint)
                require(len(frozen) >= 4 and "manifest.sha256" in frozen)
                require(os.fstat(parent_fd).st_ino == parent.stat().st_ino)
                report["private_modes_verified"] = True
                report["borrowed_parent_fd_preserved"] = True

                STAGE = "restore_checkpoint"
                restored_model, restored_optimizer = new_model(torch)
                returned_rng = active.restore_checkpoint(
                    parent_fd, checkpoint_name=receipt["checkpoint_name"],
                    manifest_sha256=receipt["manifest_sha256"],
                    model=restored_model, optimizer=restored_optimizer,
                )
                require(equal_tree(torch, saved_model, restored_model.state_dict()))
                require(equal_tree(torch, saved_optimizer, restored_optimizer.state_dict()))
                require(equal_tree(torch, saved_rng, returned_rng))
                reads_before = supervisor.summary()["resources_reserved"]["checkpoint_read_bytes"]

                STAGE = "repeated_restore"
                returned_rng = active.restore_checkpoint(
                    parent_fd, checkpoint_name=receipt["checkpoint_name"],
                    manifest_sha256=receipt["manifest_sha256"],
                    model=restored_model, optimizer=restored_optimizer,
                )
                require(supervisor.summary()["resources_reserved"]["checkpoint_read_bytes"]
                        > reads_before)
                report["repeated_restore_charged"] = True

                STAGE = "resume_equality"
                torch.set_rng_state(saved_rng)
                optimizer_step(torch, model, optimizer)
                expected_rng = torch.get_rng_state().clone()
                torch.set_rng_state(returned_rng)
                optimizer_step(torch, restored_model, restored_optimizer)
                require(equal_tree(torch, model.state_dict(), restored_model.state_dict()))
                require(equal_tree(torch, optimizer.state_dict(), restored_optimizer.state_dict()))
                require(torch.equal(expected_rng, torch.get_rng_state()))
                for key in ("model_resume_equal", "optimizer_resume_equal", "rng_resume_equal"):
                    report[key] = True
                require(private_inventory(checkpoint) == frozen)

                STAGE = "wrong_manifest_hash"
                failed = controller()
                wrong = worker(implementation, failed)
                incorrect_hash = "0" * 64 if receipt["manifest_sha256"] != "0" * 64 else "1" * 64
                expect_rejection(implementation, failed, lambda: wrong.restore_checkpoint(
                    parent_fd, checkpoint_name=receipt["checkpoint_name"],
                    manifest_sha256=incorrect_hash, model=restored_model,
                    optimizer=restored_optimizer,
                ))
                require(private_inventory(checkpoint) == frozen)
                report["wrong_hash_rejected"] = True

                STAGE = "read_cap"
                failed = controller()
                capped = worker(implementation, failed, maximum_read_bytes=1)
                expect_rejection(implementation, failed, lambda: capped.restore_checkpoint(
                    parent_fd, checkpoint_name=receipt["checkpoint_name"],
                    manifest_sha256=receipt["manifest_sha256"], model=restored_model,
                    optimizer=restored_optimizer,
                ))
                require(private_inventory(checkpoint) == frozen)
                report["read_cap_rejected"] = True

                STAGE = "existing_checkpoint"
                failed = controller()
                collision = worker(implementation, failed)
                expect_rejection(implementation, failed, lambda: collision.save_checkpoint(
                    parent_fd, model=model, optimizer=optimizer, rng_state=saved_rng,
                    training_state={"step": 1, "schema_version": "SYNTHETIC"}, step=1,
                ))
                require(private_inventory(checkpoint) == frozen)
                report["existing_checkpoint_preserved"] = True

                STAGE = "write_cap"
                failed = controller(maximum_writes=1)
                capped = worker(implementation, failed)
                expect_rejection(implementation, failed, lambda: capped.save_checkpoint(
                    parent_fd, model=model, optimizer=optimizer, rng_state=saved_rng,
                    training_state={"step": 2, "schema_version": "SYNTHETIC"}, step=2,
                ))
                require(not (parent / "checkpoint-00000002").exists())
                require(private_inventory(checkpoint) == frozen)
                require(any(path.name.startswith(".checkpoint-00000002.")
                            for path in parent.iterdir()))
                report["write_cap_rejected"] = True
                require(os.fstat(parent_fd).st_ino == parent.stat().st_ino)
                require(not torch.cuda.is_initialized())
            resources = supervisor.summary()["resources_reserved"]
            report.update({key: resources[key]
                           for key in ("checkpoint_read_bytes", "private_write_bytes")})
            report.update(counters)
            report.update({"status": "passed", "scenarios_passed": 5,
                           "real_torch_dcp_used": True,
                           "explicit_dcp_storage_verified": True,
                           "serializer_filelikes_verified": True,
                           "stock_filesystem_io_blocked": True})
        finally:
            os.close(parent_fd)
    STAGE = "complete"
    return report


def main():
    try:
        result = run()
    except BaseException:
        # Deliberately no caught exception, model values, paths or traceback text.
        diagnostic = {"status": "failed", "stage": STAGE}
        implementation = sys.modules.get(PACKAGE + ".posttraining_sft_checkpoint_worker_v1")
        if implementation is not None and CURRENT_WORKER is not None:
            stage = getattr(CURRENT_WORKER, "failure_stage", None)
            code = getattr(CURRENT_WORKER, "failure_code", None)
            if (type(stage) is str and stage in implementation.STAGES
                    and type(code) is str and code in implementation.ERROR_CODES):
                diagnostic.update({"failure_stage": stage, "failure_code": code})
        print(json.dumps(diagnostic, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
