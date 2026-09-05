#!/usr/bin/env python3
"""Publish/verify a private source-review snapshot and stdlib synthetic results.

Not an SFT runner. No cloud, data/weight reads, installation, or Git mutation.
Publication and verification each consume a distinct exclusive started marker.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import signal
import stat
import sys
import time
import unittest
from pathlib import Path

if not sys.flags.no_site or not sys.dont_write_bytecode:
    print("STOP_use_python_minus_S_minus_B")
    raise SystemExit(1)

from lumi import posttraining_sft_source_bundle_v1 as bundle

ROOT = Path(__file__).resolve().parents[1]
PLAN = "configs/lumi-4b-sft-local-readiness-v0.1.json"
OUTPUT = "artifacts/lumi-sft-local-readiness-v0.1/attempt-001/private"
ARCHIVE = OUTPUT + "/source-review-only.zip"
REPORT = "evidence/lumi-4b-sft-local-readiness-v0.1.json"
VERIFICATION = "evidence/lumi-4b-sft-local-readiness-verification-v0.1.json"
OLD_PLAN = "configs/lumi-4b-sft-launcher-development-v0.1.json"
OLD_PLAN_SHA = "c98e5283c8edaf16a6a9855e72b5a81d010e635f6d62cc554e9760e1cd83e41b"
SEEDS = [
    "scripts/train_distributed_sft_v2.py",
    "src/lumi/posttraining_checkpoint_import_v2.py",
    "src/lumi/posttraining_checkpoint_import_torch_v2.py",
    "src/lumi/posttraining_sft_launch_contract_v1.py",
    "src/lumi/posttraining_sft_supervisor_v1.py",
    "src/lumi/posttraining_sft_worker_admission_v1.py",
    "src/lumi/posttraining_sft_source_bundle_v1.py",
    "scripts/check_sft_local_readiness_v1.py",
    "tests/test_sft_source_bundle_v1.py",
    "tests/test_sft_supervisor_v1.py",
    "tests/test_sft_worker_admission_v1.py",
    "tests/test_sft_launch_contract_v1.py",
]
TESTS = {
    "tests/test_sft_source_bundle_v1.py": 26,
    "tests/test_sft_supervisor_v1.py": 51,
    "tests/test_sft_worker_admission_v1.py": 36,
    "tests/test_sft_launch_contract_v1.py": 54,
}
CONTROLS = sorted(bundle.ALLOWED_CONTROL)
LIMITS = {
    "maximum_runtime_seconds_per_pass": 600,
    "maximum_source_bytes": bundle.MAX_SOURCE_BYTES,
    "maximum_archive_bytes": bundle.MAX_BUNDLE_BYTES,
    "maximum_published_bytes": 24 * 1024 * 1024,
    "maximum_report_bytes": 65536,
    "network_operations": 0,
    "child_processes": 0,
    "model_or_dataset_content_reads": 0,
    "gpu_use": 0,
    "inference_calls": 0,
}
WRITTEN = 0


def serialized(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def strict_document(raw: bytes) -> dict:
    def pairs(values):
        result = {}
        for key, value in values:
            bundle.require(key not in result, "duplicate_control_key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    bundle.require(type(value) is dict, "control_object_required")
    bundle.canonical(value)
    return value


def safe_read(name: str, maximum: int) -> bytes:
    # Caller selects these control paths; never accept user-supplied source paths.
    path = ROOT / name
    bundle.require(path.is_relative_to(ROOT) and ".." not in Path(name).parts, "control_path")
    for parent in path.parents:
        if parent == ROOT:
            break
        bundle.require(not parent.is_symlink(), "control_symlink_parent")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb", buffering=0) as stream:
        before = os.fstat(stream.fileno())
        bundle.require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum,
                       "control_file_shape_or_size")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    bundle.require(len(raw) == before.st_size and (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                   == (after.st_size, after.st_mtime_ns, after.st_ctime_ns), "control_changed")
    return raw


def write_new(name: str, raw: bytes) -> None:
    global WRITTEN
    bundle.require(name in {ARCHIVE, REPORT, VERIFICATION, REPORT + ".started", VERIFICATION + ".started"},
                   "output_not_allowlisted")
    bundle.require(WRITTEN + len(raw) <= LIMITS["maximum_published_bytes"], "publication_write_cap")
    if name != ARCHIVE:
        bundle.require(len(raw) <= LIMITS["maximum_report_bytes"], "report_write_cap")
    path = ROOT / name
    current = ROOT
    for part in Path(name).parts[:-1]:
        current = current / part
        bundle.require(stat.S_ISDIR(current.lstat().st_mode), "output_parent_not_directory")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    WRITTEN += len(raw)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def make_private_output() -> None:
    # No chmod of an existing artifact tree; exact fresh attempt only.
    parent = ROOT / "artifacts"
    bundle.require(stat.S_ISDIR(parent.lstat().st_mode), "artifact_parent_not_directory")
    for part in Path(OUTPUT).parts[1:]:
        parent = parent / part
        os.mkdir(parent, 0o700)


def guard_io() -> dict:
    counters = {"prohibited_operation_attempts": 0}
    forbidden = {"torch", "transformers", "tokenizers", "safetensors", "numpy", "boto3", "botocore", "requests", "httpx"}
    bundle.require(not any(name.split(".")[0] in forbidden for name in sys.modules), "forbidden_module_already_loaded")
    def audit(event, args):
        denied = event.startswith("socket.") or event in {
            "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.fork", "os.forkpty", "pty.spawn", "ctypes.dlopen"
        }
        if event == "import" and args[0].split(".")[0] in forbidden:
            denied = True
        if denied:
            counters["prohibited_operation_attempts"] += 1
            raise PermissionError("local_source_and_synthetic_only")
    sys.addaudithook(audit)
    return counters


class DiscardText(io.TextIOBase):
    def write(self, value):
        return len(value)


def run_tests(files: dict[str, bytes]) -> dict:
    results = {}
    sink = DiscardText()
    for name, expected in TESTS.items():
        module_name = "bounded_local_readiness_" + Path(name).stem
        spec = importlib.util.spec_from_file_location(module_name, ROOT / name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(files[name], str(ROOT / name), "exec"), module.__dict__)
            suite = unittest.defaultTestLoader.loadTestsFromModule(module)
            outcome = unittest.TextTestRunner(stream=sink).run(suite)
        results[Path(name).stem] = {"run": outcome.testsRun, "failures": len(outcome.failures),
                                  "errors": len(outcome.errors), "skipped": len(outcome.skipped)}
        bundle.require(outcome.wasSuccessful() and not outcome.skipped and outcome.testsRun == expected,
                       "synthetic_suite_failed")
    return results


def verify_preserved(plan: dict) -> int:
    for name, expected in plan["preserved_bindings"].items():
        bundle.require(bundle.digest(safe_read(name, bundle.MAX_FILE_BYTES)) == expected, "historical_binding_changed")
    return len(plan["preserved_bindings"])


def run(mode: str, expected_plan: str) -> dict:
    start = time.monotonic()
    counters = guard_io()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("local_deadline")))
    signal.alarm(LIMITS["maximum_runtime_seconds_per_pass"])
    plan_raw = safe_read(PLAN, 65536)
    bundle.require(bundle.digest(plan_raw) == expected_plan, "plan_hash_mismatch")
    plan = strict_document(plan_raw)
    bundle.require(plan["schema_version"] == "lumi.sft-local-readiness-development.v0.1"
                   and plan["scope"] == "local_source_review_and_synthetic_tests_only"
                   and plan["gpu_execution_authorized"] is False and plan["production_eligible"] is False,
                   "plan_scope_mismatch")
    bundle.require(plan["limits"] == LIMITS and plan["seeds"] == SEEDS
                   and plan["controls"] == CONTROLS and plan["tests"] == TESTS, "plan_selection_changed")
    bundle.require(bundle.digest(safe_read(OLD_PLAN, 65536)) == OLD_PLAN_SHA, "historical_plan_changed")
    old_plan = strict_document(safe_read(OLD_PLAN, 65536))
    expected_preserved = {v["path"]: v["sha256"] for v in old_plan["bindings"].values()}
    expected_preserved[OLD_PLAN] = OLD_PLAN_SHA
    # Additional previous report/verification/markers are pinned in this new plan.
    bundle.require(all(plan["preserved_bindings"].get(k) == v for k, v in expected_preserved.items()),
                   "historical_preservation_set_changed")
    preserved = verify_preserved(plan)
    marker = REPORT + ".started" if mode == "publish" else VERIFICATION + ".started"
    if mode != "check":
        write_new(marker, b"STARTED-local-source-readiness-v1-no-retry\n")
    files, inspection = bundle.collect(ROOT, SEEDS, CONTROLS)
    actual = {name: {"bytes": len(raw), "sha256": bundle.digest(raw)} for name, raw in files.items()}
    bundle.require(actual == plan["source_bindings"], "source_bindings_changed")
    tests = run_tests(files)
    archive, manifest = bundle.make_bundle(files, inspection=inspection)
    verification = bundle.verify_bundle(archive, bundle.digest(archive), files)
    again, _ = bundle.collect(ROOT, SEEDS, CONTROLS)
    bundle.require(again == files, "source_changed_after_tests")
    bundle.require(verify_preserved(plan) == preserved, "preservation_count_changed")
    bundle.require(not counters["prohibited_operation_attempts"], "prohibited_attempt")
    result = {
        "schema_version": "lumi.sft-local-readiness-report.v0.1",
        "status": "PASS_local_source_review_and_synthetic_tests_only",
        "mode": mode, "plan_file_sha256": expected_plan, "tests": tests,
        "synthetic_tests_passed": sum(item["run"] for item in tests.values()),
        "source_file_count": manifest["file_count"], "source_bytes": manifest["source_bytes"],
        "bundle_bytes": len(archive), "bundle_sha256": bundle.digest(archive),
        "preserved_bindings_verified_before_and_after": preserved,
        "source_bytes_verified_before_and_after": True,
        "inspection": inspection, "bundle_verification": verification,
        "runtime_seconds": round(time.monotonic() - start, 6),
        "network_operations": 0, "gpu_use": 0, "inference_calls": 0, "child_processes": 0,
        "real_model_or_dataset_content_reads": 0, "synthetic_temporary_files_used": True,
        "guard_is_not_os_sandbox": True, **counters,
        "source_bundle_is_not_deployment_release": True,
        "cooperative_supervisor_is_not_process_shared_or_os_enforcing": True,
        "integrated_gpu_worker_ready": False, "gpu_launch_ready": False,
        "production_eligible": False, "remote_git_push_performed": False,
    }
    if mode == "verify":
        previous_raw = safe_read(REPORT, 65536)
        previous = strict_document(previous_raw)
        previous_hash = previous.pop("internal_sha256")
        bundle.require(bundle.digest(bundle.canonical(previous)) == previous_hash, "previous_report_hash")
        for key in ("plan_file_sha256", "tests", "source_file_count", "source_bytes", "bundle_bytes", "bundle_sha256"):
            bundle.require(previous[key] == result[key], "fresh_replay_mismatch")
        saved_archive = safe_read(ARCHIVE, bundle.MAX_BUNDLE_BYTES)
        result["saved_archive_verification"] = bundle.verify_bundle(saved_archive, previous["bundle_sha256"], files)
        bundle.require(saved_archive == archive, "archive_rebuild_mismatch")
        result["report_file_sha256"] = bundle.digest(previous_raw)
        result["fresh_process_replay_matches"] = True
    result["internal_sha256"] = bundle.digest(bundle.canonical(result))
    if mode == "publish":
        make_private_output()
        write_new(ARCHIVE, archive)
        write_new(REPORT, serialized(result))
    elif mode == "verify":
        write_new(VERIFICATION, serialized(result))
    signal.alarm(0)
    return {"status": result["status"], "mode": mode, "synthetic_tests_passed": result["synthetic_tests_passed"],
            "source_file_count": len(files), "bundle_bytes": len(archive), "bundle_sha256": result["bundle_sha256"],
            "published_bytes_this_pass": WRITTEN, "gpu_launch_ready": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "publish", "verify"), default="check")
    parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        print(json.dumps(run(args.mode, args.expected_plan_sha256), sort_keys=True))
    except BaseException:
        # No raw test failure, filename, source line or exception text is retained.
        print("STOP_local_readiness_check_failed_no_retry_if_started")
        raise SystemExit(1) from None
