#!/usr/bin/env python3
"""Bounded local POSIX synthetic-process/control tests; no real SFT dispatch."""

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

ROOT = Path(__file__).resolve().parents[1]
PLAN = "configs/lumi-4b-sft-runtime-controls-v0.1.json"
REPORT = "evidence/lumi-4b-sft-runtime-controls-v0.1.json"
VERIFICATION = "evidence/lumi-4b-sft-runtime-controls-verification-v0.1.json"
OLD_PLAN = "configs/lumi-4b-sft-local-readiness-v0.1.json"
OLD_SHA = "5a7ab18722864814a67a76fab60f8221d89057b333d2afcf45fc2ddb332ae268"
TESTS = {
    "tests/test_process_watchdog_v1.py": 33,
    "tests/test_sft_guarded_bootstrap_v1.py": 30,
    "tests/test_sft_ledger_protocol_v1.py": 43,
}
SOURCES = [
    "src/lumi/posttraining_process_watchdog_v1.py",
    "scripts/sft_guarded_bootstrap_v1.py",
    "src/lumi/posttraining_sft_ledger_protocol_v1.py",
    "src/lumi/posttraining_sft_supervisor_v1.py",
    "scripts/check_sft_runtime_controls_v1.py",
    "configs/lumi-4b-sft-next-run-budget-v0.1.json",
    *TESTS,
]
LIMITS = {"maximum_runtime_seconds_per_pass": 120,
          "maximum_control_file_bytes": 1048576, "maximum_control_read_bytes_per_pass": 16777216,
          "maximum_published_bytes_per_pass": 65536,
          "real_model_or_dataset_reads": 0, "external_network": 0, "gpu_use": 0, "inference_calls": 0}
READ_BYTES = WRITE_BYTES = 0


def require(condition, code):
    if not condition:
        raise ValueError(code) from None


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def document(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate_json_key")
            out[key] = value
        return out
    value = json.loads(raw, object_pairs_hook=pairs)
    require(type(value) is dict, "json_object_required")
    canonical(value)
    return value


def read_bound(name, expected=None):
    global READ_BYTES
    require(type(name) is str and not Path(name).is_absolute() and ".." not in Path(name).parts
            and name.split("/")[0] in {"configs", "scripts", "src", "tests", "docs", "evidence"}, "unsafe_control_path")
    path = ROOT / name
    require(path.resolve() == path, "symlink_control_path")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb", buffering=0) as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size <= LIMITS["maximum_control_file_bytes"],
                "invalid_control_file")
        require(READ_BYTES + info.st_size <= LIMITS["maximum_control_read_bytes_per_pass"], "control_read_cap")
        raw = stream.read(info.st_size)
        READ_BYTES += len(raw)
        after = os.fstat(stream.fileno())
    require(len(raw) == info.st_size and (info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            == (after.st_size, after.st_mtime_ns, after.st_ctime_ns), "control_changed")
    require(expected is None or sha(raw) == expected, "control_hash_changed")
    return raw


def write_new(name, raw):
    global WRITE_BYTES
    require(name in {REPORT, VERIFICATION, REPORT + ".started", VERIFICATION + ".started"}, "output_not_allowed")
    require(WRITE_BYTES + len(raw) <= LIMITS["maximum_published_bytes_per_pass"], "publication_cap")
    path = ROOT / name
    require(path.parent.resolve() == path.parent, "unsafe_output_parent")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    WRITE_BYTES += len(raw)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


class DiscardText(io.TextIOBase):
    def write(self, text):
        return len(text)


def install_guard():
    counts = {"observed_direct_popen_events": 0, "blocked_network_or_ml_attempts": 0}
    banned = {"torch", "transformers", "tokenizers", "safetensors", "boto3", "botocore", "requests", "httpx"}
    require(not any(n.split(".")[0] in banned for n in sys.modules), "ml_already_loaded")
    def guard(event, args):
        if event.startswith("socket.") or (event == "import" and args[0].split(".")[0] in banned):
            counts["blocked_network_or_ml_attempts"] += 1
            raise PermissionError("local_synthetic_boundary")
        if event == "subprocess.Popen":
            counts["observed_direct_popen_events"] += 1
    sys.addaudithook(guard)
    return counts


def run(mode, expected_sha):
    require(sys.flags.no_site and sys.dont_write_bytecode, "python_flags_required")
    started = time.monotonic()
    counts = install_guard()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("local_test_deadline")))
    signal.alarm(LIMITS["maximum_runtime_seconds_per_pass"])
    plan = document(read_bound(PLAN, expected_sha))
    require(plan["schema_version"] == "lumi.sft-runtime-controls-development.v0.1"
            and plan["scope"] == "local_synthetic_process_and_control_tests_only"
            and plan["limits"] == LIMITS and plan["tests"] == TESTS
            and set(plan["bindings"]) == set(SOURCES) and plan["gpu_launch_ready"] is False,
            "plan_schema_or_scope_changed")
    old = document(read_bound(OLD_PLAN, OLD_SHA))
    preserved = {**old["preserved_bindings"], **{n: b["sha256"] for n, b in old["source_bindings"].items()}}
    preserved[OLD_PLAN] = OLD_SHA
    for name, expected in preserved.items():
        read_bound(name, expected)
    loaded = {n: read_bound(n, plan["bindings"][n]) for n in SOURCES}
    output = REPORT if mode == "publish" else VERIFICATION
    if mode != "check":
        write_new(output + ".started", b"STARTED-runtime-controls-v1-no-retry\n")
    results = {}
    sink = DiscardText()
    for name, expected in TESTS.items():
        module_name = "runtime_control_test_" + Path(name).stem
        spec = importlib.util.spec_from_file_location(module_name, ROOT / name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(loaded[name], str(ROOT / name), "exec"), module.__dict__)
            result = unittest.TextTestRunner(stream=sink).run(unittest.defaultTestLoader.loadTestsFromModule(module))
        results[Path(name).stem] = {"run": result.testsRun, "failures": len(result.failures),
                                  "errors": len(result.errors), "skipped": len(result.skipped)}
        require(result.wasSuccessful() and result.testsRun == expected and not result.skipped, "synthetic_tests_failed")
    for name, expected in preserved.items():
        read_bound(name, expected)
    for name in SOURCES:
        read_bound(name, plan["bindings"][name])
    require(counts["blocked_network_or_ml_attempts"] == 0, "prohibited_attempt")
    result = {
        "schema_version": "lumi.sft-runtime-controls-report.v0.1", "mode": mode,
        "status": "PASS_local_runtime_control_components_only", "plan_file_sha256": expected_sha,
        "tests": results, "total_tests_passed": sum(r["run"] for r in results.values()),
        "bindings": plan["bindings"], "historical_binding_count_verified_before_and_after": len(preserved),
        "real_posix_synthetic_process_tests": True, "local_synthetic_fixture_files_used": True,
        "child_text_retained_in_report": False, "model_or_dataset_reads": 0,
        "network_operations_in_test_workflow": 0, "gpu_use": 0, "inference_calls": 0,
        "parent_audit_hook_is_not_inherited_os_sandbox": True,
        "direct_popen_events_exclude_grandchild_forks": True, **counts,
        "integrated_gpu_worker_ready": False, "linux_cgroup_verified": False,
        "ledger_transport_and_fresh_process_epochs_integrated": False,
        "real_receipts_authenticated": False, "cloud_instance_launched": False,
        "cloud_billing_enforcement_integrated": False, "git_commit_or_push_performed": False,
        "production_eligible": False, "elapsed_seconds": round(time.monotonic() - started, 6),
        "counted_control_read_bytes": READ_BYTES,
    }
    if mode == "verify":
        raw = read_bound(REPORT)
        previous = document(raw)
        internal = previous.pop("internal_sha256")
        require(sha(canonical(previous)) == internal, "report_internal_hash_changed")
        for field in ("status", "plan_file_sha256", "tests", "bindings", "total_tests_passed",
                      "historical_binding_count_verified_before_and_after", "observed_direct_popen_events"):
            require(previous[field] == result[field], "fresh_process_replay_mismatch")
        result["publication_report_file_sha256"] = sha(raw)
        result["fresh_process_replay_matches"] = True
    result["internal_sha256"] = sha(canonical(result))
    if mode != "check":
        write_new(output, json.dumps(result, sort_keys=True, indent=2).encode() + b"\n")
    signal.alarm(0)
    return {"status": result["status"], "mode": mode, "tests_passed": result["total_tests_passed"],
            "historical_bindings_preserved": len(preserved), "direct_popen_events": counts["observed_direct_popen_events"],
            "published_bytes": WRITE_BYTES, "gpu_launch_ready": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "publish", "verify"), default="check")
    parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        print(json.dumps(run(args.mode, args.expected_plan_sha256), sort_keys=True))
    except BaseException:
        print("STOP_runtime_controls_no_retry_after_started")
        raise SystemExit(1) from None
