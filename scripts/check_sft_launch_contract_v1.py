#!/usr/bin/env python3
"""Check the local SFT contract and synthetic tests; NEVER initialize or launch ML."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
import signal
import sys
import time
import types
import unittest
from pathlib import Path

# The historical lumi package auto-registers the ML stack when site-packages are
# visible. Require stdlib-only startup before importing that package; do not alter
# its frozen historical behavior or pretend an installed ML environment is inert.
if not sys.flags.no_site or not sys.dont_write_bytecode:
    print("STOP_use_python_minus_S_minus_B_for_stdlib_only_check")
    raise SystemExit(1)

from lumi.posttraining_gpu_smoke_readiness_v1 import MetadataReader, safe_path
from lumi.posttraining_sft_launch_contract_v1 import (
    analyze,
    deny_execution,
    digest,
    make_draft,
    require,
    strict_json,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = "configs/lumi-4b-sft-launcher-development-v0.1.json"
CONTRACT = "configs/lumi-4b-sft-gpu-smoke-launch-contract-v0.1.json"
REPORT = "evidence/lumi-4b-sft-launcher-development-v0.1.json"
VERIFICATION = "evidence/lumi-4b-sft-launcher-development-verification-v0.1.json"
BINDING_PATHS = {
    "base": "configs/lumi-4b-final-base-binding-v0.1.json",
    "data_contract": "configs/lumi-4b-sft-data-contract-v0.2.json",
    "source_registry": "configs/lumi-4b-sft-source-registry-v0.24.json",
    "cpu_plan": "configs/lumi-4b-sft-distributed-cpu-preflight-v0.2.json",
    "cpu_report": "evidence/lumi-4b-sft-distributed-cpu-preflight-v0.2.json",
    "cpu_verification": "evidence/lumi-4b-sft-distributed-cpu-preflight-verification-v0.2.json",
    "seal": "evidence/lumi-4b-sft-learning-candidate-token-consistent-seal-v0.2.json",
    "shard_receipt": "evidence/lumi-4b-sft-token-consistent-shards-v0.2.json",
    "shard_verification": "evidence/lumi-4b-sft-token-consistent-shards-verification-v0.2.json",
    "import_test_plan": "configs/lumi-4b-sft-import-runtime-test-bindings-v0.1.json",
    "import_report": "evidence/lumi-4b-sft-import-runtime-test-v0.1.json",
    "import_verification": "evidence/lumi-4b-sft-import-runtime-test-verification-v0.1.json",
    "readiness_plan": "configs/lumi-4b-sft-gpu-smoke-readiness-v0.1.json",
    "legacy_worker_cli": "scripts/train_distributed_sft_v2.py",
    "legacy_worker": "src/lumi/posttraining_distributed_sft_v2.py",
    "legacy_authorization": "src/lumi/distributed_training.py",
    "data_loader": "src/lumi/posttraining_sft_runtime_v4.py",
    "metadata_reader": "src/lumi/posttraining_gpu_smoke_readiness_v1.py",
    "package_init": "src/lumi/__init__.py",
    "compiler": "src/lumi/posttraining_sft_launch_contract_v1.py",
    "tests": "tests/test_sft_launch_contract_v1.py",
    "runner": "scripts/check_sft_launch_contract_v1.py",
    "contract": CONTRACT,
}
LIMITS = {
    "maximum_runtime_seconds": 600,
    "maximum_metadata_read_bytes": 8_388_608,
    "maximum_evidence_write_bytes_per_pass": 65_536,
    "network_bytes": 0,
    "gpu_use": 0,
    "inference_calls": 0,
    "model_or_dataset_content_reads": 0,
    "child_processes": 0,
}


def install_guard() -> dict:
    """Defense in depth for fixed stdlib tests, not an arbitrary-code sandbox."""
    counts = {"prohibited_action_attempts": 0}
    reads = {str(ROOT / name) for name in set(BINDING_PATHS.values()) | {PLAN, REPORT}}
    writes = {
        str(ROOT / name)
        for name in (REPORT, VERIFICATION, REPORT + ".started", VERIFICATION + ".started")
    }
    runtime_roots = {Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve()}
    denied_modules = {
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "numpy",
        "boto3",
        "botocore",
        "requests",
        "httpx",
    }

    def guard(event, args):
        prohibited = event.startswith("socket.") or event in {
            "subprocess.Popen",
            "os.system",
            "os.exec",
            "os.posix_spawn",
            "os.fork",
            "os.forkpty",
            "pty.spawn",
            "ctypes.dlopen",
        }
        if event == "import" and args[0].split(".")[0] in denied_modules:
            prohibited = True
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0])).absolute()
            flags = args[2] or 0
            is_write = bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
            if is_write:
                prohibited = str(path) not in writes
            else:
                prohibited = str(path) not in reads and not any(
                    path.is_relative_to(runtime) for runtime in runtime_roots
                )
        if event in {"os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.link", "os.symlink"}:
            prohibited = True
        if prohibited:
            counts["prohibited_action_attempts"] += 1
            raise PermissionError("local_metadata_only_boundary")

    require(not (denied_modules & set(sys.modules)), "ml_or_transport_already_imported")
    sys.addaudithook(guard)
    sys.dont_write_bytecode = True
    return counts


class DiscardText(io.TextIOBase):
    def write(self, text):
        return len(text)


def synthetic_tests(expected_count: int, reader: MetadataReader) -> dict:
    module = types.ModuleType("bounded_sft_launcher_tests")
    # Execute only the already hash-checked source bytes, not an unbound .pyc cache.
    exec(
        compile(reader.read(BINDING_PATHS["tests"]), BINDING_PATHS["tests"], "exec"),
        module.__dict__,
    )
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    sink = DiscardText()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        result = unittest.TextTestRunner(stream=sink).run(suite)
    return {
        "run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "passed": (
            result.wasSuccessful() and not result.skipped and result.testsRun == expected_count
        ),
    }


def write_exclusive(relative: str, raw: bytes) -> None:
    require(len(raw) <= LIMITS["maximum_evidence_write_bytes_per_pass"], "evidence_write_cap")
    path = safe_path(ROOT, relative)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def serialized(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def legacy_order_findings(reader: MetadataReader) -> dict:
    cli = ast.parse(reader.read(BINDING_PATHS["legacy_worker_cli"]))
    worker = ast.parse(reader.read(BINDING_PATHS["legacy_worker"]))
    main = next(n for n in cli.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    train = next(
        n
        for n in worker.body
        if isinstance(n, ast.FunctionDef) and n.name == "train_distributed_sft"
    )

    def line_of(fn, name):
        return min(
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
        )

    require(
        line_of(main, "initialize_distributed_from_environment")
        < line_of(main, "train_distributed_sft"),
        "legacy_cli_order_changed",
    )
    require(
        line_of(train, "_collective_load_inputs") < line_of(train, "_validate_execution_scope"),
        "legacy_worker_order_changed",
    )
    require(
        any(
            isinstance(n, ast.ImportFrom)
            and n.module == "distributed_training"
            and any(a.name == "_validate_execution_scope" for a in n.names)
            for n in worker.body
        ),
        "legacy_authorization_import_changed",
    )
    return {
        "legacy_cli_initializes_distributed_before_worker_authorization": True,
        "legacy_worker_loads_inputs_before_authorization": True,
        "legacy_worker_uses_generic_pretraining_authorization": True,
        "legacy_worker_dispatched": False,
    }


def run(expected_sha: str, *, publish: bool, verify: bool) -> str:
    start = time.monotonic()
    counts = install_guard()
    allowed = set(BINDING_PATHS.values()) | {PLAN, REPORT}
    reader = MetadataReader(ROOT, allowed)
    plan = strict_json(reader.read(PLAN, expected_sha))
    require(
        plan["schema_version"] == "lumi.sft-launcher-development.v0.1"
        and plan["scope"] == "local_metadata_and_synthetic_tests_only",
        "development_scope_changed",
    )
    require(plan["limits"] == LIMITS, "development_limits_changed")
    require(set(plan["bindings"]) == set(BINDING_PATHS), "development_bindings_changed")
    for name, relative in BINDING_PATHS.items():
        binding = plan["bindings"][name]
        require(
            set(binding) == {"path", "sha256"} and binding["path"] == relative,
            "development_binding_path_changed",
        )
        reader.read(relative, binding["sha256"])
    target = VERIFICATION if verify else REPORT
    marker = serialized(
        {
            "schema_version": "lumi.sft-launcher-local-attempt.v0.1",
            "plan_file_sha256": expected_sha,
            "verification": verify,
        }
    )
    if publish:
        require(not os.path.lexists(ROOT / target), "immutable_output_exists")
        write_exclusive(target + ".started", marker)
    reference = make_draft(
        strict_json(reader.read(BINDING_PATHS["base"])),
        strict_json(reader.read(BINDING_PATHS["cpu_plan"]))["input"],
        strict_json(reader.read(BINDING_PATHS["readiness_plan"]))["launch_draft"],
    )
    result = analyze(strict_json(reader.read(CONTRACT)), reference)
    findings = legacy_order_findings(reader)
    tests = synthetic_tests(plan["expected_synthetic_tests"], reader)
    require(tests["passed"], "synthetic_tests_failed")
    require(counts["prohibited_action_attempts"] == 0, "prohibited_action_attempted")
    # Reopen every exact binding after tests; cached reads alone cannot prove preservation.
    after = MetadataReader(ROOT, allowed)
    after.read(PLAN, expected_sha)
    for binding in plan["bindings"].values():
        after.read(binding["path"], binding["sha256"])
    read_bytes = reader.bytes_read + after.bytes_read
    require(read_bytes <= LIMITS["maximum_metadata_read_bytes"], "combined_metadata_read_cap")
    report = {
        "schema_version": "lumi.sft-launcher-development-report.v0.1",
        "status": "PASS_local_launch_contract_and_synthetic_tests_execution_blocked",
        "plan_file_sha256": expected_sha,
        "result": result,
        "synthetic_tests": tests,
        "legacy_integration_findings": findings,
        "bound_files_checked_before_and_after": len(BINDING_PATHS),
        "bound_files_unchanged": True,
        "metadata_bytes_read": read_bytes,
        "runtime_seconds": round(time.monotonic() - start, 6),
        **counts,
        "network_bytes": 0,
        "gpu_use": 0,
        "inference_calls": 0,
        "optimizer_steps": 0,
        "dataset_or_model_content_reads": 0,
        "child_processes": 0,
        "s3_accessed": False,
        "gpu_launch_ready": False,
        "production_eligible": False,
        "raw_training_text_or_exception_text_retained": False,
        "runtime_or_cost_enforcement_for_gpu_proven": False,
        "boundary_measurement": "fixed_stdlib_code_and_audit_guard_not_os_packet_meter",
    }
    require(report["runtime_seconds"] <= LIMITS["maximum_runtime_seconds"], "local_runtime_cap")
    report["report_sha256"] = digest(report)
    if verify:
        previous_raw = reader.read(REPORT)
        previous = strict_json(previous_raw)
        previous_internal = previous.pop("report_sha256")
        require(digest(previous) == previous_internal, "report_internal_hash_mismatch")
        ignore = {"report_sha256", "runtime_seconds"}

        def stable(value):
            return {k: v for k, v in value.items() if k not in ignore}

        require(stable(previous) == stable(report), "fresh_replay_mismatch")
        report = {
            "schema_version": "lumi.sft-launcher-development-verification.v0.1",
            "status": "PASS_fresh_process_local_contract_replay_execution_blocked",
            "method": "same_implementation_fresh_process_not_independent_algorithm",
            "plan_file_sha256": expected_sha,
            "report_file_sha256": hashlib.sha256(previous_raw).hexdigest(),
            "report_internal_sha256": previous_internal,
            "synthetic_tests": tests,
            "bound_files_unchanged": True,
            "metadata_bytes_read_including_report": reader.bytes_read + after.bytes_read,
            "runtime_seconds": round(time.monotonic() - start, 6),
            "gpu_launch_ready": False,
            "production_eligible": False,
            "real_model_imported": False,
            "s3_accessed": False,
        }
        require(
            report["metadata_bytes_read_including_report"] <= LIMITS["maximum_metadata_read_bytes"],
            "verification_metadata_read_cap",
        )
        report["verification_sha256"] = digest(report)
    raw = serialized(report)
    require(
        len(raw) + len(marker) <= LIMITS["maximum_evidence_write_bytes_per_pass"],
        "combined_evidence_write_cap",
    )
    if publish:
        write_exclusive(target, raw)
    return report["status"]


def main(argv=None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    # This branch precedes parsing, plan/report reads, markers, and all test work.
    if "--execute" in arguments:
        deny_execution()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-plan-sha256", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args(arguments)

    def timeout(_signum, _frame):
        raise TimeoutError("local_development_runtime_cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(LIMITS["maximum_runtime_seconds"])
    try:
        print(
            run(args.expected_plan_sha256, publish=args.publish or args.verify, verify=args.verify)
        )
    finally:
        signal.alarm(0)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, PermissionError, OSError, KeyError, StopIteration, TimeoutError):
        # Never emit path, input, test ID, or exception text from the public runner.
        print("STOP_local_contract_check_failed_no_execution")
        raise SystemExit(1) from None
