#!/usr/bin/env python3
"""Offline genuine tiny CPU import tests and contract-suite evidence, no training."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "artifacts/lumi-sft-import-runtime-v0.1"
PLAN = ROOT / "configs/lumi-4b-sft-import-runtime-test-bindings-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-import-runtime-test-v0.1.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-import-runtime-test-verification-v0.1.json"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_exclusive(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    target = VERIFICATION if args.verify else REPORT
    if target.exists():
        raise FileExistsError("immutable_test_evidence_already_exists")
    raw_plan = PLAN.read_bytes()
    if digest(raw_plan) != args.expected_plan_sha256:
        raise ValueError("test_plan_hash_mismatch")
    plan = json.loads(raw_plan)
    if plan["scope"] != "offline_synthetic_cpu_import_tests_only":
        raise ValueError("test_scope_mismatch")
    if sys.executable != str(RUNTIME / "venv/bin/python"):
        raise ValueError("test_interpreter_mismatch")
    for binding in plan["bindings"].values():
        path = ROOT / binding["path"]
        if digest(path.read_bytes()) != binding["sha256"]:
            raise ValueError("frozen_code_or_input_changed")
    install = json.loads((RUNTIME / "pip-install-report.json").read_bytes())
    versions = {}
    for entry in install["install"]:
        name, version = entry["metadata"]["name"], entry["metadata"]["version"]
        if importlib.metadata.version(name) != version:
            raise ValueError("installed_package_version_changed")
        versions[name] = version
    prohibited_events = 0

    def guard(event, values):
        nonlocal prohibited_events
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo",
                     "subprocess.Popen", "os.system"}:
            prohibited_events += 1
            raise PermissionError("offline_tests_prohibit_network_or_child_launch")
        if event == "open" and isinstance(values[0], (str, bytes)):
            name = os.fsdecode(values[0])
            path = Path(name).absolute()
            forbidden = (path.is_relative_to(ROOT / "data") or
                         path.is_relative_to(ROOT.parent / "lumi-artifacts") or
                         path.is_relative_to(ROOT / "checkpoints") or
                         (path.is_relative_to(ROOT / "artifacts")
                          and not path.is_relative_to(RUNTIME)))
            if forbidden:
                prohibited_events += 1
                raise PermissionError("real_dataset_or_checkpoint_access_prohibited")

    sys.addaudithook(guard)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                      CUDA_VISIBLE_DEVICES="", PYTORCH_ENABLE_MPS_FALLBACK="0")
    import torch

    from lumi.modeling_lumi import LumiForCausalLM

    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    results = {}
    started = time.monotonic()
    stream = io.StringIO()
    # These are stronger than offline flags alone: no model forward or optimizer
    # steps can happen during either suite, including supposedly synthetic setup.
    with (
        patch.object(LumiForCausalLM, "forward", side_effect=AssertionError("forward_forbidden")),
        patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer_forbidden")),
    ):
        for suite_name, specification in plan["test_suites"].items():
            path = ROOT / specification["path"]
            spec = importlib.util.spec_from_file_location("bounded_" + suite_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            suite = unittest.defaultTestLoader.loadTestsFromModule(module)
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                result = unittest.TextTestRunner(stream=stream).run(suite)
            results[suite_name] = {
                "run": result.testsRun, "failures": len(result.failures),
                "errors": len(result.errors), "skipped": len(result.skipped),
                "failed_test_ids": [test.id() for test, _ in result.errors + result.failures],
                "passed": (result.wasSuccessful() and not result.skipped
                           and result.testsRun == specification["expected_tests"]),
            }
    success = all(x["passed"] for x in results.values()) and prohibited_events == 0
    report = {
        "schema_version": "lumi.sft-import-runtime-test.v0.1",
        "status": "PASS_genuine_tiny_cpu_import_and_contract_tests" if success else
                  "FAIL_synthetic_import_tests_no_real_4b_action",
        "plan_file_sha256": args.expected_plan_sha256,
        "test_suites": results,
        "runtime_versions": versions,
        "runtime_seconds": round(time.monotonic() - started, 6),
        "prohibited_action_attempts": prohibited_events,
        "real_4b_import": False, "real_4b_receipt_created": False,
        "gpu_launch_ready": False, "production_eligible": False,
        "executed_network_bytes": 0, "gpu_use": 0, "inference_calls": 0,
        "optimizer_steps": 0, "real_dataset_or_checkpoint_content_reads": 0,
        "raw_tensor_or_training_text_retained": False,
        "test_stdout_stderr_retained": False,
        "import_fixture_files": "temporary_synthetic_only_removed_by_unittest_cleanup",
    }
    report["report_sha256"] = digest(canonical(report))
    if args.verify:
        previous_raw = REPORT.read_bytes()
        previous = json.loads(previous_raw)
        expected_hash = previous.pop("report_sha256")
        if digest(canonical(previous)) != expected_hash:
            raise ValueError("execution_report_internal_hash_mismatch")
        ignore = {"report_sha256", "runtime_seconds"}
        def stable(value):
            return {k: v for k, v in value.items() if k not in ignore}
        if not success or stable(previous) != stable(report):
            raise ValueError("fresh_test_replay_did_not_match_success")
        report = {
            "schema_version": "lumi.sft-import-runtime-test-verification.v0.1",
            "status": "PASS_fresh_process_genuine_tiny_cpu_test_replay",
            "method": "same_test_implementation_fresh_process_not_independent_tensor_algorithm",
            "plan_file_sha256": args.expected_plan_sha256,
            "report_file_sha256": digest(previous_raw), "report_internal_sha256": expected_hash,
            "replayed_tests": results, "runtime_seconds": report["runtime_seconds"],
            "real_4b_import": False, "production_eligible": False, "gpu_launch_ready": False,
        }
        report["verification_sha256"] = digest(canonical(report))
    write_exclusive(target, report)
    print(report["status"])
    if not success:
        print(json.dumps(results, sort_keys=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
