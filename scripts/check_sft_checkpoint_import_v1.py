#!/usr/bin/env python3
"""Bound metadata and stdlib-fixture audit; incomplete ML tests cannot be PASS."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path

from lumi import posttraining_checkpoint_import_v1 as adapter

PLAN = "configs/lumi-4b-sft-checkpoint-import-development-v0.1.json"
REPORT = "evidence/lumi-4b-sft-checkpoint-import-development-v0.1.json"
VERIFY = "evidence/lumi-4b-sft-checkpoint-import-development-verification-v0.1.json"
EXPECTED_PATHS = {
    "base": "configs/lumi-4b-final-base-binding-v0.1.json",
    "prior_plan": "configs/lumi-4b-sft-gpu-smoke-readiness-v0.1.json",
    "prior_report": "evidence/lumi-4b-sft-gpu-smoke-readiness-v0.1.json",
    "prior_verification": "evidence/lumi-4b-sft-gpu-smoke-readiness-verification-v0.1.json",
    "adapter": "src/lumi/posttraining_checkpoint_import_v1.py",
    "backend": "src/lumi/posttraining_checkpoint_import_torch_v1.py",
    "stdlib_tests": "tests/test_sft_checkpoint_import_v1.py",
    "genuine_tests": "tests/test_sft_checkpoint_import_torch_v1.py",
    "audit_script": "scripts/check_sft_checkpoint_import_v1.py",
    "consumer": "src/lumi/posttraining_sft_runtime_v4.py",
    "config_class": "src/lumi/configuration_lumi.py",
    "model_class": "src/lumi/modeling_lumi.py",
    "parameter_count": "src/lumi/parameter_count.py",
}


def build_report(root, expected_plan_hash):
    plan_raw = adapter.read_regular(root, PLAN)
    adapter.require(adapter.sha(plan_raw) == expected_plan_hash, "plan_hash_mismatch")
    plan = adapter.json_object(plan_raw)
    adapter.require(plan["schema_version"] == "lumi.sft-checkpoint-import-development.v0.1",
                    "plan_schema")
    adapter.require(plan["execution_scope"] == "local_metadata_and_stdlib_synthetic_only",
                    "plan_scope")
    adapter.require(all(v is False for v in plan["authorization_boundary"].values())
                    and set(plan["authorization_boundary"]) == {
                        "network", "real_checkpoint_read", "dataset_read", "gpu", "inference",
                        "training", "remote_action", "production_eligible", "runtime_install",
                    }, "plan_boundary")
    adapter.require(set(plan["bindings"]) == set(EXPECTED_PATHS), "binding_set")
    blobs = {}
    for key, relative in EXPECTED_PATHS.items():
        bound = plan["bindings"][key]
        adapter.require(bound["path"] == relative, "binding_path")
        raw = adapter.read_regular(root, relative)
        adapter.require(adapter.sha(raw) == bound["sha256"], "binding_hash")
        blobs[key] = raw
    prior = adapter.json_object(blobs["prior_report"])
    prior_verify = adapter.json_object(blobs["prior_verification"])
    for value, field in ((prior, "report_sha256"), (prior_verify, "verification_sha256")):
        adapter.require(value[field] == adapter.sha(adapter.canonical(
            {k: v for k, v in value.items() if k != field})), "prior_internal_hash")
    adapter.require(prior_verify["report_file_sha256"] == adapter.sha(blobs["prior_report"]),
                    "prior_verification_link")
    adapter.require(prior["plan_file_sha256"] == adapter.sha(blobs["prior_plan"]),
                    "prior_plan_link")
    for key in ("adapter", "backend", "stdlib_tests", "genuine_tests", "audit_script"):
        ast.parse(blobs[key])
    spec = importlib.util.spec_from_file_location("import_contract_fixture_tests",
                                               root / EXPECTED_PATHS["stdlib_tests"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output).run(suite)
    adapter.require(result.wasSuccessful() and not result.skipped and result.testsRun == 31,
                    "stdlib_synthetic_tests_failed")
    availability = {name: importlib.util.find_spec(name) is not None
                    for name in ("torch", "transformers", "safetensors", "numpy")}
    # A dependency change is not permission to import/execute tensor operations.
    # The successor must deliberately invoke the genuine suite and freeze results.
    count = sum(isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
                for n in ast.walk(ast.parse(blobs["genuine_tests"])))
    report = {
        "schema_version": "lumi.sft-checkpoint-import-development-report.v0.1",
        "status": "PARTIAL_contract_tests_pass_genuine_import_pending",
        "plan_file_sha256": expected_plan_hash,
        "bindings": plan["bindings"],
        "stdlib_tests": {"run": result.testsRun, "failures": len(result.failures),
                         "errors": len(result.errors), "skipped": len(result.skipped),
                         "mocked_packaging_is_not_real_tensor_evidence": True},
        "genuine_tensor_tests": {
            "defined": count, "run": 0, "passed": 0,
            "status": "PENDING_ML_runtime_recovery_and_genuine_test_execution",
        },
        "dependency_module_availability": availability,
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "prior_gate_review": "metadata_readiness_preserved_not_a_GPU_or_import_pass",
        "final_base_file_sha256": adapter.sha(blobs["base"]),
        "real_4b_import": False, "real_4b_receipt_created": False,
        "gpu_launch_ready": False, "production_eligible": False,
        "raw_training_text_retained": False,
        "executed_scope": {
            "metadata_and_code_hash_reads": True, "stdlib_temp_fixture_reads_and_writes": True,
            "real_checkpoint_or_dataset_content_reads": 0, "network_bytes": 0,
            "gpu_use": 0, "inference_calls": 0, "training_steps": 0,
            "runtime_installs": 0, "remote_actions": 0,
        },
        "next_required": [
            "recover_pinned_local_ML_runtime_under_separate_bounded_install_scope",
            "execute_14_genuine_tiny_CPU_import_tests_including_both_worker_integrations",
            "review_failures_and_freeze_successor_evidence_before_claiming_adapter_complete",
            "separately_bind_and_approve_actual_final_4b_export_import_operation",
        ],
    }
    report["report_sha256"] = adapter.sha(adapter.canonical(report))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.repository_root.resolve()
    target = VERIFY if args.verify else REPORT
    adapter.require(not (root / target).exists(), "output_exists_no_overwrite")
    result = build_report(root, args.expected_plan_sha256)
    if args.verify:
        raw = adapter.read_regular(root, REPORT)
        adapter.require(adapter.json_object(raw) == result, "fresh_replay_mismatch")
        result = {
            "schema_version": "lumi.sft-checkpoint-import-development-verification.v0.1",
            "status": "VERIFIED_partial_state_genuine_import_still_pending",
            "method": "fresh_process_same_implementation_structural_replay",
            "report_file_sha256": adapter.sha(raw),
            "report_internal_sha256": result["report_sha256"],
            "plan_file_sha256": args.expected_plan_sha256,
            "stdlib_tests_replayed": result["stdlib_tests"]["run"],
            "genuine_tensor_tests_run": 0, "real_4b_import": False,
            "production_eligible": False, "gpu_launch_ready": False,
        }
        result["verification_sha256"] = adapter.sha(adapter.canonical(result))
    adapter.write_new(root, target, json.dumps(result, sort_keys=True, indent=2).encode() + b"\n")
    print(result["status"])


if __name__ == "__main__":
    main()
