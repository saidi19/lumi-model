"""Generated control/IO fixtures and kernel doubles only, never native ML import."""

from __future__ import annotations

import ast
import copy
import importlib
import os
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_fullsize_coordinator_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
m = importlib.import_module(PACKAGE + ".posttraining_checkpoint_import_runner_v3")
kernel = importlib.import_module(PACKAGE + ".posttraining_checkpoint_import_torch_v5")
packager = importlib.import_module(PACKAGE + ".posttraining_checkpoint_package_v2")
canonical, sha, json_object = m.canonical, m.sha, m.json_object


def _generated_builder(file, name, namespace):
    """Reuse only an existing synthetic fixture function, without its test imports."""
    tree = ast.parse((ROOT / "tests" / file).read_bytes())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "synthetic-builder", "exec"),
         namespace)
    return namespace[name]


fixture_namespace = {"canonical": canonical, "sha": sha, "json_object": json_object}
documents = _generated_builder("test_sft_checkpoint_import_plan_v1.py", "documents",
                               fixture_namespace)
seal_fixture = _generated_builder("test_sft_checkpoint_import_runner_v2.py", "seal_fixture",
                                  fixture_namespace)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lumi-v3-generated-coordinator-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.checkpoint, self.tokens = self.root / "input", self.root / "tokens"
        self.attempt, self.sources = self.root / "attempt", self.root / "source"
        for path in (self.root, self.checkpoint, self.tokens, self.sources):
            path.mkdir(mode=0o700, exist_ok=True)
            path.chmod(0o700)
        (self.checkpoint / "model").mkdir(mode=0o700)
        (self.checkpoint / "model/.metadata").write_bytes(b"generated metadata")
        (self.checkpoint / "model/__0_0.distcp").write_bytes(b"generated weights")
        self.metadata = seal_fixture(self.checkpoint, self.tokens)
        self.plan = m.frozen.build_import_plan(*self.metadata,
                                             expected_base_sha256=sha(self.metadata[0]),
                                             dtype="bfloat16")
        self.source_rows = []
        for name in ("configuration_lumi.py", "modeling_lumi.py"):
            raw = b"# generated source\n"
            (self.sources / name).write_bytes(raw)
            (self.sources / name).chmod(0o600)
            self.source_rows.append({"path": name, "bytes": len(raw), "sha256": sha(raw)})
        self.pins = {row["path"]: row["sha256"] for row in self.source_rows}
        self.source_bytes = sum(row["bytes"] for row in self.source_rows)
        b = self.plan["resource_bounds"]
        self.envelope = {"runtime_sources": self.source_rows}
        self.limits = {"maximum_seconds": 300, "maximum_coordinator_read_bytes": 64 * 1024**2,
                       "maximum_private_write_bytes": b["maximum_private_output_write_bytes"],
                       "maximum_package_read_bytes": 16 * 1024**2,
                       "maximum_package_write_bytes": b["tokenizer_inventory_bytes"] + 262144,
                       "maximum_control_bytes": 131072}
        self.contract = {"contract_sha256": "c" * 64,
                         "checkpoint_read_contract": {"maximum_read_bytes": 1024**2},
                         "hf_io_contract": {"maximum_read_bytes": 8 * 1024**2,
                             "maximum_write_bytes": b["maximum_backend_output_bytes"]}}
        self.args = {"metadata": self.metadata, "expected_base_sha256": sha(self.metadata[0]),
                     "plan_raw": canonical(self.plan),
                     "expected_plan_file_sha256": sha(canonical(self.plan)),
                     "envelope_raw": b"{}", "expected_envelope_file_sha256": sha(b"{}"),
                     "checkpoint_root": self.checkpoint, "tokenizer_root": self.tokens,
                     "attempt": self.attempt, "kernel_io_contract": self.contract,
                     "expected_kernel_io_contract_sha256": "c" * 64}

    def converted(self, *args, **kwargs):
        kwargs["check_budget"]()
        (self.attempt / "checkpoint").mkdir(mode=0o700)
        b = self.plan["resource_bounds"]
        return {"runtime_code_sha256": self.pins,
                "dcp_read_verification": {"dcp_read_bytes_reserved":
                    b["checkpoint_inventory_bytes"] + 100,
                    "dcp_input_audit_bytes": b["checkpoint_inventory_bytes"],
                    "model_payload_read_bytes_reserved": 100,
                    "hash_verified_files": b["checkpoint_inventory_files"] - 1,
                    "dcp_reads_metered": True},
                "hf_content_verification": {"hf_content_read_bytes_reserved": 100,
                                             "hf_content_write_bytes_reserved": 100}}

    def packaged(self, **kwargs):
        kwargs["check_budget"]()
        return {"package_read_bytes_reserved": 100, "package_write_bytes_reserved": 100}

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.object(m, "_runtime"))
        stack.enter_context(patch.object(m, "__file__", str(self.sources / "runner.py")))
        stack.enter_context(patch.object(m, "_resource_inputs", return_value=(
            self.plan, self.envelope, self.limits, self.contract)))
        stack.enter_context(patch.object(kernel, "convert_and_verify", side_effect=self.converted))
        stack.enter_context(patch.object(packager, "finalize_package", side_effect=self.packaged))
        return stack

    def run_core(self):
        return m._coordinate_admitted_import(**self.args)

    def bound_resource_inputs(self):
        """Build a complete envelope over this generated, actually parsed plan."""
        b = self.plan["resource_bounds"]
        rows = {row["path"]: row for row in self.plan["checkpoint_inventory"]}
        dcp_payload = max(b["source_fp32_tensor_bytes_including_aliases"], sum(
            row["bytes"] for name, row in rows.items() if name.endswith(".distcp")))
        dcp_payload += 2 * rows["model/.metadata"]["bytes"]
        control, tokens = 131072, b["tokenizer_inventory_bytes"]
        backend = b["maximum_backend_output_bytes"]
        e = {"schema_version": "lumi.sft-checkpoint-resource-envelope.v0.1",
             "plan_sha256": self.plan["plan_sha256"],
             "dcp_payload_read_cap_bytes": dcp_payload,
             "dcp_read_cap_bytes": b["checkpoint_inventory_bytes"] + dcp_payload,
             "hf_read_cap_bytes": backend + 2 * self.source_bytes + len(self.metadata[3]),
             "package_read_cap_bytes": 3 * backend + 3 * tokens + 5 * control,
             "hf_write_cap_bytes": backend, "package_write_cap_bytes": tokens + 2 * control,
             "config_bytes": len(self.metadata[3]), "source_config_sha256": sha(self.metadata[3]),
             "control_bytes": control, "package_receipt_cap_bytes": control,
             "package_manifest_cap_bytes": control, "runtime_sources": self.source_rows,
             "private_write_cap_bytes": b["maximum_private_output_write_bytes"],
             "worker_memory_max_bytes": b["minimum_host_ram_planning_bytes"],
             "output_capacity_bytes": b["maximum_private_output_write_bytes"],
             "extra_host_disk_reserve_bytes": 1024**3,
             "host_disk_capacity_bytes": b["minimum_staged_disk_planning_bytes"] + 1024**3,
             "maximum_seconds": 300}
        e["coordinator_read_cap_bytes"] = (e["dcp_read_cap_bytes"] + e["hf_read_cap_bytes"]
            + b["checkpoint_inventory_bytes"] + b["tokenizer_audit_read_bytes"]
            + e["package_read_cap_bytes"] + self.source_bytes)
        io = importlib.import_module(PACKAGE + ".posttraining_checkpoint_io_contract_v1")
        contract = io.build_kernel_io_contract(self.plan, checkpoint_read_contract={
            "expected_manifest_sha256": self.plan["metadata"]["checkpoint_manifest_sha256"],
            "maximum_file_bytes": max(row["bytes"] for row in rows.values()),
            "maximum_read_bytes": e["dcp_read_cap_bytes"], "maximum_read_call_bytes": 1048576},
            hf_io_contract={"maximum_read_bytes": e["hf_read_cap_bytes"],
                            "maximum_write_bytes": backend,
                            "maximum_file_bytes": min(backend, 2**31),
                            "maximum_control_bytes": 65536, "target_shard_bytes": 268435456})
        return e, contract

    def test_integrated_resource_pin_and_stage_flow_without_ml(self):
        e, contract = self.bound_resource_inputs()
        self.args.update(envelope_raw=canonical(e), expected_envelope_file_sha256=sha(canonical(e)),
                         kernel_io_contract=contract,
                         expected_kernel_io_contract_sha256=contract["contract_sha256"])
        with patch.object(m, "_runtime"), \
                patch.object(m, "__file__", str(self.sources / "runner.py")), \
                patch.object(kernel, "convert_and_verify", side_effect=self.converted), \
                patch.object(packager, "finalize_package", side_effect=self.packaged):
            result = self.run_core()
        self.assertEqual(result["status"], "PASS_import_coordinator_requires_parent_verification")
        self.assertEqual(result["coordinator_read_bytes_reserved"], e["coordinator_read_cap_bytes"])

    def test_integrated_bad_envelope_pin_precedes_runtime_source_or_input_reads(self):
        e, contract = self.bound_resource_inputs()
        self.args.update(envelope_raw=canonical(e), expected_envelope_file_sha256="0" * 64,
                         kernel_io_contract=contract,
                         expected_kernel_io_contract_sha256=contract["contract_sha256"])
        with patch.object(m, "_runtime"), patch.object(m.os, "open") as opened:
            with self.assertRaisesRegex(m.ImportRunnerError, "resource_envelope_rejected"):
                self.run_core()
            opened.assert_not_called()

    def test_kernel_and_envelope_caps_must_match_before_source_reads(self):
        e, contract = self.bound_resource_inputs()
        contract["hf_io_contract"]["maximum_read_bytes"] += 1
        contract["contract_sha256"] = sha(canonical({k: v for k, v in contract.items()
                                                   if k != "contract_sha256"}))
        self.args.update(envelope_raw=canonical(e), expected_envelope_file_sha256=sha(canonical(e)),
                         kernel_io_contract=contract,
                         expected_kernel_io_contract_sha256=contract["contract_sha256"])
        with patch.object(m, "_runtime"), patch.object(m.os, "open") as opened:
            with self.assertRaisesRegex(m.ImportRunnerError, "resource_envelope_rejected"):
                self.run_core()
            opened.assert_not_called()

    def test_runtime_source_modes_match_frozen_hf_control_contract(self):
        for mode in (0o400, 0o600, 0o644):
            for row in self.source_rows:
                (self.sources / row["path"]).chmod(mode)
            ledger = m._Ledger(self.limits, started=m.time.monotonic())
            with patch.object(m, "__file__", str(self.sources / "runner.py")):
                pins, count = m._runtime_source_preflight(self.source_rows, ledger)
            self.assertEqual((pins, count), (self.pins, self.source_bytes))
        (self.sources / self.source_rows[0]["path"]).chmod(0o666)
        with patch.object(m, "__file__", str(self.sources / "runner.py")), \
                self.assertRaisesRegex(m.ImportRunnerError, "source_binding_rejected"):
            m._runtime_source_preflight(self.source_rows, ledger)

    def test_public_entry_refuses_before_io_or_control_parsing(self):
        with patch.object(m.os, "open") as opened, patch.object(m, "_resource_inputs") as parser:
            with self.assertRaisesRegex(m.ImportRunnerError, "^host_guard_not_bound$"):
                m.run_import(approval=True, host_bound=True, **self.args)
            opened.assert_not_called()
            parser.assert_not_called()

    def test_private_core_rejects_unisolated_runtime_before_parsing(self):
        with patch.object(m.sys, "platform", "darwin"), \
                patch.object(m, "_resource_inputs") as parser, patch.object(m.os, "open") as opened:
            with self.assertRaisesRegex(m.ImportRunnerError, "^runtime_rejected$"):
                self.run_core()
        parser.assert_not_called()
        opened.assert_not_called()

    def test_generated_flow_audits_and_accounts_preflight_separately(self):
        with self.patches(), patch.object(m, "_audit", wraps=m._audit) as audit:
            result = self.run_core()
        self.assertEqual([call.args[0] for call in audit.call_args_list],
                         [self.tokens, self.checkpoint, self.tokens])
        account = result["content_accounting"]
        self.assertEqual(account["runtime_source_preflight_read_bytes"], self.source_bytes)
        expected = (self.plan["resource_bounds"]["checkpoint_audit_read_bytes"]
                    + self.plan["resource_bounds"]["tokenizer_audit_read_bytes"]
                    + 300 + self.source_bytes)
        self.assertEqual(account["successful_content_read_bytes_requested"], expected)
        self.assertTrue((self.attempt / m.COMPLETE).is_file())
        self.assertFalse((self.attempt / m.FAILED).exists())
        self.assertFalse(result["real_4b_import"])
        self.assertTrue(result["success_requires_returned_outcome_and_parent_supervisor"])
        marker = json_object((self.attempt / m.STARTED).read_bytes())
        self.assertEqual(marker["resource_envelope_sha256"],
                         self.args["expected_envelope_file_sha256"])

    def test_prior_attempt_is_preserved_and_never_reaches_kernel(self):
        self.attempt.mkdir(mode=0o700)
        sentinel = self.attempt / "sentinel"
        sentinel.write_bytes(b"preserve generated prior attempt")
        with self.patches(), patch.object(kernel, "convert_and_verify") as convert:
            with self.assertRaisesRegex(m.ImportRunnerError, "attempt_exists_no_retry"):
                self.run_core()
            convert.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"preserve generated prior attempt")

    def test_wrong_source_pin_precedes_attempt_and_payload_audit(self):
        self.source_rows[0]["sha256"] = "0" * 64
        with self.patches(), patch.object(m, "_audit") as audit:
            with self.assertRaisesRegex(m.ImportRunnerError, "source_binding_rejected"):
                self.run_core()
            audit.assert_not_called()
        self.assertFalse(self.attempt.exists())

    def test_source_short_read_is_charged_once_not_retried(self):
        ledger = m._Ledger(self.limits, started=m.time.monotonic())
        with patch.object(m, "__file__", str(self.sources / "runner.py")), \
                patch.object(m.os, "read", return_value=b"short") as read:
            with self.assertRaisesRegex(m.ImportRunnerError, "source_binding_rejected"):
                m._runtime_source_preflight(self.source_rows, ledger)
        self.assertEqual(read.call_count, 1)
        self.assertEqual(ledger.reads, self.source_rows[0]["bytes"])

    def test_wrong_kernel_source_receipt_prevents_packaging(self):
        def wrong(*args, **kwargs):
            result = self.converted(*args, **kwargs)
            result["runtime_code_sha256"] = {}
            return result
        with self.patches(), \
                patch.object(kernel, "convert_and_verify", side_effect=wrong), \
                patch.object(packager, "finalize_package") as package_call:
            with self.assertRaisesRegex(m.ImportRunnerError, "source_binding_rejected"):
                self.run_core()
            package_call.assert_not_called()
        self.assertTrue((self.attempt / m.FAILED).is_file())

    def test_kernel_exception_never_retains_private_text_or_retries(self):
        with self.patches(), patch.object(
                kernel, "convert_and_verify",
                side_effect=ValueError("private generated detail")) as call:
            with self.assertRaisesRegex(m.ImportRunnerError, "^kernel_failed$"):
                self.run_core()
            call.assert_called_once()
            with self.assertRaisesRegex(m.ImportRunnerError, "attempt_exists_no_retry"):
                self.run_core()
            call.assert_called_once()
        self.assertNotIn(b"private generated", (self.attempt / m.FAILED).read_bytes())

    def test_source_mutation_after_conversion_prevents_package(self):
        def mutate(*args, **kwargs):
            result = self.converted(*args, **kwargs)
            (self.checkpoint / "trainer-rank-00000.pt").write_bytes(b"changed generated input")
            return result
        with self.patches(), patch.object(kernel, "convert_and_verify", side_effect=mutate), \
                patch.object(packager, "finalize_package") as package_call:
            with self.assertRaisesRegex(m.ImportRunnerError, "input_audit_failed"):
                self.run_core()
            package_call.assert_not_called()

    def test_late_failure_preserves_non_authoritative_completion(self):
        original = m._write_new
        def late(root, name, raw, ledger, **kwargs):
            original(root, name, raw, ledger, **kwargs)
            if name == m.COMPLETE:
                raise OSError("generated late close error")
        with self.patches(), patch.object(m, "_write_new", side_effect=late):
            with self.assertRaises(m.ImportRunnerError):
                self.run_core()
        saved = json_object((self.attempt / m.COMPLETE).read_bytes())
        self.assertFalse(saved["durable_completion_is_execution_authority"])
        self.assertTrue((self.attempt / m.FAILED).is_file())

    def test_ledger_full_size_limits_are_copied_and_exhaustion_is_sticky(self):
        limits = copy.deepcopy(self.limits)
        limits["maximum_coordinator_read_bytes"] = 200 * 1024**3
        limits["maximum_private_write_bytes"] = 9 * 1024**3
        ledger = m._Ledger(limits, started=m.time.monotonic())
        limits["maximum_coordinator_read_bytes"] = 1
        ledger.reserve(reads=100 * 1024**3)
        with self.assertRaisesRegex(m.frozen.ImportRunnerError, "read_budget_exhausted"):
            ledger.reserve(reads=101 * 1024**3)
        self.assertEqual(ledger.reads, 100 * 1024**3)
        with self.assertRaises(m.frozen.ImportRunnerError):
            ledger.tick()

    def test_partial_marker_preserved_without_second_kernel_attempt(self):
        original, calls = os.write, []
        def short(fd, raw):
            calls.append(True)
            return original(fd, raw[:1]) if len(calls) == 1 else original(fd, raw)
        with self.patches(), patch.object(m.os, "write", side_effect=short), \
                patch.object(kernel, "convert_and_verify") as call:
            with self.assertRaisesRegex(m.ImportRunnerError, "write_failed"):
                self.run_core()
            call.assert_not_called()
        self.assertEqual((self.attempt / m.STARTED).stat().st_size, 1)


if __name__ == "__main__":
    unittest.main()
