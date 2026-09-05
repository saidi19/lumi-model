"""Synthetic byte-only bootstrap checks; worker source is never executed."""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).parents[1] / "scripts/sft_guarded_bootstrap_v1.py"
spec = importlib.util.spec_from_file_location("synthetic_bootstrap", SCRIPT)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def fixture():
    source = b'raise RuntimeError("SYNTHETIC-SOURCE-MUST-NOT-RUN")\n'
    source_sha = hashlib.sha256(source).hexdigest()
    runtime = {"python_executable": "/synthetic/venv/python", "runtime_lock_sha256": "b" * 64,
               "world_size": 2, "rank": 0, "argv": ["--from-step", "0", "--stop-step", "2"]}
    supervisor = {"schema_version": "lumi.sft-supervisor-attachment.v0.1",
                  "validation_kind": "structural_only", "source_sha256": source_sha,
                  "runtime_binding_sha256": bootstrap.digest(runtime), "resource_caps_sha256": "c" * 64}
    ticket = {"schema_version": bootstrap.SCHEMA, "scope": bootstrap.SCOPE,
              "operation": "sft_gpu_smoke_and_fresh_resume_only", "production_eligible": False,
              "runtime": runtime,
              "bindings": {**dict.fromkeys(bootstrap.HASH_FIELDS, "a" * 64),
                           "source_sha256": source_sha, "runtime_lock_sha256": "b" * 64,
                           "resource_caps_sha256": "c" * 64,
                           "supervisor_receipt_sha256": bootstrap.digest(supervisor)}}
    subject = bootstrap.digest(ticket)
    ticket["approval"] = {"scope": bootstrap.SCOPE, "kind": "bounded_sft_execution", "approved": True,
                          "subject_sha256": subject, "issued_at_epoch_seconds": 100,
                          "expires_at_epoch_seconds": 200}
    raw = bootstrap.canonical(ticket)
    kwargs = {"expected_ticket_sha256": hashlib.sha256(raw).hexdigest(),
              "expected_subject_sha256": subject, "expected_source_sha256": source_sha,
              "now_epoch_seconds": 150, "runtime_binding": copy.deepcopy(runtime),
              "supervisor_evidence": supervisor, "no_site": 1, "no_bytecode": 1}
    return ticket, raw, source, kwargs


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.ticket, self.raw, self.source, self.kwargs = fixture()

    def check(self):
        return bootstrap.validate_bootstrap(self.raw, self.source, **self.kwargs)

    def repin(self):
        self.raw = bootstrap.canonical(self.ticket)
        self.kwargs["expected_ticket_sha256"] = hashlib.sha256(self.raw).hexdigest()

    def code(self, expected, callback=None):
        with self.assertRaises(bootstrap.BootstrapError) as caught:
            (self.check if callback is None else callback)()
        self.assertEqual(str(caught.exception), expected)
        self.assertIn(expected, bootstrap.CODES)

    def test_valid_pins_do_not_enable_execution_or_import_source(self):
        report = self.check()
        self.assertTrue(report["metadata_validation_passed"])
        self.assertFalse(report["execution_enabled"])
        self.assertFalse(report["worker_imported"])
        self.assertFalse(report["supervisor_attachment_authenticated"])
        self.assertNotIn("SYNTHETIC-SOURCE", json.dumps(report))

    def test_site_and_bytecode_flags_strict(self):
        for field in ("no_site", "no_bytecode"):
            for bad in (0, True, "1"):
                self.kwargs[field] = bad
                self.code("python_flags_required")
            self.kwargs[field] = 1

    def test_ticket_byte_pin_before_json_decode(self):
        self.raw = b"SYNTHETIC-PRIVATE-INVALID-JSON"
        self.code("ticket_hash")

    def test_source_pin_before_any_import(self):
        self.source += b"# changed"
        self.code("source_hash")

    def test_all_out_of_band_hashes_strict(self):
        for field in ("expected_ticket_sha256", "expected_subject_sha256", "expected_source_sha256"):
            original = self.kwargs[field]
            for bad in ("A" * 64, "SYNTHETIC-SECRET", None, True):
                self.kwargs[field] = bad
                self.code("invalid_hash")
            self.kwargs[field] = original

    def test_subject_pin_cannot_be_replaced_by_ticket_self_assertion(self):
        self.ticket["approval"]["subject_sha256"] = "f" * 64
        self.repin()
        self.code("approval_subject")

    def test_pretraining_approval_rejected(self):
        self.ticket["approval"]["kind"] = "pretraining"
        self.repin()
        self.code("approval_scope")

    def test_approved_value_not_truthy_integer(self):
        self.ticket["approval"]["approved"] = 1
        self.repin()
        self.code("approval_scope")

    def test_ticket_scope_and_operation_fixed(self):
        for field, bad in (("scope", "production"), ("operation", "full_sft"),
                           ("production_eligible", True), ("production_eligible", 0)):
            self.ticket, self.raw, self.source, self.kwargs = fixture()
            self.ticket[field] = bad
            self.repin()
            self.code("ticket_scope")

    def test_expiry_and_time_types(self):
        for bad in (99, 200, True, 150.0, -1, 2**63):
            self.kwargs["now_epoch_seconds"] = bad
            self.code("approval_time")

    def test_issued_at_exact_boundary_allowed(self):
        self.kwargs["now_epoch_seconds"] = 100
        self.assertTrue(self.check()["metadata_validation_passed"])

    def test_binding_hash_changed_even_with_repin_changes_subject(self):
        self.ticket["bindings"]["cuda_acceptance_sha256"] = "e" * 64
        self.repin()
        self.code("approval_subject")

    def test_extra_ticket_data_rejected(self):
        self.ticket["raw_text"] = "SYNTHETIC-SECRET"
        self.repin()
        self.code("ticket_schema")

    def test_runtime_rank_world_args_must_match_observed_binding(self):
        for field, bad in (("rank", 1), ("world_size", 3), ("argv", ["--stop-step", "999"]),
                           ("python_executable", "/synthetic/other/python")):
            self.ticket, self.raw, self.source, self.kwargs = fixture()
            self.kwargs["runtime_binding"][field] = bad
            self.code("runtime_binding")

    def test_runtime_shape_rejects_rank_bool_and_controls(self):
        for field, bad in (("rank", True), ("world_size", True), ("argv", ["bad\n"]),
                           ("python_executable", "/synthetic/../other/python")):
            self.ticket, self.raw, self.source, self.kwargs = fixture()
            self.kwargs["runtime_binding"][field] = bad
            self.code("runtime_schema")

    def test_supervisor_flag_cannot_prove_os_isolation(self):
        self.kwargs["supervisor_evidence"]["validation_kind"] = "os_enforced"
        self.code("supervisor_schema")

    def test_supervisor_binding_cannot_be_swapped(self):
        self.kwargs["supervisor_evidence"]["resource_caps_sha256"] = "e" * 64
        self.code("supervisor_binding")

    def test_duplicate_keys_rejected(self):
        self.raw = b'{"scope":"a","scope":"b"}'
        self.kwargs["expected_ticket_sha256"] = hashlib.sha256(self.raw).hexdigest()
        self.code("json_duplicate_key")

    def test_nonfinite_json_rejected(self):
        self.raw = b'{"value":NaN}'
        self.kwargs["expected_ticket_sha256"] = hashlib.sha256(self.raw).hexdigest()
        self.code("json_invalid")

    def test_input_size_limits(self):
        self.source = b"x" * (bootstrap.MAX_SOURCE_BYTES + 1)
        self.code("input_size")

    def test_control_file_real_read_and_size_cap(self):
        with tempfile.TemporaryDirectory(prefix="lumi-bootstrap-synthetic-") as root:
            path = Path(root) / "source.py"
            path.write_bytes(self.source)
            self.assertEqual(bootstrap.read_control_file(str(path), 4096), self.source)
            self.code("control_file_invalid", lambda: bootstrap.read_control_file(str(path), 1))

    def test_control_file_symlink_and_directory_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lumi-bootstrap-synthetic-") as root:
            path = Path(root) / "source.py"
            path.write_bytes(self.source)
            link = Path(root) / "link.py"
            link.symlink_to(path)
            for target in (link, Path(root)):
                self.code("control_file_invalid", lambda: bootstrap.read_control_file(str(target), 4096))

    def test_file_error_context_does_not_retain_source_path(self):
        with self.assertRaises(bootstrap.BootstrapError) as caught:
            bootstrap.read_control_file("/synthetic/SYNTHETIC-SECRET-NOT-PRESENT", 4096)
        self.assertIsNone(caught.exception.__context__)

    def test_control_file_change_detected(self):
        with tempfile.TemporaryDirectory(prefix="lumi-bootstrap-synthetic-") as root:
            path = Path(root) / "source.py"
            path.write_bytes(self.source)
            real_read = os.read
            def changing_read(fd, size):
                result = real_read(fd, size)
                if result:
                    path.write_bytes(self.source + b"# changed")
                return result
            with mock.patch.object(bootstrap.os, "read", changing_read):
                self.code("control_file_changed", lambda: bootstrap.read_control_file(str(path), 4096))

    def test_cli_execute_rejected_before_control_reads(self):
        output = io.StringIO()
        with mock.patch.object(bootstrap, "read_control_file", side_effect=AssertionError("must not read")):
            with contextlib.redirect_stdout(output):
                self.assertEqual(bootstrap.main(["--execute"]), 2)
        self.assertEqual(json.loads(output.getvalue())["failure_code"], "execute_disabled")

    def test_cli_bad_flag_outputs_no_raw_argument(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(bootstrap.main(["--SYNTHETIC-SECRET"]), 2)
        self.assertNotIn("SYNTHETIC-SECRET", output.getvalue())

    def test_cli_invalid_pin_rejected_before_control_reads(self):
        argv = []
        for field in ("ticket", "source", "supervisor-evidence", "runtime-binding"):
            argv.extend(["--" + field, "/synthetic/unread"])
        for field in ("expected-ticket-sha256", "expected-subject-sha256", "expected-source-sha256"):
            argv.extend(["--" + field, "invalid"])
        output = io.StringIO()
        with mock.patch.object(bootstrap, "read_control_file", side_effect=AssertionError("must not read")):
            with contextlib.redirect_stdout(output):
                self.assertEqual(bootstrap.main(argv), 2)
        self.assertEqual(json.loads(output.getvalue())["failure_code"], "invalid_hash")

    def test_cli_ticket_pin_rejected_before_source_read(self):
        argv = []
        for field in ("ticket", "source", "supervisor-evidence", "runtime-binding"):
            argv.extend(["--" + field, "/synthetic/" + field])
        for field in ("expected_ticket_sha256", "expected_subject_sha256", "expected_source_sha256"):
            argv.extend(["--" + field.replace("_", "-"), self.kwargs[field]])
        output = io.StringIO()
        with mock.patch.object(bootstrap, "read_control_file", return_value=b"changed") as reader:
            with contextlib.redirect_stdout(output):
                self.assertEqual(bootstrap.main(argv), 2)
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(json.loads(output.getvalue())["failure_code"], "ticket_hash")

    def test_cli_real_synthetic_control_files_pass_but_never_execute(self):
        with tempfile.TemporaryDirectory(prefix="lumi-bootstrap-synthetic-") as root:
            files = {"ticket": self.raw, "source": self.source,
                     "runtime-binding": bootstrap.canonical(self.kwargs["runtime_binding"]),
                     "supervisor-evidence": bootstrap.canonical(self.kwargs["supervisor_evidence"])}
            argv = []
            for field, data in files.items():
                path = Path(root) / field
                path.write_bytes(data)
                argv.extend(["--" + field, str(path)])
            for field in ("expected_ticket_sha256", "expected_subject_sha256", "expected_source_sha256"):
                argv.extend(["--" + field.replace("_", "-"), self.kwargs[field]])
            output = io.StringIO()
            with mock.patch.object(bootstrap.time, "time", return_value=150):
                with mock.patch.object(bootstrap.sys, "executable", "/synthetic/venv/python"):
                    with contextlib.redirect_stdout(output):
                        self.assertEqual(bootstrap.main(argv), 0)
            self.assertFalse(json.loads(output.getvalue())["execution_enabled"])

    def test_ast_has_no_ml_or_package_import_or_dispatch(self):
        tree = ast.parse(SCRIPT.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module]
            else:
                names = []
            for name in names:
                self.assertNotIn(name.split(".")[0], {"lumi", "torch", "transformers", "subprocess", "socket"})
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"eval", "exec", "compile", "__import__"})


if __name__ == "__main__":
    unittest.main()
