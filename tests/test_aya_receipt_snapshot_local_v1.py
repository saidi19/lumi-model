"""Synthetic CLI regressions; never open a private receipt or install audit hooks."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aya_receipt_snapshot_cli_tests", ROOT / "scripts/check_aya_receipt_snapshot_local_v1.py"
)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class AyaReceiptLocalTests(unittest.TestCase):
    def setUp(self):
        # These factories only authenticate and compile source-controlled helpers.
        self.lifecycle = cli.lifecycle_helpers()
        self.io = self.lifecycle.io_primitives()
        self.configure_main()

    def configure_main(self):
        self.spec = cli.read_spec()
        self.inputs = Mock(opens=0, bytes_read=0)

        def synthetic_read(_path, binding):
            self.inputs.opens += 1
            self.inputs.bytes_read += binding["bytes"]
            return b"synthetic payload; no bound file was opened"

        self.inputs.read.side_effect = synthetic_read
        self.boundary = Mock(blocked=0, roots=())
        self.summary = {
            "virtual_opens": 1,
            "logical_read_bytes_debited": 1667,
            "bound_blob_count": 1,
            "filesystem_access_performed": False,
        }
        self.snapshot = Mock(summary=Mock(return_value=self.summary))
        self.module = self.lifecycle.synthetic_module()
        self.pytest = Mock()
        self.core = Mock()
        self.core.build_test_module.return_value = (self.module, self.snapshot)
        self.phases = {
            "collected": 1,
            "phases": {"setup": "passed", "call": "passed", "teardown": "passed"},
        }
        self.read_spec = Mock(return_value=self.spec)
        self.warm_runtime = Mock(return_value=(self.pytest, self.core))
        self.run_projected_test = Mock(return_value=self.phases)
        self.alarm = Mock()

    def run_main(
        self,
        arguments,
        *,
        no_site=1,
        optimize=0,
        dont_write_bytecode=True,
        times=(0, 1, 2),
    ):
        capture = io.StringIO()
        with (
            patch.object(cli, "lifecycle_helpers", return_value=self.lifecycle),
            patch.object(self.lifecycle, "io_primitives", return_value=self.io),
            patch.object(cli, "read_spec", self.read_spec),
            patch.object(self.io, "Boundary", return_value=self.boundary) as boundary_factory,
            patch.object(self.io, "BoundInputs", return_value=self.inputs) as inputs_factory,
            patch.object(self.lifecycle, "network_denial") as network_denial,
            patch.object(cli, "warm_runtime", self.warm_runtime),
            patch.object(self.lifecycle, "run_projected_test", self.run_projected_test),
            patch.object(
                cli.sys, "flags", types.SimpleNamespace(no_site=no_site, optimize=optimize)
            ),
            patch.object(cli.sys, "dont_write_bytecode", dont_write_bytecode),
            patch.object(cli.sys, "addaudithook") as addaudithook,
            patch.object(cli.signal, "signal"),
            patch.object(cli.signal, "alarm", self.alarm),
            patch.object(cli.time, "monotonic", side_effect=times),
            contextlib.redirect_stdout(capture),
        ):
            result = cli.main(arguments)
        self.boundary_factory = boundary_factory
        self.inputs_factory = inputs_factory
        self.network_denial = network_denial
        self.addaudithook = addaudithook
        self.raw_output = capture.getvalue()
        self.assertEqual(self.alarm.call_args_list[-1], call(0))
        return result, json.loads(self.raw_output)

    def assert_stopped(self, result, output, failure_code):
        self.assertEqual(result, 1)
        self.assertEqual(output["status"], "STOP")
        self.assertEqual(output["failure_code"], failure_code)
        self.assertFalse(output["original_test_body_executed"])
        self.assertEqual(output["snapshot_backed_pytest_nodes_passed"], 0)
        self.assertNotIn("snapshot_accounting", output)
        self.assertNotIn("SYNTHETIC_SECRET", self.raw_output)

    @contextlib.contextmanager
    def isolated_core_namespace(self):
        with patch.dict(sys.modules):
            for name in tuple(sys.modules):
                if name == cli.CORE_PACKAGE or name.startswith(cli.CORE_PACKAGE + "."):
                    del sys.modules[name]
            yield

    def test_exact_new_control_bindings_and_limits(self):
        self.assertEqual((cli.MAX_FILES, cli.MAX_BYTES, cli.MAX_SECONDS), (4, 45_471, 30))
        self.assertEqual(cli.RECEIPT_BYTES, 1667)
        self.assertEqual(self.spec["nodeid"], cli.NODEID)
        self.assertEqual(
            cli.NODEID,
            "tests/test_aya_bengali_owner_review.py::"
            "test_actual_owner_receipt_is_complete_and_nonpromoting",
        )
        self.assertEqual(
            set(self.spec["bindings"]),
            {"receipt", "review_implementation", "hash_helper_source", "test_source"},
        )
        self.assertEqual(sum(b["bytes"] for b in self.spec["bindings"].values()), 45_471)
        self.assertEqual(self.spec["bindings"]["receipt"]["bytes"], 1667)
        self.assertEqual(
            self.spec["bindings"]["receipt"]["sha256"],
            "66c28d06816e6fc69cbde022fa0ebb3820f44ad0e1daf611f1857c6ca9e243be",
        )
        self.assertEqual(
            self.spec["limits"],
            {
                "maximum_bound_file_opens": 4,
                "maximum_bound_bytes": 45_471,
                "maximum_runtime_seconds": 30,
                "maximum_private_receipt_opens": 1,
                "maximum_private_receipt_bytes": 1667,
                "maximum_snapshot_opens": 1,
                "maximum_snapshot_read_bytes": 1667,
                "source_rows": 0,
                "candidate_rows": 0,
                "artifact_write_bytes": 0,
                "network_bytes": 0,
                "inference_calls": 0,
            },
        )

    def test_control_read_cap_and_hash_are_enforced(self):
        for payload in (b"too long", b"bad"):
            with self.subTest(payload=payload):
                stream = Mock()
                stream.__enter__ = Mock(return_value=stream)
                stream.__exit__ = Mock(return_value=False)
                stream.read.return_value = payload
                path = Mock(open=Mock(return_value=stream))
                with self.assertRaisesRegex(ValueError, "^control_binding_rejected$"):
                    cli.control_bytes(path, "0" * 64, 3)
                path.open.assert_called_once_with("rb")
                stream.read.assert_called_once_with(4)

    def test_receipt_identity_uses_fresh_namespace_without_mutating_old_helper(self):
        source = cli.control_bytes(cli.LIFECYCLE_PATH, cli.LIFECYCLE_SHA256, 65_536)
        original = types.ModuleType("_synthetic_old_aya_lifecycle")
        original.__file__ = str(cli.LIFECYCLE_PATH)
        exec(compile(source, str(cli.LIFECYCLE_PATH), "exec"), original.__dict__)
        old_identity = (
            original.MODULE_NAME, original.TEST_NAME, original.TEST_PATH, original.NODEID
        )
        existing = types.ModuleType("_aya_receipt_pytest_lifecycle")
        with patch.dict(sys.modules, {existing.__name__: existing}):
            fresh = cli.lifecycle_helpers()
            self.assertIs(sys.modules[existing.__name__], existing)
        self.assertIsNot(fresh, self.lifecycle)
        self.assertEqual(
            (fresh.MODULE_NAME, fresh.TEST_NAME, fresh.TEST_PATH, fresh.NODEID),
            (cli.MODULE_NAME, cli.TEST_NAME, cli.TEST_PATH, cli.NODEID),
        )
        self.assertEqual(
            (original.MODULE_NAME, original.TEST_NAME, original.TEST_PATH, original.NODEID),
            old_identity,
        )
        self.assertEqual(original.MODULE_NAME, "test_aya_bengali_repair_preflight")
        self.assertEqual((original.MAX_FILES, original.MAX_BYTES), (17, 10_433_360))
        self.assertEqual((fresh.MAX_FILES, fresh.MAX_BYTES), (17, 10_433_360))
        self.assertEqual(
            cli.control_bytes(cli.LIFECYCLE_PATH, cli.LIFECYCLE_SHA256, 65_536), source
        )

    def test_quiet_output_does_not_retain_exception_or_pytest_text(self):
        sink = self.lifecycle.QuietText()
        self.assertFalse(sink.isatty())
        self.assertEqual(sink.encoding, "utf-8")
        self.assertEqual(sink.write("SYNTHETIC_SECRET"), 16)
        self.assertIsNone(sink.flush())
        self.assertEqual(vars(sink), {})

    def test_only_exact_single_collection_is_accepted(self):
        for items in (
            [],
            [types.SimpleNamespace(nodeid="wrong")],
            [types.SimpleNamespace(nodeid=cli.NODEID)] * 2,
        ):
            with self.subTest(items=items), self.assertRaises(self.io.Stop):
                self.lifecycle.OneNode(self.io).pytest_collection_modifyitems(items)
        plugin = self.lifecycle.OneNode(self.io)
        plugin.pytest_collection_modifyitems([types.SimpleNamespace(nodeid=cli.NODEID)])
        self.assertFalse(plugin.passed())

    def test_all_three_phases_must_pass_without_skip_or_failure(self):
        for changed_phase in ("setup", "call", "teardown"):
            for outcome in ("passed", "failed", "skipped", "missing"):
                with self.subTest(phase=changed_phase, outcome=outcome):
                    plugin = self.lifecycle.OneNode(self.io)
                    plugin.pytest_collection_modifyitems(
                        [types.SimpleNamespace(nodeid=cli.NODEID)]
                    )
                    for phase in ("setup", "call", "teardown"):
                        if phase == changed_phase and outcome == "missing":
                            continue
                        plugin.pytest_runtest_logreport(
                            types.SimpleNamespace(
                                nodeid=cli.NODEID,
                                when=phase,
                                outcome=outcome if phase == changed_phase else "passed",
                                longrepr="SYNTHETIC_SECRET",
                            )
                        )
                    self.assertEqual(plugin.passed(), outcome == "passed")
                    self.assertNotIn("SYNTHETIC_SECRET", str(vars(plugin)))

    def test_foreign_unknown_and_duplicate_phase_reports_are_rejected(self):
        for nodeid, phase in (("wrong", "call"), (cli.NODEID, "unknown"), (cli.NODEID, "call")):
            with self.subTest(nodeid=nodeid, phase=phase):
                plugin = self.lifecycle.OneNode(self.io)
                plugin.pytest_runtest_logreport(
                    types.SimpleNamespace(nodeid=cli.NODEID, when="call", outcome="passed")
                )
                with self.assertRaises(self.io.Stop):
                    plugin.pytest_runtest_logreport(
                        types.SimpleNamespace(nodeid=nodeid, when=phase, outcome="passed")
                    )

    def test_preexisting_test_module_is_preserved_before_pytest_starts(self):
        existing = types.ModuleType(cli.MODULE_NAME)
        with patch.dict(sys.modules, {cli.MODULE_NAME: existing}):
            with self.assertRaises(self.io.Stop):
                self.lifecycle.run_projected_test(self.io, self.pytest, self.module)
            self.assertIs(sys.modules[cli.MODULE_NAME], existing)
        self.pytest.main.assert_not_called()

    def test_projection_cleanup_restores_outer_namespace_on_pytest_failure(self):
        for failure in (1, RuntimeError("SYNTHETIC_SECRET")):
            with self.subTest(failure=type(failure).__name__):
                existing = types.ModuleType(cli.MODULE_NAME)
                with patch.dict(sys.modules, {cli.MODULE_NAME: existing}):
                    with patch.dict(sys.modules):
                        del sys.modules[cli.MODULE_NAME]
                        fake_pytest = Mock()
                        if isinstance(failure, Exception):
                            fake_pytest.main.side_effect = failure
                        else:
                            fake_pytest.main.return_value = failure
                        with self.assertRaises((self.io.Stop, RuntimeError)):
                            self.lifecycle.run_projected_test(self.io, fake_pytest, self.module)
                        self.assertNotIn(cli.MODULE_NAME, sys.modules)
                    self.assertIs(sys.modules[cli.MODULE_NAME], existing)

    def test_projected_runner_requires_observed_lifecycle_and_exact_test_argument(self):
        def synthetic_pytest(arguments, *, plugins):
            self.assertEqual(arguments[-1], str(cli.TEST_PATH) + "::" + cli.TEST_NAME)
            for flag in ("--noconftest", "--assert=plain", "no:cacheprovider", "no:faulthandler"):
                self.assertIn(flag, arguments)
            plugin = plugins[0]
            plugin.pytest_collection_modifyitems([types.SimpleNamespace(nodeid=cli.NODEID)])
            getattr(self.module, cli.TEST_NAME)()
            for phase in ("setup", "call", "teardown"):
                plugin.pytest_runtest_logreport(
                    types.SimpleNamespace(nodeid=cli.NODEID, when=phase, outcome="passed")
                )
            return 0

        with patch.dict(sys.modules):
            sys.modules.pop(cli.MODULE_NAME, None)
            self.pytest.main.side_effect = synthetic_pytest
            self.assertEqual(
                self.lifecycle.run_projected_test(self.io, self.pytest, self.module), self.phases
            )
            self.assertNotIn(cli.MODULE_NAME, sys.modules)
            self.pytest.main.side_effect = None
            self.pytest.main.return_value = 0
            with self.assertRaises(self.io.Stop):
                self.lifecycle.run_projected_test(self.io, self.pytest, self.module)
            self.assertNotIn(cli.MODULE_NAME, sys.modules)

    def test_warm_runtime_isolates_package_and_disables_ambient_pytest_plugins(self):
        with (
            patch.dict(sys.modules, {"pytest": self.pytest}),
            patch.object(cli.sys, "path", list(sys.path)),
            patch.dict(cli.os.environ, {"PYTEST_ADDOPTS": "unsafe", "PYTEST_PLUGINS": "unsafe"}),
            patch.object(cli, "authenticated_core", return_value=self.core) as load_core,
        ):
            self.assertEqual(cli.warm_runtime(self.io), (self.pytest, self.core))
            self.assertEqual(sys.path[0], str(self.io.SITE))
            self.assertEqual(cli.os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"], "1")
            self.assertNotIn("PYTEST_ADDOPTS", cli.os.environ)
            self.assertNotIn("PYTEST_PLUGINS", cli.os.environ)
            load_core.assert_called_once_with(self.io)

    def test_warm_runtime_preserves_preexisting_package(self):
        name = cli.CORE_PACKAGE
        existing = types.ModuleType(name)
        with (
            patch.dict(sys.modules, {"pytest": self.pytest, name: existing}),
            patch.object(cli.sys, "path", list(sys.path)),
            patch.dict(cli.os.environ),
            patch.object(cli, "control_bytes") as control_read,
        ):
            with self.assertRaises(self.io.Stop):
                cli.warm_runtime(self.io)
            self.assertIs(sys.modules[name], existing)
            control_read.assert_not_called()

    def test_authenticated_core_binds_exact_three_helpers_in_dependency_order(self):
        suffixes = (
            "posttraining_safety_replay_snapshot_v1",
            "posttraining_aya_plan_snapshot_v1",
            "posttraining_aya_receipt_snapshot_v1",
        )
        self.assertEqual(tuple(item[0] for item in cli.CONTROL_SOURCES), suffixes)
        self.assertEqual(
            tuple(item[1] for item in cli.CONTROL_SOURCES),
            tuple("src/lumi/" + suffix + ".py" for suffix in suffixes),
        )
        with (
            self.isolated_core_namespace(),
            patch.object(cli, "control_bytes", wraps=cli.control_bytes) as control_read,
        ):
            core = cli.authenticated_core(self.io)
            self.assertEqual(core.__name__, cli.CORE_PACKAGE + "." + suffixes[-1])
            self.assertEqual(sys.modules[cli.CORE_PACKAGE].__path__, [])
            self.assertEqual(
                control_read.call_args_list,
                [
                    call(ROOT / relative, digest, 65_536)
                    for _suffix, relative, digest in cli.CONTROL_SOURCES
                ],
            )
            for suffix in suffixes:
                self.assertIn(cli.CORE_PACKAGE + "." + suffix, sys.modules)
            self.assertIs(
                core.Snapshot,
                sys.modules[cli.CORE_PACKAGE + "." + suffixes[0]].Snapshot,
            )
            self.assertIs(core._compile, sys.modules[cli.CORE_PACKAGE + "." + suffixes[1]]._compile)

    def test_tampered_control_prevents_all_execution_and_namespace_registration(self):
        # Only these three source-control files are read; altered bytes stay in memory.
        payloads = {
            ROOT / relative: cli.control_bytes(ROOT / relative, digest, 65_536)
            for _suffix, relative, digest in cli.CONTROL_SOURCES
        }
        for rejected in payloads:
            with self.subTest(rejected=rejected.name):
                def fake_open(path, mode, rejected=rejected):
                    self.assertEqual(mode, "rb")
                    self.assertNotIn(cli.CORE_PACKAGE, sys.modules)
                    alteration = b"\n# tampered" if path == rejected else b""
                    return io.BytesIO(payloads[path] + alteration)

                with (
                    self.isolated_core_namespace(),
                    patch.object(Path, "open", autospec=True, side_effect=fake_open),
                    patch.object(cli, "exec", create=True) as execute,
                ):
                    with self.assertRaisesRegex(ValueError, "^control_binding_rejected$"):
                        cli.authenticated_core(self.io)
                    execute.assert_not_called()
                    self.assertFalse(any(name.startswith(cli.CORE_PACKAGE) for name in sys.modules))

    def test_all_control_bytes_are_authenticated_before_any_execution(self):
        events = []

        def authenticated_bytes(path, _digest, maximum):
            self.assertEqual(maximum, 65_536)
            self.assertNotIn(cli.CORE_PACKAGE, sys.modules)
            events.append(("read", str(path)))
            return b"SYNTHETIC_MARKER = 1\n"

        def execute_authenticated(_code, namespace):
            self.assertEqual(sum(event[0] == "read" for event in events), 3)
            events.append(("exec", namespace["__name__"]))

        with (
            self.isolated_core_namespace(),
            patch.object(cli, "control_bytes", side_effect=authenticated_bytes),
            patch.object(cli, "exec", create=True, side_effect=execute_authenticated),
        ):
            cli.authenticated_core(self.io)
        self.assertEqual([event[0] for event in events], ["read"] * 3 + ["exec"] * 3)

    def test_authenticated_core_rejects_and_preserves_any_prefix_collision(self):
        for suffix in ("", ".posttraining_safety_replay_snapshot_v1", ".foreign.descendant"):
            with self.subTest(suffix=suffix), self.isolated_core_namespace():
                name = cli.CORE_PACKAGE + suffix
                existing = types.ModuleType(name)
                sys.modules[name] = existing
                with (
                    patch.object(cli, "control_bytes") as control_read,
                    patch.object(cli, "exec", create=True) as execute,
                ):
                    with self.assertRaises(self.io.Stop):
                        cli.authenticated_core(self.io)
                    control_read.assert_not_called()
                    execute.assert_not_called()
                    self.assertIs(sys.modules[name], existing)

    def test_core_execution_fault_cleans_namespace_and_main_hides_exception_text(self):
        for position in range(3):
            with self.subTest(position=position):
                self.configure_main()
                payloads = [b"SYNTHETIC_MARKER = 1\n"] * 3
                payloads[position] = b"raise RuntimeError('SYNTHETIC_SECRET')\n"
                self.warm_runtime.side_effect = lambda _io: (
                    self.pytest, cli.authenticated_core(self.io)
                )
                with (
                    self.isolated_core_namespace(),
                    patch.object(cli, "control_bytes", side_effect=payloads),
                ):
                    result, output = self.run_main([cli.REAL_MODE])
                    self.assert_stopped(result, output, "audit_failed")
                    self.assertFalse(any(name.startswith(cli.CORE_PACKAGE) for name in sys.modules))
                self.inputs_factory.assert_not_called()
                self.inputs.read.assert_not_called()
                self.addaudithook.assert_not_called()

    def test_source_authentication_failure_precedes_any_receipt_read(self):
        self.assertEqual(
            cli.READ_ROLES,
            ("review_implementation", "hash_helper_source", "test_source", "receipt"),
        )
        for rejected in cli.READ_ROLES[:-1]:
            with self.subTest(rejected=rejected):
                self.configure_main()
                original_read = self.inputs.read.side_effect

                def reject_source(path, binding, rejected=rejected, original_read=original_read):
                    if path == self.spec["bindings"][rejected]["path"]:
                        raise self.io.Stop("input_hash_mismatch")
                    return original_read(path, binding)

                self.inputs.read.side_effect = reject_source
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, "input_hash_mismatch")
                self.assertNotIn(
                    self.spec["bindings"]["receipt"]["path"],
                    [entry.args[0] for entry in self.inputs.read.call_args_list],
                )
                self.core.build_test_module.assert_not_called()
                self.inputs.close.assert_called_once_with()

    def test_runtime_flags_fail_before_manifest_and_inputs(self):
        for flags in ({"no_site": 0}, {"optimize": 1}, {"dont_write_bytecode": False}):
            with self.subTest(flags=flags):
                self.configure_main()
                result, output = self.run_main([cli.REAL_MODE], **flags)
                self.assert_stopped(result, output, "runtime_rejected")
                self.read_spec.assert_not_called()
                self.network_denial.assert_not_called()
                self.warm_runtime.assert_not_called()
                self.inputs_factory.assert_not_called()
                self.inputs.read.assert_not_called()
                self.assertEqual(output["bound_input_opens"], 0)

    def test_unknown_missing_or_multiple_arguments_fail_before_inputs(self):
        for arguments in (
            [], ["wrong"], [cli.REAL_MODE, "extra"], ["--check-runtime", cli.REAL_MODE]
        ):
            with self.subTest(arguments=arguments):
                self.configure_main()
                result, output = self.run_main(arguments)
                self.assert_stopped(result, output, "arguments_rejected")
                self.read_spec.assert_not_called()
                self.network_denial.assert_not_called()
                self.warm_runtime.assert_not_called()
                self.inputs_factory.assert_not_called()
                self.inputs.read.assert_not_called()

    def test_runtime_probe_has_no_bound_reads_or_receipt_claims(self):
        result, output = self.run_main(["--check-runtime"])
        self.assertEqual(result, 0)
        self.assertEqual(output["status"], "PASS_RUNTIME_ONLY")
        self.assertEqual(output["bound_input_opens"], 0)
        self.assertEqual(output["bound_input_bytes_read"], 0)
        self.assertFalse(output["original_test_body_executed"])
        self.assertEqual(output["snapshot_backed_pytest_nodes_passed"], 0)
        self.assertFalse(output["input_files_preserved"])
        self.assertNotIn("receipt_claim_accepted_records", output)
        self.inputs_factory.assert_not_called()
        self.inputs.read.assert_not_called()
        self.inputs.verify.assert_not_called()
        self.inputs.close.assert_not_called()
        self.core.build_test_module.assert_not_called()
        self.run_projected_test.assert_called_once()

    def test_success_reports_observed_counts_without_broader_authorization(self):
        result, output = self.run_main([cli.REAL_MODE])
        self.assertEqual(result, 0)
        self.assertEqual(output["status"], "PASS_AUTHENTICATED_AYA_RECEIPT_TEST_ONLY")
        self.assertTrue(output["original_test_body_executed"])
        self.assertEqual(output["snapshot_backed_pytest_nodes_passed"], 1)
        self.assertEqual(output["bound_input_opens"], 4)
        self.assertEqual(output["bound_input_bytes_read"], 45_471)
        self.assertEqual(output["snapshot_accounting"], self.summary)
        self.assertEqual(output["pytest_lifecycle"], self.phases)
        self.assertEqual(output["receipt_claim_accepted_records"], 32)
        self.assertEqual(output["receipt_claim_rejected_records"], 0)
        self.assertTrue(output["input_files_preserved"])
        self.assertEqual(
            (self.io.MAX_FILES, self.io.MAX_BYTES, self.io.MAX_SECONDS), (4, 45_471, 30)
        )
        self.assertEqual((self.lifecycle.MAX_FILES, self.lifecycle.MAX_BYTES), (17, 10_433_360))
        self.assertEqual(
            self.inputs.read.call_args_list,
            [
                call(self.spec["bindings"][role]["path"], self.spec["bindings"][role])
                for role in cli.READ_ROLES
            ],
        )
        self.inputs.verify.assert_called_once_with()
        self.inputs.close.assert_called_once_with()
        self.run_projected_test.assert_called_once_with(self.io, self.pytest, self.module)
        self.assertIs(self.module.pytest, self.pytest)
        for key in (
            "ordinary_checkout_test_passed",
            "original_module_fully_imported",
            "owner_decisions_reproduced",
            "human_authorship_independently_verified",
            "candidate_sample_reconstructed",
            "full_private_integration_passed",
            "execution_authorized",
            "production_eligible",
            "tokenizer_loaded",
            "network_used",
            "gpu_used",
            "inference_called",
        ):
            self.assertIs(output[key], False, key)
        for key in ("source_rows_read", "candidate_rows_read", "input_files_written"):
            self.assertEqual(output[key], 0, key)

    def test_invalid_manifest_fails_before_any_input_or_runtime_startup(self):
        for mutation in ("nodeid", "file_count", "total_bytes", "receipt_bytes"):
            with self.subTest(mutation=mutation):
                self.configure_main()
                if mutation == "nodeid":
                    self.spec["nodeid"] = "wrong"
                elif mutation == "file_count":
                    del self.spec["bindings"]["test_source"]
                elif mutation == "total_bytes":
                    self.spec["bindings"]["test_source"]["bytes"] += 1
                else:
                    self.spec["bindings"]["receipt"]["bytes"] += 1
                    self.spec["bindings"]["test_source"]["bytes"] -= 1
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, "spec_rejected")
                self.network_denial.assert_not_called()
                self.warm_runtime.assert_not_called()
                self.inputs_factory.assert_not_called()
                self.inputs.read.assert_not_called()

    def test_bound_input_counter_mismatch_cannot_pass(self):
        for field, initial in (("opens", -1), ("opens", 1), ("bytes_read", -1), ("bytes_read", 1)):
            with self.subTest(field=field, initial=initial):
                self.configure_main()
                setattr(self.inputs, field, initial)
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, "read_budget")
                self.snapshot.summary.assert_not_called()
                self.inputs.close.assert_called_once_with()

    def test_snapshot_counter_or_filesystem_access_mismatch_cannot_pass(self):
        for field, value in (
            ("virtual_opens", 0),
            ("virtual_opens", 2),
            ("logical_read_bytes_debited", 1666),
            ("logical_read_bytes_debited", 1668),
            ("bound_blob_count", 2),
            ("filesystem_access_performed", True),
        ):
            with self.subTest(field=field, value=value):
                self.configure_main()
                self.summary[field] = value
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, "reproduction_rejected")
                self.inputs.close.assert_called_once_with()

    def test_input_preservation_or_boundary_failure_cannot_pass(self):
        for failure in ("input_changed", "boundary_violation"):
            with self.subTest(failure=failure):
                self.configure_main()
                if failure == "input_changed":
                    self.inputs.verify.side_effect = self.io.Stop(failure)
                else:
                    self.boundary.blocked = 1
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, failure)
                self.inputs.close.assert_called_once_with()

    def test_cleanup_and_post_cleanup_deadline_failures_block_success(self):
        for failure in ("close_exception", "close_stop", "alarm", "deadline"):
            with self.subTest(failure=failure):
                self.configure_main()
                if failure == "close_exception":
                    self.inputs.close.side_effect = OSError("SYNTHETIC_SECRET")
                elif failure == "close_stop":
                    self.inputs.close.side_effect = self.io.Stop("input_changed")
                elif failure == "alarm":
                    self.alarm.side_effect = [None, OSError("SYNTHETIC_SECRET")]
                result, output = self.run_main(
                    [cli.REAL_MODE], times=(0, 1, 30 if failure == "deadline" else 2)
                )
                code = {
                    "close_exception": "input_cleanup_failed",
                    "close_stop": "input_cleanup_failed",
                    "alarm": "deadline_cleanup_failed",
                    "deadline": "deadline_exceeded",
                }[failure]
                self.assert_stopped(result, output, code)
                self.inputs.close.assert_called_once_with()
                self.assertEqual(self.alarm.call_args_list, [call(30), call(0)])

    def test_opaque_exceptions_at_each_stage_are_not_retained(self):
        for stage in ("spec", "runtime", "read", "projection", "pytest", "summary"):
            with self.subTest(stage=stage):
                self.configure_main()
                target = {
                    "spec": self.read_spec,
                    "runtime": self.warm_runtime,
                    "read": self.inputs.read,
                    "projection": self.core.build_test_module,
                    "pytest": self.run_projected_test,
                    "summary": self.snapshot.summary,
                }[stage]
                target.side_effect = RuntimeError("SYNTHETIC_SECRET")
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, "audit_failed")
                if stage not in {"spec", "runtime"}:
                    self.inputs.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
