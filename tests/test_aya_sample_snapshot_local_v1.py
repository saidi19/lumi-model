"""Synthetic sample-CLI guards; never open private objects or existing plan/report bodies."""

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
    "aya_sample_snapshot_cli_tests", ROOT / "scripts/check_aya_sample_snapshot_local_v1.py"
)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class AyaSampleLocalTests(unittest.TestCase):
    def setUp(self):
        # These factories authenticate source-control helpers only, never their main functions.
        self.helper = cli.helpers()
        self.lifecycle = self.helper.lifecycle_helpers()
        self.io = self.lifecycle.io_primitives()
        self.configure_main()

    def configure_main(self):
        self.spec = cli.read_spec()
        self.inputs = Mock(opens=0, bytes_read=0)

        def synthetic_read(_path, binding):
            self.inputs.opens += 1
            self.inputs.bytes_read += binding["bytes"]
            return b"synthetic bound payload; no file opened"

        self.inputs.read.side_effect = synthetic_read
        self.boundary = Mock(blocked=0, roots=())
        self.summary = {
            "virtual_opens": 9,
            "logical_read_bytes_debited": 11_523_272,
            "bound_blob_count": 7,
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
            patch.object(cli, "helpers", return_value=self.helper),
            patch.object(self.helper, "lifecycle_helpers", return_value=self.lifecycle),
            patch.object(self.lifecycle, "io_primitives", return_value=self.io),
            patch.object(self.helper, "main") as old_main,
            patch.object(self.lifecycle, "main") as older_main,
            patch.object(cli, "read_spec", self.read_spec),
            patch.object(self.io, "Boundary", return_value=self.boundary) as boundary_factory,
            patch.object(self.io, "BoundInputs", return_value=self.inputs) as inputs_factory,
            patch.object(self.lifecycle, "network_denial") as network_denial,
            patch.object(self.helper, "warm_runtime", self.warm_runtime),
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
            old_main.assert_not_called()
            older_main.assert_not_called()
        self.boundary_factory = boundary_factory
        self.inputs_factory = inputs_factory
        self.network_denial = network_denial
        self.addaudithook = addaudithook
        self.raw_output = capture.getvalue()
        self.assertEqual(self.alarm.call_args_list[-1], call(0))
        return result, json.loads(self.raw_output)

    def assert_stopped(self, result, output, code):
        self.assertEqual(result, 1)
        self.assertEqual(output["status"], "STOP")
        self.assertEqual(output["failure_code"], code)
        self.assertFalse(output["original_test_body_executed"])
        self.assertFalse(output["candidate_sample_reconstructed"])
        self.assertEqual(output["snapshot_backed_pytest_nodes_passed"], 0)
        self.assertNotIn("snapshot_accounting", output)
        self.assertNotIn("sample_records_validated", output)
        self.assertNotIn("SYNTHETIC_SECRET", self.raw_output)

    @contextlib.contextmanager
    def isolated_core_namespace(self):
        with patch.dict(sys.modules):
            for name in tuple(sys.modules):
                if name == cli.CORE_PACKAGE or name.startswith(cli.CORE_PACKAGE + "."):
                    del sys.modules[name]
            yield

    def test_manifest_identity_limits_and_candidate_last_order_are_exact(self):
        self.assertEqual((cli.MAX_FILES, cli.MAX_BYTES, cli.MAX_SECONDS), (10, 6_551_911, 30))
        self.assertEqual((cli.CANDIDATE_BYTES, cli.PRIVATE_BYTES), (5_016_358, 6_480_319))
        self.assertEqual((cli.SNAPSHOT_OPENS, cli.SNAPSHOT_BYTES), (9, 11_523_272))
        self.assertEqual(
            cli.NODEID,
            "tests/test_aya_bengali_owner_review.py::test_actual_plan_rebuilds_exact_sample",
        )
        self.assertEqual(self.spec["nodeid"], cli.NODEID)
        self.assertEqual(
            cli.READ_ROLES,
            (
                "plan", "review_implementation", "file_hash_source", "sample_helper_source",
                "test_source", "receipt_finalizer", "materialization_report",
                "owner_review_packet", "attribution_manifest", "accepted_records",
            ),
        )
        self.assertEqual(set(self.spec["bindings"]), set(cli.READ_ROLES))
        self.assertEqual(sum(b["bytes"] for b in self.spec["bindings"].values()), 6_551_911)
        self.assertEqual(
            sum(self.spec["bindings"][role]["bytes"] for role in cli.READ_ROLES[-3:]), 6_480_319
        )
        self.assertEqual(
            self.spec["bindings"]["accepted_records"]["sha256"],
            "169cc09f53e2b799058328ab3be946f151f1b2143b7288a757be0006a5901aee",
        )
        self.assertEqual(
            self.spec["limits"],
            {
                "maximum_bound_file_opens": 10,
                "maximum_bound_bytes": 6_551_911,
                "maximum_runtime_seconds": 30,
                "maximum_private_objects": 3,
                "maximum_private_object_bytes": 6_480_319,
                "candidate_file_bytes": 5_016_358,
                "candidate_records": 1478,
                "sample_records": 32,
                "maximum_snapshot_opens": 9,
                "maximum_snapshot_read_bytes": 11_523_272,
                "upstream_source_rows": 0,
                "artifact_write_bytes": 0,
                "network_bytes": 0,
                "inference_calls": 0,
            },
        )

    def test_control_reads_enforce_hash_and_cap_during_read(self):
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

    def test_helpers_are_fresh_hash_bound_and_preserve_original_namespaces(self):
        source = cli.control_bytes(cli.HELPER_PATH, cli.HELPER_SHA256, 65_536)
        original = types.ModuleType("_synthetic_original_receipt_helper")
        original.__file__ = str(cli.HELPER_PATH)
        exec(compile(source, str(cli.HELPER_PATH), "exec"), original.__dict__)
        existing = types.ModuleType("_aya_sample_control_helpers")
        with patch.dict(sys.modules, {existing.__name__: existing}):
            fresh = cli.helpers()
            self.assertIs(sys.modules[existing.__name__], existing)
        self.assertIsNot(fresh, self.helper)
        self.assertEqual(
            (fresh.MODULE_NAME, fresh.TEST_NAME, fresh.TEST_PATH, fresh.NODEID),
            (cli.MODULE_NAME, cli.TEST_NAME, cli.TEST_PATH, cli.NODEID),
        )
        self.assertEqual((fresh.CORE_PACKAGE, fresh.CONTROL_SOURCES), (
            cli.CORE_PACKAGE, cli.CONTROL_SOURCES
        ))
        self.assertEqual(
            original.TEST_NAME, "test_actual_owner_receipt_is_complete_and_nonpromoting"
        )
        self.assertEqual(original.CORE_PACKAGE, "_lumi_aya_receipt_entry")
        self.assertEqual((original.MAX_FILES, original.MAX_BYTES), (4, 45_471))
        self.assertEqual((fresh.MAX_FILES, fresh.MAX_BYTES), (4, 45_471))
        self.assertEqual(
            cli.control_bytes(cli.HELPER_PATH, cli.HELPER_SHA256, 65_536), source
        )
        self.assertEqual(self.lifecycle.NODEID, cli.NODEID)
        self.assertEqual(self.lifecycle.TEST_NAME, cli.TEST_NAME)

    def test_authenticated_core_uses_exact_pinned_helpers_without_source_fallback(self):
        suffixes = (
            "posttraining_safety_replay_snapshot_v1",
            "posttraining_aya_plan_snapshot_v1",
            "posttraining_aya_sample_snapshot_v1",
        )
        self.assertEqual(tuple(item[0] for item in cli.CONTROL_SOURCES), suffixes)
        self.assertEqual(
            tuple(item[1] for item in cli.CONTROL_SOURCES),
            tuple("src/lumi/" + suffix + ".py" for suffix in suffixes),
        )
        with (
            self.isolated_core_namespace(),
            patch.object(
                self.helper, "control_bytes", wraps=self.helper.control_bytes
            ) as control_read,
        ):
            core = self.helper.authenticated_core(self.io)
            self.assertEqual(core.__name__, cli.CORE_PACKAGE + "." + suffixes[-1])
            self.assertEqual(sys.modules[cli.CORE_PACKAGE].__path__, [])
            self.assertEqual(
                control_read.call_args_list,
                [
                    call(ROOT / relative, digest, 65_536)
                    for _suffix, relative, digest in cli.CONTROL_SOURCES
                ],
            )
            self.assertTrue(callable(core.build_test_module))

    def test_all_controls_authenticate_before_execution_or_namespace_registration(self):
        events = []

        def authenticate(path, _digest, maximum):
            self.assertEqual(maximum, 65_536)
            self.assertNotIn(cli.CORE_PACKAGE, sys.modules)
            events.append(("read", str(path)))
            return b"SYNTHETIC_MARKER = 1\n"

        def execute(_code, namespace):
            self.assertEqual(sum(kind == "read" for kind, _value in events), 3)
            events.append(("exec", namespace["__name__"]))

        with (
            self.isolated_core_namespace(),
            patch.object(self.helper, "control_bytes", side_effect=authenticate),
            patch.object(self.helper, "exec", create=True, side_effect=execute),
        ):
            self.helper.authenticated_core(self.io)
        self.assertEqual([kind for kind, _value in events], ["read"] * 3 + ["exec"] * 3)

    def test_rejected_controls_never_execute_or_open_bound_inputs(self):
        for position in range(3):
            with self.subTest(position=position):
                self.configure_main()
                payloads = [b"SYNTHETIC_MARKER = 1\n"] * 3
                payloads[position] = ValueError("SYNTHETIC_SECRET")
                self.warm_runtime.side_effect = lambda _io: (
                    self.pytest, self.helper.authenticated_core(self.io)
                )
                with (
                    self.isolated_core_namespace(),
                    patch.object(self.helper, "control_bytes", side_effect=payloads),
                    patch.object(self.helper, "exec", create=True) as execute,
                ):
                    result, output = self.run_main([cli.REAL_MODE])
                    self.assert_stopped(result, output, "audit_failed")
                    execute.assert_not_called()
                    self.assertFalse(any(name.startswith(cli.CORE_PACKAGE) for name in sys.modules))
                self.warm_runtime.assert_called_once_with(self.io)
                self.inputs_factory.assert_not_called()
                self.inputs.read.assert_not_called()
                self.addaudithook.assert_not_called()

    def test_core_prefix_collisions_are_preserved_before_control_reads(self):
        for suffix in ("", ".posttraining_safety_replay_snapshot_v1", ".foreign.descendant"):
            with self.subTest(suffix=suffix), self.isolated_core_namespace():
                name = cli.CORE_PACKAGE + suffix
                existing = types.ModuleType(name)
                sys.modules[name] = existing
                with patch.object(self.helper, "control_bytes") as control_read:
                    with self.assertRaises(self.io.Stop):
                        self.helper.authenticated_core(self.io)
                    control_read.assert_not_called()
                    self.assertIs(sys.modules[name], existing)

    def test_core_execution_failure_cleans_namespace_without_raw_main_output(self):
        for position in range(3):
            with self.subTest(position=position):
                self.configure_main()
                payloads = [b"SYNTHETIC_MARKER = 1\n"] * 3
                payloads[position] = b"raise RuntimeError('SYNTHETIC_SECRET')\n"
                self.warm_runtime.side_effect = lambda _io: (
                    self.pytest, self.helper.authenticated_core(self.io)
                )
                with (
                    self.isolated_core_namespace(),
                    patch.object(self.helper, "control_bytes", side_effect=payloads),
                ):
                    result, output = self.run_main([cli.REAL_MODE])
                    self.assert_stopped(result, output, "audit_failed")
                    self.assertFalse(any(name.startswith(cli.CORE_PACKAGE) for name in sys.modules))
                self.inputs_factory.assert_not_called()
                self.inputs.read.assert_not_called()

    def test_quiet_text_discards_untrusted_pytest_output(self):
        sink = self.lifecycle.QuietText()
        self.assertFalse(sink.isatty())
        self.assertEqual(sink.encoding, "utf-8")
        self.assertEqual(sink.write("SYNTHETIC_SECRET"), 16)
        self.assertIsNone(sink.flush())
        self.assertEqual(vars(sink), {})

    def test_exact_one_node_collection_and_all_passing_phases_are_required(self):
        for items in (
            [], [types.SimpleNamespace(nodeid="wrong")],
            [types.SimpleNamespace(nodeid=cli.NODEID)] * 2,
        ):
            with self.subTest(items=items), self.assertRaises(self.io.Stop):
                self.lifecycle.OneNode(self.io).pytest_collection_modifyitems(items)
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
                        plugin.pytest_runtest_logreport(types.SimpleNamespace(
                            nodeid=cli.NODEID, when=phase,
                            outcome=outcome if phase == changed_phase else "passed",
                            longrepr="SYNTHETIC_SECRET",
                        ))
                    self.assertEqual(plugin.passed(), outcome == "passed")
                    self.assertNotIn("SYNTHETIC_SECRET", str(vars(plugin)))

    def test_unknown_duplicate_and_foreign_phase_reports_are_rejected(self):
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

    def test_preexisting_original_test_module_is_not_overwritten(self):
        existing = types.ModuleType(cli.MODULE_NAME)
        with patch.dict(sys.modules, {cli.MODULE_NAME: existing}):
            with self.assertRaises(self.io.Stop):
                self.lifecycle.run_projected_test(self.io, self.pytest, self.module)
            self.assertIs(sys.modules[cli.MODULE_NAME], existing)
        self.pytest.main.assert_not_called()

    def test_projected_pytest_requires_one_observed_pass_and_restores_outer_namespace(self):
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

        existing = types.ModuleType(cli.MODULE_NAME)
        with patch.dict(sys.modules, {cli.MODULE_NAME: existing}):
            with patch.dict(sys.modules):
                del sys.modules[cli.MODULE_NAME]
                self.pytest.main.side_effect = synthetic_pytest
                self.assertEqual(
                    self.lifecycle.run_projected_test(self.io, self.pytest, self.module),
                    self.phases,
                )
                self.assertNotIn(cli.MODULE_NAME, sys.modules)
                self.pytest.main.side_effect = None
                self.pytest.main.return_value = 0
                with self.assertRaises(self.io.Stop):
                    self.lifecycle.run_projected_test(self.io, self.pytest, self.module)
                self.assertNotIn(cli.MODULE_NAME, sys.modules)
                self.pytest.main.side_effect = RuntimeError("SYNTHETIC_SECRET")
                with self.assertRaises(RuntimeError):
                    self.lifecycle.run_projected_test(self.io, self.pytest, self.module)
                self.assertNotIn(cli.MODULE_NAME, sys.modules)
            self.assertIs(sys.modules[cli.MODULE_NAME], existing)

    def test_runtime_flags_and_bad_arguments_stop_before_manifest_or_inputs(self):
        cases = [
            ([cli.REAL_MODE], {"no_site": 0}, "runtime_rejected"),
            ([cli.REAL_MODE], {"optimize": 1}, "runtime_rejected"),
            ([cli.REAL_MODE], {"dont_write_bytecode": False}, "runtime_rejected"),
            ([], {}, "arguments_rejected"),
            (["wrong"], {}, "arguments_rejected"),
            ([cli.REAL_MODE, "extra"], {}, "arguments_rejected"),
            (["--check-runtime", cli.REAL_MODE], {}, "arguments_rejected"),
        ]
        for arguments, flags, code in cases:
            with self.subTest(arguments=arguments, flags=flags):
                self.configure_main()
                result, output = self.run_main(arguments, **flags)
                self.assert_stopped(result, output, code)
                self.read_spec.assert_not_called()
                self.network_denial.assert_not_called()
                self.warm_runtime.assert_not_called()
                self.inputs_factory.assert_not_called()
                self.inputs.read.assert_not_called()

    def test_invalid_manifest_stops_before_runtime_or_bound_reads(self):
        for mutation in ("schema", "nodeid", "roles", "total_bytes", "candidate", "private"):
            with self.subTest(mutation=mutation):
                self.configure_main()
                if mutation == "schema":
                    self.spec["schema_version"] = "wrong"
                elif mutation == "nodeid":
                    self.spec["nodeid"] = "wrong"
                elif mutation == "roles":
                    self.spec["bindings"]["wrong"] = self.spec["bindings"].pop("test_source")
                elif mutation == "total_bytes":
                    self.spec["bindings"]["test_source"]["bytes"] += 1
                else:
                    role = "accepted_records" if mutation == "candidate" else "owner_review_packet"
                    self.spec["bindings"][role]["bytes"] += 1
                    self.spec["bindings"]["test_source"]["bytes"] -= 1
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, "spec_rejected")
                self.network_denial.assert_not_called()
                self.warm_runtime.assert_not_called()
                self.inputs_factory.assert_not_called()
                self.inputs.read.assert_not_called()

    def test_runtime_probe_reads_zero_bound_inputs_and_makes_no_sample_claim(self):
        result, output = self.run_main(["--check-runtime"])
        self.assertEqual(result, 0)
        self.assertEqual(output["status"], "PASS_RUNTIME_ONLY")
        self.assertEqual(output["bound_input_opens"], 0)
        self.assertEqual(output["bound_input_bytes_read"], 0)
        self.assertEqual(output["candidate_rows_parsed"], 0)
        self.assertFalse(output["candidate_sample_reconstructed"])
        self.assertFalse(output["original_test_body_executed"])
        self.assertEqual(output["snapshot_backed_pytest_nodes_passed"], 0)
        self.assertFalse(output["input_files_preserved"])
        self.assertNotIn("candidate_records_validated", output)
        self.assertNotIn("original_sample_identity_matched", output)
        self.inputs_factory.assert_not_called()
        self.inputs.read.assert_not_called()
        self.inputs.verify.assert_not_called()
        self.inputs.close.assert_not_called()
        self.core.build_test_module.assert_not_called()
        self.warm_runtime.assert_called_once_with(self.io)
        self.run_projected_test.assert_called_once()

    def test_success_observes_exact_candidate_last_reads_and_narrow_sample_claim(self):
        result, output = self.run_main([cli.REAL_MODE])
        self.assertEqual(result, 0)
        self.assertEqual(output["status"], "PASS_AUTHENTICATED_AYA_SAMPLE_TEST_ONLY")
        self.assertTrue(output["original_test_body_executed"])
        self.assertTrue(output["candidate_sample_reconstructed"])
        self.assertTrue(output["original_sample_identity_matched"])
        self.assertTrue(output["input_files_preserved"])
        self.assertEqual(output["snapshot_backed_pytest_nodes_passed"], 1)
        self.assertEqual(output["bound_input_opens"], 10)
        self.assertEqual(output["bound_input_bytes_read"], 6_551_911)
        self.assertEqual(output["snapshot_accounting"], self.summary)
        self.assertEqual(output["pytest_lifecycle"], self.phases)
        self.assertEqual(output["candidate_records_validated"], 1478)
        self.assertEqual(output["candidate_rows_parsed"], 1478)
        self.assertEqual(output["sample_records_validated"], 32)
        self.assertEqual(
            self.inputs.read.call_args_list,
            [
                call(self.spec["bindings"][role]["path"], self.spec["bindings"][role])
                for role in cli.READ_ROLES
            ],
        )
        self.assertEqual(
            self.inputs.read.call_args_list[-1].args[0],
            self.spec["bindings"]["accepted_records"]["path"],
        )
        self.assertEqual(
            (self.io.MAX_FILES, self.io.MAX_BYTES, self.io.MAX_SECONDS), (10, 6_551_911, 30)
        )
        self.assertEqual((self.helper.MAX_FILES, self.helper.MAX_BYTES), (4, 45_471))
        self.assertEqual((self.lifecycle.MAX_FILES, self.lifecycle.MAX_BYTES), (17, 10_433_360))
        self.inputs.verify.assert_called_once_with()
        self.inputs.close.assert_called_once_with()
        self.run_projected_test.assert_called_once_with(self.io, self.pytest, self.module)
        self.core.build_test_module.assert_called_once()
        self.assertIs(self.module.pytest, self.pytest)
        for key in (
            "ordinary_checkout_test_passed", "original_module_fully_imported",
            "owner_decisions_reproduced", "new_owner_approval_created",
            "human_authorship_independently_verified", "new_candidate_constructed",
            "full_private_integration_passed", "execution_authorized", "production_eligible",
            "tokenizer_loaded", "network_used", "gpu_used", "inference_called",
        ):
            self.assertIs(output[key], False, key)
        self.assertEqual(output["upstream_source_rows_read"], 0)
        self.assertEqual(output["input_files_written"], 0)

    def test_any_bound_read_failure_is_one_pass_without_retry_or_later_candidate_read(self):
        for position, rejected in enumerate(cli.READ_ROLES):
            with self.subTest(rejected=rejected):
                self.configure_main()
                original_read = self.inputs.read.side_effect

                def reject(path, binding, rejected=rejected, original_read=original_read):
                    if path == self.spec["bindings"][rejected]["path"]:
                        raise self.io.Stop("input_hash_mismatch")
                    return original_read(path, binding)

                self.inputs.read.side_effect = reject
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, "input_hash_mismatch")
                self.assertEqual(
                    [entry.args[0] for entry in self.inputs.read.call_args_list],
                    [self.spec["bindings"][role]["path"] for role in cli.READ_ROLES[:position + 1]],
                )
                self.assertIsNone(output["candidate_rows_parsed"])
                self.core.build_test_module.assert_not_called()
                self.run_projected_test.assert_not_called()
                self.inputs.close.assert_called_once_with()

    def test_bound_read_counter_mismatch_never_publishes_pass(self):
        for field, initial in (("opens", -1), ("opens", 1), ("bytes_read", -1), ("bytes_read", 1)):
            with self.subTest(field=field, initial=initial):
                self.configure_main()
                setattr(self.inputs, field, initial)
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, "read_budget")
                self.snapshot.summary.assert_not_called()
                self.inputs.close.assert_called_once_with()

    def test_snapshot_counters_or_filesystem_access_mismatch_cannot_pass(self):
        for field, value in (
            ("virtual_opens", 8), ("virtual_opens", 10),
            ("logical_read_bytes_debited", 11_523_271),
            ("logical_read_bytes_debited", 11_523_273),
            ("bound_blob_count", 6), ("bound_blob_count", 8),
            ("filesystem_access_performed", True),
        ):
            with self.subTest(field=field, value=value):
                self.configure_main()
                self.summary[field] = value
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, "reproduction_rejected")
                self.inputs.close.assert_called_once_with()

    def test_preservation_boundary_and_pytest_failure_stop_without_retry(self):
        for code in ("input_changed", "boundary_violation", "reproduction_rejected"):
            with self.subTest(code=code):
                self.configure_main()
                if code == "input_changed":
                    self.inputs.verify.side_effect = self.io.Stop(code)
                elif code == "boundary_violation":
                    self.boundary.blocked = 1
                else:
                    self.run_projected_test.side_effect = self.io.Stop(code)
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, code)
                self.run_projected_test.assert_called_once_with(self.io, self.pytest, self.module)
                self.inputs.close.assert_called_once_with()

    def test_cleanup_or_post_cleanup_deadline_failure_prevents_pass(self):
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

    def test_opaque_exceptions_at_each_stage_are_not_retained_or_retried(self):
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
                target.assert_called_once()
                if stage not in {"spec", "runtime"}:
                    self.inputs.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
