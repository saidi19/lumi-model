"""Synthetic prerequisite CLI guards; no private, old-plan/report, or build/lib reads."""

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
    "aya_prerequisite_cli_tests", ROOT / "scripts/check_aya_prerequisite_snapshot_local_v1.py"
)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)
HISTORICAL_SOURCE = "tests/fixtures/aya_source_registry_9f20652.py.txt"
LOGICAL_SOURCE = "src/lumi/posttraining_sources.py"
HISTORICAL_SHA256 = "9f20652e85d0aede7b6ea75799f6e9746c5f35487b28e4bf39bdf854c80330e8"


class AyaPrerequisiteLocalTests(unittest.TestCase):
    def setUp(self):
        # Factory calls authenticate only source-control helpers, never their mains.
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
            return b"synthetic payload; no bound file read or source execution"

        self.inputs.read.side_effect = synthetic_read
        self.boundary = Mock(blocked=0, roots=())
        self.summary = {
            "virtual_opens": 12,
            "logical_read_bytes_debited": 2_842_451,
            "bound_blob_count": 9,
            "filesystem_access_performed": False,
        }
        self.snapshot = Mock(summary=Mock(return_value=self.summary))
        self.module = self.lifecycle.synthetic_module()
        self.pytest, self.core = Mock(), Mock()
        self.core.build_test_module.return_value = (self.module, self.snapshot)
        self.phases = {
            "collected": 1,
            "phases": {"setup": "passed", "call": "passed", "teardown": "passed"},
        }
        self.read_spec = Mock(return_value=self.spec)
        self.warm_runtime = Mock(return_value=(self.pytest, self.core))
        self.run_projected_test = Mock(return_value=self.phases)
        self.alarm = Mock()

    def run_main(self, arguments, *, flags=None, bytecode=True, times=(0, 1, 2)):
        capture = io.StringIO()
        with (
            patch.object(cli, "helpers", return_value=self.helper),
            patch.object(self.helper, "lifecycle_helpers", return_value=self.lifecycle),
            patch.object(self.lifecycle, "io_primitives", return_value=self.io),
            patch.object(self.helper, "main") as old_main,
            patch.object(self.lifecycle, "main") as older_main,
            patch.object(cli, "read_spec", self.read_spec),
            patch.object(self.io, "Boundary", return_value=self.boundary),
            patch.object(self.io, "BoundInputs", return_value=self.inputs) as inputs_factory,
            patch.object(self.lifecycle, "network_denial") as network_denial,
            patch.object(self.helper, "warm_runtime", self.warm_runtime),
            patch.object(self.lifecycle, "run_projected_test", self.run_projected_test),
            patch.object(cli.sys, "flags", types.SimpleNamespace(**(flags or {
                "no_site": 1, "optimize": 0
            }))),
            patch.object(cli.sys, "dont_write_bytecode", bytecode),
            patch.object(cli.sys, "addaudithook") as addaudithook,
            patch.object(cli.signal, "signal"),
            patch.object(cli.signal, "alarm", self.alarm),
            patch.object(cli.time, "monotonic", side_effect=times),
            contextlib.redirect_stdout(capture),
        ):
            result = cli.main(arguments)
            old_main.assert_not_called()
            older_main.assert_not_called()
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
        self.assertFalse(output["prerequisite_evidence_verified"])
        self.assertFalse(output["historical_source_copy_authenticated"])
        self.assertEqual(output["snapshot_backed_pytest_nodes_passed"], 0)
        self.assertNotIn("snapshot_accounting", output)
        self.assertNotIn("attribution_records_validated", output)
        self.assertNotIn("SYNTHETIC_SECRET", self.raw_output)

    @contextlib.contextmanager
    def isolated_core_namespace(self):
        with patch.dict(sys.modules):
            for name in tuple(sys.modules):
                if name == cli.CORE_PACKAGE or name.startswith(cli.CORE_PACKAGE + "."):
                    del sys.modules[name]
            yield

    def expected_reads(self):
        return [
            call(cli.PHYSICAL_READ_PATHS.get(role, binding["path"]), binding)
            for role in cli.READ_ROLES
            for binding in (self.spec["bindings"][role],)
        ]

    def test_exact_manifest_limits_identity_and_historical_mapping(self):
        self.assertEqual((cli.MAX_FILES, cli.MAX_BYTES, cli.MAX_SECONDS), (11, 1_472_969, 30))
        self.assertEqual((cli.ATTRIBUTION_BYTES, cli.EVIDENCE_BYTES), (1_387_842, 1_393_247))
        self.assertEqual((cli.SNAPSHOT_OPENS, cli.SNAPSHOT_BYTES), (12, 2_842_451))
        self.assertEqual(
            cli.NODEID,
            "tests/test_aya_source_promotion.py::test_exact_reviewed_aya_prerequisites_pass",
        )
        self.assertEqual(self.spec["nodeid"], cli.NODEID)
        self.assertEqual(
            cli.READ_ROLES,
            (
                "plan", "data_contract", "predecessor_registry", "promotion_implementation",
                "promotion_runner", "source_registry_implementation", "hash_helper_source",
                "test_source", "materialization_report", "owner_review_receipt",
                "attribution_manifest",
            ),
        )
        self.assertEqual(set(self.spec["bindings"]), set(cli.READ_ROLES))
        self.assertEqual(sum(b["bytes"] for b in self.spec["bindings"].values()), 1_472_969)
        self.assertEqual(
            self.spec["bindings"]["source_registry_implementation"],
            {"path": LOGICAL_SOURCE, "bytes": 14927, "sha256": HISTORICAL_SHA256},
        )
        self.assertEqual(cli.PHYSICAL_READ_PATHS, {
            "source_registry_implementation": HISTORICAL_SOURCE
        })
        self.assertEqual(self.spec["physical_read_paths"], cli.PHYSICAL_READ_PATHS)
        self.assertEqual(
            self.spec["limits"],
            {
                "maximum_bound_file_opens": 11,
                "maximum_bound_bytes": 1_472_969,
                "maximum_runtime_seconds": 30,
                "maximum_evidence_objects": 3,
                "maximum_evidence_bytes": 1_393_247,
                "attribution_records": 1478,
                "maximum_snapshot_opens": 12,
                "maximum_snapshot_read_bytes": 2_842_451,
                "upstream_source_rows": 0,
                "candidate_rows": 0,
                "artifact_write_bytes": 0,
                "network_bytes": 0,
                "inference_calls": 0,
            },
        )

    def test_control_cap_and_hash_fail_before_untrusted_content_is_used(self):
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

    def test_new_public_source_fixture_has_exact_old_hash_length_and_capped_read(self):
        payload = cli.control_bytes(ROOT / HISTORICAL_SOURCE, HISTORICAL_SHA256, 14927)
        self.assertEqual(len(payload), 14927)
        self.assertEqual(cli.hashlib.sha256(payload).hexdigest(), HISTORICAL_SHA256)
        for content in (payload, payload + b"\n"):
            with self.subTest(length=len(content)):
                stream = io.BytesIO(content)
                path = Mock(open=Mock(return_value=stream))
                with patch.object(stream, "read", wraps=stream.read) as read:
                    if len(content) == 14927:
                        self.assertEqual(cli.control_bytes(path, HISTORICAL_SHA256, 14927), payload)
                    else:
                        with self.assertRaisesRegex(ValueError, "^control_binding_rejected$"):
                            cli.control_bytes(path, HISTORICAL_SHA256, 14927)
                    read.assert_called_once_with(14928)

    def test_fresh_helper_identity_preserves_receipt_helper_and_installed_namespace(self):
        source = cli.control_bytes(cli.HELPER_PATH, cli.HELPER_SHA256, 65_536)
        original = types.ModuleType("_synthetic_original_receipt_helper")
        original.__file__ = str(cli.HELPER_PATH)
        exec(compile(source, str(cli.HELPER_PATH), "exec"), original.__dict__)
        existing = types.ModuleType("_aya_prerequisite_control_helpers")
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
        self.assertEqual(self.lifecycle.NODEID, cli.NODEID)
        self.assertEqual(cli.control_bytes(cli.HELPER_PATH, cli.HELPER_SHA256, 65_536), source)

    def test_core_uses_only_three_authenticated_helpers_without_historical_fixture_import(self):
        suffixes = (
            "posttraining_safety_replay_snapshot_v1",
            "posttraining_aya_plan_snapshot_v1",
            "posttraining_aya_prerequisite_snapshot_v1",
        )
        self.assertEqual(tuple(item[0] for item in cli.CONTROL_SOURCES), suffixes)
        self.assertEqual(
            tuple(item[1] for item in cli.CONTROL_SOURCES),
            tuple("src/lumi/" + suffix + ".py" for suffix in suffixes),
        )
        with (
            self.isolated_core_namespace(),
            patch.object(self.helper, "control_bytes", wraps=self.helper.control_bytes) as reads,
        ):
            core = self.helper.authenticated_core(self.io)
            self.assertEqual(core.__name__, cli.CORE_PACKAGE + "." + suffixes[-1])
            self.assertEqual(sys.modules[cli.CORE_PACKAGE].__path__, [])
            self.assertTrue(callable(core.build_test_module))
            self.assertEqual(reads.call_args_list, [
                call(ROOT / relative, digest, 65_536)
                for _suffix, relative, digest in cli.CONTROL_SOURCES
            ])
            self.assertNotIn(
                ROOT / HISTORICAL_SOURCE, [item.args[0] for item in reads.call_args_list]
            )

    def test_controls_authenticate_before_execution_and_prefix_collisions_are_preserved(self):
        events = []

        def authenticate(_path, _digest, maximum):
            self.assertEqual(maximum, 65_536)
            self.assertNotIn(cli.CORE_PACKAGE, sys.modules)
            events.append("read")
            return b"SYNTHETIC_MARKER = 1\n"

        def execute(_code, _namespace):
            self.assertEqual(events.count("read"), 3)
            events.append("exec")

        with (
            self.isolated_core_namespace(),
            patch.object(self.helper, "control_bytes", side_effect=authenticate),
            patch.object(self.helper, "exec", create=True, side_effect=execute),
        ):
            self.helper.authenticated_core(self.io)
        self.assertEqual(events, ["read"] * 3 + ["exec"] * 3)
        for suffix in ("", ".posttraining_safety_replay_snapshot_v1", ".foreign.descendant"):
            with self.subTest(suffix=suffix), self.isolated_core_namespace():
                name = cli.CORE_PACKAGE + suffix
                existing = types.ModuleType(name)
                sys.modules[name] = existing
                with patch.object(self.helper, "control_bytes") as reads:
                    with self.assertRaises(self.io.Stop):
                        self.helper.authenticated_core(self.io)
                    reads.assert_not_called()
                    self.assertIs(sys.modules[name], existing)

    def test_failed_control_authentication_or_execution_stops_before_bound_inputs(self):
        for failure in ("authenticate", "execute"):
            for position in range(3):
                with self.subTest(failure=failure, position=position):
                    self.configure_main()
                    payloads = [b"SYNTHETIC_MARKER = 1\n"] * 3
                    payloads[position] = (
                        ValueError("SYNTHETIC_SECRET") if failure == "authenticate"
                        else b"raise RuntimeError('SYNTHETIC_SECRET')\n"
                    )
                    self.warm_runtime.side_effect = lambda _io: (
                        self.pytest, self.helper.authenticated_core(self.io)
                    )
                    with (
                        self.isolated_core_namespace(),
                        patch.object(self.helper, "control_bytes", side_effect=payloads),
                    ):
                        result, output = self.run_main([cli.REAL_MODE])
                        self.assert_stopped(result, output, "audit_failed")
                        self.assertFalse(any(
                            name.startswith(cli.CORE_PACKAGE) for name in sys.modules
                        ))
                    self.inputs_factory.assert_not_called()
                    self.inputs.read.assert_not_called()
                    self.addaudithook.assert_not_called()
                    self.warm_runtime.assert_called_once_with(self.io)

    def test_quiet_output_and_exact_single_node_lifecycle(self):
        sink = self.lifecycle.QuietText()
        self.assertEqual(sink.write("SYNTHETIC_SECRET"), 16)
        self.assertFalse(sink.isatty())
        self.assertEqual(sink.encoding, "utf-8")
        self.assertIsNone(sink.flush())
        self.assertEqual(vars(sink), {})
        for items in (
            [], [types.SimpleNamespace(nodeid="wrong")],
            [types.SimpleNamespace(nodeid=cli.NODEID)] * 2,
        ):
            with self.subTest(items=items), self.assertRaises(self.io.Stop):
                self.lifecycle.OneNode(self.io).pytest_collection_modifyitems(items)
        for phase in ("setup", "call", "teardown"):
            for outcome in ("passed", "failed", "skipped", "missing"):
                with self.subTest(phase=phase, outcome=outcome):
                    plugin = self.lifecycle.OneNode(self.io)
                    plugin.pytest_collection_modifyitems([types.SimpleNamespace(nodeid=cli.NODEID)])
                    for reported in ("setup", "call", "teardown"):
                        if reported == phase and outcome == "missing":
                            continue
                        plugin.pytest_runtest_logreport(types.SimpleNamespace(
                            nodeid=cli.NODEID, when=reported,
                            outcome=outcome if reported == phase else "passed",
                            longrepr="SYNTHETIC_SECRET",
                        ))
                    self.assertEqual(plugin.passed(), outcome == "passed")
                    self.assertNotIn("SYNTHETIC_SECRET", str(vars(plugin)))

    def test_foreign_duplicate_and_unknown_lifecycle_reports_cannot_pass(self):
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

    def test_test_namespace_collision_is_preserved_and_failed_projection_is_removed(self):
        existing = types.ModuleType(cli.MODULE_NAME)
        with patch.dict(sys.modules, {cli.MODULE_NAME: existing}):
            with self.assertRaises(self.io.Stop):
                self.lifecycle.run_projected_test(self.io, self.pytest, self.module)
            self.assertIs(sys.modules[cli.MODULE_NAME], existing)
            self.pytest.main.assert_not_called()
            with patch.dict(sys.modules):
                del sys.modules[cli.MODULE_NAME]
                for failure in (None, RuntimeError("SYNTHETIC_SECRET")):
                    with self.subTest(failure=type(failure).__name__):
                        self.pytest.main.side_effect = failure
                        self.pytest.main.return_value = 0
                        with self.assertRaises((self.io.Stop, RuntimeError)):
                            self.lifecycle.run_projected_test(self.io, self.pytest, self.module)
                        self.assertNotIn(cli.MODULE_NAME, sys.modules)
            self.assertIs(sys.modules[cli.MODULE_NAME], existing)

    def test_successful_projected_runner_observes_only_the_exact_test(self):
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
        self.pytest.main.assert_called_once()

    def test_invalid_flags_or_arguments_stop_before_manifest_and_inputs(self):
        cases = (
            ([cli.REAL_MODE], {"flags": {"no_site": 0, "optimize": 0}}, "runtime_rejected"),
            ([cli.REAL_MODE], {"flags": {"no_site": 1, "optimize": 1}}, "runtime_rejected"),
            ([cli.REAL_MODE], {"bytecode": False}, "runtime_rejected"),
            ([], {}, "arguments_rejected"),
            (["wrong"], {}, "arguments_rejected"),
            ([cli.REAL_MODE, "extra"], {}, "arguments_rejected"),
            (["--check-runtime", cli.REAL_MODE], {}, "arguments_rejected"),
        )
        for arguments, options, code in cases:
            with self.subTest(arguments=arguments, options=options):
                self.configure_main()
                result, output = self.run_main(arguments, **options)
                self.assert_stopped(result, output, code)
                self.read_spec.assert_not_called()
                self.warm_runtime.assert_not_called()
                self.network_denial.assert_not_called()
                self.inputs_factory.assert_not_called()
                self.inputs.read.assert_not_called()

    def test_unknown_missing_or_modified_physical_mapping_is_rejected_without_fallback(self):
        for mapping in (
            {},
            {"unknown": HISTORICAL_SOURCE},
            {"source_registry_implementation": LOGICAL_SOURCE},
            {"source_registry_implementation": "build/lib/lumi/unbound.py"},
            {"source_registry_implementation": HISTORICAL_SOURCE, "plan": "unbound.json"},
        ):
            with self.subTest(mapping=mapping):
                self.configure_main()
                self.spec["physical_read_paths"] = mapping
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, "spec_rejected")
                self.warm_runtime.assert_not_called()
                self.inputs_factory.assert_not_called()
                self.inputs.read.assert_not_called()

    def test_invalid_manifest_schema_identity_roles_and_budgets_stop_before_reads(self):
        for mutation in ("schema", "nodeid", "roles", "total_bytes", "attribution", "evidence"):
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
                    role = (
                        "attribution_manifest"
                        if mutation == "attribution" else "owner_review_receipt"
                    )
                    self.spec["bindings"][role]["bytes"] += 1
                    self.spec["bindings"]["test_source"]["bytes"] -= 1
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, "spec_rejected")
                self.warm_runtime.assert_not_called()
                self.inputs_factory.assert_not_called()
                self.inputs.read.assert_not_called()

    def test_runtime_probe_has_zero_bound_reads_and_no_prerequisite_claim(self):
        result, output = self.run_main(["--check-runtime"])
        self.assertEqual(result, 0)
        self.assertEqual(output["status"], "PASS_RUNTIME_ONLY")
        for key in (
            "bound_input_opens", "bound_input_bytes_read", "candidate_rows_read",
            "attribution_records_parsed", "snapshot_backed_pytest_nodes_passed",
        ):
            self.assertEqual(output[key], 0, key)
        self.assertFalse(output["prerequisite_evidence_verified"])
        self.assertFalse(output["historical_source_copy_authenticated"])
        self.assertFalse(output["original_test_body_executed"])
        self.assertFalse(output["input_files_preserved"])
        self.inputs_factory.assert_not_called()
        self.inputs.read.assert_not_called()
        self.inputs.verify.assert_not_called()
        self.inputs.close.assert_not_called()
        self.core.build_test_module.assert_not_called()
        self.run_projected_test.assert_called_once()

    def test_success_keeps_logical_binding_hash_and_reads_evidence_last_without_promotion(self):
        binding = dict(self.spec["bindings"]["source_registry_implementation"])
        result, output = self.run_main([cli.REAL_MODE])
        self.assertEqual(result, 0)
        self.assertEqual(output["status"], "PASS_AUTHENTICATED_AYA_PREREQUISITE_TEST_ONLY")
        self.assertTrue(output["original_test_body_executed"])
        self.assertTrue(output["prerequisite_evidence_verified"])
        self.assertTrue(output["historical_source_copy_authenticated"])
        self.assertTrue(output["input_files_preserved"])
        self.assertEqual(output["snapshot_backed_pytest_nodes_passed"], 1)
        self.assertEqual(output["bound_input_opens"], 11)
        self.assertEqual(output["bound_input_bytes_read"], 1_472_969)
        self.assertEqual(output["snapshot_accounting"], self.summary)
        self.assertEqual(output["pytest_lifecycle"], self.phases)
        self.assertEqual(output["attribution_records_validated"], 1478)
        self.assertEqual(output["attribution_records_parsed"], 1478)
        self.assertEqual(output["materialization_supervised_tokens_verified"], 577453)
        self.assertEqual(output["owner_receipt_accepted_records_verified"], 32)
        self.assertEqual(self.inputs.read.call_args_list, self.expected_reads())
        self.assertIn(call(HISTORICAL_SOURCE, binding), self.inputs.read.call_args_list)
        self.assertNotIn(LOGICAL_SOURCE, [item.args[0] for item in self.inputs.read.call_args_list])
        self.assertEqual(self.spec["bindings"]["source_registry_implementation"], binding)
        self.assertEqual(
            self.core.build_test_module.call_args.args[1]["source_registry_implementation"], binding
        )
        self.inputs.verify.assert_called_once_with()
        self.inputs.close.assert_called_once_with()
        self.run_projected_test.assert_called_once_with(self.io, self.pytest, self.module)
        self.core.build_test_module.assert_called_once()
        self.assertIs(self.module.pytest, self.pytest)
        self.assertEqual(
            (self.io.MAX_FILES, self.io.MAX_BYTES, self.io.MAX_SECONDS), (11, 1_472_969, 30)
        )
        self.assertEqual((self.helper.MAX_FILES, self.helper.MAX_BYTES), (4, 45_471))
        for key in (
            "ordinary_checkout_test_passed", "original_module_fully_imported",
            "owner_decisions_reproduced", "new_owner_approval_created",
            "human_authorship_independently_verified", "source_registry_code_executed",
            "current_source_replaced", "candidate_promotion_performed", "source_registry_changed",
            "full_private_integration_passed", "execution_authorized", "production_eligible",
            "tokenizer_loaded", "network_used", "gpu_used", "inference_called",
        ):
            self.assertIs(output[key], False, key)
        for key in ("upstream_source_rows_read", "candidate_rows_read", "input_files_written"):
            self.assertEqual(output[key], 0, key)

    def test_each_read_failure_stops_once_without_current_source_fallback_or_later_evidence(self):
        for position, rejected in enumerate(cli.READ_ROLES):
            with self.subTest(rejected=rejected):
                self.configure_main()
                rejected_path = self.expected_reads()[position].args[0]
                original_read = self.inputs.read.side_effect

                def reject(path, binding, rejected_path=rejected_path, original_read=original_read):
                    if path == rejected_path:
                        raise self.io.Stop("input_hash_mismatch")
                    return original_read(path, binding)

                self.inputs.read.side_effect = reject
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, "input_hash_mismatch")
                self.assertEqual(
                    self.inputs.read.call_args_list, self.expected_reads()[:position + 1]
                )
                self.assertNotIn(
                    LOGICAL_SOURCE, [item.args[0] for item in self.inputs.read.call_args_list]
                )
                self.assertEqual(output["candidate_rows_read"], 0)
                self.assertIsNone(output["attribution_records_parsed"])
                self.core.build_test_module.assert_not_called()
                self.run_projected_test.assert_not_called()
                self.inputs.close.assert_called_once_with()

    def test_read_and_snapshot_accounting_mismatches_cannot_pass(self):
        cases = (
            ("opens", -1, "read_budget"), ("opens", 1, "read_budget"),
            ("bytes_read", -1, "read_budget"), ("bytes_read", 1, "read_budget"),
            ("virtual_opens", 11, "reproduction_rejected"),
            ("virtual_opens", 13, "reproduction_rejected"),
            ("logical_read_bytes_debited", 2_842_450, "reproduction_rejected"),
            ("logical_read_bytes_debited", 2_842_452, "reproduction_rejected"),
            ("bound_blob_count", 8, "reproduction_rejected"),
            ("bound_blob_count", 10, "reproduction_rejected"),
            ("filesystem_access_performed", True, "reproduction_rejected"),
        )
        for field, value, code in cases:
            with self.subTest(field=field, value=value):
                self.configure_main()
                if code == "read_budget":
                    setattr(self.inputs, field, value)
                else:
                    self.summary[field] = value
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, code)
                self.inputs.close.assert_called_once_with()

    def test_preservation_boundary_and_reproduction_failures_stop_without_retry(self):
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

    def test_close_alarm_and_post_cleanup_deadline_failures_prevent_pass(self):
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
                    "close_exception": "input_cleanup_failed", "close_stop": "input_cleanup_failed",
                    "alarm": "deadline_cleanup_failed", "deadline": "deadline_exceeded",
                }[failure]
                self.assert_stopped(result, output, code)
                self.inputs.close.assert_called_once_with()
                self.assertEqual(self.alarm.call_args_list, [call(30), call(0)])

    def test_opaque_errors_are_not_retained_or_retried(self):
        for stage in ("spec", "runtime", "read", "projection", "pytest", "summary"):
            with self.subTest(stage=stage):
                self.configure_main()
                target = {
                    "spec": self.read_spec, "runtime": self.warm_runtime,
                    "read": self.inputs.read, "projection": self.core.build_test_module,
                    "pytest": self.run_projected_test, "summary": self.snapshot.summary,
                }[stage]
                target.side_effect = RuntimeError("SYNTHETIC_SECRET")
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, "audit_failed")
                target.assert_called_once()
                if stage not in {"spec", "runtime"}:
                    self.inputs.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
