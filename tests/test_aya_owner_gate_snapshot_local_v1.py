"""Synthetic owner-gate CLI guards; no private inputs, artifact bodies or real CLI runs."""

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
    "aya_owner_gate_cli_tests", ROOT / "scripts/check_aya_owner_gate_snapshot_local_v1.py"
)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)
HISTORICAL_SOURCE = "tests/fixtures/aya_source_registry_9f20652.py.txt"
LOGICAL_SOURCE = "src/lumi/posttraining_sources.py"
HISTORICAL_SHA256 = "9f20652e85d0aede7b6ea75799f6e9746c5f35487b28e4bf39bdf854c80330e8"
PHASES = ("setup", "call", "teardown")
EVENTS = tuple((cli.NODEID, phase) for phase in PHASES)
OLD_NAMES = (
    "test_successor_changes_only_the_existing_aya_entry",
    "test_promotion_receipt_is_reproducible_raw_free_and_nonproduction",
)
SUCCESS_CLAIMS = (
    "original_test_body_executed", "tampered_owner_seal_rejected",
    "owner_receipt_copy_changed_in_memory", "materialization_prerequisite_checked",
    "historical_source_copy_authenticated", "input_files_preserved",
)
NONCLAIMS = (
    "ordinary_checkout_test_passed", "original_module_fully_imported",
    "owner_decisions_reproduced", "new_owner_approval_created",
    "human_authorship_independently_verified", "prerequisite_evidence_verified",
    "attribution_provenance_revalidated", "owner_policy_rejection_independently_verified",
    "source_registry_code_executed", "current_source_replaced", "candidate_promotion_performed",
    "source_registry_changed", "full_private_integration_passed", "execution_authorized",
    "production_eligible",
)


class AyaOwnerGateLocalTests(unittest.TestCase):
    def setUp(self):
        # Authenticate public helper source only; never invoke an earlier checker main.
        self.helper = cli.helpers()
        self.shared = self.helper.lifecycle_helpers()
        self.io = self.shared.io_primitives()
        self.lifecycle = self.fresh_lifecycle()
        key = cli.CORE_PACKAGE + ".posttraining_aya_promotion_lifecycle_v1"
        with (
            patch.dict(sys.modules, {key: self.lifecycle}),
            patch.object(self.helper, "warm_runtime", return_value=(Mock(), Mock())),
        ):
            cli.warm_runtime(self.helper, self.io)
        self.configure_main()

    def fresh_lifecycle(self):
        _suffix, relative, digest = cli.CONTROL_SOURCES[3]
        source = cli.control_bytes(ROOT / relative, digest, 65_536)
        module = types.ModuleType("_synthetic_owner_gate_lifecycle")
        exec(compile(source, str(ROOT / relative), "exec"), module.__dict__)
        return module

    def configure_main(self):
        self.spec = cli.read_spec()
        self.inputs = Mock(opens=0, bytes_read=0)

        def synthetic_read(_path, binding):
            self.inputs.opens += 1
            self.inputs.bytes_read += binding["bytes"]
            return b"synthetic payload; no bound artifact opened"

        self.inputs.read.side_effect = synthetic_read
        self.boundary = Mock(blocked=0, roots=())
        self.summary = {
            "virtual_opens": 12, "logical_read_bytes_debited": 2_842_451,
            "bound_blob_count": 9, "filesystem_access_performed": False,
        }
        self.snapshot = Mock(summary=Mock(return_value=self.summary))
        self.module = self.lifecycle.synthetic_module(str(ROOT))
        self.pytest, self.core = Mock(), Mock()
        self.core.build_test_module.return_value = (self.module, self.snapshot)
        self.phases = {
            "collected": 1, "nodeids": [cli.NODEID],
            "phases": {cli.NODEID: dict.fromkeys(PHASES, "passed")},
        }
        self.read_spec = Mock(return_value=self.spec)
        self.warm_runtime = Mock(return_value=(self.pytest, self.core, self.lifecycle))
        self.runner = Mock(return_value=self.phases)
        self.synthetic = Mock(wraps=self.lifecycle.synthetic_module)
        self.alarm = Mock()

    def run_main(self, arguments, *, flags=None, bytecode=True, times=(0, 1, 2)):
        capture = io.StringIO()
        with contextlib.ExitStack() as stack:
            for replacement in (
                patch.object(cli, "helpers", return_value=self.helper),
                patch.object(self.helper, "lifecycle_helpers", return_value=self.shared),
                patch.object(self.shared, "io_primitives", return_value=self.io),
                patch.object(cli, "read_spec", self.read_spec),
                patch.object(self.io, "Boundary", return_value=self.boundary),
                patch.object(cli, "warm_runtime", self.warm_runtime),
                patch.object(self.lifecycle, "run_projected_tests", self.runner),
                patch.object(self.lifecycle, "synthetic_module", self.synthetic),
                patch.object(cli.sys, "flags", types.SimpleNamespace(**(flags or {
                    "no_site": 1, "optimize": 0
                }))),
                patch.object(cli.sys, "dont_write_bytecode", bytecode),
                patch.object(cli.signal, "signal"),
                patch.object(cli.signal, "alarm", self.alarm),
                patch.object(cli.time, "monotonic", side_effect=times),
                contextlib.redirect_stdout(capture),
            ):
                stack.enter_context(replacement)
            unused_helpers = [
                stack.enter_context(patch.object(target, name))
                for target, name in (
                    (self.helper, "main"), (self.shared, "main"),
                    (self.shared, "run_projected_test"), (self.shared, "synthetic_module"),
                )
            ]
            self.inputs_factory = stack.enter_context(
                patch.object(self.io, "BoundInputs", return_value=self.inputs)
            )
            self.network_denial = stack.enter_context(patch.object(self.shared, "network_denial"))
            self.addaudithook = stack.enter_context(patch.object(cli.sys, "addaudithook"))
            result = cli.main(arguments)
            for unused in unused_helpers:
                unused.assert_not_called()
        self.raw_output = capture.getvalue()
        self.assertEqual(self.alarm.call_args_list[-1], call(0))
        return result, json.loads(self.raw_output)

    def assert_stopped(self, result, output, code):
        self.assertEqual((result, output["status"], output["failure_code"]), (1, "STOP", code))
        for key in SUCCESS_CLAIMS[:-1] + NONCLAIMS:
            self.assertIs(output[key], False, key)
        self.assertEqual(output["snapshot_backed_pytest_nodes_passed"], 0)
        for key in ("snapshot_accounting", "attribution_records_validated", "pytest_lifecycle"):
            self.assertNotIn(key, output)
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

    def collect(self, plugin):
        plugin.pytest_collection_modifyitems([types.SimpleNamespace(nodeid=cli.NODEID)])

    def report(self, plugin, event, **changes):
        values = {"nodeid": event[0], "when": event[1], "outcome": "passed",
                  "longrepr": "SYNTHETIC_SECRET", "capstdout": "SYNTHETIC_SECRET"}
        values.update(changes)
        plugin.pytest_runtest_logreport(types.SimpleNamespace(**values))

    def complete(self, plugin):
        self.collect(plugin)
        for event in EVENTS:
            self.report(plugin, event)

    def test_exact_single_target_limits_manifest_and_historical_mapping(self):
        self.assertEqual(cli.TEST_NAME, "test_rejected_or_tampered_owner_gate_cannot_be_promoted")
        self.assertEqual(cli.NODEID, "tests/test_aya_source_promotion.py::" + cli.TEST_NAME)
        self.assertEqual(self.spec["nodeid"], cli.NODEID)
        self.assertEqual((cli.MAX_FILES, cli.MAX_BYTES, cli.MAX_SECONDS), (11, 1_472_969, 30))
        self.assertEqual((cli.ATTRIBUTION_BYTES, cli.EVIDENCE_BYTES), (1_387_842, 1_393_247))
        self.assertEqual((cli.SNAPSHOT_OPENS, cli.SNAPSHOT_BYTES), (12, 2_842_451))
        self.assertEqual(cli.READ_ROLES, (
            "plan", "data_contract", "predecessor_registry", "promotion_implementation",
            "promotion_runner", "source_registry_implementation", "hash_helper_source",
            "test_source", "materialization_report", "owner_review_receipt",
            "attribution_manifest",
        ))
        self.assertEqual(set(self.spec["bindings"]), set(cli.READ_ROLES))
        self.assertEqual(sum(b["bytes"] for b in self.spec["bindings"].values()), 1_472_969)
        self.assertEqual(self.spec["bindings"]["source_registry_implementation"], {
            "path": LOGICAL_SOURCE, "bytes": 14927, "sha256": HISTORICAL_SHA256,
        })
        self.assertEqual(cli.PHYSICAL_READ_PATHS, {
            "source_registry_implementation": HISTORICAL_SOURCE
        })
        self.assertEqual(self.spec["physical_read_paths"], cli.PHYSICAL_READ_PATHS)
        self.assertEqual(self.spec["limits"], {
            "maximum_bound_file_opens": 11, "maximum_bound_bytes": 1_472_969,
            "maximum_runtime_seconds": 30, "maximum_evidence_objects": 3,
            "maximum_evidence_bytes": 1_393_247, "attribution_records": 1478,
            "maximum_snapshot_opens": 12, "maximum_snapshot_read_bytes": 2_842_451,
            "selected_original_tests": 1, "upstream_source_rows": 0, "candidate_rows": 0,
            "artifact_write_bytes": 0, "network_bytes": 0, "inference_calls": 0,
        })

    def test_control_cap_and_hash_reject_before_content_use(self):
        for payload in (b"too long", b"bad"):
            with self.subTest(payload=payload):
                stream = io.BytesIO(payload)
                path = Mock(open=Mock(return_value=stream))
                with patch.object(stream, "read", wraps=stream.read) as read:
                    with self.assertRaisesRegex(ValueError, "^control_binding_rejected$"):
                        cli.control_bytes(path, "0" * 64, 3)
                    read.assert_called_once_with(4)
                path.open.assert_called_once_with("rb")

    def test_fresh_helper_does_not_change_prior_identity_namespace_or_source(self):
        source = cli.control_bytes(cli.HELPER_PATH, cli.HELPER_SHA256, 65_536)
        original = types.ModuleType("_synthetic_original_receipt_helper")
        original.__file__ = str(cli.HELPER_PATH)
        exec(compile(source, str(cli.HELPER_PATH), "exec"), original.__dict__)
        existing = types.ModuleType("_aya_owner_gate_control_helpers")
        with patch.dict(sys.modules, {existing.__name__: existing}):
            fresh = cli.helpers()
            self.assertIs(sys.modules[existing.__name__], existing)
        self.assertIsNot(fresh, self.helper)
        self.assertEqual((fresh.CORE_PACKAGE, fresh.CONTROL_SOURCES), (
            cli.CORE_PACKAGE, cli.CONTROL_SOURCES
        ))
        self.assertEqual((fresh.MODULE_NAME, fresh.TEST_NAME, fresh.NODEID), (
            original.MODULE_NAME, original.TEST_NAME, original.NODEID
        ))
        self.assertEqual(
            original.TEST_NAME, "test_actual_owner_receipt_is_complete_and_nonpromoting"
        )
        self.assertEqual(
            (original.CORE_PACKAGE, original.MAX_FILES), ("_lumi_aya_receipt_entry", 4)
        )
        self.assertEqual(self.shared.MODULE_NAME, original.MODULE_NAME)
        self.assertEqual(cli.control_bytes(cli.HELPER_PATH, cli.HELPER_SHA256, 65_536), source)

    def test_all_five_authenticated_controls_have_exact_order_and_no_source_fallback(self):
        suffixes = (
            "posttraining_safety_replay_snapshot_v1", "posttraining_aya_plan_snapshot_v1",
            "posttraining_aya_prerequisite_snapshot_v1", "posttraining_aya_promotion_lifecycle_v1",
            "posttraining_aya_owner_gate_snapshot_v1",
        )
        self.assertEqual(tuple(item[0] for item in cli.CONTROL_SOURCES), suffixes)
        self.assertEqual(tuple(item[1] for item in cli.CONTROL_SOURCES), tuple(
            "src/lumi/" + suffix + ".py" for suffix in suffixes
        ))
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

    def test_all_five_controls_authenticate_before_execution_or_namespace_registration(self):
        events = []

        def authenticate(_path, _digest, maximum):
            self.assertEqual(maximum, 65_536)
            self.assertNotIn(cli.CORE_PACKAGE, sys.modules)
            events.append("read")
            return b"SYNTHETIC_MARKER = 1\n"

        def execute(_code, _namespace):
            self.assertEqual(events.count("read"), 5)
            events.append("exec")

        with (
            self.isolated_core_namespace(),
            patch.object(self.helper, "control_bytes", side_effect=authenticate),
            patch.object(self.helper, "exec", create=True, side_effect=execute),
        ):
            self.helper.authenticated_core(self.io)
        self.assertEqual(events, ["read"] * 5 + ["exec"] * 5)

    def test_core_prefix_collisions_are_preserved_before_any_control_read(self):
        for suffix in ("", ".posttraining_aya_promotion_lifecycle_v1", ".foreign.descendant"):
            with self.subTest(suffix=suffix), self.isolated_core_namespace():
                name = cli.CORE_PACKAGE + suffix
                existing = types.ModuleType(name)
                sys.modules[name] = existing
                with patch.object(self.helper, "control_bytes") as reads:
                    with self.assertRaises(self.io.Stop):
                        self.helper.authenticated_core(self.io)
                    reads.assert_not_called()
                    self.assertIs(sys.modules[name], existing)

    def test_fresh_authenticated_lifecycle_only_gets_singleton_identity_and_three_events(self):
        original = self.fresh_lifecycle()
        original_state = (original.TEST_NAMES, original.NODEIDS, original._EXPECTED_EVENTS)
        key = cli.CORE_PACKAGE + ".posttraining_aya_promotion_lifecycle_v1"
        old_key = "_lumi_aya_promotion_entry.posttraining_aya_promotion_lifecycle_v1"
        with (
            self.isolated_core_namespace(),
            patch.dict(sys.modules, {old_key: original}),
            patch.object(self.helper, "warm_runtime", side_effect=lambda _io: (
                self.pytest, self.helper.authenticated_core(self.io)
            )) as warm,
        ):
            pytest, _core, configured = cli.warm_runtime(self.helper, self.io)
            self.assertIs(pytest, self.pytest)
            self.assertIs(configured, sys.modules[key])
            self.assertIsNot(configured, original)
            self.assertEqual(
                (configured.TEST_NAMES, configured.NODEIDS, configured._EXPECTED_EVENTS),
                ((cli.TEST_NAME,), (cli.NODEID,), EVENTS),
            )
            self.assertEqual((original.TEST_NAMES, original.NODEIDS, original._EXPECTED_EVENTS),
                             original_state)
            self.assertIs(sys.modules[old_key], original)
            warm.assert_called_once_with(self.io)
        self.assertEqual(self.fresh_lifecycle().TEST_NAMES, OLD_NAMES)

    def test_warm_runtime_rejects_nonoriginal_identity_phases_or_event_order(self):
        key = cli.CORE_PACKAGE + ".posttraining_aya_promotion_lifecycle_v1"
        for field, value in (
            ("MODULE_NAME", "wrong"), ("TEST_NAMES", (cli.TEST_NAME,)),
            ("TEST_NAMES", OLD_NAMES[::-1]), ("NODEIDS", (cli.NODEID,)),
            ("_PHASES", PHASES[::-1]), ("_EXPECTED_EVENTS", EVENTS),
            ("_EXPECTED_EVENTS", ()),
        ):
            lifecycle = self.fresh_lifecycle()
            with (
                self.subTest(field=field, value=value),
                patch.dict(sys.modules, {key: lifecycle}),
                patch.object(self.helper, "warm_runtime", return_value=(self.pytest, self.core)),
                patch.object(lifecycle, field, value),
            ):
                with self.assertRaises(self.io.Stop) as caught:
                    cli.warm_runtime(self.helper, self.io)
                self.assertEqual(caught.exception.code, "runtime_rejected")

    def test_each_failed_control_authentication_or_execution_cleans_up_before_inputs(self):
        for failure in ("authenticate", "execute"):
            for position in range(5):
                with self.subTest(failure=failure, position=position):
                    self.configure_main()
                    payloads = [b"SYNTHETIC_MARKER = 1\n"] * 5
                    payloads[position] = (
                        ValueError("SYNTHETIC_SECRET") if failure == "authenticate"
                        else b"raise RuntimeError('SYNTHETIC_SECRET')\n"
                    )
                    self.warm_runtime.side_effect = lambda helper, io: (
                        self.pytest, helper.authenticated_core(io), self.lifecycle
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
                    self.warm_runtime.assert_called_once_with(self.helper, self.io)

    def test_singleton_collection_rejects_prior_tests_empty_duplicate_or_foreign_nodes(self):
        old_nodes = tuple("tests/test_aya_source_promotion.py::" + name for name in OLD_NAMES)
        for nodeids in ((), old_nodes, old_nodes[:1], (cli.NODEID,) * 2, ("wrong",), (None,)):
            with self.subTest(nodeids=nodeids):
                plugin = self.lifecycle.TwoNodes(self.io)
                with self.assertRaises(self.io.Stop):
                    plugin.pytest_collection_modifyitems([
                        types.SimpleNamespace(nodeid=nodeid) for nodeid in nodeids
                    ])
                self.assertFalse(plugin.passed())
                with self.assertRaises(self.io.Stop):
                    self.collect(plugin)

    def test_all_three_event_positions_reject_bad_reports_stickily(self):
        for position, event in enumerate(EVENTS):
            for change in (
                {"outcome": "failed"}, {"outcome": "skipped"}, {"outcome": "rerun"},
                {"wasxfail": False}, {"wasxfail": "SYNTHETIC_SECRET"},
                {"nodeid": "wrong"}, {"when": EVENTS[(position + 1) % 3][1]},
            ):
                with self.subTest(position=position, change=change):
                    plugin = self.lifecycle.TwoNodes(self.io)
                    self.collect(plugin)
                    for previous in EVENTS[:position]:
                        self.report(plugin, previous)
                    with self.assertRaises(self.io.Stop):
                        self.report(plugin, event, **change)
                    self.assertFalse(plugin.passed())
                    self.assertNotIn("SYNTHETIC_SECRET", str(vars(plugin)))
                    with self.assertRaises(self.io.Stop):
                        self.report(plugin, event)
                    with self.assertRaises(self.io.Stop):
                        plugin.summary()
            plugin = self.lifecycle.TwoNodes(self.io)
            self.collect(plugin)
            for previous in EVENTS[:position + 1]:
                self.report(plugin, previous)
            with self.assertRaises(self.io.Stop):
                self.report(plugin, event)
            self.assertFalse(plugin.passed())

    def test_reports_before_collection_repeated_collection_and_missing_reports_cannot_pass(self):
        plugin = self.lifecycle.TwoNodes(self.io)
        with self.assertRaises(self.io.Stop):
            self.report(plugin, EVENTS[0])
        with self.assertRaises(self.io.Stop):
            self.collect(plugin)
        for length in range(4):
            with self.subTest(length=length):
                plugin = self.lifecycle.TwoNodes(self.io)
                self.collect(plugin)
                for event in EVENTS[:length]:
                    self.report(plugin, event)
                self.assertEqual(plugin.passed(), length == 3)
                if length == 3:
                    self.assertEqual(plugin.summary(), self.phases)
                else:
                    with self.assertRaises(self.io.Stop):
                        plugin.summary()
                with self.assertRaises(self.io.Stop):
                    self.collect(plugin)
                self.assertFalse(plugin.passed())

    def test_one_pytest_invocation_has_only_exact_owner_gate_node_and_restrictive_flags(self):
        self.assertEqual({name for name in vars(self.module) if name.startswith("test_")},
                         {cli.TEST_NAME})

        def synthetic_pytest(arguments, *, plugins):
            self.assertEqual(arguments, [
                "-c", "/dev/null", "--rootdir", str(ROOT), "--noconftest", "--assert=plain",
                "-p", "no:cacheprovider", "-p", "no:logging", "-p", "no:faulthandler",
                "-s", "-q", "-x", "-o", "addopts=", str(cli.TEST_PATH) + "::" + cli.TEST_NAME,
            ])
            self.assertEqual(len(plugins), 1)
            self.assertIs(sys.modules[cli.MODULE_NAME], self.module)
            getattr(self.module, cli.TEST_NAME)()
            self.complete(plugins[0])
            return 0

        with patch.dict(sys.modules):
            sys.modules.pop(cli.MODULE_NAME, None)
            self.pytest.main.side_effect = synthetic_pytest
            self.assertEqual(self.lifecycle.run_projected_tests(
                self.io, self.pytest, self.module, str(ROOT)
            ), self.phases)
            self.assertNotIn(cli.MODULE_NAME, sys.modules)
        self.pytest.main.assert_called_once()

    def test_module_collisions_preserve_existing_names_without_pytest(self):
        for suffix in ("", ".foreign", ".nested.descendant"):
            with self.subTest(suffix=suffix), patch.dict(sys.modules):
                sys.modules.pop(cli.MODULE_NAME, None)
                name = cli.MODULE_NAME + suffix
                existing = types.ModuleType(name)
                sys.modules[name] = existing
                with self.assertRaises(self.io.Stop):
                    self.lifecycle.run_projected_tests(self.io, self.pytest, self.module, str(ROOT))
                self.assertIs(sys.modules[name], existing)
                self.pytest.main.assert_not_called()

    def test_failed_pytest_always_cleans_only_owned_projection_and_never_retries(self):
        for failure in ("missing", "nonzero", "exception", "caught_rejection", "replacement"):
            with self.subTest(failure=failure), patch.dict(sys.modules):
                sys.modules.pop(cli.MODULE_NAME, None)
                replacement = types.ModuleType(cli.MODULE_NAME)
                pytest = Mock()

                def run(_arguments, *, plugins, failure=failure, replacement=replacement):
                    if failure == "exception":
                        raise RuntimeError("SYNTHETIC_SECRET")
                    if failure != "missing":
                        self.complete(plugins[0])
                    if failure == "caught_rejection":
                        with self.assertRaises(self.io.Stop):
                            self.report(plugins[0], EVENTS[-1])
                    if failure == "replacement":
                        sys.modules[cli.MODULE_NAME] = replacement
                    return 1 if failure == "nonzero" else 0

                pytest.main.side_effect = run
                with self.assertRaises((self.io.Stop, RuntimeError)):
                    self.lifecycle.run_projected_tests(self.io, pytest, self.module, str(ROOT))
                if failure == "replacement":
                    self.assertIs(sys.modules[cli.MODULE_NAME], replacement)
                else:
                    self.assertNotIn(cli.MODULE_NAME, sys.modules)
                pytest.main.assert_called_once()

    def test_invalid_arguments_flags_or_manifest_stop_before_runtime_and_inputs(self):
        for arguments, options, code in (
            ([cli.REAL_MODE], {"flags": {"no_site": 0, "optimize": 0}}, "runtime_rejected"),
            ([cli.REAL_MODE], {"flags": {"no_site": 1, "optimize": 1}}, "runtime_rejected"),
            ([cli.REAL_MODE], {"bytecode": False}, "runtime_rejected"),
            ([], {}, "arguments_rejected"), (["wrong"], {}, "arguments_rejected"),
            ([cli.REAL_MODE, "extra"], {}, "arguments_rejected"),
            (["--check-runtime", cli.REAL_MODE], {}, "arguments_rejected"),
        ):
            with self.subTest(arguments=arguments, options=options):
                self.configure_main()
                result, output = self.run_main(arguments, **options)
                self.assert_stopped(result, output, code)
                self.read_spec.assert_not_called()
                self.warm_runtime.assert_not_called()
                self.network_denial.assert_not_called()
                self.inputs_factory.assert_not_called()
        for mutation in ("schema", "nodeid", "roles", "total_bytes", "attribution", "evidence"):
            with self.subTest(mutation=mutation):
                self.configure_main()
                if mutation == "schema":
                    self.spec["schema_version"] = "wrong"
                elif mutation == "nodeid":
                    self.spec["nodeid"] = "tests/test_aya_source_promotion.py::" + OLD_NAMES[0]
                elif mutation == "roles":
                    self.spec["bindings"]["wrong"] = self.spec["bindings"].pop("test_source")
                elif mutation == "total_bytes":
                    self.spec["bindings"]["test_source"]["bytes"] += 1
                else:
                    role = (
                        "attribution_manifest" if mutation == "attribution"
                        else "owner_review_receipt"
                    )
                    self.spec["bindings"][role]["bytes"] += 1
                    self.spec["bindings"]["test_source"]["bytes"] -= 1
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, "spec_rejected")
                self.warm_runtime.assert_not_called()
                self.inputs_factory.assert_not_called()
                self.inputs.read.assert_not_called()

    def test_unknown_or_modified_physical_mapping_cannot_fall_back_to_current_source(self):
        for mapping in (
            {}, {"unknown": HISTORICAL_SOURCE},
            {"source_registry_implementation": LOGICAL_SOURCE},
            {"source_registry_implementation": "build/lib/lumi/posttraining_sources.py"},
            {"source_registry_implementation": HISTORICAL_SOURCE, "plan": "unbound.json"},
        ):
            with self.subTest(mapping=mapping):
                self.configure_main()
                self.spec["physical_read_paths"] = mapping
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, "spec_rejected")
                self.warm_runtime.assert_not_called()
                self.inputs_factory.assert_not_called()

    def test_runtime_probe_uses_one_synthetic_test_and_no_bound_inputs_or_owner_claim(self):
        result, output = self.run_main(["--check-runtime"])
        self.assertEqual((result, output["status"]), (0, "PASS_RUNTIME_ONLY"))
        for key in SUCCESS_CLAIMS + NONCLAIMS:
            self.assertIs(output[key], False, key)
        for key in ("bound_input_opens", "bound_input_bytes_read", "candidate_rows_read",
                    "snapshot_backed_pytest_nodes_passed"):
            self.assertEqual(output[key], 0, key)
        self.assertNotIn("attribution_records_validated", output)
        self.assertNotIn("snapshot_accounting", output)
        self.inputs_factory.assert_not_called()
        for operation in (self.inputs.read, self.inputs.verify, self.inputs.close,
                          self.core.build_test_module):
            operation.assert_not_called()
        self.synthetic.assert_called_once_with(str(ROOT))
        self.runner.assert_called_once()
        module = self.runner.call_args.args[2]
        self.assertEqual(
            {name for name in vars(module) if name.startswith("test_")}, {cli.TEST_NAME}
        )
        self.assertEqual(self.runner.call_args.args, (self.io, self.pytest, module, str(ROOT)))
        self.assertEqual(output["pytest_lifecycle"], self.phases)

    def test_success_claims_only_tampered_seal_rejection_not_full_owner_policy_or_provenance(self):
        binding = dict(self.spec["bindings"]["source_registry_implementation"])
        result, output = self.run_main([cli.REAL_MODE])
        self.assertEqual((result, output["status"]), (
            0, "PASS_AUTHENTICATED_AYA_OWNER_GATE_TEST_ONLY"
        ))
        for key in SUCCESS_CLAIMS:
            self.assertIs(output[key], True, key)
        for key in NONCLAIMS + ("tokenizer_loaded", "network_used", "gpu_used", "inference_called"):
            self.assertIs(output[key], False, key)
        for key in ("attribution_records_validated", "attribution_records_parsed",
                    "owner_receipt_accepted_records_verified", "existing_successor_matched",
                    "existing_promotion_receipt_matched"):
            self.assertNotIn(key, output)
        self.assertEqual(output["snapshot_backed_pytest_nodes_passed"], 1)
        self.assertEqual((output["bound_input_opens"], output["bound_input_bytes_read"]),
                         (11, 1_472_969))
        self.assertEqual(output["snapshot_accounting"], self.summary)
        self.assertEqual(output["pytest_lifecycle"], self.phases)
        self.assertEqual(self.inputs.read.call_args_list, self.expected_reads())
        self.assertIn(call(HISTORICAL_SOURCE, binding), self.inputs.read.call_args_list)
        self.assertNotIn(LOGICAL_SOURCE, [item.args[0] for item in self.inputs.read.call_args_list])
        self.assertEqual(self.spec["bindings"]["source_registry_implementation"], binding)
        self.assertEqual(self.core.build_test_module.call_args.args[1]["source_registry_implementation"],
                         binding)
        self.core.build_test_module.assert_called_once()
        self.inputs.verify.assert_called_once_with()
        self.inputs.close.assert_called_once_with()
        self.runner.assert_called_once_with(self.io, self.pytest, self.module, str(ROOT))
        self.synthetic.assert_not_called()
        self.assertIs(self.module.pytest, self.pytest)
        self.assertEqual((self.io.MAX_FILES, self.io.MAX_BYTES, self.io.MAX_SECONDS),
                         (11, 1_472_969, 30))
        self.assertEqual((self.helper.MAX_FILES, self.helper.MAX_BYTES), (4, 45_471))
        for key in ("upstream_source_rows_read", "candidate_rows_read", "input_files_written"):
            self.assertEqual(output[key], 0, key)

    def test_each_read_failure_stops_without_later_evidence_retry_or_source_fallback(self):
        for position, role in enumerate(cli.READ_ROLES):
            with self.subTest(role=role):
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
                self.core.build_test_module.assert_not_called()
                self.runner.assert_not_called()
                self.inputs.close.assert_called_once_with()

    def test_any_physical_or_snapshot_accounting_mismatch_prevents_success(self):
        for field, value, code in (
            ("opens", -1, "read_budget"), ("opens", 1, "read_budget"),
            ("bytes_read", -1, "read_budget"), ("bytes_read", 1, "read_budget"),
            ("virtual_opens", 11, "reproduction_rejected"),
            ("virtual_opens", 13, "reproduction_rejected"),
            ("logical_read_bytes_debited", 2_842_450, "reproduction_rejected"),
            ("logical_read_bytes_debited", 2_842_452, "reproduction_rejected"),
            ("bound_blob_count", 8, "reproduction_rejected"),
            ("bound_blob_count", 10, "reproduction_rejected"),
            ("filesystem_access_performed", True, "reproduction_rejected"),
        ):
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
                    self.runner.side_effect = self.io.Stop(code)
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, code)
                self.runner.assert_called_once_with(self.io, self.pytest, self.module, str(ROOT))
                self.inputs.close.assert_called_once_with()

    def test_cleanup_alarm_and_deadline_failures_cannot_publish_partial_pass(self):
        for failure in ("close_exception", "close_stop", "alarm", "deadline", "early_deadline"):
            with self.subTest(failure=failure):
                self.configure_main()
                if failure == "close_exception":
                    self.inputs.close.side_effect = OSError("SYNTHETIC_SECRET")
                elif failure == "close_stop":
                    self.inputs.close.side_effect = self.io.Stop("input_changed")
                elif failure == "alarm":
                    self.alarm.side_effect = [None, OSError("SYNTHETIC_SECRET")]
                times = (0, 30, 30) if failure == "early_deadline" else (
                    0, 1, 30 if failure == "deadline" else 2
                )
                result, output = self.run_main([cli.REAL_MODE], times=times)
                code = {
                    "close_exception": "input_cleanup_failed", "close_stop": "input_cleanup_failed",
                    "alarm": "deadline_cleanup_failed", "deadline": "deadline_exceeded",
                    "early_deadline": "deadline_exceeded",
                }[failure]
                self.assert_stopped(result, output, code)
                self.inputs.close.assert_called_once_with()
                self.assertEqual(self.alarm.call_args_list, [call(30), call(0)])

    def test_opaque_failures_never_leak_exception_content_or_retry(self):
        for stage in ("spec", "runtime", "read", "projection", "pytest", "summary"):
            with self.subTest(stage=stage):
                self.configure_main()
                target = {
                    "spec": self.read_spec, "runtime": self.warm_runtime,
                    "read": self.inputs.read, "projection": self.core.build_test_module,
                    "pytest": self.runner, "summary": self.snapshot.summary,
                }[stage]
                target.side_effect = RuntimeError("SYNTHETIC_SECRET")
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, "audit_failed")
                target.assert_called_once()
                if stage not in {"spec", "runtime"}:
                    self.inputs.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
