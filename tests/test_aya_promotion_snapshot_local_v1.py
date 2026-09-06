"""Synthetic combined CLI guards; never read private inputs or existing artifact bodies."""

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
    "aya_promotion_cli_tests", ROOT / "scripts/check_aya_promotion_snapshot_local_v1.py"
)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)
HISTORICAL_SOURCE = "tests/fixtures/aya_source_registry_9f20652.py.txt"
LOGICAL_SOURCE = "src/lumi/posttraining_sources.py"
HISTORICAL_SHA256 = "9f20652e85d0aede7b6ea75799f6e9746c5f35487b28e4bf39bdf854c80330e8"


class AyaPromotionLocalTests(unittest.TestCase):
    def setUp(self):
        # Only authenticated source-control helpers are loaded, never earlier mains.
        self.helper = cli.helpers()
        self.shared = self.helper.lifecycle_helpers()
        self.io = self.shared.io_primitives()
        _suffix, relative, digest = cli.CONTROL_SOURCES[2]
        source = cli.control_bytes(ROOT / relative, digest, 65_536)
        self.lifecycle = types.ModuleType("_synthetic_promotion_lifecycle")
        exec(compile(source, str(ROOT / relative), "exec"), self.lifecycle.__dict__)
        self.configure_main()

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
            "virtual_opens": 35,
            "logical_read_bytes_debited": 4_413_108,
            "bound_blob_count": 11,
            "filesystem_access_performed": False,
        }
        self.snapshot = Mock(summary=Mock(return_value=self.summary))
        self.module = self.lifecycle.synthetic_module(str(ROOT))
        self.pytest, self.core = Mock(), Mock()
        self.core.build_test_module.return_value = (self.module, self.snapshot)
        self.phases = {
            "collected": 2,
            "nodeids": list(cli.NODEIDS),
            "phases": {
                nodeid: {"setup": "passed", "call": "passed", "teardown": "passed"}
                for nodeid in cli.NODEIDS
            },
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
            inputs_factory = stack.enter_context(
                patch.object(self.io, "BoundInputs", return_value=self.inputs)
            )
            network_denial = stack.enter_context(patch.object(self.shared, "network_denial"))
            addaudithook = stack.enter_context(patch.object(cli.sys, "addaudithook"))
            result = cli.main(arguments)
            for unused in unused_helpers:
                unused.assert_not_called()
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
        for key in (
            "original_test_bodies_executed", "existing_successor_matched",
            "existing_promotion_receipt_matched", "historical_registry_validators_executed",
            "candidate_promotion_performed", "source_registry_changed",
        ):
            self.assertIs(output[key], False, key)
        self.assertEqual(output["snapshot_backed_pytest_nodes_passed"], 0)
        for key in (
            "snapshot_accounting", "existing_receipt_claim_candidate_promotions",
            "attribution_records_validated",
        ):
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

    def test_exact_ordered_targets_limits_and_historical_mapping(self):
        self.assertEqual(cli.TEST_NAMES, (
            "test_successor_changes_only_the_existing_aya_entry",
            "test_promotion_receipt_is_reproducible_raw_free_and_nonproduction",
        ))
        self.assertEqual(cli.NODEIDS, tuple(
            "tests/test_aya_source_promotion.py::" + name for name in cli.TEST_NAMES
        ))
        self.assertEqual(self.spec["nodeids"], list(cli.NODEIDS))
        self.assertEqual((cli.MAX_FILES, cli.MAX_BYTES, cli.MAX_SECONDS), (14, 1_524_967, 30))
        self.assertEqual((cli.ATTRIBUTION_BYTES, cli.EVIDENCE_BYTES), (1_387_842, 1_397_494))
        self.assertEqual((cli.SNAPSHOT_OPENS, cli.SNAPSHOT_BYTES), (35, 4_413_108))
        self.assertEqual(cli.READ_ROLES, (
            "plan", "data_contract", "predecessor_registry", "promotion_implementation",
            "promotion_runner", "source_registry_implementation", "hash_helper_source",
            "test_source", "policy_source", "successor_registry", "materialization_report",
            "owner_review_receipt", "promotion_receipt", "attribution_manifest",
        ))
        self.assertEqual(set(self.spec["bindings"]), set(cli.READ_ROLES))
        self.assertEqual(sum(b["bytes"] for b in self.spec["bindings"].values()), 1_524_967)
        self.assertEqual(
            sum(self.spec["bindings"][role]["bytes"] for role in cli.READ_ROLES[-4:]), 1_397_494
        )
        self.assertEqual(self.spec["bindings"]["source_registry_implementation"], {
            "path": LOGICAL_SOURCE, "bytes": 14927, "sha256": HISTORICAL_SHA256
        })
        self.assertEqual(cli.PHYSICAL_READ_PATHS, {
            "source_registry_implementation": HISTORICAL_SOURCE
        })
        self.assertEqual(self.spec["physical_read_paths"], cli.PHYSICAL_READ_PATHS)
        self.assertEqual(self.spec["limits"], {
            "maximum_bound_file_opens": 14, "maximum_bound_bytes": 1_524_967,
            "maximum_runtime_seconds": 30, "maximum_evidence_objects": 4,
            "maximum_evidence_bytes": 1_397_494, "attribution_records": 1478,
            "maximum_snapshot_opens": 35, "maximum_snapshot_read_bytes": 4_413_108,
            "selected_original_tests": 2, "upstream_source_rows": 0, "candidate_rows": 0,
            "artifact_write_bytes": 0, "network_bytes": 0, "inference_calls": 0,
        })

    def test_control_read_cap_and_hash_reject_untrusted_content(self):
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

    def test_fresh_helper_only_changes_authenticated_core_controls_not_old_test_identity(self):
        source = cli.control_bytes(cli.HELPER_PATH, cli.HELPER_SHA256, 65_536)
        original = types.ModuleType("_synthetic_original_receipt_helper")
        original.__file__ = str(cli.HELPER_PATH)
        exec(compile(source, str(cli.HELPER_PATH), "exec"), original.__dict__)
        existing = types.ModuleType("_aya_promotion_control_helpers")
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
        self.assertEqual(fresh.MODULE_NAME, "test_aya_bengali_owner_review")
        self.assertEqual(original.CORE_PACKAGE, "_lumi_aya_receipt_entry")
        self.assertEqual(self.shared.MODULE_NAME, original.MODULE_NAME)
        self.assertEqual(cli.control_bytes(cli.HELPER_PATH, cli.HELPER_SHA256, 65_536), source)

    def test_exact_four_control_helpers_authenticate_and_load_without_source_fallback(self):
        suffixes = (
            "posttraining_safety_replay_snapshot_v1", "posttraining_aya_plan_snapshot_v1",
            "posttraining_aya_promotion_lifecycle_v1", "posttraining_aya_promotion_snapshot_v1",
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
            loaded = sys.modules[cli.CORE_PACKAGE + "." + suffixes[2]]
            self.assertEqual(loaded.NODEIDS, cli.NODEIDS)
            self.assertTrue(callable(loaded.run_projected_tests))

    def test_all_four_controls_authenticate_before_any_execution_or_registration(self):
        events = []

        def authenticate(_path, _digest, maximum):
            self.assertEqual(maximum, 65_536)
            self.assertNotIn(cli.CORE_PACKAGE, sys.modules)
            events.append("read")
            return b"SYNTHETIC_MARKER = 1\n"

        def execute(_code, _namespace):
            self.assertEqual(events.count("read"), 4)
            events.append("exec")

        with (
            self.isolated_core_namespace(),
            patch.object(self.helper, "control_bytes", side_effect=authenticate),
            patch.object(self.helper, "exec", create=True, side_effect=execute),
        ):
            self.helper.authenticated_core(self.io)
        self.assertEqual(events, ["read"] * 4 + ["exec"] * 4)

    def test_core_prefix_collisions_are_preserved_before_control_reads(self):
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

    def test_warm_runtime_selects_exact_authenticated_lifecycle_namespace(self):
        key = cli.CORE_PACKAGE + ".posttraining_aya_promotion_lifecycle_v1"
        decoy = types.ModuleType("_decoy")
        with (
            patch.dict(sys.modules, {key: self.lifecycle, "_other." + key: decoy}),
            patch.object(
                self.helper, "warm_runtime", return_value=(self.pytest, self.core)
            ) as warm,
        ):
            self.assertEqual(
                cli.warm_runtime(self.helper, self.io), (self.pytest, self.core, self.lifecycle)
            )
            warm.assert_called_once_with(self.io)

    def test_warm_runtime_rejects_wrong_lifecycle_identity_or_target_order(self):
        key = cli.CORE_PACKAGE + ".posttraining_aya_promotion_lifecycle_v1"
        for field, value in (
            ("MODULE_NAME", "wrong"), ("TEST_NAMES", cli.TEST_NAMES[::-1]),
            ("TEST_NAMES", cli.TEST_NAMES[:1]), ("NODEIDS", cli.NODEIDS[::-1]),
            ("NODEIDS", ("wrong", "wrong")),
        ):
            with (
                self.subTest(field=field, value=value),
                patch.dict(sys.modules, {key: self.lifecycle}),
                patch.object(self.helper, "warm_runtime", return_value=(self.pytest, self.core)),
                patch.object(self.lifecycle, field, value),
            ):
                with self.assertRaises(self.io.Stop) as caught:
                    cli.warm_runtime(self.helper, self.io)
                self.assertEqual(caught.exception.code, "runtime_rejected")

    def test_failed_control_authentication_or_execution_stops_before_private_inputs(self):
        for failure in ("authenticate", "execute"):
            for position in range(4):
                with self.subTest(failure=failure, position=position):
                    self.configure_main()
                    payloads = [b"SYNTHETIC_MARKER = 1\n"] * 4
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

    def test_invalid_flags_or_arguments_stop_before_manifest_and_inputs(self):
        cases = (
            ([cli.REAL_MODE], {"flags": {"no_site": 0, "optimize": 0}}, "runtime_rejected"),
            ([cli.REAL_MODE], {"flags": {"no_site": 1, "optimize": 1}}, "runtime_rejected"),
            ([cli.REAL_MODE], {"bytecode": False}, "runtime_rejected"),
            ([], {}, "arguments_rejected"), (["wrong"], {}, "arguments_rejected"),
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

    def test_reversed_missing_duplicate_or_foreign_manifest_nodes_stop_before_inputs(self):
        for nodeids in (
            [], list(cli.NODEIDS[:1]), list(cli.NODEIDS[::-1]),
            [cli.NODEIDS[0]] * 2, [cli.NODEIDS[0], "wrong"], [*cli.NODEIDS, cli.NODEIDS[0]],
        ):
            with self.subTest(nodeids=nodeids):
                self.configure_main()
                self.spec["nodeids"] = nodeids
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
                self.inputs.read.assert_not_called()

    def test_invalid_manifest_schema_roles_or_byte_budgets_stop_before_reads(self):
        for mutation in ("schema", "roles", "total_bytes", "attribution", "evidence"):
            with self.subTest(mutation=mutation):
                self.configure_main()
                if mutation == "schema":
                    self.spec["schema_version"] = "wrong"
                elif mutation == "roles":
                    self.spec["bindings"]["wrong"] = self.spec["bindings"].pop("test_source")
                elif mutation == "total_bytes":
                    self.spec["bindings"]["test_source"]["bytes"] += 1
                else:
                    role = (
                        "attribution_manifest"
                        if mutation == "attribution" else "promotion_receipt"
                    )
                    self.spec["bindings"][role]["bytes"] += 1
                    self.spec["bindings"]["test_source"]["bytes"] -= 1
                result, output = self.run_main([cli.REAL_MODE])
                self.assert_stopped(result, output, "spec_rejected")
                self.warm_runtime.assert_not_called()
                self.inputs_factory.assert_not_called()
                self.inputs.read.assert_not_called()

    def test_runtime_probe_uses_two_synthetic_tests_and_zero_bound_inputs(self):
        result, output = self.run_main(["--check-runtime"])
        self.assertEqual(result, 0)
        self.assertEqual(output["status"], "PASS_RUNTIME_ONLY")
        for key in ("bound_input_opens", "bound_input_bytes_read", "candidate_rows_read"):
            self.assertEqual(output[key], 0, key)
        self.assertEqual(output["snapshot_backed_pytest_nodes_passed"], 0)
        self.assertEqual(output["pytest_lifecycle"], self.phases)
        self.assertFalse(output["original_test_bodies_executed"])
        self.assertFalse(output["existing_successor_matched"])
        self.assertFalse(output["existing_promotion_receipt_matched"])
        self.assertFalse(output["historical_registry_validators_executed"])
        self.assertNotIn("existing_receipt_claim_candidate_promotions", output)
        self.inputs_factory.assert_not_called()
        self.inputs.read.assert_not_called()
        self.inputs.verify.assert_not_called()
        self.inputs.close.assert_not_called()
        self.core.build_test_module.assert_not_called()
        self.synthetic.assert_called_once_with(str(ROOT))
        self.runner.assert_called_once()
        module = self.runner.call_args.args[2]
        self.assertEqual(
            {name for name in vars(module) if name.startswith("test_")}, set(cli.TEST_NAMES)
        )
        self.assertEqual(self.runner.call_args.args, (self.io, self.pytest, module, str(ROOT)))

    def test_success_reports_two_existing_comparisons_not_a_new_promotion(self):
        binding = dict(self.spec["bindings"]["source_registry_implementation"])
        result, output = self.run_main([cli.REAL_MODE])
        self.assertEqual(result, 0)
        self.assertEqual(output["status"], "PASS_AUTHENTICATED_AYA_PROMOTION_TESTS_ONLY")
        for key in (
            "original_test_bodies_executed", "existing_successor_matched",
            "existing_promotion_receipt_matched", "historical_registry_validators_executed",
            "input_files_preserved",
        ):
            self.assertIs(output[key], True, key)
        self.assertEqual(output["snapshot_backed_pytest_nodes_passed"], 2)
        self.assertEqual(output["bound_input_opens"], 14)
        self.assertEqual(output["bound_input_bytes_read"], 1_524_967)
        self.assertEqual(output["snapshot_accounting"], self.summary)
        self.assertEqual(output["pytest_lifecycle"], self.phases)
        self.assertEqual(output["predecessor_source_count_verified"], 10)
        self.assertEqual(output["successor_source_count_verified"], 10)
        self.assertEqual(output["existing_registry_changed_entries_verified"], 1)
        self.assertEqual(output["existing_receipt_claim_candidate_promotions"], 1)
        self.assertEqual(output["existing_receipt_claim_production_promotions"], 0)
        self.assertEqual(output["attribution_records_validated"], 1478)
        self.assertEqual(self.inputs.read.call_args_list, self.expected_reads())
        self.assertIn(call(HISTORICAL_SOURCE, binding), self.inputs.read.call_args_list)
        self.assertNotIn(LOGICAL_SOURCE, [item.args[0] for item in self.inputs.read.call_args_list])
        self.assertEqual(self.spec["bindings"]["source_registry_implementation"], binding)
        self.assertEqual(
            self.core.build_test_module.call_args.args[1]["source_registry_implementation"], binding
        )
        self.inputs.verify.assert_called_once_with()
        self.inputs.close.assert_called_once_with()
        self.runner.assert_called_once_with(self.io, self.pytest, self.module, str(ROOT))
        self.core.build_test_module.assert_called_once()
        self.synthetic.assert_not_called()
        self.assertIs(self.module.pytest, self.pytest)
        self.assertEqual(
            (self.io.MAX_FILES, self.io.MAX_BYTES, self.io.MAX_SECONDS), (14, 1_524_967, 30)
        )
        self.assertEqual((self.helper.MAX_FILES, self.helper.MAX_BYTES), (4, 45_471))
        for key in (
            "ordinary_checkout_tests_passed", "original_modules_fully_imported",
            "owner_decisions_reproduced", "new_owner_approval_created",
            "human_authorship_independently_verified", "current_source_replaced",
            "candidate_promotion_performed", "source_registry_changed",
            "full_private_integration_passed", "execution_authorized", "production_eligible",
            "tokenizer_loaded", "network_used", "gpu_used", "inference_called",
        ):
            self.assertIs(output[key], False, key)
        for key in ("upstream_source_rows_read", "candidate_rows_read", "input_files_written"):
            self.assertEqual(output[key], 0, key)

    def test_any_bound_read_failure_stops_once_without_later_evidence_or_source_fallback(self):
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
                self.core.build_test_module.assert_not_called()
                self.runner.assert_not_called()
                self.inputs.close.assert_called_once_with()

    def test_accounting_mismatches_cannot_publish_two_test_pass(self):
        for field, value, code in (
            ("opens", -1, "read_budget"), ("opens", 1, "read_budget"),
            ("bytes_read", -1, "read_budget"), ("bytes_read", 1, "read_budget"),
            ("virtual_opens", 34, "reproduction_rejected"),
            ("virtual_opens", 36, "reproduction_rejected"),
            ("logical_read_bytes_debited", 4_413_107, "reproduction_rejected"),
            ("logical_read_bytes_debited", 4_413_109, "reproduction_rejected"),
            ("bound_blob_count", 10, "reproduction_rejected"),
            ("bound_blob_count", 12, "reproduction_rejected"),
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

    def test_real_lifecycle_collision_is_preserved_and_never_invokes_pytest(self):
        self.runner.side_effect = self.lifecycle.run_projected_tests
        existing = types.ModuleType(cli.MODULE_NAME)
        with patch.dict(sys.modules, {cli.MODULE_NAME: existing}):
            result, output = self.run_main([cli.REAL_MODE])
            self.assert_stopped(result, output, "runtime_rejected")
            self.assertIs(sys.modules[cli.MODULE_NAME], existing)
        self.pytest.main.assert_not_called()
        self.runner.assert_called_once()
        self.inputs.close.assert_called_once_with()

    def test_incomplete_two_node_lifecycle_stops_and_restores_outer_namespace(self):
        for outcome in ("missing_second", "skipped_second", "reversed_collection"):
            with self.subTest(outcome=outcome):
                self.configure_main()
                self.runner.side_effect = self.lifecycle.run_projected_tests

                def run(_arguments, *, plugins, outcome=outcome):
                    plugin = plugins[0]
                    nodeids = cli.NODEIDS[::-1] if outcome == "reversed_collection" else cli.NODEIDS
                    plugin.pytest_collection_modifyitems([
                        types.SimpleNamespace(nodeid=nodeid) for nodeid in nodeids
                    ])
                    for nodeid in cli.NODEIDS:
                        if nodeid == cli.NODEIDS[1] and outcome == "missing_second":
                            break
                        for phase in ("setup", "call", "teardown"):
                            plugin.pytest_runtest_logreport(types.SimpleNamespace(
                                nodeid=nodeid, when=phase,
                                outcome="skipped" if nodeid == cli.NODEIDS[1] else "passed",
                                longrepr="SYNTHETIC_SECRET",
                            ))
                    return 0

                self.pytest.main.side_effect = run
                existing = types.ModuleType(cli.MODULE_NAME)
                with patch.dict(sys.modules, {cli.MODULE_NAME: existing}):
                    with patch.dict(sys.modules):
                        del sys.modules[cli.MODULE_NAME]
                        result, output = self.run_main([cli.REAL_MODE])
                        self.assert_stopped(result, output, "reproduction_rejected")
                        self.assertNotIn(cli.MODULE_NAME, sys.modules)
                    self.assertIs(sys.modules[cli.MODULE_NAME], existing)
                self.pytest.main.assert_called_once()
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

    def test_opaque_errors_do_not_leak_or_publish_partial_success_or_retry(self):
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
