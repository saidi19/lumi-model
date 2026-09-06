"""Synthetic CI routing guards; no genuine private inputs or test execution."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "source_partition_guards", ROOT / "scripts/ci_source_partition_v1.py",
)
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)
TEST = "tests/test_synthetic.py"
IMPLEMENTATION = "src/lumi/synthetic.py"
PRIVATE = "../lumi-artifacts/synthetic-private/candidate.jsonl"
NODE = TEST + "::test_private"
SOURCE = b"# Synthetic reviewed source, never executed.\n# Second witness line.\n"
BINDING = {"bytes": len(SOURCE), "sha256": hashlib.sha256(SOURCE).hexdigest()}


def fixtures():
    baseline = {
        "entries": [{"nodeid": NODE, "review_lane": "dependency_review_pending"}],
        "test_files": {TEST: copy.deepcopy(BINDING)},
    }
    manifest = {
        "schema_version": policy.SCHEMA,
        "baseline_inventory_sha256": policy.BASELINE_SHA256,
        "entries": [{
            "nodeid": NODE, "classification": "private_fixture", "status": "OPEN",
            "reason_code": "synthetic_private_dependency",
            "witness": {
                "test_path": TEST, "test_line": 1, "callee": "_load",
                "implementation_path": IMPLEMENTATION, "implementation_line": 2,
                "binding_key": "bindings.synthetic", "required_path": PRIVATE,
            },
        }],
        "bindings": {TEST: copy.deepcopy(BINDING), IMPLEMENTATION: copy.deepcopy(BINDING)},
    }
    return manifest, baseline


def arguments():
    return ["-p", "scripts.ci_source_partition_v1", "--lumi-source-suite", *[
        "--ignore=" + name for name in policy.IGNORES
    ], *["--deselect=" + name for name in policy.HISTORY]]


def fake_pytest():
    return types.SimpleNamespace(UsageError=RuntimeError)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.baseline = fixtures()

    def rejected(self, value=None):
        with self.assertRaises(policy.PartitionError):
            policy.validate_manifest(self.manifest if value is None else value, self.baseline)

    def test_exact_synthetic_manifest_accepts_reviewed_source_witness(self):
        self.assertIs(policy.validate_manifest(self.manifest, self.baseline), self.manifest)

    def test_direct_read_may_use_same_test_and_implementation_path(self):
        self.manifest["entries"][0]["witness"]["implementation_path"] = TEST
        del self.manifest["bindings"][IMPLEMENTATION]
        policy.validate_manifest(self.manifest, self.baseline)

    def test_optional_reviewed_plan_support_is_not_a_private_path(self):
        self.manifest["bindings"]["configs/synthetic-support.json"] = copy.deepcopy(BINDING)
        policy.validate_manifest(self.manifest, self.baseline)

    def test_private_required_path_cannot_be_a_read_binding(self):
        self.manifest["entries"][0]["witness"]["required_path"] = "configs/private.json"
        self.manifest["bindings"]["configs/private.json"] = copy.deepcopy(BINDING)
        self.rejected()

    def test_unknown_node_cannot_be_routed(self):
        self.manifest["entries"][0]["nodeid"] = TEST + "::test_new"
        self.rejected()

    def test_duplicate_and_empty_node_inventories_are_rejected(self):
        for entries in ([], self.manifest["entries"] * 2):
            changed = copy.deepcopy(self.manifest)
            changed["entries"] = entries
            with self.subTest(count=len(entries)):
                self.rejected(changed)

    def test_closed_skip_pass_statuses_are_rejected(self):
        for status in ("PASS", "SKIP", "XFAIL", "CLOSED", True):
            self.manifest["entries"][0]["status"] = status
            with self.subTest(status=status):
                self.rejected()

    def test_historical_subtype_cannot_hide_source_drift(self):
        self.baseline["entries"][0]["review_lane"] = "historical_source_and_private_integration"
        self.rejected()
        self.manifest["entries"][0]["classification"] = "historical_source"
        policy.validate_manifest(self.manifest, self.baseline)

    def test_normal_private_node_cannot_be_relabelled_historical(self):
        self.manifest["entries"][0]["classification"] = "historical_source"
        self.rejected()

    def test_source_test_hash_cannot_be_refreshed(self):
        self.manifest["bindings"][TEST]["sha256"] = "0" * 64
        self.rejected()

    def test_missing_witness_source_binding_rejects(self):
        del self.manifest["bindings"][IMPLEMENTATION]
        self.rejected()

    def test_explicit_reviewed_source_support_binding_is_allowed(self):
        self.manifest["bindings"]["src/lumi/synthetic_support.py"] = copy.deepcopy(BINDING)
        policy.validate_manifest(self.manifest, self.baseline)

    def test_non_source_and_nested_config_support_paths_reject(self):
        for name in ("private/synthetic.py", "configs/nested/synthetic.json", "data/private.jsonl"):
            changed = copy.deepcopy(self.manifest)
            changed["bindings"][name] = copy.deepcopy(BINDING)
            with self.subTest(name=name):
                self.rejected(changed)

    def test_witness_must_name_its_exact_test_file(self):
        self.manifest["entries"][0]["witness"]["test_path"] = "tests/test_other.py"
        self.rejected()

    def test_boolean_zero_and_negative_witness_lines_reject(self):
        for line in (True, 0, -1, 1.0):
            self.manifest["entries"][0]["witness"]["test_line"] = line
            with self.subTest(line=line):
                self.rejected()

    def test_unsafe_required_paths_reject_without_inspection(self):
        for name in ("/private/value", "../other/value", "../../value", "a/../value", "one"):
            self.manifest["entries"][0]["witness"]["required_path"] = name
            with self.subTest(name=name):
                self.rejected()

    def test_unknown_fields_and_wrong_baseline_binding_reject(self):
        for field, value in (("authorization", True), ("baseline_inventory_sha256", "0" * 64)):
            changed = copy.deepcopy(self.manifest)
            changed[field] = value
            with self.subTest(field=field):
                self.rejected(changed)

    def test_binding_schema_and_size_are_strict(self):
        for binding in ({}, {"bytes": True, "sha256": "0" * 64},
                        {"bytes": 1, "sha256": "not-a-hash"}):
            changed = copy.deepcopy(self.manifest)
            changed["bindings"][IMPLEMENTATION] = binding
            with self.subTest(binding=binding):
                self.rejected(changed)

    def test_unrecognized_reason_or_source_labels_reject(self):
        for field, value in (("callee", "a()\nprivate"), ("binding_key", "")):
            self.manifest["entries"][0]["witness"][field] = value
            with self.subTest(field=field):
                self.rejected()


class FileChecks(unittest.TestCase):
    def synthetic_check(self, *, changed_source=False, extra_line=False, wrong_pin=False):
        manifest, baseline = fixtures()
        baseline_bytes = json.dumps(baseline, sort_keys=True).encode()
        digest = hashlib.sha256(baseline_bytes).hexdigest()
        manifest["baseline_inventory_sha256"] = digest
        if extra_line:
            manifest["entries"][0]["witness"]["test_line"] = 3
        manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
        payloads = {policy.BASELINE: baseline_bytes, policy.MANIFEST: manifest_bytes,
                    TEST: SOURCE, IMPLEMENTATION: b"changed" if changed_source else SOURCE}
        reads = []

        def read(_root, name, limit=2_097_152):
            reads.append(name)
            return payloads[name]

        with patch.object(policy, "BASELINE_SHA256", digest), patch.object(
            policy, "SOURCE_PARTITION_MANIFEST_SHA256",
            "0" * 64 if wrong_pin else hashlib.sha256(manifest_bytes).hexdigest(),
        ), patch.object(policy, "read_bound_file", side_effect=read):
            result = policy.check(Path("/synthetic"))
        self.assertNotIn(PRIVATE, reads)
        self.assertEqual(set(reads), set(payloads))
        return result

    def test_bound_check_never_reads_or_stats_private_required_path(self):
        with patch.object(Path, "stat", side_effect=AssertionError("no stat")):
            self.synthetic_check()

    def test_changed_source_rejects(self):
        with self.assertRaisesRegex(policy.PartitionError, "bound_file_changed"):
            self.synthetic_check(changed_source=True)

    def test_manifest_hash_tamper_rejects_before_source_reads(self):
        with self.assertRaisesRegex(policy.PartitionError, "manifest_hash"):
            self.synthetic_check(wrong_pin=True)

    def test_out_of_range_review_witness_rejects(self):
        with self.assertRaisesRegex(policy.PartitionError, "witness_line"):
            self.synthetic_check(extra_line=True)

    def test_malformed_and_duplicate_json_reject(self):
        for payload in (b"not JSON", b"[]", b'{"a":1,"a":2}', b"\xff"):
            with self.subTest(payload=payload), self.assertRaises(policy.PartitionError):
                policy.decode(payload)

    def test_file_reader_rejects_missing_and_symlink_leaf_or_parent(self):
        with tempfile.TemporaryDirectory(prefix="lumi-partition-synthetic-") as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            for name in (TEST, "src/lumi/synthetic.py"):
                with self.subTest(name=name), self.assertRaises(policy.PartitionError):
                    policy.read_bound_file(root, name)
            (root / TEST).symlink_to(root / "missing")
            with self.assertRaises(policy.PartitionError):
                policy.read_bound_file(root, TEST)
            (root / "src").symlink_to(root / "tests")
            with self.assertRaises(policy.PartitionError):
                policy.read_bound_file(root, "src/synthetic.py")

    def test_file_reader_rejects_directory_hardlink_and_path_escape(self):
        with tempfile.TemporaryDirectory(prefix="lumi-partition-shape-") as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            (root / TEST).mkdir()
            with self.assertRaises(policy.PartitionError):
                policy.read_bound_file(root, TEST)
            (root / "tests/first.py").write_bytes(SOURCE)
            os.link(root / "tests/first.py", root / "tests/second.py")
            with self.assertRaises(policy.PartitionError):
                policy.read_bound_file(root, "tests/second.py")
            with self.assertRaises(policy.PartitionError):
                policy.read_bound_file(root, "tests/../../private.json")


class ScopeTests(unittest.TestCase):
    def validate(self, args=None, environment=None, roots=None, ini=None):
        policy.validate_scope(arguments() if args is None else args,
                              {} if environment is None else environment,
                              ["tests"] if roots is None else roots,
                              {"addopts": ["-q"], "testpaths": ["tests"]} if ini is None else ini)

    def test_exact_full_scope_and_optional_tests_and_quiet_are_valid(self):
        self.validate()
        self.validate(arguments() + ["tests", "-q"])

    def test_filter_subset_and_shortcut_flags_reject(self):
        for extra in (["-k", "easy"], ["-m", "not slow"], ["--lf"], ["--collect-only"],
                      ["--ignore-glob=*private*"], ["--maxfail=1"], ["tests/test_other.py"],
                      ["--deselect=tests/test_other.py::test_other"]):
            with self.subTest(extra=extra), self.assertRaises(policy.PartitionError):
                self.validate(arguments() + extra)

    def test_missing_or_broader_routes_reject(self):
        for index in (-1, -3):
            changed = arguments()
            changed.pop(index)
            with self.subTest(index=index), self.assertRaises(policy.PartitionError):
                self.validate(changed)

    def test_env_ini_and_resolved_subset_bypasses_reject(self):
        for kwargs in ({"environment": {"PYTEST_ADDOPTS": "-k easy"}},
                       {"environment": {"PYTEST_PLUGINS": "unreviewed_plugin"}},
                       {"roots": ["tests/test_other.py"]},
                       {"ini": {"addopts": ["-q", "-k", "easy"], "testpaths": ["tests"]}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(policy.PartitionError):
                self.validate(**kwargs)

    def test_assertions_must_be_enabled(self):
        policy.require_assertions(0)
        for optimize in (1, 2, False, 0.0):
            with self.subTest(optimize=optimize), self.assertRaises(policy.PartitionError):
                policy.require_assertions(optimize)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.manifest, _ = fixtures()
        self.new_node = "tests/test_new.py::test_new_failure_stays_real"
        self.nodes = [NODE, *policy.HISTORY, self.new_node]

    def test_only_exact_reviewed_node_routes_and_new_test_runs(self):
        source, private = policy.partition_nodes(self.nodes, self.manifest)
        self.assertEqual(source, {self.new_node})
        self.assertEqual(private, {NODE})
        self.assertEqual(len(source) + len(private) + len(policy.HISTORY), len(self.nodes))

    def test_missing_private_or_history_node_and_duplicate_collection_reject(self):
        for nodes in (self.nodes[1:], self.nodes[:1] + self.nodes[2:], self.nodes + [NODE]):
            with self.subTest(nodes=nodes), self.assertRaises(policy.PartitionError):
                policy.partition_nodes(nodes, self.manifest)

    def test_no_opt_in_has_no_manifest_reads_or_item_changes(self):
        config = types.SimpleNamespace(getoption=lambda _name: False)
        items = [types.SimpleNamespace(nodeid=node, config=config) for node in self.nodes]
        original = list(items)
        with patch.object(policy, "check", side_effect=AssertionError("inert")):
            policy.pytest_configure(config)
            policy.pytest_collection_modifyitems(config, items)
        self.assertEqual(items, original)

    def test_hook_routes_no_skip_marker_and_final_accounting_is_exact(self):
        config = types.SimpleNamespace(hook=types.SimpleNamespace(pytest_deselected=Mock()))
        state = {"manifest": self.manifest, "collected": list(self.nodes), "validated": False}
        setattr(config, policy.STATE, state)
        items = [types.SimpleNamespace(nodeid=node) for node in self.nodes]
        with patch.dict("sys.modules", {"pytest": fake_pytest()}):
            policy.pytest_collection_modifyitems(config, items)
            self.assertEqual({item.nodeid for item in items}, {self.new_node, *policy.HISTORY})
            self.assertEqual(config.hook.pytest_deselected.call_count, 1)
            # Model pytest's original exact two-node deselection hook.
            items[:] = [item for item in items if item.nodeid not in policy.HISTORY]
            session = types.SimpleNamespace(config=config, items=items, testsfailed=0)
            policy.pytest_collection_finish(session)
        self.assertTrue(state["validated"])

    def test_hidden_source_deselection_or_collection_error_is_blocking(self):
        for failed, items in ((0, []), (1, [types.SimpleNamespace(nodeid=self.new_node)])):
            config = types.SimpleNamespace()
            setattr(config, policy.STATE, {"source": {self.new_node}, "validated": False})
            session = types.SimpleNamespace(config=config, items=items, testsfailed=failed)
            with patch.dict("sys.modules", {"pytest": fake_pytest()}):
                with self.subTest(failed=failed), self.assertRaises(RuntimeError):
                    policy.pytest_collection_finish(session)

    def test_private_readiness_stays_blocking_not_passed(self):
        with patch.object(policy, "check", return_value=self.manifest):
            report = policy.private_readiness(ROOT)
        self.assertEqual(report["status"], "BLOCKED_PRIVATE_INTEGRATION_UNAVAILABLE")
        self.assertEqual(report["open_baseline_nodes"], 327)
        self.assertEqual(report["reviewed_private_nodes"], 1)
        self.assertEqual(report["private_tests_executed"], 0)
        self.assertIs(report["full_ci_passed"], False)

    def test_readiness_cli_always_exits_nonzero_on_valid_or_invalid_inputs(self):
        for broken in (False, True):
            output = io.StringIO()
            with patch.object(policy.sys, "argv", ["checker", "--private-readiness"]), patch.object(
                policy, "check", side_effect=policy.PartitionError("blocked") if broken else None,
                return_value=self.manifest,
            ), contextlib.redirect_stdout(output):
                result = policy.main()
            self.assertEqual(result, 1)
            report = json.loads(output.getvalue())
            self.assertIs(report["full_ci_passed"], False)
            self.assertIs(report["private_integration_passed"], False)


if __name__ == "__main__":
    unittest.main()
