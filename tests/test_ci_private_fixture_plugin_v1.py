"""Synthetic metadata-only tests; no private fixtures or model imports."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ci_private_fixture_plugin_v1.py"
SPEC = importlib.util.spec_from_file_location("ci_private_fixture_plugin", SCRIPT)
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)

ROOT = Path("/synthetic/repository")
SOURCE = b"# synthetic source fixture; not executed\n"
NODES = sorted(policy.REVIEWED_NODES)
FIXTURE = "evidence/synthetic-private-report.json"
SECOND = "configs/synthetic-private-plan.json"


def manifest():
    return {
        "schema_version": "lumi.ci-private-fixture-requirements.v0.1",
        "entries": [
            {
                "nodeid": nodeid,
                "test_file_sha256": hashlib.sha256(SOURCE).hexdigest(),
                "static_reason": "Direct unconditional fixture read verified in synthetic test.",
                "fixtures": [{"path": FIXTURE, "kind": "file"}],
            }
            for nodeid in NODES
        ],
    }


def metadata(mode=stat.S_IFREG | 0o600, **changes):
    fields = {
        "st_mode": mode,
        "st_nlink": 1,
        "st_size": len(SOURCE),
        "st_dev": 1,
        "st_ino": 2,
        "st_mtime_ns": 3,
        "st_ctime_ns": 4,
    }
    fields.update(changes)
    return types.SimpleNamespace(**fields)


class ReviewedManifestTests(unittest.TestCase):
    def test_real_reviewed_manifest_passes_hash_and_schema_without_fixture_reads(self):
        with (
            mock.patch.object(policy, "fixture_present") as presence,
            mock.patch.object(policy, "tracked_fixtures") as inventory,
        ):
            value = policy.load_manifest(policy.ROOT)
        self.assertEqual({entry["nodeid"] for entry in value["entries"]}, set(NODES))
        presence.assert_not_called()
        inventory.assert_not_called()


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.value = manifest()

    def fails(self, code):
        with self.assertRaisesRegex(policy.FixturePolicyError, "^" + code + "$"):
            policy.validate_manifest(self.value)

    def test_exact_four_nodes_validate(self):
        self.assertEqual(policy.validate_manifest(self.value), self.value)

    def test_unknown_manifest_key_rejected(self):
        self.value["skip_all"] = True
        self.fails("manifest_schema")

    def test_wrong_schema_rejected(self):
        self.value["schema_version"] = "other"
        self.fails("manifest_schema")

    def test_extra_or_missing_node_rejected(self):
        for length in (0, 3, 5):
            with self.subTest(length=length):
                self.value = manifest()
                self.value["entries"] = (self.value["entries"] * 2)[:length]
                self.fails("manifest_inventory")

    def test_duplicate_node_rejected(self):
        self.value["entries"][1] = copy.deepcopy(self.value["entries"][0])
        self.fails("manifest_duplicate_nodeid")

    def test_unknown_or_pattern_node_rejected(self):
        for nodeid in ("tests/test_new.py::test_missing", "tests/*", NODES[0] + "[other]", True):
            with self.subTest(nodeid=nodeid):
                self.value["entries"][0]["nodeid"] = nodeid
                self.fails("manifest_nodeid")

    def test_test_hash_requires_lowercase_sha256(self):
        for digest in ("a" * 63, "A" * 64, True):
            with self.subTest(digest=digest):
                self.value["entries"][0]["test_file_sha256"] = digest
                self.fails("manifest_test_hash")

    def test_same_test_file_cannot_have_conflicting_hashes(self):
        self.value["entries"][-1]["test_file_sha256"] = "a" * 64
        self.fails("manifest_test_hash_conflict")

    def test_static_reason_is_required_and_single_line(self):
        for reason in ("", "too short", "x" * 401, "Direct read\nwith hidden line", 1):
            with self.subTest(reason=reason):
                self.value["entries"][0]["static_reason"] = reason
                self.fails("manifest_static_reason")

    def test_fixture_path_must_be_strict_relative_allowlisted(self):
        for name in (
            "../evidence/private.json",
            "/evidence/private.json",
            "evidence/../private.json",
            "evidence//private.json",
            "evidence/./private.json",
            "evidence/private.json/",
            "evidence/private.json\n",
            "evidence\\private.json",
            "evidence/*",
            "data/private.json",
            "evidence",
            "evidence/private\x00.json",
            True,
        ):
            with self.subTest(name=name):
                self.value["entries"][0]["fixtures"][0]["path"] = name
                self.fails("manifest_fixture_path")

    def test_fixture_inventory_and_kind_are_strict(self):
        self.value["entries"][0]["fixtures"] = []
        self.fails("manifest_fixtures")
        self.value = manifest()
        self.value["entries"][0]["fixtures"][0]["kind"] = "directory"
        self.fails("manifest_fixture_kind")

    def test_duplicate_fixture_rejected(self):
        self.value["entries"][0]["fixtures"] *= 2
        self.fails("manifest_duplicate_fixture")

    def test_unknown_entry_and_fixture_fields_rejected(self):
        self.value["entries"][0]["exception_class"] = "ValueError"
        self.fails("manifest_entry")
        self.value = manifest()
        self.value["entries"][0]["fixtures"][0]["ignore_bad_contents"] = True
        self.fails("manifest_fixture")

    def test_manifest_hash_is_checked_before_json_decode(self):
        with mock.patch.object(policy, "read_reviewed_file", return_value=b"not JSON"):
            with self.assertRaisesRegex(policy.FixturePolicyError, "manifest_hash_mismatch"):
                policy.load_manifest(ROOT)

    def test_pinned_synthetic_manifest_loads(self):
        data = json.dumps(self.value).encode()
        with (
            mock.patch.object(policy, "read_reviewed_file", return_value=data) as reader,
            mock.patch.object(policy, "MANIFEST_SHA256", hashlib.sha256(data).hexdigest()),
        ):
            self.assertEqual(policy.load_manifest(ROOT), self.value)
        reader.assert_called_once_with(ROOT, policy.MANIFEST, 16_384)

    def test_invalid_json_is_a_fixed_policy_error(self):
        data = b"not JSON"
        with (
            mock.patch.object(policy, "read_reviewed_file", return_value=data),
            mock.patch.object(policy, "MANIFEST_SHA256", hashlib.sha256(data).hexdigest()),
            self.assertRaisesRegex(policy.FixturePolicyError, "manifest_json"),
        ):
            policy.load_manifest(ROOT)


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.value = manifest()
        self.reader = self.enterContext(
            mock.patch.object(policy, "read_reviewed_file", return_value=SOURCE)
        )
        self.inventory = self.enterContext(
            mock.patch.object(policy, "tracked_fixtures", return_value=set())
        )
        self.presence = self.enterContext(
            mock.patch.object(policy, "fixture_present", return_value=False)
        )

    def assess(self, nodeids=None):
        return policy.assess_requirements(ROOT, self.value, NODES if nodeids is None else nodeids)

    def test_absent_untracked_fixtures_affect_only_the_four_exact_nodes(self):
        result = self.assess([*NODES, "tests/test_new.py::test_new_failure"])
        self.assertEqual(result, dict.fromkeys(NODES, (FIXTURE,)))
        self.assertEqual(self.reader.call_count, 2)
        self.presence.assert_called_once_with(ROOT, FIXTURE)

    def test_present_fixtures_mean_no_skips(self):
        self.presence.return_value = True
        self.assertEqual(self.assess(), {})

    def test_present_tracked_fixture_runs_normally(self):
        self.inventory.return_value = {FIXTURE}
        self.presence.return_value = True
        self.assertEqual(self.assess(), {})

    def test_missing_tracked_fixture_fails(self):
        self.inventory.return_value = {FIXTURE}
        with self.assertRaisesRegex(policy.FixturePolicyError, "tracked_fixture_missing"):
            self.assess()

    def test_stale_missing_or_duplicate_collected_nodes_fail_before_reads(self):
        for nodeids in (NODES[:-1], [*NODES, NODES[0]], []):
            with self.subTest(nodeids=nodeids):
                with self.assertRaisesRegex(
                    policy.FixturePolicyError, "manifest_node_not_collected_once"
                ):
                    self.assess(nodeids)
        self.reader.assert_not_called()

    def test_changed_test_source_fails_before_fixture_checks(self):
        self.reader.return_value = SOURCE + b"# changed\n"
        with self.assertRaisesRegex(policy.FixturePolicyError, "test_source_hash_mismatch"):
            self.assess()
        self.inventory.assert_not_called()
        self.presence.assert_not_called()

    def test_unexpected_git_inventory_rejected(self):
        self.inventory.return_value = {"evidence/unreviewed.json"}
        with self.assertRaisesRegex(policy.FixturePolicyError, "git_index_output"):
            self.assess()

    def test_metadata_failure_propagates_never_becomes_skip(self):
        self.presence.side_effect = policy.FixturePolicyError("fixture_symlink")
        with self.assertRaisesRegex(policy.FixturePolicyError, "fixture_symlink"):
            self.assess()

    def test_partial_presence_lists_only_missing_declared_fixture(self):
        self.value["entries"][0]["fixtures"].append({"path": SECOND, "kind": "file"})
        self.presence.side_effect = lambda root, name: name == FIXTURE
        self.assertEqual(self.assess(), {NODES[0]: (SECOND,)})

    def test_bad_metadata_result_does_not_count_as_absence(self):
        self.presence.return_value = None
        with self.assertRaisesRegex(policy.FixturePolicyError, "fixture_metadata_result"):
            self.assess()


class FilesystemTests(unittest.TestCase):
    def setUp(self):
        self.opener = self.enterContext(mock.patch.object(policy.os, "open", return_value=10))
        self.closer = self.enterContext(mock.patch.object(policy.os, "close"))
        self.reader = self.enterContext(mock.patch.object(policy.os, "read"))
        self.statter = self.enterContext(mock.patch.object(policy.os, "stat"))
        self.fstatter = self.enterContext(mock.patch.object(policy.os, "fstat"))

    def test_fixture_check_never_opens_leaf_or_reads_contents(self):
        self.statter.side_effect = [metadata(stat.S_IFDIR | 0o700), metadata()]
        self.assertTrue(policy.fixture_present(ROOT, FIXTURE))
        self.assertEqual([call.args[0] for call in self.opener.call_args_list], [ROOT, "evidence"])
        self.reader.assert_not_called()
        self.assertTrue(all(call.kwargs["follow_symlinks"] is False
                            for call in self.statter.call_args_list))

    def test_missing_leaf_is_absent_and_closes_descriptors(self):
        self.statter.side_effect = [metadata(stat.S_IFDIR | 0o700), FileNotFoundError()]
        self.assertFalse(policy.fixture_present(ROOT, FIXTURE))
        self.assertEqual(self.closer.call_count, 2)
        self.reader.assert_not_called()

    def test_missing_parent_is_absent(self):
        self.statter.side_effect = FileNotFoundError()
        self.assertFalse(policy.fixture_present(ROOT, FIXTURE))
        self.opener.assert_called_once()

    def test_symlink_leaf_or_parent_fails(self):
        for values in (
            [metadata(stat.S_IFLNK)],
            [metadata(stat.S_IFDIR), metadata(stat.S_IFLNK)],
        ):
            with self.subTest(values=values):
                self.statter.side_effect = values
                with self.assertRaisesRegex(policy.FixturePolicyError, "fixture_symlink"):
                    policy.fixture_present(ROOT, FIXTURE)

    def test_directory_fifo_and_socket_leaf_fail(self):
        for mode in (stat.S_IFDIR, stat.S_IFIFO, stat.S_IFSOCK):
            with self.subTest(mode=mode):
                self.statter.side_effect = [metadata(stat.S_IFDIR), metadata(mode)]
                with self.assertRaisesRegex(policy.FixturePolicyError, "fixture_wrong_type"):
                    policy.fixture_present(ROOT, FIXTURE)

    def test_hardlink_leaf_fails(self):
        self.statter.side_effect = [metadata(stat.S_IFDIR), metadata(st_nlink=2)]
        with self.assertRaisesRegex(policy.FixturePolicyError, "fixture_hardlink"):
            policy.fixture_present(ROOT, FIXTURE)

    def test_regular_file_parent_fails(self):
        self.statter.side_effect = [metadata()]
        with self.assertRaisesRegex(policy.FixturePolicyError, "fixture_parent_wrong_type"):
            policy.fixture_present(ROOT, FIXTURE)

    def test_permission_error_is_not_absence(self):
        self.statter.side_effect = PermissionError("synthetic secret must not escape")
        with self.assertRaisesRegex(policy.FixturePolicyError, "^fixture_metadata_unreadable$"):
            policy.fixture_present(ROOT, FIXTURE)

    def test_source_read_is_bounded_and_no_follow(self):
        self.fstatter.return_value = metadata()
        self.reader.side_effect = [SOURCE, b""]
        name = NODES[0].split("::", 1)[0]
        self.assertEqual(policy.read_reviewed_file(ROOT, name, 1024), SOURCE)
        self.assertTrue(all(call.args[1] & os.O_NOFOLLOW for call in self.opener.call_args_list))
        self.assertTrue(self.opener.call_args_list[-1].args[1] & os.O_NONBLOCK)
        self.assertLessEqual(sum(call.args[1] for call in self.reader.call_args_list), 2050)

    def test_source_reader_cannot_read_fixture_path(self):
        with self.assertRaisesRegex(policy.FixturePolicyError, "reviewed_path"):
            policy.read_reviewed_file(ROOT, FIXTURE, 1024)
        self.opener.assert_not_called()

    def test_oversized_source_fails_before_read(self):
        self.fstatter.return_value = metadata(st_size=1025)
        with self.assertRaisesRegex(policy.FixturePolicyError, "reviewed_file_shape"):
            policy.read_reviewed_file(ROOT, policy.MANIFEST, 1024)
        self.reader.assert_not_called()

    def test_growth_during_read_is_bounded_and_rejected(self):
        self.fstatter.return_value = metadata(st_size=1)
        self.reader.return_value = b"a" * 11
        with self.assertRaisesRegex(policy.FixturePolicyError, "reviewed_file_oversize"):
            policy.read_reviewed_file(ROOT, policy.MANIFEST, 10)
        self.reader.assert_called_once_with(10, 11)

    def test_changed_source_metadata_fails(self):
        self.fstatter.side_effect = [metadata(), metadata(st_ctime_ns=5)]
        self.reader.side_effect = [SOURCE, b""]
        with self.assertRaisesRegex(policy.FixturePolicyError, "reviewed_file_changed"):
            policy.read_reviewed_file(ROOT, policy.MANIFEST, 1024)


class GitTests(unittest.TestCase):
    def setUp(self):
        self.which = self.enterContext(
            mock.patch.object(policy.shutil, "which", return_value="/usr/bin/git")
        )
        self.run = self.enterContext(mock.patch.object(policy.subprocess, "run"))
        self.run.return_value = types.SimpleNamespace(returncode=0, stdout=b"")

    def test_git_checks_only_explicit_local_index_paths(self):
        self.assertEqual(policy.tracked_fixtures(ROOT, {FIXTURE}), set())
        args, kwargs = self.run.call_args
        self.assertEqual(
            args[0],
            ["/usr/bin/git", "--no-optional-locks", "-C", str(ROOT), "ls-files",
             "--cached", "--full-name", "-z", "--", FIXTURE],
        )
        self.assertEqual(kwargs["timeout"], 10)
        self.assertEqual(kwargs["env"]["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(kwargs["env"]["GIT_CONFIG_GLOBAL"], os.devnull)
        self.assertNotIn("AWS_ACCESS_KEY_ID", kwargs["env"])

    def test_tracked_filename_is_returned(self):
        self.run.return_value.stdout = FIXTURE.encode() + b"\0"
        self.assertEqual(policy.tracked_fixtures(ROOT, {FIXTURE}), {FIXTURE})

    def test_bad_duplicate_or_unexpected_git_output_fails(self):
        for output in (b"no-nul", b"other\0", (FIXTURE.encode() + b"\0") * 2, b"x" * 16_385):
            with self.subTest(output_bytes=len(output)):
                self.run.return_value.stdout = output
                with self.assertRaisesRegex(policy.FixturePolicyError, "git_index_output"):
                    policy.tracked_fixtures(ROOT, {FIXTURE})

    def test_git_failure_and_timeout_are_not_skips(self):
        self.run.return_value.returncode = 1
        with self.assertRaisesRegex(policy.FixturePolicyError, "git_index_failed"):
            policy.tracked_fixtures(ROOT, {FIXTURE})
        self.run.side_effect = subprocess.TimeoutExpired("git", 10)
        with self.assertRaisesRegex(policy.FixturePolicyError, "git_index_unavailable"):
            policy.tracked_fixtures(ROOT, {FIXTURE})

    def test_git_unavailable_fails_before_execution(self):
        self.which.return_value = None
        with self.assertRaisesRegex(policy.FixturePolicyError, "git_unavailable"):
            policy.tracked_fixtures(ROOT, {FIXTURE})
        self.run.assert_not_called()


class HookTests(unittest.TestCase):
    def setUp(self):
        self.config = types.SimpleNamespace(
            rootpath=policy.ROOT, getoption=mock.Mock(return_value=True)
        )
        self.items = [types.SimpleNamespace(nodeid=n, add_marker=mock.Mock()) for n in NODES]
        self.extra = types.SimpleNamespace(
            nodeid="tests/test_new.py::test_new", add_marker=mock.Mock()
        )
        self.items.append(self.extra)
        self.loader = self.enterContext(mock.patch.object(policy, "load_manifest"))
        self.assessor = self.enterContext(mock.patch.object(policy, "assess_requirements"))
        self.assessor.return_value = {NODES[0]: (FIXTURE,)}
        self.fake_pytest = types.ModuleType("pytest")
        self.fake_pytest.mark = types.SimpleNamespace(skip=mock.Mock(return_value="synthetic skip"))
        self.fake_pytest.UsageError = type("SyntheticUsageError", (Exception,), {})
        self.enterContext(mock.patch.dict(sys.modules, {"pytest": self.fake_pytest}))

    def test_no_flag_means_no_reads_no_state_and_no_markers(self):
        self.config.getoption.return_value = False
        policy.pytest_collection_modifyitems(self.config, self.items)
        self.loader.assert_not_called()
        self.assessor.assert_not_called()
        self.assertFalse(hasattr(self.config, policy.STATE))
        for item in self.items:
            item.add_marker.assert_not_called()

    def test_only_exact_assessed_node_is_marked(self):
        policy.pytest_collection_modifyitems(self.config, self.items)
        self.items[0].add_marker.assert_called_once_with("synthetic skip")
        for item in self.items[1:]:
            item.add_marker.assert_not_called()
        reason = self.fake_pytest.mark.skip.call_args.kwargs["reason"]
        self.assertIn("unavailable (not passed)", reason)

    def test_all_present_means_no_markers(self):
        self.assessor.return_value = {}
        policy.pytest_collection_modifyitems(self.config, self.items)
        for item in self.items:
            item.add_marker.assert_not_called()

    def test_policy_error_is_usage_error_without_partial_markers(self):
        self.assessor.side_effect = policy.FixturePolicyError("tracked_fixture_missing")
        with self.assertRaisesRegex(self.fake_pytest.UsageError, "tracked_fixture_missing"):
            policy.pytest_collection_modifyitems(self.config, self.items)
        for item in self.items:
            item.add_marker.assert_not_called()

    def test_wrong_repository_root_fails_before_reads(self):
        self.config.rootpath = ROOT
        with self.assertRaisesRegex(self.fake_pytest.UsageError, "repository_root_mismatch"):
            policy.pytest_collection_modifyitems(self.config, self.items)
        self.loader.assert_not_called()

    def test_summary_calls_unavailable_tests_not_passed(self):
        setattr(self.config, policy.STATE, {NODES[0]: (FIXTURE,)})
        reporter = mock.Mock()
        policy.pytest_terminal_summary(reporter, 1, self.config)
        self.assertIn("1 unavailable; not counted as passed", reporter.write_sep.call_args.args[1])
        lines = [call.args[0] for call in reporter.write_line.call_args_list]
        self.assertTrue(any(NODES[0] in line and FIXTURE in line for line in lines))
        self.assertIn("not a private integration-test pass", lines[-1])

    def test_no_flag_means_no_summary(self):
        self.config.getoption.return_value = False
        setattr(self.config, policy.STATE, {NODES[0]: (FIXTURE,)})
        reporter = mock.Mock()
        policy.pytest_terminal_summary(reporter, 0, self.config)
        self.assertEqual(reporter.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
