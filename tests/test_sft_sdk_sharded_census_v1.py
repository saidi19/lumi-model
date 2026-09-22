"""Synthetic sharded metadata tests; never access a real SDK or install native guards."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def definitions(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = definitions("scripts/diagnose_sft_sdk_sharded_census_v1.py", "_sdk_sharded_test_definitions")


def identity(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid


def local_os(**overrides):
    """Do not patch Python's shared os module."""
    return types.SimpleNamespace(**{**vars(os), **overrides})


def changed_info(info, **changes):
    return types.SimpleNamespace(**{
        **{key: getattr(info, key) for key in dir(info) if key.startswith("st_")}, **changes,
    })


class ShardFixtures(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory(prefix="sdk-shards-"))
        self.root = Path(self.directory) / "sdk"
        self.root.mkdir()
        self.first, self.second = self.root / "first", self.root / "second"
        self.first.mkdir()
        self.second.mkdir()
        (self.first / "a").write_bytes(b"synthetic")
        (self.second / "nested").mkdir()
        (self.second / "nested" / "b").write_bytes(b"synthetic")
        self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, self.fd)
        self.helper = types.SimpleNamespace(identity=identity)

    def census(self, name="first", result=None):
        result = m.empty_report() if result is None else result
        m.shard_census(self.fd, name, self.helper, result)
        return result

    def assert_budget(self, name, result, kind):
        with self.assertRaises(m.CensusError) as caught:
            self.census(name, result)
        self.assertEqual(caught.exception.code, "budget_hit")
        self.assertEqual(result["budget_kind"], kind)
        self.assertGreater(result["attempted_value"], result["budget_limit"])

    def test_roots_and_descendants_counted_with_no_names_retained(self):
        result = self.census()
        self.census("second", result)
        self.assertEqual(result["visited_nodes"], 5)
        self.assertEqual(result["active_shard_nodes"], 3)
        self.assertEqual(result["maximum_directory_depth"], 2)
        self.assertNotIn("first", m.canonical(result).decode())
        self.assertNotIn("second", m.canonical(result).decode())

    def test_exact_shard_cap_includes_root_and_allows_relisting(self):
        with patch.object(m, "SHARD_NODES", 2):
            result = self.census()
        self.assertEqual(result["visited_nodes"], 2)
        self.assertEqual(result["active_shard_nodes"], 2)

    def test_exact_nested_shard_cap_allows_both_relistings(self):
        with patch.object(m, "SHARD_NODES", 3):
            result = self.census("second")
        self.assertEqual(result["visited_nodes"], 3)

    def test_exact_aggregate_cap_completes(self):
        result = m.empty_report()
        with patch.object(m, "AGGREGATE_NODES", 5):
            self.census("first", result)
            self.census("second", result)
        self.assertEqual(result["visited_nodes"], 5)

    def test_shard_counters_reset_between_shards(self):
        result = self.census("second")
        prior_names = result["accepted_relative_name_bytes"]
        prior_blocks = result["accepted_allocated_bytes"]
        self.census("first", result)
        self.assertEqual(result["active_shard_nodes"], 2)
        self.assertEqual(result["active_shard_name_bytes"], len("first/") + len("first/a/"))
        self.assertEqual(result["accepted_relative_name_bytes"],
                         prior_names + result["active_shard_name_bytes"])
        self.assertEqual(result["accepted_allocated_bytes"],
                         prior_blocks + result["active_shard_allocated_bytes"])

    def test_no_payload_reads_and_only_directories_opened(self):
        calls = []

        def open_directory(path, flags, *args, **kwargs):
            self.assertTrue(flags & os.O_DIRECTORY)
            self.assertTrue(flags & os.O_NOFOLLOW)
            self.assertFalse(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            calls.append(path)
            return os.open(path, flags, *args, **kwargs)

        with patch.object(m, "os", local_os(
                read=Mock(side_effect=AssertionError("payload read")), open=open_directory)):
            result = self.census("second")
        self.assertEqual(calls, ["second", "nested"])
        self.assertFalse(result["sdk_payload_read"])

    def test_regular_file_shard_is_metadata_only(self):
        (self.root / "regular").write_bytes(b"synthetic payload")
        with patch.object(m, "os", local_os(open=Mock(side_effect=AssertionError("file opened")))):
            result = self.census("regular")
        self.assertEqual(result["visited_nodes"], 1)
        self.assertEqual(result["maximum_directory_depth"], 0)

    def test_root_symlink_counted_without_following(self):
        outside = Path(self.directory) / "outside"
        outside.mkdir()
        (outside / "sentinel").write_bytes(b"untouched")
        (self.root / "link").symlink_to(outside, target_is_directory=True)
        opener = Mock(side_effect=AssertionError("link followed"))
        with patch.object(m, "os", local_os(open=opener)):
            result = self.census("link")
        self.assertEqual(result["visited_nodes"], 1)
        self.assertEqual((outside / "sentinel").read_bytes(), b"untouched")

    def test_nested_symlink_is_leaf(self):
        (self.first / "link").symlink_to(self.second, target_is_directory=True)
        result = self.census()
        self.assertEqual(result["visited_nodes"], 3)
        self.assertEqual(result["maximum_directory_depth"], 1)

    def test_shard_node_overflow_not_accepted(self):
        (self.first / "0dir").mkdir()
        (self.first / "0dir" / "leaf").write_bytes(b"synthetic")
        result = m.empty_report()
        with patch.object(m, "SHARD_NODES", 3):
            self.assert_budget("first", result, "shard_nodes")
        self.assertEqual(result["visited_nodes"], 3)
        self.assertEqual(result["active_shard_nodes"], 3)

    def test_aggregate_overflow_preserves_counts(self):
        result = self.census()
        with patch.object(m, "AGGREGATE_NODES", 4):
            self.assert_budget("second", result, "aggregate_nodes")
        self.assertEqual(result["visited_nodes"], 4)
        self.assertEqual(result["active_shard_nodes"], 2)
        self.assertTrue((self.second / "nested" / "b").exists())

    def test_listing_cap_before_traversal(self):
        (self.first / "b").write_bytes(b"x")
        result = m.empty_report()
        with patch.object(m, "SHARD_NODES", 1):
            self.assert_budget("first", result, "shard_listing_entries")
        self.assertEqual(result["visited_nodes"], 1)

    def test_shard_name_cap_exact_and_root_inclusive(self):
        cap = len("first/") + len("first/a/")
        with patch.object(m, "MAXIMUM_NAME_BYTES", cap):
            self.assertEqual(self.census()["active_shard_name_bytes"], cap)
        result = m.empty_report()
        with patch.object(m, "MAXIMUM_NAME_BYTES", cap - 1):
            self.assert_budget("first", result, "shard_relative_name_bytes")
        self.assertEqual(result["active_shard_name_bytes"], len("first/"))

    def test_aggregate_name_cap_counts_previous_shards(self):
        result = self.census()
        accepted = result["accepted_relative_name_bytes"]
        with patch.object(m, "AGGREGATE_NAME_BYTES", accepted):
            self.assert_budget("second", result, "aggregate_relative_name_bytes")
        self.assertEqual(result["accepted_relative_name_bytes"], accepted)
        self.assertEqual(result["active_shard_name_bytes"], 0)

    def test_shard_allocated_cap_exact(self):
        expected = sum(path.lstat().st_blocks * 512 for path in (self.first, self.first / "a"))
        self.assertGreater(expected, 0)
        with patch.object(m, "MAXIMUM_ALLOCATED_BYTES", expected):
            self.assertEqual(self.census()["active_shard_allocated_bytes"], expected)
        result = m.empty_report()
        with patch.object(m, "MAXIMUM_ALLOCATED_BYTES", expected - 1):
            self.assert_budget("first", result, "shard_allocated_bytes")

    def test_aggregate_allocated_cap_counts_previous_shards(self):
        result = self.census()
        accepted = result["accepted_allocated_bytes"]
        with patch.object(m, "AGGREGATE_ALLOCATED_BYTES", accepted):
            self.assert_budget("second", result, "aggregate_allocated_bytes")
        self.assertEqual(result["accepted_allocated_bytes"], accepted)

    def test_directory_depth_sdk_relative_and_exclusive(self):
        (self.first / "nested").mkdir()
        result = m.empty_report()
        with patch.object(m, "MAXIMUM_DEPTH", 2):
            self.assert_budget("first", result, "shard_depth")
        self.assertEqual(result["maximum_directory_depth"], 1)

    def test_leaf_at_directory_depth_boundary_valid(self):
        with patch.object(m, "MAXIMUM_DEPTH", 2):
            result = self.census()
        self.assertEqual(result["maximum_directory_depth"], 1)

    def test_hardlinked_regular_file_rejected(self):
        os.link(self.first / "a", self.root / "hardlink")
        with self.assertRaises(m.CensusError) as caught:
            self.census("hardlink")
        self.assertEqual(caught.exception.code, "tree_rejected")

    def test_special_node_rejected(self):
        os.mkfifo(self.root / "fifo")
        with self.assertRaises(m.CensusError) as caught:
            self.census("fifo")
        self.assertEqual(caught.exception.code, "tree_rejected")

    def test_cross_device_rejected(self):
        def other_device(*args, **kwargs):
            info = os.stat(*args, **kwargs)
            return changed_info(info, st_dev=info.st_dev + 1)

        with patch.object(m, "os", local_os(stat=other_device)), \
                self.assertRaises(m.CensusError) as caught:
            self.census()
        self.assertEqual(caught.exception.code, "tree_rejected")

    def test_opened_directory_identity_change_rejected(self):
        def changed_fstat(fd):
            info = os.fstat(fd)
            return info if fd == self.fd else changed_info(info, st_ino=info.st_ino + 1)

        with patch.object(m, "os", local_os(fstat=changed_fstat)), \
                self.assertRaises(m.CensusError) as caught:
            self.census()
        self.assertEqual(caught.exception.code, "identity_changed")

    def test_changed_directory_listing_rejected(self):
        original_names = m.names
        calls = 0

        def changed_names(*args):
            nonlocal calls
            calls += 1
            return original_names(*args) if calls == 1 else []

        with patch.object(m, "names", side_effect=changed_names), \
                self.assertRaises(m.CensusError) as caught:
            self.census()
        self.assertEqual(caught.exception.code, "identity_changed")

    def test_negative_allocated_metadata_rejected(self):
        def negative_blocks(*args, **kwargs):
            return changed_info(os.stat(*args, **kwargs), st_blocks=-1)

        with patch.object(m, "os", local_os(stat=negative_blocks)), \
                self.assertRaises(m.CensusError) as caught:
            self.census()
        self.assertEqual(caught.exception.code, "tree_rejected")


class ContractTests(unittest.TestCase):
    def budget_report(self):
        value = m.empty_report()
        value.update(status="observed", stage="census", code="budget_hit",
                     budget_kind="aggregate_nodes", budget_limit=m.AGGREGATE_NODES,
                     visited_nodes=m.AGGREGATE_NODES, attempted_value=m.AGGREGATE_NODES + 1)
        return value

    def test_closed_budget_report_and_false_flags(self):
        value = self.budget_report()
        m.validate_report(value)
        self.assertTrue(all(value[key] is False for key in m.FALSE_FLAGS))

    def test_false_flags_not_promotable(self):
        for key in m.FALSE_FLAGS:
            with self.subTest(key=key), self.assertRaises(m.CensusError):
                m.validate_report({**m.empty_report(), key: True})

    def test_unknown_fields_rejected(self):
        with self.assertRaises(m.CensusError):
            m.validate_report({**m.empty_report(), "path": "/synthetic/private"})

    def test_counter_bounds_and_bool_counters_rejected(self):
        bounds = {
            "visited_nodes": m.AGGREGATE_NODES,
            "active_shard_nodes": m.SHARD_NODES,
            "active_shard_name_bytes": m.MAXIMUM_NAME_BYTES,
            "active_shard_allocated_bytes": m.MAXIMUM_ALLOCATED_BYTES,
            "accepted_relative_name_bytes": m.AGGREGATE_NAME_BYTES,
            "accepted_allocated_bytes": m.AGGREGATE_ALLOCATED_BYTES,
            "maximum_directory_depth": m.MAXIMUM_DEPTH - 1,
            "largest_shard_listing_entries": m.SHARD_NODES,
        }
        for key, maximum in bounds.items():
            for value in (-1, maximum + 1, True):
                with self.subTest(key=key, value=value), self.assertRaises(m.CensusError):
                    m.validate_report({**m.empty_report(), key: value})

    def test_active_counters_cannot_exceed_aggregate(self):
        fields = ("active_shard_nodes", "active_shard_name_bytes", "active_shard_allocated_bytes")
        for key in fields:
            with self.subTest(key=key), self.assertRaises(m.CensusError):
                m.validate_report({**m.empty_report(), key: 1})

    def test_forged_budget_limits_and_non_exceedances_rejected(self):
        for changes in ({"budget_limit": 1}, {"attempted_value": m.AGGREGATE_NODES},
                        {"budget_kind": "unbounded"}, {"budget_limit": True}):
            with self.subTest(changes=changes), self.assertRaises(m.CensusError):
                m.validate_report({**self.budget_report(), **changes})

    def test_complete_requires_all_started_shards_completed(self):
        value = m.empty_report()
        value.update(status="observed", stage="complete", code="census_complete",
                     census_complete=True, shards_started=1)
        with self.assertRaises(m.CensusError):
            m.validate_report(value)

    def test_budget_report_cannot_claim_completion(self):
        with self.assertRaises(m.CensusError):
            m.validate_report({**self.budget_report(), "census_complete": True})

    def test_failed_report_cannot_claim_completion(self):
        with self.assertRaises(m.CensusError):
            m.validate_report({**m.empty_report(), "census_complete": True})

    def test_read_only_guard_rejects_mutation_and_processes(self):
        for event in ("os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.link", "os.symlink",
                      "os.chmod", "os.chown", "os.fchmod", "os.fchown", "os.truncate",
                      "os.utime", "os.setxattr", "os.removexattr", "os.fork", "os.forkpty",
                      "os.exec", "os.posix_spawn", "os.spawn", "os.system", "shutil.rmtree",
                      "socket.__new__", "subprocess.Popen", "ctypes.dlopen"):
            with self.subTest(event=event), self.assertRaises(m.CensusError) as caught:
                m.audit_guard(event, ())
            self.assertEqual(caught.exception.code, "read_only_rejected")

    def test_read_only_guard_rejects_payload_opens_and_write_flags(self):
        for path, mode, flags in (("x", None, os.O_RDONLY), ("x", "r", os.O_RDONLY),
                                  ("/proc/self/mountinfo", "rb", os.O_RDWR),
                                  ("x", None, os.O_DIRECTORY | os.O_CREAT),
                                  ("x", None, os.O_DIRECTORY | os.O_WRONLY),
                                  ("x", None, os.O_DIRECTORY | os.O_TRUNC),
                                  ("x", None, os.O_DIRECTORY | os.O_APPEND)):
            with self.subTest(flags=flags), self.assertRaises(m.CensusError):
                m.audit_guard("open", (path, mode, flags))

    def test_read_only_guard_allows_directory_and_mount_metadata(self):
        m.audit_guard("open", ("synthetic", None, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW))
        m.audit_guard("open", ("/proc/self/mountinfo", "r", os.O_RDONLY))
        m.audit_guard("open", ("/proc/self/mountinfo", "rb", os.O_RDONLY))
        m.audit_guard("os.scandir", (123,))

    def test_workflow_binds_script_and_single_guarded_run(self):
        script = ROOT / "scripts/diagnose_sft_sdk_sharded_census_v1.py"
        workflow = (ROOT / ".github/workflows/sft-sdk-sharded-census-v1.yml").read_text()
        self.assertIn(hashlib.sha256(script.read_bytes()).hexdigest(), workflow)
        self.assertNotRegex(workflow, r"\brm\s|\bcurl\s|\bwget\s|\bdocker\s|\baws\s|git push")
        self.assertEqual(workflow.count("sudo -n /usr/bin/env -i"), 1)
        self.assertIn("--read-only-sharded-sdk-census", workflow)
        self.assertIn("github.run_attempt == 1", workflow)
        self.assertIn("github.event.before == '7da33c657497386dc0e5cf4c6fb1f66bf03cfc05'", workflow)


class FakeReclaimError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(m, "_ATTEMPTED", False))
        self.directory = self.enterContext(tempfile.TemporaryDirectory(prefix="sdk-shard-run-"))
        self.root = Path(self.directory) / "sdk"
        self.root.mkdir()
        (self.root / "one").mkdir()
        (self.root / "one" / "a").write_bytes(b"synthetic")
        self.handles = []
        self.addCleanup(self.close_owned)
        self.helper = types.SimpleNamespace(
            PARTS=("sdk",), identity=identity,
            open_target=Mock(side_effect=self.open_owned),
            check_mounts=Mock(), ReclaimError=FakeReclaimError,
        )
        self.enterContext(patch.object(m, "runtime"))
        self.load = self.enterContext(patch.object(m, "load_helper", return_value=self.helper))
        self.guard = self.enterContext(patch.object(m, "install_guard"))
        self.arm = self.enterContext(patch.object(m, "arm_deadline"))
        self.disarm = self.enterContext(patch.object(m, "disarm_deadline"))

    def open_owned(self):
        parent = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        root = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        self.handles.extend((fd, identity(os.fstat(fd))) for fd in (parent, root))
        return [parent, root], identity(os.fstat(root))

    def close_owned(self):
        for fd, bound in self.handles:
            try:
                if identity(os.fstat(fd)) == bound:
                    os.close(fd)
            except OSError:
                pass

    def test_run_completes_without_mutation(self):
        result = m.run()
        self.assertEqual(result["code"], "census_complete")
        self.assertTrue(result["census_complete"])
        self.assertEqual(result["visited_nodes"], 2)
        self.assertEqual(result["shards_started"], 1)
        self.assertEqual(result["shards_completed"], 1)
        self.helper.check_mounts.assert_called_once_with()
        self.assertEqual((self.root / "one" / "a").read_bytes(), b"synthetic")
        self.guard.assert_called_once_with()
        self.arm.assert_called_once_with()
        self.disarm.assert_called_once_with()

    def test_run_accepts_regular_and_symlink_roots(self):
        (self.root / "leaf").write_bytes(b"synthetic")
        (self.root / "link").symlink_to(self.root / "one", target_is_directory=True)
        result = m.run()
        self.assertEqual(result["code"], "census_complete")
        self.assertEqual(result["shards_completed"], 3)
        self.assertEqual(result["visited_nodes"], 4)

    def test_run_budget_observed_and_never_retried(self):
        with patch.object(m, "AGGREGATE_NODES", 1):
            result = m.run()
            self.assertEqual(result["status"], "observed")
            self.assertEqual(result["code"], "budget_hit")
            self.assertEqual(result["budget_kind"], "aggregate_nodes")
            second = m.run()
        self.assertEqual(second["code"], "already_attempted")
        self.helper.open_target.assert_called_once_with()
        self.load.assert_called_once_with()

    def test_root_listing_budget_has_census_stage(self):
        (self.root / "two").mkdir()
        with patch.object(m, "SHARD_NODES", 1):
            result = m.run()
        self.assertEqual(result["stage"], "census")
        self.assertEqual(result["code"], "budget_hit")
        self.assertEqual(result["budget_kind"], "shard_listing_entries")
        self.assertEqual(result["shards_started"], 0)

    def test_exception_text_never_enters_report(self):
        with patch.object(m, "shard_census", side_effect=RuntimeError("SENSITIVE NAME")):
            result = m.run()
        self.assertEqual(result["code"], "operation_failed")
        self.assertNotIn("SENSITIVE", m.canonical(result).decode())
        self.assertFalse(result["census_complete"])

    def test_deadline_fails_closed(self):
        with patch.object(m, "shard_census", side_effect=m.CensusError("deadline_exhausted")):
            result = m.run()
        self.assertEqual(result["code"], "deadline_exhausted")
        self.assertEqual(result["status"], "failed")
        self.disarm.assert_called_once_with()

    def test_identity_change_after_census_rejected(self):
        self.helper.check_mounts.side_effect = FakeReclaimError("identity_changed")
        result = m.run()
        self.assertEqual(result["code"], "identity_changed")
        self.assertFalse(result["census_complete"])

    def test_close_failure_clears_success(self):
        def close_then_fail(fd):
            os.close(fd)
            raise OSError("synthetic close failure")

        with patch.object(m, "os", local_os(close=close_then_fail)):
            result = m.run()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["code"], "operation_failed")
        self.assertFalse(result["census_complete"])

    def test_invalid_runtime_does_not_load_helper(self):
        with patch.object(m, "runtime", side_effect=m.CensusError("runtime_rejected")):
            result = m.run()
        self.assertEqual(result["code"], "runtime_rejected")
        self.load.assert_not_called()
        self.helper.open_target.assert_not_called()


if __name__ == "__main__":
    unittest.main()
