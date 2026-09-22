"""Owned synthetic fixtures only; never run the real SDK or native setup probe."""

from __future__ import annotations

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


m = definitions("scripts/reclaim_sft_sdk_prefix_v1.py", "_sdk_prefix_test_definitions")
p = definitions("scripts/probe_sft_prepared_setup_v1.py", "_sdk_prefix_probe_definitions")


def helper_definitions():
    return definitions("scripts/reclaim_sft_android_sdk_v1.py", "_sdk_prefix_helper_definitions")


def local_os(**overrides):
    """Module-local replacements avoid changing the test runner's shared os module."""
    return types.SimpleNamespace(**{**vars(os), **overrides})


def changed_info(info, **changes):
    return types.SimpleNamespace(**{
        **{key: getattr(info, key) for key in dir(info) if key.startswith("st_")}, **changes,
    })


def synthetic_space(enough=True):
    return types.SimpleNamespace(f_bavail=(32 if enough else 1) * 1024**3 // 4096,
                                 f_frsize=4096)


def setup_observation(enough=True):
    value = p.empty_report()
    value.update(status="observed", stage="complete", code="ok" if enough else
                 "disk_floor_insufficient", source_bound_files=78,
                 owned_temp_directories_created=2, owned_temp_directories_removed=2,
                 setup_subreaper_installed=True, ready_for_unchanged_setup=enough)
    value.update(p.capacity(synthetic_space(enough)))
    return p.validate_report(value)


class ShardTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory(prefix="sdk-prefix-shard-"))
        self.root = Path(self.directory) / "sdk"
        self.root.mkdir()
        self.first = self.root / "first"
        self.first.mkdir()
        (self.first / "a").write_bytes(b"synthetic fixture")
        self.second = self.root / "second"
        self.second.mkdir()
        (self.second / "nested").mkdir()
        (self.second / "nested" / "b").write_bytes(b"synthetic fixture")
        self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, self.fd)
        self.helper = helper_definitions()

    def census(self, name="first", report=None):
        report = m.empty_report() if report is None else report
        return m.census_shard(self.fd, name, self.helper, report), report

    def test_root_included_and_relative_maps_compatible_with_frozen_remover(self):
        shard, report = self.census()
        self.assertEqual(report["census_nodes"], 2)
        self.assertEqual(shard[0], "first")
        self.assertEqual(shard[1], self.helper.file_state(self.first.lstat()))
        self.assertEqual(set(shard[2]), {("a",)})
        self.assertEqual(shard[3], {(): ["a"]})
        self.assertEqual(report["census_name_bytes"], len("first/") + len("first/a/"))

    def test_exact_shard_node_cap_allows_relisting(self):
        with patch.object(m, "SHARD_NODES", 2):
            _, report = self.census()
        self.assertEqual(report["census_nodes"], 2)

    def test_exact_nested_node_cap_allows_both_relistings(self):
        with patch.object(m, "SHARD_NODES", 3):
            _, report = self.census("second")
        self.assertEqual(report["census_nodes"], 3)

    def test_exact_aggregate_node_cap_allows_multiple_shards(self):
        with patch.object(m, "AGGREGATE_NODES", 5):
            _, report = self.census()
            self.census("second", report)
        self.assertEqual(report["census_nodes"], 5)

    def test_shard_node_overflow_preserves_accepted_counts(self):
        report = m.empty_report()
        with patch.object(m, "SHARD_NODES", 1), self.assertRaises(ValueError):
            self.census(report=report)
        self.assertEqual(report["census_nodes"], 1)
        self.assertTrue((self.first / "a").is_file())

    def test_aggregate_node_overflow_before_next_acceptance(self):
        _, report = self.census()
        with patch.object(m, "AGGREGATE_NODES", 4), self.assertRaises(ValueError):
            self.census("second", report)
        self.assertEqual(report["census_nodes"], 4)

    def test_shard_name_cap_includes_root(self):
        cap = len("first/") + len("first/a/")
        with patch.object(m, "MAXIMUM_NAME_BYTES", cap):
            _, report = self.census()
        self.assertEqual(report["census_name_bytes"], cap)
        with patch.object(m, "MAXIMUM_NAME_BYTES", cap - 1), self.assertRaises(ValueError):
            self.census()

    def test_aggregate_name_cap_counts_previous_shards(self):
        _, report = self.census()
        accepted = report["census_name_bytes"]
        with patch.object(m, "AGGREGATE_NAME_BYTES", accepted), self.assertRaises(ValueError):
            self.census("second", report)
        self.assertEqual(report["census_name_bytes"], accepted)

    def test_exact_shard_allocated_cap(self):
        allocation = sum(path.lstat().st_blocks * 512 for path in (self.first, self.first / "a"))
        self.assertGreater(allocation, 0)
        with patch.object(m, "MAXIMUM_ALLOCATED_BYTES", allocation):
            _, report = self.census()
        self.assertEqual(report["census_allocated_bytes"], allocation)
        with patch.object(m, "MAXIMUM_ALLOCATED_BYTES", allocation - 1), \
                self.assertRaises(ValueError):
            self.census()

    def test_aggregate_allocation_cap_counts_previous_shards(self):
        _, report = self.census()
        allocation = report["census_allocated_bytes"]
        with patch.object(m, "AGGREGATE_ALLOCATED_BYTES", allocation), \
                self.assertRaises(ValueError):
            self.census("second", report)
        self.assertEqual(report["census_allocated_bytes"], allocation)

    def test_directory_depth_exclusive_but_leaf_at_boundary_allowed(self):
        with patch.object(m, "MAXIMUM_DEPTH", 2):
            _, report = self.census()
            with self.assertRaises(ValueError):
                self.census("second")
        self.assertEqual(report["census_nodes"], 2)

    def test_no_payload_reads_and_only_nofollow_directories_opened(self):
        opened = []

        def directories_only(path, flags, *args, **kwargs):
            self.assertTrue(flags & os.O_DIRECTORY)
            self.assertTrue(flags & os.O_NOFOLLOW)
            self.assertFalse(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            opened.append(path)
            return os.open(path, flags, *args, **kwargs)

        with patch.object(m, "os", local_os(open=directories_only,
                                            read=Mock(side_effect=AssertionError("payload")))):
            self.census("second")
        self.assertEqual(opened, ["second", "nested"])

    def test_regular_and_symlink_roots_are_unopened_leaves(self):
        outside = Path(self.directory) / "outside"
        outside.write_bytes(b"external sentinel")
        (self.root / "regular").write_bytes(b"synthetic leaf")
        (self.root / "link").symlink_to(outside)
        for name in ("regular", "link"):
            with self.subTest(name=name), patch.object(m, "os", local_os(
                    open=Mock(side_effect=AssertionError("leaf opened")))):
                shard, report = self.census(name)
            self.assertEqual(report["census_nodes"], 1)
            m.remove_shard(self.fd, shard, self.helper, report)
            self.assertEqual(report["removed_leaves"], 1)
        self.assertEqual(outside.read_bytes(), b"external sentinel")

    def test_nested_symlink_is_removed_without_target_access(self):
        outside = Path(self.directory) / "outside"
        outside.mkdir()
        sentinel = outside / "sentinel"
        sentinel.write_bytes(b"untouched")
        (self.first / "link").symlink_to(outside, target_is_directory=True)
        shard, report = self.census()
        m.verify_shard(self.fd, shard, self.helper)
        m.remove_shard(self.fd, shard, self.helper, report)
        self.assertEqual(report["census_nodes"], 3)
        self.assertEqual(report["removed_leaves"], 2)
        self.assertEqual(report["removed_directories"], 1)
        self.assertEqual(sentinel.read_bytes(), b"untouched")

    def test_hardlinked_regular_rejected(self):
        os.link(self.first / "a", self.root / "hardlink")
        with self.assertRaises(ValueError):
            self.census("hardlink")

    def test_special_node_rejected(self):
        os.mkfifo(self.root / "fifo")
        with self.assertRaises(ValueError):
            self.census("fifo")

    def test_cross_device_rejected(self):
        def other_device(*args, **kwargs):
            info = os.stat(*args, **kwargs)
            return changed_info(info, st_dev=info.st_dev + 1)

        with patch.object(m, "os", local_os(stat=other_device)), self.assertRaises(ValueError):
            self.census()

    def test_negative_allocation_rejected(self):
        def negative_blocks(*args, **kwargs):
            return changed_info(os.stat(*args, **kwargs), st_blocks=-1)

        with patch.object(m, "os", local_os(stat=negative_blocks)), self.assertRaises(ValueError):
            self.census()

    def test_changed_open_directory_identity_rejected(self):
        def changed_fstat(fd):
            info = os.fstat(fd)
            return info if fd == self.fd else changed_info(info, st_ino=info.st_ino + 1)

        with patch.object(m, "os", local_os(fstat=changed_fstat)), self.assertRaises(ValueError):
            self.census()

    def test_predelete_verification_rejects_changed_leaf(self):
        shard, _ = self.census()
        (self.first / "a").write_bytes(b"changed size and mtime")
        with self.assertRaises(ValueError):
            m.verify_shard(self.fd, shard, self.helper)
        self.assertTrue(self.first.is_dir())

    def test_predelete_verification_rejects_changed_listing(self):
        shard, _ = self.census()
        (self.first / "new").write_bytes(b"new")
        with self.assertRaises(ValueError):
            m.verify_shard(self.fd, shard, self.helper)
        self.assertTrue((self.first / "a").is_file())

    def test_remove_bound_shard_preserves_sdk_and_other_children(self):
        root_identity = self.helper.identity(self.root.lstat())
        shard, report = self.census("second")
        m.verify_shard(self.fd, shard, self.helper)
        m.remove_shard(self.fd, shard, self.helper, report)
        self.assertFalse(self.second.exists())
        self.assertEqual(self.helper.identity(self.root.lstat()), root_identity)
        self.assertTrue((self.first / "a").is_file())
        self.assertEqual(report["removed_directories"], 2)
        self.assertEqual(report["removed_leaves"], 1)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(m, "_ATTEMPTED", False))
        self.directory = self.enterContext(tempfile.TemporaryDirectory(prefix="sdk-prefix-run-"))
        self.root = Path(self.directory) / "sdk"
        self.root.mkdir()
        for index in reversed(range(11)):
            child = self.root / f"child{index:02}"
            child.mkdir()
            (child / "sentinel").write_bytes(b"synthetic sentinel")
        self.handles = []
        self.helper = helper_definitions()
        self.addCleanup(self.close_owned)
        self.probe = types.SimpleNamespace(**{
            **vars(p), "runtime": Mock(), "install_guard": Mock(),
            "load_harness": Mock(return_value=types.SimpleNamespace(runtime=Mock())),
            "run": Mock(return_value=setup_observation()),
        })
        self.helper.PARTS = ("sdk",)
        self.helper.open_target = Mock(side_effect=self.open_owned)
        self.helper.check_mounts = Mock()
        self.helper.load_probe = Mock(return_value=self.probe)
        self.runtime = self.enterContext(patch.object(m, "runtime"))
        self.load = self.enterContext(patch.object(m, "load_helper", return_value=self.helper))
        self.arm = self.enterContext(patch.object(m, "arm_deadline"))
        self.disarm = self.enterContext(patch.object(m, "disarm_deadline"))
        self.enterContext(patch.object(m, "os", local_os(
            statvfs=Mock(return_value=synthetic_space()))))

    def open_owned(self):
        parent = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        root = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        self.handles.extend((fd, self.helper.identity(os.fstat(fd))) for fd in (parent, root))
        return [parent, root], self.helper.identity(os.fstat(root))

    def close_owned(self):
        for fd, bound in self.handles:
            try:
                if self.helper.identity(os.fstat(fd)) == bound:
                    os.close(fd)
            except OSError:
                pass

    def test_first_ten_sorted_deleted_sdk_root_and_eleventh_preserved(self):
        before = self.helper.identity(self.root.lstat())
        result = m.run()
        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["code"], "ok")
        self.assertEqual(result["selected_shards"], 10)
        self.assertEqual(result["audited_shards"], 10)
        self.assertEqual(result["deleted_shards"], 10)
        self.assertEqual(result["census_nodes"], 20)
        self.assertEqual(result["removed_leaves"], 10)
        self.assertEqual(result["removed_directories"], 10)
        self.assertTrue(result["sdk_root_preserved"])
        self.assertTrue(result["selected_absence_verified"])
        self.assertTrue(result["unselected_root_names_unchanged"])
        self.assertEqual(self.helper.identity(self.root.lstat()), before)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["child10"])
        self.assertEqual((self.root / "child10" / "sentinel").read_bytes(), b"synthetic sentinel")
        self.probe.run.assert_called_once_with()
        self.assertTrue(result["ready_for_unchanged_setup"])

    def test_eleventh_child_never_statted_or_opened(self):
        touched = []

        def selected_stat(path, *args, **kwargs):
            self.assertNotEqual(path, "child10")
            touched.append(("stat", path))
            return os.stat(path, *args, **kwargs)

        def selected_open(path, *args, **kwargs):
            self.assertNotEqual(path, "child10")
            touched.append(("open", path))
            return os.open(path, *args, **kwargs)

        replacement = local_os(stat=selected_stat, open=selected_open,
                               statvfs=Mock(return_value=synthetic_space()))
        with patch.object(m, "os", replacement), patch.object(self.helper, "os", replacement):
            result = m.run()
        self.assertEqual(result["code"], "ok")
        self.assertTrue(touched)

    def test_all_selected_audited_and_verified_before_first_delete(self):
        events = []
        original_census, original_verify, original_remove = (
            m.census_shard, m.verify_shard, m.remove_shard)

        def census(fd, name, helper, report):
            events.append(("census", name))
            return original_census(fd, name, helper, report)

        def verify(fd, shard, helper):
            events.append(("verify", shard[0]))
            return original_verify(fd, shard, helper)

        def remove(fd, shard, helper, report):
            events.append(("remove", shard[0]))
            return original_remove(fd, shard, helper, report)

        with patch.object(m, "census_shard", side_effect=census), \
                patch.object(m, "verify_shard", side_effect=verify), \
                patch.object(m, "remove_shard", side_effect=remove):
            result = m.run()
        self.assertEqual(result["code"], "ok")
        before_delete = events[:next(i for i, event in enumerate(events) if event[0] == "remove")]
        self.assertEqual(sum(kind == "census" for kind, _ in before_delete), 10)
        self.assertEqual(sum(kind == "verify" for kind, _ in before_delete), 10)

    def test_tenth_shard_failure_leaves_all_targets_intact(self):
        os.mkfifo(self.root / "child09" / "rejected")
        result = m.run()
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["removal_started"])
        self.assertEqual(result["deleted_shards"], 0)
        self.assertEqual(len(list(self.root.iterdir())), 11)
        self.probe.run.assert_not_called()

    def test_verify_failure_before_removal_leaves_all_targets_intact(self):
        with patch.object(m, "verify_shard", side_effect=RuntimeError("private name")):
            result = m.run()
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["removal_started"])
        self.assertEqual(len(list(self.root.iterdir())), 11)
        self.assertNotIn("private name", m.canonical(result).decode())
        self.probe.run.assert_not_called()

    def test_mount_recheck_failure_leaves_all_targets_intact(self):
        self.helper.check_mounts.side_effect = self.helper.ReclaimError("target_rejected")
        result = m.run()
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["removal_started"])
        self.assertEqual(len(list(self.root.iterdir())), 11)

    def test_root_listing_change_before_removal_rejects_without_deletion(self):
        original_verify = m.verify_shard

        def add_child(fd, shard, helper):
            original_verify(fd, shard, helper)
            if shard[0] == "child09":
                (self.root / "new-child").mkdir()

        with patch.object(m, "verify_shard", side_effect=add_child):
            result = m.run()
        self.assertEqual(result["code"], "identity_changed")
        self.assertFalse(result["removal_started"])
        self.assertEqual(len(list(self.root.iterdir())), 12)

    def test_fewer_than_ten_children_selects_only_available_prefix(self):
        for index in range(1, 11):
            child = self.root / f"child{index:02}"
            (child / "sentinel").unlink()
            child.rmdir()
        result = m.run()
        self.assertEqual(result["code"], "ok")
        self.assertEqual(result["selected_shards"], 1)
        self.assertEqual(result["deleted_shards"], 1)
        self.assertTrue(self.root.is_dir())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_empty_prefix_rejects_before_mutation(self):
        for index in range(11):
            child = self.root / f"child{index:02}"
            (child / "sentinel").unlink()
            child.rmdir()
        result = m.run()
        self.assertEqual(result["code"], "tree_rejected")
        self.assertFalse(result["removal_started"])
        self.probe.run.assert_not_called()

    def test_partial_deletion_error_preserves_honest_counts(self):
        original_remove = m.remove_shard
        calls = 0

        def fail_second(fd, shard, helper, report):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("private deletion detail")
            return original_remove(fd, shard, helper, report)

        with patch.object(m, "remove_shard", side_effect=fail_second):
            result = m.run()
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["audit_complete"])
        self.assertTrue(result["removal_started"])
        self.assertEqual(result["deleted_shards"], 1)
        self.assertEqual(result["removed_leaves"] + result["removed_directories"], 2)
        self.assertFalse(result["ready_for_unchanged_setup"])
        self.assertNotIn("private deletion detail", m.canonical(result).decode())
        self.probe.run.assert_not_called()

    def test_attempt_never_retried_after_failure(self):
        with patch.object(m, "census_shard", side_effect=RuntimeError("synthetic")):
            first = m.run()
        second = m.run()
        self.assertEqual(first["status"], "failed")
        self.assertEqual(second["code"], "already_attempted")
        self.load.assert_called_once_with()
        self.helper.open_target.assert_called_once_with()

    def test_low_capacity_does_not_run_probe_or_expand_selection(self):
        with patch.object(m, "os", local_os(statvfs=Mock(return_value=synthetic_space(False)))):
            result = m.run()
        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["code"], "capacity_insufficient")
        self.assertEqual(result["deleted_shards"], 10)
        self.assertTrue((self.root / "child10" / "sentinel").is_file())
        self.assertFalse(result["ready_for_unchanged_setup"])
        self.probe.run.assert_not_called()

    def test_cleanup_alarm_disarmed_before_setup_probe(self):
        def setup():
            self.disarm.assert_called_once_with()
            for fd, _ in self.handles:
                with self.assertRaises(OSError):
                    os.fstat(fd)
            return setup_observation()

        self.probe.run.side_effect = setup
        result = m.run()
        self.assertEqual(result["code"], "ok")
        self.arm.assert_called_once_with()
        self.disarm.assert_called_once_with()

    def test_malformed_nested_setup_report_never_retained(self):
        self.probe.run.return_value = {**setup_observation(), "private_path": "SENSITIVE"}
        result = m.run()
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["setup_report"])
        self.assertNotIn("SENSITIVE", m.canonical(result).decode())
        self.assertFalse(result["ready_for_unchanged_setup"])

    def test_failed_valid_nested_setup_report_never_claims_ready(self):
        self.probe.run.return_value = p.empty_report()
        result = m.run()
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["ready_for_unchanged_setup"])
        self.assertEqual(result["setup_report"], p.empty_report())

    def test_invalid_runtime_rejects_before_loading_or_target_access(self):
        self.runtime.side_effect = RuntimeError("wrong environment")
        result = m.run()
        self.assertEqual(result["status"], "failed")
        self.load.assert_not_called()
        self.helper.open_target.assert_not_called()

    def test_frozen_probe_runtime_checked_before_target_access(self):
        self.probe.runtime.side_effect = p.PreflightError("runtime_rejected")
        result = m.run()
        self.assertEqual(result["status"], "failed")
        self.helper.open_target.assert_not_called()
        self.arm.assert_not_called()

    def test_frozen_harness_runtime_checked_before_target_access(self):
        self.probe.load_harness.return_value.runtime.side_effect = RuntimeError("synthetic")
        result = m.run()
        self.assertEqual(result["status"], "failed")
        self.helper.open_target.assert_not_called()
        self.arm.assert_not_called()

    def test_deadline_error_before_removal_is_fixed_code_and_disarms(self):
        with patch.object(m, "verify_shard", side_effect=m.PrefixError("deadline_exhausted")):
            result = m.run()
        self.assertEqual(result["code"], "deadline_exhausted")
        self.assertFalse(result["removal_started"])
        self.disarm.assert_called_once_with()

    def test_close_failure_preserves_deletion_evidence_and_skips_probe(self):
        failed = False

        def close_then_fail(fd):
            nonlocal failed
            os.close(fd)
            if len(self.handles) == 2 and fd == self.handles[1][0] and not failed:
                failed = True
                raise OSError("synthetic close failure")

        with patch.object(m, "os", local_os(close=close_then_fail,
                                            statvfs=Mock(return_value=synthetic_space()))):
            result = m.run()
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["removal_started"])
        self.assertEqual(result["deleted_shards"], 10)
        self.assertTrue(result["sdk_root_preserved"])
        self.probe.run.assert_not_called()


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.flags = dict.fromkeys(("LUMI_SDK_PREFIX_CAPACITY_CI",
                                   "LUMI_PREPARED_SETUP_PREFLIGHT_CI",
                                   "LUMI_CHECKPOINT_GUARD_QUALIFICATION_CI"), "1")
        self.fake_os = local_os(environ=self.flags, uname=lambda: types.SimpleNamespace(
            machine="x86_64"), getresuid=lambda: (0, 0, 0), getresgid=lambda: (0, 0, 0))
        self.fake_sys = types.SimpleNamespace(
            platform="linux", implementation=types.SimpleNamespace(name="cpython"),
            version_info=(3, 12, 0), flags=types.SimpleNamespace(isolated=True, no_site=True),
            dont_write_bytecode=True)
        self.fake_signal = types.SimpleNamespace(**{
            **vars(m.signal), "getsignal": lambda _signum: m.signal.SIG_DFL,
            "getitimer": lambda _timer: (0.0, 0.0),
        })
        current = object()
        self.enterContext(patch.object(m, "os", self.fake_os))
        self.enterContext(patch.object(m, "sys", self.fake_sys))
        self.enterContext(patch.object(m, "signal", self.fake_signal))
        self.enterContext(patch.object(m, "threading", types.SimpleNamespace(
            current_thread=lambda: current, main_thread=lambda: current, active_count=lambda: 1)))

    def test_exact_synthetic_runtime_contract_passes_without_side_effects(self):
        m.runtime()

    def test_every_required_flag_is_mandatory(self):
        for flag in tuple(self.flags):
            with self.subTest(flag=flag), patch.dict(self.flags, {flag: "0"}), \
                    self.assertRaises(m.PrefixError) as caught:
                m.runtime()
            self.assertEqual(caught.exception.code, "runtime_rejected")

    def test_mac_and_wrong_python_version_rejected(self):
        for field, value in (("platform", "darwin"), ("version_info", (3, 13, 0)),
                             ("dont_write_bytecode", False)):
            with self.subTest(field=field), patch.object(self.fake_sys, field, value), \
                    self.assertRaises(m.PrefixError):
                m.runtime()

    def test_nonroot_identity_and_existing_alarm_rejected(self):
        with patch.object(self.fake_os, "getresuid", return_value=(1, 1, 1)), \
                self.assertRaises(m.PrefixError):
            m.runtime()
        with patch.object(self.fake_signal, "getitimer", return_value=(1.0, 0.0)), \
                self.assertRaises(m.PrefixError):
            m.runtime()

    def test_existing_signal_handler_rejected(self):
        with patch.object(self.fake_signal, "getsignal", return_value=lambda: None), \
                self.assertRaises(m.PrefixError):
            m.runtime()


class ContractTests(unittest.TestCase):
    def test_empty_report_is_closed_and_valid(self):
        value = m.empty_report()
        m.validate_report(value)
        self.assertLessEqual(len(m.canonical(value)), m.MAXIMUM_REPORT_BYTES)
        self.assertTrue(all(value[key] is False for key in m.FALSE_FLAGS))

    def test_unknown_fields_and_promotions_rejected(self):
        with self.assertRaises(ValueError):
            m.validate_report({**m.empty_report(), "path": "/private/synthetic"})
        for key in m.FALSE_FLAGS:
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.validate_report({**m.empty_report(), key: True})

    def test_counter_types_and_bounds_rejected(self):
        bounds = {
            "census_nodes": m.AGGREGATE_NODES,
            "census_name_bytes": m.AGGREGATE_NAME_BYTES,
            "census_allocated_bytes": m.AGGREGATE_ALLOCATED_BYTES,
            "selected_shards": m.MAXIMUM_SHARDS,
            "audited_shards": m.MAXIMUM_SHARDS,
            "deleted_shards": m.MAXIMUM_SHARDS,
            "removed_leaves": m.AGGREGATE_NODES,
            "removed_directories": m.AGGREGATE_NODES,
        }
        for key, maximum in bounds.items():
            for value in (-1, True, maximum + 1):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    m.validate_report({**m.empty_report(), key: value})

    def test_forged_removal_without_complete_audit_rejected(self):
        for changes in ({"removal_started": True}, {"removed_leaves": 1},
                        {"deleted_shards": 1}, {"audited_shards": 1},
                        {"ready_for_unchanged_setup": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                m.validate_report({**m.empty_report(), **changes})

    def test_capacity_arithmetic_and_schema_rejected(self):
        capacity = p.capacity(synthetic_space())
        for changes in ({"private_name": "leak"}, {"available_disk_bytes": True},
                        {"minimum_preacquisition_free_disk_bytes": 1},
                        {"acquisition_headroom_met": False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                m.validate_report({**m.empty_report(), "capacity_before": {
                    **capacity, **changes}}, p)

    def test_unknown_codes_cannot_retain_exception_text(self):
        for field in ("code", "stage", "status"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                m.validate_report({**m.empty_report(), field: "PRIVATE EXCEPTION TEXT"})

    def test_budget_report_accepts_only_bound_limit_and_actual_exceedance(self):
        value = m.empty_report()
        value.update(stage="census", code="budget_hit", budget_kind="aggregate_nodes",
                     budget_limit=m.AGGREGATE_NODES, attempted_value=m.AGGREGATE_NODES + 1)
        m.validate_report(value)
        for changes in ({"budget_limit": 1}, {"budget_limit": True},
                        {"attempted_value": m.AGGREGATE_NODES}, {"budget_kind": "private name"},
                        {"attempted_value": 2**63}, {"code": "operation_failed"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                m.validate_report({**value, **changes})


if __name__ == "__main__":
    unittest.main()
