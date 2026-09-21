"""Owned-directory and mocked orchestration tests; never remove the actual SDK.

The frozen setup probe is loaded as definitions only. Its native run, runtime,
audit installation and signal handling are always replaced for orchestration.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

ROOT = Path(__file__).resolve().parents[1]


def definitions(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = definitions("scripts/reclaim_sft_android_sdk_v1.py", "_sdk_capacity_test_definitions")
p = definitions("scripts/probe_sft_prepared_setup_v1.py", "_sdk_capacity_probe_definitions")


def namespace(module):
    return types.SimpleNamespace(**{name: getattr(module, name) for name in dir(module)})


def space(gib=32):
    return types.SimpleNamespace(f_bavail=gib * 1024**3 // 4096, f_frsize=4096)


def observed_setup(gib=32):
    result = p.empty_report()
    result.update(p.capacity(space(gib)))
    result.update(status="observed", stage="complete", source_bound_files=78,
                  setup_subreaper_installed=True, owned_temp_directories_created=2,
                  owned_temp_directories_removed=2)
    result["code"] = ("disk_floor_insufficient" if not result["disk_floor_met"] else
                      "acquisition_headroom_insufficient"
                      if not result["acquisition_headroom_met"] else "ok")
    result["ready_for_unchanged_setup"] = result["code"] == "ok"
    return p.validate_report(result)


class OwnedTreeTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory(prefix="sdk-unit-"))
        self.parent = Path(self.directory).resolve()
        self.sdk = self.parent / "sdk"
        self.sdk.mkdir()
        self.fd = os.open(self.sdk, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, self.fd)
        self.enterContext(patch.object(m, "os", namespace(os)))

    def file(self, name, body=b"synthetic"):
        path = self.sdk / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return path

    def rejected(self, callback, code):
        with self.assertRaises(m.ReclaimError) as error:
            callback()
        self.assertEqual(error.exception.code, code)

    def test_census_and_removal_read_no_payloads(self):
        self.file("a")
        self.file("nested/b")
        with patch.object(m.os, "read", side_effect=AssertionError("payload read")):
            tree = m.census(self.fd)
            result = m.empty_report()
            m.remove_tree(self.fd, tree, result)
        self.assertEqual(len(tree[0]), 3)
        self.assertGreaterEqual(tree[2], 0)
        self.assertEqual(result["removed_leaves"], 2)
        self.assertEqual(result["removed_directories"], 1)
        self.assertEqual(list(self.sdk.iterdir()), [])
        self.assertTrue(self.sdk.is_dir())

    def test_symlink_leaf_removed_without_following_target(self):
        outside = self.parent / "outside"
        outside.mkdir()
        sentinel = outside / "sentinel"
        sentinel.write_bytes(b"synthetic sentinel")
        (self.sdk / "link").symlink_to(outside, target_is_directory=True)
        (self.sdk / "dangling").symlink_to(self.parent / "missing")
        result = m.empty_report()
        with patch.object(m.os, "read", side_effect=AssertionError("payload read")):
            m.remove_tree(self.fd, m.census(self.fd), result)
        self.assertEqual(result["removed_leaves"], 2)
        self.assertEqual(sentinel.read_bytes(), b"synthetic sentinel")

    def test_regular_hardlink_rejected_before_any_removal(self):
        original = self.file("a")
        os.link(original, self.sdk / "b")
        with patch.object(m.os, "unlink") as unlink:
            self.rejected(lambda: m.census(self.fd), "tree_rejected")
        unlink.assert_not_called()
        self.assertEqual(original.stat().st_nlink, 2)

    def test_special_fifo_rejected_before_any_removal(self):
        os.mkfifo(self.sdk / "fifo")
        with patch.object(m.os, "unlink") as unlink:
            self.rejected(lambda: m.census(self.fd), "tree_rejected")
        unlink.assert_not_called()

    def test_cross_device_node_rejected(self):
        self.file("a")
        original = m.os.stat

        def different(*args, **kwargs):
            info = namespace(original(*args, **kwargs))
            info.st_dev += 1
            return info

        with patch.object(m.os, "stat", side_effect=different):
            self.rejected(lambda: m.census(self.fd), "tree_rejected")

    def test_added_entry_rejected_before_deletion(self):
        self.file("a")
        tree = m.census(self.fd)
        self.file("new")
        result = m.empty_report()
        self.rejected(lambda: m.remove_tree(self.fd, tree, result), "identity_changed")
        self.assertEqual(result["removed_leaves"], 0)
        self.assertTrue((self.sdk / "a").exists())

    def test_changed_file_state_rejected(self):
        target = self.file("a")
        tree = m.census(self.fd)
        target.write_bytes(b"different synthetic size")
        self.rejected(lambda: m.remove_tree(self.fd, tree, m.empty_report()),
                      "identity_changed")
        self.assertTrue(target.exists())

    def test_changed_owner_or_mode_rejected(self):
        self.file("a")
        tree = m.census(self.fd)
        original = m.os.stat
        for field in ("st_uid", "st_gid", "st_mode"):
            def changed(*args, selected=field, **kwargs):
                info = namespace(original(*args, **kwargs))
                setattr(info, selected, getattr(info, selected) ^ 1)
                return info

            with self.subTest(field=field), patch.object(m.os, "stat", side_effect=changed):
                self.rejected(lambda: m.remove_tree(self.fd, tree, m.empty_report()),
                              "identity_changed")
        self.assertTrue((self.sdk / "a").exists())

    def test_changed_directory_identity_rejected(self):
        self.file("nested/a")
        tree = m.census(self.fd)
        (self.sdk / "nested").rename(self.parent / "prior-nested")
        (self.sdk / "nested").mkdir()
        self.rejected(lambda: m.remove_tree(self.fd, tree, m.empty_report()),
                      "identity_changed")

    def test_directory_replaced_between_stat_and_open_rejected(self):
        self.file("nested/a")
        actual_fstat = m.os.fstat

        def changed(fd):
            result = namespace(actual_fstat(fd))
            if fd != self.fd:
                result.st_ino += 1
            return result

        with patch.object(m.os, "fstat", side_effect=changed):
            self.rejected(lambda: m.census(self.fd), "identity_changed")

    def test_directory_entries_change_during_census_rejected(self):
        self.file("a")
        with patch.object(m, "names", side_effect=[["a"], ["a", "new"]]):
            self.rejected(lambda: m.census(self.fd), "identity_changed")

    def test_node_bound(self):
        self.file("a")
        self.file("b")
        with patch.object(m, "MAXIMUM_NODES", 1):
            self.rejected(lambda: m.census(self.fd), "budget_exhausted")

    def test_name_storage_bound(self):
        self.file("long-name")
        with patch.object(m, "MAXIMUM_NAME_BYTES", 2):
            self.rejected(lambda: m.census(self.fd), "budget_exhausted")

    def test_depth_bound(self):
        self.file("one/two/a")
        with patch.object(m, "MAXIMUM_DEPTH", 2):
            self.rejected(lambda: m.census(self.fd), "budget_exhausted")

    def test_allocated_block_bound(self):
        self.file("a")
        original = m.os.stat

        def allocated(*args, **kwargs):
            info = namespace(original(*args, **kwargs))
            info.st_blocks = 2
            return info

        with patch.object(m.os, "stat", side_effect=allocated), \
                patch.object(m, "MAXIMUM_ALLOCATED_BYTES", 512):
            self.rejected(lambda: m.census(self.fd), "budget_exhausted")

    def test_partial_failure_preserves_counts(self):
        self.file("a")
        self.file("b")
        tree = m.census(self.fd)
        result = m.empty_report()
        original = m.os.unlink

        def fail_second(name, **kwargs):
            if name == "b":
                raise OSError("synthetic detail must not be retained")
            return original(name, **kwargs)

        with patch.object(m.os, "unlink", side_effect=fail_second), self.assertRaises(OSError):
            m.remove_tree(self.fd, tree, result)
        self.assertEqual(result["removed_leaves"], 1)
        self.assertFalse((self.sdk / "a").exists())
        self.assertTrue((self.sdk / "b").exists())


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(m, "os", namespace(os)))

    def target_setup(self):
        info = types.SimpleNamespace(st_dev=5, st_ino=10, st_mode=stat.S_IFDIR | 0o755,
                                     st_uid=0, st_gid=0)
        self.enterContext(patch.object(m.os, "open", side_effect=range(20, 26)))
        self.stat = self.enterContext(patch.object(m.os, "stat", return_value=info))
        self.fstat = self.enterContext(patch.object(m.os, "fstat", return_value=info))
        self.close = self.enterContext(patch.object(m.os, "close"))
        self.mounts = self.enterContext(patch.object(m, "check_mounts"))
        return info

    def test_exact_fixed_target_and_ancestor_descriptor_walk(self):
        self.target_setup()
        handles, _bound = m.open_target()
        self.assertEqual(m.TARGET, "/usr/local/lib/android/sdk")
        self.assertEqual(m.PARTS, ("usr", "local", "lib", "android", "sdk"))
        self.assertEqual(handles, list(range(20, 26)))
        calls = m.os.open.call_args_list
        self.assertEqual(calls[0].args[0], "/")
        for index, component in enumerate(m.PARTS, 1):
            self.assertEqual(calls[index].args[0], component)
            self.assertEqual(calls[index].kwargs, {"dir_fd": handles[index - 1]})
            self.assertTrue(calls[index].args[1] & os.O_NOFOLLOW)
        self.mounts.assert_called_once_with()

    def test_symlink_ancestor_rejected_and_descriptors_closed(self):
        info = self.target_setup()
        altered = copy.copy(info)
        altered.st_mode = stat.S_IFLNK | 0o777
        self.stat.return_value = altered
        with self.assertRaises(m.ReclaimError) as error:
            m.open_target()
        self.assertEqual(error.exception.code, "target_rejected")
        self.close.assert_called_once_with(20)

    def test_open_identity_race_rejected_and_descriptors_closed(self):
        info = self.target_setup()
        changed = copy.copy(info)
        changed.st_ino += 1
        self.fstat.return_value = changed
        with self.assertRaises(m.ReclaimError) as error:
            m.open_target()
        self.assertEqual(error.exception.code, "identity_changed")
        self.assertEqual([call.args[0] for call in self.close.call_args_list], [21, 20])

    def test_target_filesystem_must_match_temporary_filesystem(self):
        info = self.target_setup()
        other = copy.copy(info)
        other.st_dev += 1
        self.stat.side_effect = lambda path, **_kwargs: other if path == "/tmp" else info
        with self.assertRaises(m.ReclaimError) as error:
            m.open_target()
        self.assertEqual(error.exception.code, "target_rejected")
        self.assertEqual(self.close.call_count, 6)

    def test_same_device_mount_at_root_or_below_rejected(self):
        for mount in (m.TARGET, m.TARGET + "/nested"):
            with self.subTest(mount=mount), patch("builtins.open", mock_open(
                    read_data=f"1 2 0:1 / {mount} rw - ext4 /dev/root rw\n".encode())):
                with self.assertRaises(m.ReclaimError) as error:
                    m.check_mounts()
                self.assertEqual(error.exception.code, "target_rejected")

    def test_unrelated_mount_prefix_is_not_target(self):
        raw = b"1 2 0:1 / /usr/local/lib/android/sdk-other rw - ext4 /dev/root rw\n"
        with patch("builtins.open", mock_open(read_data=raw)):
            m.check_mounts()

    def test_malformed_or_unbounded_mount_table_rejected(self):
        for raw in (b"", b"invalid\n", b"x" * (1024**2 + 1)):
            with self.subTest(size=len(raw)), patch("builtins.open", mock_open(read_data=raw)):
                with self.assertRaises(m.ReclaimError):
                    m.check_mounts()

    def test_probe_source_hash_is_bound(self):
        self.assertEqual(hashlib.sha256((ROOT / m.PROBE).read_bytes()).hexdigest(),
                         m.PROBE_SHA256)

    def test_synthetic_probe_definitions_require_exact_hash(self):
        directory = self.enterContext(tempfile.TemporaryDirectory(prefix="sdk-probe-unit-"))
        root = Path(directory).resolve()
        source = root / m.PROBE
        source.parent.mkdir()
        raw = b"SYNTHETIC_DEFINITION_ONLY = True\n"
        source.write_bytes(raw)
        source.chmod(0o600)
        with patch.object(m, "ROOT", root):
            with self.assertRaises(m.ReclaimError) as error:
                m.load_probe()
            self.assertEqual(error.exception.code, "binding_rejected")
            with patch.object(m, "PROBE_SHA256", hashlib.sha256(raw).hexdigest()):
                loaded = m.load_probe()
                self.assertTrue(loaded.SYNTHETIC_DEFINITION_ONLY)
        self.assertNotIn("_lumi_sdk_capacity_probe_v1", sys.modules)

    def test_probe_symlink_hardlink_or_writable_code_rejected(self):
        directory = self.enterContext(tempfile.TemporaryDirectory(prefix="sdk-probe-unit-"))
        root = Path(directory).resolve()
        source = root / m.PROBE
        source.parent.mkdir()
        original = root / "synthetic.py"
        raw = b"SYNTHETIC_DEFINITION_ONLY = True\n"
        original.write_bytes(raw)
        with patch.object(m, "ROOT", root), \
                patch.object(m, "PROBE_SHA256", hashlib.sha256(raw).hexdigest()):
            source.symlink_to(original)
            with self.assertRaises(m.ReclaimError):
                m.load_probe()
            source.unlink()
            os.link(original, source)
            with self.assertRaises(m.ReclaimError):
                m.load_probe()
            source.unlink()
            source.write_bytes(raw)
            source.chmod(0o666)
            with self.assertRaises(m.ReclaimError):
                m.load_probe()


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory(prefix="sdk-run-unit-"))
        self.parent = Path(directory).resolve()
        self.sdk = self.parent / "sdk"
        self.sdk.mkdir()
        (self.sdk / "a").write_bytes(b"synthetic")
        (self.sdk / "b").write_bytes(b"synthetic")
        self.enterContext(patch.object(m, "os", namespace(os)))
        self.enterContext(patch.object(m, "_ATTEMPTED", False))
        self.runtime = self.enterContext(patch.object(m, "runtime"))
        self.arm = self.enterContext(patch.object(m, "arm_deadline"))
        self.disarm = self.enterContext(patch.object(m, "disarm_deadline"))
        self.mounts = self.enterContext(patch.object(m, "check_mounts"))
        self.probe = types.SimpleNamespace(
            runtime=Mock(), install_guard=Mock(), capacity=p.capacity,
            validate_report=p.validate_report, DISK_INTS=p.DISK_INTS, DISK_BOOLS=p.DISK_BOOLS,
            run=Mock(return_value=observed_setup()))
        self.loader = self.enterContext(patch.object(m, "load_probe", return_value=self.probe))
        self.enterContext(patch.object(m.os, "statvfs", return_value=space()))
        self.opened = []

        def fresh_owned_target():
            parent = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY)
            sdk = os.open(self.sdk, os.O_RDONLY | os.O_DIRECTORY)
            self.opened += [parent, sdk]
            return [parent, sdk], m.identity(os.fstat(sdk))

        self.target = self.enterContext(patch.object(m, "open_target",
                                                     side_effect=fresh_owned_target))

    def assert_closed(self):
        for fd in self.opened:
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_success_removes_only_owned_sdk_and_probes_once(self):
        sentinel = self.parent / "untouched"
        sentinel.write_bytes(b"synthetic")
        result = m.run()
        self.assertEqual(result["code"], "ok")
        self.assertTrue(result["sdk_root_absence_verified"])
        self.assertTrue(result["ready_for_unchanged_setup"])
        self.assertFalse(self.sdk.exists())
        self.assertEqual(sentinel.read_bytes(), b"synthetic")
        self.probe.run.assert_called_once_with()
        self.probe.install_guard.assert_called_once_with()
        self.arm.assert_called_once_with()
        self.disarm.assert_called_once_with()
        self.assert_closed()
        self.assertTrue(all(result[name] is False for name in m.FALSE_FLAGS))

    def test_low_disk_remains_observed_not_ready(self):
        self.probe.run.return_value = observed_setup(13)
        result = m.run()
        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["code"], "capacity_insufficient")
        self.assertFalse(result["ready_for_unchanged_setup"])

    def test_low_capacity_after_removal_skips_setup_probe(self):
        with patch.object(m.os, "statvfs", return_value=space(13)):
            result = m.run()
        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["code"], "capacity_insufficient")
        self.assertFalse(result["ready_for_unchanged_setup"])
        self.assertIsNone(result["setup_report"])
        self.assertTrue(result["sdk_root_absence_verified"])
        self.probe.run.assert_not_called()
        self.assert_closed()
        forged = copy.deepcopy(result)
        forged["capacity_after"] = p.capacity(space())
        with self.assertRaises(m.ReclaimError):
            m.validate_report(forged, self.probe)

    def test_malformed_setup_result_is_not_retained(self):
        self.probe.run.return_value = {"private_exception_text": "synthetic private detail"}
        result = m.run()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["code"], "operation_failed")
        self.assertIsNone(result["setup_report"])
        self.assertFalse(result["ready_for_unchanged_setup"])
        self.assertNotIn(b"private", m.canonical(result))

    def test_partial_failure_no_probe_no_retry_and_no_exception_text(self):
        actual = m.os.unlink

        def fail_second(name, **kwargs):
            if name == "b":
                raise OSError("PRIVATE SYNTHETIC ERROR")
            actual(name, **kwargs)

        with patch.object(m.os, "unlink", side_effect=fail_second):
            result = m.run()
        self.assertEqual(result["code"], "operation_failed")
        self.assertEqual(result["removed_leaves"], 1)
        self.assertTrue(result["removal_started"])
        self.assertFalse(result["sdk_root_removed"])
        self.assertNotIn(b"PRIVATE", m.canonical(result))
        self.probe.run.assert_not_called()
        self.assertEqual(m.run()["code"], "already_attempted")
        self.assertEqual(self.target.call_count, 1)
        self.assert_closed()

    def test_census_rejection_does_not_begin_removal(self):
        with patch.object(m, "census", side_effect=m.ReclaimError("tree_rejected")):
            result = m.run()
        self.assertEqual(result["stage"], "census")
        self.assertFalse(result["removal_started"])
        self.assertTrue(self.sdk.exists())
        self.probe.run.assert_not_called()
        self.assert_closed()

    def test_mount_change_before_removal_stops_without_mutation(self):
        self.mounts.side_effect = m.ReclaimError("target_rejected")
        result = m.run()
        self.assertEqual(result["code"], "target_rejected")
        self.assertFalse(result["removal_started"])
        self.assertTrue((self.sdk / "a").exists())

    def test_empty_target_rejected_without_deletion(self):
        (self.sdk / "a").unlink()
        (self.sdk / "b").unlink()
        result = m.run()
        self.assertEqual(result["code"], "tree_rejected")
        self.assertFalse(result["removal_started"])
        self.assertTrue(self.sdk.is_dir())

    def test_deadline_failure_is_recorded_and_not_retried(self):
        with patch.object(m, "remove_tree", side_effect=m.ReclaimError("deadline_exhausted")):
            result = m.run()
        self.assertEqual(result["code"], "deadline_exhausted")
        self.assertTrue(result["removal_started"])
        self.assertEqual(m.run()["code"], "already_attempted")
        self.probe.run.assert_not_called()

    def test_probe_setup_failure_does_not_claim_readiness(self):
        self.probe.run.return_value = p.empty_report()
        result = m.run()
        self.assertEqual(result["stage"], "setup_probe")
        self.assertEqual(result["code"], "setup_failed")
        self.assertFalse(result["ready_for_unchanged_setup"])
        self.assertTrue(result["sdk_root_absence_verified"])

    def test_target_binding_failure_retires_deadline_and_never_censuses(self):
        self.target.side_effect = m.ReclaimError("target_rejected")
        with patch.object(m, "census") as census:
            result = m.run()
        self.assertEqual(result["stage"], "target_binding")
        census.assert_not_called()
        self.disarm.assert_called_once_with()

    def test_runtime_rejection_before_source_binding(self):
        self.runtime.side_effect = m.ReclaimError("runtime_rejected")
        result = m.run()
        self.assertEqual(result["stage"], "runtime")
        self.loader.assert_not_called()
        self.target.assert_not_called()
        self.arm.assert_not_called()


class ReportAndRuntimeTests(unittest.TestCase):
    def test_mac_rejected_without_filesystem_or_signal_actions(self):
        with patch.object(m, "sys", namespace(sys)), patch.object(m, "os", namespace(os)):
            m.sys.platform = "darwin"
            with patch.object(m.os, "open") as opening:
                with self.assertRaises(m.ReclaimError) as error:
                    m.runtime()
                self.assertEqual(error.exception.code, "runtime_rejected")
                opening.assert_not_called()

    def test_unknown_error_detail_becomes_fixed_code(self):
        self.assertEqual(str(m.ReclaimError("private detail")), "operation_failed")
        with self.assertRaises(m.ReclaimError) as error:
            m.expired(None, None)
        self.assertEqual(error.exception.code, "deadline_exhausted")

    def test_report_rejects_extra_keys_raw_details_and_boolean_counts(self):
        for change in ({"raw_text": "private"}, {"code": "private detail"},
                       {"census_nodes": True}, {"stage": "unknown"},
                       {"removed_leaves": 1}, {"sdk_root_absence_verified": True},
                       {"ready_for_unchanged_setup": True}):
            with self.subTest(keys=list(change)):
                result = m.empty_report()
                result.update(change)
                with self.assertRaises(m.ReclaimError):
                    m.validate_report(result)

    def test_all_nonclaim_flags_must_remain_false(self):
        for flag in m.FALSE_FLAGS:
            with self.subTest(flag=flag):
                result = m.empty_report()
                result[flag] = True
                with self.assertRaises(m.ReclaimError):
                    m.validate_report(result)

    def test_tampered_capacity_rejected(self):
        result = m.empty_report()
        result["capacity_before"] = p.capacity(space())
        result["capacity_before"]["acquisition_headroom_met"] = False
        with self.assertRaises(m.ReclaimError):
            m.validate_report(result, p)

    def test_wrong_cli_never_enters_run(self):
        for args in ([], ["--help"], ["/usr/local/lib/android/sdk"],
                     ["--fresh-sdk-capacity-check", "extra"]):
            with self.subTest(args=args), patch.object(m, "sys", namespace(sys)), \
                    patch.object(m, "os", namespace(os)), patch.object(m, "run") as run:
                m.sys.argv = ["synthetic-script", *args]
                with patch.object(m.os, "write", side_effect=lambda _fd, raw: len(raw)) as write:
                    self.assertEqual(m.main(), 2)
                run.assert_not_called()
                result = json.loads(write.call_args.args[1])
                self.assertEqual(result["code"], "runtime_rejected")

    def test_cli_short_report_write_rejected(self):
        with patch.object(m, "sys", namespace(sys)), patch.object(m, "os", namespace(os)):
            m.sys.argv = ["synthetic-script", "wrong"]
            with patch.object(m.os, "write", return_value=0), self.assertRaises(m.ReclaimError):
                m.main()


class WorkflowContractTests(unittest.TestCase):
    def setUp(self):
        self.workflow = (ROOT / ".github/workflows/sft-sdk-capacity-preflight-v1.yml").read_text()

    def test_one_branch_one_path_push_only(self):
        trigger = self.workflow.split("\non:\n", 1)[1].split("\npermissions:\n", 1)[0]
        self.assertEqual(trigger, "  push:\n    branches:\n"
                         "      - codex/lumi-posttraining-product-v0.1\n    paths:\n"
                         "      - .github/workflows/sft-sdk-capacity-preflight-v1.yml\n")

    def test_private_repository_parent_and_attempt_are_bound(self):
        conditions = re.findall(r"^    if: (.+)$", self.workflow, re.MULTILINE)
        self.assertEqual(conditions, [
            "github.event.repository.private == true && github.repository_id == '1300831104'"
            " && github.run_attempt == 1 && github.event.before == "
            "'716774386cc3eba22f860c7699d6a33654882a02'"
            " && github.event.after == github.sha"])

    def test_one_bounded_cpu_job(self):
        jobs = self.workflow.split("\njobs:\n", 1)[1]
        self.assertEqual(re.findall(r"^  ([a-zA-Z0-9_-]+):$", jobs, re.MULTILINE),
                         ["sdk-capacity"])
        self.assertIn("    runs-on: ubuntu-24.04\n    timeout-minutes: 10\n", jobs)
        self.assertIn("  cancel-in-progress: false\n", self.workflow)
        self.assertEqual(re.findall(r"^        timeout-minutes: (\d+)$", jobs, re.MULTILINE),
                         ["1", "1", "1", "6", "1"])

    def test_exact_cleanup_and_frozen_probe_hashes(self):
        pins = re.findall(r"^          ([0-9a-f]{64})  (\S+)$", self.workflow, re.MULTILINE)
        self.assertEqual({relative for _digest, relative in pins}, {
            "scripts/reclaim_sft_android_sdk_v1.py", "scripts/probe_sft_prepared_setup_v1.py"})
        self.assertEqual(len(pins), 2)
        for digest, relative in pins:
            with self.subTest(relative=relative):
                self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest)
        self.assertIn("sha256sum --check --strict <<'LUMI_PINS'", self.workflow)

    def test_no_write_permission_or_persistent_checkout_credentials(self):
        permissions = self.workflow.split("\npermissions:\n", 1)[1].split(
            "\nconcurrency:\n", 1)[0]
        self.assertEqual(permissions, "  contents: read\n")
        self.assertIn("          persist-credentials: false\n          fetch-depth: 2\n",
                      self.workflow)

    def test_expected_synthetic_suites_only(self):
        selected = re.findall(r"unittest discover -q -s tests -p (\S+)", self.workflow)
        self.assertEqual(selected, ["test_sft_sdk_capacity_preflight_v1.py",
                                    "test_sft_prepared_setup_preflight_v1.py",
                                    "test_sft_checkpoint_prepared_qualification_v2.py"])
        self.assertEqual(self.workflow.count("/usr/bin/python3 -I -S -B -m unittest"), 3)

    def test_single_fresh_entry_with_clean_environment_and_external_timeout(self):
        command = "/usr/bin/python3 -I -S -B scripts/reclaim_sft_android_sdk_v1.py"
        self.assertEqual(self.workflow.count(command), 1)
        self.assertIn(command + " --fresh-sdk-capacity-check", self.workflow)
        self.assertEqual(self.workflow.count("sudo -n /usr/bin/env -i"), 1)
        self.assertIn("timeout --signal=TERM --kill-after=5s 330s", self.workflow)
        self.assertIn("PATH=/usr/bin:/bin LC_ALL=C.UTF-8", self.workflow)
        self.assertIn("LUMI_SDK_CAPACITY_CI=1 LUMI_PREPARED_SETUP_PREFLIGHT_CI=1", self.workflow)
        self.assertIn("LUMI_CHECKPOINT_GUARD_QUALIFICATION_CI=1", self.workflow)
        self.assertIn("set -euC\n          umask 077", self.workflow)
        self.assertIn("mkdir artifacts/lumi-sft-sdk-capacity-preflight-v0.1\n", self.workflow)

    def test_only_closed_aggregate_uploaded_with_bounded_retention(self):
        target = "artifacts/lumi-sft-sdk-capacity-preflight-v0.1/aggregate-capacity.json"
        self.assertIn("> " + target + " 2>/dev/null", self.workflow)
        self.assertEqual(re.findall(r"^          path: (.+)$", self.workflow, re.MULTILINE),
                         [target])
        self.assertIn("        if: always()", self.workflow)
        self.assertIn("          if-no-files-found: error\n          retention-days: 14\n",
                      self.workflow)
        self.assertEqual(re.findall(r"uses: (\S+)", self.workflow),
                         ["actions/checkout@v6", "actions/upload-artifact@v4"])

    def test_offline_no_gpu_no_extra_acquisition_or_deletion_commands(self):
        for setting in ('CUDA_VISIBLE_DEVICES: ""', 'HF_HUB_OFFLINE: "1"',
                        'TRANSFORMERS_OFFLINE: "1"', 'AWS_EC2_METADATA_DISABLED: "true"'):
            self.assertIn(setting, self.workflow)
        forbidden = (r"\brm\s", r"\bcurl\s", r"\bwget\s", r"\bdocker\s",
                     r"\bgh\s", r"\baws\s", r"\bpip\s", r"\bsleep\s",
                     r"\bwhile\s", r"\bfor\s", r"\bgit push\b", r"\bgit clean\b")
        for pattern in forbidden:
            with self.subTest(pattern=pattern):
                self.assertNotRegex(self.workflow, pattern)


if __name__ == "__main__":
    unittest.main()
