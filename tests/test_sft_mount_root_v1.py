"""Syscall doubles only; never mount, unshare, pivot, or inspect live procfs."""

from __future__ import annotations

import ctypes
import importlib
import os
import stat
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_synthetic_mount_root_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
mount = importlib.import_module(PACKAGE + ".posttraining_sft_mount_root_v1")
ADDITIONS = set(sys.modules) - before_import
OPTIONS = {
    "root_path": "/tmp/lumi-sft-root-" + "a" * 32,
    "runtime_root": "/opt/hostedtoolcache/Python/3.12.9/x64",
    "source_root": "/home/runner/work/example/example/src",
    "script_path": "/home/runner/work/example/example/scripts/"
                   "run_sft_worker_run_torch_fixture_v1.py",
    "uid": 1001, "gid": 1001, "scratch_bytes": 8 * 1024**2,
}


def info(device=1, inode=2, mode=stat.S_IFDIR | 0o700, uid=0, gid=0):
    return types.SimpleNamespace(st_dev=device, st_ino=inode, st_mode=mode,
                                 st_uid=uid, st_gid=gid)


class MountRootTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(mount, "_ATTEMPT_OWNER", None))
        self.runtime = self.enterContext(patch.object(mount.netns, "_runtime"))
        self.next_fd, self.root_opens, self.ns_opens = 10, 0, 0
        self.metadata, self.closed, self.events = {}, [], []
        self.pin_records = []
        self.old_root = info()
        self.new_root = info(2, 1, stat.S_IFDIR | 0o755)
        self.parent = info(mode=stat.S_IFDIR | 0o1777)
        self.os = types.SimpleNamespace(
            **{name: getattr(os, name) for name in (
                "O_RDONLY", "O_WRONLY", "O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC",
                "O_NONBLOCK", "O_CREAT", "O_EXCL",
            )}, getpid=Mock(return_value=700), open=Mock(side_effect=self.open),
            fstat=Mock(side_effect=lambda fd: self.metadata[fd]),
            close=Mock(side_effect=self.close), listdir=Mock(return_value=[]),
            readlink=Mock(side_effect=lambda path: mount._LINKS[path[1:]]),
            mkdir=Mock(), chmod=Mock(), symlink=Mock(), mknod=Mock(),
            makedev=lambda major, minor: (major << 8) | minor,
            chdir=Mock(), rmdir=Mock(), statvfs=Mock(side_effect=self.statvfs),
        )
        self.enterContext(patch.object(mount, "os", self.os))
        self.pin = self.enterContext(patch.object(mount, "_pin", side_effect=self.pin_path))
        self.namespace = self.enterContext(patch.object(
            mount, "_open_namespace", side_effect=self.open_namespace,
        ))
        self.identity = self.enterContext(patch.object(
            mount, "_namespace_identity", side_effect=lambda fd: (42, fd),
        ))
        self.unshare = self.enterContext(patch.object(
            mount, "_unshare", side_effect=lambda: self.events.append("unshare"),
        ))
        self.mount = self.enterContext(patch.object(
            mount, "_mount", side_effect=self.mount_call,
        ))
        self.seal = self.enterContext(patch.object(
            mount, "_seal", side_effect=lambda *args, **kw: self.events.append(("seal", args, kw)),
        ))
        self.pivot = self.enterContext(patch.object(
            mount, "_pivot", side_effect=lambda: self.events.append("pivot"),
        ))
        self.filesystem = self.enterContext(patch.object(mount, "_filesystem"))
        self.native = self.enterContext(patch.object(
            mount.ctypes, "CDLL", side_effect=AssertionError("no_live_native"),
        ))

    def tearDown(self):
        self.assertFalse(mount._SETUP_LOCK.locked())
        self.assertEqual(len(self.closed), len(set(self.closed)))
        self.native.assert_not_called()

    def allocate(self, value):
        fd, self.next_fd = self.next_fd, self.next_fd + 1
        self.metadata[fd] = value
        return fd

    def pin_path(self, path, *, directory, owned):
        if path == OPTIONS["root_path"]:
            value = self.old_root if self.root_opens == 0 else self.new_root
            self.root_opens += 1
        elif path == "/tmp":
            value = self.parent
        else:
            value = info(mode=(stat.S_IFDIR if directory else stat.S_IFREG) | 0o755)
        fd = self.allocate(value)
        owned.append(fd)
        self.pin_records.append((path, fd, "unshare" in self.events))
        return fd

    def mount_call(self, *args, **kwargs):
        if args[3] & mount.MS_BIND:
            fd = int(args[0].rsplit("/", 1)[1])
            matching = [record for record in self.pin_records if record[1] == fd]
            self.assertEqual(len(matching), 1)
            path, _, in_new_namespace = matching[0]
            self.assertTrue(in_new_namespace, "old_namespace_bind_source")
            originals = [record[1] for record in self.pin_records
                         if record[0] == path and not record[2]]
            self.assertEqual(len(originals), 1)
            self.assertIn(originals[0], self.closed)
            self.assertIn(fd, self.metadata)
            self.assertEqual(kwargs, {"failure_code": "bind_mount_failed"})
        self.events.append(("mount", args))

    def open_namespace(self, owned):
        self.ns_opens += 1
        fd = self.allocate(info(inode=self.ns_opens))
        owned.append(fd)
        return fd

    def open(self, path, flags, *args, **kwargs):
        return self.allocate(self.old_root if path.startswith("/proc/self/fd/") else info())

    def close(self, fd):
        self.assertIn(fd, self.metadata)
        self.closed.append(fd)
        del self.metadata[fd]

    def statvfs(self, path):
        flags = 14 if path == "/tmp" else 11 if path == "/dev" else 15 if path == "/proc" else 7
        return types.SimpleNamespace(f_flag=flags, f_frsize=4096,
                                     f_blocks=OPTIONS["scratch_bytes"] // 4096)

    def run_setup(self, **changes):
        return mount.enter_mount_root(**(OPTIONS | changes))

    def error(self, code, **changes):
        with self.assertRaises(mount.MountRootError) as caught:
            self.run_setup(**changes)
        self.assertEqual(caught.exception.args, (code,))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_import_is_stdlib_only(self):
        self.assertFalse(any(name.split(".")[0] in mount.netns.ML_ROOTS for name in ADDITIONS))

    def test_success_exact_receipt_and_no_leaked_owned_fds(self):
        self.assertEqual(self.run_setup(), {
            "schema": mount.SCHEMA, "readonly_root": True, "readonly_binds": True,
            "old_root_detached": True, "private_propagation": True, "private_tmpfs": True,
            "scratch_capacity_bytes": 8 * 1024**2, "host_mounts_modified": False,
            "real_sft_execution_permitted": False, "production_eligible": False,
        })
        self.assertEqual(self.metadata, {})
        self.runtime.assert_called_once_with()

    def test_private_propagation_precedes_all_generated_mounts(self):
        self.run_setup()
        self.assertEqual(self.events[:2], ["unshare", ("mount", (
            None, "/", None, mount.MS_REC | mount.MS_PRIVATE,
        ))])
        self.assertEqual(self.events[-1], ("seal", ("/", 7), {"recursive": False}))
        self.assertLess(self.events.index("pivot"), len(self.events) - 1)

    def test_every_bind_repins_same_source_in_new_namespace(self):
        self.run_setup()
        binds = [item for item in self.mount.call_args_list if item.args[3] & mount.MS_BIND]
        self.assertEqual(len(binds), 10)
        self.assertEqual(len([item for item in self.pin_records
                              if item[0] == OPTIONS["source_root"]]), 2)
        self.assertEqual(self.metadata, {})

    def reject_changed_source(self, field):
        def changed(path, *, directory, owned):
            fd = self.pin_path(path, directory=directory, owned=owned)
            if path == "/usr" and "unshare" in self.events:
                value = self.metadata[fd]
                setattr(value, field, getattr(value, field) + 1)
            return fd
        self.pin.side_effect = changed
        self.error("source_rejected")
        self.mount.assert_called_once_with(None, "/", None, mount.MS_REC | mount.MS_PRIVATE)
        self.pivot.assert_not_called()
        self.assertEqual(self.metadata, {})
        self.error("already_attempted")

    def test_repin_changed_device_denied_before_root_construction(self):
        self.reject_changed_source("st_dev")

    def test_repin_changed_inode_denied_before_root_construction(self):
        self.reject_changed_source("st_ino")

    def test_repin_changed_mode_denied_before_root_construction(self):
        self.reject_changed_source("st_mode")

    def test_repin_changed_owner_denied_before_root_construction(self):
        self.reject_changed_source("st_uid")

    def test_repin_changed_group_denied_before_root_construction(self):
        self.reject_changed_source("st_gid")

    def test_repin_open_failure_closes_both_namespaces_without_retry(self):
        def failed(path, *, directory, owned):
            if path == "/usr" and "unshare" in self.events:
                raise OSError("PRIVATE")
            return self.pin_path(path, directory=directory, owned=owned)
        self.pin.side_effect = failed
        self.error("setup_failed")
        self.assertEqual(self.metadata, {})
        self.pivot.assert_not_called()
        self.error("already_attempted")

    def test_repin_original_close_failure_is_not_closed_twice(self):
        def failed(fd):
            original = any(path == "/usr" and pinned == fd and not fresh
                           for path, pinned, fresh in self.pin_records)
            self.close(fd)
            if original:
                raise OSError("PRIVATE")
        self.os.close.side_effect = failed
        self.error("close_failed")
        self.assertEqual(self.metadata, {})
        self.pivot.assert_not_called()

    def test_scaffold_chmod_overrides_restrictive_umask_only_on_new_dirs(self):
        self.run_setup()
        created = [item.args[0] for item in self.os.mkdir.call_args_list]
        self.assertTrue(created)
        self.assertTrue(all(path.startswith(OPTIONS["root_path"] + "/") for path in created))
        for path in created:
            expected = 0o1777 if path.endswith("/tmp/shm") else 0o755
            self.assertIn(call(path, expected), self.os.chmod.call_args_list)

    def test_devices_sealed_without_nodev_and_tmp_size_fixed(self):
        self.run_setup()
        self.assertIn(call(OPTIONS["root_path"] + "/dev", 11, recursive=False),
                      self.seal.call_args_list)
        self.assertIn(call("tmpfs", OPTIONS["root_path"] + "/tmp", "tmpfs", 14,
                           "size=8388608,nr_inodes=16384,mode=1777"), self.mount.call_args_list)
        self.assertEqual(self.os.mknod.call_count, 4)

    def test_bad_path_and_bool_arguments_have_no_side_effects(self):
        for field, values in {
            "root_path": ("/", "/tmp/../bad", "/tmp/not-owned"),
            "runtime_root": ("/usr", "/opt/hostedtoolcache/Python/3.13.1/x64"),
            "source_root": ("/tmp/foo/src", "/usr/src", "/bad//src"),
            "script_path": ("/etc/passwd",), "uid": (0, True, 2**31),
            "gid": (-1, False), "scratch_bytes": (True, 1, 8 * 1024**2 + 1),
        }.items():
            for value in values:
                self.error("input_rejected", **{field: value})
        self.pin.assert_not_called()
        self.unshare.assert_not_called()

    def test_runtime_rejection_precedes_pin_and_mutation(self):
        self.runtime.side_effect = OSError("PRIVATE")
        self.error("runtime_rejected")
        self.pin.assert_not_called()

    def test_second_success_and_fork_reuse_are_denied(self):
        self.run_setup()
        count = self.pin.call_count
        self.error("already_attempted")
        self.os.getpid.return_value = 701
        self.error("owner_mismatch")
        self.assertEqual(self.pin.call_count, count)

    def test_failed_mount_is_not_retried_or_rolled_back(self):
        self.mount.side_effect = OSError("PRIVATE")
        self.error("setup_failed")
        self.error("already_attempted")
        self.mount.assert_called_once()
        self.pivot.assert_not_called()
        self.assertEqual(self.metadata, {})

    def test_parent_must_be_root_owned_and_sticky(self):
        self.parent.st_mode = stat.S_IFDIR | 0o777
        self.error("root_rejected")
        self.unshare.assert_not_called()

    def test_existing_root_must_be_empty(self):
        self.os.listdir.return_value = ["owned-by-someone-else"]
        self.error("root_rejected")
        self.unshare.assert_not_called()

    def test_existing_root_mode_must_be_private(self):
        self.old_root.st_mode = stat.S_IFDIR | 0o755
        self.error("root_rejected")

    def test_host_merged_usr_link_must_match(self):
        self.os.readlink.side_effect = None
        self.os.readlink.return_value = "unexpected"
        self.error("source_rejected")
        self.unshare.assert_not_called()

    def test_unchanged_namespace_rejected_before_mount(self):
        self.identity.side_effect = lambda fd: (42, 1)
        self.error("namespace_unchanged")
        self.mount.assert_not_called()

    def test_pre_mount_root_fd_is_not_accepted_as_tmpfs_root(self):
        self.new_root.st_dev, self.new_root.st_ino = self.old_root.st_dev, self.old_root.st_ino
        self.error("root_rejected")
        self.pivot.assert_not_called()

    def test_required_seal_failure_is_terminal(self):
        self.seal.side_effect = mount.MountRootError("seal_failed")
        self.error("seal_failed")
        self.pivot.assert_not_called()

    def test_pivot_failure_cannot_return_success(self):
        self.pivot.side_effect = mount.MountRootError("pivot_failed")
        self.error("pivot_failed")
        self.assertEqual(self.metadata, {})

    def test_writable_root_readback_denied(self):
        self.os.statvfs.side_effect = lambda path: types.SimpleNamespace(f_flag=0)
        self.error("readback_failed")

    def test_scratch_capacity_must_equal_bound(self):
        original = self.statvfs
        def wrong(path):
            value = original(path)
            value.f_blocks += 1
            return value
        self.os.statvfs.side_effect = wrong
        self.error("readback_failed")

    def test_cleanup_close_failure_is_once_and_overrides_success(self):
        original = self.close
        def fail_once(fd):
            original(fd)
            if fd == 10:
                raise OSError("PRIVATE")
        self.os.close.side_effect = fail_once
        self.error("close_failed")
        self.assertEqual(self.metadata, {})

    def test_error_values_are_allowlisted(self):
        for value in ("PRIVATE", None, True, {}):
            self.assertEqual(mount.MountRootError(value).args, ("setup_failed",))


class NativeSignatureTests(unittest.TestCase):
    def test_fd_walk_pins_no_symlink_components_and_closes_previous_only(self):
        fake = types.SimpleNamespace(
            **{name: getattr(os, name) for name in (
                "O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC",
            )}, open=Mock(side_effect=[10, 11, 12]), close=Mock(),
            fstat=Mock(side_effect=[info(mode=stat.S_IFDIR | 0o755),
                                   info(mode=stat.S_IFREG | 0o644)]),
        )
        flags = mount.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC
        owned = []
        with (patch.object(mount, "os", fake),
              patch.object(mount.fcntl, "fcntl", return_value=mount.fcntl.FD_CLOEXEC)):
            self.assertEqual(mount._pin("/safe/file", directory=False, owned=owned), 12)
        self.assertEqual(owned, [12])
        self.assertEqual(fake.open.call_args_list, [
            call("/", flags | os.O_DIRECTORY), call("safe", flags | os.O_DIRECTORY, dir_fd=10),
            call("file", flags, dir_fd=11),
        ])
        self.assertEqual(fake.close.call_args_list, [call(10), call(11)])

    def test_symlink_fd_is_rejected_with_cleanup_ownership_retained(self):
        fake = types.SimpleNamespace(
            **{name: getattr(os, name) for name in (
                "O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC",
            )}, open=Mock(side_effect=[10, 11]), close=Mock(),
            fstat=Mock(return_value=info(mode=stat.S_IFLNK | 0o777)),
        )
        owned = []
        with patch.object(mount, "os", fake), self.assertRaises(mount.MountRootError):
            mount._pin("/bad", directory=False, owned=owned)
        self.assertEqual(owned, [10, 11])
        fake.close.assert_not_called()

    def test_native_mount_and_unshare_have_exact_flags_and_signatures(self):
        library = types.SimpleNamespace(mount=Mock(return_value=0), unshare=Mock(return_value=0))
        with patch.object(mount.ctypes, "CDLL", return_value=library):
            mount._unshare()
            mount._mount(None, "/", None, mount.MS_REC | mount.MS_PRIVATE)
        library.unshare.assert_called_once_with(0x20000)
        library.mount.assert_called_once_with(None, b"/", None, 278528, None)
        self.assertEqual(library.unshare.argtypes, [ctypes.c_int])
        self.assertIs(library.mount.restype, ctypes.c_int)

    def test_native_seal_failure_is_fixed_code_without_fallback(self):
        library = types.SimpleNamespace(syscall=Mock(return_value=-1))
        with (patch.object(mount.ctypes, "CDLL", return_value=library),
              self.assertRaises(mount.MountRootError) as caught):
            mount._seal("/owned", 7, recursive=True)
        self.assertEqual(caught.exception.args, ("seal_failed",))
        library.syscall.assert_called_once()

    def test_native_bind_failure_has_fixed_stage_code(self):
        library = types.SimpleNamespace(mount=Mock(return_value=-1))
        with (patch.object(mount.ctypes, "CDLL", return_value=library),
              self.assertRaises(mount.MountRootError) as caught):
            mount._mount("/proc/self/fd/20", "/owned", None, mount.MS_BIND,
                         failure_code="bind_mount_failed")
        self.assertEqual(caught.exception.args, ("bind_mount_failed",))
        library.mount.assert_called_once()

    def test_native_unshare_failure_is_distinct_from_bind_failure(self):
        library = types.SimpleNamespace(unshare=Mock(return_value=-1))
        with (patch.object(mount.ctypes, "CDLL", return_value=library),
              self.assertRaises(mount.MountRootError) as caught):
            mount._unshare()
        self.assertEqual(caught.exception.args, ("namespace_unshare_failed",))
        library.unshare.assert_called_once()

    def test_mount_setattr_and_pivot_use_full_width_arguments(self):
        library = types.SimpleNamespace(syscall=Mock(return_value=0), umount2=Mock(return_value=0))
        with (patch.object(mount.ctypes, "CDLL", return_value=library),
              patch.object(mount, "os", types.SimpleNamespace(chdir=Mock(), rmdir=Mock())) as fake):
            mount._seal("/synthetic", 7, recursive=True)
            args = library.syscall.call_args.args
            self.assertEqual([args[0].value, args[1].value, args[2].value, args[3].value],
                             [442, -100, b"/synthetic", 0x8000])
            self.assertEqual(args[4]._obj.attr_set, 7)
            self.assertEqual(args[5].value, 32)
            mount._pivot()
            self.assertEqual(library.syscall.call_args.args[0].value, 155)
            library.umount2.assert_called_once_with(b"/.oldroot", 2)
            fake.chdir.assert_called_once_with("/")
            fake.rmdir.assert_called_once_with("/.oldroot")
        self.assertEqual(ctypes.sizeof(mount.MountAttr), 32)


if __name__ == "__main__":
    unittest.main()
