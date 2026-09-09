"""Synthetic syscall doubles and AST only; never inspect live procfs or mount."""

from __future__ import annotations

import ast
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
PACKAGE = "_synthetic_oci_mount_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
mount = importlib.import_module(PACKAGE + ".posttraining_sft_oci_mount_v1")
IMPORT_ADDITIONS = set(sys.modules) - before_import
OPTIONS = {
    "rootfs_path": "/tmp/lumi-private-" + "a" * 32 + "/rootfs",
    "mountpoint_path": "/tmp/lumi-private-" + "a" * 32 + "/mountpoint",
    "uid": 10001, "gid": 10001, "scratch_bytes": 536870912, "shm_bytes": 67108864,
}
PRIVATE = "synthetic-private-path-error"


def metadata(inode=2, mode=stat.S_IFDIR | 0o755, uid=0, gid=0, device=7, rdev=0):
    return types.SimpleNamespace(st_dev=device, st_ino=inode, st_mode=mode,
                                 st_uid=uid, st_gid=gid, st_rdev=rdev)


class OCIMountTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(mount, "_ATTEMPT_OWNER", None))
        self.runtime = self.enterContext(patch.object(mount.netns, "_runtime"))
        self.enterContext(patch.object(mount, "sys", types.SimpleNamespace(version_info=(3, 12))))
        self.events, self.fds, self.closed, self.pins = [], {}, [], []
        self.next_fd, self.namespace_generation, self.bound = 20, 0, False
        self.root_info = metadata()
        self.parent_info = metadata(inode=1, mode=stat.S_IFDIR | 0o700)
        self.target_info = metadata(inode=3, mode=stat.S_IFDIR | 0o700)
        self.scratch_bytes, self.shm_bytes = OPTIONS["scratch_bytes"], OPTIONS["shm_bytes"]
        self.os = types.SimpleNamespace(
            **{name: getattr(os, name) for name in (
                "O_RDONLY", "O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK",
            )}, getpid=Mock(return_value=4000), open=Mock(side_effect=self.open),
            fstat=Mock(side_effect=lambda fd: self.fds[fd]),
            close=Mock(side_effect=self.close), listdir=Mock(return_value=[]),
            mknod=Mock(), chmod=Mock(), symlink=Mock(), mkdir=Mock(),
            makedev=os.makedev, major=os.major, minor=os.minor,
            fchdir=Mock(side_effect=lambda fd: self.events.append(("fchdir", fd))),
            chdir=Mock(), stat=Mock(side_effect=self.stat),
        )
        self.enterContext(patch.object(mount, "os", self.os))
        self.pin = self.enterContext(patch.object(mount, "_pin", side_effect=self.pin_path))
        self.legacy_pin = self.enterContext(patch.object(
            mount.mount_root, "_pin", side_effect=self.legacy_pin_path,
        ))
        self.identity = self.enterContext(patch.object(
            mount.mount_root, "_source_identity", side_effect=self.identity_of,
        ))
        self.namespace = self.enterContext(patch.object(
            mount.mount_root, "_open_namespace", side_effect=self.open_namespace,
        ))
        self.namespace_identity = self.enterContext(patch.object(
            mount.mount_root, "_namespace_identity", side_effect=lambda fd: (8, fd),
        ))
        self.unshare = self.enterContext(patch.object(
            mount.mount_root, "_unshare", side_effect=self.unshare_call,
        ))
        self.mount_call = self.enterContext(patch.object(
            mount.mount_root, "_mount", side_effect=self.mounting,
        ))
        self.seal = self.enterContext(patch.object(
            mount.mount_root, "_seal", side_effect=lambda *a, **k: self.events.append(("seal", a)),
        ))
        self.pivot = self.enterContext(patch.object(
            mount, "_pivot", side_effect=lambda: self.events.append(("pivot",)),
        ))
        self.filesystem = self.enterContext(patch.object(mount.mount_root, "_filesystem"))
        self.readback = self.enterContext(patch.object(
            mount.mount_root, "_readback", side_effect=self.view,
        ))
        self.native = self.enterContext(patch.object(
            mount.ctypes, "CDLL", side_effect=AssertionError("no_native_execution"),
        ))

    def tearDown(self):
        self.assertFalse(mount._SETUP_LOCK.locked())
        self.assertEqual(self.closed, list(dict.fromkeys(self.closed)))
        self.native.assert_not_called()

    def allocate(self, value):
        descriptor, self.next_fd = self.next_fd, self.next_fd + 1
        self.fds[descriptor] = value
        return descriptor

    def identity_of(self, descriptor):
        value = self.fds[descriptor]
        return value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid

    def pin_path(self, path, owned):
        if path == OPTIONS["rootfs_path"] or path == "/":
            value = self.root_info
        elif path == OPTIONS["mountpoint_path"]:
            value = self.root_info if self.bound else self.target_info
        elif path == OPTIONS["rootfs_path"].rpartition("/")[0]:
            value = self.parent_info
        elif path in ("/tmp", "/dev/shm"):
            value = metadata(mode=stat.S_IFDIR | 0o1777, uid=10001, gid=10001)
        elif path == OPTIONS["rootfs_path"] + "/tmp":
            value = metadata(mode=stat.S_IFDIR | 0o1777)
        else:
            value = metadata()
        descriptor = self.allocate(value)
        owned.append(descriptor)
        self.pins.append((path, descriptor, self.namespace_generation))
        return descriptor

    def legacy_pin_path(self, path, *, directory, owned):
        self.assertFalse(directory)
        self.assertEqual(path, OPTIONS["rootfs_path"] + "/usr/local/bin/python3.12")
        descriptor = self.allocate(metadata(mode=stat.S_IFREG | 0o755))
        owned.append(descriptor)
        return descriptor

    def open(self, path, flags):
        self.assertTrue(path.startswith("/proc/self/fd/"))
        self.assertEqual(flags, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NONBLOCK)
        source = int(path.rpartition("/")[2])
        return self.allocate(self.fds[source])

    def close(self, descriptor):
        self.assertIn(descriptor, self.fds)
        self.assertNotIn(descriptor, self.closed)
        self.closed.append(descriptor)
        del self.fds[descriptor]

    def open_namespace(self, owned):
        descriptor = self.allocate(metadata(inode=self.namespace_generation + 20))
        owned.append(descriptor)
        return descriptor

    def unshare_call(self):
        self.namespace_generation += 1
        self.events.append(("unshare",))

    def mounting(self, *args, **kwargs):
        self.events.append(("mount", args, kwargs))
        if args[3] & mount.mount_root.MS_BIND:
            self.assertEqual(args[3], mount.mount_root.MS_BIND)
            source, target = (int(value.rpartition("/")[2]) for value in args[:2])
            for descriptor, path in ((source, OPTIONS["rootfs_path"]),
                                     (target, OPTIONS["mountpoint_path"])):
                self.assertIn((path, descriptor, 1), self.pins)
            self.assertEqual(kwargs, {"failure_code": "bind_mount_failed"})
            self.bound = True

    def view(self, path, required, forbidden=0):
        self.events.append(("readback", path, required, forbidden))
        size = self.shm_bytes if path == "/dev/shm" else self.scratch_bytes
        return types.SimpleNamespace(f_flag=required, f_frsize=4096, f_blocks=size // 4096)

    def stat(self, path, *, follow_symlinks):
        self.assertFalse(follow_symlinks)
        major, minor = mount.DEVICES[path.rpartition("/")[2]]
        return metadata(mode=stat.S_IFCHR | 0o666, rdev=os.makedev(major, minor))

    def run_setup(self, **changes):
        return mount.enter_oci_rootfs(**(OPTIONS | changes))

    def error(self, code, **changes):
        with self.assertRaises(mount.OCIMountError) as caught:
            self.run_setup(**changes)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.args, (code,))
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertNotIn(PRIVATE, str(caught.exception))

    def test_success_exact_receipt_and_owned_fds_closed(self):
        value = self.run_setup()
        self.assertEqual(value, {
            "schema": mount.SCHEMA, "rootfs_only_bind": True, "readonly_root": True,
            "private_propagation": True, "old_root_detached": True,
            "private_tmpfs": True, "private_shm_tmpfs": True, "fresh_procfs": True,
            "scratch_capacity_bytes": 536870912, "shm_capacity_bytes": 67108864,
            "host_software_binds": False, "host_mounts_modified": False,
            "rootfs_content_authenticated": False, "ambient_descriptors_closed": False,
            "complete_os_sandbox_verified": False, "worker_dispatched": False,
            "real_sft_execution_permitted": False, "production_eligible": False,
        })
        self.assertEqual(self.fds, {})
        self.runtime.assert_called_once_with()

    def test_only_one_nonrecursive_fresh_rootfs_bind(self):
        self.run_setup()
        binds = [item for item in self.mount_call.call_args_list
                 if item.args[3] & mount.mount_root.MS_BIND]
        self.assertEqual(len(binds), 1)
        self.assertEqual(binds[0].args[3], mount.mount_root.MS_BIND)
        self.assertEqual([item.args[0] for item in self.mount_call.call_args_list
                          if not item.args[3] & mount.mount_root.MS_BIND],
                         [None, "tmpfs", "tmpfs", "tmpfs", "proc"])

    def test_private_propagation_precedes_bind_and_root_seal_precedes_overlays(self):
        self.run_setup()
        self.assertEqual(self.events[:2], [
            ("unshare",), ("mount", (None, "/", None,
                                     mount.mount_root.MS_REC | mount.mount_root.MS_PRIVATE), {}),
        ])
        self.assertEqual(self.events[3], ("seal", (OPTIONS["mountpoint_path"], 7)))
        self.assertEqual(self.events[4][1][2], "tmpfs")

    def test_fchdir_pivot_then_readback(self):
        self.run_setup()
        index = self.events.index(("pivot",))
        self.assertEqual(self.events[index - 1][0], "fchdir")
        self.assertEqual(self.events[index + 1], ("readback", "/", 7, 8))
        self.os.chdir.assert_not_called()

    def test_scratch_and_shm_are_separate_sized_tmpfs(self):
        self.run_setup()
        self.assertIn(call("tmpfs", OPTIONS["mountpoint_path"] + "/tmp", "tmpfs", 14,
                           "size=536870912,nr_inodes=16384,mode=1777,uid=10001,gid=10001"),
                      self.mount_call.call_args_list)
        self.assertIn(call("tmpfs", OPTIONS["mountpoint_path"] + "/dev/shm", "tmpfs", 14,
                           "size=67108864,nr_inodes=1024,mode=1777,uid=10001,gid=10001"),
                      self.mount_call.call_args_list)
        self.assertIn(call(OPTIONS["mountpoint_path"] + "/dev", 11, recursive=False),
                      self.seal.call_args_list)

    def test_proc_is_new_filesystem_not_host_bind(self):
        self.run_setup()
        self.assertIn(call("proc", OPTIONS["mountpoint_path"] + "/proc", "proc", 14,
                           "hidepid=2"), self.mount_call.call_args_list)
        self.assertIn(call(OPTIONS["mountpoint_path"] + "/proc", 15, recursive=False),
                      self.seal.call_args_list)

    def test_only_four_minimal_device_nodes_and_overlay_shm_directory_created(self):
        self.run_setup()
        self.assertEqual(self.os.mknod.call_count, 4)
        self.assertEqual(self.os.symlink.call_count, 4)
        self.os.mkdir.assert_called_once_with(OPTIONS["mountpoint_path"] + "/dev/shm", 0o1777)
        for invocation in self.os.mknod.call_args_list:
            self.assertTrue(invocation.args[0].startswith(OPTIONS["mountpoint_path"] + "/dev/"))
        self.assertFalse(any("oldroot" in str(item) for item in self.os.mkdir.call_args_list))

    def test_export_metadata_is_not_changed(self):
        self.run_setup()
        for operation in (self.os.mkdir, self.os.chmod, self.os.mknod, self.os.symlink):
            self.assertTrue(all(OPTIONS["rootfs_path"] not in str(item)
                                for item in operation.call_args_list))

    def test_invalid_inputs_have_no_native_or_filesystem_effects(self):
        for name, values in {
            "rootfs_path": ("/", "relative", "/tmp/a/../rootfs", "/tmp//bad/rootfs",
                            "/tmp/a/rootfs/", "/tmp/a/rootfs\n"),
            "mountpoint_path": (OPTIONS["rootfs_path"], "/tmp/elsewhere/mountpoint",
                                OPTIONS["rootfs_path"] + "/nested"),
            "uid": (True, 0, -1, 2**31), "gid": (False, -1),
            "scratch_bytes": (True, 4096, 536870913, 8388609),
            "shm_bytes": (False, 4096, 67108865, 1048577),
        }.items():
            for value in values:
                self.error("input_rejected", **{name: value})
        self.runtime.assert_not_called()
        self.pin.assert_not_called()
        self.unshare.assert_not_called()

    def test_runtime_failure_precedes_first_source_open(self):
        self.runtime.side_effect = OSError(PRIVATE)
        self.error("runtime_rejected")
        self.pin.assert_not_called()
        self.unshare.assert_not_called()

    def test_cpython_312_required(self):
        mount.sys.version_info = (3, 13)
        self.error("runtime_rejected")
        self.pin.assert_not_called()

    def test_parent_mode_or_owner_rejected_before_unshare(self):
        self.parent_info.st_mode = stat.S_IFDIR | 0o755
        self.error("root_rejected")
        self.unshare.assert_not_called()
        self.assertEqual(self.fds, {})
        self.error("already_attempted")

    def test_rootfs_owner_rejected(self):
        self.root_info.st_uid = 10001
        self.error("root_rejected")
        self.unshare.assert_not_called()

    def test_nonempty_mountpoint_rejected(self):
        self.os.listdir.return_value = ["synthetic-entry"]
        self.error("root_rejected")
        self.unshare.assert_not_called()
        self.assertEqual(self.fds, {})

    def test_nested_source_symlink_failure_is_terminal_without_mount(self):
        original = self.pin.side_effect

        def reject(path, owned):
            if path.endswith("/usr/local"):
                raise mount.mount_root.MountRootError("source_rejected")
            return original(path, owned)

        self.pin.side_effect = reject
        self.error("source_rejected")
        self.mount_call.assert_not_called()
        self.assertEqual(self.fds, {})

    def test_runtime_file_is_metadata_checked_and_never_read(self):
        self.run_setup()
        self.legacy_pin.assert_called_once()
        self.assertFalse(hasattr(self.os, "read"))

    def test_changed_namespace_denied(self):
        self.namespace_identity.side_effect = lambda fd: (8, 1)
        self.error("namespace_unchanged")
        self.mount_call.assert_not_called()
        self.pivot.assert_not_called()

    def test_repin_root_identity_mismatch_denied_before_bind(self):
        original = self.pin.side_effect

        def changed(path, owned):
            descriptor = original(path, owned)
            if path == OPTIONS["rootfs_path"] and self.namespace_generation:
                self.fds[descriptor] = metadata(inode=99)
            return descriptor

        self.pin.side_effect = changed
        self.error("source_rejected")
        self.assertEqual(self.mount_call.call_count, 1)
        self.pivot.assert_not_called()
        self.assertEqual(self.fds, {})

    def test_bound_root_identity_mismatch_prevents_overlays(self):
        original = self.pin.side_effect

        def changed(path, owned):
            descriptor = original(path, owned)
            if path == OPTIONS["mountpoint_path"] and self.bound:
                self.fds[descriptor] = metadata(inode=99)
            return descriptor

        self.pin.side_effect = changed
        self.error("source_rejected")
        self.assertEqual(self.mount_call.call_count, 2)
        self.seal.assert_not_called()

    def test_source_namespace_fds_closed_once_even_on_bind_failure(self):
        original = self.mount_call.side_effect

        def fail(*args, **kwargs):
            if args[3] & mount.mount_root.MS_BIND:
                raise mount.mount_root.MountRootError("bind_mount_failed")
            return original(*args, **kwargs)

        self.mount_call.side_effect = fail
        self.error("bind_mount_failed")
        self.assertEqual(self.fds, {})
        self.pivot.assert_not_called()
        self.error("already_attempted")

    def test_started_unshare_failure_is_terminal(self):
        self.unshare.side_effect = mount.mount_root.MountRootError("namespace_unshare_failed")
        self.error("namespace_unshare_failed")
        self.error("already_attempted")
        self.mount_call.assert_not_called()

    def test_scratch_wrong_size_readback_rejected(self):
        self.scratch_bytes -= 4096
        self.error("readback_failed")
        self.pivot.assert_called_once()
        self.assertEqual(self.fds, {})

    def test_shm_wrong_size_readback_rejected(self):
        self.shm_bytes -= 4096
        self.error("readback_failed")

    def test_unexpected_exception_text_is_not_exposed(self):
        self.pivot.side_effect = OSError(PRIVATE)
        self.error("setup_failed")
        self.assertEqual(self.fds, {})

    def test_close_failure_not_retried(self):
        original = self.os.close.side_effect

        def failed(descriptor):
            original(descriptor)
            if len(self.closed) == 1:
                raise OSError(PRIVATE)

        self.os.close.side_effect = failed
        self.error("close_failed")
        self.assertEqual(self.fds, {})

    def test_same_owner_and_changed_owner_reentry_rejected(self):
        self.run_setup()
        self.error("already_attempted")
        self.os.getpid.return_value = 4001
        self.error("owner_mismatch")

    def test_lock_reentry_is_rejected_without_releasing_another_owner_lock(self):
        self.assertTrue(mount._SETUP_LOCK.acquire(blocking=False))
        try:
            self.error("reentry")
            self.assertTrue(mount._SETUP_LOCK.locked())
            self.pin.assert_not_called()
        finally:
            mount._SETUP_LOCK.release()

    def test_wrong_device_number_rejected_after_pivot(self):
        self.os.stat.return_value = metadata(mode=stat.S_IFCHR | 0o666,
                                             rdev=os.makedev(1, 99))
        self.os.stat.side_effect = None
        self.error("device_rejected")
        self.pivot.assert_called_once()

    def test_wrong_scratch_owner_rejected_after_pivot(self):
        original = self.pin.side_effect

        def changed(path, owned):
            descriptor = original(path, owned)
            if path == "/tmp":
                self.fds[descriptor].st_uid = 0
            return descriptor

        self.pin.side_effect = changed
        self.error("readback_failed")
        self.pivot.assert_called_once()


class NativeAdapterAndSourceTests(unittest.TestCase):
    def test_import_has_no_ml_dependencies(self):
        self.assertFalse(any(name.split(".", 1)[0] in mount.netns.ML_ROOTS
                             for name in IMPORT_ADDITIONS))

    def test_pivot_exact_documented_same_directory_then_detach_then_chdir(self):
        events = []
        syscall = Mock(side_effect=lambda *args: events.append(("pivot", args)) or 0)
        umount2 = Mock(side_effect=lambda *args: events.append(("detach", args)) or 0)
        libc = types.SimpleNamespace(syscall=syscall, umount2=umount2)
        with patch.object(mount.ctypes, "CDLL", return_value=libc) as library, \
                patch.object(mount.os, "chdir", side_effect=lambda p: events.append(("chdir", p))):
            mount._pivot()
        library.assert_called_once_with(None, use_errno=True)
        self.assertEqual(events[0][0], "pivot")
        arguments = events[0][1]
        self.assertEqual([item.value for item in arguments], [155, b".", b"."])
        self.assertEqual(events[1:], [("detach", (b".", 2)), ("chdir", "/")])
        self.assertEqual(syscall.argtypes, [ctypes.c_long])
        self.assertEqual(umount2.argtypes, [ctypes.c_char_p, ctypes.c_int])

    def test_pivot_failure_never_detaches_or_retries(self):
        libc = types.SimpleNamespace(syscall=Mock(return_value=-1), umount2=Mock())
        with patch.object(mount.ctypes, "CDLL", return_value=libc), \
                patch.object(mount.os, "chdir") as chdir:
            with self.assertRaises(mount.OCIMountError) as caught:
                mount._pivot()
        self.assertEqual(caught.exception.code, "pivot_failed")
        self.assertEqual(libc.syscall.call_count, 1)
        libc.umount2.assert_not_called()
        chdir.assert_not_called()

    def test_detach_failure_never_changes_directory_or_retries(self):
        libc = types.SimpleNamespace(syscall=Mock(return_value=0), umount2=Mock(return_value=-1))
        with patch.object(mount.ctypes, "CDLL", return_value=libc), \
                patch.object(mount.os, "chdir") as chdir:
            with self.assertRaises(mount.OCIMountError):
                mount._pivot()
        self.assertEqual(libc.umount2.call_count, 1)
        chdir.assert_not_called()

    def test_only_new_overlay_directories_are_created_in_source(self):
        tree = ast.parse((ROOT / "src/lumi/posttraining_sft_oci_mount_v1.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        forbidden = {"chroot", "setns", "execv", "execve", "system", "Popen", "rmdir",
                     "unlink", "remove", "rmtree", "chmod_rootfs"}
        self.assertFalse(any(isinstance(node.func, ast.Attribute)
                             and isinstance(node.func.value, ast.Name)
                             and node.func.value.id in ("os", "subprocess", "shutil")
                             and node.func.attr in forbidden for node in calls))
        mkdir = [node for node in calls if isinstance(node.func, ast.Attribute)
                 and node.func.attr == "mkdir"]
        self.assertEqual(len(mkdir), 1)
        self.assertEqual(ast.unparse(mkdir[0].args[0]), "shm_path")

    def test_api_defaults_and_no_worker_dispatch_entrypoint(self):
        tree = ast.parse((ROOT / "src/lumi/posttraining_sft_oci_mount_v1.py").read_text())
        entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "enter_oci_rootfs")
        self.assertEqual([argument.arg for argument in entry.args.args],
                         ["rootfs_path", "mountpoint_path"])
        self.assertEqual([argument.arg for argument in entry.args.kwonlyargs],
                         ["uid", "gid", "scratch_bytes", "shm_bytes"])
        self.assertEqual([ast.literal_eval(node) for node in entry.args.kw_defaults],
                         [10001, 10001, 536870912, 67108864])
        self.assertFalse(any(isinstance(node, ast.If) for node in tree.body))

    def test_low_level_source_pin_is_preserved_no_symlink_resolution(self):
        tree = ast.parse((ROOT / "src/lumi/posttraining_sft_oci_mount_v1.py").read_text())
        pin = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "_pin")
        source = ast.unparse(pin)
        self.assertIn("mount_root._pin(path, directory=True, owned=owned)", source)
        self.assertIn("os.O_NOFOLLOW", source)
        self.assertNotIn("resolve", source)
        self.assertNotIn("realpath", source)


if __name__ == "__main__":
    unittest.main()
