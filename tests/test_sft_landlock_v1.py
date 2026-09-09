"""Source-AST contract checks and syscall doubles; no live procfs, libc or ML."""

from __future__ import annotations
import __future__

import ast
import ctypes
import fcntl
import importlib
import os
import stat
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_landlock_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
landlock = importlib.import_module(PACKAGE + ".posttraining_sft_landlock_v1")
IMPORT_ADDITIONS = set(sys.modules) - before_import
PRIVATE = "SYNTHETIC-PRIVATE-EXCEPTION"


class MemoryLock:
    def __init__(self):
        self.held = False

    def acquire(self, *, blocking):
        if blocking is not False or self.held:
            return False
        self.held = True
        return True

    def release(self):
        if not self.held:
            raise AssertionError("lock_not_owned")
        self.held = False


class MemoryKernel:
    def __init__(self):
        self.events, self.closed, self.fds = [], [], {}
        self.next_fd, self.abi, self.ruleset_fd = 20, 5, 10
        self.cloexec, self.path_flags = True, True
        self.fail_open, self.fail_close, self.fail_rule = None, None, False
        self.restrict_result, self.open_hook = 0, None
        self.special = {"/code/main.py": stat.S_IFREG | 0o444,
                        "/dev/null": stat.S_IFCHR | 0o666,
                        "/dev/zero": stat.S_IFCHR | 0o666,
                        "/dev/random": stat.S_IFCHR | 0o666,
                        "/dev/urandom": stat.S_IFCHR | 0o666}
        self.device_numbers = {"/dev/null": 3, "/dev/zero": 5,
                               "/dev/random": 8, "/dev/urandom": 9}
        self.rules = []

    def open(self, path, flags, *, dir_fd=None):
        full = path if dir_fd is None else self.fds[dir_fd].path.rstrip("/") + "/" + path
        self.events.append(("open", full, flags))
        if full == self.fail_open:
            raise OSError(PRIVATE)
        fd, self.next_fd = self.next_fd, self.next_fd + 1
        self.fds[fd] = types.SimpleNamespace(
            path=full, st_mode=self.special.get(full, stat.S_IFDIR | 0o555),
            st_uid=0, st_gid=0, st_dev=1, st_ino=fd,
            st_rdev=os.makedev(1, self.device_numbers.get(full, 0)),
        )
        if self.open_hook is not None:
            self.open_hook(fd, self.fds[fd])
        return fd

    def fstat(self, fd):
        self.events.append(("fstat", fd))
        return self.fds[fd]

    def fcntl(self, fd, command):
        self.events.append(("fcntl", fd, command))
        if command == fcntl.F_GETFD:
            return fcntl.FD_CLOEXEC if self.cloexec else 0
        if command == fcntl.F_GETFL:
            return landlock.O_PATH if self.path_flags else 0
        raise AssertionError("unexpected_fcntl")

    def close(self, fd):
        self.events.append(("close", fd))
        if fd in self.closed:
            raise AssertionError("repeated_close")
        self.closed.append(fd)
        if fd == self.fail_close:
            raise OSError(PRIVATE)

    def syscall(self, number, *arguments):
        self.events.append(("syscall", number))
        if number == landlock.SYS_CREATE_RULESET:
            if arguments[-1].value == landlock.LANDLOCK_CREATE_RULESET_VERSION:
                if arguments[0].value is not None or arguments[1].value != 0:
                    raise AssertionError("wrong_abi_query")
                return self.abi
            if (arguments[0]._obj.handled_access_fs != (1 << 16) - 1
                    or arguments[1].value != 8 or arguments[2].value != 0):
                raise AssertionError("wrong_ruleset_layout")
            return self.ruleset_fd
        if number == landlock.SYS_ADD_RULE:
            value = arguments[2]._obj
            if (arguments[0].value != 10 or arguments[1].value != 1
                    or arguments[3].value != 0 or ctypes.sizeof(value) != 12):
                raise AssertionError("wrong_path_rule_layout")
            self.rules.append((self.fds[value.parent_fd].path, value.allowed_access))
            return -1 if self.fail_rule else 0
        if number == landlock.SYS_RESTRICT_SELF:
            if arguments[0].value != 10 or arguments[1].value != 0:
                raise AssertionError("wrong_restrict_arguments")
            return self.restrict_result
        raise AssertionError("unexpected_syscall")


class LandlockTests(unittest.TestCase):
    def setUp(self):
        self.kernel, self.lock = MemoryKernel(), MemoryLock()
        self.runtime, self.privileges = Mock(), Mock()
        self.root_filesystem = Mock()
        self.root_view = types.SimpleNamespace(f_flag=7, f_frsize=4096, f_blocks=256)
        replacements = {
            "_runtime": self.runtime, "_no_new_privileges": self.privileges,
            "_syscall": self.kernel.syscall, "_SETUP_LOCK": self.lock, "_ATTEMPT_OWNER": None,
            "_filesystem": self.root_filesystem,
        }
        for name, value in replacements.items():
            selected = patch.object(landlock, name, value)
            selected.start()
            self.addCleanup(selected.stop)
        for module, name, value in (
            (landlock.os, "open", self.kernel.open), (landlock.os, "fstat", self.kernel.fstat),
            (landlock.os, "close", self.kernel.close), (landlock.os, "getpid", lambda: 100),
            (landlock.threading, "get_ident", lambda: 200),
            (landlock.fcntl, "fcntl", self.kernel.fcntl),
            (landlock.os, "fstatvfs", lambda fd: self.root_view),
        ):
            selected = patch.object(module, name, value)
            selected.start()
            self.addCleanup(selected.stop)

    def invoke(self, **updates):
        values = {"read_only_paths": ("/code", "/proc"), "writable_paths": ("/tmp",),
                  "device_paths": ("/dev/null",)}
        values.update(updates)
        return landlock.restrict_filesystem(**values)

    def rejects(self, code, **updates):
        with self.assertRaises(landlock.LandlockError) as raised:
            self.invoke(**updates)
        self.assertEqual(raised.exception.code, code)
        self.assertEqual(str(raised.exception), code)
        self.assertNotIn(PRIVATE, str(raised.exception))
        self.assertFalse(self.lock.held)

    def test_success_exact_receipt_masks_and_close_ownership(self):
        self.assertEqual(self.invoke(), {
            "schema": "lumi.sft-landlock.v0.1", "abi_version": 5,
            "private_root_listing_only": False,
            "filesystem_content_policy_enforced": True, "ptrace_domain_restricted": True,
            "full_pathname_unix_isolation_verified": False,
            "metadata_policy_alone_sufficient": False, "real_sft_execution_permitted": False,
            "production_eligible": False,
        })
        self.assertEqual(self.kernel.rules, [("/code", 13), ("/proc", 13),
                                        ("/tmp", 30654), ("/dev/null", 6)])
        self.assertEqual(set(self.kernel.closed), set(self.kernel.fds) | {10})
        self.assertEqual(len(set(self.kernel.closed)), len(self.kernel.closed))
        self.runtime.assert_called_once_with()
        self.privileges.assert_called_once_with()
        self.assertEqual(landlock._ATTEMPT_OWNER, (100, 200))

    def test_readonly_file_excludes_read_dir(self):
        self.invoke(read_only_paths=("/code/main.py",))
        self.assertEqual(self.kernel.rules[0], ("/code/main.py", 5))

    def private_root(self):
        self.kernel.special["/"] = stat.S_IFDIR | 0o755

    def test_private_root_listing_is_opt_in_and_directory_names_only(self):
        self.private_root()
        receipt = self.invoke(allow_private_root_listing=True)
        self.assertIs(receipt["private_root_listing_only"], True)
        self.assertEqual(self.kernel.rules[0], ("/", 8))
        self.assertEqual(self.kernel.rules[0][1] & (landlock.READ_FILE | landlock.WRITE_FILE
                                                   | landlock.EXECUTE), 0)
        self.root_filesystem.assert_called_once()
        self.assertEqual(self.root_filesystem.call_args.args[1], landlock.TMPFS_MAGIC)

    def test_ordinary_host_filesystem_cannot_receive_root_rule(self):
        self.private_root()
        self.root_filesystem.side_effect = RuntimeError(PRIVATE)
        self.rejects("rule_rejected", allow_private_root_listing=True)
        self.assertEqual(self.kernel.rules, [])

    def test_writable_root_cannot_receive_listing_rule(self):
        self.private_root()
        self.root_view.f_flag = 6
        self.rejects("path_rejected", allow_private_root_listing=True)

    def test_root_capacity_must_match_private_scaffold(self):
        self.private_root()
        self.root_view.f_blocks = 512
        self.rejects("path_rejected", allow_private_root_listing=True)

    def test_private_root_mode_is_exact(self):
        self.rejects("path_rejected", allow_private_root_listing=True)

    def test_root_listing_flag_rejects_truthy_nonbool(self):
        for value in (1, "yes", [], None):
            self.rejects("paths_rejected", allow_private_root_listing=value)
        self.assertEqual(self.kernel.events, [])

    def test_private_root_rule_counts_toward_existing_sixteen_rule_cap(self):
        self.rejects("paths_rejected", read_only_paths=tuple(f"/code{i}" for i in range(14)),
                     allow_private_root_listing=True)
        self.assertEqual(self.kernel.events, [])

    def secure_loader_walk(self):
        # Compile only these three pure source definitions, never import the
        # ML runtime. The unmodified path walk operates solely on syscall doubles.
        source = ast.parse((ROOT / "src/lumi/posttraining_sft_runtime_v4.py").read_bytes())
        names = {"_absolute_lexical", "_secure_open_flags", "_open_secure_root"}
        nodes = [node for node in source.body if isinstance(node, ast.FunctionDef)
                 and node.name in names]
        self.assertEqual({node.name for node in nodes}, names)
        descriptors, closed = {}, []
        def opened(path, flags, *, dir_fd=None):
            full = path if dir_fd is None else descriptors[dir_fd].rstrip("/") + "/" + path
            if full == "/" and ("/", landlock.READ_DIR) not in self.kernel.rules:
                raise PermissionError("SYNTHETIC_DIRECTORY_RULE_REQUIRED")
            self.assertEqual(flags & landlock.O_PATH, 0)
            self.assertTrue(flags & os.O_DIRECTORY and flags & os.O_NOFOLLOW)
            fd = 1000 + len(descriptors)
            descriptors[fd] = full
            return fd
        fake = types.SimpleNamespace(
            path=types.SimpleNamespace(abspath=lambda path: str(path)), O_RDONLY=os.O_RDONLY,
            O_NOFOLLOW=os.O_NOFOLLOW, O_CLOEXEC=os.O_CLOEXEC, O_DIRECTORY=os.O_DIRECTORY,
            open=opened, close=closed.append, geteuid=lambda: 1001,
            fstat=lambda fd: types.SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=1001),
        )
        scope = {"os": fake, "stat": stat, "Path": Path,
                 "_secure_file_identity": lambda value: value}
        code = compile(ast.Module(body=nodes, type_ignores=[]), "<secure-loader-contract>", "exec",
                       flags=__future__.annotations.compiler_flag)
        exec(code, scope)
        result = scope["_open_secure_root"](Path("/tmp/generated"))
        self.assertEqual(descriptors[result[1]], "/tmp/generated")
        self.assertEqual(len(closed), 2)

    def test_original_secure_loader_walk_rejects_missing_root_directory_rule(self):
        self.invoke()
        with self.assertRaises(PermissionError):
            self.secure_loader_walk()

    def test_original_secure_loader_walk_accepts_names_only_private_root_rule(self):
        self.private_root()
        self.invoke(allow_private_root_listing=True)
        self.secure_loader_walk()

    def test_device_rules_exact_character_identity_and_masks(self):
        self.invoke(device_paths=tuple(landlock.DEVICES))
        self.assertEqual(self.kernel.rules[-4:], [("/dev/null", 6), ("/dev/zero", 6),
                                                 ("/dev/random", 4), ("/dev/urandom", 4)])

    def test_writable_mask_forbids_execute_mknod_and_ioctl(self):
        mask = landlock.WRITABLE_DIRECTORY
        self.assertEqual(mask & (landlock.EXECUTE | landlock.MAKE_CHAR
                                 | landlock.MAKE_BLOCK | landlock.IOCTL_DEV), 0)
        self.assertEqual(mask & (landlock.REFER | landlock.TRUNCATE), 24576)

    def test_symlink_leaf_is_not_followed(self):
        self.kernel.special["/code"] = stat.S_IFLNK | 0o777
        self.rejects("path_rejected")
        self.assertFalse(self.kernel.rules)

    def test_symlink_parent_is_not_followed(self):
        self.kernel.special["/code"] = stat.S_IFLNK | 0o777
        self.rejects("path_rejected", read_only_paths=("/code/main.py",))
        self.assertNotIn("/code/main.py", [item[1] for item in self.kernel.events
                                          if item[0] == "open"])

    def test_nondirectory_parent_denied(self):
        self.kernel.special["/code"] = stat.S_IFREG | 0o444
        self.rejects("path_rejected", read_only_paths=("/code/main.py",))

    def test_special_readonly_object_denied(self):
        self.kernel.special["/code"] = stat.S_IFSOCK | 0o666
        self.rejects("path_rejected")

    def test_writable_file_denied(self):
        self.kernel.special["/tmp"] = stat.S_IFREG | 0o600
        self.rejects("path_rejected")

    def test_wrong_device_number_denied(self):
        self.kernel.device_numbers["/dev/null"] = 1
        self.rejects("device_rejected")

    def test_noncharacter_device_denied(self):
        self.kernel.special["/dev/null"] = stat.S_IFREG | 0o600
        self.rejects("device_rejected")

    def test_unowned_device_denied(self):
        def modify(_fd, info):
            if info.path == "/dev/null":
                info.st_uid = 1000
        self.kernel.open_hook = modify
        self.rejects("device_rejected")

    def test_no_cloexec_ruleset_denied_and_closed(self):
        self.kernel.cloexec = False
        self.rejects("ruleset_rejected")
        self.assertEqual(self.kernel.closed, [10])

    def test_no_opath_denied_and_all_fds_closed(self):
        self.kernel.path_flags = False
        self.rejects("path_rejected")
        self.assertEqual(set(self.kernel.closed), set(self.kernel.fds) | {10})

    def test_path_open_error_is_fixed_and_no_payload_release(self):
        self.kernel.fail_open = "/code"
        self.rejects("path_rejected")
        self.privileges.assert_not_called()
        self.assertEqual(set(self.kernel.closed), set(self.kernel.fds) | {10})

    def test_rule_error_closes_path_and_ruleset_once(self):
        self.kernel.fail_rule = True
        self.rejects("rule_rejected")
        self.assertEqual(set(self.kernel.closed), set(self.kernel.fds) | {10})
        self.privileges.assert_not_called()

    def test_parent_close_error_does_not_retry_and_closes_child(self):
        self.kernel.fail_close = 20
        self.rejects("close_failed")
        self.assertEqual(set(self.kernel.closed), set(self.kernel.fds) | {10})

    def test_ruleset_close_error_overrides_success_and_retires_attempt(self):
        self.kernel.fail_close = 10
        self.rejects("close_failed")
        self.assertEqual(self.kernel.closed.count(10), 1)
        self.rejects("already_attempted")

    def test_restrict_failure_is_terminal(self):
        self.kernel.restrict_result = -1
        self.rejects("restrict_failed")
        self.rejects("already_attempted")
        self.assertEqual(self.kernel.events.count(("syscall", 446)), 1)

    def test_privilege_failure_does_not_restrict_or_retry(self):
        self.privileges.side_effect = landlock.LandlockError("privilege_rejected")
        self.rejects("privilege_rejected")
        self.assertNotIn(("syscall", 446), self.kernel.events)
        self.rejects("already_attempted")

    def test_abi_rejection_has_no_feature_degradation(self):
        for abi in (-1, 0, 1, 2, 3, 4, True, 5.0, 2**31):
            with self.subTest(abi=abi):
                landlock._ATTEMPT_OWNER = None
                self.kernel.abi = abi
                before = len(self.kernel.events)
                self.rejects("abi_rejected")
                self.assertEqual(self.kernel.events[before:], [("syscall", 444)])
        self.rejects("already_attempted")

    def test_newer_abi_does_not_silently_claim_new_rights(self):
        self.kernel.abi = 9
        receipt = self.invoke()
        self.assertEqual(receipt["abi_version"], 9)
        self.assertFalse(receipt["full_pathname_unix_isolation_verified"])

    def test_ruleset_failure_is_fixed_and_has_no_invalid_close(self):
        self.kernel.ruleset_fd = -1
        self.rejects("ruleset_rejected")
        self.assertEqual(self.kernel.closed, [])

    def test_runtime_failure_precedes_consumption(self):
        self.runtime.side_effect = RuntimeError(PRIVATE)
        self.rejects("runtime_rejected")
        self.assertIsNone(landlock._ATTEMPT_OWNER)
        self.assertEqual(self.kernel.events, [])

    def test_success_cannot_be_reused(self):
        self.invoke()
        before = len(self.kernel.events)
        self.rejects("already_attempted")
        self.assertEqual(len(self.kernel.events), before)

    def test_fork_or_thread_owner_change_denied(self):
        landlock._ATTEMPT_OWNER = (99, 200)
        self.rejects("owner_mismatch")
        self.assertEqual(self.kernel.events, [])

    def test_reentry_denied_before_any_setup(self):
        self.lock.held = True
        with self.assertRaisesRegex(landlock.LandlockError, "reentry"):
            self.invoke()
        self.assertEqual(self.kernel.events, [])
        self.assertTrue(self.lock.held)
        self.lock.held = False

    def test_invalid_path_inputs_rejected_without_syscalls(self):
        cases = [
            {"read_only_paths": ["/code"]}, {"read_only_paths": ()},
            {"writable_paths": ["/tmp"]}, {"device_paths": ["/dev/null"]},
            *({"read_only_paths": (path,)} for path in
              ("/", "relative", "//code", "/code/", "/code/../tmp", "/code/./a",
               "/co\x00de", "/co\nde", "/co de", "/é", "/" + "a" * 256, True)),
            {"read_only_paths": tuple(f"/code{i}" for i in range(17))},
            {"device_paths": ("/dev/tty",)}, {"read_only_paths": ("/code", "/code")},
            {"writable_paths": ("/code/private",)},
            {"read_only_paths": ("/tmp/read",)},
            {"writable_paths": ("/tmp", "/tmp/nested")},
            {"read_only_paths": ("/dev",)}, {"writable_paths": ("/dev",)},
        ]
        for case in cases:
            with self.subTest(case=case):
                self.rejects("paths_rejected", **case)
                self.assertEqual(self.kernel.events, [])
                self.assertIsNone(landlock._ATTEMPT_OWNER)

    def test_component_open_flags_and_paths_are_bounded(self):
        self.invoke(read_only_paths=("/code/main.py",))
        for _, path, flags in (item for item in self.kernel.events if item[0] == "open"):
            self.assertTrue(flags & landlock.O_PATH)
            self.assertTrue(flags & os.O_NOFOLLOW)
            self.assertTrue(flags & os.O_CLOEXEC)
            if path in ("/", "/code", "/dev"):
                self.assertTrue(flags & os.O_DIRECTORY)

    def test_unknown_exception_code_is_redacted(self):
        self.assertEqual(str(landlock.LandlockError(PRIVATE)), "setup_failed")
        self.assertEqual(str(landlock.LandlockError(None)), "setup_failed")


class NativeBindingDoubleTests(unittest.TestCase):
    def test_native_syscall_keeps_explicit_variadic_argument_widths(self):
        call = Mock(return_value=5)
        libc = types.SimpleNamespace(syscall=call)
        arguments = ctypes.c_void_p(), ctypes.c_size_t(0), ctypes.c_uint(1)
        with patch.object(landlock.ctypes, "CDLL", return_value=libc):
            self.assertEqual(landlock._syscall(444, *arguments), 5)
        self.assertEqual(call.argtypes, [ctypes.c_long])
        self.assertIs(call.restype, ctypes.c_long)
        call.assert_called_once_with(444, *arguments)

    def test_nnp_set_and_get_are_both_required(self):
        for results, passes in (([0, 1], True), ([-1], False), ([0, 0], False)):
            with self.subTest(results=results):
                call = Mock(side_effect=results)
                libc = types.SimpleNamespace(prctl=call)
                with patch.object(landlock.ctypes, "CDLL", return_value=libc):
                    if passes:
                        landlock._no_new_privileges()
                    else:
                        with self.assertRaisesRegex(landlock.LandlockError, "privilege_rejected"):
                            landlock._no_new_privileges()
                self.assertEqual(call.call_args_list[0].args, (38, 1, 0, 0, 0))
                if len(results) == 2:
                    self.assertEqual(call.call_args_list[1].args, (39, 0, 0, 0, 0))

    def test_import_has_no_ml_side_effects(self):
        roots = {name.split(".", 1)[0] for name in IMPORT_ADDITIONS}
        self.assertFalse(roots & {"torch", "numpy", "transformers", "safetensors",
                                  "tokenizers", "tensorflow", "jax", "cupy"})


if __name__ == "__main__":
    unittest.main()
