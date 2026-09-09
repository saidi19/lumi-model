"""Mocked runtime/filesystem/syscalls and source-AST only; no native execution."""

from __future__ import annotations

import ast
import ctypes
import importlib
import stat
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_synthetic_oci_landlock_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
policy = importlib.import_module(PACKAGE + ".posttraining_sft_oci_landlock_v1")
IMPORT_ADDITIONS = set(sys.modules) - before_import
IDENTITY = (7, 19, stat.S_IFDIR, 0, 0)
OPTIONS = {
    "read_only_paths": ("/usr", "/etc", "/opt/lumi-release", "/proc"),
    "writable_paths": ("/tmp", "/dev/shm"),
    "device_paths": ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom"),
    "expected_root_identity": IDENTITY,
}
PRIVATE = "synthetic-private-error"


class OCILandlockTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(policy, "_ATTEMPT_OWNER", None))
        self.runtime = self.enterContext(patch.object(policy.landlock, "_runtime"))
        self.metadata = types.SimpleNamespace(st_dev=7, st_ino=19,
                                              st_mode=stat.S_IFDIR | 0o755,
                                              st_uid=0, st_gid=0)
        self.view = types.SimpleNamespace(f_flag=7)
        self.events, self.closed, self.root_rules = [], [], []
        self.abi, self.creation, self.add_result, self.restrict_result = 5, 20, 0, 0
        self.os = types.SimpleNamespace(getpid=Mock(return_value=4000),
                                        close=Mock(side_effect=self.close),
                                        fstatvfs=Mock(return_value=self.view))
        self.enterContext(patch.object(policy, "os", self.os))
        self.fcntl = self.enterContext(patch.object(
            policy.fcntl, "fcntl", return_value=policy.fcntl.FD_CLOEXEC,
        ))
        self.open_path = self.enterContext(patch.object(
            policy.landlock, "_open_path", return_value=42,
        ))
        self.fd_info = self.enterContext(patch.object(
            policy.landlock, "_fd_info", return_value=self.metadata,
        ))
        self.syscall = self.enterContext(patch.object(
            policy.landlock, "_syscall", side_effect=self.system_call,
        ))
        self.rules = self.enterContext(patch.object(
            policy.landlock, "_add_rule", side_effect=lambda *a: self.events.append(("rule", a)),
        ))
        self.privileges = self.enterContext(patch.object(
            policy.landlock, "_no_new_privileges", side_effect=lambda: self.events.append(("nnp",)),
        ))
        self.old_private_root = self.enterContext(patch.object(
            policy.landlock, "_private_root", side_effect=AssertionError("old_tmpfs_policy"),
        ))
        self.native = self.enterContext(patch.object(
            policy.ctypes, "CDLL", side_effect=AssertionError("no_native_execution"),
        ))

    def tearDown(self):
        self.assertFalse(policy._SETUP_LOCK.locked())
        self.assertEqual(len(self.closed), len(set(self.closed)))
        self.native.assert_not_called()
        self.old_private_root.assert_not_called()

    def close(self, descriptor):
        self.assertIn(descriptor, (20, 42))
        self.assertNotIn(descriptor, self.closed)
        self.closed.append(descriptor)
        self.events.append(("close", descriptor))

    def system_call(self, number, *arguments):
        self.events.append(("syscall", number))
        if number == policy.landlock.SYS_CREATE_RULESET:
            if arguments[-1].value == policy.landlock.LANDLOCK_CREATE_RULESET_VERSION:
                self.assertIsNone(arguments[0].value)
                self.assertEqual(arguments[1].value, 0)
                return self.abi
            self.assertEqual(arguments[0]._obj.handled_access_fs, policy.landlock.HANDLED_FS)
            self.assertEqual(arguments[1].value, ctypes.sizeof(policy.landlock.RulesetAttr))
            return self.creation
        if number == policy.landlock.SYS_ADD_RULE:
            self.assertEqual(arguments[0].value, 20)
            self.assertEqual(arguments[1].value, policy.landlock.LANDLOCK_RULE_PATH_BENEATH)
            self.assertEqual(arguments[3].value, 0)
            attributes = arguments[2]._obj
            self.root_rules.append((attributes.allowed_access, attributes.parent_fd))
            return self.add_result
        if number == policy.landlock.SYS_RESTRICT_SELF:
            self.assertEqual([value.value for value in arguments], [20, 0])
            return self.restrict_result
        raise AssertionError("unexpected_syscall")

    def run_policy(self, **changes):
        return policy.restrict_filesystem(**(OPTIONS | changes))

    def error(self, code, **changes):
        with self.assertRaises(policy.OCILandlockError) as caught:
            self.run_policy(**changes)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.args, (code,))
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertNotIn(PRIVATE, str(caught.exception))

    def test_success_exact_receipt_and_both_owned_fds_closed(self):
        self.assertEqual(self.run_policy(), {
            "schema": policy.SCHEMA, "abi_version": 5,
            "filesystem_content_policy_enforced": True,
            "exported_root_identity_verified": True, "root_readonly_mount_verified": True,
            "private_root_listing_only": True, "ptrace_domain_restricted": True,
            "rootfs_content_authenticated": False,
            "full_pathname_unix_isolation_verified": False,
            "metadata_policy_alone_sufficient": False, "inherited_authority_closed": False,
            "complete_os_sandbox_verified": False, "real_host_admission_verified": False,
            "real_sft_execution_permitted": False, "production_eligible": False,
        })
        self.assertEqual(self.closed, [42, 20])
        self.runtime.assert_called_once_with()

    def test_root_grants_only_read_dir_not_read_file_write_or_execute(self):
        self.run_policy()
        self.assertEqual(self.root_rules, [(policy.landlock.READ_DIR, 42)])
        self.open_path.assert_called_once_with("/")
        self.fd_info.assert_called_once_with(42)
        self.os.fstatvfs.assert_called_once_with(42)

    def test_existing_roles_delegated_unchanged(self):
        self.run_policy()
        expected = [call(20, path, role) for role, field in (
            ("read_only", "read_only_paths"), ("writable", "writable_paths"),
            ("device", "device_paths"),
        ) for path in OPTIONS[field]]
        self.assertEqual(self.rules.call_args_list, expected)

    def test_root_rule_precedes_normal_rules_and_restriction(self):
        self.run_policy()
        root_index = self.events.index(("syscall", policy.landlock.SYS_ADD_RULE))
        first_rule_index = next(i for i, value in enumerate(self.events) if value[0] == "rule")
        self.assertLess(root_index, first_rule_index)
        self.assertEqual(self.events[-3:], [("nnp",),
                         ("syscall", policy.landlock.SYS_RESTRICT_SELF), ("close", 20)])

    def test_non_tmpfs_is_accepted_without_filesystem_type_or_size_assumption(self):
        # The fake statvfs deliberately provides no tmpfs magic or capacity fields.
        self.run_policy()
        self.assertEqual(vars(self.view), {"f_flag": 7})

    def test_invalid_identity_types_rejected_before_runtime_or_syscalls(self):
        for value in (None, list(IDENTITY), (), IDENTITY + (0,), (True, 19, stat.S_IFDIR, 0, 0),
                      (-1, 19, stat.S_IFDIR, 0, 0), (7, 0, stat.S_IFDIR, 0, 0),
                      (7, 2**64, stat.S_IFDIR, 0, 0), (7, 19, stat.S_IFREG, 0, 0),
                      (7, 19, stat.S_IFDIR, 10001, 0), (7, 19, stat.S_IFDIR, 0, 10001)):
            self.error("root_identity_rejected", expected_root_identity=value)
        self.runtime.assert_not_called()
        self.syscall.assert_not_called()

    def test_invalid_paths_keep_old_rejection(self):
        self.error("paths_rejected", writable_paths=("/usr/scratch",))
        self.runtime.assert_not_called()
        self.syscall.assert_not_called()

    def test_root_rule_counts_against_sixteen_rule_budget(self):
        self.error("paths_rejected", read_only_paths=tuple(f"/readonly{i}" for i in range(10)))
        self.runtime.assert_not_called()

    def test_runtime_rejection_precedes_started_syscalls(self):
        self.runtime.side_effect = OSError(PRIVATE)
        self.error("runtime_rejected")
        self.syscall.assert_not_called()
        self.open_path.assert_not_called()

    def test_abi_below_five_is_consumed_once(self):
        self.abi = 4
        self.error("abi_rejected")
        self.error("already_attempted")
        self.assertEqual(self.syscall.call_count, 1)

    def test_boolean_abi_rejected(self):
        self.abi = True
        self.error("abi_rejected")

    def test_failed_ruleset_creation_does_not_close_unowned_fd(self):
        self.creation = -1
        self.error("ruleset_rejected")
        self.assertEqual(self.closed, [])
        self.open_path.assert_not_called()

    def test_non_cloexec_ruleset_rejected_and_closed(self):
        self.fcntl.return_value = 0
        self.error("ruleset_rejected")
        self.assertEqual(self.closed, [20])

    def test_wrong_root_inode_rejected_before_any_rule(self):
        self.metadata.st_ino += 1
        self.error("root_identity_rejected")
        self.assertEqual(self.root_rules, [])
        self.rules.assert_not_called()
        self.assertEqual(self.closed, [42, 20])

    def test_wrong_root_device_rejected(self):
        self.metadata.st_dev += 1
        self.error("root_identity_rejected")

    def test_wrong_root_owner_rejected(self):
        self.metadata.st_uid = 10001
        self.error("root_identity_rejected")

    def test_wrong_root_mode_rejected(self):
        self.metadata.st_mode = stat.S_IFDIR | 0o777
        self.error("root_identity_rejected")

    def test_wrong_root_type_rejected(self):
        self.metadata.st_mode = stat.S_IFREG | 0o755
        self.error("root_identity_rejected")

    def test_writable_root_rejected(self):
        self.view.f_flag = 6
        self.error("root_mount_rejected")
        self.assertEqual(self.root_rules, [])

    def test_suid_root_rejected(self):
        self.view.f_flag = 5
        self.error("root_mount_rejected")

    def test_device_enabled_root_rejected(self):
        self.view.f_flag = 3
        self.error("root_mount_rejected")

    def test_noexec_root_rejected_for_runtime_contract(self):
        self.view.f_flag = 15
        self.error("root_mount_rejected")

    def test_root_rule_failure_does_not_restrict_and_closes_once(self):
        self.add_result = -1
        self.error("rule_rejected")
        self.rules.assert_not_called()
        self.privileges.assert_not_called()
        self.assertEqual(self.closed, [42, 20])

    def test_existing_device_rule_error_is_fixed_and_preserved(self):
        self.rules.side_effect = policy.landlock.LandlockError("device_rejected")
        self.error("device_rejected")
        self.privileges.assert_not_called()
        self.assertEqual(self.closed, [42, 20])

    def test_restrict_failure_is_terminal(self):
        self.restrict_result = -1
        self.error("restrict_failed")
        self.error("already_attempted")
        self.assertEqual(self.closed, [42, 20])

    def test_unexpected_exception_text_is_suppressed(self):
        self.open_path.side_effect = OSError(PRIVATE)
        self.error("rule_rejected")
        self.assertEqual(self.closed, [20])

    def test_root_close_failure_does_not_retry_close(self):
        original = self.os.close.side_effect

        def failed(descriptor):
            original(descriptor)
            if descriptor == 42:
                raise OSError(PRIVATE)

        self.os.close.side_effect = failed
        self.error("close_failed")
        self.assertEqual(self.closed, [42, 20])

    def test_ruleset_close_failure_remains_terminal_after_restriction(self):
        original = self.os.close.side_effect

        def failed(descriptor):
            original(descriptor)
            if descriptor == 20:
                raise OSError(PRIVATE)

        self.os.close.side_effect = failed
        self.error("close_failed")
        self.error("already_attempted")

    def test_same_and_other_owner_reentry(self):
        self.run_policy()
        self.error("already_attempted")
        self.os.getpid.return_value = 4001
        self.error("owner_mismatch")

    def test_concurrent_reentry_does_not_unlock_other_owner(self):
        self.assertTrue(policy._SETUP_LOCK.acquire(blocking=False))
        try:
            self.error("reentry")
            self.assertTrue(policy._SETUP_LOCK.locked())
        finally:
            policy._SETUP_LOCK.release()
        self.syscall.assert_not_called()


class SourceTests(unittest.TestCase):
    def test_import_is_stdlib_and_pure_existing_modules_only(self):
        blocked = {"torch", "numpy", "transformers", "safetensors", "tokenizers", "jax"}
        self.assertFalse(any(name.split(".", 1)[0] in blocked for name in IMPORT_ADDITIONS))

    def test_old_globals_never_assigned_and_no_native_entrypoint(self):
        tree = ast.parse((ROOT / "src/lumi/posttraining_sft_oci_landlock_v1.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    self.assertFalse(isinstance(target, ast.Attribute)
                                     and isinstance(target.value, ast.Name)
                                     and target.value.id == "landlock")
        self.assertFalse(any(isinstance(node, ast.If) for node in tree.body))

    def test_source_root_rule_has_no_tmpfs_fallback_or_content_permission(self):
        tree = ast.parse((ROOT / "src/lumi/posttraining_sft_oci_landlock_v1.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "_add_root_rule")
        source = ast.unparse(function)
        self.assertIn("landlock.PathBeneathAttr(landlock.READ_DIR, descriptor)", source)
        self.assertNotIn("TMPFS_MAGIC", source)
        self.assertNotIn("_private_root", source)
        self.assertNotIn("READ_FILE", source)
        self.assertNotIn("WRITE_FILE", source)
        self.assertNotIn("EXECUTE", source)
        self.assertIn("observed == expected_root_identity", source)


if __name__ == "__main__":
    unittest.main()
