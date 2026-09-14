"""Bounded workspace ownership/cleanup syscall doubles; no filesystem mutations."""

from __future__ import annotations

import importlib
import os
import stat
import sys
import types
import unittest
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_synthetic_checkpoint_fixture_workspace_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
subject = importlib.import_module(PACKAGE + ".posttraining_checkpoint_fixture_workspace_v1")
PARENT = Path("/synthetic-private-harness")
DEADLINE = 1000000
TOKEN = "a" * 32
NAME = "lumi-fixture-workspace-importer-" + TOKEN


class Node:
    def __init__(self, inode, mode=stat.S_IFDIR | 0o700, *, uid=0, size=0, dev=9, mount=7):
        self.ino, self.mode, self.uid, self.gid = inode, mode, uid, uid
        self.size, self.dev, self.mount = size, dev, mount
        self.nlink = 2 if stat.S_ISDIR(mode) else 1
        self.mtime = self.ctime = 1
        self.children = {}

    def changed(self):
        self.mtime += 1
        self.ctime += 1

    def info(self):
        return types.SimpleNamespace(st_dev=self.dev, st_ino=self.ino, st_mode=self.mode,
            st_uid=self.uid, st_gid=self.gid, st_nlink=self.nlink, st_size=self.size,
            st_mtime_ns=self.mtime, st_ctime_ns=self.ctime)


class Entries:
    def __init__(self, names):
        self.names = names

    def __enter__(self):
        return iter(types.SimpleNamespace(name=name) for name in self.names)

    def __exit__(self, *_args):
        return False


class FileSystem:
    O_DIRECTORY = os.O_DIRECTORY

    def __init__(self):
        self.parent, self.proc, self.fdinfo = Node(1), Node(2), Node(3)
        self.next_inode, self.next_fd, self.pid = 4, 10, 4000
        self.fds, self.events, self.removed = {}, [], []
        self.scans, self.closes = Counter(), Counter()
        self.before = lambda *_args: None
        self.on_scan = lambda *_args: None
        self.fdinfo_bytes = lambda node: b"pos:\t0\nflags:\t0100000\nmnt_id:\t%d\n" % node.mount
        self.absolute = {str(PARENT): self.parent, "/proc": self.proc}
        self.proc.children["self/fdinfo"] = self.fdinfo

    def node(self, **kwargs):
        self.next_inode += 1
        return Node(self.next_inode, **kwargs)

    def add(self, parent, name, *, directory=True, **kwargs):
        if name in parent.children:
            raise AssertionError("synthetic_target_exists")
        mode = kwargs.pop("mode", (stat.S_IFDIR | 0o700) if directory else (stat.S_IFREG | 0o600))
        value = self.node(mode=mode, **kwargs)
        parent.children[name] = value
        parent.changed()
        return value

    def event(self, operation, *args):
        self.events.append((operation, *args))
        self.before(operation, *args)

    def descriptor(self, node, flags, *, raw=None):
        self.next_fd += 1
        self.fds[self.next_fd] = {"node": node, "flags": flags, "raw": raw, "offset": 0}
        return self.next_fd

    def getpid(self):
        return self.pid

    def open_absolute(self, value, *, directory):
        self.event("open_absolute", str(value))
        if not directory:
            raise AssertionError("synthetic_directory_only")
        return self.descriptor(self.absolute[str(value)], subject.READ_FLAGS | os.O_DIRECTORY)

    def open(self, name, flags, *, dir_fd):
        self.event("open", name, dir_fd, flags)
        if not flags & os.O_NOFOLLOW:
            raise AssertionError("nofollow_required")
        parent = self.fds[dir_fd]["node"]
        if parent is self.fdinfo:
            target = self.fds[int(name)]["node"]
            raw = self.fdinfo_bytes(target)
            return self.descriptor(self.node(mode=stat.S_IFREG | 0o400), flags, raw=raw)
        node = parent.children[name]
        if flags & os.O_DIRECTORY and not stat.S_ISDIR(node.mode):
            raise NotADirectoryError("synthetic")
        return self.descriptor(node, flags)

    def fstat(self, fd):
        self.event("fstat", fd)
        return self.fds[fd]["node"].info()

    def stat(self, name, *, dir_fd, follow_symlinks):
        self.event("stat", name, dir_fd)
        if follow_symlinks is not False:
            raise AssertionError("nofollow_required")
        return self.fds[dir_fd]["node"].children[name].info()

    def mkdir(self, name, mode, *, dir_fd):
        self.event("mkdir", name, dir_fd, mode)
        parent = self.fds[dir_fd]["node"]
        if name in parent.children:
            raise FileExistsError("synthetic")
        self.add(parent, name, mode=stat.S_IFDIR | mode)

    def scandir(self, fd):
        self.event("scandir", fd)
        record = self.fds[fd]
        if record["flags"] & subject.mount_root.O_PATH:
            raise AssertionError("Linux_scandir_rejects_O_PATH")
        node = record["node"]
        self.scans[node.ino] += 1
        self.on_scan(node, self.scans[node.ino])
        return Entries(list(node.children))

    def read(self, fd, count):
        self.event("read", fd, count)
        record = self.fds[fd]
        if record["raw"] is None:
            raise AssertionError("payload_read_forbidden")
        offset = record["offset"]
        raw = record["raw"][offset:offset + count]
        record["offset"] += len(raw)
        return raw

    def rmdir(self, name, *, dir_fd):
        self.event("rmdir", name, dir_fd)
        parent = self.fds[dir_fd]["node"]
        node = parent.children[name]
        if not stat.S_ISDIR(node.mode) or node.children:
            raise OSError("synthetic_not_empty_directory")
        self.removed.append(("directory", node.ino))
        del parent.children[name]
        parent.changed()

    def unlink(self, name, *, dir_fd):
        self.event("unlink", name, dir_fd)
        parent = self.fds[dir_fd]["node"]
        node = parent.children[name]
        if not stat.S_ISREG(node.mode):
            raise AssertionError("synthetic_regular_only")
        self.removed.append(("file", node.ino))
        del parent.children[name]
        parent.changed()

    def close(self, fd):
        self.closes[fd] += 1
        self.event("close", fd)
        if fd not in self.fds:
            raise AssertionError("repeated_close")
        del self.fds[fd]


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.fs = FileSystem()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(subject, "os", self.fs))
        self.runtime = self.stack.enter_context(patch.object(subject.runtime, "_runtime"))
        self.stack.enter_context(patch.object(subject.rootfs, "_open_absolute",
                                              side_effect=self.fs.open_absolute))
        self.filesystem = self.stack.enter_context(patch.object(subject.mount_root, "_filesystem"))
        self.clock = self.stack.enter_context(patch.object(subject.time, "monotonic_ns",
                                                          return_value=1))
        self.stack.enter_context(patch.object(subject.uuid, "uuid4",
                                              return_value=types.SimpleNamespace(hex=TOKEN)))

    def create(self, role="importer"):
        return subject.create_workspace(PARENT, role=role, deadline_ns=DEADLINE)

    def cleanup(self, workspace, **kwargs):
        return workspace.cleanup(**({"descendants_reaped": True, "native_mounts_released": True,
                                     "deadline_ns": DEADLINE} | kwargs))

    def tree(self):
        workspace = self.create()
        root = self.fs.parent.children[NAME]
        checkpoint = self.fs.add(root, "checkpoint", uid=10001)
        nested = self.fs.add(checkpoint, "model", uid=10001)
        self.fs.add(nested, ".metadata", directory=False, uid=10001, size=11)
        self.fs.add(nested, "private.distcp", directory=False, uid=10001, size=19)
        tokenizer = self.fs.add(root, "tokenizer", uid=10001)
        self.fs.add(tokenizer, "vocab.json", directory=False, uid=10001, size=7)
        output = self.fs.add(root, "output", uid=10001)
        attempt = self.fs.add(output, "import", uid=10001)
        self.fs.add(attempt, "attempt-001", uid=10001)
        self.fs.add(output, ".marker", directory=False, uid=10001, size=0,
                    mode=stat.S_IFREG | 0o400)
        self.fs.add(root, "mountpoint")
        return workspace, root, checkpoint, output

    def rejected(self, workspace, code="scope_rejected", *, no_removals=True, **kwargs):
        with self.assertRaises(subject.WorkspaceError) as raised:
            self.cleanup(workspace, **kwargs)
        self.assertEqual(raised.exception.code, code)
        self.assertEqual(str(raised.exception), code)
        if no_removals:
            self.assertEqual(self.fs.removed, [])
        self.assertEqual(self.fs.fds, {})
        with self.assertRaisesRegex(subject.WorkspaceError, "already_attempted"):
            self.cleanup(workspace)

    def test_fresh_exclusive_root_owned_private_workspace(self):
        workspace = self.create()
        self.assertEqual(workspace.path, PARENT / NAME)
        root = self.fs.parent.children[NAME]
        self.assertEqual((root.uid, root.gid, stat.S_IMODE(root.mode)), (0, 0, 0o700))
        self.assertEqual(len(self.fs.fds), 2)
        self.assertEqual(sum(event[0] == "mkdir" for event in self.fs.events), 1)
        self.cleanup(workspace)

    def test_empty_generator_cleanup_is_aggregate_only_and_one_shot(self):
        workspace = self.create(role="generator")
        receipt = self.cleanup(workspace)
        self.assertEqual(receipt["schema"], subject.SCHEMA)
        self.assertEqual((receipt["status"], receipt["role"]), ("removed", "generator"))
        self.assertEqual(receipt["census_nodes"], 1)
        self.assertEqual(receipt["removed_directories"], 1)
        self.assertEqual(receipt["removed_files"], 0)
        self.assertEqual(receipt["directory_entries_observed"], 0)
        self.assertEqual(receipt["source_payload_bytes_read"], 0)
        self.assertEqual(receipt["declared_file_bytes"], 0)
        self.assertNotIn(str(PARENT), repr(receipt))
        self.assertNotIn(TOKEN, repr(receipt))
        self.assertEqual(self.fs.fds, {})
        self.assertTrue(all(count == 1 for count in self.fs.closes.values()))
        with self.assertRaisesRegex(subject.WorkspaceError, "workspace_closed"):
            _ = workspace.path
        with self.assertRaisesRegex(subject.WorkspaceError, "already_attempted"):
            self.cleanup(workspace)

    def test_partial_layout_and_readonly_output_files_removed_bottom_up(self):
        workspace, root, _checkpoint, _output = self.tree()
        receipt = self.cleanup(workspace)
        self.assertEqual(receipt["census_nodes"], 12)
        self.assertEqual((receipt["removed_files"], receipt["removed_directories"]), (4, 8))
        self.assertEqual(receipt["declared_file_bytes"], 37)
        self.assertEqual(receipt["directory_entries_observed"], 22)
        self.assertEqual(self.fs.removed[-1], ("directory", root.ino))
        self.assertEqual(self.fs.parent.children, {})
        self.assertEqual(self.fs.fds, {})
        for key in ("descendant_reaping_verified_by_adapter",
                    "native_mount_release_verified_by_adapter",
                    "adopted_existing_directory", "external_mutation_excluded", "retry_authorized",
                    "production_eligible"):
            self.assertIs(receipt[key], False)
        for key in ("workspace_removed", "identity_checked_removal", "mount_boundary_checks_passed",
                    "descendants_reaped_premise", "native_mounts_released_premise"):
            self.assertIs(receipt[key], True)

    def test_directory_descriptors_are_readable_and_regular_pins_are_o_path(self):
        workspace, _root, _checkpoint, _output = self.tree()
        self.cleanup(workspace)
        calls = [event for event in self.fs.events if event[0] == "open"]
        directories = [event for event in calls if event[3] & os.O_DIRECTORY]
        self.assertTrue(directories)
        self.assertTrue(all(not event[3] & subject.mount_root.O_PATH for event in directories))
        files = [event for event in calls
                 if event[1] in (".metadata", "private.distcp", "vocab.json")]
        self.assertTrue(files)
        self.assertTrue(all(event[3] & subject.mount_root.O_PATH for event in files))

    def test_unknown_top_level_or_unremoved_software_root_rejects_before_deletion(self):
        workspace, root, _checkpoint, _output = self.tree()
        self.fs.add(root, "owned-software-root-must-be-closed")
        self.rejected(workspace)

    def test_top_level_regular_file_is_not_a_valid_layout_sibling(self):
        workspace = self.create()
        self.fs.add(self.fs.parent.children[NAME], "checkpoint", directory=False)
        self.rejected(workspace)

    def test_symlink_special_file_and_hardlink_shapes_reject_before_deletion(self):
        for mode in (stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o600, stat.S_IFCHR | 0o600,
                     stat.S_IFSOCK | 0o600, stat.S_IFBLK | 0o600):
            with self.subTest(mode=mode):
                self.setUp()
                workspace, _root, checkpoint, _output = self.tree()
                self.fs.add(checkpoint, "unsafe", mode=mode)
                self.rejected(workspace)

    def test_regular_file_hardlink_rejects(self):
        workspace, _root, checkpoint, _output = self.tree()
        self.fs.add(checkpoint, "aliased", directory=False).nlink = 2
        self.rejected(workspace)

    def test_directory_inode_alias_rejects(self):
        workspace, root, checkpoint, _output = self.tree()
        checkpoint.children["alias"] = root.children["tokenizer"]
        self.rejected(workspace)

    def test_cross_device_node_rejects_before_deletion(self):
        workspace, _root, checkpoint, _output = self.tree()
        self.fs.add(checkpoint, "foreign", directory=False, dev=10)
        self.rejected(workspace)

    def test_same_device_bind_mount_rejects_before_deletion(self):
        workspace, _root, checkpoint, _output = self.tree()
        self.fs.add(checkpoint, "foreign", mount=8)
        self.rejected(workspace, "mount_rejected")

    def test_workspace_parent_mount_mismatch_rejects(self):
        workspace = self.create()
        self.fs.parent.children[NAME].mount = 8
        self.rejected(workspace, "mount_rejected")

    def test_exact_size_node_and_depth_caps_are_accepted(self):
        workspace = self.create()
        root = self.fs.parent.children[NAME]
        current = self.fs.add(root, "checkpoint", uid=10001)
        for index in range(14):
            current = self.fs.add(current, "nested-" + str(index), uid=10001)
        # Root plus 15 directories plus 496 regular files is exactly 512 nodes.
        for index in range(496):
            self.fs.add(current, "file-" + str(index), directory=False, uid=10001,
                        size=subject.MAXIMUM_FILE_BYTES if index == 0 else 0)
        receipt = self.cleanup(workspace)
        self.assertEqual(receipt["census_nodes"], 512)
        self.assertEqual(receipt["declared_file_bytes"], 16777216)
        self.assertEqual(receipt["directory_entries_observed"], 1022)
        self.assertLessEqual(receipt["kernel_metadata_read_calls"], 4096)
        self.assertLessEqual(receipt["kernel_metadata_bytes_read"], 4096 * 1026)

    def test_one_over_size_cap_rejects_before_deletion(self):
        workspace = self.create()
        checkpoint = self.fs.add(self.fs.parent.children[NAME], "checkpoint")
        self.fs.add(checkpoint, "oversize", directory=False, size=16777217)
        self.rejected(workspace, "bounds_exhausted")

    def test_one_over_node_cap_rejects_before_deletion(self):
        workspace = self.create()
        checkpoint = self.fs.add(self.fs.parent.children[NAME], "checkpoint")
        for index in range(511):
            self.fs.add(checkpoint, "node-" + str(index), directory=False)
        self.rejected(workspace, "bounds_exhausted")

    def test_one_over_depth_cap_rejects_before_deletion(self):
        workspace = self.create()
        current = self.fs.add(self.fs.parent.children[NAME], "checkpoint")
        for index in range(16):
            current = self.fs.add(current, "nested-" + str(index))
        self.rejected(workspace, "bounds_exhausted")

    def test_second_census_detects_new_child_before_any_deletion(self):
        workspace, root, _checkpoint, _output = self.tree()

        def mutate(node, count):
            if node is root and count == 2:
                self.fs.add(root, "unknown-new-child")

        self.fs.on_scan = mutate
        self.rejected(workspace, "identity_changed")

    def test_second_census_entry_observations_are_globally_bounded(self):
        workspace, root, _checkpoint, _output = self.tree()

        def mutate(node, count):
            if node is root and count == 2:
                for index in range(513):
                    self.fs.add(root, "added-" + str(index))

        self.fs.on_scan = mutate
        self.rejected(workspace, "bounds_exhausted")
        self.assertLessEqual(workspace._directory_entries, 1025)

    def test_entry_budget_spans_multiple_directories_and_revalidation(self):
        workspace = self.create()
        root = self.fs.parent.children[NAME]
        checkpoint = self.fs.add(root, "checkpoint")
        output = self.fs.add(root, "output")
        for index in range(255):
            self.fs.add(checkpoint, "file-" + str(index), directory=False)
        for index in range(254):
            self.fs.add(output, "file-" + str(index), directory=False)

        def mutate(node, count):
            if node is output and count == 2:
                for index in range(3):
                    self.fs.add(output, "new-" + str(index), directory=False)

        self.fs.on_scan = mutate
        self.rejected(workspace, "bounds_exhausted")
        self.assertEqual(workspace._directory_entries, 1025)
        self.assertLess(len(output.children), 512)

    def test_replaced_file_identity_is_never_deleted(self):
        workspace = self.create()
        checkpoint = self.fs.add(self.fs.parent.children[NAME], "checkpoint")
        old = self.fs.add(checkpoint, "private", directory=False)
        opens = 0

        def mutate(operation, *args):
            nonlocal opens
            if operation == "open" and args[0] == "private":
                opens += 1
                if opens == 2:
                    checkpoint.children["private"] = self.fs.node(mode=stat.S_IFREG | 0o600)

        self.fs.before = mutate
        self.rejected(workspace, "identity_changed")
        self.assertNotEqual(checkpoint.children["private"].ino, old.ino)

    def test_replaced_directory_identity_is_never_deleted(self):
        workspace = self.create()
        root = self.fs.parent.children[NAME]
        old = self.fs.add(root, "checkpoint")
        opens = 0

        def mutate(operation, *args):
            nonlocal opens
            if operation == "open" and args[0] == "checkpoint":
                opens += 1
                if opens == 2:
                    root.children["checkpoint"] = self.fs.node()

        self.fs.before = mutate
        self.rejected(workspace, "identity_changed")
        self.assertNotEqual(root.children["checkpoint"].ino, old.ino)

    def test_new_child_at_removal_is_preserved(self):
        workspace = self.create()
        checkpoint = self.fs.add(self.fs.parent.children[NAME], "checkpoint")

        def mutate(node, count):
            if node is checkpoint and count == 3:
                self.fs.add(checkpoint, "late-private", directory=False)

        self.fs.on_scan = mutate
        self.rejected(workspace, "identity_changed")
        self.assertIn("late-private", checkpoint.children)

    def test_nonliteral_or_false_premises_consume_attempt_without_scanning(self):
        for key, value in (("descendants_reaped", False), ("descendants_reaped", 1),
                           ("native_mounts_released", False), ("native_mounts_released", "true")):
            with self.subTest(key=key, value=value):
                self.setUp()
                workspace = self.create()
                self.rejected(workspace, "arguments_rejected", **{key: value})
                self.assertFalse(any(event[0] in ("read", "scandir") for event in self.fs.events))

    def test_initial_cleanup_deadline_is_one_shot_and_preserves_workspace(self):
        workspace = self.create()
        self.clock.return_value = DEADLINE
        self.rejected(workspace, "deadline_exhausted")
        self.assertIn(NAME, self.fs.parent.children)

    def test_late_cleanup_deadline_stops_before_deletion(self):
        workspace, _root, _checkpoint, _output = self.tree()

        def expire(_node, _count):
            self.clock.return_value = DEADLINE

        self.fs.on_scan = expire
        self.rejected(workspace, "deadline_exhausted")

    def test_deadline_after_last_mount_read_prevents_empty_directory_removal(self):
        workspace = self.create()
        reads = 0

        def expire(operation, *_args):
            nonlocal reads
            if operation == "read":
                reads += 1
                if reads == 8:
                    self.clock.return_value = DEADLINE

        self.fs.before = expire
        self.rejected(workspace, "deadline_exhausted")
        self.assertEqual(reads, 8)

    def test_parent_resolution_deadline_prevents_workspace_creation(self):
        def expire(operation, *_args):
            if operation == "fstat":
                self.clock.return_value = DEADLINE

        self.fs.before = expire
        with self.assertRaisesRegex(subject.WorkspaceError, "deadline_exhausted"):
            self.create()
        self.assertFalse(any(event[0] == "mkdir" for event in self.fs.events))
        self.assertEqual(self.fs.fds, {})

    def test_final_removal_over_deadline_cannot_return_success_receipt(self):
        workspace = self.create()

        def expire(operation, *_args):
            if operation == "rmdir":
                self.clock.return_value = DEADLINE

        self.fs.before = expire
        self.rejected(workspace, "deadline_exhausted", no_removals=False)
        self.assertNotIn(NAME, self.fs.parent.children)
        self.assertEqual(len(self.fs.removed), 1)

    def test_cleanup_owner_mismatch_rejects_before_io(self):
        workspace = self.create()
        previous = len(self.fs.events)
        self.fs.pid += 1
        with self.assertRaisesRegex(subject.WorkspaceError, "owner_mismatch"):
            self.cleanup(workspace)
        self.assertEqual(len(self.fs.events), previous)
        self.assertFalse(workspace._cleanup_attempted)
        self.fs.pid -= 1
        self.cleanup(workspace)

    def test_runtime_preflight_before_create_or_cleanup_io(self):
        self.runtime.side_effect = ValueError("synthetic-private-exception")
        with self.assertRaisesRegex(subject.WorkspaceError, "runtime_rejected"):
            self.create()
        self.assertEqual(self.fs.events, [])
        self.runtime.side_effect = None
        workspace = self.create()
        previous = len(self.fs.events)
        self.runtime.side_effect = ValueError("synthetic-private-exception")
        with self.assertRaisesRegex(subject.WorkspaceError, "runtime_rejected"):
            self.cleanup(workspace)
        self.assertEqual(len(self.fs.events), previous)
        self.runtime.side_effect = None
        self.cleanup(workspace)

    def test_parent_path_role_and_deadline_reject_before_io(self):
        for parent, role, deadline in ((PARENT, [], DEADLINE), (PARENT, "other", DEADLINE),
                (str(PARENT), "importer", DEADLINE), (Path("/"), "importer", DEADLINE),
                (Path("relative"), "importer", DEADLINE), (Path("/a/../b"), "importer", DEADLINE),
                (Path("/bad\\name"), "importer", DEADLINE), (PARENT, "importer", True),
                (PARENT, "importer", 0)):
            with self.subTest(role=role, parent=str(parent), deadline=deadline):
                with self.assertRaisesRegex(subject.WorkspaceError, "arguments_rejected"):
                    subject.create_workspace(parent, role=role, deadline_ns=deadline)
        self.assertEqual(self.fs.events, [])

    def test_nonprivate_or_nonroot_parent_never_creates_workspace(self):
        self.fs.parent.mode = stat.S_IFDIR | 0o755
        with self.assertRaisesRegex(subject.WorkspaceError, "scope_rejected"):
            self.create()
        self.assertFalse(any(event[0] == "mkdir" for event in self.fs.events))
        self.assertEqual(self.fs.fds, {})

    def test_existing_directory_is_not_adopted_or_deleted(self):
        old = self.fs.add(self.fs.parent, NAME)
        self.fs.add(old, "prior-evidence", directory=False, size=44)
        with self.assertRaisesRegex(subject.WorkspaceError, "creation_failed"):
            self.create()
        self.assertIs(self.fs.parent.children[NAME], old)
        self.assertIn("prior-evidence", old.children)
        self.assertEqual(self.fs.removed, [])
        self.assertEqual(self.fs.fds, {})

    def test_failed_open_after_known_creation_removes_only_empty_owned_inode(self):
        def fail(operation, *args):
            if operation == "open" and args[0] == NAME:
                raise OSError("synthetic-private-error")

        self.fs.before = fail
        with self.assertRaisesRegex(subject.WorkspaceError, "creation_failed"):
            self.create()
        self.assertEqual(len(self.fs.removed), 1)
        self.assertNotIn(NAME, self.fs.parent.children)
        self.assertEqual(self.fs.fds, {})

    def test_failed_identity_capture_never_adopts_new_path_for_cleanup(self):
        def fail(operation, *_args):
            if operation == "stat":
                raise OSError("synthetic-private-error")

        self.fs.before = fail
        with self.assertRaisesRegex(subject.WorkspaceError, "creation_cleanup_unconfirmed"):
            self.create()
        self.assertIn(NAME, self.fs.parent.children)
        self.assertEqual(self.fs.removed, [])
        self.assertEqual(self.fs.fds, {})

    def test_failed_creation_cleanup_preserves_replacement_inode(self):
        def fail(operation, *args):
            if operation == "open" and args[0] == NAME:
                self.fs.parent.children[NAME] = self.fs.node()
                raise OSError("synthetic-private-error")

        self.fs.before = fail
        with self.assertRaisesRegex(subject.WorkspaceError, "creation_cleanup_unconfirmed"):
            self.create()
        self.assertIn(NAME, self.fs.parent.children)
        self.assertEqual(self.fs.removed, [])
        self.assertEqual(self.fs.fds, {})

    def test_failed_creation_does_not_traverse_nonempty_new_workspace(self):
        def fail(operation, *args):
            if operation == "open" and args[0] == NAME:
                self.fs.add(self.fs.parent.children[NAME], "unexpected-new-evidence")
                raise OSError("synthetic-private-error")

        self.fs.before = fail
        with self.assertRaisesRegex(subject.WorkspaceError, "creation_cleanup_unconfirmed"):
            self.create()
        self.assertEqual(self.fs.removed, [])
        self.assertIn("unexpected-new-evidence", self.fs.parent.children[NAME].children)
        self.assertFalse(any(event[0] == "scandir" for event in self.fs.events))

    def test_ambiguous_close_is_never_repeated(self):
        workspace = self.create()
        failed = False

        def fail(operation, *_args):
            nonlocal failed
            if operation == "close" and not failed:
                failed = True
                raise OSError("synthetic-private-error")

        self.fs.before = fail
        with self.assertRaisesRegex(subject.WorkspaceError, "close_failed"):
            self.cleanup(workspace)
        self.assertTrue(all(count == 1 for count in self.fs.closes.values()))
        self.assertEqual(self.fs.removed, [])
        with self.assertRaisesRegex(subject.WorkspaceError, "already_attempted"):
            self.cleanup(workspace)

    def test_deletion_failure_is_not_retried(self):
        workspace, _root, _checkpoint, _output = self.tree()

        def fail(operation, *_args):
            if operation == "unlink":
                raise OSError("synthetic-private-error")

        self.fs.before = fail
        self.rejected(workspace, "cleanup_failed", no_removals=False)
        self.assertEqual(sum(event[0] == "unlink" for event in self.fs.events), 1)

    def test_proc_filesystem_authentication_failure_never_reads_or_deletes(self):
        workspace = self.create()
        self.filesystem.side_effect = ValueError("synthetic-private-error")
        self.rejected(workspace, "cleanup_failed")
        self.assertFalse(any(event[0] == "read" for event in self.fs.events))

    def test_malformed_duplicate_oversize_or_absent_mount_id_rejects(self):
        for raw in (b"", b"pos:\t0\n", b"mnt_id:\t7\nmnt_id:\t7\n", b"mnt_id:\t0\n",
                    b"mnt_id:\t-7\n", b"mnt_id:\t18446744073709551616\n", b"z" * 1025):
            with self.subTest(raw_shape=len(raw)):
                self.setUp()
                workspace = self.create()
                self.fs.fdinfo_bytes = lambda _node, data=raw: data
                self.rejected(workspace, "mount_rejected")

    def test_kernel_read_budget_is_charged_before_read(self):
        workspace = self.create()
        workspace._kernel_calls = subject.MAXIMUM_KERNEL_READ_CALLS
        self.rejected(workspace, "bounds_exhausted")
        self.assertFalse(any(event[0] == "read" for event in self.fs.events))

    def test_unauthorized_constructor_cannot_adopt_existing_directory(self):
        with self.assertRaisesRegex(subject.WorkspaceError, "arguments_rejected"):
            subject.OwnedFixtureWorkspace(None, PARENT, "importer", NAME)
        self.assertEqual(self.fs.events, [])

    def test_pure_receipt_validator_requires_every_typed_field_and_exact_role(self):
        workspace = self.create()
        value = workspace.cleanup(descendants_reaped=True, native_mounts_released=True,
                                  deadline_ns=DEADLINE)
        copied = subject.validate_cleanup_receipt(value, role="importer")
        self.assertEqual(copied, value)
        self.assertIsNot(copied, value)
        for key in value:
            selected = dict(value)
            del selected[key]
            with self.assertRaises(subject.WorkspaceError):
                subject.validate_cleanup_receipt(selected, role="importer")
        with self.assertRaises(subject.WorkspaceError):
            subject.validate_cleanup_receipt(value, role="generator")

    def test_pure_receipt_validator_rejects_absent_reads_and_inconsistent_census(self):
        workspace = self.create()
        value = workspace.cleanup(descendants_reaped=True, native_mounts_released=True,
                                  deadline_ns=DEADLINE)
        for key, part in (("kernel_metadata_read_calls", 0), ("kernel_metadata_bytes_read", 0),
                          ("kernel_metadata_read_calls", 9), ("census_nodes", True),
                          ("removed_directories", 2), ("directory_entries_observed", 1),
                          ("production_eligible", True), ("private_path", "never public")):
            with self.assertRaises(subject.WorkspaceError):
                subject.validate_cleanup_receipt(dict(value, **{key: part}), role="importer")


if __name__ == "__main__":
    unittest.main()
