"""Temporary local files and injected errors only; no provider/private inputs."""

from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/lumi/posttraining_sft_host_journal_v1.py"
SPEC = importlib.util.spec_from_file_location("_synthetic_host_journal_v1", MODULE)
assert SPEC is not None and SPEC.loader is not None
journal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(journal)
ATTEMPT = "a" * 64
RECORD = b'{"event":"synthetic"}\n'
SUMMARY_KEYS = {
    "claimed", "failed", "closed", "records_reserved", "records_synced",
    "capacity_reserved_bytes", "write_calls", "fsync_calls", "close_calls",
}


class JournalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lumi-journal-synthetic-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.directory.chmod(0o700)
        self.root_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.addCleanup(os.close, self.root_fd)
        self.path = self.directory / (ATTEMPT + ".lifecycle.jsonl")

    def make(self, identifier=ATTEMPT):
        result = journal.Journal(self.root_fd, identifier)
        self.addCleanup(result.close)
        return result

    def rejected(self, callable_, *arguments):
        with self.assertRaises(journal.JournalError) as caught:
            callable_(*arguments)
        self.assertEqual(caught.exception.code, "journal_rejected")
        self.assertEqual(str(caught.exception), "journal_rejected")
        self.assertNotIn("SECRET", repr(caught.exception))

    def capture_constructor(self):
        created = []
        original = journal.Journal

        class CapturedJournal(original):
            def __init__(self, *arguments):
                created.append(self)
                super().__init__(*arguments)

        return created, patch.object(journal, "Journal", CapturedJournal)

    def test_exclusive_empty_claim_has_private_metadata_and_cloexec_ownership(self):
        item = self.make()
        self.assertTrue(self.path.is_file())
        self.assertEqual(self.path.read_bytes(), b"")
        info = self.path.stat()
        self.assertEqual(stat.S_IMODE(info.st_mode), 0o600)
        self.assertEqual(info.st_uid, os.getuid())
        self.assertEqual(info.st_nlink, 1)
        self.assertNotEqual(item._root_fd, self.root_fd)
        self.assertFalse(os.get_inheritable(item._root_fd))
        self.assertFalse(os.get_inheritable(item._file_fd))
        self.assertEqual(item.summary(), {
            "claimed": True, "failed": False, "closed": False, "records_reserved": 0,
            "records_synced": 0, "capacity_reserved_bytes": 0, "write_calls": 0,
            "fsync_calls": 2, "close_calls": 0,
        })

    def test_constructor_syncs_file_then_root_before_return(self):
        calls = []
        original = os.fsync
        with patch.object(
            journal.os, "fsync", side_effect=lambda fd: calls.append(fd) or original(fd),
        ):
            item = self.make()
        self.assertEqual(calls, [item._file_fd, item._root_fd])

    def test_append_precharges_verifies_writes_once_syncs_and_rechecks(self):
        item = self.make()
        events = []
        verify, write, fsync = item._verify_identity, os.write, os.fsync

        def checked_identity():
            self.assertEqual(item.summary()["records_reserved"], 1)
            self.assertEqual(item.summary()["capacity_reserved_bytes"], 4096)
            events.append("verify")
            verify()

        with patch.object(item, "_verify_identity", side_effect=checked_identity), patch.object(
            journal.os, "write",
            side_effect=lambda fd, raw: events.append("write") or write(fd, raw),
        ), patch.object(
            journal.os, "fsync", side_effect=lambda fd: events.append("fsync") or fsync(fd),
        ):
            self.assertIsNone(item.append(RECORD))
        self.assertEqual(events, ["verify", "write", "fsync", "verify"])
        self.assertEqual(self.path.read_bytes(), RECORD)
        self.assertEqual(item.summary()["records_synced"], 1)
        self.assertEqual(item.summary()["write_calls"], 1)
        self.assertEqual(item.summary()["fsync_calls"], 3)

    def test_append_uses_current_position_and_never_reopens(self):
        item = self.make()
        with patch.object(journal.os, "open") as opened:
            item.append(RECORD)
            item.append(b"second\n")
        opened.assert_not_called()
        self.assertEqual(self.path.read_bytes(), RECORD + b"second\n")

    def test_maximum_record_and_cumulative_capacity_bounds(self):
        item = self.make()
        record = b"x" * 4095 + b"\n"
        # This test still performs each actual fsync against only its temporary file.
        for _index in range(256):
            item.append(record)
        before = item.summary()
        self.assertEqual(before["capacity_reserved_bytes"], 1_048_576)
        self.assertEqual(before["records_reserved"], 256)
        self.assertEqual(before["records_synced"], 256)
        self.assertEqual(self.path.stat().st_size, 1_048_576)
        self.rejected(item.append, b"overflow\n")
        self.assertTrue(item.summary()["failed"])
        for key in ("capacity_reserved_bytes", "records_reserved", "records_synced", "write_calls"):
            self.assertEqual(item.summary()[key], before[key])

    def test_invalid_record_framing_marks_sticky_failure_without_write_or_reservation(self):
        for index, record in enumerate((b"", b"missing newline", b"two\nlines\n",
                                        b"x" * 4096 + b"\n",
                                        "text\n", bytearray(b"text\n"), None, True)):
            item = self.make(f"{index:064x}")
            with self.subTest(type_name=type(record).__name__), patch.object(
                journal.os, "write",
            ) as write:
                self.rejected(item.append, record)
                self.rejected(item.append, RECORD)
                write.assert_not_called()
                self.assertEqual(item.summary()["records_reserved"], 0)
                self.assertTrue(item.summary()["failed"])

    def test_newline_only_is_valid_framing_but_not_semantic_schema_validation(self):
        item = self.make()
        item.append(b"\n")
        self.assertEqual(self.path.read_bytes(), b"\n")

    def test_short_zero_and_exception_write_are_sticky_without_retry_or_sync(self):
        for index, outcome in enumerate((0, 1, OSError("SECRET write"), InterruptedError(), True)):
            item = self.make(f"{index:064x}")
            behavior = {"side_effect": outcome} if isinstance(outcome, BaseException) else {
                "return_value": outcome,
            }
            with self.subTest(outcome_type=type(outcome).__name__), patch.object(
                journal.os, "write", **behavior,
            ) as write, patch.object(journal.os, "fsync") as fsync:
                self.rejected(item.append, RECORD)
                self.rejected(item.append, RECORD)
                write.assert_called_once()
                fsync.assert_not_called()
                self.assertEqual(item.summary()["records_reserved"], 1)
                self.assertEqual(item.summary()["capacity_reserved_bytes"], 4096)
                self.assertEqual(item.summary()["records_synced"], 0)

    def test_actual_partial_write_is_preserved_as_consumed_claim(self):
        item = self.make()
        write = os.write
        with patch.object(
            journal.os, "write", side_effect=lambda fd, raw: write(fd, raw[:3]),
        ) as call:
            self.rejected(item.append, RECORD)
        call.assert_called_once()
        self.assertTrue(item.close())
        self.assertEqual(self.path.read_bytes(), RECORD[:3])
        self.rejected(journal.Journal, self.root_fd, ATTEMPT)
        self.assertEqual(self.path.read_bytes(), RECORD[:3])

    def test_append_fsync_failure_preserves_full_bytes_and_never_retries(self):
        item = self.make()
        with patch.object(journal.os, "fsync", side_effect=OSError("SECRET fsync")) as fsync:
            self.rejected(item.append, RECORD)
            self.rejected(item.append, RECORD)
        fsync.assert_called_once()
        self.assertEqual(self.path.read_bytes(), RECORD)
        self.assertEqual(item.summary()["fsync_calls"], 3)
        self.assertEqual(item.summary()["records_synced"], 0)

    def test_existing_claim_survives_close_and_cannot_be_reopened(self):
        item = self.make()
        item.append(RECORD)
        self.assertTrue(item.close())
        self.rejected(journal.Journal, self.root_fd, ATTEMPT)
        self.assertEqual(self.path.read_bytes(), RECORD)
        self.rejected(item.append, RECORD)

    def test_preexisting_empty_regular_symlink_and_hardlink_collisions_never_overwrite(self):
        self.path.write_bytes(b"synthetic existing")
        self.rejected(journal.Journal, self.root_fd, ATTEMPT)
        self.assertEqual(self.path.read_bytes(), b"synthetic existing")
        for identifier, kind in (("b" * 64, "symlink"), ("c" * 64, "hardlink")):
            target = self.directory / (identifier + ".lifecycle.jsonl")
            if kind == "symlink":
                target.symlink_to(self.path.name)
            else:
                os.link(self.path, target)
            self.rejected(journal.Journal, self.root_fd, identifier)
            self.assertEqual(self.path.read_bytes(), b"synthetic existing")

    def test_bad_root_mode_and_nondirectory_descriptor_rejected(self):
        self.directory.chmod(0o750)
        self.rejected(journal.Journal, self.root_fd, ATTEMPT)
        self.assertFalse(self.path.exists())
        self.directory.chmod(0o700)
        with tempfile.TemporaryFile(dir=self.directory) as plain:
            self.rejected(journal.Journal, plain.fileno(), ATTEMPT)
        self.assertFalse(self.path.exists())

    def test_root_identity_change_before_append_reserves_but_does_not_write(self):
        item = self.make()
        item._root_identity = (0, 0)
        with patch.object(journal.os, "write") as write:
            self.rejected(item.append, RECORD)
        write.assert_not_called()
        self.assertEqual(item.summary()["records_reserved"], 1)

    def test_stat_errors_precharge_once_without_write_or_retry(self):
        for index, method in enumerate(("stat", "fstat")):
            item = self.make(f"{index:064x}")
            with self.subTest(method=method), patch.object(
                journal.os, method, side_effect=OSError("SECRET metadata"),
            ) as metadata, patch.object(journal.os, "write") as write:
                self.rejected(item.append, RECORD)
                self.rejected(item.append, RECORD)
                metadata.assert_called_once()
                write.assert_not_called()
            self.assertEqual(item.summary()["capacity_reserved_bytes"], 4096)

    def test_file_mode_hardlink_and_size_changes_block_append(self):
        for index, mutation in enumerate(("mode", "link", "size")):
            identifier = f"{index:064x}"
            item = self.make(identifier)
            path = self.directory / (identifier + ".lifecycle.jsonl")
            if mutation == "mode":
                path.chmod(0o640)
            elif mutation == "link":
                os.link(path, self.directory / "extra-link")
            else:
                path.write_bytes(b"external")
            with self.subTest(mutation=mutation), patch.object(journal.os, "write") as write:
                self.rejected(item.append, RECORD)
                write.assert_not_called()

    def test_path_replacement_symlink_or_regular_is_rejected_without_overwrite(self):
        for index, replacement in enumerate(("regular", "symlink")):
            identifier = f"{index:064x}"
            item = self.make(identifier)
            path = self.directory / (identifier + ".lifecycle.jsonl")
            moved = self.directory / ("original-" + str(index))
            path.rename(moved)
            if replacement == "regular":
                path.write_bytes(b"SECRET replacement")
            else:
                path.symlink_to(moved.name)
            with self.subTest(replacement=replacement), patch.object(journal.os, "write") as write:
                self.rejected(item.append, RECORD)
                write.assert_not_called()
            self.assertEqual(moved.read_bytes(), b"")

    def test_identity_replacement_during_write_prevents_synced_record_acceptance(self):
        item = self.make()
        write = os.write
        moved = self.directory / "original"

        def replacing(fd, raw):
            result = write(fd, raw)
            self.path.rename(moved)
            self.path.write_bytes(b"SECRET replacement")
            return result

        with patch.object(journal.os, "write", side_effect=replacing):
            self.rejected(item.append, RECORD)
        self.assertEqual(item.summary()["records_synced"], 0)
        self.assertEqual(moved.read_bytes(), RECORD)
        self.assertEqual(self.path.read_bytes(), b"SECRET replacement")

    def test_bad_owner_or_file_type_stat_metadata_rejected(self):
        item = self.make()
        original = os.fstat

        def wrong_owner(fd):
            observed = original(fd)
            if fd == item._file_fd:
                return types.SimpleNamespace(
                    st_mode=observed.st_mode, st_uid=os.getuid() + 1,
                    st_nlink=observed.st_nlink, st_size=observed.st_size,
                    st_dev=observed.st_dev, st_ino=observed.st_ino,
                )
            return observed

        with patch.object(journal.os, "fstat", side_effect=wrong_owner):
            self.rejected(item.append, RECORD)
        self.assertEqual(item.summary()["write_calls"], 0)

    def test_root_mode_change_or_cloexec_removal_rejected(self):
        item = self.make()
        self.directory.chmod(0o710)
        self.rejected(item.append, RECORD)
        self.directory.chmod(0o700)
        other = self.make("b" * 64)
        os.set_inheritable(other._file_fd, True)
        self.rejected(other.append, RECORD)
        self.assertEqual(other.summary()["write_calls"], 0)

    def test_constructor_file_or_directory_fsync_failure_leaves_empty_claim_and_closes(self):
        for index in (0, 1):
            identifier = f"{index:064x}"
            captured, capture = self.capture_constructor()
            with capture, patch.object(journal.os, "fsync", side_effect=[None] * index + [OSError]):
                self.rejected(journal.Journal, self.root_fd, identifier)
            item = captured[0]
            self.assertTrue(item.summary()["claimed"])
            self.assertTrue(item.summary()["failed"])
            self.assertTrue(item.summary()["closed"])
            self.assertEqual(item.summary()["close_calls"], 2)
            self.assertEqual(item.summary()["fsync_calls"], index + 1)
            self.rejected(journal.Journal, self.root_fd, identifier)

    def test_duplicate_and_open_errors_close_only_descriptors_actually_owned(self):
        captured, capture = self.capture_constructor()
        with capture, patch.object(journal.fcntl, "fcntl", side_effect=OSError("SECRET duplicate")):
            self.rejected(journal.Journal, self.root_fd, ATTEMPT)
        self.assertEqual(captured[0].summary()["close_calls"], 0)
        with capture, patch.object(journal.os, "open", side_effect=OSError("SECRET open")):
            self.rejected(journal.Journal, self.root_fd, ATTEMPT)
        self.assertEqual(captured[-1].summary()["close_calls"], 1)
        os.fstat(self.root_fd)

    def test_duplicate_result_cannot_transfer_ownership_of_borrowed_root(self):
        captured, capture = self.capture_constructor()
        with capture, patch.object(journal.fcntl, "fcntl", return_value=self.root_fd):
            self.rejected(journal.Journal, self.root_fd, ATTEMPT)
        self.assertEqual(captured[0].summary()["close_calls"], 0)
        self.assertFalse(self.path.exists())
        os.fstat(self.root_fd)

    def test_constructor_fstat_error_after_claim_preserves_file_and_closes_owned_fds(self):
        captured, capture = self.capture_constructor()
        original = os.fstat
        count = 0

        def fail_after_claim(fd):
            nonlocal count
            count += 1
            if count == 3:
                raise OSError("SECRET fstat")
            return original(fd)

        with capture, patch.object(journal.os, "fstat", side_effect=fail_after_claim):
            self.rejected(journal.Journal, self.root_fd, ATTEMPT)
        self.assertTrue(self.path.exists())
        self.assertEqual(captured[0].summary()["close_calls"], 2)

    def test_close_preserves_borrowed_root_and_repeated_close_has_no_extra_calls(self):
        item = self.make()
        self.assertTrue(item.close())
        self.assertTrue(item.close())
        self.assertEqual(item.summary()["close_calls"], 2)
        os.fstat(self.root_fd)
        self.assertTrue(self.path.exists())

    def test_failed_close_still_attempts_other_fd_once_and_never_retries(self):
        for index, fail_name in enumerate(("file", "root")):
            item = self.make(f"{index:064x}")
            target = getattr(item, "_" + fail_name + "_fd")
            original = os.close
            calls = []

            def fail_one(fd, *, calls=calls, original=original, target=target):
                calls.append(fd)
                original(fd)
                if fd == target:
                    # Model ambiguous close completion without leaking a fixture descriptor.
                    raise OSError("SECRET close")

            with patch.object(journal.os, "close", side_effect=fail_one):
                self.assertFalse(item.close())
                self.assertFalse(item.close())
            self.assertEqual(calls, [item._file_fd, item._root_fd])
            self.assertEqual(item.summary()["close_calls"], 2)
            self.assertTrue(item.summary()["failed"])
            self.assertFalse(item.summary()["closed"])

    def test_foreign_process_cannot_append_but_may_close_descriptor_copies(self):
        item = self.make()
        with patch.object(journal.os, "getpid", return_value=item._owner_pid + 1), patch.object(
            journal.os, "write",
        ) as write, patch.object(journal.os, "fsync") as sync:
            self.rejected(item.append, RECORD)
            self.assertTrue(item.close())
        write.assert_not_called()
        sync.assert_not_called()
        self.assertEqual(self.path.read_bytes(), b"")
        self.assertEqual(item.summary()["records_reserved"], 0)

    def test_owner_is_rechecked_immediately_before_write_and_fsync(self):
        for index, boundary in enumerate(("identity", "write")):
            item = self.make(f"{index:064x}")
            original = item._verify_identity if boundary == "identity" else os.write

            def changed(*arguments, original=original, item=item):
                result = original(*arguments)
                item._owner_pid += 1
                return result

            target, name = (item, "_verify_identity") if boundary == "identity" else (
                journal.os, "write",
            )
            with self.subTest(boundary=boundary), patch.object(target, name, side_effect=changed):
                self.rejected(item.append, RECORD)
            self.assertEqual(item.summary()["write_calls"], int(boundary == "write"))
            self.assertEqual(item.summary()["fsync_calls"], 2)
            self.assertEqual(item.summary()["records_synced"], 0)

    def test_unrelated_threads_do_not_block_the_owned_thread(self):
        item = self.make()
        with patch.object(journal.threading, "active_count", return_value=2):
            item.append(RECORD)
        self.assertEqual(item.summary()["records_synced"], 1)

    def test_thread_object_identity_not_reusable_numeric_ident_controls_write_and_close(self):
        item = self.make()
        foreign = types.SimpleNamespace(ident=item._owner_thread.ident)
        with patch.object(journal.threading, "current_thread", return_value=foreign):
            self.rejected(item.append, RECORD)
            self.assertFalse(item.close())
        self.assertEqual(item.summary()["write_calls"], 0)
        self.assertEqual(item.summary()["close_calls"], 0)
        self.assertTrue(item.close())

    def test_reentrant_close_during_write_never_clobbers_descriptors_or_accepts_record(self):
        item = self.make()
        write = os.write

        def callback(fd, raw):
            self.assertFalse(item.close())
            self.assertEqual(item.summary()["close_calls"], 0)
            return write(fd, raw)

        with patch.object(journal.os, "write", side_effect=callback), patch.object(
            journal.os, "fsync",
        ) as fsync:
            self.rejected(item.append, RECORD)
        fsync.assert_not_called()
        self.assertEqual(item.summary()["records_synced"], 0)
        self.assertTrue(item.close())

    def test_reentrant_append_is_sticky_and_does_not_reset_outer_busy_guard(self):
        item = self.make()
        write = os.write

        def callback(fd, raw):
            self.rejected(item.append, b"nested\n")
            self.assertTrue(item._busy)
            return write(fd, raw)

        with patch.object(journal.os, "write", side_effect=callback) as called:
            self.rejected(item.append, RECORD)
        called.assert_called_once()
        self.assertEqual(item.summary()["records_reserved"], 1)
        self.assertEqual(item.summary()["records_synced"], 0)
        self.assertFalse(item._busy)

    def test_reentrant_close_calls_do_not_retry_or_skip_other_owned_descriptor(self):
        item = self.make()
        close = os.close

        def callback(fd):
            self.assertFalse(item.close())
            close(fd)

        with patch.object(journal.os, "close", side_effect=callback) as called:
            self.assertTrue(item.close())
        self.assertEqual(called.call_count, 2)
        self.assertEqual(item.summary()["close_calls"], 2)
        self.assertTrue(item.summary()["failed"])

    def test_configuration_types_and_attempt_id_profile_rejected_without_claim(self):
        for identifier in (None, True, 1, [], {}, "", "a" * 63, "a" * 65, "A" * 64,
                           "../" + "a" * 61, "a" * 63 + "\n"):
            with self.subTest(type_name=type(identifier).__name__):
                self.rejected(journal.Journal, self.root_fd, identifier)
        for descriptor in (None, True, -1, 1.0, "3", []):
            with self.subTest(type_name=type(descriptor).__name__):
                self.rejected(journal.Journal, descriptor, ATTEMPT)
        self.assertFalse(self.path.exists())

    def test_summary_and_errors_have_only_allowlisted_scalars(self):
        item = self.make()
        item.append(RECORD)
        value = item.summary()
        self.assertEqual(value.keys(), SUMMARY_KEYS)
        self.assertTrue(all(type(number) in (bool, int) for number in value.values()))
        for forbidden in (ATTEMPT, str(self.directory), "synthetic", "exception", "SECRET"):
            self.assertNotIn(forbidden, repr(value))
        for supplied in ("SECRET error", None, [], {}, True):
            self.assertEqual(str(journal.JournalError(supplied)), "journal_rejected")

    def test_abrupt_process_exit_leaves_consumed_marker_without_close_or_resume(self):
        code = (
            "import importlib.util, os, sys\n"
            f"spec=importlib.util.spec_from_file_location('fixed_journal', {str(MODULE)!r})\n"
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
            "root_fd=os.open(sys.argv[1], os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)\n"
            f"item=module.Journal(root_fd, {ATTEMPT!r})\n"
            f"item.append({RECORD!r})\n"
            "os._exit(7)\n"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-B", "-c", code, str(self.directory)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=5, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        self.assertEqual(result.returncode, 7)
        self.assertEqual(self.path.read_bytes(), RECORD)
        self.rejected(journal.Journal, self.root_fd, ATTEMPT)
        self.assertEqual(self.path.read_bytes(), RECORD)


if __name__ == "__main__":
    unittest.main()
