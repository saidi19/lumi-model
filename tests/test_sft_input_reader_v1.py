"""Tiny stdlib-only private-FD fixtures; no ML, real inputs or external I/O."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_synthetic_sft_input_reader_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
reader_module = importlib.import_module(PACKAGE + ".posttraining_sft_input_reader_v1")
metered = importlib.import_module(PACKAGE + ".posttraining_sft_metered_io_v1")
METADATA = "encoded_metadata_read_bytes"
BINARY = "encoded_binary_read_bytes"
INDEX = "encoded_index_read_bytes"


class InputReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lumi-SYNTHETIC-input-reader-v1-")
        self.root = Path(self.temporary.name)
        self.root.chmod(0o700)
        self.handles, self.reservations, self.interrupts = [], [], []
        self.descriptors = set()
        self.counter = 0

    def tearDown(self):
        for handle in reversed(self.handles):
            try:
                handle.close()
            except metered.MeteredIOError:
                pass
        for descriptor in self.descriptors:
            os.close(descriptor)
        self.temporary.cleanup()

    def file(self, payload=b"abcdef", *, flags=os.O_RDONLY, mode=0o600):
        self.assertLessEqual(len(payload), 512)
        self.counter += 1
        path = self.root / f"synthetic-{self.counter}.bin"
        path.write_bytes(payload)
        path.chmod(mode)
        descriptor = os.open(path, flags)
        self.descriptors.add(descriptor)
        return descriptor, path

    def reserve(self, rank, **amounts):
        self.reservations.append((rank, amounts))

    def interrupt(self):
        self.interrupts.append(True)

    def reader(self, fd=None, *, resource=BINARY, rank=0, reserve=None, interrupt=None):
        if fd is None:
            fd, _ = self.file()
        result = reader_module.InputReadHandle(
            fd, rank=rank, resource=resource,
            reserve=self.reserve if reserve is None else reserve,
            interrupt=self.interrupt if interrupt is None else interrupt,
        )
        self.descriptors.remove(fd)
        self.handles.append(result)
        return result

    def error(self, code, operation, *args, **kwargs):
        with self.assertRaises(metered.MeteredIOError) as caught:
            operation(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.args, (code,))
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertNotIn("SYNTHETIC-SECRET", repr(caught.exception))
        self.assertNotIn(str(self.root), repr(caught.exception))

    def test_metadata_chunks_prepay_each_pread_at_exact_offsets(self):
        fd, _ = self.file(b"abcdefghij")
        events = []
        original = os.pread

        def reserve(rank, **amounts):
            events.append(("reserve", rank, amounts))

        def pread(descriptor, count, offset):
            self.assertEqual(events[-1], ("reserve", 11, {METADATA: count}))
            events.append(("pread", count, offset))
            return original(descriptor, count, offset)

        handle = self.reader(fd, resource=METADATA, rank=11, reserve=reserve)
        with patch.object(reader_module, "MAX_OPERATION_BYTES", 4), \
                patch.object(reader_module.os, "pread", side_effect=pread) as io:
            self.assertEqual(handle.read(16 * 1024 * 1024 + 1), b"abcdefghij")
        self.assertEqual(io.call_count, 3)
        self.assertEqual(events, [
            ("reserve", 11, {METADATA: 4}), ("pread", 4, 0),
            ("reserve", 11, {METADATA: 4}), ("pread", 4, 4),
            ("reserve", 11, {METADATA: 2}), ("pread", 2, 8),
        ])
        self.assertEqual(handle.tell(), 10)
        self.assertEqual(handle.summary(), {
            "bytes_read": 10, "index_rows_reserved": 0, "closed": False, "failed": False,
        })

    def test_binary_reads_charge_only_requested_available_prefix(self):
        fd, _ = self.file()
        handle = self.reader(fd, rank=1)
        original = os.pread

        def pread(descriptor, count, offset):
            self.assertEqual(self.reservations[-1], (1, {BINARY: count}))
            return original(descriptor, count, offset)

        with patch.object(reader_module.os, "pread", side_effect=pread) as io:
            self.assertEqual(handle.read(2), b"ab")
            self.assertEqual(handle.read(4095 * 4), b"cdef")
        self.assertEqual(io.call_args_list[0].args, (fd, 2, 0))
        self.assertEqual(io.call_args_list[1].args, (fd, 4, 2))
        self.assertEqual(self.reservations, [(1, {BINARY: 2}), (1, {BINARY: 4})])
        self.assertEqual(os.lseek(fd, 0, os.SEEK_CUR), 0)

    def test_zero_reads_and_eof_have_no_byte_reservation_or_pread(self):
        for resource in (METADATA, BINARY):
            with self.subTest(resource=resource):
                handle = self.reader(resource=resource)
                with patch.object(reader_module.os, "pread", wraps=os.pread) as io:
                    self.assertEqual(handle.read(0), b"")
                    io.assert_not_called()
                    self.assertEqual(handle.read(6), b"abcdef")
                    self.assertEqual(handle.read(0), b"")
                    self.assertEqual(handle.read(1), b"")
                self.assertEqual(io.call_count, 1)
        self.assertEqual(self.reservations, [(0, {METADATA: 6}), (0, {BINARY: 6})])

    def test_index_rows_and_every_byte_are_prepaid_without_next_record_overread(self):
        fd, _ = self.file(b"ab\nSYNTHETIC-SECRET\n")
        events = []
        original = os.pread

        def reserve(rank, **amounts):
            events.append(("reserve", rank, amounts))

        def pread(descriptor, count, offset):
            self.assertEqual(events[-1], ("reserve", 2, {INDEX: 1}))
            self.assertEqual(count, 1)
            self.assertLess(offset, 3)
            events.append(("pread", count, offset))
            return original(descriptor, count, offset)

        handle = self.reader(fd, resource=INDEX, rank=2, reserve=reserve)
        with patch.object(reader_module.os, "pread", side_effect=pread) as io:
            self.assertEqual(handle.readline(176), b"ab\n")
        self.assertEqual(io.call_count, 3)
        self.assertEqual(events, [
            ("reserve", 2, {"encoded_index_rows": 1}),
            ("reserve", 2, {INDEX: 1}), ("pread", 1, 0),
            ("reserve", 2, {INDEX: 1}), ("pread", 1, 1),
            ("reserve", 2, {INDEX: 1}), ("pread", 1, 2),
        ])
        self.assertEqual(handle.tell(), 3)
        self.assertEqual(handle.summary()["index_rows_reserved"], 1)

    def test_index_newline_at_first_and_maximum_byte_never_prefetches(self):
        maximum_line = b"a" * 175 + b"\n"
        for payload, expected in ((b"\nsecret", b"\n"),
                                  (maximum_line + b"secret", maximum_line)):
            with self.subTest(length=len(expected)):
                fd, _ = self.file(payload)
                handle = self.reader(fd, resource=INDEX)
                with patch.object(reader_module.os, "pread", wraps=os.pread) as io:
                    self.assertEqual(handle.readline(176), expected)
                self.assertEqual(io.call_count, len(expected))
                self.assertEqual(io.call_args.args, (fd, 1, len(expected) - 1))
                self.assertEqual(handle.tell(), len(expected))

    def test_index_limit_and_eof_charge_one_row_per_valid_call(self):
        fd, _ = self.file(b"abcd")
        handle = self.reader(fd, resource=INDEX)
        with patch.object(reader_module.os, "pread", wraps=os.pread) as io:
            self.assertEqual(handle.readline(2), b"ab")
            self.assertEqual(handle.readline(176), b"cd")
            self.assertEqual(handle.readline(176), b"")
        self.assertEqual(io.call_count, 4)
        self.assertEqual([amount for _, amount in self.reservations], [
            {"encoded_index_rows": 1}, {INDEX: 1}, {INDEX: 1},
            {"encoded_index_rows": 1}, {INDEX: 1}, {INDEX: 1}, {"encoded_index_rows": 1},
        ])
        self.assertEqual(handle.summary()["index_rows_reserved"], 3)

    def test_denied_byte_reservations_never_read_or_retry(self):
        def refuse(rank, **amounts):
            self.reserve(rank, **amounts)
            raise RuntimeError("SYNTHETIC-SECRET")

        for resource in (METADATA, BINARY):
            with self.subTest(resource=resource):
                handle = self.reader(resource=resource, reserve=refuse)
                with patch.object(reader_module.os, "pread") as io:
                    self.error("reserve_failed", handle.read, 3)
                    self.error("reserve_failed", handle.read, 3)
                    self.error("reserve_failed", handle.tell)
                io.assert_not_called()
                self.assertEqual(handle.summary()["bytes_read"], 0)
        self.assertEqual(self.reservations, [(0, {METADATA: 3}), (0, {BINARY: 3})])
        self.assertEqual(len(self.interrupts), 2)

    def test_denied_row_reservation_prevents_byte_charges_and_content(self):
        def refuse(rank, **amounts):
            self.reserve(rank, **amounts)
            raise RuntimeError("SYNTHETIC-SECRET")

        handle = self.reader(resource=INDEX, reserve=refuse)
        with patch.object(reader_module.os, "pread") as io:
            self.error("reserve_failed", handle.readline, 176)
            self.error("reserve_failed", handle.readline, 176)
        io.assert_not_called()
        self.assertEqual(self.reservations, [(0, {"encoded_index_rows": 1})])
        self.assertEqual(handle.summary()["index_rows_reserved"], 0)
        self.assertEqual(self.interrupts, [True])

    def test_row_and_byte_reservations_require_exact_none(self):
        for resource in (METADATA, BINARY, INDEX):
            for result in (True, False, 0, b"", "approved", []):
                with self.subTest(resource=resource, result=result):
                    handle = self.reader(
                        resource=resource, reserve=lambda *a, result=result, **kw: result,
                    )
                    operation = handle.readline if resource == INDEX else handle.read
                    with patch.object(reader_module.os, "pread") as io:
                        self.error("reserve_failed", operation, 1)
                    io.assert_not_called()
                    self.assertEqual(handle.summary()["bytes_read"], 0)
                    self.assertEqual(handle.summary()["index_rows_reserved"], 0)

    def test_index_byte_denial_keeps_prepaid_row_without_read(self):
        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            if INDEX in amounts:
                raise RuntimeError("SYNTHETIC-SECRET")

        handle = self.reader(resource=INDEX, reserve=reserve)
        with patch.object(reader_module.os, "pread") as io:
            self.error("reserve_failed", handle.readline, 176)
            self.error("reserve_failed", handle.readline, 176)
        io.assert_not_called()
        self.assertEqual(self.reservations, [(0, {"encoded_index_rows": 1}), (0, {INDEX: 1})])
        self.assertEqual(handle.summary()["index_rows_reserved"], 1)
        self.assertEqual(handle.summary()["bytes_read"], 0)

    def test_later_chunk_denial_keeps_prior_read_and_all_reservations(self):
        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            if len(self.reservations) == 2:
                raise RuntimeError("SYNTHETIC-SECRET")

        handle = self.reader(resource=METADATA, reserve=reserve)
        with patch.object(reader_module, "MAX_OPERATION_BYTES", 2), \
                patch.object(reader_module.os, "pread", wraps=os.pread) as io:
            self.error("reserve_failed", handle.read, 6)
            self.error("reserve_failed", handle.read, 6)
        self.assertEqual(io.call_count, 1)
        self.assertEqual(self.reservations, [(0, {METADATA: 2}), (0, {METADATA: 2})])
        self.assertEqual(handle.summary()["bytes_read"], 2)
        self.assertEqual(self.interrupts, [True])

    def test_short_and_wrong_type_reads_never_retry_or_refund(self):
        for returned in (b"a", b"", b"abcd", bytearray(b"abc"), None):
            with self.subTest(returned=returned):
                handle = self.reader()
                start = len(self.reservations)
                with patch.object(reader_module.os, "pread", return_value=returned) as io:
                    self.error("short_read", handle.read, 3)
                    self.error("short_read", handle.read, 3)
                self.assertEqual(io.call_count, 1)
                self.assertEqual(self.reservations[start:], [(0, {BINARY: 3})])
                expected_bytes = len(returned) if type(returned) is bytes else 0
                self.assertEqual(handle.summary()["bytes_read"], expected_bytes)

    def test_index_short_read_preserves_row_and_byte_reservations(self):
        handle = self.reader(resource=INDEX)
        with patch.object(reader_module.os, "pread", side_effect=[b"a", b""]) as io:
            self.error("short_read", handle.readline, 176)
            self.error("short_read", handle.readline, 176)
        self.assertEqual(io.call_count, 2)
        self.assertEqual(self.reservations, [
            (0, {"encoded_index_rows": 1}), (0, {INDEX: 1}), (0, {INDEX: 1}),
        ])
        self.assertEqual(handle.summary()["bytes_read"], 1)
        self.assertEqual(handle.summary()["index_rows_reserved"], 1)

    def test_read_syscall_error_is_redacted_terminal_and_not_refunded(self):
        handle = self.reader()
        with patch.object(reader_module.os, "pread", side_effect=OSError("SYNTHETIC-SECRET")) as io:
            self.error("read_failed", handle.read, 3)
            self.error("read_failed", handle.read, 3)
        self.assertEqual(io.call_count, 1)
        self.assertEqual(self.reservations, [(0, {BINARY: 3})])
        self.assertEqual(handle.summary()["bytes_read"], 0)
        self.assertEqual(self.interrupts, [True])

    def test_interrupt_exception_does_not_mask_first_error_or_repeat(self):
        def interrupt():
            self.interrupt()
            raise KeyboardInterrupt("SYNTHETIC-SECRET")

        handle = self.reader(interrupt=interrupt)
        self.error("invalid_arguments", handle.read, -1)
        self.error("invalid_arguments", handle.read, 1)
        self.error("invalid_arguments", handle.tell)
        self.assertEqual(self.interrupts, [True])
        self.assertEqual(self.reservations, [])

    def test_changed_file_before_operation_prevents_reservation_and_read(self):
        fd, path = self.file()
        handle = self.reader(fd)
        path.write_bytes(b"changed-size")
        with patch.object(reader_module.os, "pread") as io:
            self.error("descriptor_changed", handle.read, 1)
        io.assert_not_called()
        self.assertEqual(self.reservations, [])
        self.assertEqual(self.interrupts, [True])

    def test_descriptor_rechecked_after_byte_reservation(self):
        fd, path = self.file()

        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            path.write_bytes(b"changed-size")

        handle = self.reader(fd, reserve=reserve)
        with patch.object(reader_module.os, "pread") as io:
            self.error("descriptor_changed", handle.read, 3)
        io.assert_not_called()
        self.assertEqual(self.reservations, [(0, {BINARY: 3})])
        self.assertEqual(handle.summary()["bytes_read"], 0)

    def test_descriptor_rechecked_after_row_reservation(self):
        fd, path = self.file()

        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            path.write_bytes(b"changed-size")

        handle = self.reader(fd, resource=INDEX, reserve=reserve)
        with patch.object(reader_module.os, "pread") as io:
            self.error("descriptor_changed", handle.readline, 176)
        io.assert_not_called()
        self.assertEqual(self.reservations, [(0, {"encoded_index_rows": 1})])
        self.assertEqual(handle.summary()["index_rows_reserved"], 1)
        self.assertEqual(handle.summary()["bytes_read"], 0)

    def test_mutation_after_pread_records_bytes_but_never_returns_payload(self):
        fd, path = self.file()
        handle = self.reader(fd)
        original = os.pread

        def pread(descriptor, count, offset):
            result = original(descriptor, count, offset)
            path.write_bytes(b"changed-size")
            return result

        with patch.object(reader_module.os, "pread", side_effect=pread) as io:
            self.error("descriptor_changed", handle.read, 3)
            self.error("descriptor_changed", handle.read, 3)
        self.assertEqual(io.call_count, 1)
        self.assertEqual(self.reservations, [(0, {BINARY: 3})])
        self.assertEqual(handle.summary()["bytes_read"], 3)

    def test_changed_descriptor_flags_prevent_content_and_reservations(self):
        fd, _ = self.file()
        handle = self.reader(fd)
        os.set_inheritable(fd, True)
        with patch.object(reader_module.os, "pread") as io:
            self.error("descriptor_changed", handle.read, 1)
        io.assert_not_called()
        self.assertEqual(self.reservations, [])

    def test_reused_descriptor_is_neither_read_nor_closed(self):
        fd, _ = self.file()
        handle = self.reader(fd)
        replacement, path = self.file(b"replacement")
        os.dup2(replacement, fd, inheritable=False)
        with patch.object(reader_module.os, "pread") as io, \
                patch.object(reader_module.os, "close") as close:
            self.error("descriptor_changed", handle.read, 1)
            self.error("descriptor_changed", handle.close)
            handle.close()
        io.assert_not_called()
        close.assert_not_called()
        self.descriptors.add(fd)
        self.assertEqual(os.fstat(fd).st_ino, path.stat().st_ino)
        self.assertTrue(handle.closed)
        self.assertEqual(self.reservations, [])
        self.assertEqual(self.interrupts, [True])

    def test_fd_reuse_in_reservation_prevents_content_and_preserves_replacement(self):
        fd, _ = self.file()
        replacement, path = self.file(b"replacement")

        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            os.dup2(replacement, fd, inheritable=False)

        handle = self.reader(fd, reserve=reserve)
        with patch.object(reader_module.os, "pread") as io, \
                patch.object(reader_module.os, "close") as close:
            self.error("descriptor_changed", handle.read, 2)
            self.error("descriptor_changed", handle.close)
        io.assert_not_called()
        close.assert_not_called()
        self.descriptors.add(fd)
        self.assertEqual(os.fstat(fd).st_ino, path.stat().st_ino)
        self.assertEqual(self.reservations, [(0, {BINARY: 2})])

    def test_process_and_thread_owner_mismatch_precede_reservation(self):
        for provider in ("getpid", "get_ident"):
            with self.subTest(provider=provider):
                handle = self.reader()
                target = metered.os if provider == "getpid" else metered.threading
                with patch.object(target, provider, return_value=-1), \
                        patch.object(reader_module.os, "pread") as io:
                    self.error("owner_mismatch", handle.read, 1)
                io.assert_not_called()
                self.error("owner_mismatch", handle.read, 1)
        self.assertEqual(self.reservations, [])
        self.assertEqual(len(self.interrupts), 2)

    def test_caught_reservation_reentry_still_fails_outer_read(self):
        for resource, category in ((BINARY, BINARY), (INDEX, "encoded_index_rows"), (INDEX, INDEX)):
            for nested in ("read", "tell", "close"):
                with self.subTest(resource=resource, category=category, nested=nested):
                    holder = {}

                    def reserve(rank, *, holder=holder, nested=nested, category=category,
                                **amounts):
                        self.reserve(rank, **amounts)
                        if category not in amounts:
                            return
                        handle = holder["handle"]
                        args = (1,) if nested == "read" else ()
                        self.error("reentry", getattr(handle, nested), *args)

                    handle = self.reader(resource=resource, reserve=reserve)
                    holder["handle"] = handle
                    operation = handle.readline if resource == INDEX else handle.read
                    with patch.object(reader_module.os, "pread") as io:
                        self.error("reentry", operation, 1)
                    io.assert_not_called()
                    self.assertFalse(handle.closed)
                    self.assertEqual(handle.summary()["bytes_read"], 0)
                    expected_rows = 1 if category == INDEX else 0
                    self.assertEqual(handle.summary()["index_rows_reserved"], expected_rows)
        self.assertEqual(len(self.reservations), 12)
        self.assertEqual(len(self.interrupts), 9)

    def test_constructor_invalid_ranks_roles_and_callbacks_keep_caller_fd(self):
        cases = [{"rank": value} for value in (-1, 12, True, 1.0, "0", None)]
        cases += [{"resource": value} for value in (
            "checkpoint_read_bytes", "private_write_bytes", "metadata_read_bytes", None, [],
        )]
        cases += [{"reserve": None}, {"interrupt": None}]
        for changes in cases:
            with self.subTest(changes=changes):
                fd, _ = self.file()
                options = {"rank": 0, "resource": BINARY, "reserve": self.reserve,
                           "interrupt": self.interrupt}
                options.update(changes)
                with patch.object(metered, "_snapshot") as snapshot:
                    self.error("invalid_arguments", reader_module.InputReadHandle, fd, **options)
                snapshot.assert_not_called()
                self.assertGreater(os.fstat(fd).st_ino, 0)
        self.assertEqual(self.reservations, [])
        self.assertEqual(self.interrupts, [])

    def test_invalid_fd_values_reject_before_metadata_or_ownership_transfer(self):
        for fd in (0, 1, 2, -1, True, 3.0, None):
            with self.subTest(fd=fd), patch.object(metered, "_snapshot") as snapshot:
                self.error("invalid_arguments", reader_module.InputReadHandle, fd, rank=0,
                           resource=BINARY, reserve=self.reserve, interrupt=self.interrupt)
                snapshot.assert_not_called()
        self.assertEqual(self.interrupts, [])

    def test_invalid_private_file_descriptors_remain_caller_owned(self):
        for flags, mode, inheritable, linked in (
            (os.O_RDWR, 0o600, False, False), (os.O_WRONLY, 0o600, False, False),
            (os.O_RDONLY, 0o644, False, False), (os.O_RDONLY, 0o600, True, False),
            (os.O_RDONLY, 0o600, False, True),
        ):
            with self.subTest(flags=flags, mode=mode, inheritable=inheritable, linked=linked):
                fd, path = self.file(flags=flags, mode=mode)
                os.set_inheritable(fd, inheritable)
                if linked:
                    os.link(path, self.root / "synthetic-hardlink")
                self.error("invalid_descriptor", reader_module.InputReadHandle, fd, rank=0,
                           resource=BINARY, reserve=self.reserve, interrupt=self.interrupt)
                self.assertGreater(os.fstat(fd).st_ino, 0)
        self.assertEqual(self.interrupts, [])

    def test_read_size_rejections_are_terminal_and_precede_content(self):
        for resource, maximum in ((METADATA, 16 * 1024 * 1024 + 1), (BINARY, 4095 * 4)):
            for size in (-1, True, False, 1.0, "1", None, maximum + 1):
                with self.subTest(resource=resource, size=size):
                    handle = self.reader(resource=resource)
                    with patch.object(reader_module.os, "pread") as io:
                        self.error("invalid_arguments", handle.read, size)
                        self.error("invalid_arguments", handle.read, 1)
                    io.assert_not_called()
        self.assertEqual(self.reservations, [])

    def test_readline_size_rejections_precede_row_and_byte_reservations(self):
        for size in (-1, 0, True, False, 1.0, "1", None, 177):
            with self.subTest(size=size):
                handle = self.reader(resource=INDEX)
                with patch.object(reader_module.os, "pread") as io:
                    self.error("invalid_arguments", handle.readline, size)
                io.assert_not_called()
                self.assertEqual(handle.summary()["index_rows_reserved"], 0)
        self.assertEqual(self.reservations, [])

    def test_resource_roles_cannot_use_other_content_api(self):
        for resource in (METADATA, BINARY, INDEX):
            with self.subTest(resource=resource):
                handle = self.reader(resource=resource)
                operation = handle.read if resource == INDEX else handle.readline
                with patch.object(reader_module.os, "pread") as io:
                    self.error("invalid_arguments", operation, 1)
                io.assert_not_called()
        self.assertEqual(self.reservations, [])

    def test_close_is_idempotent_and_public_surface_exposes_no_content_escape(self):
        fd, _ = self.file()
        handle = self.reader(fd)
        for name in ("fileno", "seek", "readinto", "raw", "buffer", "write"):
            self.assertFalse(hasattr(handle, name), name)
        self.assertEqual(repr(handle), "<InputReadHandle private>")
        self.assertEqual(set(handle.summary()),
                         {"bytes_read", "index_rows_reserved", "closed", "failed"})
        with patch.object(reader_module.os, "close", wraps=os.close) as close:
            handle.close()
            handle.close()
        close.assert_called_once_with(fd)
        self.assertTrue(handle.closed)
        self.assertFalse(handle.summary()["failed"])
        self.assertEqual(self.interrupts, [])

    def test_close_failure_is_terminal_and_never_retries_fd_number(self):
        fd, _ = self.file()
        handle = self.reader(fd)
        failure = OSError("SYNTHETIC-SECRET")
        with patch.object(reader_module.os, "close", side_effect=failure) as close:
            self.error("close_failed", handle.close)
            handle.close()
        close.assert_called_once_with(fd)
        self.descriptors.add(fd)  # The injected close did not close this synthetic descriptor.
        self.assertTrue(handle.closed)
        self.error("close_failed", handle.read, 1)
        self.assertEqual(self.interrupts, [True])

    def test_retired_closed_handle_cannot_interrupt_new_epoch(self):
        current_epoch = ["old"]
        interrupted_epochs = []

        def interrupt():
            interrupted_epochs.append(current_epoch[0])

        old = self.reader(interrupt=interrupt)
        old.close()
        current_epoch[0] = "new"
        new = self.reader(interrupt=interrupt)
        with patch.object(reader_module.os, "pread") as io:
            self.error("closed", old.read, 1)
            self.error("closed", old.tell)
            with patch.object(metered.os, "getpid", return_value=-1):
                self.error("closed", old.read, 1)
            old.close()
        io.assert_not_called()
        self.assertEqual(interrupted_epochs, [])
        self.assertEqual(self.reservations, [])
        self.assertEqual(new.read(2), b"ab")
        self.assertFalse(new.summary()["failed"])
        self.assertEqual(interrupted_epochs, [])

    def test_first_failure_survives_close_and_interrupts_only_once(self):
        handle = self.reader()
        self.error("invalid_arguments", handle.read, -1)
        handle.close()
        self.error("invalid_arguments", handle.read, 1)
        self.error("invalid_arguments", handle.readline, 1)
        handle.close()
        self.assertTrue(handle.closed)
        self.assertEqual(self.interrupts, [True])
        self.assertEqual(self.reservations, [])

    def test_reinitialization_fails_without_taking_replacement_fd(self):
        fd, _ = self.file()
        handle = self.reader(fd)
        replacement, _ = self.file(b"replacement")
        self.error("invalid_arguments", handle.__init__, replacement, rank=0, resource=BINARY,
                   reserve=self.reserve, interrupt=self.interrupt)
        self.assertGreater(os.fstat(replacement).st_ino, 0)
        handle.close()
        with self.assertRaises(OSError):
            os.fstat(fd)
        self.assertEqual(self.interrupts, [True])


if __name__ == "__main__":
    unittest.main()
