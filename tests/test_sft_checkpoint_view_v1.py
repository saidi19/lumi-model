"""Tiny synthetic checkpoint bytes only; no ML package entrypoint import."""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import tempfile
import types
import unittest
from array import array
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_checkpoint_view_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
views = importlib.import_module(f"{PACKAGE}.posttraining_sft_checkpoint_view_v1")
readers = importlib.import_module(f"{PACKAGE}.posttraining_sft_checkpoint_reader_v1")


class CheckpointViewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lumi-SYNTHETIC-checkpoint-view-v1-")
        self.root = Path(self.temporary.name)
        self.root.chmod(0o700)
        (self.root / "model").mkdir(mode=0o700)
        self.payloads = {"config.json": b"{}", "model/.metadata": b"ab\ncdef\ngh",
                         "model/__0_0.distcp": b"0123456789", "empty": b""}
        for name, data in self.payloads.items():
            self.write(name, data)
        manifest = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n"
                           for name, data in self.payloads.items()).encode("ascii")
        self.write("manifest.sha256", manifest)
        self.pin = hashlib.sha256(manifest).hexdigest()
        self.sessions, self.charges = [], []

    def tearDown(self):
        for session in self.sessions:
            try:
                session.close()
            except readers.CheckpointReadError:
                pass
        self.temporary.cleanup()

    def write(self, name, data):
        path = self.root / name
        path.write_bytes(data)
        path.chmod(0o600)

    def reserve(self, rank, **amounts):
        self.charges.append((rank, amounts["checkpoint_read_bytes"]))

    def session(self, *, verify=True):
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        session = readers.CheckpointReadSession(
            descriptor, rank=0, reserve=self.reserve, expected_manifest_sha256=self.pin,
            maximum_file_bytes=1024, maximum_read_bytes=8192)
        self.sessions.append(session)
        if verify:
            session.verify()
        return session

    def view(self, relative="model/__0_0.distcp", **options):
        session = self.session()
        options.setdefault("maximum_read_call_bytes", 64)
        view = views.CheckpointReadView(session, relative, **options)
        self.charges.clear()
        return view, session

    def assert_error(self, code, function, *args, **kwargs):
        with self.assertRaises(views.CheckpointViewError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.args, (code,))
        self.assertEqual(caught.exception.code, code)
        self.assertNotIn(str(self.root), repr(caught.exception))

    def test_read_seek_and_reread_are_all_metered(self):
        view, session = self.view()
        self.assertEqual(view.read(3), b"012")
        self.assertEqual(view.seek(1), 1)
        self.assertEqual(view.read(3), b"123")
        self.assertEqual(view.tell(), 4)
        self.assertEqual(self.charges, [(0, 3), (0, 3)])
        self.assertFalse(session._failed)

    def test_default_read_is_remaining_extent(self):
        view, _ = self.view(offset=2)
        self.assertEqual(view.read(), b"23456789")
        self.assertEqual(view.read(), b"")
        self.assertEqual(self.charges, [(0, 8)])

    def test_explicit_large_read_clamps_before_call_cap(self):
        view, _ = self.view(offset=8, maximum_read_call_bytes=2)
        self.assertEqual(view.read(2**63), b"89")
        self.assertEqual(view.read(2**63), b"")
        self.assertEqual(self.charges, [(0, 2)])

    def test_chunk_reads_have_explicit_bounded_sizes(self):
        view, session = self.view()
        with (patch.object(views, "MAX_OPERATION_BYTES", 3),
              patch.object(session, "read_file", wraps=session.read_file) as reading):
            self.assertEqual(view.read(), b"0123456789")
        self.assertEqual([call.args[2] for call in reading.call_args_list], [3, 3, 3, 1])
        self.assertEqual(self.charges, [(0, 3), (0, 3), (0, 3), (0, 1)])

    def test_read_cap_failure_occurs_before_content_access(self):
        view, session = self.view(maximum_read_call_bytes=4)
        with patch.object(session, "read_file", wraps=session.read_file) as reading:
            self.assert_error("read_cap_exceeded", view.read)
        reading.assert_not_called()
        self.assertEqual(self.charges, [])
        self.assertTrue(session._failed)
        self.assert_error("terminal_failure", view.read, 1)

    def test_invalid_read_sizes_poison_shared_session(self):
        for size in (True, False, None, -2, 1.0, "1"):
            with self.subTest(size=size):
                view, session = self.view()
                self.assert_error("invalid_arguments", view.read, size)
                self.assertTrue(session._failed)
                self.assertEqual(self.charges, [])

    def test_readinto_bytearray_preserves_suffix_at_eof(self):
        view, _ = self.view(offset=7)
        buffer = bytearray(b"XXXXXX")
        self.assertEqual(view.readinto(buffer), 3)
        self.assertEqual(buffer, b"789XXX")
        self.assertEqual(view.readinto(buffer), 0)
        self.assertEqual(self.charges, [(0, 3)])

    def test_readinto_writable_memoryview_slice(self):
        view, _ = self.view()
        buffer = bytearray(b"XXXXXX")
        with memoryview(buffer)[1:5] as target:
            self.assertEqual(view.readinto(target), 4)
        self.assertEqual(buffer, b"X0123X")

    def test_readinto_signed_byte_format(self):
        view, _ = self.view()
        buffer = bytearray(3)
        with memoryview(buffer).cast("b") as target:
            self.assertEqual(view.readinto(target), 3)
        self.assertEqual(buffer, b"012")

    def test_readinto_chunks_without_full_buffer_clone(self):
        view, session = self.view()
        buffer = bytearray(10)
        with (patch.object(views, "MAX_OPERATION_BYTES", 4),
              patch.object(session, "read_file", wraps=session.read_file) as reading):
            self.assertEqual(view.readinto(buffer), 10)
        self.assertEqual(buffer, b"0123456789")
        self.assertEqual([call.args[2] for call in reading.call_args_list], [4, 4, 2])

    def test_readinto_cap_uses_actual_remaining_not_buffer_size(self):
        view, _ = self.view(offset=8, maximum_read_call_bytes=2)
        buffer = bytearray(b"XXXXXX")
        self.assertEqual(view.readinto(buffer), 2)
        self.assertEqual(buffer, b"89XXXX")
        self.assertEqual(view.readinto(buffer), 0)

    def test_readinto_cap_rejects_before_buffer_mutation(self):
        view, session = self.view(maximum_read_call_bytes=2)
        buffer = bytearray(b"XXXX")
        self.assert_error("read_cap_exceeded", view.readinto, buffer)
        self.assertEqual(buffer, b"XXXX")
        self.assertTrue(session._failed)
        self.assertEqual(self.charges, [])

    def test_readinto_later_chunk_failure_keeps_partial_buffer_and_charges(self):
        view, session = self.view()
        buffer = bytearray(b"XXXXXX")

        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            if len(self.charges) == 2:
                raise RuntimeError("SYNTHETIC-BUDGET")

        session._reserve = reserve
        with patch.object(views, "MAX_OPERATION_BYTES", 4):
            self.assert_error("read_failed", view.readinto, buffer)
        self.assertEqual(buffer, b"0123XX")
        self.assertEqual(self.charges, [(0, 4), (0, 2)])
        self.assertTrue(session._failed)
        self.assert_error("terminal_failure", view.readinto, buffer)

    def test_readinto_invalid_buffers_fail_closed(self):
        buffers = (b"abc", [0], array("B", [0]), memoryview(b"abc"),
                   memoryview(bytearray(6))[::2], memoryview(array("I", [0])),
                   memoryview(bytearray(4)).cast("B", shape=[2, 2]))
        try:
            for buffer in buffers:
                with self.subTest(buffer_type=type(buffer).__name__):
                    view, session = self.view()
                    self.assert_error("invalid_arguments", view.readinto, buffer)
                    self.assertTrue(session._failed)
        finally:
            for buffer in buffers:
                if isinstance(buffer, memoryview):
                    buffer.release()

    def test_readinto_released_memoryview_has_fixed_error(self):
        view, session = self.view()
        target = memoryview(bytearray(2))
        target.release()
        self.assert_error("read_failed", view.readinto, target)
        self.assertTrue(session._failed)

    def test_readline_read_ahead_is_charged_then_reread(self):
        view, _ = self.view("model/.metadata")
        self.assertEqual(view.readline(), b"ab\n")
        self.assertEqual(view.tell(), 3)
        self.assertEqual(view.read(4), b"cdef")
        self.assertEqual(self.charges, [(0, 10), (0, 4)])

    def test_readline_chunk_boundary_and_independent_next_line(self):
        view, _ = self.view("model/.metadata")
        with patch.object(views, "MAX_OPERATION_BYTES", 4):
            self.assertEqual(view.readline(), b"ab\n")
            self.assertEqual(view.readline(), b"cdef\n")
            self.assertEqual(view.readline(), b"gh")
            self.assertEqual(view.readline(), b"")
        self.assertEqual(self.charges, [(0, 4), (0, 4), (0, 3), (0, 2)])

    def test_readline_explicit_size_and_zero(self):
        view, _ = self.view("model/.metadata")
        self.assertEqual(view.readline(0), b"")
        self.assertEqual(view.readline(2), b"ab")
        self.assertEqual(view.readline(1), b"\n")
        self.assertEqual(view.tell(), 3)
        self.assertEqual(self.charges, [(0, 2), (0, 1)])

    def test_readline_without_newline_is_bounded_remaining(self):
        view, _ = self.view()
        self.assertEqual(view.readline(), b"0123456789")
        self.assertEqual(view.readline(), b"")

    def test_readline_cap_is_checked_even_if_newline_would_be_early(self):
        view, session = self.view("model/.metadata", maximum_read_call_bytes=4)
        self.assert_error("read_cap_exceeded", view.readline)
        self.assertEqual(self.charges, [])
        self.assertTrue(session._failed)

    def test_seek_end_uses_standard_positive_and_negative_offsets(self):
        view, _ = self.view(offset=2, length=6)
        self.assertEqual(view.seek(-2, os.SEEK_END), 4)
        self.assertEqual(view.read(), b"67")
        self.assertEqual(view.seek(-3, os.SEEK_CUR), 3)
        self.assertEqual(view.read(1), b"5")
        self.assertEqual(view.seek(0, os.SEEK_END), 6)
        self.assertEqual(view.seek(0), 0)

    def test_seek_rejects_escaping_and_invalid_arguments(self):
        for offset, whence in ((-1, os.SEEK_SET), (11, os.SEEK_SET), (1, os.SEEK_END),
                               (-11, os.SEEK_END), (1, 99), (True, 0), (0, True), (1.0, 0)):
            with self.subTest(offset=offset, whence=whence):
                view, session = self.view()
                self.assert_error("invalid_arguments", view.seek, offset, whence)
                self.assertTrue(session._failed)
                self.assertEqual(self.charges, [])

    def test_slice_has_independent_cursor_and_absolute_file_offset(self):
        view, session = self.view(offset=2, length=6)
        view.seek(4)
        child = view.slice(1, 3)
        self.assertEqual(child.read(), b"345")
        self.assertEqual(view.tell(), 4)
        self.assertEqual(view.read(), b"67")
        child.seek(0)
        self.assertEqual(child.read(1), b"3")
        self.assertFalse(session._failed)

    def test_nested_slice_is_bounded_by_parent_not_file(self):
        view, _ = self.view()
        child = view.slice(2, 6).slice(2, 2)
        self.assertEqual(child.read(), b"45")
        self.assertEqual(child.seek(0, os.SEEK_END), 2)

    def test_invalid_slice_bounds_poison_session(self):
        for offset, length in ((-1, 1), (1, -1), (3, 2), (5, 0), (True, 1), (0, True)):
            with self.subTest(offset=offset, length=length):
                view, session = self.view(length=4)
                self.assert_error("invalid_arguments", view.slice, offset, length)
                self.assertTrue(session._failed)

    def test_empty_file_and_end_slice_do_not_read_content(self):
        for name, options in (("empty", {}), ("model/__0_0.distcp", {"offset": 10})):
            with self.subTest(name=name):
                view, _ = self.view(name, **options)
                self.assertEqual(view.read(), b"")
                self.assertEqual(view.readline(), b"")
                self.assertEqual(view.readinto(bytearray(10)), 0)
                self.assertEqual(view.slice(0, 0).read(), b"")
                self.assertEqual(self.charges, [])

    def test_close_does_not_close_session_or_child(self):
        view, session = self.view()
        child = view.slice(2, 2)
        view.close()
        view.close()
        self.assertTrue(view.closed)
        self.assertFalse(session.closed)
        self.assertEqual(child.read(), b"23")
        self.assertEqual(session.read_file("config.json", 0, 2), b"{}")

    def test_close_after_failure_is_allowed_without_descriptor_close(self):
        view, session = self.view()
        self.assert_error("invalid_arguments", view.read, -2)
        with patch.object(views.os, "close", wraps=os.close) as closing:
            view.close()
            view.close()
        closing.assert_not_called()
        self.assertTrue(view.closed)
        self.assertFalse(session.closed)

    def test_use_after_view_close_poison_session(self):
        view, session = self.view()
        view.close()
        self.assert_error("closed", view.tell)
        self.assertTrue(session._failed)

    def test_session_close_invalidates_open_views(self):
        view, session = self.view()
        session.close()
        self.assert_error("closed", view.read, 1)
        view.close()

    def test_session_failure_invalidates_siblings(self):
        view, session = self.view()
        sibling = view.slice(0, 1)
        self.assert_error("invalid_arguments", view.seek, -1)
        self.assert_error("terminal_failure", sibling.read)
        self.assertTrue(session._failed)

    def test_constructor_requires_verified_session(self):
        session = self.session(verify=False)
        self.assert_error("not_verified", views.CheckpointReadView, session, "config.json",
                          maximum_read_call_bytes=4)
        self.assertTrue(session._failed)

    def test_constructor_validates_names_bounds_and_cap(self):
        changes = ({"relative": "manifest.sha256"}, {"relative": "missing"},
                   {"relative": "../config.json"}, {"relative": Path("config.json")},
                   {"offset": -1}, {"offset": 11}, {"offset": True}, {"length": True},
                   {"length": -1}, {"length": 11}, {"maximum_read_call_bytes": True},
                   {"maximum_read_call_bytes": 0}, {"maximum_read_call_bytes": -1},
                   {"maximum_read_call_bytes": 2**63})
        for changed in changes:
            with self.subTest(changed=changed):
                session = self.session()
                options = {"relative": "model/__0_0.distcp", "maximum_read_call_bytes": 4}
                options.update(changed)
                self.assert_error("invalid_arguments", views.CheckpointReadView, session, **options)
                self.assertTrue(session._failed)

    def test_constructor_rejects_unrelated_session_object(self):
        for session in (None, object(), {}, True):
            self.assert_error("invalid_session", views.CheckpointReadView, session, "config.json",
                              maximum_read_call_bytes=4)

    def test_constructor_cannot_reinitialize_existing_view(self):
        view, session = self.view()
        self.assert_error("invalid_arguments", view.__init__, session, "config.json",
                          maximum_read_call_bytes=4)
        self.assertTrue(session._failed)

    def test_constructor_rejects_closed_session(self):
        session = self.session()
        session.close()
        self.assert_error("closed", views.CheckpointReadView, session, "config.json",
                          maximum_read_call_bytes=4)

    def test_observed_payload_mutation_fails_before_content_read(self):
        view, session = self.view()
        self.write("model/__0_0.distcp", b"CHANGED")
        self.assert_error("read_failed", view.read, 1)
        self.assertTrue(session._failed)
        self.assertEqual(self.charges, [])

    def test_root_metadata_change_invalidates_even_cursor_operation(self):
        view, session = self.view()
        self.write("unexpected", b"SYNTHETIC")
        self.assert_error("read_failed", view.tell)
        self.assertTrue(session._failed)

    def test_reservation_callback_mutation_is_nonrefundable(self):
        view, session = self.view()

        def mutate(rank, **amounts):
            self.reserve(rank, **amounts)
            self.write("model/__0_0.distcp", b"CHANGED")

        session._reserve = mutate
        self.assert_error("read_failed", view.read, 2)
        self.assertEqual(self.charges, [(0, 2)])
        self.assertTrue(session._failed)

    def test_swallowed_same_view_reentry_poison_outer_read(self):
        view, session = self.view()

        def reenter(rank, **amounts):
            self.reserve(rank, **amounts)
            self.assert_error("reentry", view.tell)

        session._reserve = reenter
        self.assert_error("read_failed", view.read, 2)
        self.assertTrue(session._failed)
        self.assertEqual(self.charges, [(0, 2)])

    def test_swallowed_sibling_reentry_poison_outer_read(self):
        view, session = self.view()
        sibling = view.slice(0, 2)

        def reenter(rank, **amounts):
            self.reserve(rank, **amounts)
            self.assert_error("read_failed", sibling.tell)

        session._reserve = reenter
        self.assert_error("read_failed", view.read, 2)
        self.assertTrue(session._failed)

    def test_close_reentry_cannot_silently_cancel_operation(self):
        view, session = self.view()

        def reenter(rank, **amounts):
            self.reserve(rank, **amounts)
            self.assert_error("reentry", view.close)

        session._reserve = reenter
        self.assert_error("read_failed", view.read, 2)
        self.assertFalse(view.closed)
        view.close()

    def test_process_owner_mismatch_is_terminal(self):
        view, session = self.view()
        with patch.object(views.os, "getpid", return_value=os.getpid() + 1):
            self.assert_error("owner_mismatch", view.read, 1)
        self.assertTrue(session._failed)
        view.close()

    def test_thread_owner_mismatch_is_terminal(self):
        view, session = self.view()
        with patch.object(views.threading, "get_ident", return_value=-1):
            self.assert_error("owner_mismatch", view.close)
        self.assertTrue(session._failed)
        view.close()

    def test_reservation_error_does_not_leak_private_error_text(self):
        view, session = self.view()

        def fail(rank, **amounts):
            self.reserve(rank, **amounts)
            raise RuntimeError("SYNTHETIC-PRIVATE-CALLBACK")

        session._reserve = fail
        self.assert_error("read_failed", view.read, 1)
        self.assertEqual(self.charges, [(0, 1)])
        self.assertTrue(session._failed)

    def test_reservation_non_none_return_is_terminal_and_not_refunded(self):
        view, session = self.view()

        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            return True

        session._reserve = reserve
        self.assert_error("read_failed", view.read, 1)
        self.assertEqual(self.charges, [(0, 1)])
        self.assertTrue(session._failed)

    def test_unexpected_chunk_type_or_short_read_is_terminal(self):
        for chunk in (b"", b"12", bytearray(b"1")):
            with self.subTest(chunk_type=type(chunk).__name__):
                view, session = self.view()
                with patch.object(session, "read_file", return_value=chunk):
                    self.assert_error("read_failed", view.read, 1)
                self.assertTrue(session._failed)

    def test_capabilities_and_repr_never_expose_native_bypass(self):
        view, _ = self.view()
        self.assertTrue(view.readable())
        self.assertTrue(view.seekable())
        self.assertFalse(view.closed)
        self.assertEqual(repr(view), "<CheckpointReadView private>")
        for attribute in ("fileno", "raw", "buffer", "path", "name", "write", "mmap"):
            self.assertFalse(hasattr(view, attribute), attribute)

    def test_unknown_error_code_is_redacted(self):
        self.assertEqual(views.CheckpointViewError("SYNTHETIC-PRIVATE").args, ("internal_error",))


if __name__ == "__main__":
    unittest.main()
