"""Tiny stdlib file-I/O checks; no ML, real data, training or subprocesses."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/lumi" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metered = load("synthetic_metered_io", "posttraining_sft_metered_io_v1.py")
supervision = load("synthetic_metered_supervisor", "posttraining_sft_supervisor_v1.py")


def controller():
    counts = {
        "sequential_invocations": 3, "global_optimizer_steps": 4,
        "rank_optimizer_steps": 8, "logical_sample_events": 8,
        "maximum_unique_train_records": 8, "rank_local_input_loads": 6,
        "maximum_encoded_index_rows": 48,
        "maximum_selected_binary_bytes": 48 * 32 * 8,
        "maximum_selected_index_bytes": 48 * 176,
        "includes_checkpoint_hashing_or_final_digest": False,
        "is_total_io_or_cost_budget": False,
    }
    caps = {
        "maximum_combined_runtime_seconds": 60, "maximum_cost_microusd": 1000,
        "total_host_hourly_rate_microusd": 60_000, "maximum_host_ram_bytes": 4096,
        "maximum_per_gpu_memory_bytes": 2048, "maximum_private_write_bytes": 32,
        "maximum_checkpoint_read_bytes": 6,
        "maximum_encoded_binary_read_bytes": counts["maximum_selected_binary_bytes"],
        "maximum_encoded_index_read_bytes": counts["maximum_selected_index_bytes"],
        "maximum_log_bytes": 16, "maximum_invocations": 3,
        "maximum_global_optimizer_steps": 4, "maximum_external_network_bytes": 0,
        "maximum_inference_calls": 0,
    }
    result = supervision.Supervisor(caps, counts, world_size=2, clock_ns=lambda: 0)
    result.begin_invocation("reference", 0, 2, False)
    return result


class MeteredIOTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lumi-SYNTHETIC-metered-io-v1-")
        self.root = Path(self.temporary.name)
        self.root.chmod(0o700)
        self.handles = []
        self.descriptors = set()
        self.reservations = []
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

    def file(self, payload=b"abcdef", *, mode=0o600, flags=os.O_RDONLY):
        self.counter += 1
        path = self.root / f"synthetic-{self.counter}.bin"
        path.write_bytes(payload)
        path.chmod(mode)
        descriptor = os.open(path, flags)
        self.descriptors.add(descriptor)
        return descriptor, path

    def reserve(self, rank, **amounts):
        self.reservations.append((rank, amounts))

    def reader(self, fd=None, *, reserve=None, resource="checkpoint_read_bytes", rank=0):
        if fd is None:
            fd, _ = self.file()
        result = metered.MeteredReader(
            fd, rank=rank, resource=resource, reserve=reserve or self.reserve,
        )
        self.descriptors.remove(fd)
        self.handles.append(result)
        return result

    def writer(self, *, reserve=None, resource="private_write_bytes", maximum=16, rank=0):
        fd, path = self.file(b"", flags=os.O_WRONLY)
        result = metered.MeteredWriter(
            fd, rank=rank, resource=resource, reserve=reserve or self.reserve,
            maximum_file_bytes=maximum,
        )
        self.descriptors.remove(fd)
        self.handles.append(result)
        return result, path, fd

    def deny(self, operation):
        with self.assertRaises(metered.MeteredIOError) as caught:
            operation()
        self.assertEqual(len(caught.exception.args), 1)
        self.assertIn(caught.exception.args[0], metered.ERROR_CODES)
        self.assertNotIn("SYNTHETIC-SECRET", str(caught.exception))
        self.assertNotIn(str(self.root), str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_reads_reserve_before_syscall_and_charge_repeated_bytes(self):
        events = []
        original = os.pread

        def reserve(rank, **amounts):
            events.append(("reserve", rank, amounts))

        def read(fd, count, offset):
            self.assertEqual(events[-1], ("reserve", 1, {"checkpoint_read_bytes": count}))
            events.append(("read", count, offset))
            return original(fd, count, offset)

        reader = self.reader(reserve=reserve, rank=1)
        with patch.object(metered.os, "pread", side_effect=read) as io:
            self.assertEqual(reader.read(3), b"abc")
            self.assertEqual(reader.seek(0), 0)
            self.assertEqual(reader.read(3), b"abc")
        self.assertEqual(io.call_count, 2)
        self.assertEqual([item[0] for item in events], ["reserve", "read"] * 2)

    def test_zero_and_eof_never_reserve_or_read(self):
        reader = self.reader()
        self.assertEqual(reader.read(6), b"abcdef")
        self.reservations.clear()
        with patch.object(metered.os, "pread") as io:
            self.assertEqual(reader.read(0), b"")
            self.assertEqual(reader.read(1), b"")
            self.assertEqual(reader.readinto(bytearray()), 0)
        io.assert_not_called()
        self.assertEqual(self.reservations, [])

    def test_readinto_and_seek_end_charge_only_actual_available_bytes(self):
        reader = self.reader()
        self.assertEqual(reader.seek(-2, os.SEEK_END), 4)
        output = bytearray(b"XXXXXXXX")
        self.assertEqual(reader.readinto(output), 2)
        self.assertEqual(output, b"efXXXXXX")
        self.assertEqual(reader.tell(), 6)
        self.assertEqual(self.reservations, [(0, {"checkpoint_read_bytes": 2})])

    def test_reservation_failure_prevents_all_content_io_and_is_terminal(self):
        def refuse(*args, **kwargs):
            raise RuntimeError("SYNTHETIC-SECRET")

        reader = self.reader(reserve=refuse)
        with patch.object(metered.os, "pread") as io:
            self.deny(lambda: reader.read(1))
            self.deny(lambda: reader.read(1))
        io.assert_not_called()

    def test_reservation_requires_exact_none_ack(self):
        for result in (True, False, 0, b"", "approved"):
            with self.subTest(kind=type(result).__name__):
                reader = self.reader(reserve=lambda *a, result=result, **kw: result)
                with patch.object(metered.os, "pread") as io:
                    self.deny(lambda reader=reader: reader.read(1))
                io.assert_not_called()

    def test_short_read_keeps_reservation_without_retry(self):
        reader = self.reader()
        with patch.object(metered.os, "pread", return_value=b"a") as io:
            self.deny(lambda: reader.read(3))
            self.deny(lambda: reader.read(3))
        self.assertEqual(io.call_count, 1)
        self.assertEqual(self.reservations, [(0, {"checkpoint_read_bytes": 3})])

    def test_read_os_error_is_redacted_and_not_retried(self):
        reader = self.reader()
        with patch.object(metered.os, "pread", side_effect=OSError("SYNTHETIC-SECRET")) as io:
            self.deny(lambda: reader.read(3))
            self.deny(lambda: reader.read(3))
        self.assertEqual(io.call_count, 1)
        self.assertEqual(self.reservations, [(0, {"checkpoint_read_bytes": 3})])

    def test_invalid_read_sizes_and_seek_are_rejected_before_reserving(self):
        for value in (-1, True, 1.0, None, metered.MAX_OPERATION_BYTES + 1):
            with self.subTest(value=value):
                reader = self.reader()
                self.deny(lambda reader=reader, value=value: reader.read(value))
        for offset, whence in ((-1, os.SEEK_SET), (7, os.SEEK_SET), (True, 0), (0, 999)):
            reader = self.reader()
            self.deny(lambda reader=reader, offset=offset, whence=whence:
                      reader.seek(offset, whence))
        self.assertEqual(self.reservations, [])

    def test_readinto_rejects_non_bytearray(self):
        for value in (b"xxx", memoryview(bytearray(3)), "xxx", None):
            reader = self.reader()
            self.deny(lambda reader=reader, value=value: reader.readinto(value))
        self.assertEqual(self.reservations, [])

    def test_reader_detects_change_before_content_read(self):
        fd, path = self.file()
        reader = self.reader(fd)
        path.write_bytes(b"mutated")
        with patch.object(metered.os, "pread") as io:
            self.deny(lambda: reader.read(2))
        io.assert_not_called()
        self.assertEqual(self.reservations, [])

    def test_reader_rechecks_identity_after_reservation_callback(self):
        fd, path = self.file()

        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            path.write_bytes(b"mutated")

        reader = self.reader(fd, reserve=reserve)
        with patch.object(metered.os, "pread") as io:
            self.deny(lambda: reader.read(2))
        io.assert_not_called()
        self.assertEqual(self.reservations, [(0, {"checkpoint_read_bytes": 2})])

    def test_read_mutation_after_syscall_never_returns_payload(self):
        fd, path = self.file()
        reader = self.reader(fd)
        original = os.pread

        def changing_read(descriptor, count, offset):
            payload = original(descriptor, count, offset)
            path.write_bytes(b"mutated")
            return payload

        with patch.object(metered.os, "pread", side_effect=changing_read):
            self.deny(lambda: reader.read(2))

    def test_writer_reserves_before_each_append(self):
        original = os.pwrite
        writer, path, _ = self.writer()
        observed = []

        def write(fd, payload, offset):
            self.assertEqual(self.reservations[-1], (0, {"private_write_bytes": len(payload)}))
            observed.append(offset)
            return original(fd, payload, offset)

        with patch.object(metered.os, "pwrite", side_effect=write):
            self.assertEqual(writer.write(b"ab"), 2)
            self.assertEqual(writer.write(b"cde"), 3)
        self.assertEqual(observed, [0, 2])
        self.assertEqual(writer.tell(), 5)
        self.assertEqual(path.read_bytes(), b"abcde")

    def test_writer_empty_write_has_no_reservation_or_io(self):
        writer, _, _ = self.writer()
        with patch.object(metered.os, "pwrite") as io:
            self.assertEqual(writer.write(b""), 0)
        io.assert_not_called()
        self.assertEqual(self.reservations, [])

    def test_writer_cap_rejects_before_second_reservation(self):
        writer, path, _ = self.writer(maximum=3)
        writer.write(b"ab")
        with patch.object(metered.os, "pwrite") as io:
            self.deny(lambda: writer.write(b"cd"))
        io.assert_not_called()
        self.assertEqual(self.reservations, [(0, {"private_write_bytes": 2})])
        self.assertEqual(path.read_bytes(), b"ab")

    def test_writer_invalid_payload_types_reject_without_io(self):
        for value in (bytearray(b"a"), memoryview(b"a"), "a", None, True):
            writer, _, _ = self.writer()
            with patch.object(metered.os, "pwrite") as io:
                self.deny(lambda writer=writer, value=value: writer.write(value))
            io.assert_not_called()

    def test_writer_reservation_failure_prevents_write(self):
        def refuse(*args, **kwargs):
            raise RuntimeError("SYNTHETIC-SECRET")

        writer, path, _ = self.writer(reserve=refuse)
        with patch.object(metered.os, "pwrite") as io:
            self.deny(lambda: writer.write(b"abc"))
        io.assert_not_called()
        self.assertEqual(path.read_bytes(), b"")

    def test_short_write_is_terminal_without_refund_or_retry(self):
        writer, _, _ = self.writer()
        with patch.object(metered.os, "pwrite", return_value=1) as io:
            self.deny(lambda: writer.write(b"abc"))
            self.deny(lambda: writer.write(b"abc"))
        self.assertEqual(io.call_count, 1)
        self.assertEqual(self.reservations, [(0, {"private_write_bytes": 3})])

    def test_write_os_error_is_terminal_without_refund_or_retry(self):
        writer, _, _ = self.writer()
        with patch.object(metered.os, "pwrite", side_effect=OSError("SYNTHETIC-SECRET")) as io:
            self.deny(lambda: writer.write(b"abc"))
            self.deny(lambda: writer.write(b"abc"))
        self.assertEqual(io.call_count, 1)
        self.assertEqual(self.reservations, [(0, {"private_write_bytes": 3})])

    def test_writer_rechecks_size_after_reservation(self):
        holder = {}

        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            holder["path"].write_bytes(b"changed")

        writer, path, _ = self.writer(reserve=reserve)
        holder["path"] = path
        with patch.object(metered.os, "pwrite") as io:
            self.deny(lambda: writer.write(b"abc"))
        io.assert_not_called()

    def test_constructor_rejection_does_not_take_descriptor_ownership(self):
        for mode, flags in ((0o644, os.O_RDONLY), (0o600, os.O_RDWR)):
            fd, _ = self.file(mode=mode, flags=flags)
            self.deny(lambda fd=fd: metered.MeteredReader(
                fd, rank=0, resource="checkpoint_read_bytes", reserve=self.reserve,
            ))
            self.assertGreater(os.fstat(fd).st_ino, 0)

    def test_reader_rejects_hardlink(self):
        fd, path = self.file()
        os.link(path, self.root / "second-link")
        self.deny(lambda: metered.MeteredReader(
            fd, rank=0, resource="checkpoint_read_bytes", reserve=self.reserve,
        ))
        self.assertGreater(os.fstat(fd).st_ino, 0)

    def test_writer_rejects_nonempty_and_append_descriptors(self):
        for payload, flags in ((b"old", os.O_WRONLY), (b"", os.O_WRONLY | os.O_APPEND)):
            fd, _ = self.file(payload, flags=flags)
            self.deny(lambda fd=fd: metered.MeteredWriter(
                fd, rank=0, resource="private_write_bytes", reserve=self.reserve,
                maximum_file_bytes=16,
            ))
            self.assertGreater(os.fstat(fd).st_ino, 0)

    def test_constructor_rejects_invalid_resources_ranks_and_callbacks(self):
        for rank, resource, reserve in (
            (True, "checkpoint_read_bytes", self.reserve),
            (-1, "checkpoint_read_bytes", self.reserve),
            (12, "checkpoint_read_bytes", self.reserve),
            (0, "private_write_bytes", self.reserve),
            (0, "SYNTHETIC-SECRET", self.reserve),
            (0, "checkpoint_read_bytes", None),
        ):
            fd, _ = self.file()
            self.deny(lambda fd=fd, rank=rank, resource=resource, reserve=reserve:
                      metered.MeteredReader(fd, rank=rank, resource=resource, reserve=reserve))

    def test_close_is_idempotent_and_never_exports_fd_or_paths(self):
        reader = self.reader()
        self.assertFalse(hasattr(reader, "fileno"))
        self.assertNotIn(str(self.root), repr(reader))
        with patch.object(metered.os, "close", wraps=os.close) as close:
            reader.close()
            reader.close()
        self.assertEqual(close.call_count, 1)
        self.assertTrue(reader.closed)
        self.deny(lambda: reader.read(1))

    def test_close_failure_never_retries_descriptor_number(self):
        fd, _ = self.file()
        reader = self.reader(fd)
        with patch.object(metered.os, "close", side_effect=OSError("SYNTHETIC-SECRET")) as close:
            self.deny(reader.close)
            try:
                reader.close()
            except metered.MeteredIOError:
                pass
        self.assertEqual(close.call_count, 1)
        os.close(fd)  # The injected syscall did not actually close this test-owned descriptor.

    def test_reused_descriptor_is_not_read_or_closed(self):
        fd, _ = self.file()
        reader = self.reader(fd)
        replacement, replacement_path = self.file(b"replacement")
        os.dup2(replacement, fd)
        with patch.object(metered.os, "pread") as read, patch.object(metered.os, "close") as close:
            self.deny(lambda: reader.read(1))
            try:
                reader.close()
            except metered.MeteredIOError:
                pass
        read.assert_not_called()
        close.assert_not_called()
        self.assertEqual(os.fstat(fd).st_ino, replacement_path.stat().st_ino)
        os.close(fd)

    def test_cross_process_and_thread_calls_deny_before_reserve(self):
        for provider in ("getpid", "get_ident"):
            reader = self.reader()
            target = metered.os if provider == "getpid" else metered.threading
            with patch.object(target, provider, return_value=-1):
                self.deny(lambda reader=reader: reader.read(1))
        self.assertEqual(self.reservations, [])

    def test_caught_reentry_still_prevents_outer_operation(self):
        holder = {}

        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            self.deny(lambda: holder["reader"].read(1))

        reader = self.reader(reserve=reserve)
        holder["reader"] = reader
        with patch.object(metered.os, "pread") as read:
            self.deny(lambda: reader.read(1))
        read.assert_not_called()
        self.assertEqual(len(self.reservations), 1)

    def test_real_shared_supervisor_denies_second_rank_read_before_io(self):
        ledger = controller()
        first = self.reader(reserve=ledger.reserve, rank=0)
        second = self.reader(reserve=ledger.reserve, rank=1)
        self.assertEqual(first.read(4), b"abcd")
        with patch.object(metered.os, "pread") as read:
            self.deny(lambda: second.read(4))
        read.assert_not_called()
        summary = ledger.summary()
        self.assertEqual(summary["resources_reserved"]["checkpoint_read_bytes"], 4)
        self.assertEqual(summary["state"], "failed")

    def test_log_category_is_not_double_charged_to_private_writes(self):
        ledger = controller()
        writer, path, _ = self.writer(reserve=ledger.reserve, resource="log_bytes")
        writer.write(b"abc")
        summary = ledger.summary()
        self.assertEqual(summary["resources_reserved"]["log_bytes"], 3)
        self.assertEqual(summary["resources_reserved"]["private_write_bytes"], 3)
        self.assertEqual(path.read_bytes(), b"abc")

    def test_all_reader_categories_route_exact_amount(self):
        for resource in ("checkpoint_read_bytes", "encoded_binary_read_bytes",
                         "encoded_index_read_bytes"):
            reader = self.reader(resource=resource)
            self.assertEqual(reader.read(1), b"a")
            self.assertEqual(self.reservations[-1], (0, {resource: 1}))

    def test_stdio_descriptors_reject_before_metadata_access(self):
        for fd in (0, 1, 2, -1, True, None):
            with patch.object(metered, "_snapshot") as snapshot:
                self.deny(lambda fd=fd: metered.MeteredReader(
                    fd, rank=0, resource="checkpoint_read_bytes", reserve=self.reserve,
                ))
            snapshot.assert_not_called()

    def test_inheritable_descriptor_rejects_without_ownership_transfer(self):
        fd, _ = self.file()
        os.set_inheritable(fd, True)
        self.deny(lambda: metered.MeteredReader(
            fd, rank=0, resource="checkpoint_read_bytes", reserve=self.reserve,
        ))
        self.assertTrue(os.get_inheritable(fd))
        os.set_inheritable(fd, False)

    def test_changed_status_flags_reject_before_reservation_or_read(self):
        fd, _ = self.file()
        reader = self.reader(fd)
        os.set_blocking(fd, False)
        with patch.object(metered.os, "pread") as read:
            self.deny(lambda: reader.read(1))
        read.assert_not_called()
        self.assertEqual(self.reservations, [])

    def test_reinitialization_preserves_original_fd_ownership(self):
        original_fd, _ = self.file()
        reader = self.reader(original_fd)
        new_fd, _ = self.file()
        self.deny(lambda: reader.__init__(
            new_fd, rank=0, resource="checkpoint_read_bytes", reserve=self.reserve,
        ))
        self.assertGreater(os.fstat(new_fd).st_ino, 0)
        reader.close()
        with self.assertRaises(OSError):
            os.fstat(original_fd)

    def test_writer_invalid_limits_reject_before_taking_fd(self):
        for maximum in (0, -1, True, None, 1.0, 2**63):
            fd, _ = self.file(b"", flags=os.O_WRONLY)
            with patch.object(metered, "_snapshot") as snapshot:
                self.deny(lambda fd=fd, maximum=maximum: metered.MeteredWriter(
                    fd, rank=0, resource="private_write_bytes", reserve=self.reserve,
                    maximum_file_bytes=maximum,
                ))
            snapshot.assert_not_called()
            self.assertGreater(os.fstat(fd).st_ino, 0)

    def test_per_operation_write_limit_with_tiny_fixture(self):
        writer, path, _ = self.writer()
        with patch.object(metered, "MAX_OPERATION_BYTES", 4):
            self.deny(lambda: writer.write(b"abcde"))
        self.assertEqual(self.reservations, [])
        self.assertEqual(path.read_bytes(), b"")

    def test_darwin_written_flag_transition_is_narrow_and_platform_bound(self):
        bit = 0x00010000
        with patch.object(metered.sys, "platform", "darwin"):
            self.assertTrue(metered._writer_flags_match(os.O_WRONLY, os.O_WRONLY | bit))
            self.assertTrue(metered._writer_flags_match(os.O_WRONLY | bit, os.O_WRONLY | bit))
            self.assertFalse(metered._writer_flags_match(os.O_WRONLY | bit, os.O_WRONLY))
            for other in (os.O_APPEND, os.O_NONBLOCK, os.O_RDWR):
                self.assertFalse(metered._writer_flags_match(os.O_WRONLY, os.O_WRONLY | other))
        with patch.object(metered.sys, "platform", "linux"):
            self.assertTrue(metered._writer_flags_match(os.O_WRONLY, os.O_WRONLY))
            self.assertFalse(metered._writer_flags_match(os.O_WRONLY, os.O_WRONLY | bit))

    def test_error_constructor_never_copies_unknown_error_text(self):
        for value in ("SYNTHETIC-SECRET", None, True, {}, 0):
            error = metered.MeteredIOError(value)
            self.assertEqual(error.args, ("internal_error",))
            self.assertNotIn("SYNTHETIC-SECRET", repr(error))


if __name__ == "__main__":
    unittest.main()
