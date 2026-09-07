"""Tiny stdlib staging files only: no model/data, ML imports or external services."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_checkpoint_saver_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(
    f"{PACKAGE}.posttraining_sft_checkpoint_saver_v1",
    ROOT / "src/lumi/posttraining_sft_checkpoint_saver_v1.py",
)
saver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(saver)
metered = sys.modules[f"{PACKAGE}.posttraining_sft_metered_io_v1"]
supervisor_spec = importlib.util.spec_from_file_location(
    f"{PACKAGE}.posttraining_sft_supervisor_v1",
    ROOT / "src/lumi/posttraining_sft_supervisor_v1.py",
)
supervision = importlib.util.module_from_spec(supervisor_spec)
supervisor_spec.loader.exec_module(supervision)


def receipt(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def controller(*, writes, reads):
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
        "maximum_per_gpu_memory_bytes": 2048, "maximum_private_write_bytes": writes,
        "maximum_checkpoint_read_bytes": reads,
        "maximum_encoded_binary_read_bytes": counts["maximum_selected_binary_bytes"],
        "maximum_encoded_index_read_bytes": counts["maximum_selected_index_bytes"],
        "maximum_log_bytes": 1, "maximum_invocations": 3,
        "maximum_global_optimizer_steps": 4, "maximum_external_network_bytes": 0,
        "maximum_inference_calls": 0,
    }
    result = supervision.Supervisor(caps, counts, world_size=2, clock_ns=lambda: 0)
    result.begin_invocation("reference", 0, 2, False)
    return result


class SyntheticSaveFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lumi-SYNTHETIC-checkpoint-save-v1-")
        self.root = Path(self.temporary.name)
        self.root.chmod(0o700)
        self.handles, self.descriptors, self.reservations = [], set(), []
        self.counter = 0

    def tearDown(self):
        for handle in reversed(self.handles):
            try:
                handle.close()
            except saver.CheckpointSaveError:
                pass
        for fd in self.descriptors:
            os.close(fd)
        self.temporary.cleanup()

    def reserve(self, rank, **amounts):
        self.reservations.append((rank, amounts))

    def total(self, resource):
        return sum(amounts.get(resource, 0) for _, amounts in self.reservations)

    def deny(self, call, *args, code=None):
        with self.assertRaises(saver.CheckpointSaveError) as caught:
            call(*args)
        error = caught.exception
        self.assertEqual(error.args, (error.code,))
        self.assertIn(error.code, saver.ERROR_CODES)
        if code is not None:
            self.assertEqual(error.code, code)
        self.assertNotIn(str(self.root), repr(error))
        self.assertNotIn("SYNTHETIC-SECRET", repr(error))
        return error

    def file(self, name, data=b"", *, mode=0o600):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_bytes(data)
        path.chmod(mode)
        return path

    def stream(self, *, payload=b"", mode=0o600, flags=os.O_WRONLY, **overrides):
        self.counter += 1
        path = self.file(f"stream-{self.counter}.bin", payload, mode=mode)
        fd = os.open(path, flags | os.O_CLOEXEC)
        self.descriptors.add(fd)
        options = {"rank": 0, "reserve": self.reserve, "maximum_file_bytes": 1024,
                   "maximum_write_call_bytes": 1024, "maximum_sync_operations": 8}
        options.update(overrides)
        result = saver.CheckpointWriteStream(fd, **options)
        self.descriptors.remove(fd)
        self.handles.append(result)
        return result, path, fd

    def completed_streams(self, reserve):
        expected = {}
        for rank, data in enumerate((b"abc", b"defg")):
            stream, path, _ = self.stream(rank=rank, reserve=reserve)
            stream.write(data)
            stream.close()
            expected[path.name] = stream.receipt()
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual(expected[path.name], receipt(data))
        return expected

    def bind_finalizer(self, expected, reserve):
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.descriptors.add(fd)
        finalizer = saver.CheckpointStagingFinalizer(
            fd, rank=0, reserve=reserve, expected_files=expected,
            maximum_file_bytes=1024, maximum_read_bytes=8192,
            maximum_manifest_write_bytes=4096, maximum_sync_operations=32,
        )
        self.descriptors.remove(fd)
        self.handles.append(finalizer)
        return finalizer


class CheckpointWriteStreamTests(SyntheticSaveFixture):
    def test_payload_streams_and_finalizer_share_one_real_ledger(self):
        ledger = controller(writes=4096, reads=8192)
        expected = self.completed_streams(ledger.reserve)
        result = self.bind_finalizer(expected, ledger.reserve).finish()
        manifest = (self.root / "manifest.sha256").read_bytes()
        payload_bytes = sum(item["bytes"] for item in expected.values())
        self.assertEqual(result["payload_bytes"], payload_bytes)
        self.assertEqual(result["private_write_bytes"], len(manifest))
        used = ledger.summary()["resources_reserved"]
        self.assertEqual(used["private_write_bytes"], payload_bytes + len(manifest))
        self.assertEqual(used["checkpoint_read_bytes"], 2 * payload_bytes + len(manifest))
        self.assertEqual(used["log_bytes"], 0)
        self.assertFalse(ledger.summary()["production_eligible"])

    def test_shared_ledger_failure_prevents_subsequent_file_read_and_write(self):
        ledger = controller(writes=7, reads=8192)
        expected = self.completed_streams(ledger.reserve)
        finalizer = self.bind_finalizer(expected, ledger.reserve)
        self.deny(finalizer.finish)
        summary = ledger.summary()
        self.assertEqual(summary["failure_code"], "resource_exhausted")
        self.assertEqual(summary["resources_reserved"]["private_write_bytes"], 7)
        self.assertEqual((self.root / "manifest.sha256").read_bytes(), b"")
        stream, _, _ = self.stream(rank=1, reserve=ledger.reserve)
        with patch.object(saver.os, "pwrite", wraps=os.pwrite) as write:
            self.deny(stream.write, b"x")
        write.assert_not_called()
        fd = os.open(self.root / "stream-1.bin", os.O_RDONLY | os.O_CLOEXEC)
        reader = metered.MeteredReader(fd, rank=1, resource="checkpoint_read_bytes",
                                       reserve=ledger.reserve)
        try:
            with patch.object(saver.os, "pread", wraps=os.pread) as read:
                with self.assertRaises(metered.MeteredIOError):
                    reader.read(1)
            read.assert_not_called()
        finally:
            reader.close()
        self.assertEqual(ledger.summary()["resources_reserved"], summary["resources_reserved"])

    def test_bounded_buffers_hash_receipt_and_targeted_sync(self):
        stream, path, fd = self.stream(rank=2)
        with patch.object(saver.os, "fsync", wraps=os.fsync) as sync:
            self.assertEqual(stream.tell(), 0)
            self.assertEqual(stream.write(b"ab"), 2)
            self.assertEqual(stream.write(bytearray(b"cd")), 2)
            self.assertEqual(stream.write(memoryview(b"ef")), 2)
            self.assertEqual(stream.tell(), 6)
            stream.flush()
            stream.flush()
            stream.close()
            stream.close()
        self.assertEqual(sync.call_count, 1)
        self.assertEqual(sync.call_args.args, (fd,))
        self.assertTrue(stream.closed)
        self.assertEqual(path.read_bytes(), b"abcdef")
        self.assertEqual(stream.receipt(), receipt(b"abcdef"))
        self.assertEqual(self.total("private_write_bytes"), 6)
        self.assertTrue(all(rank == 2 for rank, _ in self.reservations))
        self.assertFalse(hasattr(stream, "fileno"))
        self.assertNotIn(str(path), repr(stream))

    def test_empty_close_syncs_once_and_has_empty_digest(self):
        stream, path, _ = self.stream()
        with patch.object(saver.os, "fsync", wraps=os.fsync) as sync:
            self.assertEqual(stream.write(b""), 0)
            stream.close()
        self.assertEqual(sync.call_count, 1)
        self.assertEqual(stream.receipt(), receipt(b""))
        self.assertEqual(path.read_bytes(), b"")
        self.assertEqual(self.reservations, [])

    def test_flush_after_new_writes_resyncs_but_clean_flush_is_noop(self):
        stream, _, _ = self.stream()
        with patch.object(saver.os, "fsync", wraps=os.fsync) as sync:
            stream.flush()
            stream.flush()
            stream.write(b"abc")
            stream.flush()
            stream.close()
        self.assertEqual(sync.call_count, 2)

    def test_write_splits_chunks_and_reserves_before_each_system_write(self):
        events = []
        original = os.pwrite

        def reserve(rank, **amounts):
            events.append(("reserve", amounts["private_write_bytes"]))
            self.reserve(rank, **amounts)

        def write(fd, data, offset):
            events.append(("write", len(data)))
            return original(fd, data, offset)

        stream, path, _ = self.stream(reserve=reserve)
        with patch.object(saver, "MAX_OPERATION_BYTES", 4), \
                patch.object(saver.os, "pwrite", side_effect=write):
            self.assertEqual(stream.write(b"abcdefghij"), 10)
        stream.close()
        self.assertEqual(events, [("reserve", 4), ("write", 4), ("reserve", 4),
                                  ("write", 4), ("reserve", 2), ("write", 2)])
        self.assertEqual(path.read_bytes(), b"abcdefghij")
        self.assertEqual(stream.receipt(), receipt(b"abcdefghij"))

    def test_call_cap_rejects_before_any_reserve_or_write(self):
        stream, path, _ = self.stream(maximum_write_call_bytes=3)
        with patch.object(saver.os, "pwrite", wraps=os.pwrite) as write:
            self.deny(stream.write, b"abcd")
        write.assert_not_called()
        self.assertEqual(self.reservations, [])
        self.assertEqual(path.read_bytes(), b"")
        self.deny(stream.write, b"a", code="terminal_failure")

    def test_remaining_file_cap_preflights_entire_call(self):
        stream, path, _ = self.stream(maximum_file_bytes=6)
        stream.write(b"abc")
        with patch.object(saver.os, "pwrite", wraps=os.pwrite) as write:
            self.deny(stream.write, b"defg")
        write.assert_not_called()
        self.assertEqual(self.total("private_write_bytes"), 3)
        self.assertEqual(path.read_bytes(), b"abc")

    def test_exact_file_and_call_caps_are_allowed(self):
        stream, _, _ = self.stream(maximum_file_bytes=4, maximum_write_call_bytes=4)
        self.assertEqual(stream.write(b"abcd"), 4)
        stream.close()
        self.assertEqual(stream.receipt(), receipt(b"abcd"))

    def test_rejects_unsupported_buffers_before_reservation(self):
        class BytesSubclass(bytes):
            pass

        invalid = ["abc", 3, None, [1, 2], BytesSubclass(b"a"),
                   memoryview(b"abcd")[::2], memoryview(b"abcd").cast("B", shape=[2, 2]),
                   memoryview(b"abcd").cast("I")]
        for data in invalid:
            with self.subTest(kind=type(data).__name__):
                stream, path, _ = self.stream()
                before = self.total("private_write_bytes")
                self.deny(stream.write, data)
                self.assertEqual(self.total("private_write_bytes"), before)
                self.assertEqual(path.read_bytes(), b"")

    def test_short_write_is_not_retried_or_refunded(self):
        stream, _, _ = self.stream()
        with patch.object(saver.os, "pwrite", return_value=1) as write:
            self.deny(stream.write, b"abcd")
        self.assertEqual(write.call_count, 1)
        self.assertEqual(self.total("private_write_bytes"), 4)
        self.deny(stream.write, b"a", code="terminal_failure")
        stream.close()
        self.deny(stream.receipt)

    def test_failed_second_chunk_keeps_first_chunk_and_all_reservations(self):
        original = os.pwrite
        calls = []

        def write(fd, data, offset):
            calls.append(offset)
            if len(calls) == 2:
                raise OSError("SYNTHETIC-SECRET")
            return original(fd, data, offset)

        stream, path, _ = self.stream()
        with patch.object(saver, "MAX_OPERATION_BYTES", 4), \
                patch.object(saver.os, "pwrite", side_effect=write):
            self.deny(stream.write, b"abcdefghij")
        self.assertEqual(calls, [0, 4])
        self.assertEqual(path.read_bytes(), b"abcd")
        self.assertEqual(self.total("private_write_bytes"), 8)

    def test_reserve_exception_cannot_write_or_leak_text(self):
        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            raise RuntimeError("SYNTHETIC-SECRET")

        stream, path, _ = self.stream(reserve=reserve)
        with patch.object(saver.os, "pwrite", wraps=os.pwrite) as write:
            self.deny(stream.write, b"abc")
        write.assert_not_called()
        self.assertEqual(path.read_bytes(), b"")
        self.assertEqual(self.total("private_write_bytes"), 3)

    def test_non_none_reserve_result_rejects(self):
        stream, path, _ = self.stream(reserve=lambda rank, **amounts: False)
        self.deny(stream.write, b"abc")
        self.assertEqual(path.read_bytes(), b"")

    def test_swallowed_reentry_poisoning_stops_outer_write(self):
        holder = []

        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            self.deny(holder[0].tell)

        stream, path, _ = self.stream(reserve=reserve)
        holder.append(stream)
        self.deny(stream.write, b"abc")
        self.assertEqual(path.read_bytes(), b"")
        self.assertEqual(self.total("private_write_bytes"), 3)

    def test_callback_file_metadata_change_rejects_before_write(self):
        target = []

        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            target[0].chmod(0o640)

        stream, path, _ = self.stream(reserve=reserve)
        target.append(path)
        self.deny(stream.write, b"abc")
        self.assertEqual(path.read_bytes(), b"")

    def test_sync_failure_is_terminal_and_never_host_wide(self):
        stream, _, _ = self.stream()
        stream.write(b"abc")
        with patch.object(saver.os, "fsync", side_effect=OSError("SYNTHETIC-SECRET")) as sync, \
                patch.object(saver.os, "sync", create=True) as global_sync:
            self.deny(stream.flush)
            self.deny(stream.flush, code="terminal_failure")
            stream.close()
        self.assertEqual(sync.call_count, 1)
        global_sync.assert_not_called()
        self.deny(stream.receipt)

    def test_sync_cap_checked_before_second_dirty_sync(self):
        stream, _, _ = self.stream(maximum_sync_operations=1)
        with patch.object(saver.os, "fsync", wraps=os.fsync) as sync:
            stream.flush()
            stream.write(b"a")
            self.deny(stream.flush)
            stream.close()
        self.assertEqual(sync.call_count, 1)

    def test_receipt_before_close_is_rejected(self):
        stream, _, _ = self.stream()
        self.deny(stream.receipt, code="not_closed")

    def test_closed_stream_rejects_further_writes(self):
        stream, _, _ = self.stream()
        stream.close()
        self.deny(stream.write, b"a", code="closed")

    def test_invalid_constructor_preserves_callers_fd(self):
        path = self.file("constructor.bin")
        fd = os.open(path, os.O_WRONLY | os.O_CLOEXEC)
        self.descriptors.add(fd)
        options = {"rank": 0, "reserve": self.reserve, "maximum_file_bytes": 1024,
                   "maximum_write_call_bytes": 1024, "maximum_sync_operations": 8}
        for key, value in [("rank", True), ("rank", 12), ("reserve", None),
                           ("maximum_file_bytes", 0), ("maximum_file_bytes", True),
                           ("maximum_write_call_bytes", -1), ("maximum_sync_operations", 0)]:
            with self.subTest(key=key, value=value):
                with self.assertRaises(saver.CheckpointSaveError):
                    saver.CheckpointWriteStream(fd, **{**options, key: value})
                self.assertEqual(os.fstat(fd).st_size, 0)

    def test_nonempty_readonly_append_or_insecure_files_reject(self):
        cases = [{"payload": b"x"}, {"flags": os.O_RDONLY},
                 {"flags": os.O_WRONLY | os.O_APPEND}, {"mode": 0o644}]
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(saver.CheckpointSaveError):
                    self.stream(**options)

    def test_inheritable_descriptor_rejects_and_remains_owned_by_caller(self):
        path = self.file("inheritable.bin")
        fd = os.open(path, os.O_WRONLY)
        self.descriptors.add(fd)
        os.set_inheritable(fd, True)
        with self.assertRaises(saver.CheckpointSaveError):
            saver.CheckpointWriteStream(fd, rank=0, reserve=self.reserve,
                                        maximum_file_bytes=32, maximum_write_call_bytes=32,
                                        maximum_sync_operations=2)
        self.assertEqual(os.fstat(fd).st_size, 0)

    def test_wrong_thread_owner_is_terminal(self):
        stream, path, _ = self.stream()
        with patch.object(saver.threading, "get_ident", return_value=-1):
            self.deny(stream.write, b"a", code="owner_mismatch")
        self.assertEqual(path.read_bytes(), b"")
        self.deny(stream.write, b"a", code="terminal_failure")

    def test_close_failure_revokes_handle_without_retry(self):
        stream, _, fd = self.stream()
        stream.write(b"a")
        original = os.close
        calls = []

        def close(descriptor):
            calls.append(descriptor)
            original(descriptor)
            raise OSError("SYNTHETIC-SECRET")

        with patch.object(saver.os, "close", side_effect=close):
            self.deny(stream.close)
            stream.close()
        self.assertEqual(calls, [fd])
        self.assertTrue(stream.closed)
        self.deny(stream.receipt)

    def test_swallowed_reentry_during_close_sync_rejects_receipt(self):
        stream, _, _ = self.stream()
        stream.write(b"a")
        original = os.fsync

        def sync(fd):
            self.deny(stream.tell)
            return original(fd)

        with patch.object(saver.os, "fsync", side_effect=sync) as calls:
            self.deny(stream.close)
            stream.close()
        self.assertEqual(calls.call_count, 1)
        self.assertTrue(stream.closed)
        self.deny(stream.receipt)

    def test_unknown_error_code_is_redacted(self):
        error = saver.CheckpointSaveError("SYNTHETIC-SECRET")
        self.assertEqual(error.args, ("internal_error",))


class CheckpointStagingFinalizerTests(SyntheticSaveFixture):
    def setUp(self):
        super().setUp()
        self.payloads = {"config.json": b"{}", "model/.metadata": b"NOT-PICKLE",
                         "model/__0_0.distcp": b"abcdef"}
        for name, data in self.payloads.items():
            self.file(name, data)
        self.expected = {name: receipt(data) for name, data in self.payloads.items()}

    def finalizer(self, **overrides):
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.descriptors.add(fd)
        options = {"rank": 0, "reserve": self.reserve, "expected_files": self.expected,
                   "maximum_file_bytes": 1024, "maximum_read_bytes": 8192,
                   "maximum_manifest_write_bytes": 4096, "maximum_sync_operations": 32}
        options.update(overrides)
        result = saver.CheckpointStagingFinalizer(fd, **options)
        self.descriptors.remove(fd)
        self.handles.append(result)
        return result

    def manifest_bytes(self):
        return "".join(f"{hashlib.sha256(self.payloads[str(path)]).hexdigest()}  {path}\n"
                       for path in sorted(map(Path, self.payloads))).encode("ascii")

    def assert_staging_receipt(self, result, manifest):
        size = sum(map(len, self.payloads.values()))
        self.assertEqual(result["status"], "verified_staging")
        self.assertEqual(result["manifest_sha256"], hashlib.sha256(manifest).hexdigest())
        self.assertEqual(result["payload_files"], len(self.payloads))
        self.assertEqual(result["payload_bytes"], size)
        self.assertEqual(result["checkpoint_read_bytes"], 2 * size + len(manifest))
        self.assertEqual(result["private_write_bytes"], len(manifest))
        self.assertIs(type(result["reserved_sync_operations"]), int)
        self.assertGreater(result["reserved_sync_operations"], 0)
        self.assertLessEqual(result["reserved_sync_operations"], 32)
        self.assertIs(result["published"], False)
        self.assertIs(result["production_eligible"], False)
        self.assertNotIn(str(self.root), repr(result))

    def test_finished_staging_is_reverified_metered_private_and_not_published(self):
        finalizer = self.finalizer(rank=3)
        manifest = self.manifest_bytes()
        with patch.object(saver.os, "rename", wraps=os.rename) as rename, \
                patch.object(saver.os, "replace", wraps=os.replace) as replace, \
                patch.object(saver.os, "sync", create=True) as global_sync:
            result = finalizer.finish()
        self.assert_staging_receipt(result, manifest)
        self.assertEqual((self.root / "manifest.sha256").read_bytes(), manifest)
        self.assertEqual((self.root / "manifest.sha256").stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.total("checkpoint_read_bytes"), result["checkpoint_read_bytes"])
        self.assertEqual(self.total("private_write_bytes"), len(manifest))
        self.assertTrue(all(rank == 3 for rank, _ in self.reservations))
        self.assertTrue(all(set(amounts) <= {"private_write_bytes", "checkpoint_read_bytes"}
                            for _, amounts in self.reservations))
        rename.assert_not_called()
        replace.assert_not_called()
        global_sync.assert_not_called()
        for name, data in self.payloads.items():
            self.assertEqual((self.root / name).read_bytes(), data)

    def test_manifest_order_matches_path_objects_not_full_strings(self):
        self.payloads.update({"a.zz": b"z", "a/file": b"x"})
        for name in ("a.zz", "a/file"):
            self.file(name, self.payloads[name])
            self.expected[name] = receipt(self.payloads[name])
        self.finalizer().finish()
        manifest = (self.root / "manifest.sha256").read_bytes()
        self.assertEqual(manifest, self.manifest_bytes())
        self.assertLess(manifest.index(b"  a/file\n"), manifest.index(b"  a.zz\n"))

    def test_directory_create_link_transition_is_platform_specific_and_exact(self):
        before = (11, 22, 0o40700, 3, 501, 96, 100, 100)
        for platform in ("darwin", "linux"):
            with patch.object(saver.sys, "platform", platform):
                self.assertTrue(saver._created_file_directory_match(before, before))
                for delta in (-1, 1, 2):
                    after = (*before[:3], before[3] + delta, before[4], 128, 101, 101)
                    self.assertEqual(saver._created_file_directory_match(before, after),
                                     platform == "darwin" and delta == 1)
                for index in (0, 1, 2, 4):
                    after = list(before)
                    after[index] += 1
                    self.assertFalse(saver._created_file_directory_match(before, tuple(after)))

    def test_empty_payload_and_empty_private_directory_are_valid(self):
        self.payloads["empty.bin"] = b""
        self.file("empty.bin")
        self.expected["empty.bin"] = receipt(b"")
        (self.root / "empty-directory").mkdir(mode=0o700)
        result = self.finalizer().finish()
        self.assert_staging_receipt(result, self.manifest_bytes())

    def test_exact_content_caps_allow_completion(self):
        manifest = self.manifest_bytes()
        expected_reads = 2 * sum(map(len, self.payloads.values())) + len(manifest)
        result = self.finalizer(maximum_read_bytes=expected_reads,
                                maximum_manifest_write_bytes=len(manifest)).finish()
        self.assert_staging_receipt(result, manifest)

    def test_finalize_twice_cannot_reread_or_rewrite(self):
        finalizer = self.finalizer()
        finalizer.finish()
        counts = list(self.reservations)
        before = (self.root / "manifest.sha256").read_bytes()
        self.deny(finalizer.finish)
        self.assertEqual(counts, self.reservations)
        self.assertEqual(before, (self.root / "manifest.sha256").read_bytes())

    def test_closed_finalizer_cannot_start(self):
        finalizer = self.finalizer()
        finalizer.close()
        finalizer.close()
        self.assertTrue(finalizer.closed)
        self.deny(finalizer.finish, code="closed")
        self.assertEqual(self.reservations, [])

    def test_extra_payload_rejects_without_content_reads_or_manifest(self):
        self.file("extra.bin", b"SYNTHETIC-SECRET")
        self.deny(self.finalizer().finish)
        self.assertEqual(self.reservations, [])
        self.assertFalse((self.root / "manifest.sha256").exists())

    def test_missing_payload_rejects_without_content_reads_or_manifest(self):
        (self.root / "config.json").unlink()
        self.deny(self.finalizer().finish)
        self.assertEqual(self.reservations, [])
        self.assertFalse((self.root / "manifest.sha256").exists())

    def test_prior_manifest_is_preserved_and_never_read(self):
        path = self.file("manifest.sha256", b"SYNTHETIC-SECRET")
        self.deny(self.finalizer().finish)
        self.assertEqual(path.read_bytes(), b"SYNTHETIC-SECRET")
        self.assertEqual(self.reservations, [])

    def test_wrong_digest_stops_before_manifest_creation_and_no_retry(self):
        self.expected["config.json"] = {"bytes": 2, "sha256": "0" * 64}
        finalizer = self.finalizer()
        self.deny(finalizer.finish)
        spent = list(self.reservations)
        self.assertGreater(self.total("checkpoint_read_bytes"), 0)
        self.assertEqual(self.total("private_write_bytes"), 0)
        self.assertFalse((self.root / "manifest.sha256").exists())
        self.deny(finalizer.finish, code="terminal_failure")
        self.assertEqual(spent, self.reservations)

    def test_wrong_recorded_size_rejects_without_manifest(self):
        self.expected["config.json"] = {**self.expected["config.json"], "bytes": 3}
        self.deny(self.finalizer().finish)
        self.assertFalse((self.root / "manifest.sha256").exists())
        self.assertEqual(self.total("private_write_bytes"), 0)

    def test_payload_over_per_file_cap_rejects_without_read(self):
        with self.assertRaises(saver.CheckpointSaveError):
            self.finalizer(maximum_file_bytes=3).finish()
        self.assertEqual(self.reservations, [])

    def test_read_cap_never_overspends(self):
        finalizer = self.finalizer(maximum_read_bytes=1)
        self.deny(finalizer.finish)
        self.assertLessEqual(self.total("checkpoint_read_bytes"), 1)
        self.assertEqual(self.total("private_write_bytes"), 0)

    def test_manifest_write_cap_never_overspends(self):
        with self.assertRaises(saver.CheckpointSaveError):
            self.finalizer(maximum_manifest_write_bytes=1).finish()
        self.assertEqual(self.total("private_write_bytes"), 0)

    def test_sync_cap_never_calls_extra_sync(self):
        finalizer = self.finalizer(maximum_sync_operations=1)
        with patch.object(saver.os, "fsync", wraps=os.fsync) as sync:
            self.deny(finalizer.finish)
        self.assertLessEqual(sync.call_count, 1)

    def test_payload_sync_failure_preserves_payloads_without_publication(self):
        finalizer = self.finalizer()
        with patch.object(saver.os, "fsync", side_effect=OSError("SYNTHETIC-SECRET")) as sync:
            self.deny(finalizer.finish)
        self.assertEqual(sync.call_count, 1)
        self.assertFalse((self.root / "manifest.sha256").exists())
        for name, data in self.payloads.items():
            self.assertEqual((self.root / name).read_bytes(), data)

    def test_symlink_and_hardlink_are_not_followed(self):
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind):
                target = self.root / "alias"
                if kind == "symlink":
                    target.symlink_to(self.root / "config.json")
                else:
                    os.link(self.root / "config.json", target)
                try:
                    self.deny(self.finalizer().finish)
                    self.assertEqual(self.reservations, [])
                finally:
                    target.unlink()

    def test_wrong_payload_or_directory_permissions_reject(self):
        for path, bad, restored in [(self.root / "config.json", 0o644, 0o600),
                                    (self.root / "model", 0o755, 0o700)]:
            with self.subTest(directory=path.is_dir()):
                path.chmod(bad)
                self.deny(self.finalizer().finish)
                self.assertEqual(self.reservations, [])
                path.chmod(restored)

    def test_reserve_error_is_redacted_and_keeps_attempted_charge(self):
        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            raise RuntimeError("SYNTHETIC-SECRET")

        self.deny(self.finalizer(reserve=reserve).finish)
        self.assertGreater(self.total("checkpoint_read_bytes"), 0)
        self.assertFalse((self.root / "manifest.sha256").exists())

    def test_callback_injected_file_is_not_accepted_as_own_mutation(self):
        injected = []

        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            if not injected:
                injected.append(self.file("injected.bin", b"SYNTHETIC-SECRET"))

        finalizer = self.finalizer(reserve=reserve)
        self.deny(finalizer.finish)
        self.assertFalse((self.root / "manifest.sha256").exists())
        self.assertEqual(injected[0].read_bytes(), b"SYNTHETIC-SECRET")

    def test_caller_receipt_mutation_does_not_change_constructor_binding(self):
        finalizer = self.finalizer()
        self.expected["config.json"]["sha256"] = "0" * 64
        self.expected["config.json"]["bytes"] = 999
        result = finalizer.finish()
        self.assert_staging_receipt(result, self.manifest_bytes())

    def test_verification_failure_preserves_written_manifest_without_restart(self):
        def reserve(rank, **amounts):
            already_written = self.total("private_write_bytes")
            self.reserve(rank, **amounts)
            if already_written and "checkpoint_read_bytes" in amounts:
                raise RuntimeError("SYNTHETIC-SECRET")

        finalizer = self.finalizer(reserve=reserve)
        self.deny(finalizer.finish)
        self.assertEqual((self.root / "manifest.sha256").read_bytes(), self.manifest_bytes())
        counts = list(self.reservations)
        self.deny(finalizer.finish, code="terminal_failure")
        finalizer.close()
        self.assertEqual(counts, self.reservations)
        self.assertTrue((self.root / "manifest.sha256").exists())

    def test_directory_sync_failure_preserves_staging_evidence(self):
        original = os.fsync
        count = 0

        def sync(fd):
            nonlocal count
            count += 1
            if count > len(self.payloads) + 1:
                raise OSError("SYNTHETIC-SECRET")
            return original(fd)

        finalizer = self.finalizer()
        with patch.object(saver.os, "fsync", side_effect=sync):
            self.deny(finalizer.finish)
            finalizer.close()
        self.assertEqual(count, len(self.payloads) + 2)
        self.assertEqual((self.root / "manifest.sha256").read_bytes(), self.manifest_bytes())

    def test_short_read_is_not_retried_and_leaves_no_manifest(self):
        finalizer = self.finalizer()
        with patch.object(saver.os, "pread", return_value=b"") as read:
            self.deny(finalizer.finish)
        self.assertEqual(read.call_count, 1)
        self.assertGreater(self.total("checkpoint_read_bytes"), 0)
        self.assertFalse((self.root / "manifest.sha256").exists())

    def test_swallowed_reentry_stops_outer_finalization(self):
        holder = []

        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            self.deny(holder[0].finish)

        finalizer = self.finalizer(reserve=reserve)
        holder.append(finalizer)
        self.deny(finalizer.finish)
        self.assertFalse((self.root / "manifest.sha256").exists())

    def test_invalid_names_and_receipts_reject_without_fd_transfer(self):
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.descriptors.add(fd)
        valid = receipt(b"x")
        invalid = [{}, {"a": valid}, {"../escape": valid, "b": valid},
                   {"/absolute": valid, "b": valid}, {"a/./b": valid, "b": valid},
                   {"a//b": valid, "b": valid}, {"manifest.sha256": valid, "b": valid},
                   {"a/manifest.sha256": valid, "b": valid}, {"a b": valid, "b": valid},
                   {"a": {"bytes": True, "sha256": "a" * 64}, "b": valid},
                   {"a": {"bytes": 1, "sha256": "A" * 64}, "b": valid},
                   {"a": {"bytes": 1}, "b": valid},
                   {"a": {**valid, "extra": False}, "b": valid}]
        for index, expected in enumerate(invalid):
            with self.subTest(index=index):
                with self.assertRaises(saver.CheckpointSaveError):
                    saver.CheckpointStagingFinalizer(
                        fd, rank=0, reserve=self.reserve, expected_files=expected,
                        maximum_file_bytes=1024, maximum_read_bytes=8192,
                        maximum_manifest_write_bytes=4096, maximum_sync_operations=32,
                    )
                self.assertTrue(os.fstat(fd).st_mode & 0o40000)
        self.assertEqual(self.reservations, [])

    def test_invalid_caps_preserve_root_fd(self):
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.descriptors.add(fd)
        options = {"rank": 0, "reserve": self.reserve, "expected_files": self.expected,
                   "maximum_file_bytes": 1024, "maximum_read_bytes": 8192,
                   "maximum_manifest_write_bytes": 4096, "maximum_sync_operations": 32}
        for key, value in [("rank", True), ("rank", 12), ("reserve", None),
                           ("maximum_file_bytes", 0), ("maximum_read_bytes", True),
                           ("maximum_manifest_write_bytes", -1), ("maximum_sync_operations", 0)]:
            with self.subTest(key=key, value=value):
                with self.assertRaises(saver.CheckpointSaveError):
                    saver.CheckpointStagingFinalizer(fd, **{**options, key: value})
                self.assertTrue(os.fstat(fd).st_mode & 0o40000)

    def test_insecure_root_rejects_constructor(self):
        self.root.chmod(0o755)
        with self.assertRaises(saver.CheckpointSaveError):
            self.finalizer()


if __name__ == "__main__":
    unittest.main()
