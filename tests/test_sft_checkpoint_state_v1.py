"""Tiny owned hash-manifest fixtures; no ML, model loading, or real checkpoints."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_checkpoint_state_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
checkpoint = importlib.import_module(f"{PACKAGE}.posttraining_sft_checkpoint_state_v1")
IMPORTED_MODULES = set(sys.modules) - before_import
workers = importlib.import_module(f"{PACKAGE}.posttraining_sft_checkpoint_worker_v1")


class CheckpointStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lumi-SYNTHETIC-checkpoint-state-v1-")
        self.parent = Path(self.temporary.name)
        self.parent.chmod(0o700)
        self.name = "checkpoint-00000003"
        self.root = self.parent / self.name
        self.root.mkdir(mode=0o700)
        (self.root / "model").mkdir(mode=0o700)
        self.parent_fd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.state = {"latest_checkpoint": self.name, "status": "checkpoint", "step": 3,
                      "synthetic_private_value": "SYNTHETIC-PRIVATE"}
        self.payloads = {
            "model/.metadata": b"SYNTHETIC-NOT-PICKLE",
            "model/__0_0.distcp": b"SYNTHETIC-NOT-TENSOR",
            "trainer-rank-00000.pt": b"SYNTHETIC-NOT-PICKLE",
            workers.STATE_NAME: self.canonical(self.state),
        }
        self.charges, self.interrupts = [], []
        self.pin_files()

    def tearDown(self):
        os.close(self.parent_fd)
        self.temporary.cleanup()

    @staticmethod
    def canonical(state):
        return (json.dumps(state, sort_keys=True, ensure_ascii=True, allow_nan=False,
                           separators=(",", ":")) + "\n").encode("ascii")

    def write(self, name, data):
        path = self.root / name
        path.write_bytes(data)
        path.chmod(0o600)

    def pin_files(self):
        for name, data in self.payloads.items():
            self.write(name, data)
        self.manifest = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n"
                                for name, data in self.payloads.items()).encode("ascii")
        self.write("manifest.sha256", self.manifest)
        self.pin = hashlib.sha256(self.manifest).hexdigest()

    def reserve(self, rank, **amounts):
        self.charges.append((rank, dict(amounts)))

    def worker(self, **changes):
        limits = {"maximum_file_bytes": 131_072, "maximum_read_bytes": 524_288,
                  "maximum_read_call_bytes": 17, "maximum_write_call_bytes": 64,
                  "maximum_sync_operations": 8}
        limits.update(changes.pop("limits", {}))
        options = {"rank": 0, "world_size": 1, "reserve": self.reserve,
                   "interrupt": lambda: self.interrupts.append(True), "limits": limits}
        options.update(changes)
        return workers.CheckpointWorker(**options)

    def read(self, worker, **changes):
        options = {"checkpoint_name": self.name, "manifest_sha256": self.pin,
                   "checkpoint_worker": worker}
        options.update(changes)
        return checkpoint.read_training_state(self.parent_fd, **options)

    def total(self):
        return sum(amounts["checkpoint_read_bytes"] for _, amounts in self.charges)

    def expected_reads(self):
        return len(self.manifest) + sum(map(len, self.payloads.values())) + len(
            self.payloads[workers.STATE_NAME])

    def assert_failure(self, worker, code="storage_rejected", **changes):
        with self.assertRaises(workers.CheckpointWorkerError) as caught:
            self.read(worker, **changes)
        self.assertEqual(caught.exception.args, ("operation_failed",))
        self.assertEqual(worker.failure_code, code)
        self.assertTrue(worker._failed)
        self.assertFalse(worker._busy)
        self.assertFalse(workers._SERIALIZER_LOCK.locked())
        self.assertNotIn(str(self.parent), repr(caught.exception))
        self.assertNotIn("SYNTHETIC-PRIVATE", repr(caught.exception))
        os.fstat(self.parent_fd)

    def test_import_does_not_load_ml(self):
        forbidden = {"torch", "transformers", "numpy", "safetensors", "tokenizers"}
        self.assertFalse({name.split(".")[0] for name in IMPORTED_MODULES} & forbidden)

    def test_exact_private_state_all_reads_charged_parent_borrowed_and_files_unchanged(self):
        worker = self.worker()
        parent_before = os.fstat(self.parent_fd)
        result = self.read(worker)
        self.assertEqual(result, self.state)
        self.assertIs(type(result), dict)
        self.assertEqual(self.total(), self.expected_reads())
        self.assertEqual(worker._reads, self.expected_reads())
        self.assertEqual(os.fstat(self.parent_fd), parent_before)
        self.assertEqual(self.interrupts, [])
        self.assertFalse(worker._failed)
        for name, data in {**self.payloads, "manifest.sha256": self.manifest}.items():
            self.assertEqual((self.root / name).read_bytes(), data)

    def test_call_cap_chunks_state_reread_without_extra_allowance(self):
        worker = self.worker(rank=1, world_size=2, limits={"maximum_read_call_bytes": 3})
        self.assertEqual(self.read(worker), self.state)
        state_bytes = len(self.payloads[workers.STATE_NAME])
        chunk_count = (state_bytes + 2) // 3
        rereads = self.charges[-chunk_count:]
        self.assertEqual(sum(amount["checkpoint_read_bytes"] for _, amount in rereads), state_bytes)
        self.assertTrue(all(0 < amount["checkpoint_read_bytes"] <= 3 for _, amount in rereads))
        self.assertTrue(all(rank == 1 for rank, _ in self.charges))
        self.assertEqual(worker._reads, self.expected_reads())

    def test_preexisting_ledger_and_repeated_calls_remain_cumulative(self):
        worker = self.worker()
        worker._reserve(0, checkpoint_read_bytes=7)
        self.read(worker)
        self.read(worker)
        self.assertEqual(worker._reads, 7 + 2 * self.expected_reads())
        self.assertEqual(self.total(), worker._reads)

    def test_existing_shared_budget_exhaustion_poisoned_without_retry(self):
        expected = self.expected_reads()
        worker = self.worker(limits={"maximum_read_bytes": expected})
        self.read(worker)
        self.assert_failure(worker, "read_cap_exceeded")
        self.assertEqual(self.total(), expected)
        self.assertEqual(worker._reads, expected)
        self.assertEqual(self.interrupts, [True])
        with self.assertRaises(workers.CheckpointWorkerError) as caught:
            self.read(worker)
        self.assertEqual(caught.exception.args, ("terminal_failure",))
        self.assertEqual(self.total(), expected)
        self.assertEqual(self.interrupts, [True])

    def test_insufficient_state_reread_allowance_does_not_mint_quota(self):
        worker = self.worker(limits={"maximum_read_bytes": self.expected_reads() - 1})
        self.assert_failure(worker)
        self.assertLess(self.total(), self.expected_reads())
        self.assertEqual(self.total(), worker._reads)
        self.assertEqual(self.interrupts, [True])

    def test_shared_reservation_failure_is_nonrefundable(self):
        def reject(rank, **amounts):
            self.reserve(rank, **amounts)
            raise RuntimeError("SYNTHETIC-PRIVATE")

        worker = self.worker(reserve=reject)
        self.assert_failure(worker, "reserve_failed")
        self.assertEqual(self.total(), len(self.manifest))
        self.assertEqual(worker._reads, len(self.manifest))
        self.assertEqual(self.interrupts, [True])

    def test_exact_worker_type_rejects_ducks_and_subclasses_without_calling_them(self):
        class Subclass(workers.CheckpointWorker):
            pass

        for value in (None, object(), Subclass.__new__(Subclass)):
            with self.subTest(kind=type(value).__name__):
                with self.assertRaises(workers.CheckpointWorkerError) as caught:
                    self.read(value)
                self.assertEqual(caught.exception.args, ("invalid_arguments",))
        self.assertEqual(self.charges, [])
        self.assertEqual(self.interrupts, [])

    def test_invalid_names_and_hashes_fail_before_read_or_directory_search(self):
        changes = [{"checkpoint_name": value} for value in (
            "../checkpoint-00000003", str(self.root), "checkpoint-3", self.name + "/model",
            self.name + "\n", "checkpoint-0000000٣", True, None,
        )]
        changes += [{"manifest_sha256": value} for value in (
            "A" * 64, self.pin + "\n", "0" * 63, 0, None,
        )]
        for change in changes:
            with self.subTest(change=change):
                with patch.object(checkpoint.os, "scandir", wraps=os.scandir) as scan:
                    self.assert_failure(self.worker(), "invalid_arguments", **change)
                scan.assert_not_called()
        self.assertEqual(self.charges, [])

    def test_checkpoint_directory_is_exact_no_parent_scan_or_latest_fallback(self):
        with patch.object(checkpoint.os, "scandir", wraps=os.scandir) as scan:
            self.assert_failure(self.worker(), checkpoint_name="checkpoint-00000004")
        scan.assert_not_called()
        self.assertEqual(self.charges, [])

    def test_symlink_checkpoint_is_not_followed(self):
        alias = "checkpoint-00000004"
        (self.parent / alias).symlink_to(self.root)
        self.assert_failure(self.worker(), checkpoint_name=alias)
        self.assertEqual(self.charges, [])

    def test_wrong_manifest_pin_never_parses_state(self):
        with patch.object(checkpoint, "_read_state", wraps=checkpoint._read_state) as read:
            self.assert_failure(self.worker(), manifest_sha256="0" * 64)
        read.assert_not_called()
        self.assertEqual(self.total(), len(self.manifest))

    def test_corrupt_state_hash_never_parses_state(self):
        self.write(workers.STATE_NAME, b"SYNTHETIC-PRIVATE-CORRUPT\n")
        with patch.object(checkpoint, "_read_state", wraps=checkpoint._read_state) as read:
            self.assert_failure(self.worker())
        read.assert_not_called()

    def test_absent_state_does_not_substitute_other_payload(self):
        del self.payloads[workers.STATE_NAME]
        (self.root / workers.STATE_NAME).unlink()
        self.pin_files()
        self.assert_failure(self.worker())
        self.assertEqual(self.total(), len(self.manifest) + sum(map(len, self.payloads.values())))

    def test_noncanonical_duplicate_nonfinite_nonobject_and_nonascii_json_rejected(self):
        invalid = [b"", b"{}", b"{}\n\n", b" {}\n", b'{"a": 1}\n',
                   b'{"z":1,"a":2}\n', b'{"a":1,"a":2}\n',
                   b'{"a":{"b":1,"b":2}}\n', b'{"a":NaN}\n',
                   b'{"a":Infinity}\n', b'{"a":-Infinity}\n', b'{"a":1e999}\n',
                   b'{"a":1e0}\n', b'{"a":-0}\n', b'{"a":"\\u0061"}\n',
                   b'[]\n', b'null\n', b'true\n', b'"private"\n',
                   '{"a":"é"}\n'.encode(), b'\xef\xbb\xbf{}\n']
        for payload in invalid:
            with self.subTest(case=invalid.index(payload)):
                self.payloads[workers.STATE_NAME] = payload
                self.pin_files()
                self.assert_failure(self.worker())

    def test_ascii_escaped_unicode_from_frozen_writer_is_canonical(self):
        self.state = {"private": "বাংলা", "step": 3}
        self.payloads[workers.STATE_NAME] = self.canonical(self.state)
        self.pin_files()
        self.assertEqual(self.read(self.worker()), self.state)

    def test_state_size_exact_boundary_and_oversize_rejected_before_reread(self):
        state = {"p": "x" * (checkpoint.MAX_STATE_BYTES - len(self.canonical({"p": ""})))}
        self.payloads[workers.STATE_NAME] = self.canonical(state)
        self.assertEqual(len(self.payloads[workers.STATE_NAME]), checkpoint.MAX_STATE_BYTES)
        self.pin_files()
        self.assertEqual(self.read(self.worker()), state)
        self.charges.clear()
        state["p"] += "x"
        self.payloads[workers.STATE_NAME] = self.canonical(state)
        self.pin_files()
        self.assert_failure(self.worker())
        self.assertEqual(self.total(), len(self.manifest) + sum(map(len, self.payloads.values())))

    def test_parent_mutation_during_reservation_rejects_after_verified_read(self):
        def mutate(rank, **amounts):
            self.reserve(rank, **amounts)
            if len(self.charges) == 1:
                (self.parent / "SYNTHETIC-sibling").mkdir(mode=0o700)

        self.assert_failure(self.worker(reserve=mutate), "descriptor_changed")
        self.assertEqual(self.interrupts, [True])

    def test_session_construction_failure_closes_duplicate_exactly_once(self):
        duplicate, original_dup, original_close = [], os.dup, os.close

        def observe_dup(fd):
            result = original_dup(fd)
            duplicate.append(result)
            return result

        with patch.object(checkpoint.os, "dup", side_effect=observe_dup):
            with patch.object(checkpoint.os, "close", wraps=original_close) as close:
                with patch.object(checkpoint, "CheckpointReadSession", side_effect=ValueError):
                    self.assert_failure(self.worker())
        self.assertEqual(len(duplicate), 1)
        self.assertEqual(sum(call.args == (duplicate[0],) for call in close.call_args_list), 1)
        with self.assertRaises(OSError):
            os.fstat(duplicate[0])
        os.fstat(self.parent_fd)

    def test_success_and_parse_failure_close_both_owned_descriptors(self):
        for malformed in (False, True):
            with self.subTest(malformed=malformed):
                if malformed:
                    self.payloads[workers.STATE_NAME] = b"SYNTHETIC-PRIVATE"
                    self.pin_files()
                duplicate, opened, original_dup, original_open = [], [], os.dup, os.open

                def observe_dup(fd, original_dup=original_dup, duplicate=duplicate):
                    result = original_dup(fd)
                    duplicate.append(result)
                    return result

                def observe_open(*args, original_open=original_open, opened=opened, **kwargs):
                    result = original_open(*args, **kwargs)
                    opened.append(result)
                    return result

                with (patch.object(checkpoint.os, "dup", side_effect=observe_dup),
                      patch.object(checkpoint.os, "open", side_effect=observe_open)):
                    if malformed:
                        self.assert_failure(self.worker())
                    else:
                        self.read(self.worker())
                self.assertEqual(len(duplicate), 1)
                for fd in duplicate + opened:
                    with self.assertRaises(OSError):
                        os.fstat(fd)
                os.fstat(self.parent_fd)

    def test_session_close_error_is_terminal_and_never_retried(self):
        original_close = checkpoint.CheckpointReadSession.close
        sessions = []

        def close_then_fail(session):
            sessions.append(session)
            original_close(session)
            raise RuntimeError("SYNTHETIC-PRIVATE-CLOSE")

        with patch.object(checkpoint.CheckpointReadSession, "close", close_then_fail):
            self.assert_failure(self.worker())
        self.assertEqual(len(sessions), 1)
        self.assertTrue(sessions[0].closed)
        self.assertEqual(self.interrupts, [True])

    def test_callback_reentry_poisoned_and_never_uses_new_allowance(self):
        worker = self.worker()

        def reenter(rank, **amounts):
            self.reserve(rank, **amounts)
            with self.assertRaises(workers.CheckpointWorkerError):
                self.read(worker)

        worker._reservation = reenter
        self.assert_failure(worker, "reentry")
        self.assertEqual(self.total(), len(self.manifest))
        self.assertEqual(worker._reads, len(self.manifest))
        self.assertEqual(self.interrupts, [True])


if __name__ == "__main__":
    unittest.main()
