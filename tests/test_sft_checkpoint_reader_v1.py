"""Only tiny synthetic files and stdlib; never import the ML package entrypoint."""

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
PACKAGE = "_lumi_synthetic_checkpoint_reader_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(
    f"{PACKAGE}.posttraining_sft_checkpoint_reader_v1",
    ROOT / "src/lumi/posttraining_sft_checkpoint_reader_v1.py",
)
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)


class CheckpointReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lumi-SYNTHETIC-checkpoint-read-v1-")
        self.root = Path(self.temporary.name)
        self.root.chmod(0o700)
        self.payloads = {"config.json": b"{}", "model/.metadata": b"NOT-PICKLE",
                         "model/__0_0.distcp": b"abcdef"}
        for name, data in self.payloads.items():
            self.write(name, data)
        self.pin_manifest()
        self.handles, self.descriptors, self.reservations = [], set(), []

    def tearDown(self):
        for handle in self.handles:
            try:
                handle.close()
            except checkpoint.CheckpointReadError:
                pass
        for fd in self.descriptors:
            os.close(fd)
        self.temporary.cleanup()

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o600)
        return path

    def pin_manifest(self, data=None):
        if data is None:
            data = "".join(f"{hashlib.sha256(value).hexdigest()}  {name}\n"
                           for name, value in self.payloads.items()).encode("ascii")
        self.manifest_bytes = data
        self.write("manifest.sha256", data)
        self.pin = hashlib.sha256(data).hexdigest()

    def reserve(self, rank, **amounts):
        self.reservations.append((rank, amounts))

    def session(self, **overrides):
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.descriptors.add(fd)
        options = {"rank": 0, "reserve": self.reserve, "expected_manifest_sha256": self.pin,
                   "maximum_file_bytes": 1024, "maximum_read_bytes": 4096}
        options.update(overrides)
        result = checkpoint.CheckpointReadSession(fd, **options)
        self.descriptors.remove(fd)
        self.handles.append(result)
        return result

    def assert_error(self, code, call, *args):
        with self.assertRaises(checkpoint.CheckpointReadError) as caught:
            call(*args)
        self.assertEqual(caught.exception.args, (code,))
        self.assertEqual(caught.exception.code, code)
        self.assertNotIn(str(self.root), repr(caught.exception))

    def total(self):
        return sum(amounts["checkpoint_read_bytes"] for _, amounts in self.reservations)

    def test_verify_and_repeated_slices_count_every_read(self):
        reader = self.session(rank=1)
        receipt = reader.verify()
        expected = len(self.manifest_bytes) + sum(map(len, self.payloads.values()))
        self.assertEqual(receipt, {"status": "verified", "payload_files": 3,
                                  "payload_bytes": 18, "reserved_read_bytes": expected,
                                  "deserialization_performed": False, "production_eligible": False})
        self.assertEqual(reader.read_file("model/__0_0.distcp", 1, 3), b"bcd")
        self.assertEqual(reader.read_file("model/__0_0.distcp", 1, 3), b"bcd")
        self.assertEqual(self.total(), expected + 6)
        self.assertTrue(all(rank == 1 for rank, _ in self.reservations))
        self.assertEqual(repr(reader), "<CheckpointReadSession private>")

    def test_manifest_pin_rejects_before_any_payload_read(self):
        reader = self.session(expected_manifest_sha256="0" * 64)
        with patch.object(checkpoint.os, "pread", wraps=os.pread) as pread:
            self.assert_error("manifest_mismatch", reader.verify)
        self.assertEqual(pread.call_count, 1)
        self.assertEqual(self.total(), len(self.manifest_bytes))
        self.assert_error("terminal_failure", reader.verify)

    def test_corrupt_payload_digest_fails_terminally(self):
        self.write("config.json", b"[]")
        reader = self.session()
        self.assert_error("checksum_mismatch", reader.verify)
        self.assert_error("terminal_failure", reader.read_file, "config.json", 0, 1)

    def test_manifest_parser_rejects_traversal_duplicates_and_noncanonical_forms(self):
        digest = "a" * 64
        base = f"{digest}  config.json\n"
        invalid = [b"", base.encode(), (base * 2).encode(),
                   (base + f"{digest}  ../escape\n").encode(),
                   (base + f"{digest}  /absolute\n").encode(),
                   (base + f"{digest}  model/../escape\n").encode(),
                   (base + f"{digest}  manifest.sha256\n").encode(),
                   (base + f"{digest}  model/manifest.sha256\n").encode(),
                   (base + f"{digest}  a//b\n").encode(),
                   (base + f"{digest}  a\\b\n").encode(),
                   (base + f"{digest}  a b\n").encode(),
                   (base + f"{digest}  a\r\n").encode(),
                   (base + f"{digest.upper()}  b\n").encode(),
                   (base + f"{digest}  a\n\n").encode(),
                   (base + f"{digest}  a").encode(),
                   (base + f"{digest}  a/./b\n").encode(),
                   (base + f"{digest}  ").encode() + b"\xff\n"]
        for data in invalid:
            with self.subTest(case=invalid.index(data)):
                self.pin_manifest(data)
                reader = self.session()
                before = self.total()
                self.assert_error("invalid_manifest", reader.verify)
                self.assertEqual(self.total() - before, len(data))

    def test_unsorted_manifest_and_empty_payload_and_empty_directory_are_valid(self):
        self.payloads["a.zz"] = b""
        self.write("a.zz", b"")
        (self.root / "empty").mkdir(mode=0o700)
        self.pin_manifest()
        reader = self.session()
        self.assertEqual(reader.verify()["payload_files"], 4)
        before = self.total()
        with patch.object(checkpoint.os, "pread", wraps=os.pread) as pread:
            self.assertEqual(reader.read_file("a.zz", 0, 0), b"")
            self.assertEqual(reader.read_file("config.json", 2, 0), b"")
        pread.assert_not_called()
        self.assertEqual(before, self.total())

    def test_missing_declared_file_rejects_before_payload_reads(self):
        (self.root / "config.json").unlink()
        self.assert_error("inventory_mismatch", self.session().verify)
        self.assertEqual(self.total(), len(self.manifest_bytes))

    def test_extra_undeclared_file_is_not_read(self):
        self.write("extra", b"PRIVATE-SYNTHETIC")
        self.assert_error("inventory_mismatch", self.session().verify)
        self.assertEqual(self.total(), len(self.manifest_bytes))

    def test_missing_manifest_has_zero_content_reads(self):
        (self.root / "manifest.sha256").unlink()
        self.assert_error("inventory_mismatch", self.session().verify)
        self.assertEqual(self.total(), 0)

    def test_symlink_is_not_followed(self):
        (self.root / "alias").symlink_to(self.root / "config.json")
        self.assert_error("inventory_mismatch", self.session().verify)
        self.assertEqual(self.total(), 0)

    def test_hardlinks_reject_without_content_reads(self):
        os.link(self.root / "config.json", self.root / "alias")
        self.assert_error("inventory_mismatch", self.session().verify)
        self.assertEqual(self.total(), 0)

    def test_fifo_rejects_without_opening_or_blocking(self):
        os.mkfifo(self.root / "fifo", mode=0o600)
        self.assert_error("inventory_mismatch", self.session().verify)
        self.assertEqual(self.total(), 0)

    def test_nonprivate_file_rejects(self):
        (self.root / "config.json").chmod(0o644)
        self.assert_error("inventory_mismatch", self.session().verify)
        self.assertEqual(self.total(), 0)

    def test_nonprivate_subdirectory_rejects(self):
        (self.root / "model").chmod(0o755)
        self.assert_error("inventory_mismatch", self.session().verify)
        self.assertEqual(self.total(), 0)

    def test_file_cap_checked_before_content_reads(self):
        self.assert_error("structure_cap_exceeded", self.session(maximum_file_bytes=5).verify)
        self.assertEqual(self.total(), 0)

    def test_manifest_cap_checked_before_content_reads(self):
        with patch.object(checkpoint, "MAX_MANIFEST_BYTES", 4):
            self.assert_error("structure_cap_exceeded", self.session().verify)
        self.assertEqual(self.total(), 0)

    def test_tree_cap_checked_before_content_reads(self):
        with patch.object(checkpoint, "MAX_TREE_ENTRIES", 1):
            self.assert_error("structure_cap_exceeded", self.session().verify)
        self.assertEqual(self.total(), 0)

    def test_payload_count_cap(self):
        with patch.object(checkpoint, "MAX_PAYLOAD_FILES", 2):
            self.assert_error("structure_cap_exceeded", self.session().verify)
        self.assertEqual(self.total(), 0)

    def test_manifest_path_and_count_caps(self):
        digest = "a" * 64
        for value in ["a/" * 8 + "b", "b" * 129, "a" * 128 + "/" + "b" * 128]:
            with self.subTest(length=len(value)):
                data = f"{digest}  config.json\n{digest}  {value}\n".encode()
                self.assert_error("invalid_manifest", checkpoint._parse_manifest, data)
        with patch.object(checkpoint, "MAX_PAYLOAD_FILES", 2):
            self.assert_error("structure_cap_exceeded", checkpoint._parse_manifest,
                              self.manifest_bytes)

    def test_manifest_read_budget_is_prepaid(self):
        reader = self.session(maximum_read_bytes=len(self.manifest_bytes) - 1)
        with patch.object(checkpoint.os, "pread", wraps=os.pread) as pread:
            self.assert_error("io_failed", reader.verify)
        pread.assert_not_called()
        self.assertEqual(self.total(), 0)

    def test_total_hash_budget_checked_before_payloads(self):
        reader = self.session(maximum_read_bytes=len(self.manifest_bytes))
        self.assert_error("read_cap_exceeded", reader.verify)
        self.assertEqual(self.total(), len(self.manifest_bytes))

    def test_reread_budget_fails_before_next_pread(self):
        cap = len(self.manifest_bytes) + sum(map(len, self.payloads.values())) + 2
        reader = self.session(maximum_read_bytes=cap)
        reader.verify()
        self.assertEqual(reader.read_file("config.json", 0, 2), b"{}")
        with patch.object(checkpoint.os, "pread", wraps=os.pread) as pread:
            self.assert_error("io_failed", reader.read_file, "config.json", 0, 1)
        pread.assert_not_called()
        self.assertEqual(self.total(), cap)

    def test_shared_budget_across_sessions_and_ranks(self):
        # Tiny real Supervisor, loaded by filename without any ML package import.
        spec = importlib.util.spec_from_file_location(
            "synthetic_checkpoint_supervisor", ROOT / "src/lumi/posttraining_sft_supervisor_v1.py")
        supervision = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(supervision)
        from tests.test_sft_metered_io_v1 import controller

        template = controller()
        verification_bytes = len(self.manifest_bytes) + 18
        caps = dict(template._caps)
        caps["maximum_checkpoint_read_bytes"] = verification_bytes * 2 + 2
        shared = supervision.Supervisor(caps, template._counts, world_size=2, clock_ns=lambda: 0)
        shared.begin_invocation("reference", 0, 2, False)
        reader = self.session(rank=0, reserve=shared.reserve)
        reader2 = self.session(rank=1, reserve=shared.reserve)
        reader.verify()
        reader2.verify()
        self.assertEqual(reader.read_file("config.json", 0, 2), b"{}")
        with patch.object(checkpoint.os, "pread", wraps=os.pread) as pread:
            self.assert_error("io_failed", reader2.read_file, "config.json", 0, 1)
        pread.assert_not_called()
        self.assertEqual(shared._used["checkpoint_read_bytes"], verification_bytes * 2 + 2)
        # The shared lifecycle is now failed, not just rank one's local reader.
        self.assert_error("io_failed", reader.read_file, "config.json", 0, 1)

    def test_hashing_streams_bounded_chunks(self):
        with patch.object(checkpoint, "MAX_OPERATION_BYTES", 4):
            self.session().verify()
        self.assertTrue(all(n["checkpoint_read_bytes"] <= 4 for _, n in self.reservations[1:]))
        self.assertEqual(self.total(), len(self.manifest_bytes) + 18)

    def test_shared_reserve_runs_before_actual_reads(self):
        actual = os.pread
        def observed(fd, size, offset):
            self.assertEqual(self.reservations[-1][1], {"checkpoint_read_bytes": size})
            return actual(fd, size, offset)
        with patch.object(checkpoint.os, "pread", side_effect=observed):
            self.session().verify()

    def test_callback_failure_and_short_read_keep_reservations(self):
        def failing(rank, **amounts):
            self.reserve(rank, **amounts)
            raise RuntimeError("PRIVATE-SYNTHETIC-ERROR")
        reader = self.session(reserve=failing)
        self.assert_error("io_failed", reader.verify)
        self.assertEqual(self.total(), len(self.manifest_bytes))
        reader2 = self.session()
        with patch.object(checkpoint.os, "pread", return_value=b"x"):
            self.assert_error("io_failed", reader2.verify)
        self.assertEqual(self.total(), 2 * len(self.manifest_bytes))

    def test_non_none_callback_result_rejects_before_read(self):
        reader = self.session(reserve=lambda *args, **kwargs: True)
        with patch.object(checkpoint.os, "pread", wraps=os.pread) as pread:
            self.assert_error("io_failed", reader.verify)
        pread.assert_not_called()

    def test_callback_swallowing_reentry_still_poisoned(self):
        def reenter(rank, **amounts):
            self.reserve(rank, **amounts)
            self.assert_error("reentry", reader.verify)
        reader = self.session(reserve=reenter)
        with patch.object(checkpoint.os, "pread", wraps=os.pread) as pread:
            self.assert_error("io_failed", reader.verify)
        pread.assert_not_called()
        self.assert_error("terminal_failure", reader.verify)

    def test_root_mutation_during_callback_fails_before_read(self):
        def mutate(rank, **amounts):
            self.reserve(rank, **amounts)
            self.write("unexpected", b"tiny")
        reader = self.session(reserve=mutate)
        with patch.object(checkpoint.os, "pread", wraps=os.pread) as pread:
            self.assert_error("io_failed", reader.verify)
        pread.assert_not_called()

    def test_file_mutation_after_verification_is_rejected(self):
        reader = self.session()
        reader.verify()
        self.write("config.json", b"[]")
        with patch.object(checkpoint.os, "pread", wraps=os.pread) as pread:
            self.assert_error("descriptor_changed", reader.read_file, "config.json", 0, 2)
        pread.assert_not_called()

    def test_parent_mutation_after_verification_is_rejected(self):
        reader = self.session()
        reader.verify()
        self.write("model/extra", b"tiny")
        with patch.object(checkpoint.os, "pread", wraps=os.pread) as pread:
            self.assert_error("descriptor_changed", reader.read_file, "model/.metadata", 0, 1)
        pread.assert_not_called()

    def test_file_change_inside_callback_is_rejected(self):
        reader = self.session()
        reader.verify()
        def mutate(rank, **amounts):
            self.reserve(rank, **amounts)
            self.write("config.json", b"[]")
        reader._reserve = mutate
        with patch.object(checkpoint.os, "pread", wraps=os.pread) as pread:
            self.assert_error("io_failed", reader.read_file, "config.json", 0, 2)
        pread.assert_not_called()

    def test_parent_change_inside_callback_returns_no_payload(self):
        reader = self.session()
        reader.verify()
        def mutate(rank, **amounts):
            self.reserve(rank, **amounts)
            self.write("model/extra", b"tiny")
        reader._reserve = mutate
        self.assert_error("descriptor_changed", reader.read_file, "model/.metadata", 0, 1)

    def test_final_inventory_catches_change_to_already_hashed_file(self):
        calls = 0
        def mutate(rank, **amounts):
            nonlocal calls
            calls += 1
            self.reserve(rank, **amounts)
            if calls == 3:  # Manifest and config were already read; hashing metadata now.
                self.write("config.json", b"[]")
        self.assert_error("descriptor_changed", self.session(reserve=mutate).verify)

    def test_failed_content_syscall_is_not_retried(self):
        reader = self.session()
        with patch.object(checkpoint.os, "pread", side_effect=OSError("SYNTHETIC")) as pread:
            self.assert_error("io_failed", reader.verify)
        self.assertEqual(pread.call_count, 1)
        self.assertEqual(self.total(), len(self.manifest_bytes))
        self.assert_error("terminal_failure", reader.verify)

    def test_internal_opened_fds_are_closed_on_success_and_failure(self):
        actual_open, actual_close = os.open, os.close
        held = set()
        def opening(*args, **kwargs):
            fd = actual_open(*args, **kwargs)
            held.add(fd)
            return fd
        def closing(fd):
            held.remove(fd)
            return actual_close(fd)
        reader = self.session()
        with (patch.object(checkpoint.os, "open", side_effect=opening),
              patch.object(checkpoint.os, "close", side_effect=closing)):
            reader.verify()
            self.assertEqual(held, set())
            with patch.object(checkpoint.os, "pread", side_effect=OSError("SYNTHETIC")):
                self.assert_error("io_failed", reader.read_file, "model/.metadata", 0, 1)
            self.assertEqual(held, set())

    def test_root_cloexec_change_is_rejected_before_read(self):
        reader = self.session()
        os.set_inheritable(reader._fd, True)
        with patch.object(checkpoint.os, "pread", wraps=os.pread) as pread:
            self.assert_error("descriptor_changed", reader.verify)
        pread.assert_not_called()

    def test_replaced_file_is_not_opened_for_content(self):
        reader = self.session()
        reader.verify()
        (self.root / "config.json").unlink()
        os.mkfifo(self.root / "config.json", mode=0o600)
        with patch.object(checkpoint.os, "open", wraps=os.open) as opening:
            self.assert_error("descriptor_changed", reader.read_file, "config.json", 0, 1)
        opening.assert_not_called()

    def test_read_requires_verification_and_verify_cannot_repeat(self):
        self.assert_error("not_verified", self.session().read_file, "config.json", 0, 1)
        reader = self.session()
        reader.verify()
        before = self.total()
        self.assert_error("already_verified", reader.verify)
        self.assertEqual(self.total(), before)

    def test_invalid_slice_arguments_are_terminal(self):
        for args in [("manifest.sha256", 0, 1), ("missing", 0, 1), ("../config.json", 0, 1),
                     ([], 0, 1), ("config.json", True, 1), ("config.json", 0, False),
                     ("config.json", -1, 1), ("config.json", 0, -1),
                     ("config.json", 0, 1_048_577), ("config.json", 2, 1)]:
            with self.subTest(case=str(args)):
                reader = self.session()
                reader.verify()
                self.assert_error("invalid_arguments", reader.read_file, *args)
                self.assert_error("terminal_failure", reader.read_file, "config.json", 0, 1)

    def test_constructor_rejection_retains_callers_fd(self):
        for overrides in [{"rank": True}, {"rank": 12}, {"reserve": None},
                          {"maximum_file_bytes": False}, {"maximum_read_bytes": 0},
                          {"expected_manifest_sha256": "A" * 64}]:
            with self.subTest(field=next(iter(overrides))):
                before = len(self.descriptors)
                self.assert_error("invalid_arguments",
                                  lambda values=overrides: self.session(**values))
                self.assertEqual(len(self.descriptors), before + 1)
                for fd in self.descriptors:
                    os.fstat(fd)

    def test_nonprivate_root_rejects_without_transferring_fd(self):
        self.root.chmod(0o755)
        self.assert_error("invalid_descriptor", self.session)
        self.assertEqual(len(self.descriptors), 1)
        self.root.chmod(0o700)

    def test_constructor_rejects_non_directory_fd(self):
        fd = os.open(self.root / "config.json", os.O_RDONLY)
        self.descriptors.add(fd)
        self.assert_error("invalid_descriptor", lambda: checkpoint.CheckpointReadSession(
            fd, rank=0, reserve=self.reserve, expected_manifest_sha256=self.pin,
            maximum_file_bytes=1024, maximum_read_bytes=4096))
        os.fstat(fd)

    def test_closed_and_owner_mismatch_are_terminal(self):
        reader = self.session()
        reader.close()
        reader.close()
        self.assertTrue(reader.closed)
        self.assert_error("closed", reader.verify)
        for attribute in ("getpid", "get_ident"):
            reader = self.session()
            module = checkpoint.os if attribute == "getpid" else checkpoint.threading
            with patch.object(module, attribute, return_value=-1):
                self.assert_error("owner_mismatch", reader.verify)
            self.assert_error("terminal_failure", reader.verify)

    def test_reinitialization_does_not_overwrite_ownership(self):
        reader = self.session()
        original = reader._fd
        self.assert_error("invalid_arguments", lambda: reader.__init__(
            original, rank=0, reserve=self.reserve, expected_manifest_sha256=self.pin,
            maximum_file_bytes=1024, maximum_read_bytes=4096))
        self.assertEqual(reader._fd, original)
        self.assert_error("terminal_failure", reader.verify)

    def test_close_never_retries_an_ambiguous_failure(self):
        reader = self.session()
        fd = reader._fd
        failure = OSError("PRIVATE-SYNTHETIC")
        with patch.object(checkpoint.os, "close", side_effect=failure) as close:
            self.assert_error("close_failed", reader.close)
            reader.close()
        close.assert_called_once_with(fd)
        self.descriptors.add(fd)  # Mock did not close it; fixture still owns actual FD.

    def test_reused_fd_for_different_directory_is_not_closed(self):
        reader = self.session()
        fd = reader._fd
        replacement = os.open(self.root / "model", os.O_RDONLY | os.O_DIRECTORY)
        self.descriptors.add(replacement)
        os.dup2(replacement, fd, inheritable=False)
        self.descriptors.add(fd)
        self.assert_error("descriptor_changed", reader.close)
        os.fstat(fd)
        self.assertTrue(reader.closed)

    def test_unknown_error_code_is_redacted(self):
        self.assertEqual(checkpoint.CheckpointReadError("PRIVATE-SYNTHETIC").args,
                         ("internal_error",))


if __name__ == "__main__":
    unittest.main()
