"""Tiny synthetic directory publication only; no ML, model/data or network access."""

from __future__ import annotations

import ctypes
import errno
import importlib.util
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_checkpoint_publication_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package


def load_module(name):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE}.{name}", ROOT / "src/lumi" / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


publication = load_module("posttraining_sft_checkpoint_publication_v1")
saver = load_module("posttraining_sft_checkpoint_saver_v1")
supervision = load_module("posttraining_sft_supervisor_v1")
metered = sys.modules[f"{PACKAGE}.posttraining_sft_metered_io_v1"]


def controller(reads):
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
        "maximum_per_gpu_memory_bytes": 2048, "maximum_private_write_bytes": 4096,
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


class PublicationFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lumi-SYNTHETIC-publication-v1-")
        self.root = Path(self.temporary.name)
        self.root.chmod(0o700)
        self.checkpoint_name = "checkpoint-00000001"
        self.staging_name = f".{self.checkpoint_name}.{'a' * 32}.tmp"
        self.staging = self.root / self.staging_name
        self.destination = self.root / self.checkpoint_name
        self.staging.mkdir(mode=0o700)
        (self.staging / "model").mkdir(mode=0o700)
        self.payloads = {"config.json": b"{}", "model/.metadata": b"NOT-PICKLE",
                         "model/__0_0.distcp": b"abc"}
        expected = {}
        for name, data in self.payloads.items():
            fd = os.open(self.staging / name,
                         os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
            stream = saver.CheckpointWriteStream(
                fd, rank=0, reserve=lambda rank, **amounts: None,
                maximum_file_bytes=1024, maximum_write_call_bytes=1024,
                maximum_sync_operations=1,
            )
            stream.write(data)
            stream.close()
            expected[name] = stream.receipt()
        fd = os.open(self.staging, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        finalizer = saver.CheckpointStagingFinalizer(
            fd, rank=0, reserve=lambda rank, **amounts: None, expected_files=expected,
            maximum_file_bytes=1024, maximum_read_bytes=4096,
            maximum_manifest_write_bytes=4096, maximum_sync_operations=32,
        )
        self.staging_receipt = finalizer.finish()
        finalizer.close()
        self.manifest_bytes = (self.staging / "manifest.sha256").stat().st_size
        self.one_pass = self.manifest_bytes + sum(map(len, self.payloads.values()))
        self.handles, self.descriptors, self.reservations = [], set(), []

    def tearDown(self):
        for handle in reversed(self.handles):
            try:
                handle.close()
            except publication.CheckpointPublicationError:
                pass
        for fd in self.descriptors:
            os.close(fd)
        self.temporary.cleanup()

    def reserve(self, rank, **amounts):
        self.reservations.append((rank, amounts))

    def total(self):
        return sum(amounts.get("checkpoint_read_bytes", 0)
                   for _, amounts in self.reservations)

    def publisher(self, **overrides):
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.descriptors.add(fd)
        options = {"staging_name": self.staging_name, "checkpoint_name": self.checkpoint_name,
                   "staging_receipt": self.staging_receipt, "rank": 0, "reserve": self.reserve,
                   "maximum_file_bytes": 1024, "maximum_read_bytes": self.one_pass * 2}
        options.update(overrides)
        handle = publication.CheckpointPublisher(fd, **options)
        self.descriptors.remove(fd)
        self.handles.append(handle)
        return handle, fd

    def deny(self, call, *args):
        with self.assertRaises(publication.CheckpointPublicationError) as caught:
            call(*args)
        error = caught.exception
        self.assertEqual(error.args, (error.code,))
        self.assertIn(error.code, publication.ERROR_CODES)
        self.assertNotIn(str(self.root), repr(error))
        self.assertNotIn("SYNTHETIC-SECRET", repr(error))
        return error

    def assert_unpublished(self, handle):
        self.assertTrue(self.staging.is_dir())
        self.assertFalse(self.destination.exists())
        self.assertFalse(handle.state()["publication_verified"])


class CheckpointPublicationTests(PublicationFixture):
    def test_stream_finalizer_publication_round_trip(self):
        previous = self.root / "checkpoint-00000000"
        previous.mkdir(mode=0o700)
        marker = previous / "retained.bin"
        marker.write_bytes(b"PRIOR-EVIDENCE")
        marker.chmod(0o600)
        previous_inode = previous.stat().st_ino
        source_inode = self.staging.stat().st_ino
        handle, fd = self.publisher(rank=1)
        real_fsync = os.fsync
        with patch.object(publication.os, "fsync", wraps=real_fsync) as sync:
            receipt = handle.publish()
        self.assertEqual(sync.call_args_list, [unittest.mock.call(fd)])
        self.assertEqual(receipt, {
            "status": "published_checkpoint",
            "manifest_sha256": self.staging_receipt["manifest_sha256"],
            "payload_files": len(self.payloads),
            "payload_bytes": sum(map(len, self.payloads.values())),
            "checkpoint_read_bytes": 2 * self.one_pass, "private_write_bytes": 0,
            "rename_attempts": 1, "parent_sync_attempts": 1,
            "published": True, "production_eligible": False,
        })
        self.assertFalse(self.staging.exists())
        self.assertEqual(self.destination.stat().st_ino, source_inode)
        for name, data in self.payloads.items():
            self.assertEqual((self.destination / name).read_bytes(), data)
        self.assertEqual(marker.read_bytes(), b"PRIOR-EVIDENCE")
        self.assertEqual(previous.stat().st_ino, previous_inode)
        self.assertEqual(self.total(), 2 * self.one_pass)
        self.assertTrue(all(rank == 1 and set(amounts) == {"checkpoint_read_bytes"}
                            for rank, amounts in self.reservations))
        self.assertEqual(handle.state(), {
            "phase": "complete", "failed": False, "rename_attempts": 1,
            "parent_sync_attempts": 1, "checkpoint_read_bytes": 2 * self.one_pass,
            "publication_verified": True, "closed": False,
        })

    def test_success_never_uses_plain_rename_replace_or_payload_write(self):
        handle, _ = self.publisher()
        with patch.object(publication.os, "rename") as rename, \
                patch.object(publication.os, "replace") as replace, \
                patch.object(publication.os, "pwrite") as write:
            handle.publish()
        rename.assert_not_called()
        replace.assert_not_called()
        write.assert_not_called()

    def test_publication_debits_the_same_real_supervisor(self):
        ledger = controller(self.one_pass * 2)
        handle, _ = self.publisher(rank=1, reserve=ledger.reserve)
        receipt = handle.publish()
        summary = ledger.summary()
        used = summary["resources_reserved"]
        self.assertEqual(used["checkpoint_read_bytes"], self.one_pass * 2)
        self.assertEqual(used["private_write_bytes"], 0)
        self.assertEqual(used["log_bytes"], 0)
        self.assertTrue(receipt["published"])
        self.assertFalse(summary["production_eligible"])

    def test_failed_supervisor_blocks_other_metered_reads_after_visible_rename(self):
        ledger = controller(self.one_pass)
        handle, _ = self.publisher(reserve=ledger.reserve)
        self.deny(handle.publish)
        summary = ledger.summary()
        self.assertEqual(summary["failure_code"], "resource_exhausted")
        self.assertEqual(summary["resources_reserved"]["checkpoint_read_bytes"], self.one_pass)
        self.assertTrue(self.destination.is_dir())
        self.assertFalse(handle.state()["publication_verified"])
        fd = os.open(self.destination / "config.json", os.O_RDONLY | os.O_CLOEXEC)
        reader = metered.MeteredReader(fd, rank=1, resource="checkpoint_read_bytes",
                                       reserve=ledger.reserve)
        try:
            with patch.object(publication.os, "pread") as read:
                with self.assertRaises(metered.MeteredIOError):
                    reader.read(1)
            read.assert_not_called()
        finally:
            reader.close()

    def test_publication_is_one_shot(self):
        handle, _ = self.publisher()
        handle.publish()
        with patch.object(publication, "_native_rename_noreplace") as rename, \
                patch.object(publication.os, "pread") as read:
            self.deny(handle.publish)
        rename.assert_not_called()
        read.assert_not_called()

    def test_exact_read_cap_is_preflighted_before_first_content_read(self):
        handle, _ = self.publisher(maximum_read_bytes=self.one_pass * 2 - 1)
        with patch.object(publication.os, "pread") as read, \
                patch.object(publication, "_native_rename_noreplace") as rename:
            self.deny(handle.publish)
        read.assert_not_called()
        rename.assert_not_called()
        self.assertEqual(self.total(), 0)
        self.assert_unpublished(handle)

    def test_existing_destination_directory_is_never_overwritten(self):
        self.destination.mkdir(mode=0o700)
        inode = self.destination.stat().st_ino
        handle, _ = self.publisher()
        with patch.object(publication.os, "pread") as read, \
                patch.object(publication, "_native_rename_noreplace") as rename:
            self.deny(handle.publish)
        read.assert_not_called()
        rename.assert_not_called()
        self.assertEqual(self.destination.stat().st_ino, inode)
        self.assertTrue(self.staging.is_dir())

    def test_racing_empty_directory_cannot_be_replaced(self):
        handle, _ = self.publisher()
        real = publication._native_rename_noreplace

        def race(*args):
            self.destination.mkdir(mode=0o700)
            return real(*args)

        with patch.object(publication, "_native_rename_noreplace", side_effect=race) as rename:
            self.deny(handle.publish)
        rename.assert_called_once()
        self.assertTrue(self.destination.is_dir())
        self.assertEqual(list(self.destination.iterdir()), [])
        self.assertTrue(self.staging.is_dir())
        self.assertEqual(handle.state()["rename_attempts"], 1)

    def test_racing_file_cannot_be_replaced(self):
        handle, _ = self.publisher()
        real = publication._native_rename_noreplace

        def race(*args):
            self.destination.write_bytes(b"RETAINED-RACING-FILE")
            self.destination.chmod(0o600)
            return real(*args)

        with patch.object(publication, "_native_rename_noreplace", side_effect=race):
            self.deny(handle.publish)
        self.assertEqual(self.destination.read_bytes(), b"RETAINED-RACING-FILE")
        self.assertTrue(self.staging.is_dir())

    def test_racing_symlink_cannot_be_replaced_or_followed(self):
        handle, _ = self.publisher()
        real = publication._native_rename_noreplace

        def race(*args):
            self.destination.symlink_to("nonexistent-SYNTHETIC-target")
            return real(*args)

        with patch.object(publication, "_native_rename_noreplace", side_effect=race):
            self.deny(handle.publish)
        self.assertTrue(self.destination.is_symlink())
        self.assertEqual(os.readlink(self.destination), "nonexistent-SYNTHETIC-target")
        self.assertTrue(self.staging.is_dir())

    def test_syscall_failure_is_never_retried(self):
        handle, _ = self.publisher()
        with patch.object(publication, "_native_rename_noreplace",
                          side_effect=OSError("SYNTHETIC-SECRET")) as rename:
            self.deny(handle.publish)
            self.deny(handle.publish)
        rename.assert_called_once()
        self.assertEqual(handle.state()["phase"], "rename_attempted")
        self.assertEqual(handle.state()["rename_attempts"], 1)
        self.assert_unpublished(handle)

    def test_exception_after_rename_retains_ambiguous_state_without_rollback(self):
        handle, _ = self.publisher()
        real = publication._native_rename_noreplace

        def rename_then_raise(*args):
            real(*args)
            raise OSError("SYNTHETIC-SECRET")

        with patch.object(publication, "_native_rename_noreplace", side_effect=rename_then_raise):
            self.deny(handle.publish)
        self.assertFalse(self.staging.exists())
        self.assertTrue(self.destination.is_dir())
        self.assertEqual(handle.state()["phase"], "rename_attempted")
        self.assertTrue(handle.state()["failed"])
        self.assertFalse(handle.state()["publication_verified"])
        self.assertEqual(handle.state()["parent_sync_attempts"], 0)

    def test_parent_sync_failure_retains_renamed_checkpoint_and_never_retries(self):
        handle, fd = self.publisher()
        with patch.object(publication.os, "fsync", side_effect=OSError("SYNTHETIC-SECRET")) as sync:
            self.deny(handle.publish)
            self.deny(handle.publish)
            handle.close()
        self.assertEqual(sync.call_args_list, [unittest.mock.call(fd)])
        self.assertFalse(self.staging.exists())
        self.assertTrue(self.destination.is_dir())
        self.assertEqual(handle.state()["phase"], "renamed")
        self.assertEqual(handle.state()["parent_sync_attempts"], 1)
        self.assertFalse(handle.state()["publication_verified"])

    def test_postrename_payload_mutation_is_detected_without_deletion(self):
        handle, _ = self.publisher()
        real = publication._native_rename_noreplace

        def corrupt(*args):
            real(*args)
            (self.destination / "model/__0_0.distcp").write_bytes(b"xyz")

        with patch.object(publication, "_native_rename_noreplace", side_effect=corrupt):
            self.deny(handle.publish)
        self.assertTrue(self.destination.is_dir())
        self.assertEqual((self.destination / "model/__0_0.distcp").read_bytes(), b"xyz")
        self.assertFalse(handle.state()["publication_verified"])

    def test_shared_budget_denial_after_rename_does_not_claim_publication_verified(self):
        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            if self.total() > self.one_pass:
                raise RuntimeError("SYNTHETIC-SECRET")

        handle, _ = self.publisher(reserve=reserve)
        self.deny(handle.publish)
        self.assertTrue(self.destination.is_dir())
        self.assertFalse(self.staging.exists())
        self.assertFalse(handle.state()["publication_verified"])
        self.assertGreater(handle.state()["checkpoint_read_bytes"], self.one_pass)

    def test_manifest_pin_mismatch_stops_before_rename(self):
        changed = dict(self.staging_receipt, manifest_sha256="0" * 64)
        handle, _ = self.publisher(staging_receipt=changed)
        with patch.object(publication, "_native_rename_noreplace") as rename:
            self.deny(handle.publish)
        rename.assert_not_called()
        self.assertEqual(self.total(), self.manifest_bytes)
        self.assert_unpublished(handle)

    def test_payload_corruption_stops_before_rename(self):
        (self.staging / "model/__0_0.distcp").write_bytes(b"xyz")
        handle, _ = self.publisher()
        with patch.object(publication, "_native_rename_noreplace") as rename:
            self.deny(handle.publish)
        rename.assert_not_called()
        self.assert_unpublished(handle)

    def test_unlisted_payload_stops_before_rename(self):
        extra = self.staging / "undeclared.bin"
        extra.write_bytes(b"x")
        extra.chmod(0o600)
        handle, _ = self.publisher()
        with patch.object(publication, "_native_rename_noreplace") as rename:
            self.deny(handle.publish)
        rename.assert_not_called()
        self.assert_unpublished(handle)

    def test_staging_payload_symlink_is_rejected_without_following(self):
        (self.staging / "config.json").unlink()
        (self.staging / "config.json").symlink_to("../../nonexistent-SYNTHETIC-target")
        handle, _ = self.publisher()
        with patch.object(publication.os, "pread") as read:
            self.deny(handle.publish)
        read.assert_not_called()
        self.assert_unpublished(handle)

    def test_descriptor_mutation_by_callback_fails_before_content_read(self):
        handle, fd = self.publisher()

        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            os.set_inheritable(fd, True)

        handle._reserve = reserve
        with patch.object(publication.os, "pread") as read:
            self.deny(handle.publish)
        read.assert_not_called()
        self.assertGreater(self.total(), 0)
        self.assert_unpublished(handle)

    def test_swallowed_callback_reentry_is_terminal(self):
        handle, _ = self.publisher()

        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            self.deny(handle.publish)

        handle._reserve = reserve
        with patch.object(publication.os, "pread") as read:
            self.deny(handle.publish)
        read.assert_not_called()
        self.assert_unpublished(handle)

    def test_callback_failure_is_raw_text_free_and_nonrefundable(self):
        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            raise RuntimeError("SYNTHETIC-SECRET " + str(self.root))

        handle, _ = self.publisher(reserve=reserve)
        self.deny(handle.publish)
        self.assertEqual(self.total(), self.manifest_bytes)
        self.assertEqual(handle.state()["checkpoint_read_bytes"], self.manifest_bytes)
        self.assert_unpublished(handle)

    def test_non_none_callback_result_is_rejected(self):
        handle, _ = self.publisher(reserve=lambda rank, **amounts: True)
        with patch.object(publication.os, "pread") as read:
            self.deny(handle.publish)
        read.assert_not_called()
        self.assert_unpublished(handle)

    def test_parent_inventory_change_in_callback_is_detected(self):
        def reserve(rank, **amounts):
            self.reserve(rank, **amounts)
            (self.root / "unexpected-private-dir").mkdir(mode=0o700)

        handle, _ = self.publisher(reserve=reserve)
        with patch.object(publication, "_native_rename_noreplace") as rename:
            self.deny(handle.publish)
        rename.assert_not_called()
        self.assert_unpublished(handle)

    def test_parent_entry_limit_is_enforced_before_content_read(self):
        (self.root / "one-more-directory").mkdir(mode=0o700)
        handle, _ = self.publisher()
        with patch.object(publication, "MAX_PARENT_ENTRIES", 1), \
                patch.object(publication.os, "pread") as read:
            self.deny(handle.publish)
        read.assert_not_called()
        self.assert_unpublished(handle)

    def test_unsupported_native_binding_fails_before_reads(self):
        handle, _ = self.publisher()
        with patch.object(publication, "_rename_binding", side_effect=AttributeError("secret")), \
                patch.object(publication.os, "pread") as read:
            self.deny(handle.publish)
        read.assert_not_called()
        self.assertEqual(handle.state()["rename_attempts"], 0)
        self.assert_unpublished(handle)

    def test_wrong_owner_thread_is_rejected(self):
        handle, _ = self.publisher()
        observed = []

        def other_thread():
            try:
                handle.publish()
            except publication.CheckpointPublicationError as error:
                observed.append(error.code)

        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(observed), 1)
        self.assert_unpublished(handle)

    def test_wrong_owner_process_is_rejected(self):
        handle, _ = self.publisher()
        with patch.object(publication.os, "getpid", return_value=os.getpid() + 1):
            self.deny(handle.publish)
        self.assert_unpublished(handle)

    def test_successful_constructor_takes_fd_and_close_is_idempotent(self):
        handle, fd = self.publisher()
        self.assertFalse(handle.closed)
        handle.close()
        handle.close()
        self.assertTrue(handle.closed)
        self.assertTrue(handle.state()["closed"])
        with self.assertRaises(OSError):
            os.fstat(fd)
        self.deny(handle.publish)

    def test_invalid_arguments_leave_fd_ownership_with_caller(self):
        invalids = [
            {"rank": True}, {"rank": -1}, {"rank": 12}, {"reserve": None},
            {"maximum_file_bytes": True}, {"maximum_file_bytes": 0},
            {"maximum_read_bytes": True}, {"maximum_read_bytes": 0},
            {"checkpoint_name": "../checkpoint-00000001"},
            {"checkpoint_name": "checkpoint-1"}, {"checkpoint_name": "checkpoint-0000000a"},
            {"staging_name": self.checkpoint_name},
            {"staging_name": f".{self.checkpoint_name}.{'A' * 32}.tmp"},
            {"staging_name": f".checkpoint-00000002.{'a' * 32}.tmp"},
            {"staging_name": "../" + self.staging_name},
        ]
        for options in invalids:
            with self.subTest(options=options):
                before = set(self.descriptors)
                self.deny(lambda options=options: self.publisher(**options))
                added = self.descriptors - before
                self.assertEqual(len(added), 1)
                for fd in added:
                    self.assertTrue(os.fstat(fd).st_ino)

    def test_staging_receipt_is_strict_and_copied(self):
        bad_receipts = [None, [], {}, dict(self.staging_receipt, extra=True)]
        for key, value in [("published", True), ("production_eligible", True),
                           ("status", "published_checkpoint"), ("payload_files", True),
                           ("payload_files", 1), ("payload_bytes", -1),
                           ("manifest_sha256", "A" * 64)]:
            bad_receipts.append(dict(self.staging_receipt, **{key: value}))
        for receipt in bad_receipts:
            with self.subTest(receipt=receipt):
                self.deny(lambda receipt=receipt: self.publisher(staging_receipt=receipt))
        copied = dict(self.staging_receipt)
        handle, _ = self.publisher(staging_receipt=copied)
        copied["manifest_sha256"] = "0" * 64
        handle.publish()

    def test_bad_parent_permissions_are_rejected_without_ownership_transfer(self):
        self.root.chmod(0o755)
        try:
            self.deny(self.publisher)
            self.assertEqual(len(self.descriptors), 1)
            self.assertTrue(os.fstat(next(iter(self.descriptors))).st_ino)
        finally:
            self.root.chmod(0o700)

    def test_inheritable_parent_is_rejected_without_ownership_transfer(self):
        real_open = os.open

        def inheritable(*args, **kwargs):
            fd = real_open(*args, **kwargs)
            os.set_inheritable(fd, True)
            return fd

        with patch.object(publication.os, "open", side_effect=inheritable):
            self.deny(self.publisher)
        self.assertEqual(len(self.descriptors), 1)
        self.assertTrue(os.get_inheritable(next(iter(self.descriptors))))

    def test_same_filesystem_requirement_is_checked_without_reading_payloads(self):
        real_stat = os.stat

        def changed_device(name, *args, **kwargs):
            info = real_stat(name, *args, **kwargs)
            if name == self.staging_name and "dir_fd" in kwargs:
                values = {key: getattr(info, key) for key in (
                    "st_dev", "st_ino", "st_mode", "st_nlink", "st_uid",
                    "st_size", "st_mtime_ns", "st_ctime_ns",
                )}
                values["st_dev"] += 1
                return types.SimpleNamespace(**values)
            return info

        handle, _ = self.publisher()
        with patch.object(publication.os, "stat", side_effect=changed_device), \
                patch.object(publication.os, "pread") as read:
            self.deny(handle.publish)
        read.assert_not_called()
        self.assert_unpublished(handle)

    def test_receipt_file_count_mismatch_stops_before_publication(self):
        changed = dict(self.staging_receipt, payload_files=2)
        handle, _ = self.publisher(staging_receipt=changed)
        with patch.object(publication, "_native_rename_noreplace") as rename:
            self.deny(handle.publish)
        rename.assert_not_called()
        self.assert_unpublished(handle)

    def test_receipt_manifest_size_mismatch_stops_before_reads(self):
        changed = dict(self.staging_receipt)
        changed["private_write_bytes"] += 1
        changed["checkpoint_read_bytes"] += 1
        handle, _ = self.publisher(staging_receipt=changed)
        with patch.object(publication.os, "pread") as read:
            self.deny(handle.publish)
        read.assert_not_called()
        self.assert_unpublished(handle)

    def test_close_does_not_close_a_descriptor_reused_for_another_inode(self):
        other = self.root / "another-SYNTHETIC-directory"
        other.mkdir(mode=0o700)
        handle, fd = self.publisher()
        os.close(fd)
        replacement = os.open(other, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        if replacement != fd:
            os.dup2(replacement, fd, inheritable=False)
            os.close(replacement)
        self.descriptors.add(fd)
        self.deny(handle.close)
        self.assertEqual(os.fstat(fd).st_ino, other.stat().st_ino)
        self.assertTrue(handle.closed)

    def test_close_never_retries_an_ambiguous_failure(self):
        handle, fd = self.publisher()
        real_close = os.close

        def close_then_raise(value):
            real_close(value)
            raise OSError("SYNTHETIC-SECRET")

        with patch.object(publication.os, "close", side_effect=close_then_raise) as close:
            self.deny(handle.close)
            handle.close()
        close.assert_called_once_with(fd)
        self.assertTrue(handle.closed)


class FakeNative:
    def __init__(self, result=0):
        self.result, self.calls = result, []

    def __call__(self, *args):
        self.calls.append(args)
        return self.result


class NativeRenameBindingTests(unittest.TestCase):
    def test_platform_bindings_use_exact_abi_flags_and_same_parent(self):
        for platform, symbol, flag in [("linux", "renameat2", 1),
                                       ("darwin", "renameatx_np", 4)]:
            with self.subTest(platform=platform):
                function = FakeNative()
                libc = types.SimpleNamespace(**{symbol: function})
                with patch.object(publication.sys, "platform", platform), \
                        patch.object(publication.ctypes, "CDLL", return_value=libc) as load:
                    binding = publication._rename_binding()
                    publication._native_rename_noreplace(binding, 23, ".synthetic", "checkpoint")
                load.assert_called_once_with(None, use_errno=True)
                self.assertEqual(function.argtypes,
                                 [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                  ctypes.c_char_p, ctypes.c_uint])
                self.assertIs(function.restype, ctypes.c_int)
                self.assertEqual(function.calls,
                                 [(23, b".synthetic", 23, b"checkpoint", flag)])

    def test_native_errno_never_exposes_paths_or_retries(self):
        function = FakeNative(-1)
        libc = types.SimpleNamespace(renameat2=function)
        with patch.object(publication.sys, "platform", "linux"), \
                patch.object(publication.ctypes, "CDLL", return_value=libc), \
                patch.object(publication.ctypes, "get_errno", return_value=errno.EEXIST):
            binding = publication._rename_binding()
            with self.assertRaises(publication.CheckpointPublicationError) as caught:
                publication._native_rename_noreplace(binding, 23, "SYNTHETIC-SECRET", "checkpoint")
        self.assertEqual(len(function.calls), 1)
        self.assertEqual(caught.exception.args, (caught.exception.code,))
        self.assertNotIn("SYNTHETIC-SECRET", repr(caught.exception))

    def test_native_errno_mapping_fails_closed_for_unsupported_filesystems(self):
        for code, expected in [(errno.ENOTEMPTY, "destination_exists"),
                               (errno.ENOSYS, "platform_unsupported"),
                               (errno.EINVAL, "platform_unsupported"),
                               (errno.ENOTSUP, "platform_unsupported"),
                               (errno.EIO, "rename_failed")]:
            with self.subTest(errno=code):
                function = FakeNative(-1)
                binding = (object(), function, 1)
                with patch.object(publication.ctypes, "get_errno", return_value=code), \
                        self.assertRaises(publication.CheckpointPublicationError) as caught:
                    publication._native_rename_noreplace(binding, 23, "source", "destination")
                self.assertEqual(caught.exception.code, expected)
                self.assertEqual(len(function.calls), 1)

    def test_unknown_platform_does_not_load_any_library(self):
        with patch.object(publication.sys, "platform", "unsupported-SYNTHETIC"), \
                patch.object(publication.ctypes, "CDLL") as load:
            with self.assertRaises(publication.CheckpointPublicationError):
                publication._rename_binding()
        load.assert_not_called()

    def test_missing_symbol_has_no_fallback(self):
        with patch.object(publication.sys, "platform", "linux"), \
                patch.object(publication.ctypes, "CDLL", return_value=types.SimpleNamespace()):
            with self.assertRaises(publication.CheckpointPublicationError):
                publication._rename_binding()


if __name__ == "__main__":
    unittest.main()
