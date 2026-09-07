"""Stdlib-only worker/storage boundary tests using tiny owned synthetic files."""

from __future__ import annotations

import hashlib
import importlib
import os
import stat
import sys
import tempfile
import types
import unittest
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_checkpoint_worker_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
worker_module = importlib.import_module(f"{PACKAGE}.posttraining_sft_checkpoint_worker_v1")
readers = importlib.import_module(f"{PACKAGE}.posttraining_sft_checkpoint_reader_v1")
views = importlib.import_module(f"{PACKAGE}.posttraining_sft_checkpoint_view_v1")


class FakeTensor:
    """Tensor-like value whose direct container equality is intentionally invalid."""

    def __init__(self, values, *, dtype="float64", shape=(2,)):
        self.values, self.dtype, self.shape = tuple(values), dtype, shape

    def __eq__(self, _other):
        raise AssertionError("fixture_tensor_direct_equality_is_ambiguous")


class FakeTorch:
    Tensor = FakeTensor

    @staticmethod
    def equal(left, right):
        return left.values == right.values


class CheckpointFixtureComparatorTests(unittest.TestCase):
    """Pure-stdlib regression for the CI assertion helper, not tensor execution."""

    @classmethod
    def setUpClass(cls):
        before = set(sys.modules)
        script = ROOT / "scripts/run_sft_checkpoint_worker_torch_fixture_v1.py"
        spec = importlib.util.spec_from_file_location("_pure_checkpoint_comparator", script)
        cls.fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fixture)
        forbidden = {"torch", "transformers", "numpy", "safetensors", "tokenizers"}
        if any(name.split(".")[0] in forbidden for name in set(sys.modules) - before):
            raise AssertionError("fixture_import_loaded_ml")

    def equal(self, left, right):
        return self.fixture.equal_tree(FakeTorch, left, right)

    def test_direct_ordered_container_equality_is_invalid_fixture_control(self):
        left = OrderedDict(weight=FakeTensor([1, 2]))
        right = OrderedDict(weight=FakeTensor([1, 2]))
        with self.assertRaisesRegex(AssertionError, "fixture_tensor_direct_equality_is_ambiguous"):
            _ = left == right

    def test_ordered_model_state_recurses_instead_of_comparing_tensors_directly(self):
        left = OrderedDict(weight=FakeTensor([1, 2]), bias=FakeTensor([3, 4]))
        right = OrderedDict(weight=FakeTensor([1, 2]), bias=FakeTensor([3, 4]))
        self.assertIs(self.equal(left, right), True)

    def test_ordered_key_order_is_preserved(self):
        left = OrderedDict(weight=FakeTensor([1, 2]), bias=FakeTensor([3, 4]))
        right = OrderedDict(bias=FakeTensor([3, 4]), weight=FakeTensor([1, 2]))
        self.assertIs(self.equal(left, right), False)

    def test_plain_and_ordered_mapping_types_are_not_interchangeable(self):
        plain = {"weight": FakeTensor([1, 2])}
        ordered = OrderedDict(weight=FakeTensor([1, 2]))
        self.assertIs(self.equal(plain, ordered), False)
        self.assertIs(self.equal(ordered, plain), False)

    def test_plain_dict_keeps_existing_key_set_semantics(self):
        left = {"step": 1, "value": FakeTensor([1, 2])}
        right = {"value": FakeTensor([1, 2]), "step": 1}
        self.assertIs(self.equal(left, right), True)

    def test_nested_optimizer_state_and_sequence_types(self):
        left = {"state": {0: {"exp_avg": FakeTensor([1, 2])}}, "param_groups": [{"params": [0]}]}
        right = {"state": {0: {"exp_avg": FakeTensor([1, 2])}}, "param_groups": [{"params": [0]}]}
        self.assertIs(self.equal(left, right), True)
        self.assertIs(self.equal([FakeTensor([1, 2])], (FakeTensor([1, 2]),)), False)

    def test_missing_and_extra_ordered_keys_are_rejected(self):
        left = OrderedDict(weight=FakeTensor([1, 2]))
        right = OrderedDict(weight=FakeTensor([1, 2]), bias=FakeTensor([3, 4]))
        self.assertIs(self.equal(left, right), False)
        self.assertIs(self.equal(right, left), False)

    def test_tensor_value_dtype_and_shape_mismatches_are_rejected(self):
        left = OrderedDict(weight=FakeTensor([1, 2]))
        for right in (FakeTensor([1, 3]), FakeTensor([1, 2], dtype="float32"),
                      FakeTensor([1, 2], shape=(1, 2))):
            with self.subTest(dtype=right.dtype, shape=right.shape):
                self.assertIs(self.equal(left, OrderedDict(weight=right)), False)


class CheckpointWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lumi-SYNTHETIC-checkpoint-worker-v1-")
        self.root = Path(self.temporary.name)
        self.root.chmod(0o700)
        (self.root / "model").mkdir(mode=0o700)
        self.root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.model_fd = os.open(self.root / "model", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.descriptors, self.sessions = [self.root_fd, self.model_fd], []
        self.charges, self.interrupts = [], []

    def tearDown(self):
        for session in self.sessions:
            try:
                session.close()
            except readers.CheckpointReadError:
                pass
        for descriptor in self.descriptors:
            os.close(descriptor)
        self.temporary.cleanup()

    def limits(self, **changes):
        result = {"maximum_file_bytes": 2048, "maximum_read_bytes": 8192,
                  "maximum_read_call_bytes": 64, "maximum_write_call_bytes": 64,
                  "maximum_sync_operations": 8}
        result.update(changes)
        return result

    def reserve(self, rank, **amounts):
        self.charges.append((rank, dict(amounts)))

    def interrupt(self):
        self.interrupts.append(True)

    def worker(self, **changes):
        options = {"rank": 0, "world_size": 1, "reserve": self.reserve,
                   "interrupt": self.interrupt, "limits": self.limits()}
        options.update(changes)
        return worker_module.CheckpointWorker(**options)

    def backend(self, worker, *, model=True):
        return worker_module._WriteDirectory(self.model_fd if model else self.root_fd,
                                             worker, model_directory=model)

    def write(self, backend, name, data, *, mode="wb"):
        with backend.worker._operation():
            path = backend.concat_path(worker_module.OPAQUE_ROOT, name)
            with backend.create_stream(path, mode) as stream:
                stream.write(data)
            return stream

    def path(self, name):
        return PurePosixPath(worker_module.OPAQUE_ROOT) / name

    def private_file(self, name, data):
        path = self.root / name
        path.write_bytes(data)
        path.chmod(0o600)
        return path

    def read_session(self, worker):
        payloads = {"model/.metadata": b"SYNTHETIC", "model/__0_0.distcp": b"0123456789",
                    "trainer-rank-00000.pt": b"NOT-PICKLE"}
        for name, data in payloads.items():
            self.private_file(name, data)
        manifest = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n"
                           for name, data in payloads.items()).encode("ascii")
        self.private_file("manifest.sha256", manifest)
        session = readers.CheckpointReadSession(
            os.dup(self.root_fd), rank=worker.rank, reserve=worker._reserve,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            maximum_file_bytes=2048, maximum_read_bytes=8192)
        self.sessions.append(session)
        session.verify()
        self.charges.clear()
        return session

    def assert_error(self, code, function, *args, **kwargs):
        with self.assertRaises(worker_module.CheckpointWorkerError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.args, (code,))
        self.assertEqual(caught.exception.code, code)
        self.assertNotIn(str(self.root), repr(caught.exception))

    def test_constructor_copies_limits_and_exposes_immutable_mapping(self):
        limits = self.limits()
        worker = self.worker(limits=limits)
        limits["maximum_read_bytes"] = 1
        self.assertEqual(worker.limits["maximum_read_bytes"], 8192)
        with self.assertRaises(TypeError):
            worker.limits["maximum_read_bytes"] = 1

    def test_constructor_rejects_invalid_scope_and_caps(self):
        changes = ({"rank": True}, {"rank": -1}, {"rank": 1}, {"world_size": True},
                   {"world_size": 0}, {"world_size": 13}, {"reserve": None},
                   {"interrupt": None}, {"limits": {}},
                   {"limits": self.limits(maximum_file_bytes=True)},
                   {"limits": self.limits(maximum_file_bytes=0)},
                   {"limits": self.limits(maximum_file_bytes=2**63)})
        for changed in changes:
            with self.subTest(changed=changed):
                self.assert_error("invalid_arguments", self.worker, **changed)
        self.assertEqual(self.interrupts, [])

    def test_reinitialization_cannot_reset_failed_ledger(self):
        worker = self.worker()
        worker._reserve(0, checkpoint_read_bytes=3)
        self.assert_error("invalid_arguments", worker.__init__, rank=0, world_size=1,
                          reserve=self.reserve, interrupt=self.interrupt, limits=self.limits())
        self.assertEqual(worker._reads, 3)
        self.assertTrue(worker._failed)
        self.assertEqual(self.interrupts, [True])
        self.assert_error("terminal_failure", worker._check)

    def test_read_cap_is_cumulative_and_rejected_reservation_not_forwarded(self):
        worker = self.worker(limits=self.limits(maximum_read_bytes=5))
        worker._reserve(0, checkpoint_read_bytes=3)
        worker._reserve(0, private_write_bytes=2)
        worker._reserve(0, checkpoint_read_bytes=2)
        self.assert_error("read_cap_exceeded", worker._reserve, 0, checkpoint_read_bytes=1)
        self.assertEqual(worker._reads, 5)
        self.assertEqual(len(self.charges), 3)
        self.assertEqual(self.interrupts, [True])

    def test_reservation_validates_rank_resources_and_exact_positive_integers(self):
        cases = ((True, {"checkpoint_read_bytes": 1}), (1, {"checkpoint_read_bytes": 1}),
                 (0, {}), (0, {"unknown": 1}), (0, {"checkpoint_read_bytes": -1}),
                 (0, {"checkpoint_read_bytes": 0}), (0, {"checkpoint_read_bytes": True}),
                 (0, {"private_write_bytes": 1.0}), (0, {"private_write_bytes": 2**63}))
        for rank, amounts in cases:
            with self.subTest(rank=rank, amounts=amounts):
                worker = self.worker()
                before = len(self.charges)
                self.assert_error("invalid_arguments", worker._reserve, rank, **amounts)
                self.assertEqual(worker._reads, 0)
                self.assertEqual(len(self.charges), before)

    def test_failed_callback_keeps_read_debit(self):
        def fail(rank, **amounts):
            self.reserve(rank, **amounts)
            raise RuntimeError("SYNTHETIC-PRIVATE-ERROR")

        worker = self.worker(reserve=fail)
        self.assert_error("reserve_failed", worker._reserve, 0, checkpoint_read_bytes=3)
        self.assertEqual(worker._reads, 3)
        self.assertEqual(self.interrupts, [True])

    def test_non_none_callback_result_is_terminal(self):
        worker = self.worker(reserve=lambda rank, **amounts: True)
        self.assert_error("reserve_failed", worker._reserve, 0, checkpoint_read_bytes=3)
        self.assertEqual(worker._reads, 3)
        self.assertEqual(self.interrupts, [True])

    def test_swallowed_reservation_reentry_poison_outer_call(self):
        worker = self.worker()

        def reenter(rank, **amounts):
            self.reserve(rank, **amounts)
            self.assert_error("reentry", worker._reserve, rank, checkpoint_read_bytes=1)

        worker._reservation = reenter
        self.assert_error("terminal_failure", worker._reserve, 0, checkpoint_read_bytes=3)
        self.assertEqual(worker._reads, 3)
        self.assertEqual(self.interrupts, [True])

    def test_interrupt_runs_once_even_if_it_raises(self):
        def fail():
            self.interrupt()
            raise RuntimeError("SYNTHETIC-INTERRUPT")

        worker = self.worker(interrupt=fail)
        self.assert_error("storage_rejected", worker._fail, "storage_rejected")
        self.assert_error("terminal_failure", worker._check)
        self.assertEqual(self.interrupts, [True])

    def test_operation_reentry_poison_outer_and_releases_global_lock(self):
        worker = self.worker()
        with self.assertRaises(worker_module.CheckpointWorkerError):
            with worker._operation():
                with self.assertRaises(worker_module.CheckpointWorkerError):
                    with worker._operation():
                        self.fail("reentry ran")
        self.assertFalse(worker_module._SERIALIZER_LOCK.locked())
        self.assertEqual(self.interrupts, [True])

    def test_different_worker_cannot_enter_global_serializer_scope(self):
        first, second = self.worker(), self.worker()
        with first._operation():
            with self.assertRaises(worker_module.CheckpointWorkerError):
                with second._operation():
                    self.fail("concurrent serializer ran")
        self.assertFalse(first._failed)
        self.assertTrue(second._failed)
        self.assertFalse(worker_module._SERIALIZER_LOCK.locked())

    def test_operation_failure_redacts_and_releases_lock(self):
        worker = self.worker()
        with self.assertRaises(worker_module.CheckpointWorkerError) as caught:
            with worker._operation():
                raise RuntimeError("SYNTHETIC-PRIVATE-TEXT")
        self.assertEqual(caught.exception.args, ("operation_failed",))
        self.assertEqual(self.interrupts, [True])
        self.assertFalse(worker_module._SERIALIZER_LOCK.locked())

    def test_owner_checks_are_terminal(self):
        worker = self.worker()
        with patch.object(worker_module.os, "getpid", return_value=os.getpid() + 1):
            self.assert_error("owner_mismatch", worker._check)
        worker = self.worker()
        with patch.object(worker_module.threading, "get_ident", return_value=-1):
            self.assert_error("owner_mismatch", worker._check)

    def test_first_failure_code_and_stage_are_retained(self):
        worker = self.worker()
        worker._stage = "idle"
        self.assert_error("storage_rejected", worker._fail, "storage_rejected")
        self.assert_error("terminal_failure", worker._check)
        self.assertEqual(worker.failure_stage, "idle")
        self.assertEqual(worker.failure_code, "storage_rejected")
        self.assertEqual(repr(worker), "<CheckpointWorker private>")
        self.assertEqual(worker_module.CheckpointWorkerError("SYNTHETIC-SECRET").args,
                         ("operation_failed",))

    def test_directory_descriptor_requires_private_directory_and_cloexec(self):
        self.assertEqual(worker_module._directory_info(self.root_fd)[0][0:2],
                         (os.fstat(self.root_fd).st_dev, os.fstat(self.root_fd).st_ino))
        for descriptor in (True, -1, 0, 1, 2):
            self.assert_error("invalid_descriptor", worker_module._directory_info, descriptor)
        os.set_inheritable(self.root_fd, True)
        self.assert_error("invalid_descriptor", worker_module._directory_info, self.root_fd)
        os.set_inheritable(self.root_fd, False)
        self.root.chmod(0o755)
        self.assert_error("invalid_descriptor", worker_module._directory_info, self.root_fd)
        self.root.chmod(0o700)

    def test_child_directory_no_follow_and_borrowed_parent(self):
        with worker_module._child_directory(self.root_fd, "model") as child:
            self.assertEqual(os.fstat(child).st_ino, os.fstat(self.model_fd).st_ino)
        os.fstat(self.root_fd)
        (self.root / "link").symlink_to(self.root / "model")
        for name in ("link", "..", ".", "model/.."):
            with self.subTest(name=name), self.assertRaises(worker_module.CheckpointWorkerError):
                with worker_module._child_directory(self.root_fd, name):
                    self.fail("unsafe child opened")

    def test_write_stream_produces_private_receipt_and_charges(self):
        worker = self.worker()
        backend = self.backend(worker)
        stream = self.write(backend, "__0_0.distcp", b"abc")
        self.assertTrue(stream.closed)
        self.assertEqual(backend.receipts, {"__0_0.distcp": {
            "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}})
        self.assertEqual(self.charges, [(0, {"private_write_bytes": 3})])
        self.assertEqual(stat.S_IMODE((self.root / "model/__0_0.distcp").stat().st_mode), 0o600)
        self.assertFalse(hasattr(stream, "fileno"))

    def test_write_file_and_call_caps_fail_before_content(self):
        for name, limits in (("__0_0.distcp", self.limits(maximum_file_bytes=2)),
                             (".metadata.tmp", self.limits(maximum_write_call_bytes=2))):
            with self.subTest(name=name):
                worker = self.worker(limits=limits)
                backend = self.backend(worker)
                self.assert_error("operation_failed", self.write, backend, name, b"abc")
                self.assertEqual((self.root / "model" / name).stat().st_size, 0)
                self.assertNotIn(name, backend.receipts)
                self.assertTrue(worker._failed)
        self.assertEqual(self.charges, [])

    def test_write_budget_failure_retains_empty_evidence_and_interrupts(self):
        def deny(rank, **amounts):
            self.reserve(rank, **amounts)
            raise RuntimeError("SYNTHETIC-BUDGET")

        worker = self.worker(reserve=deny)
        backend = self.backend(worker)
        self.assert_error("operation_failed", self.write, backend, "__0_0.distcp", b"abc")
        self.assertEqual((self.root / "model/__0_0.distcp").stat().st_size, 0)
        self.assertEqual(self.charges, [(0, {"private_write_bytes": 3})])
        self.assertEqual(self.interrupts, [True])

    def test_write_rejects_wrong_rank_foreign_names_and_modes(self):
        for name, mode in (("__1_0.distcp", "wb"), ("../outside", "wb"),
                           (".metadata", "wb"), ("__0_1.distcp", "wb"),
                           ("__0_0.distcp", "rb")):
            with self.subTest(name=name, mode=mode):
                worker = self.worker()
                self.assert_error("operation_failed", self.write, self.backend(worker),
                                  name, b"abc", mode=mode)
                self.assertTrue(worker._failed)
        self.assertEqual(list((self.root / "model").iterdir()), [])

    def test_existing_file_and_symlink_are_never_overwritten(self):
        self.private_file("model/__0_0.distcp", b"OLD")
        (self.root / "model/.metadata.tmp").symlink_to(self.root / "not-created")
        for name in ("__0_0.distcp", ".metadata.tmp"):
            worker = self.worker()
            self.assert_error("operation_failed", self.write, self.backend(worker), name, b"NEW")
        self.assertEqual((self.root / "model/__0_0.distcp").read_bytes(), b"OLD")
        self.assertTrue((self.root / "model/.metadata.tmp").is_symlink())
        self.assertFalse((self.root / "not-created").exists())
        self.assertEqual(self.charges, [])

    def test_rank_one_cannot_write_coordinator_metadata_or_state(self):
        for name, model in ((".metadata.tmp", True), (worker_module.STATE_NAME, False)):
            worker = self.worker(rank=1, world_size=2)
            self.assert_error("operation_failed", self.write, self.backend(worker, model=model),
                              name, b"abc")
        self.assertEqual(self.charges, [])

    def test_distinct_rank_file_creation_is_allowed_directory_metadata_change(self):
        worker = self.worker()
        backend = self.backend(worker)
        self.private_file("model/__1_0.distcp", b"OTHER-RANK")
        backend.check()
        self.write(backend, "__0_0.distcp", b"MINE")
        self.assertFalse(worker._failed)

    def test_directory_mode_change_is_terminal(self):
        worker = self.worker()
        backend = self.backend(worker)
        (self.root / "model").chmod(0o755)
        with self.assertRaises(worker_module.CheckpointWorkerError):
            with worker._operation():
                backend.check()
        self.assertTrue(worker._failed)
        (self.root / "model").chmod(0o700)

    def test_pathname_swap_after_close_cannot_receive_original_receipt(self):
        worker = self.worker()
        backend = self.backend(worker)
        original_close = worker_module.CheckpointWriteStream.close

        def swap(stream):
            already_closed = stream.closed
            original_close(stream)
            if not already_closed:
                (self.root / "model/__0_0.distcp").rename(self.root / "model/retained-original")
                self.private_file("model/__0_0.distcp", b"xyz")

        with patch.object(worker_module.CheckpointWriteStream, "close", swap):
            self.assert_error("operation_failed", self.write, backend, "__0_0.distcp", b"abc")
        self.assertEqual(backend.receipts, {})
        self.assertTrue(worker._failed)
        self.assertEqual((self.root / "model/retained-original").read_bytes(), b"abc")

    def test_metadata_rename_is_no_replace_and_preserves_receipt(self):
        worker = self.worker()
        backend = self.backend(worker)
        self.write(backend, ".metadata.tmp", b"SYNTHETIC")
        receipt = dict(backend.receipts[".metadata.tmp"])
        with worker._operation():
            backend.rename(self.path(".metadata.tmp"), self.path(".metadata"))
        self.assertEqual(backend.receipts, {".metadata": receipt})
        self.assertFalse((self.root / "model/.metadata.tmp").exists())
        self.assertEqual((self.root / "model/.metadata").read_bytes(), b"SYNTHETIC")

    def test_metadata_existing_destination_is_preserved_without_retry(self):
        worker = self.worker()
        backend = self.backend(worker)
        self.write(backend, ".metadata.tmp", b"NEW")
        self.private_file("model/.metadata", b"OLD")
        with self.assertRaises(worker_module.CheckpointWorkerError):
            with worker._operation():
                backend.rename(self.path(".metadata.tmp"), self.path(".metadata"))
        self.assertEqual((self.root / "model/.metadata").read_bytes(), b"OLD")
        self.assertEqual((self.root / "model/.metadata.tmp").read_bytes(), b"NEW")
        self.assertTrue(worker._failed)

    def test_metadata_mutation_is_rejected_before_rename(self):
        worker = self.worker()
        backend = self.backend(worker)
        self.write(backend, ".metadata.tmp", b"OLD")
        self.private_file("model/.metadata.tmp", b"CHANGED")
        with patch.object(worker_module, "_native_rename_noreplace") as rename:
            self.assert_error("descriptor_changed", backend.rename,
                              self.path(".metadata.tmp"), self.path(".metadata"))
        rename.assert_not_called()

    def test_remove_is_always_rejected(self):
        worker = self.worker()
        backend = self.backend(worker)
        self.private_file("model/__0_0.distcp", b"RETAIN")
        self.assert_error("storage_rejected", backend.rm_file, self.path("__0_0.distcp"))
        self.assertEqual((self.root / "model/__0_0.distcp").read_bytes(), b"RETAIN")

    def test_read_factory_metered_view_closes_without_closing_session(self):
        worker = self.worker()
        session = self.read_session(worker)
        backend = worker_module._ReadDirectory(session, worker)
        with backend.create_stream(self.path("__0_0.distcp"), "rb") as view:
            self.assertEqual(view.read(3), b"012")
            self.assertFalse(hasattr(view, "fileno"))
        self.assertTrue(view.closed)
        self.assertFalse(session.closed)
        self.assertEqual(self.charges, [(0, {"checkpoint_read_bytes": 3})])

    def test_read_directory_rejects_unsafe_names_and_modes(self):
        for name, mode in (("../config.json", "rb"), (".metadata.tmp", "rb"),
                           ("trainer-rank-00000.pt", "rb"), ("__0_0.distcp", "wb")):
            with self.subTest(name=name, mode=mode):
                worker = self.worker()
                backend = worker_module._ReadDirectory(None, worker)
                with self.assertRaises(worker_module.CheckpointWorkerError):
                    with backend.create_stream(self.path(name), mode):
                        self.fail("unsafe reader opened")
                self.assertTrue(worker._failed)

    def test_read_call_cap_before_content_poison_worker_operation(self):
        worker = self.worker(limits=self.limits(maximum_read_call_bytes=2))
        session = self.read_session(worker)
        backend = worker_module._ReadDirectory(session, worker)
        with self.assertRaises(worker_module.CheckpointWorkerError):
            with worker._operation():
                with backend.create_stream(self.path("__0_0.distcp"), "rb") as view:
                    view.read()
        self.assertEqual(self.charges, [])
        self.assertTrue(worker._failed)
        self.assertTrue(session._failed)

    def test_reader_unexpected_declared_shape_is_still_blocked_by_session(self):
        worker = self.worker()
        session = self.read_session(worker)
        backend = worker_module._ReadDirectory(session, worker)
        with self.assertRaises(worker_module.CheckpointWorkerError):
            with worker._operation():
                with backend.create_stream(self.path("__9_0.distcp"), "rb"):
                    self.fail("undeclared file read")
        self.assertTrue(worker._failed)
        self.assertEqual(self.charges, [])

    def test_fake_dcp_writer_receives_only_explicit_safe_options(self):
        captured = {}

        class FakeWriter:
            def __init__(self, path, **options):
                captured.update(path=path, **options)
                self.sync_files, self.overwrite, self.thread_count = False, False, 1
                self.serialization_format = types.SimpleNamespace(name="TORCH_SAVE")

        worker = self.worker()
        writer, backend = worker_module._dcp_writer(
            self.model_fd, worker, types.SimpleNamespace(FileSystemWriter=FakeWriter))
        self.assertEqual(captured, {"path": worker_module.OPAQUE_ROOT,
                                   "single_file_per_rank": True, "thread_count": 1,
                                   "per_thread_copy_ahead": 0, "sync_files": False,
                                   "overwrite": False, "_extensions": ()})
        self.assertIs(writer.fs, backend)
        self.assertEqual(writer.path, PurePosixPath(worker_module.OPAQUE_ROOT))

    def test_fake_dcp_writer_capability_drift_rejected(self):
        class UnsafeWriter:
            def __init__(self, path, **options):
                self.sync_files, self.overwrite, self.thread_count = True, False, 1
                self.serialization_format = types.SimpleNamespace(name="TORCH_SAVE")

        worker = self.worker()
        self.assert_error("runtime_rejected", worker_module._dcp_writer, self.model_fd, worker,
                          types.SimpleNamespace(FileSystemWriter=UnsafeWriter))

    def test_fake_dcp_reader_uses_bounded_independent_slice(self):
        class FakeReader:
            def __init__(self, path):
                self.path = path

        worker = self.worker()
        session = self.read_session(worker)
        reader = worker_module._dcp_reader(session, worker,
                                            types.SimpleNamespace(FileSystemReader=FakeReader))
        with reader.fs.create_stream(self.path("__0_0.distcp"), "rb") as view:
            view.seek(8)
            sliced = reader._slice_file(view, types.SimpleNamespace(
                offset=2, length=3, transform_descriptors=None))
            self.assertEqual(sliced.read(), b"234")
            self.assertEqual(view.tell(), 8)
            sliced.close()
        self.assertEqual(self.charges, [(0, {"checkpoint_read_bytes": 3})])

    def test_no_read_transforms_rejects_extensions_without_touching_stream(self):
        transforms = worker_module._NoReadTransforms()
        raw = object()
        self.assertIs(transforms.transform_load_stream(None, (), raw), raw)
        self.assert_error("storage_rejected", transforms.transform_load_stream,
                          None, ("SYNTHETIC-EXTENSION",), raw)

    def test_single_rank_collective_helpers_use_only_success_or_fixed_failure(self):
        worker = self.worker()
        self.assertEqual(worker._rank_zero(None, lambda: {"count": 1}), {"count": 1})
        self.assertEqual(worker._all_local(None, lambda: 2), [2])

        def fail():
            raise RuntimeError("SYNTHETIC-PRIVATE-COLLECTIVE")

        self.assert_error("collective_failed", worker._all_local, None, fail)
        self.assertEqual(self.interrupts, [True])


if __name__ == "__main__":
    unittest.main()
