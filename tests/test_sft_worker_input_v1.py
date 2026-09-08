"""Tiny stdlib guards/hooks only; no emulated Torch success or real input access."""

from __future__ import annotations

import importlib
import os
import stat
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_synthetic_worker_input_guards_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
worker = importlib.import_module(PACKAGE + ".posttraining_sft_worker_input_v1")
reader_module = importlib.import_module(PACKAGE + ".posttraining_sft_input_reader_v1")


@dataclass(frozen=True)
class Identity:
    changed_ns: int
    device: int
    inode: int
    mode: int
    modified_ns: int
    links: int
    owner: int
    size: int


def identity(info):
    return Identity(info.st_ctime_ns, info.st_dev, info.st_ino, stat.S_IMODE(info.st_mode),
                    info.st_mtime_ns, info.st_nlink, info.st_uid, info.st_size)


def flags():
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


class WorkerInputGuards(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lumi-input-guards-")
        self.root = Path(self.temporary.name)
        self.root.chmod(0o700)
        self.interrupts, self.reservations = [], []
        self.inputs, self.roots = [], []
        self.pins = dict.fromkeys(worker.PIN_FIELDS, "a" * 64)

    def tearDown(self):
        for source in self.inputs:
            for handle, _, _ in source._handles:
                if not handle.closed:
                    handle.close()
        for descriptor in self.roots:
            os.close(descriptor)
        self.temporary.cleanup()

    def source(self, **changes):
        options = {"rank": 0, "reserve": lambda rank, **amounts: self.reservations.append(amounts),
                   "interrupt": lambda: self.interrupts.append(True)}
        options.update(changes)
        result = worker.WorkerInput(**options)
        self.inputs.append(result)
        return result

    def hook_source(self):
        source = self.source()
        source._state = "loading"
        source._runtime = types.SimpleNamespace(_secure_open_flags=flags,
                                               _secure_file_identity=identity)
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.roots.append(directory)
        return source, directory

    def file(self, name="manifest.json", data=b"{}\n"):
        path = self.root / name
        with path.open("xb") as handle:
            self.assertEqual(handle.write(data), len(data))
        path.chmod(0o600)
        return path

    def error(self, code, operation, *args, **kwargs):
        with self.assertRaises(worker.WorkerInputError) as caught:
            operation(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.args, (code,))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_constructor_exact_rank_and_callbacks(self):
        for change in ({"rank": True}, {"rank": -1}, {"rank": 12}, {"rank": 1.0},
                       {"reserve": None}, {"interrupt": None}):
            with self.subTest(change=tuple(change)):
                self.error("invalid_arguments", self.source, **change)
        self.assertEqual(self.interrupts, [])

    def test_reinitialization_is_sticky_and_interrupts_once(self):
        source = self.source()
        self.error("already_consumed", source.__init__, rank=1, reserve=lambda *a: None,
                   interrupt=lambda: None)
        self.error("already_consumed", source.load, self.root, record_limit=1, **self.pins)
        self.assertEqual(self.interrupts, [True])

    def test_invalid_prefix_never_opens_or_imports_runtime(self):
        for limit in (None, True, False, 0, -1, 13, 1.0, "1"):
            source = self.source()
            with self.subTest(limit=limit), patch.object(os, "open") as opened:
                self.error("invalid_arguments", source.load, self.root,
                           record_limit=limit, **self.pins)
                opened.assert_not_called()

    def test_all_five_pins_are_mandatory_and_exact_before_io(self):
        for field in worker.PIN_FIELDS:
            for value in (None, "", "A" * 64, "g" * 64, "a" * 63, True):
                source, pins = self.source(), dict(self.pins)
                pins[field] = value
                with self.subTest(field=field, value=value), patch.object(os, "open") as opened:
                    self.error("invalid_arguments", source.load, self.root,
                               record_limit=1, **pins)
                    opened.assert_not_called()

    def test_unsafe_root_forms_reject_before_io(self):
        for path in ("relative", "/", "/tmp/../x", "/tmp/\0x", 3):
            source = self.source()
            with self.subTest(path=path), patch.object(os, "open") as opened:
                self.error("invalid_arguments", source.load, path, record_limit=1, **self.pins)
                opened.assert_not_called()

    def test_missing_runtime_cannot_make_first_ml_import_or_read(self):
        source = self.source()
        with patch.dict(sys.modules, {"torch": None, worker.RUNTIME_MODULE: None}), \
                patch.object(os, "open") as opened:
            self.error("runtime_rejected", source.load, self.root, record_limit=1, **self.pins)
            self.error("runtime_rejected", source.load, self.root, record_limit=1, **self.pins)
            opened.assert_not_called()
        self.assertEqual(self.interrupts, [True])

    def test_changed_owner_rejected(self):
        source = self.source()
        with patch.object(worker.os, "getpid", return_value=source._owner[0] + 1):
            self.error("owner_mismatch", source.load, self.root, record_limit=1, **self.pins)

    def test_reentry_poison_even_when_reservation_callback_swallows_error(self):
        source = self.source()
        source._state = "loading"

        def reenter(rank, **amounts):
            self.error("reentry", source.load, self.root, record_limit=1, **self.pins)

        source._parent_reserve = reenter
        self.error("reentry", source._reserve, 0, encoded_metadata_read_bytes=1)
        self.assertEqual(self.interrupts, [True])

    def test_parent_callback_return_exception_and_rank_are_rejected(self):
        for callback in (lambda *a, **k: True, lambda *a, **k: 1):
            source = self.source(reserve=callback)
            source._state = "loading"
            self.error("reserve_failed", source._reserve, 0, encoded_metadata_read_bytes=1)
        source = self.source()
        source._state = "loading"
        self.error("reserve_failed", source._reserve, False, encoded_metadata_read_bytes=1)

    def test_idle_or_retired_hooks_cannot_open_inputs(self):
        for state in ("idle", "complete"):
            source = self.source()
            source._state = state
            with patch.object(os, "open") as opened:
                self.error("already_consumed", source._open, 123, "manifest.json",
                           require_private_mode=False)
                opened.assert_not_called()

    def test_validation_and_unrecognized_names_reject_before_open(self):
        for name in ("validation-00000.tokens.u32le", "train-00000.extra", "../manifest.json",
                     "train-00000.index.jsonl/child", "train-0000.tokens.u32le"):
            source, directory = self.hook_source()
            with self.subTest(name=name), patch.object(os, "open") as opened:
                self.error("input_rejected", source._open, directory, name,
                           require_private_mode=True)
                opened.assert_not_called()

    def test_private_root_and_file_required_before_content(self):
        self.file()
        for root_mode, file_mode in ((0o755, 0o600), (0o700, 0o644)):
            source, directory = self.hook_source()
            self.root.chmod(root_mode)
            (self.root / "manifest.json").chmod(file_mode)
            with patch.object(os, "pread") as read:
                self.error("input_rejected", source._open, directory, "manifest.json",
                           require_private_mode=False)
                read.assert_not_called()
        self.assertEqual(self.reservations, [])

    def test_fifo_open_is_nonblocking_and_rejected_without_content(self):
        os.mkfifo(self.root / "manifest.json", 0o600)
        source, directory = self.hook_source()
        with patch.object(os, "pread") as read:
            self.error("input_rejected", source._open, directory, "manifest.json",
                       require_private_mode=False)
            read.assert_not_called()

    def test_successful_manifest_hook_separates_metadata_and_closes(self):
        self.file()
        source, directory = self.hook_source()
        handle, initial = source._open(directory, "manifest.json", require_private_mode=False)
        self.assertEqual(handle.read(16 * 1024 * 1024 + 1), b"{}\n")
        self.assertEqual(handle.tell(), 3)
        for attribute in ("fileno", "raw", "buffer", "seek", "name"):
            self.assertFalse(hasattr(handle, attribute))
        source._finish(handle, directory, "manifest.json", initial)
        self.assertTrue(handle.closed)
        self.assertEqual(self.reservations, [{"encoded_metadata_read_bytes": 3}])
        self.assertEqual(source.summary()["metadata_bytes_read"], 3)
        self.assertEqual(source.summary()["binary_bytes_read"], 0)

    def test_duplicate_file_open_is_rejected(self):
        self.file()
        source, directory = self.hook_source()
        source._open(directory, "manifest.json", require_private_mode=False)
        with patch.object(os, "open") as opened:
            self.error("input_rejected", source._open, directory, "manifest.json",
                       require_private_mode=False)
            opened.assert_not_called()

    def test_proxy_allocation_failure_closes_transferred_fd_once(self):
        self.file()
        source, directory = self.hook_source()
        close = os.close
        closed = []

        def observed_close(descriptor):
            closed.append(descriptor)
            return close(descriptor)

        with patch.object(worker, "_LoaderHandle", side_effect=MemoryError("synthetic")), \
                patch.object(os, "close", observed_close):
            with self.assertRaises(MemoryError):
                source._open(directory, "manifest.json", require_private_mode=False)
        self.assertEqual(len(closed), 1)
        self.assertEqual(source._handles, [])
        with self.assertRaises(OSError):
            os.fstat(closed[0])

    def test_changed_path_at_finish_poison_and_disown_handle(self):
        path = self.file()
        source, directory = self.hook_source()
        handle, initial = source._open(directory, "manifest.json", require_private_mode=False)
        path.rename(self.root / "old-generated.json")
        self.file(data=b"[]\n")
        with self.assertRaises((worker.WorkerInputError, reader_module.MeteredIOError)):
            source._finish(handle, directory, "manifest.json", initial)
        self.assertTrue(handle.closed)
        self.assertEqual(self.interrupts, [True])

    def test_close_failure_does_not_skip_frozen_style_root_finally_or_retry(self):
        self.file()
        source, directory = self.hook_source()
        handle, _ = source._open(directory, "manifest.json", require_private_mode=False)
        attempts, reached = [], []

        def ambiguous_file_close():
            descriptor, handle._reader._fd = handle._reader._fd, None
            attempts.append(descriptor)
            os.close(descriptor)
            raise OSError("SYNTHETIC_PRIVATE_CLOSE")

        with patch.object(handle._reader, "close", ambiguous_file_close):
            try:
                raise ValueError("synthetic_primary")
            except ValueError:
                pass
            finally:
                if not handle.closed:
                    handle.close()
                os.close(directory)
                reached.append(True)
                self.roots.remove(directory)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(reached, [True])
        self.assertTrue(handle.closed)
        self.assertEqual(source.summary()["state"], "failed")
        self.error("load_failed", source._check)
        self.assertEqual(self.interrupts, [True])
        with self.assertRaises(OSError):
            os.fstat(directory)

    def test_retired_proxy_cannot_interrupt_later_epoch(self):
        self.file()
        source, directory = self.hook_source()
        handle, initial = source._open(directory, "manifest.json", require_private_mode=False)
        source._finish(handle, directory, "manifest.json", initial)
        source._state = "complete"
        with patch.object(handle._reader, "close", side_effect=OSError("synthetic_late")):
            handle.close()
        self.assertEqual(source.summary()["state"], "complete")
        self.assertEqual(self.interrupts, [])
        self.error("already_consumed", handle.read, 1)

    def test_clone_preserves_code_and_original_globals_and_defaults(self):
        namespace = {"factor": 2}
        exec("def sample(value=3, *, extra=4): return value * factor + extra", namespace)
        original = namespace["sample"]
        clone = worker._clone(original, {"factor": 5})
        self.assertIs(clone.__code__, original.__code__)
        self.assertEqual(original(), 10)
        self.assertEqual(clone(), 19)
        clone.__kwdefaults__["extra"] = 1
        self.assertEqual(original(), 10)
        self.assertEqual(namespace["factor"], 2)

    def test_idle_summary_and_repr_contain_no_private_paths_or_pins(self):
        source = self.source()
        self.assertNotIn(str(self.root), str(source.summary()))
        self.assertNotIn("a" * 64, str(source.summary()))
        self.assertEqual(repr(source), "<WorkerInput private>")
        self.assertFalse(source.summary()["real_input_execution_permitted"])


if __name__ == "__main__":
    unittest.main()
