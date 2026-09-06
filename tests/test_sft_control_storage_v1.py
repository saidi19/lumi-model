"""Synthetic CI storage guards and deterministic pre-spawn deadline reproduction.

No hosted runner, private input, real child process, mount or timeout relaxation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "synthetic_control_storage", ROOT / "scripts/run_sft_control_ci_v1.py"
)
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


def mountinfo(device=0, filesystem=b"tmpfs", mount=b"/dev/shm"):
    identity = f"{os.major(device)}:{os.minor(device)}".encode("ascii")
    return b"1 2 " + identity + b" / " + mount + b" rw - " + filesystem + b" tmpfs rw\n"


class ControlStorageTests(unittest.TestCase):
    def test_only_exact_tmpfs_mount_and_device_are_accepted(self):
        ci.validate_tmpfs_mountinfo(mountinfo(), 0)
        for value in (
            b"", b"x" * (ci.MAXIMUM_MOUNTINFO_BYTES + 1),
            mountinfo(filesystem=b"ext4"), mountinfo(mount=b"/tmp"),
            mountinfo() * 2, mountinfo().replace(b"0:0", b"1:1"),
            b"1 2 0:0 / /dev/shm rw", b"1 2 0:0 / /dev/shm rw - \n",
        ):
            with self.subTest(size=len(value)), self.assertRaisesRegex(
                ci.ControlCIError, "watchdog_tmpfs_mount_invalid"
            ):
                ci.validate_tmpfs_mountinfo(value, 0)

    def test_unrelated_mounts_are_not_mistaken_for_target(self):
        ci.validate_tmpfs_mountinfo(mountinfo(mount=b"/synthetic") + mountinfo(), 0)

    def storage_probe(self, *, mode=0o1777, owner=0, blocks=4096, inodes=64):
        info = types.SimpleNamespace(st_mode=stat.S_IFDIR | mode, st_uid=owner, st_dev=0)
        space = types.SimpleNamespace(f_bavail=blocks, f_frsize=4096, f_favail=inodes)
        return (
            patch.object(ci.os, "open", return_value=42),
            patch.object(ci.os, "fstat", return_value=info),
            patch.object(ci.os, "fstatvfs", return_value=space),
            patch.object(ci.os, "close"),
            patch("builtins.open", mock_open(read_data=mountinfo())),
        )

    def test_kernel_probe_is_read_only_nofollow_bounded_and_closes_fd(self):
        opened, stated, space, closed, proc = self.storage_probe()
        with opened as op, stated, space, closed as close, proc as read:
            self.assertEqual(ci.require_watchdog_tmpfs(), 0)
        self.assertEqual(op.call_args.args[0], "/dev/shm")
        self.assertTrue(op.call_args.args[1] & os.O_NOFOLLOW)
        self.assertFalse(op.call_args.args[1] & os.O_CREAT)
        read.assert_called_once_with("/proc/self/mountinfo", "rb")
        read().read.assert_called_once_with(ci.MAXIMUM_MOUNTINFO_BYTES + 1)
        close.assert_called_once_with(42)

    def test_insecure_or_capacity_deficient_mount_fails_closed(self):
        for changes, code in (
            ({"mode": 0o777}, "watchdog_tmpfs_root_invalid"),
            ({"owner": 1001}, "watchdog_tmpfs_root_invalid"),
            ({"blocks": 4095}, "watchdog_tmpfs_capacity_insufficient"),
            ({"inodes": 63}, "watchdog_tmpfs_capacity_insufficient"),
        ):
            opened, stated, space, closed, proc = self.storage_probe(**changes)
            with self.subTest(changes=changes), opened, stated, space, closed as close, proc:
                with self.assertRaisesRegex(ci.ControlCIError, code):
                    ci.require_watchdog_tmpfs()
            close.assert_called_once_with(42)

    def test_mount_read_failure_still_closes_descriptor(self):
        opened, stated, space, closed, _ = self.storage_probe()
        with opened, stated, space, closed as close:
            with patch("builtins.open", side_effect=OSError("SYNTHETIC_PRIVATE")):
                with self.assertRaises(OSError):
                    ci.require_watchdog_tmpfs()
        close.assert_called_once_with(42)

    def test_nonlinux_keeps_original_tempfile_environment(self):
        previous = tempfile.tempdir
        with patch.object(ci.sys, "platform", "darwin"), patch.object(
            ci, "require_watchdog_tmpfs", side_effect=AssertionError("must not probe")
        ):
            with ci.watchdog_test_storage():
                self.assertEqual(tempfile.tempdir, previous)
        self.assertEqual(tempfile.tempdir, previous)

    def test_private_storage_restored_and_removed_after_success_or_failure(self):
        with tempfile.TemporaryDirectory() as root:
            device = Path(root).stat().st_dev
            for fail in (False, True):
                previous = tempfile.tempdir
                created = None
                try:
                    with (
                        patch.object(ci.sys, "platform", "linux"),
                        patch.object(ci, "WATCHDOG_TMPFS", root),
                        patch.object(ci, "require_watchdog_tmpfs", return_value=device),
                        ci.watchdog_test_storage(),
                    ):
                        created = Path(tempfile.tempdir)
                        self.assertEqual(created.parent, Path(root))
                        self.assertEqual(stat.S_IMODE(created.stat().st_mode), 0o700)
                        with tempfile.TemporaryDirectory() as inner:
                            self.assertEqual(Path(inner).parent, created)
                        if fail:
                            raise AssertionError("SYNTHETIC propagated, never waived")
                except AssertionError as error:
                    if not fail or str(error) != "SYNTHETIC propagated, never waived":
                        raise
                self.assertIsNotNone(created)
                self.assertFalse(created.exists())
                self.assertEqual(tempfile.tempdir, previous)

    def test_probe_failure_never_runs_suite_or_falls_back_to_disk(self):
        executed = False
        with (
            patch.object(ci.sys, "platform", "linux"),
            patch.object(ci, "require_watchdog_tmpfs", side_effect=OSError("SYNTHETIC_PRIVATE")),
            self.assertRaisesRegex(ci.ControlCIError, "^watchdog_tmpfs_unavailable$") as caught,
        ):
            with ci.watchdog_test_storage():
                executed = True
        self.assertFalse(executed)
        self.assertNotIn("SYNTHETIC_PRIVATE", str(caught.exception))

    def test_wrong_device_private_directory_is_rejected_before_suite(self):
        with tempfile.TemporaryDirectory() as root:
            with (
                patch.object(ci.sys, "platform", "linux"),
                patch.object(ci, "WATCHDOG_TMPFS", root),
                patch.object(ci, "require_watchdog_tmpfs", return_value=-1),
                self.assertRaisesRegex(
                    ci.ControlCIError, "watchdog_tmpfs_private_directory_invalid"
                ),
            ):
                with ci.watchdog_test_storage():
                    self.fail("must not enter suite")
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_storage_only_wraps_exact_original_watchdog_suite(self):
        executed, contexts = [], []

        class FakeSuite:
            def __init__(self, count):
                self.count = count

            def countTestCases(self):
                return self.count

            def run(self, result):
                result.testsRun = self.count
                executed.append((self.count, len(contexts)))

        context = Mock()
        context.__enter__ = Mock(side_effect=lambda: contexts.append(True))
        context.__exit__ = Mock(side_effect=lambda *_: contexts.clear())
        fake_spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda _: None))
        fake_module = types.SimpleNamespace(WatchdogTests=unittest.TestCase)
        with (
            patch.object(ci, "require_startup"),
            patch.object(ci.importlib.util, "spec_from_file_location", return_value=fake_spec),
            patch.object(ci.importlib.util, "module_from_spec", return_value=fake_module),
            patch.object(ci.unittest.defaultTestLoader, "loadTestsFromModule", side_effect=[
                FakeSuite(count) for _, count in ci.SUITES
            ]),
            patch.object(ci, "watchdog_test_storage", return_value=context) as storage,
        ):
            report = ci.execute_suites(ROOT)
        storage.assert_called_once_with()
        self.assertEqual(executed, [(count, int(count == 33)) for _, count in ci.SUITES])
        self.assertEqual(report["tests_run"], 273)

    def test_storage_admission_failure_codes_survive_parent_without_raw_text(self):
        for code in ci.WATCHDOG_STORAGE_FAILURE_CODES:
            raw = json.dumps({
                "schema_version": ci.SCHEMA, "status": "STOP", "failure_code": code,
            }).encode()
            with (
                self.subTest(code=code), patch.object(ci, "require_parent_startup"),
                patch.object(ci.subprocess, "run", return_value=types.SimpleNamespace(
                    returncode=1, stdout=raw
                )),
                self.assertRaisesRegex(ci.ControlCIError, "^" + code + "$") as caught,
            ):
                ci.run_fresh_process(ROOT)
            self.assertIsNone(caught.exception.diagnostic)

    def test_storage_failure_parser_rejects_unknown_and_duplicate_fields(self):
        good = {
            "schema_version": ci.SCHEMA, "status": "STOP",
            "failure_code": "watchdog_tmpfs_unavailable",
        }
        for changed in (
            {**good, "raw_text": "SYNTHETIC_PRIVATE"}, {**good, "failure_code": "PRIVATE"},
            {**good, "failure_code": []}, {**good, "status": "PASS"},
            {**good, "schema_version": "unknown"}, [], None,
        ):
            self.assertIsNone(ci.storage_failure_code(json.dumps(changed).encode()))
        for raw in (
            b"", b"PRIVATE", b"\xff", b"x" * (ci.MAXIMUM_REPORT_BYTES + 1), "not bytes",
            b'{"status":"STOP",' + json.dumps(good).encode()[1:],
        ):
            self.assertIsNone(ci.storage_failure_code(raw))


class WatchdogSetupDeadlineTests(unittest.TestCase):
    def test_setup_flush_latency_can_exhaust_both_original_budgets_before_spawn(self):
        spec = importlib.util.spec_from_file_location(
            "synthetic_watchdog_setup_deadline",
            ROOT / "src/lumi/posttraining_process_watchdog_v1.py",
        )
        watchdog = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: watchdog}):
            spec.loader.exec_module(watchdog)
            real_fsync = os.fsync
            for budget in (100, 800):
                with self.subTest(budget=budget), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory).resolve()
                    source = root / "fixture.py"
                    source.write_bytes(b"pass\n")
                    contract = watchdog.ProcessContract(
                        python_executable=str(Path(sys.executable).resolve()),
                        script_path=str(source), script_sha256=watchdog.digest(b"pass\n"),
                        arguments=(), runtime_milliseconds=budget, maximum_stream_bytes=4096,
                        term_grace_milliseconds=10, maximum_cleanup_milliseconds=1000,
                    )
                    clock = [1_000_000_000]
                    flushes = []

                    def delayed_flush(fd, flushes=flushes, clock=clock, budget=budget):
                        real_fsync(fd)
                        flushes.append(fd)
                        clock[0] += (budget // 5 + 1) * 1_000_000

                    with (
                        patch.object(
                            watchdog.time, "monotonic_ns", side_effect=lambda clock=clock: clock[0]
                        ),
                        patch.object(watchdog.os, "fsync", side_effect=delayed_flush),
                        patch.object(watchdog.subprocess, "Popen") as spawn,
                    ):
                        result = watchdog.run_process(contract, root / "attempt")
                    spawn.assert_not_called()
                    self.assertEqual(len(flushes), 7)  # five setup, two final result flushes
                    self.assertEqual(result["reason"], "deadline_exhausted_before_spawn")
                    self.assertFalse(result["child_spawned"])
                    self.assertFalse(result["worker_status_reported"])
                    self.assertTrue((root / "attempt" / "started.json").is_file())
                    self.assertEqual(result["retained_child_text_bytes"], 0)
                    self.assertEqual(result["execution_deadline_milliseconds"], budget)
                    self.assertGreater(result["elapsed_milliseconds"], budget)


if __name__ == "__main__":
    unittest.main()
