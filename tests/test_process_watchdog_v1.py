"""Real local POSIX subprocess tests, synthetic code only. No model/data/network."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "src/lumi/posttraining_process_watchdog_v1.py"
SPEC = importlib.util.spec_from_file_location("lumi_watchdog_test_only", PATH)
watchdog = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = watchdog
SPEC.loader.exec_module(watchdog)


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lumi-watchdog-test-")
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "fixture.py"
        self.attempt = self.root / "attempt"

    def tearDown(self):
        self.temporary.cleanup()

    def contract(self, source=b"pass\n", **changes):
        self.source.write_bytes(source)
        value = watchdog.ProcessContract(
            python_executable=str(Path(sys.executable).resolve()), script_path=str(self.source),
            script_sha256=watchdog.digest(source), arguments=(), runtime_milliseconds=800,
            maximum_stream_bytes=4096, term_grace_milliseconds=10, maximum_cleanup_milliseconds=1000,
        )
        return dataclasses.replace(value, **changes)

    def run_child(self, source=b"pass\n", **changes):
        return watchdog.run_process(self.contract(source, **changes), self.attempt)

    def test_successful_real_child(self):
        result = self.run_child()
        self.assertEqual(result["status"], "PASS_child_process_only")
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["direct_child_reaped"])

    def test_child_failure_is_code_only(self):
        result = self.run_child(b"raise RuntimeError('SYNTHETIC-SECRET-EXCEPTION')\n")
        self.assertEqual(result["reason"], "child_failed")
        self.assertEqual(result["retained_child_text_bytes"], 0)
        self.assertNotIn("SYNTHETIC-SECRET-EXCEPTION", (self.attempt / "result.json").read_text())

    def test_sleeping_child_hits_deadline(self):
        result = self.run_child(b"import time\ntime.sleep(10)\n", runtime_milliseconds=100)
        self.assertEqual(result["reason"], "deadline_exhausted")
        self.assertLess(result["elapsed_milliseconds"], 2000)

    def test_child_ignoring_term_is_killed(self):
        source = b"import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(10)\n"
        result = self.run_child(source, runtime_milliseconds=150)
        self.assertEqual(result["reason"], "deadline_exhausted")
        self.assertEqual(result["guardian_exit_code"], -9)

    def test_stdout_flood_cannot_deadlock_or_be_retained(self):
        result = self.run_child(b"import os\nwhile True: os.write(1,b'x'*4096)\n", maximum_stream_bytes=100)
        self.assertEqual(result["reason"], "stream_budget_exhausted")
        self.assertEqual(result["observed_stream_bytes"], 101)
        self.assertEqual(result["retained_child_text_bytes"], 0)

    def test_stderr_flood_is_combined_budget(self):
        result = self.run_child(b"import os\nos.write(1,b'x'*40)\nos.write(2,b'y'*70)\n", maximum_stream_bytes=100)
        self.assertEqual(result["reason"], "stream_budget_exhausted")
        self.assertEqual(result["observed_stream_bytes"], 101)

    def test_short_output_under_budget_succeeds(self):
        result = self.run_child(b"import os\nos.write(1,b'123')\nos.write(2,b'456')\n", maximum_stream_bytes=6)
        self.assertEqual(result["status"], "PASS_child_process_only")
        self.assertEqual(result["observed_stream_bytes"], 6)

    def test_fast_exit_output_is_still_budgeted(self):
        result = self.run_child(b"import os\nos.write(1,b'z'*8000)\n", maximum_stream_bytes=5000)
        self.assertEqual(result["reason"], "stream_budget_exhausted")

    def test_no_environment_credentials_inherited(self):
        source = b"import os\nassert 'AWS_SECRET_ACCESS_KEY' not in os.environ\nassert 'GITHUB_TOKEN' not in os.environ\nassert 'PYTHONPATH' not in os.environ\n"
        with patch.dict(os.environ, {"AWS_SECRET_ACCESS_KEY": "SYNTHETIC", "GITHUB_TOKEN": "SYNTHETIC", "PYTHONPATH": "/synthetic"}):
            result = self.run_child(source)
        self.assertEqual(result["status"], "PASS_child_process_only")

    def test_arguments_are_not_shell_interpreted(self):
        source = b"import sys\nassert sys.argv[1] == '; touch SHOULD_NOT_EXIST'\n"
        result = self.run_child(source, arguments=("; touch SHOULD_NOT_EXIST",))
        self.assertEqual(result["status"], "PASS_child_process_only")
        self.assertFalse((self.attempt / "SHOULD_NOT_EXIST").exists())

    def test_attempt_is_private_and_consumed(self):
        contract = self.contract()
        watchdog.run_process(contract, self.attempt)
        with self.assertRaisesRegex(watchdog.WatchdogError, "attempt_already_exists"):
            watchdog.run_process(contract, self.attempt)
        self.assertEqual(self.attempt.stat().st_mode & 0o777, 0o700)
        for item in self.attempt.iterdir():
            self.assertEqual(item.stat().st_mode & 0o777, 0o600)

    def test_spawn_failure_consumes_marker(self):
        contract = self.contract()
        with patch.object(watchdog.subprocess, "Popen", side_effect=OSError("SECRET")):
            result = watchdog.run_process(contract, self.attempt)
        self.assertEqual(result["reason"], "spawn_failed")
        self.assertTrue((self.attempt / "started.json").exists())
        self.assertNotIn("SECRET", json.dumps(result))

    def test_parent_interrupt_kills_child(self):
        contract = self.contract(b"import time\ntime.sleep(10)\n")
        with patch.object(selectors.KqueueSelector if hasattr(selectors, "KqueueSelector") else selectors.EpollSelector,
                          "select", side_effect=KeyboardInterrupt):
            result = watchdog.run_process(contract, self.attempt)
        self.assertEqual(result["reason"], "parent_interrupted")
        self.assertTrue(result["direct_child_reaped"])

    def test_leader_exit_does_not_leave_same_group_grandchild(self):
        # A heartbeat file is a synthetic test side effect, not retained worker log.
        source = b"import os,time\npid=os.fork()\nif pid == 0:\n while True:\n  with open('heartbeat','ab') as f: f.write(b'x')\n  time.sleep(0.01)\ntime.sleep(0.05)\n"
        result = self.run_child(source)
        self.assertEqual(result["status"], "PASS_child_process_only")
        heartbeat = self.attempt / "heartbeat"
        self.assertTrue(heartbeat.exists())
        size = heartbeat.stat().st_size
        time.sleep(0.06)
        self.assertEqual(heartbeat.stat().st_size, size)

    def test_parent_fault_kills_child_and_scrubs_exception(self):
        contract = self.contract(b"import time\ntime.sleep(10)\n")
        with patch.object(selectors.KqueueSelector if hasattr(selectors, "KqueueSelector") else selectors.EpollSelector,
                          "select", side_effect=RuntimeError("PRIVATE")):
            result = watchdog.run_process(contract, self.attempt)
        self.assertEqual(result["reason"], "supervisor_internal_error")
        self.assertTrue(result["direct_child_reaped"])
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_wrong_script_hash_rejects_before_marker_or_spawn(self):
        contract = self.contract(script_sha256="0" * 64)
        with self.assertRaisesRegex(watchdog.WatchdogError, "source_digest_mismatch"):
            watchdog.run_process(contract, self.attempt)
        self.assertFalse(self.attempt.exists())

    def test_changed_script_rejects(self):
        contract = self.contract()
        self.source.write_bytes(b"raise Exception()\n")
        with self.assertRaises(watchdog.WatchdogError):
            watchdog.run_process(contract, self.attempt)

    def test_symlink_source_rejects(self):
        contract = self.contract()
        link = self.root / "link.py"
        link.symlink_to(self.source)
        with self.assertRaises(watchdog.WatchdogError):
            watchdog.run_process(dataclasses.replace(contract, script_path=str(link)), self.attempt)

    def test_hardlink_source_rejects(self):
        contract = self.contract()
        os.link(self.source, self.root / "linked.py")
        with self.assertRaises(watchdog.WatchdogError):
            watchdog.run_process(contract, self.attempt)

    def test_fifo_source_rejects_without_blocking(self):
        contract = self.contract()
        fifo = self.root / "fifo.py"
        os.mkfifo(fifo)
        with self.assertRaises(watchdog.WatchdogError):
            watchdog.run_process(dataclasses.replace(contract, script_path=str(fifo)), self.attempt)

    def test_source_size_checked_before_read(self):
        contract = self.contract()
        with patch.object(watchdog, "MAX_SCRIPT_BYTES", 1), self.assertRaises(watchdog.WatchdogError):
            watchdog.run_process(contract, self.attempt)

    def test_booleans_and_unbounded_caps_rejected(self):
        contract = self.contract()
        for field in ("runtime_milliseconds", "maximum_stream_bytes", "term_grace_milliseconds", "maximum_cleanup_milliseconds"):
            for bad in (True, 0, -1, 2**63, 1.5):
                with self.subTest(field=field, value=bad), self.assertRaises(watchdog.WatchdogError):
                    watchdog.validate_contract(dataclasses.replace(contract, **{field: bad}))

    def test_unsafe_environment_keys_rejected(self):
        contract = self.contract()
        for name in ("LD_PRELOAD", "PYTHONPATH", "AWS_SESSION_TOKEN", "HOME", "PATH"):
            with self.assertRaises(watchdog.WatchdogError):
                watchdog.validate_contract(dataclasses.replace(contract, environment=((name, "value"),)))

    def test_allowed_environment_is_explicit(self):
        source = b"import os\nassert os.environ['OMP_NUM_THREADS']=='1'\nassert os.environ['HF_HUB_OFFLINE']=='1'\n"
        self.assertEqual(self.run_child(source, environment=(("OMP_NUM_THREADS", "1"),))["status"], "PASS_child_process_only")

    def test_no_false_os_gpu_or_billing_claim(self):
        result = self.run_child()
        for field in ("setsid_escape_containment", "linux_cgroup_kill_verified", "gpu_execution_authorized_by_this_utility",
                      "network_or_filesystem_isolation_enforced", "cloud_instance_terminated", "cloud_billing_stopped"):
            self.assertIs(result[field], False)

    def _kill_outer_supervisor(self, sig):
        source = b"import time\nwhile True:\n with open('heartbeat','ab') as f: f.write(b'x')\n time.sleep(.01)\n"
        contract = self.contract(source, runtime_milliseconds=4000)
        harness = self.root / "supervisor_fixture.py"
        harness.write_text(
            "import importlib.util,sys\n"
            f"s=importlib.util.spec_from_file_location('watchdog_fixture',{str(PATH)!r})\n"
            "m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m)\n"
            f"c=m.ProcessContract(**{dataclasses.asdict(contract)!r})\n"
            f"m.run_process(c,{str(self.attempt)!r})\n"
        )
        process = subprocess.Popen([str(Path(sys.executable).resolve()), "-S", "-B", str(harness)],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        heartbeat = self.attempt / "heartbeat"
        try:
            deadline = time.monotonic() + 3
            while not heartbeat.exists() and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertTrue(heartbeat.exists())
            process.send_signal(sig)
            process.wait(timeout=3)
            time.sleep(.15)
            previous = heartbeat.stat().st_size
            time.sleep(.06)
            self.assertEqual(heartbeat.stat().st_size, previous)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)
            # Closing the sole lifetime writer guarantees guardian EOF on any
            # test-failure path; allow it to perform its own group cleanup.
            time.sleep(.03)

    def test_real_parent_sigterm_stops_worker(self):
        self._kill_outer_supervisor(signal.SIGTERM)

    def test_real_parent_sigkill_triggers_guardian_eof_cleanup(self):
        self._kill_outer_supervisor(signal.SIGKILL)

    def test_real_parent_sighup_stops_worker(self):
        self._kill_outer_supervisor(signal.SIGHUP)

    def test_partial_pipe_setup_failure_cannot_block_cleanup(self):
        contract = self.contract(b"import time\ntime.sleep(10)\n")
        selector_type = selectors.KqueueSelector if hasattr(selectors, "KqueueSelector") else selectors.EpollSelector
        with patch.object(selector_type, "register", side_effect=OSError("PRIVATE")):
            result = watchdog.run_process(contract, self.attempt)
        self.assertEqual(result["reason"], "supervisor_internal_error")
        self.assertTrue(result["direct_child_reaped"])
        self.assertLess(result["elapsed_milliseconds"], 2000)

    def test_original_parent_signal_handlers_restored(self):
        before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
        self.run_child()
        self.assertEqual(before, {sig: signal.getsignal(sig) for sig in before})

    def test_late_success_cannot_override_deadline(self):
        self.assertEqual(watchdog._completion_reason(0, False, 100, 100), "deadline_exhausted")

    def test_success_cannot_override_cancellation(self):
        self.assertEqual(watchdog._completion_reason(0, True, 50, 100), "parent_interrupted")

    def test_auto_reaping_sigchld_is_rejected_before_spawn(self):
        contract = self.contract()
        original = signal.getsignal
        with patch.object(signal, "getsignal", side_effect=lambda sig: signal.SIG_IGN if sig == signal.SIGCHLD else original(sig)):
            with self.assertRaisesRegex(watchdog.WatchdogError, "platform_unsupported"):
                watchdog.run_process(contract, self.attempt)
        self.assertFalse(self.attempt.exists())


if __name__ == "__main__":
    unittest.main()
