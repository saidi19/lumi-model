"""Source, JSON and syscall doubles only; never start the Linux supervisor."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import io
import os
import signal
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/supervise_checkpoint_guard_qualification_v1.py"
spec = importlib.util.spec_from_file_location("_checkpoint_guard_supervisor_v1", SOURCE)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def report():
    return {"schema": subject.HARNESS_SCHEMA, "status": "passed", "code": "ok",
            "stage": "complete", **dict.fromkeys(subject.FALSE_FLAGS, False),
            "cases": [{"status": "passed", "sha256": "a" * 64}]}


class ReportTests(unittest.TestCase):
    def reject(self, value, *, returncode=0, raw=False):
        with self.assertRaises(subject.SupervisorError) as caught:
            subject.parse_report(value if raw else subject.canonical(value), returncode)
        self.assertEqual(caught.exception.code, "output_rejected")

    def test_exact_basic_aggregate_copied_without_nested_semantic_claim(self):
        value = report()
        self.assertEqual(subject.parse_report(subject.canonical(value), 0), value)
        self.assertEqual(subject.parse_report(subject.canonical(value).rstrip(b"\n"), 0), value)

    def test_failed_report_requires_nonzero_exit(self):
        value = report()
        value["status"] = "failed"
        self.assertEqual(subject.parse_report(subject.canonical(value), 2), value)
        self.reject(value)
        self.reject(report(), returncode=2)

    def test_success_requires_complete_ok_codes(self):
        for key, replacement in (("code", "failed"), ("stage", "preflight")):
            value = report()
            value[key] = replacement
            self.reject(value)

    def test_required_false_authority_flags(self):
        for key in subject.FALSE_FLAGS:
            for bad in (True, 0, None, "false"):
                value = report()
                value[key] = bad
                self.reject(value)
            value = report()
            del value[key]
            self.reject(value)

    def test_non_ascii_paths_controls_and_oversized_strings_rejected(self):
        for text in ("/private/input", "C:\\input", "x\nx", "x\x00x", "é", "x" * 257):
            value = report()
            value["extra"] = text
            self.reject(value)
        for key in ("bad/key", "bad\\key", "bad\nkey", ""):
            value = report()
            value[key] = "x"
            self.reject(value)

    def test_duplicate_keys_and_noncanonical_or_extra_packets_rejected(self):
        raw = subject.canonical(report())
        duplicate = raw.replace(b'"status":', b'"status":"passed","status":')
        for altered in (raw + raw, b" " + raw, duplicate,
                        b"[]", b"NaN", b"{", b"{}\n", b"x" * 32769):
            self.reject(altered, raw=True)

    def test_floats_negative_excessive_ints_and_list_limits_rejected(self):
        for extra in (1.0, -1, 2**63, float("inf"), [0] * 65):
            value = report()
            value["extra"] = extra
            if extra == float("inf"):
                raw = subject.canonical(report()).replace(b'"cases":', b'"bad":Infinity,"cases":')
                self.reject(raw, raw=True)
            else:
                self.reject(value)

    def test_depth_and_total_node_limits(self):
        value = report()
        value["extra"] = [[[[[[[[[0]]]]]]]]]
        self.reject(value)
        value["extra"] = [[0] * 64 for _ in range(17)]
        self.reject(value)


class SourceTests(unittest.TestCase):
    def test_tiny_readonly_harness_pin_and_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "entry.py"
            path.write_bytes(b"# generated source\n")
            path.chmod(0o444)
            with patch.object(subject, "HARNESS", path):
                state = subject.verify_harness(hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(state[6], 19)
                with self.assertRaises(subject.SupervisorError):
                    subject.verify_harness("0" * 64)

    def test_symlink_hardlink_and_writable_source_denied(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            source = parent / "source.py"
            source.write_bytes(b"x")
            link = parent / "link.py"
            link.symlink_to(source)
            with self.assertRaises(subject.SupervisorError):
                subject._path(str(link))
            source.chmod(0o666)
            with self.assertRaises(subject.SupervisorError):
                subject._path(str(source))
            source.chmod(0o400)
            os.link(source, parent / "hard.py")
            with self.assertRaises(subject.SupervisorError):
                subject._path(str(source))

    def test_invalid_pin_rejected_before_filesystem_read(self):
        with patch.object(subject, "_path") as path:
            for pin in ("A" * 64, "a" * 63, None, 123):
                with self.assertRaises(subject.SupervisorError):
                    subject.verify_harness(pin)
            path.assert_not_called()

    def test_frozen_fixed_command_environment_and_no_broad_signals(self):
        tree = ast.parse(SOURCE.read_bytes())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        names = {ast.unparse(node.func) for node in calls}
        self.assertFalse({"os.kill", "os.killpg", "process.kill", "process.terminate",
                          "subprocess.run", "os.system"} & names)
        self.assertEqual(subject.FIXED_ENV, {"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8",
                         "LUMI_CHECKPOINT_GUARD_QUALIFICATION_CI": "1"})
        self.assertEqual(subject.MAXIMUM_HARNESS_SECONDS + subject.CLEANUP_SECONDS, 1020)
        self.assertEqual(subject.PYTHON, "/usr/bin/python3")
        self.assertEqual(subject.HARNESS.name, "run_checkpoint_guard_qualification_v1.py")
        self.assertTrue(ast.unparse(tree.body[-1].test) == "__name__ == '__main__'")

    def test_native_runtime_rejects_before_children_probe_on_local_host(self):
        with (patch.object(subject.sys, "platform", "darwin"),
              patch.object(subject.Path, "iterdir") as walk):
            with self.assertRaises(subject.SupervisorError) as caught:
                subject.runtime()
            self.assertEqual(caught.exception.code, "runtime_rejected")
            walk.assert_not_called()

    def test_nondefault_sigchld_denied_before_native_checks_or_spawn(self):
        for disposition in (signal.SIG_IGN, lambda _signum, _frame: None):
            with (patch.object(subject.sys, "platform", "linux"),
                  patch.object(subject.signal, "getsignal", return_value=disposition) as getsignal,
                  patch.object(subject.platform, "machine") as machine,
                  patch.object(subject.Path, "iterdir") as walk,
                  patch.object(subject.subprocess, "Popen") as spawn):
                with self.assertRaises(subject.SupervisorError) as caught:
                    subject.supervise("/tmp/bound.tar", "a" * 64)
                self.assertEqual(caught.exception.code, "runtime_rejected")
                getsignal.assert_called_once_with(signal.SIGCHLD)
                machine.assert_not_called()
                walk.assert_not_called()
                spawn.assert_not_called()


class WatchTests(unittest.TestCase):
    def doubles(self, *, chunks=None, exit_at=0.1, pidfd=81, signal_error=False,
                read_error=False, blocking_error=False, fileno_error=False,
                terminal_poll_delay=0.0, adopted=()):
        stack = ExitStack()
        self.addCleanup(stack.close)
        clock = [0.0]
        chunks = list([subject.canonical(report()), b""] if chunks is None else chunks)
        process = Mock()
        process.stdout.fileno.return_value = 71
        if fileno_error:
            process.stdout.fileno.side_effect = OSError("private fileno error")
        completion_observed = [False]
        def poll():
            if clock[0] < exit_at:
                return None
            if not completion_observed[0]:
                clock[0] += terminal_poll_delay
                completion_observed[0] = True
            return 0
        process.poll.side_effect = poll
        stack.enter_context(patch.object(subject.time, "monotonic", side_effect=lambda: clock[0]))
        def selected(read, _write, _error, timeout):
            clock[0] += max(timeout, 0.05)
            return (read if chunks else [], [], [])
        stack.enter_context(patch.object(subject.select, "select", side_effect=selected))
        def read(_fd, maximum):
            if read_error:
                raise OSError("private read error")
            part = chunks.pop(0)
            self.assertLessEqual(len(part), maximum)
            return part
        stack.enter_context(patch.object(subject.os, "read", side_effect=read))
        blocking = stack.enter_context(patch.object(subject.os, "set_blocking"))
        if blocking_error:
            blocking.side_effect = OSError("private blocking error")
        signals = stack.enter_context(patch.object(subject.signal, "pidfd_send_signal",
                                                  create=True))
        if signal_error:
            signals.side_effect = OSError("private signal error")
        statuses = iter([*adopted, ChildProcessError()])
        def reap(*_args):
            value = next(statuses)
            if isinstance(value, BaseException):
                raise value
            return value
        stack.enter_context(patch.object(subject.os, "waitpid", side_effect=reap))
        return subject.watch(process, pidfd, 0), signals, clock

    def test_normal_exit_collects_exact_one_report_no_signals(self):
        result, signals, clock = self.doubles()
        self.assertIsNone(result["failure"])
        self.assertEqual(result["raw"], subject.canonical(report()))
        self.assertTrue(result["harness_reaped"] and result["all_children_reaped"])
        signals.assert_not_called()
        self.assertLess(clock[0], 1)

    def test_deadline_terms_then_kills_only_exact_pidfd(self):
        result, signals, clock = self.doubles(chunks=[b""], exit_at=915)
        self.assertEqual(result["failure"], "deadline_exhausted")
        self.assertEqual(signals.call_args_list[0].args, (81, signal.SIGTERM, None, 0))
        self.assertEqual(signals.call_args_list[1].args, (81, signal.SIGKILL, None, 0))
        self.assertEqual(signals.call_count, 2)
        self.assertTrue(result["all_children_reaped"])
        self.assertLess(clock[0], 1020)

    def test_completed_harness_first_observed_after_deadline_cannot_pass(self):
        result, signals, clock = self.doubles(exit_at=900.001)
        self.assertGreater(clock[0], subject.MAXIMUM_HARNESS_SECONDS)
        self.assertEqual(result["failure"], "deadline_exhausted")
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["harness_reaped"] and result["all_children_reaped"])
        signals.assert_not_called()

    def test_completion_poll_crossing_deadline_cannot_use_stale_clock(self):
        result, signals, clock = self.doubles(terminal_poll_delay=900)
        self.assertGreater(clock[0], subject.MAXIMUM_HARNESS_SECONDS)
        self.assertEqual(result["failure"], "deadline_exhausted")
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["harness_reaped"] and result["all_children_reaped"])
        signals.assert_not_called()

    def test_missing_pidfd_fails_without_numeric_pid_fallback(self):
        result, signals, clock = self.doubles(chunks=[b""], exit_at=10, pidfd=None)
        self.assertEqual(result["failure"], "cleanup_unconfirmed")
        signals.assert_not_called()
        self.assertLess(clock[0], 121)

    def test_oversized_stdout_stops_with_bounded_buffer(self):
        chunks = [b"x" * 4096] * 8 + [b"x"]
        result, signals, _clock = self.doubles(chunks=chunks, exit_at=2)
        self.assertEqual(result["failure"], "output_rejected")
        self.assertEqual(len(result["raw"]), 32769)
        self.assertEqual(signals.call_count, 1)

    def test_signal_failure_not_retried_or_masked_as_cleanup(self):
        result, signals, _clock = self.doubles(chunks=[b""], exit_at=915, signal_error=True)
        self.assertEqual(result["failure"], "cleanup_unconfirmed")
        self.assertEqual(signals.call_count, 2)

    def test_live_unreaped_harness_stops_without_hostwide_cleanup(self):
        result, signals, clock = self.doubles(chunks=[b""], exit_at=2000)
        self.assertEqual(result["failure"], "cleanup_unconfirmed")
        self.assertFalse(result["harness_reaped"] or result["all_children_reaped"])
        self.assertEqual(signals.call_count, 2)
        self.assertLess(clock[0], 1021)

    def test_pipe_setup_or_read_failure_still_supervises_owned_child(self):
        for options in ({"read_error": True}, {"blocking_error": True},
                        {"fileno_error": True}):
            result, signals, _clock = self.doubles(exit_at=2, **options)
            self.assertEqual(result["failure"], "supervision_failed")
            self.assertTrue(result["harness_reaped"])
            self.assertEqual(signals.call_count, 1)

    def test_adopted_children_counted_and_abnormal_status_rejected(self):
        result, _signals, _clock = self.doubles(adopted=((812, 0), (813, 256)))
        self.assertEqual(result["adopted_children_reaped"], 2)
        self.assertFalse(result["all_children_reaped"])
        self.assertEqual(result["failure"], "cleanup_unconfirmed")

    def test_even_zero_exit_adopted_child_prevents_success(self):
        result, signals, _clock = self.doubles(adopted=((812, 0),))
        self.assertEqual(result["adopted_children_reaped"], 1)
        self.assertTrue(result["harness_reaped"] and result["all_children_reaped"])
        self.assertEqual(result["failure"], "cleanup_unconfirmed")
        signals.assert_not_called()

    def test_adopted_wait_has_only_120_seconds_after_early_harness_exit(self):
        with patch.object(subject, "_reap", return_value=(0, False)) as reap:
            result, _signals, clock = self.doubles()
        self.assertEqual(result["failure"], "cleanup_unconfirmed")
        self.assertLessEqual(reap.call_args.args[0], clock[0] + 120)

    def test_reap_waits_only_until_deadline_without_signalling_orphans(self):
        clock = [0.0]
        def selected(_read, _write, _error, timeout):
            clock[0] += timeout
        with (patch.object(subject.time, "monotonic", side_effect=lambda: clock[0]),
              patch.object(subject.select, "select", side_effect=selected),
              patch.object(subject.os, "waitpid", return_value=(0, 0)),
              patch.object(subject.signal, "pidfd_send_signal", create=True) as signals):
            self.assertEqual(subject._reap(0.2), (0, False))
            signals.assert_not_called()

    def test_reap_operating_system_failure_is_fixed_unconfirmed(self):
        with patch.object(subject.os, "waitpid", side_effect=OSError("private wait error")):
            self.assertEqual(subject._reap(subject.time.monotonic() + 1), (0, False))

    def test_reap_nondefault_sigchld_cannot_claim_status_evidence(self):
        for disposition in (signal.SIG_IGN, lambda _signum, _frame: None):
            with (patch.object(subject.signal, "getsignal", return_value=disposition),
                  patch.object(subject.os, "waitpid") as waitpid):
                self.assertEqual(subject._reap(1), (0, False))
                waitpid.assert_not_called()


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(subject, "_ATTEMPTED", False))

    def configured(self):
        self.stack.enter_context(patch.object(subject, "runtime"))
        self.stack.enter_context(patch.object(subject, "verify_harness", return_value=(1,)))
        self.stack.enter_context(patch.object(subject, "_path", return_value=(1,)))
        self.stack.enter_context(patch.object(subject, "subreaper"))
        process = Mock(pid=811)
        spawn = self.stack.enter_context(patch.object(subject.subprocess, "Popen",
                                                      return_value=process))
        self.stack.enter_context(patch.object(subject.os, "pidfd_open", return_value=81,
                                             create=True))
        self.stack.enter_context(patch.object(subject.os, "close"))
        value = {"raw": subject.canonical(report()), "returncode": 0, "failure": None,
                 "harness_term_attempts": 0, "harness_kill_attempts": 0,
                 "harness_reaped": True, "adopted_children_reaped": 0, "all_children_reaped": True}
        self.stack.enter_context(patch.object(subject, "watch", return_value=copy.deepcopy(value)))
        return spawn, process

    def test_fixed_launch_and_exact_nested_report_no_inherited_credentials(self):
        spawn, _process = self.configured()
        value = subject.supervise("/tmp/bound.tar", "a" * 64)
        self.assertEqual(value["status"], "passed")
        self.assertEqual(value["harness_report"], report())
        self.assertEqual(spawn.call_args.args[0], [subject.PYTHON, "-I", "-S", "-B",
                         str(subject.HARNESS), "--synthetic-only", "/tmp/bound.tar"])
        self.assertEqual(spawn.call_args.kwargs["env"], subject.FIXED_ENV)
        self.assertTrue(spawn.call_args.kwargs["close_fds"])
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        with self.assertRaises(subject.SupervisorError) as caught:
            subject.supervise("/tmp/bound.tar", "a" * 64)
        self.assertEqual(caught.exception.code, "already_attempted")
        self.assertEqual(spawn.call_count, 1)

    def test_started_spawn_failure_is_terminal(self):
        spawn, _process = self.configured()
        spawn.side_effect = OSError("private spawn error")
        with self.assertRaises(OSError):
            subject.supervise("/tmp/bound.tar", "a" * 64)
        self.assertTrue(subject._ATTEMPTED)

    def test_invalid_source_state_prevents_spawn(self):
        spawn, _process = self.configured()
        with patch.object(subject, "_path", side_effect=[(1,), (2,)]):
            with self.assertRaises(subject.SupervisorError):
                subject.supervise("/tmp/bound.tar", "a" * 64)
        spawn.assert_not_called()
        self.assertFalse(subject._ATTEMPTED)

    def test_missing_pidfd_binding_cannot_pass_even_if_child_completed(self):
        _spawn, _process = self.configured()
        with patch.object(subject.os, "pidfd_open", side_effect=OSError("private pidfd error")):
            value = subject.supervise("/tmp/bound.tar", "a" * 64)
        self.assertEqual(value["code"], "cleanup_unconfirmed")
        self.assertFalse(value["harness_pidfd_bound"])

    def test_nonzero_harness_result_retained_but_supervisor_fails(self):
        _spawn, _process = self.configured()
        value = report()
        value["status"] = "failed"
        observed = {"raw": subject.canonical(value), "returncode": 2, "failure": None,
                    "harness_term_attempts": 0, "harness_kill_attempts": 0,
                    "harness_reaped": True, "adopted_children_reaped": 0,
                    "all_children_reaped": True}
        with patch.object(subject, "watch", return_value=observed):
            result = subject.supervise("/tmp/bound.tar", "a" * 64)
        self.assertEqual(result["code"], "process_failed")
        self.assertEqual(result["harness_report"], value)

    def test_descriptor_cleanup_failure_cannot_pass(self):
        _spawn, process = self.configured()
        process.stdout.close.side_effect = OSError("private close error")
        value = subject.supervise("/tmp/bound.tar", "a" * 64)
        self.assertEqual(value["status"], "failed")
        self.assertEqual(value["code"], "cleanup_unconfirmed")
        self.assertIsNone(value["harness_report"])

    def test_main_only_fixed_error_no_private_exception(self):
        output = io.BytesIO()
        with patch.object(subject.sys, "argv", ["entry", "--synthetic-only", "/tmp/a", "a" * 64]), \
             patch.object(subject, "supervise", side_effect=RuntimeError("PRIVATE_SECRET")), \
             patch.object(subject.sys, "stdout", Mock(buffer=output)):
            self.assertEqual(subject.main(), 2)
        self.assertNotIn(b"PRIVATE_SECRET", output.getvalue())
        self.assertLess(len(output.getvalue()), 4096)


if __name__ == "__main__":
    unittest.main()
