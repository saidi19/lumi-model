"""Synthetic doubles only: no native guardian, fork, Docker, ML or real payload."""

import copy
import stat
import types
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

from lumi import posttraining_checkpoint_guard_qualification_v1 as subject
from tests.test_sft_checkpoint_guard_composition_v1 import (
    ExitNative,
    Lease,
    environment,
    exit_native,
    worker,
)


def receipt(case):
    if case == "positive":
        return worker()
    return {"schema": subject.frozen.WORKER_SCHEMA, "status": "failed",
            "code": "entry_binding_rejected" if case == "wrong_entry"
            else "source_binding_rejected", "synthetic_only": True,
            "real_4b_import": False, "training_authorized": False,
            "production_eligible": False}


def final(case):
    return {"status": "passed", "case": "worker_run", "reason": "workload_empty",
            "empty_and_removed": True, "workload_processes_reaped": 1,
            "kill_attempts": 1, "worker_exit_code": 0 if case == "positive" else 1,
            "worker": receipt(case)}


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(subject, "_ATTEMPTED", False))

    def invoke(self, case="positive", **overrides):
        args = {"case": case, "rootfs_path": "/owned/root", "mountpoint_path": "/owned/mount",
                "root_identity": (1, 2, stat.S_IFDIR, 0, 0), "environment": environment(),
                "base_script_path": "/release/scripts/run_sft_cgroup_fixture_v1.py",
                "expected_base_sha256": subject.BASE_SHA256, "deadline_ns": 90_000_000_000}
        args.update(overrides)
        return subject.run_prepared_case(**args)

    def api(self, case="positive", result=None):
        self.stack.enter_context(patch.object(subject.frozen.child_bootstrap, "_runtime"))
        self.stack.enter_context(patch.object(subject.time, "monotonic_ns", return_value=100))
        base = types.SimpleNamespace(subreaper=Mock(), wait_statuses=Mock(),
                                     run_case=Mock(return_value=final(case)
                                                   if result is None else result))
        loader = self.stack.enter_context(patch.object(subject.frozen, "_load_base",
                                                       return_value=base))
        return base, loader

    def guarded(self, case="positive", *, ready=None, report=None, statuses=None,
                kill_error=False, close_error=False, late_close=False, duplicate_release=False):
        with ExitStack() as stack:
            lease, events = Lease(), []
            lease.kill_error, lease.close_error = kill_error, close_error
            outer, channel, child = Mock(), Mock(), Mock()
            outer.fileno.return_value = 80
            channel.send.side_effect = lambda raw: events.append(("release", raw)) or len(raw)
            sent = []

            def send(passed, value):
                self.assertIs(passed, outer)
                sent.append(copy.deepcopy(value))

            def receive(_channel, _deadline):
                if "ready" not in events:
                    events.append("ready")
                    lease.populated = 1
                    return copy.deepcopy(subject.frozen.child_bootstrap.READY
                                         if ready is None else ready)
                return copy.deepcopy(receipt(case) if report is None else report)

            def watch(passed, *, expected_parent_pid, deadline_ns, cleanup_timeout_ms, release):
                self.assertIs(passed, lease)
                self.assertEqual((expected_parent_pid, cleanup_timeout_ms), (77, 10000))
                events.append("watch")
                release()
                if duplicate_release:
                    release()
                lease.kill()
                return {"reason": "workload_empty", "kill_attempted": True,
                        "empty_verified": True, "parent_pidfd_bound": True,
                        "release_called": True, "pidfd_closed": True}

            factory = Mock(return_value=lease)
            modules = {
                "posttraining_sft_cgroup_v1": types.SimpleNamespace(CgroupLease=factory),
                "posttraining_sft_cgroup_guardian_v1": types.SimpleNamespace(watch_parent=watch),
            }
            base = types.SimpleNamespace(close_except=Mock(), subreaper=Mock(), send=send,
                                         module=modules.__getitem__, wait_statuses=Mock(
                    return_value={123: 0 if case == "positive" else 256}
                    if statuses is None else statuses))
            stack.enter_context(patch.object(subject.os, "open", return_value=51))
            closer = stack.enter_context(patch.object(subject.os, "close"))
            if late_close:
                closer.side_effect = OSError("private error")
            fork = stack.enter_context(patch.object(subject.os, "fork", return_value=123))
            stack.enter_context(patch.object(subject.os, "_exit", side_effect=exit_native))
            stack.enter_context(patch.object(subject.socket, "socketpair",
                                             return_value=(channel, child)))
            stack.enter_context(patch.object(subject.time, "monotonic_ns", return_value=100))
            stack.enter_context(patch.object(subject.frozen, "_receive", side_effect=receive))
            stack.enter_context(patch.object(subject.frozen, "_packet", return_value=b""))
            bootstrap = stack.enter_context(patch.object(subject.frozen.child_bootstrap,
                                                         "bootstrap"))
            context = {"rootfs_path": "/owned/root", "mountpoint_path": "/owned/mount",
                       "root_identity": (1, 2, stat.S_IFDIR, 0, 0),
                       "environment": subject._environment(case, environment())}
            with self.assertRaises(ExitNative) as captured:
                subject._guard(base, context, case, "worker_run", outer, 77, 10001, 10001,
                               subject.frozen.child_bootstrap.RUNTIME, 90_000_000_000)
            fork.assert_called_once_with()
            bootstrap.assert_not_called()
            return sent[-1], lease, events, captured.exception.code, factory

    def test_exact_case_names(self):
        self.assertEqual(subject.CASES, ("wrong_entry", "wrong_manifest", "positive"))

    def test_copied_environment_changes_exactly_one_negative_pin(self):
        genuine = environment()
        for case, key in (("wrong_entry", "LUMI_IMPORT_ENTRY_SHA256"),
                          ("wrong_manifest", "LUMI_IMPORT_SOURCE_MANIFEST_SHA256")):
            selected = subject._environment(case, genuine)
            self.assertEqual({name for name in selected if selected[name] != genuine[name]}, {key})
            self.assertEqual(selected[key], "0" * 64)
        self.assertEqual(genuine, environment())
        selected = subject._environment("positive", genuine)
        self.assertEqual(selected, genuine)
        self.assertIsNot(selected, genuine)

    def test_missing_zero_bool_and_extra_environment_rejected(self):
        for case in subject.CASES:
            for key in subject.frozen.child_bootstrap.HASH_ENV:
                for value in ("0" * 64, True, 1, "A" * 64):
                    with self.subTest(case=case, key=key, value=value), self.assertRaises(
                            ValueError):
                        subject._environment(case, {**environment(), key: value})
        with self.assertRaises(ValueError):
            subject._environment("positive", {**environment(), "LD_PRELOAD": "x"})

    def test_negative_exact_receipts_and_exit_codes(self):
        for case in subject.CASES:
            expected_exit = 0 if case == "positive" else 1
            self.assertEqual(subject._worker(receipt(case), case, environment(), expected_exit),
                             receipt(case))
            for observed in (True, False, 256, 2, 1 - expected_exit):
                with self.subTest(case=case, observed=observed), self.assertRaises(ValueError):
                    subject._worker(receipt(case), case, environment(), observed)

    def test_negative_wrong_code_status_bool_or_extra_never_passes(self):
        for case in subject.CASES[:2]:
            for key, value in (("status", "passed"), ("code", "fixture_failed"),
                               ("code", receipt(subject.CASES[1] if case == subject.CASES[0]
                                                else subject.CASES[0])["code"]),
                               ("synthetic_only", 1), ("real_4b_import", 0), ("extra", False)):
                with self.subTest(case=case, key=key), self.assertRaises(ValueError):
                    subject._worker({**receipt(case), key: value}, case, environment(), 1)

    def test_positive_still_requires_original_complete_worker_schema(self):
        for value in (receipt("wrong_entry"), {**worker(), "checkpoint_full_audits": 3},
                      {**worker(), "entry_sha256": "0" * 64}):
            with self.assertRaises(ValueError):
                subject._worker(value, "positive", environment(), 0)

    def test_guard_positive_and_both_negative_success(self):
        for case in subject.CASES:
            with self.subTest(case=case):
                result, lease, events, code, factory = self.guarded(case)
                self.assertEqual(result, final(case))
                self.assertEqual((lease.kills, lease.closes, code), (1, 1, 0))
                self.assertLess(events.index("watch"), events.index(("release", b"G")))
                self.assertEqual(factory.call_args.kwargs["limits"], subject.frozen.LIMITS)

    def test_guard_negative_exit_zero_signal_bool_or_other_child_failure_rejected(self):
        for statuses in ({123: 0}, {123: 9}, {123: True}, {123: 256, 124: 256}, {124: 256}):
            with self.subTest(statuses=statuses):
                result, lease, *_ = self.guarded("wrong_entry", statuses=statuses)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(lease.kills, 1)

    def test_guard_wrong_negative_receipt_rejected(self):
        result, lease, *_ = self.guarded("wrong_manifest", report=receipt("wrong_entry"))
        self.assertEqual((result["status"], lease.kills), ("failed", 1))

    def test_strict_ready_no_integer_boolean(self):
        ready = {**subject.frozen.child_bootstrap.READY, "isolation_installed": 1}
        result, lease, events, *_ = self.guarded("wrong_entry", ready=ready)
        self.assertEqual((result["status"], lease.kills), ("failed", 1))
        self.assertNotIn("watch", events)

    def test_ambiguous_kill_close_and_late_close_never_retry_or_pass(self):
        for options in ({"kill_error": True}, {"close_error": True}, {"late_close": True}):
            with self.subTest(options=options):
                result, lease, *_ = self.guarded("wrong_manifest", **options)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(lease.kills, 1)
                self.assertLessEqual(lease.closes, 1)
                if not options.get("kill_error"):
                    self.assertEqual(result["code"], "cleanup_unconfirmed")

    def test_duplicate_release_denies_before_second_byte(self):
        result, lease, events, *_ = self.guarded("wrong_entry", duplicate_release=True)
        self.assertEqual((result["status"], lease.kills), ("failed", 1))
        self.assertEqual(events.count(("release", b"G")), 1)

    def test_positive_api_exact_binding_and_false_claims(self):
        base, loader = self.api()
        result = self.invoke()
        self.assertEqual((result["case"], result["expects_negative"], result["worker_exit_code"]),
                         ("positive", False, 0))
        self.assertTrue(all(result[key] is False for key in subject.frozen.NONCLAIMS))
        self.assertEqual(result["worker"], worker())
        self.assertIs(base.receive, subject.frozen._receive)
        loader.assert_called_once_with("/release/scripts/run_sft_cgroup_fixture_v1.py",
                                       subject.BASE_SHA256)
        base.run_case.assert_called_once_with("worker_run", 10001, 10001,
                                             subject.frozen.child_bootstrap.RUNTIME,
                                             90_000_000_000)

    def test_negative_api_uses_private_changed_env_and_validated_receipt(self):
        base, _ = self.api("wrong_entry")
        captured_context = []

        def guard(_base, context, case, *args):
            captured_context.append((context, case))

        with patch.object(subject, "_guard", side_effect=guard):
            base.run_case.side_effect = lambda *args: base.guardian("dummy") or final("wrong_entry")
            result = self.invoke("wrong_entry")
        context, case = captured_context[0]
        self.assertEqual(case, "wrong_entry")
        self.assertEqual(context["environment"]["LUMI_IMPORT_ENTRY_SHA256"], "0" * 64)
        self.assertEqual((result["expects_negative"], result["worker_exit_code"]), (True, 1))
        self.assertEqual(result["worker"], receipt("wrong_entry"))

    def test_api_wrong_report_exit_bool_denied(self):
        self.api("wrong_entry", {**final("wrong_entry"), "worker_exit_code": True})
        with self.assertRaisesRegex(subject.QualificationError, "worker_rejected"):
            self.invoke("wrong_entry")

    def test_native_preflight_before_io(self):
        with patch.object(subject.frozen.child_bootstrap, "_runtime", side_effect=ValueError), \
                patch.object(subject.frozen, "_load_base") as load:
            with self.assertRaisesRegex(subject.QualificationError, "runtime_rejected"):
                self.invoke()
            load.assert_not_called()
        self.assertFalse(subject._ATTEMPTED)

    def test_bad_cases_or_arguments_before_attempt(self):
        _, loader = self.api()
        for case in (True, "wrong_adapter", "", None):
            with self.assertRaisesRegex(subject.QualificationError, "case_rejected"):
                self.invoke(case)
        for override in ({"deadline_ns": True}, {"deadline_ns": 100},
                         {"deadline_ns": 91_000_000_000}, {"rootfs_path": "/owned/mount"},
                         {"root_identity": (1, 2, stat.S_IFDIR, 10001, 0)},
                         {"environment": {**environment(), "LUMI_IMPORT_ENTRY_SHA256": True}}):
            with self.assertRaises(subject.QualificationError):
                self.invoke(**override)
        loader.assert_not_called()
        self.assertFalse(subject._ATTEMPTED)

    def test_started_load_failure_consumes_attempt(self):
        _, loader = self.api()
        loader.side_effect = OSError("private path")
        for expected in ("guard_failed", "already_attempted"):
            with self.assertRaisesRegex(subject.QualificationError, expected):
                self.invoke()
        loader.assert_called_once()

    def test_success_cannot_run_second_case_same_process(self):
        base, _ = self.api("wrong_entry")
        self.invoke("wrong_entry")
        with self.assertRaisesRegex(subject.QualificationError, "already_attempted"):
            self.invoke("wrong_manifest")
        base.run_case.assert_called_once()

    def test_failed_case_reaps_then_fixed_error(self):
        base, _ = self.api()
        base.run_case.side_effect = OSError("private error")
        with self.assertRaisesRegex(subject.QualificationError, "guard_failed"):
            self.invoke()
        base.wait_statuses.assert_called_once()

    def test_cleanup_failure_and_late_deadline_cannot_pass(self):
        base, _ = self.api()
        base.run_case.side_effect = OSError("private error")
        base.wait_statuses.side_effect = OSError("private cleanup error")
        with self.assertRaisesRegex(subject.QualificationError, "cleanup_unconfirmed"):
            self.invoke()

    def test_late_deadline_denies_success(self):
        self.api()
        with patch.object(subject.time, "monotonic_ns",
                          side_effect=[100, 90_000_000_001, 90_000_000_002]):
            with self.assertRaisesRegex(subject.QualificationError, "deadline_exhausted"):
                self.invoke()


if __name__ == "__main__":
    unittest.main()
