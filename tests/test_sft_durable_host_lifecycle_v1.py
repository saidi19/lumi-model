"""Synthetic durable lifecycle tests; real I/O is confined to owned temp journals."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_test_sft_durable_host_lifecycle_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
durable = importlib.import_module(PACKAGE + ".posttraining_sft_durable_host_lifecycle_v1")
NS = 1_000_000_000
OWNED, PROTECTED = "a" * 32, "b" * 32
ATTEMPT = "c" * 64


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def plan(**changes):
    value = {
        "schema_version": durable.model.PLAN_SCHEMA, "mode": "local_decision_model_only",
        "attempt_id": ATTEMPT, "account_scope_sha256": "d" * 64,
        "approval_sha256": "e" * 64, "launch_request_sha256": "f" * 64,
        "protected_instance_ids": [PROTECTED], "region": "us-east-1",
        "instance_type": "gpu_8x_a100_80gb_sxm4", "gpu_count": 8,
        "total_host_hourly_rate_microusd": 22_320_000,
        "maximum_cost_microusd": 150_000_000, "noncompute_reserve_microusd": 10_000_000,
        "maximum_lifecycle_seconds": 14400, "shutdown_reserve_seconds": 600,
        "create_not_before_unix": 999, "create_expires_unix": 2000, "maximum_requests": 8,
    }
    return {**value, **changes}


def instance(status="active"):
    return {
        "id": OWNED, "status": status, "region": {"name": "us-east-1"},
        "instance_type": {"name": "gpu_8x_a100_80gb_sxm4", "price_cents_per_hour": 2232,
                          "specs": {"gpus": 8}},
        "name": "SECRET", "ip": "192.0.2.123", "provider_note": "SECRET",
    }


class Fixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lumi-durable-lifecycle-synthetic-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.directory.chmod(0o700)
        self.root_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.addCleanup(os.close, self.root_fd)
        self.now = 0
        self.clock = Mock(side_effect=lambda: self.now)
        self.wall = Mock(return_value=1000)
        self.raw_plan = encode(plan())
        self.plan_sha = hashlib.sha256(self.raw_plan).hexdigest()
        self.path = self.directory / (ATTEMPT + ".lifecycle.jsonl")

    def create(self, **kwargs):
        controller = durable.DurableLifecycle(
            self.root_fd, self.raw_plan, self.plan_sha,
            clock_ns=kwargs.get("clock_ns", self.clock),
            unix_seconds=kwargs.get("unix_seconds", self.wall),
        )
        self.addCleanup(controller.close)
        return controller

    def bound(self):
        controller = self.create()
        intent = controller.begin_create()
        self.now = 1
        controller.receive(intent.sequence, http_status=200,
                           body=encode({"data": {"instance_ids": [OWNED]}}))
        return controller

    def active(self):
        controller = self.bound()
        self.now = 2
        intent = controller.begin_observe()
        self.now = 3
        controller.receive(intent.sequence, http_status=200, body=encode({"data": instance()}))
        return controller

    def assert_controller_code(self, code, operation):
        with self.assertRaises(durable.ControllerError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)

    def assert_model_code(self, code, operation):
        with self.assertRaises(durable.model.LifecycleError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def records(self):
        return [json.loads(line) for line in self.path.read_bytes().splitlines()]

    def boundaries(self, controller):
        report = controller.report()
        for key in durable.model.NONCLAIMS:
            self.assertIs(report[key], False, key)
        for key in ("crash_replay_or_resume_enabled", "same_account_global_exclusivity_verified",
                    "storage_power_loss_durability_verified",
                    "journal_storage_encryption_verified"):
            self.assertIs(report[key], False, key)
        for key in ("provider_calls_executed", "external_network_bytes", "gpu_operations",
                    "private_artifact_reads"):
            self.assertEqual(report[key], 0, key)
        self.assertNotIn("private_artifact_writes", report)
        self.assertEqual(report["local_journal_write_capacity_reserved_bytes"],
                         report["journal"]["capacity_reserved_bytes"])
        for secret in (str(self.directory), OWNED, PROTECTED, ATTEMPT, "d" * 64, "e" * 64,
                       "f" * 64, "SECRET", "192.0.2.123", "Traceback"):
            self.assertNotIn(secret, json.dumps(report))
        self.assertTrue(report["model_snapshot_is_last_synced_record"])
        if report["controller_failed"] or report["controller_closed"]:
            self.assertFalse(report["simulated_work_window_open"])
        return report

    def assert_poisoned(self, controller, code):
        before = self.boundaries(controller)
        self.assertTrue(before["controller_failed"])
        self.assertEqual(before["controller_failure_code"], code)
        self.assertTrue(before["reconciliation_required"])
        with patch.object(controller._journal, "append",
                          wraps=controller._journal.append) as append:
            for operation in (controller.begin_create, controller.begin_observe,
                              controller.begin_terminate, controller.finish_work, controller.tick):
                self.assert_controller_code("controller_failed", operation)
            append.assert_not_called()
        self.assertEqual(controller.report()["journal_records_synced"],
                         before["journal_records_synced"])

    def after_append(self, controller, action):
        original = controller._journal.append

        def appended(raw):
            original(raw)
            action()

        return patch.object(controller._journal, "append", side_effect=appended)


class PersistenceTests(Fixture):
    def test_constructor_claims_and_syncs_header_without_borrowed_fd_ownership(self):
        controller = self.create()
        report = self.boundaries(controller)
        self.assertEqual(report["state"], "prepared")
        self.assertEqual(report["journal_records_synced"], 1)
        self.assertEqual(report["journal"]["records_reserved"], 1)
        self.assertEqual(report["journal"]["fsync_calls"], 3)
        self.assertEqual(report["local_journal_write_capacity_reserved_bytes"], 4096)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.records()[0]["event"], "initialized")
        self.wall.assert_not_called()
        controller.close()
        self.assertEqual(os.fstat(self.root_fd).st_mode & 0o777, 0o700)

    def test_bad_plan_or_clock_configuration_precedes_journal_claim(self):
        with patch.object(durable.journal, "Journal") as journal:
            self.assert_model_code("plan_rejected", lambda: durable.DurableLifecycle(
                self.root_fd, self.raw_plan, "0" * 64,
            ))
            for options in ({"clock_ns": 0}, {"unix_seconds": "SECRET"}):
                self.assert_controller_code("controller_rejected", lambda options=options:
                                            durable.DurableLifecycle(
                                                self.root_fd, self.raw_plan, self.plan_sha,
                                                **options,
                                            ))
            journal.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_create_intent_predebit_is_fsynced_before_return(self):
        controller = self.create()
        events = []
        transition = controller._model.begin_create
        append = controller._journal.append
        fsync = durable.journal.os.fsync

        def transitioned(**kwargs):
            events.append("model_transition")
            return transition(**kwargs)

        def appended(raw):
            record = json.loads(raw)
            self.assertEqual(record["snapshot"]["create_attempts_modeled"], 1)
            self.assertEqual(record["snapshot"]["request_reservations"], 1)
            self.assertTrue(record["snapshot"]["pending_request"])
            events.append("append")
            append(raw)
            events.append("append_complete")

        def synced(fd):
            fsync(fd)
            events.append("fsync")

        with (
            patch.object(controller._model, "begin_create", side_effect=transitioned),
            patch.object(controller._journal, "append", side_effect=appended),
            patch.object(durable.journal.os, "fsync", side_effect=synced),
        ):
            intent = controller.begin_create()
            events.append("intent_released")
        self.assertEqual(events, ["model_transition", "append", "fsync",
                                  "append_complete", "intent_released"])
        self.assertEqual(intent, ("create", 1, None, 15 * NS, 65536))
        self.assertEqual(self.wall.call_count, 2)
        self.assertEqual(self.boundaries(controller)["journal_records_synced"], 2)

    def test_raised_model_denial_is_persisted_and_synced_before_exception_escapes(self):
        controller = self.create()
        controller.begin_create()
        before = controller.report()["journal"]["fsync_calls"]
        self.assert_model_code("state_rejected", controller.begin_create)
        record = self.records()[-1]
        self.assertEqual((record["event"], record["error_code"]),
                         ("begin_create", "state_rejected"))
        self.assertIsNone(record["intent_kind"])
        report = self.boundaries(controller)
        self.assertFalse(report["controller_failed"])
        self.assertEqual(report["journal"]["fsync_calls"], before + 1)
        self.assertEqual(report["request_reservations"], 1)

    def test_persisted_poll_denial_does_not_poison_later_valid_observation(self):
        controller = self.active()
        self.now = 4
        self.assert_model_code("poll_too_soon", controller.begin_observe)
        self.assertEqual(self.records()[-1]["error_code"], "poll_too_soon")
        self.now = 15 * NS + 2
        intent = controller.begin_observe()
        self.assertEqual((intent.kind, intent.sequence, intent.instance_id), ("observe", 3, OWNED))
        self.assertFalse(self.boundaries(controller)["controller_failed"])

    def test_fsync_hook_sees_only_previous_durable_snapshot_and_no_work_permission(self):
        controller = self.bound()
        self.now = 2
        intent = controller.begin_observe()
        before = controller.report()
        reports = []
        fsync = durable.journal.os.fsync

        def syncing(fd):
            reports.append(controller.report())
            fsync(fd)

        self.now = 3
        with patch.object(durable.journal.os, "fsync", side_effect=syncing):
            result = controller.receive(intent.sequence, http_status=200,
                                        body=encode({"data": instance()}))
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["responses_consumed"], before["responses_consumed"])
        self.assertEqual(reports[0]["journal_tail_sha256"], before["journal_tail_sha256"])
        self.assertFalse(reports[0]["simulated_work_window_open"])
        self.assertTrue(result["simulated_work_window_open"])
        self.boundaries(controller)

    def test_complete_transcript_persists_all_events_without_provider_or_network_calls(self):
        with (
            patch.object(socket, "socket", side_effect=AssertionError("network")) as network,
            patch.object(subprocess, "Popen", side_effect=AssertionError("provider")) as process,
        ):
            controller = self.active()
            self.now = 4
            controller.finish_work()
            self.now = 5
            intent = controller.begin_terminate()
            self.now = 6
            controller.receive(intent.sequence, http_status=200, body=encode({
                "data": {"terminated_instances": [instance("terminated")]},
            }))
            self.assertFalse(controller.report()["terminal_observation_modeled"])
            self.now = 15 * NS + 2
            intent = controller.begin_observe()
            self.now += 1
            controller.receive(intent.sequence, http_status=200,
                               body=encode({"data": instance("terminated")}))
            report = controller.close()
            network.assert_not_called()
            process.assert_not_called()
        self.assertTrue(report["terminal_observation_modeled"])
        self.assertTrue(report["controller_closed"])
        self.assertFalse(report["controller_failed"])
        self.assertEqual([record["event"] for record in self.records()], [
            "initialized", "begin_create", "receive", "begin_observe", "receive", "finish_work",
            "begin_terminate", "receive", "begin_observe", "receive", "close",
        ])
        self.assertEqual(self.wall.call_count, 2)
        self.boundaries(controller)

    def test_report_is_a_detached_snapshot_without_clock_or_journal_io(self):
        controller = self.active()
        self.clock.reset_mock()
        with patch.object(controller._journal, "append") as append:
            snapshot = controller.report()
            snapshot["state"] = "SECRET"
            snapshot["journal"]["records_synced"] = 999
            self.assertNotEqual(controller.report()["state"], "SECRET")
            self.assertNotEqual(controller.report()["journal"]["records_synced"], 999)
            self.clock.assert_not_called()
            append.assert_not_called()


class FailureTests(Fixture):
    def test_write_failure_never_releases_intent_or_unsynced_predebit_and_is_sticky(self):
        controller = self.create()
        with patch.object(durable.journal.os, "write", side_effect=OSError("SECRET")) as write:
            self.assert_controller_code("journal_rejected", controller.begin_create)
            write.assert_called_once()
        self.assert_poisoned(controller, "journal_rejected")
        report = controller.report()
        self.assertEqual(report["create_attempts_modeled"], 0)
        self.assertEqual(report["request_reservations"], 0)
        self.assertEqual(report["journal_records_synced"], 1)
        self.assertEqual(report["journal"]["records_reserved"], 2)
        self.assertEqual(report["journal"]["capacity_reserved_bytes"], 8192)

    def test_partial_write_is_not_retried_and_keeps_consumed_attempt_prefix(self):
        controller = self.create()
        original = durable.journal.os.write

        def partial(fd, raw):
            return original(fd, raw[:len(raw) // 2])

        with patch.object(durable.journal.os, "write", side_effect=partial) as write:
            self.assert_controller_code("journal_rejected", controller.begin_create)
            write.assert_called_once()
        controller.close()
        raw = self.path.read_bytes()
        self.assertFalse(raw.endswith(b"\n"))
        self.assertEqual(len(raw.splitlines()), 2)
        self.assert_controller_code("controller_rejected", self.create)
        self.assertEqual(self.path.read_bytes(), raw)

    def test_fsync_failure_on_active_reply_never_exposes_unsynced_active_state(self):
        controller = self.bound()
        self.now = 2
        intent = controller.begin_observe()
        before = controller.report()
        self.now = 3
        with patch.object(durable.journal.os, "fsync", side_effect=OSError("SECRET")):
            self.assert_controller_code("journal_rejected", lambda: controller.receive(
                intent.sequence, http_status=200, body=encode({"data": instance()}),
            ))
        report = self.boundaries(controller)
        self.assertFalse(report["simulated_work_window_open"])
        self.assertEqual(report["responses_consumed"], before["responses_consumed"])
        self.assertEqual(report["journal_tail_sha256"], before["journal_tail_sha256"])
        self.assert_poisoned(controller, "journal_rejected")

    def test_fsync_failure_on_terminal_reply_never_exposes_unsynced_terminal_evidence(self):
        controller = self.bound()
        self.now = 2
        intent = controller.begin_observe()
        self.now = 3
        with patch.object(durable.journal.os, "fsync", side_effect=OSError("SECRET")):
            self.assert_controller_code("journal_rejected", lambda: controller.receive(
                intent.sequence, http_status=200, body=encode({"data": instance("terminated")}),
            ))
        self.assertFalse(self.boundaries(controller)["terminal_observation_modeled"])
        self.assert_poisoned(controller, "journal_rejected")

    def test_storage_failure_while_persisting_model_denial_overrides_unpersisted_denial(self):
        controller = self.create()
        controller.begin_create()
        with patch.object(durable.journal.os, "write", side_effect=OSError("SECRET")):
            self.assert_controller_code("journal_rejected", controller.begin_create)
        self.assert_poisoned(controller, "journal_rejected")
        self.assertEqual(controller.report()["request_reservations"], 1)

    def test_record_capacity_is_global_and_never_refunded_or_reset(self):
        controller = self.create()
        for _ in range(255):
            controller.tick()
        report = controller.report()
        self.assertEqual(report["journal_records_synced"], 256)
        self.assertEqual(report["local_journal_write_capacity_reserved_bytes"], 1_048_576)
        self.assert_controller_code("journal_rejected", controller.tick)
        self.assert_poisoned(controller, "journal_rejected")
        self.assertEqual(controller.report()["journal_records_synced"], 256)

    def test_invalid_preoperation_monotonic_clock_poison_happens_before_model_or_journal(self):
        controller = self.create()
        self.clock.side_effect = lambda: True
        with patch.object(controller._journal, "append") as append:
            self.assert_controller_code("clock_rejected", controller.begin_create)
            append.assert_not_called()
        self.assert_poisoned(controller, "clock_rejected")
        self.assertEqual(controller.report()["create_attempts_modeled"], 0)
        self.wall.assert_not_called()

    def test_clock_callable_exception_is_fixed_and_sticky(self):
        controller = self.create()
        self.clock.side_effect = RuntimeError("SECRET")
        self.assert_controller_code("clock_rejected", controller.tick)
        self.assert_poisoned(controller, "clock_rejected")

    def test_postsync_backward_clock_denies_intent_even_after_durable_record(self):
        controller = self.create()
        self.now = 10
        with self.after_append(controller, lambda: setattr(self, "now", 9)):
            self.assert_controller_code("clock_rejected", controller.begin_create)
        self.assertEqual(self.records()[-1]["event"], "begin_create")
        self.assertEqual(controller.report()["journal_records_synced"], 2)
        self.assert_poisoned(controller, "clock_rejected")

    def test_postsync_request_deadline_withholds_create_intent_without_retry(self):
        controller = self.create()
        with self.after_append(controller, lambda: setattr(self, "now", 15 * NS)):
            self.assert_controller_code("durability_deadline_rejected", controller.begin_create)
        self.assertEqual(controller.report()["create_attempts_modeled"], 1)
        self.assertTrue(controller.report()["pending_request"])
        self.assert_poisoned(controller, "durability_deadline_rejected")

    def test_postsync_expired_create_wall_window_withholds_already_recorded_intent(self):
        controller = self.create()
        self.wall.side_effect = [1000, 2000]
        self.assert_controller_code("authorization_window_closed", controller.begin_create)
        self.assertEqual(self.records()[-1]["intent_kind"], "create")
        self.assertEqual(self.wall.call_count, 2)
        self.assert_poisoned(controller, "authorization_window_closed")

    def test_postsync_hard_lifecycle_deadline_discards_return_and_forces_work_closed(self):
        controller = self.active()
        self.now = 10
        with self.after_append(controller, lambda: setattr(self, "now", 14400 * NS)):
            self.assert_controller_code("durability_deadline_rejected", controller.tick)
        self.assert_poisoned(controller, "durability_deadline_rejected")

    def test_postsync_work_shutdown_cutoff_discards_otherwise_active_snapshot(self):
        controller = self.active()
        self.now = 13800 * NS - 1
        with self.after_append(controller, lambda: setattr(self, "now", 13800 * NS)):
            self.assert_controller_code("durability_deadline_rejected", controller.tick)
        self.assertTrue(self.records()[-1]["snapshot"]["simulated_work_window_open"])
        self.assertFalse(self.boundaries(controller)["simulated_work_window_open"])
        self.assert_poisoned(controller, "durability_deadline_rejected")

    def test_reentrant_operation_cannot_release_outer_intent_even_if_callback_swallows_error(self):
        controller = self.create()
        original = controller._journal.append
        errors = []

        def reenter(raw):
            try:
                controller.tick()
            except durable.ControllerError as error:
                errors.append(error.code)
            original(raw)

        with patch.object(controller._journal, "append", side_effect=reenter) as append:
            self.assert_controller_code("controller_rejected", controller.begin_create)
            append.assert_called_once()
        self.assertEqual(errors, ["controller_rejected"])
        self.assert_poisoned(controller, "controller_rejected")

    def test_reentrant_close_is_sticky_and_does_not_close_inflight_descriptors(self):
        controller = self.create()
        with self.after_append(controller, controller.close):
            self.assert_controller_code("controller_rejected", controller.begin_create)
        self.assertFalse(controller.report()["controller_closed"])
        self.assertEqual(controller.report()["journal"]["close_calls"], 0)
        self.assert_poisoned(controller, "controller_rejected")

    def test_close_without_terminal_observation_preserves_reconciliation_obligation(self):
        controller = self.bound()
        result = controller.close()
        self.assertTrue(result["controller_closed"])
        self.assertFalse(result["controller_failed"])
        self.assertTrue(result["reconciliation_required"])
        self.assertFalse(result["terminal_observation_modeled"])
        self.assertFalse(result["simulated_work_window_open"])

    def test_out_of_order_response_is_durable_and_clears_pending_ownership(self):
        controller = self.bound()
        self.now = 2
        intent = controller.begin_observe()
        self.now = 3
        self.assert_model_code("response_out_of_order", lambda: controller.receive(
            intent.sequence + 1, http_status=200, body=encode({"data": instance()}),
        ))
        report = self.boundaries(controller)
        self.assertFalse(report["controller_failed"])
        self.assertFalse(report["pending_request"])
        self.assertTrue(report["reconciliation_required"])
        self.assertEqual(self.records()[-1]["error_code"], "response_out_of_order")
        self.now = 13800 * NS
        controller.tick()
        self.assert_model_code("state_rejected", controller.begin_terminate)


class ClosingAndClaimsTests(Fixture):
    def test_duplicate_live_or_closed_attempt_is_rejected_without_changing_marker(self):
        controller = self.create()
        raw = self.path.read_bytes()
        self.assert_controller_code("controller_rejected", self.create)
        self.assertEqual(self.path.read_bytes(), raw)
        controller.close()
        raw = self.path.read_bytes()
        self.assert_controller_code("controller_rejected", self.create)
        self.assertEqual(self.path.read_bytes(), raw)

    def test_empty_or_partial_crash_marker_is_never_reopened_or_truncated(self):
        self.path.write_bytes(b"")
        self.path.chmod(0o600)
        self.assert_controller_code("controller_rejected", self.create)
        self.assertEqual(self.path.read_bytes(), b"")
        self.path.write_bytes(b'{"synthetic_crash_prefix":')
        raw = self.path.read_bytes()
        self.assert_controller_code("controller_rejected", self.create)
        self.assertEqual(self.path.read_bytes(), raw)

    def test_close_record_fsync_precedes_descriptor_closes_and_second_close_is_noop(self):
        controller = self.active()
        events = []
        sync, close = durable.journal.os.fsync, durable.journal.os.close

        def synced(fd):
            sync(fd)
            events.append("fsync")

        def closed(fd):
            close(fd)
            events.append("close")

        with (
            patch.object(durable.journal.os, "fsync", side_effect=synced),
            patch.object(durable.journal.os, "close", side_effect=closed),
        ):
            first = controller.close()
            second = controller.close()
        self.assertEqual(events, ["fsync", "close", "close"])
        self.assertEqual(first, second)
        self.assertFalse(first["simulated_work_window_open"])
        self.assertEqual(self.records()[-1]["event"], "close")
        self.assert_controller_code("controller_closed", controller.tick)
        self.boundaries(controller)

    def test_close_failure_is_one_shot_and_does_not_erase_persisted_close_record(self):
        controller = self.create()
        original = durable.journal.os.close

        def failed(fd):
            # Release the test-only FD before injecting an uncertain close result.
            original(fd)
            raise OSError("SECRET")

        with patch.object(durable.journal.os, "close", side_effect=failed) as close:
            first = controller.close()
            second = controller.close()
        self.assertEqual(close.call_count, 2)
        self.assertEqual(first, second)
        self.assertTrue(first["controller_failed"])
        self.assertEqual(first["controller_failure_code"], "journal_rejected")
        self.assertEqual(self.records()[-1]["event"], "close")
        self.assertEqual(os.fstat(self.root_fd).st_mode & 0o777, 0o700)

    def test_close_after_storage_failure_never_appends_retry_or_close_record(self):
        controller = self.create()
        with patch.object(durable.journal.os, "write", side_effect=OSError("SECRET")):
            self.assert_controller_code("journal_rejected", controller.tick)
        with patch.object(controller._journal, "append",
                          wraps=controller._journal.append) as append:
            report = controller.close()
            append.assert_not_called()
        self.assertTrue(report["controller_closed"])
        self.assertEqual([item["event"] for item in self.records()], ["initialized"])

    def test_cleanup_does_not_consult_expired_create_wall_window(self):
        controller = self.bound()
        self.wall.side_effect = AssertionError("wall must not gate cleanup")
        self.now = 2
        intent = controller.begin_terminate()
        self.assertEqual(intent.instance_id, OWNED)
        self.now = 3
        controller.fail_pending(intent.sequence)
        controller.close()
        self.boundaries(controller)


class JournalEvidenceTests(Fixture):
    def evidence(self):
        controller = self.bound()
        controller.close()
        return self.path.read_bytes(), controller.report()["journal_tail_sha256"]

    def verify(self, raw, tail):
        return durable.verify_journal(raw, self.plan_sha, tail)

    def reject(self, raw, tail):
        self.assert_controller_code("journal_evidence_rejected", lambda: self.verify(raw, tail))

    def rebuild(self, records):
        previous = durable.ZERO
        lines = []
        for number, record in enumerate(records, 1):
            record["sequence"] = number
            record["previous_sha256"] = previous
            record.pop("sha256", None)
            record["sha256"] = durable.digest(record)
            previous = record["sha256"]
            lines.append(durable.canonical(record) + b"\n")
        return b"".join(lines), previous

    def test_closed_chain_is_only_structural_evidence_with_no_admission_or_replay(self):
        raw, tail = self.evidence()
        with (
            patch.object(os, "open", side_effect=AssertionError("audit filesystem")) as opened,
            patch.object(socket, "socket", side_effect=AssertionError("audit network")) as network,
            patch.object(subprocess, "Popen",
                         side_effect=AssertionError("audit dispatch")) as process,
        ):
            result = self.verify(raw, tail)
            for operation in (opened, network, process):
                operation.assert_not_called()
        self.assertTrue(result["structural_chain_verified"])
        self.assertTrue(result["final_close_record_present"])
        self.assertEqual(result["record_count"], 4)
        for key in ("dispatch_or_resume_authorized", "provider_termination_verified",
                    "cloud_billing_stopped", "execution_authorized", "production_eligible"):
            self.assertIs(result[key], False)
        for secret in (OWNED, PROTECTED, ATTEMPT, "SECRET", "192.0.2.123", str(self.directory)):
            self.assertNotIn(secret.encode(), raw)
            self.assertNotIn(secret, json.dumps(result))

    def test_tail_pin_detects_removed_suffix_and_prefix_has_no_resume_authority(self):
        raw, tail = self.evidence()
        lines = raw.splitlines(keepends=True)
        prefix = b"".join(lines[:2])
        self.reject(prefix, tail)
        result = self.verify(prefix, json.loads(lines[1])["sha256"])
        self.assertTrue(result["structural_chain_verified"])
        self.assertFalse(result["final_close_record_present"])
        self.assertFalse(result["dispatch_or_resume_authorized"])
        self.reject(raw[:-1], tail)
        self.reject(raw[:-10], tail)

    def test_external_plan_and_tail_pins_are_exact_and_not_guessed_from_file(self):
        raw, tail = self.evidence()
        for plan_pin, tail_pin in (("0" * 64, tail), (self.plan_sha, "0" * 64),
                                   (True, tail), (self.plan_sha, tail.upper())):
            self.assert_controller_code(
                "journal_evidence_rejected", lambda plan_pin=plan_pin, tail_pin=tail_pin:
                durable.verify_journal(raw, plan_pin, tail_pin),
            )

    def test_noncanonical_duplicate_oversized_or_nonbytes_audit_input_fails_closed(self):
        raw, tail = self.evidence()
        line = raw.splitlines(keepends=True)[0]
        duplicate = line.replace(b'{', b'{"sequence":1,', 1)
        for invalid in (b" " + raw, duplicate, raw.decode(), bytearray(raw), b"", b"x" * 1048577,
                        b'{"value":NaN}\n', b"\xff\n", line.rstrip(b"\n") + b"\r\n"):
            self.reject(invalid, tail)

    def test_tampered_counter_order_hash_or_raw_sensitive_field_is_rejected(self):
        raw, tail = self.evidence()
        records = [json.loads(line) for line in raw.splitlines()]
        tampered = copy.deepcopy(records)
        tampered[1]["snapshot"]["request_reservations"] = 0
        self.reject(b"\n".join(encode(record) for record in tampered) + b"\n", tail)
        mutations = (
            lambda items: items[1].update({"SECRET": "SECRET"}),
            lambda items: items[1]["snapshot"].update({"events": True}),
            lambda items: items[2]["snapshot"].update({"events": 0}),
            lambda items: items[1].update({"intent_kind": "terminate"}),
        )
        for mutate in mutations:
            changed = copy.deepcopy(records)
            mutate(changed)
            altered, new_tail = self.rebuild(changed)
            self.reject(altered, new_tail)

    def test_chain_cannot_change_bound_target_or_continue_after_close(self):
        raw, _tail = self.evidence()
        records = [json.loads(line) for line in raw.splitlines()]
        changed = copy.deepcopy(records)
        changed[-1]["target_sha256"] = "f" * 64
        self.reject(*self.rebuild(changed))
        changed = copy.deepcopy(records)
        changed.append(copy.deepcopy(changed[-1]))
        changed[-1]["event"] = "tick"
        self.reject(*self.rebuild(changed))


if __name__ == "__main__":
    unittest.main()
