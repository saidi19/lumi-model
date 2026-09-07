"""Synthetic-only service driver tests with fake DNS, sockets and TLS installed.

The frozen bridge fixture provides real owned temporary journals and fake network
entry points before any controller is created. Only virtual time advances here.
"""

from __future__ import annotations

import importlib
import json
import threading
import unittest
from unittest.mock import Mock, patch

from tests import test_sft_provider_controller_v1 as previous

service = importlib.import_module(previous.PACKAGE + ".posttraining_sft_controller_service_v1")
NS = previous.NS


class Control:
    def __init__(self, fixture, decision=None):
        self.fixture = fixture
        self.decision = decision or (lambda: "continue")
        self.timeouts = []

    def poll(self, timeout):
        if type(timeout) not in {int, float} or not 0 <= timeout <= 0.25:
            raise AssertionError("invalid synthetic timeout")
        self.timeouts.append(timeout)
        self.fixture.now += int(timeout * NS)
        return self.decision()


class ServiceTests(previous.Fixture):
    def make_driver(self, decision=None):
        self.controller = self.make()
        self.control = Control(self, decision)
        return self.controller

    def run_driver(self, **options):
        result = service.run_controller(self.controller, self.control,
                                        clock_ns=lambda: self.now, **options)
        self.assertEqual(service.validate_report(result), result)
        self.assertLessEqual(len(json.dumps(result).encode()), 4096)
        self.assertTrue(all(result[key] is False for key in service.FALSE_FIELDS))
        self.assertTrue(result["local_controller_closed"])
        return result

    def queue_create(self):
        self.queue({"data": {"instance_ids": [previous.OWNED]}})

    def queue_observe(self, status="active"):
        self.queue({"data": previous.instance(status)})

    def queue_terminate(self, status="terminating"):
        self.queue({"data": {"terminated_instances": [previous.instance(status)]}})

    def stop_after_observe(self):
        return "stop" if self.controller.report()["poll_attempts_modeled"] else "continue"

    def happy(self):
        self.make_driver(self.stop_after_observe)
        self.queue_create()
        self.queue_observe()
        self.queue_terminate()
        self.queue_observe("terminated")

    def test_normal_stop_runs_only_reserved_cleanup_and_no_durable_ticks(self):
        self.happy()
        with patch.object(self.controller, "tick", wraps=self.controller.tick) as ticks:
            result = self.run_driver()
        self.assertEqual(result["status"], "terminal_modeled")
        self.assertEqual(result["stop_code"], "stop_requested")
        self.assertEqual(result["provider_requests"], 4)
        self.assertEqual(result["service_actions"], 4)
        self.assertEqual(result["journal_records"], 10)
        self.assertGreaterEqual(self.now, 15 * NS)
        ticks.assert_not_called()
        self.assertEqual(self.dns.call_count, 4)

    def test_stop_before_create_uses_zero_requests(self):
        self.make_driver(lambda: "stop")
        result = self.run_driver()
        self.assertEqual(result["provider_requests"], 0)
        self.assertEqual(result["service_actions"], 0)
        self.assertEqual(result["journal_records"], 2)
        self.dns.assert_not_called()

    def test_parent_eof_before_create_uses_zero_requests(self):
        self.make_driver(lambda: "eof")
        result = self.run_driver()
        self.assertEqual(result["stop_code"], "parent_eof")
        self.assertEqual(result["provider_requests"], 0)

    def test_signal_before_create_uses_zero_requests(self):
        self.make_driver(lambda: "signal")
        result = self.run_driver()
        self.assertEqual(result["stop_code"], "signal_requested")
        self.dns.assert_not_called()

    def test_control_exception_before_create_is_redacted(self):
        self.make_driver()
        self.control.poll = Mock(side_effect=RuntimeError("SECRET CONTROL"))
        result = self.run_driver()
        self.assertEqual(result["stop_code"], "control_fault")
        self.assertNotIn("SECRET", repr(result))
        self.dns.assert_not_called()

    def test_invalid_stop_command_cannot_select_an_operation(self):
        self.make_driver(lambda: {"terminate": previous.PROTECTED})
        result = self.run_driver()
        self.assertEqual(result["stop_code"], "control_fault")
        self.dns.assert_not_called()

    def test_eof_after_create_cleans_up_without_first_active_get(self):
        self.make_driver(lambda: "eof" if self.controller.report()["create_attempts_modeled"]
                         else "continue")
        self.queue_create()
        self.queue_terminate()
        self.queue_observe("terminated")
        result = self.run_driver()
        self.assertEqual(result["stop_code"], "parent_eof")
        self.assertEqual(result["provider_requests"], 3)
        self.assertTrue(result["modeled_terminal"])

    def test_repeated_signal_after_create_terminates_only_once(self):
        self.make_driver(lambda: "signal" if self.controller.report()["create_attempts_modeled"]
                         else "continue")
        self.queue_create()
        self.queue_terminate()
        self.queue_observe("terminating")
        self.queue_observe("terminated")
        result = self.run_driver()
        self.assertEqual(result["provider_requests"], 4)
        self.assertEqual(self.controller.report()["termination_attempts_modeled"], 1)

    def test_failed_observation_proceeds_to_one_termination(self):
        self.make_driver()
        self.queue_create()
        self.queue(b"SECRET BAD RECEIPT")
        self.queue_terminate()
        self.queue_observe("terminated")
        result = self.run_driver()
        self.assertTrue(result["modeled_terminal"])
        self.assertEqual(result["provider_requests"], 4)

    def test_transport_failed_observation_proceeds_to_termination(self):
        self.make_driver()
        self.queue_create()
        self.queue({}, status=b"503 SECRET")
        self.queue_terminate()
        self.queue_observe("terminated")
        # The failed HTTP parser does not consume its artificial EOF fragment.
        receive = self.tls.recv.side_effect
        skipped = False

        def recv(size):
            nonlocal skipped
            if not skipped and self.dns.call_count == 3 and self.fragments[:1] == [b""]:
                self.fragments.pop(0)
                skipped = True
            return receive(size)

        self.tls.recv.side_effect = recv
        result = self.run_driver()
        self.assertTrue(result["modeled_terminal"])
        self.assertEqual(result["provider_requests"], 4)

    def test_failed_termination_is_not_retried(self):
        self.make_driver(self.stop_after_observe)
        self.queue_create()
        self.queue_observe()
        self.queue(b"SECRET FAILED TERMINATE")
        self.queue_observe("terminated")
        result = self.run_driver()
        self.assertTrue(result["modeled_terminal"])
        self.assertEqual(self.controller.report()["termination_attempts_modeled"], 1)

    def test_ambiguous_create_stops_without_guessing_target(self):
        self.make_driver()
        self.queue({"data": {"instance_ids": [previous.OWNED, previous.OTHER]}})
        result = self.run_driver()
        self.assertEqual(result["provider_requests"], 1)
        self.assertFalse(result["modeled_terminal"])
        self.assertTrue(result["reconciliation_required"])
        self.assertEqual(self.dns.call_count, 1)

    def test_protected_create_target_stays_reconciliation(self):
        self.make_driver()
        self.queue({"data": {"instance_ids": [previous.PROTECTED]}})
        result = self.run_driver()
        self.assertEqual(result["provider_requests"], 1)
        self.assertTrue(result["reconciliation_required"])

    def test_request_ceiling_preserves_cleanup_and_journal_room(self):
        self.rebind(maximum_requests=64)
        self.make_driver()
        self.queue_create()
        for _ in range(61):
            self.queue_observe("booting")
        self.queue_terminate()
        self.queue_observe("terminated")
        result = self.run_driver()
        self.assertEqual(result["provider_requests"], 64)
        self.assertEqual(result["service_actions"], 65)
        self.assertEqual(result["journal_records"], 131)
        self.assertTrue(result["modeled_terminal"])

    def test_cutoff_triggers_cleanup_while_waiting_without_tick(self):
        self.rebind(maximum_lifecycle_seconds=61, shutdown_reserve_seconds=60)
        self.make_driver()
        self.queue_create()
        self.queue_observe("booting")
        self.queue_terminate()
        self.queue_observe("terminated")
        result = self.run_driver()
        self.assertEqual(result["stop_code"], "shutdown_cutoff")
        self.assertTrue(result["modeled_terminal"])
        self.assertEqual(result["journal_records"], 10)

    def test_hard_deadline_never_guesses_late_cleanup(self):
        self.rebind(maximum_lifecycle_seconds=61, shutdown_reserve_seconds=60)
        self.make_driver()
        self.queue_create()
        self.queue_observe()
        original = self.control.poll

        def poll(timeout):
            answer = original(timeout)
            if timeout:
                self.now = 61 * NS
            return answer

        self.control.poll = poll
        result = self.run_driver()
        self.assertEqual(result["stop_code"], "hard_deadline")
        self.assertEqual(result["provider_requests"], 2)
        self.assertTrue(result["reconciliation_required"])

    def test_termination_acknowledgment_does_not_replace_terminal_get(self):
        self.make_driver(self.stop_after_observe)
        self.queue_create()
        self.queue_observe()
        self.queue_terminate("terminated")
        self.queue_observe("terminated")
        result = self.run_driver()
        self.assertEqual(result["provider_requests"], 4)

    def test_preempted_is_not_terminal(self):
        self.make_driver()
        self.queue_create()
        self.queue_observe("preempted")
        self.queue_terminate()
        self.queue_observe("terminated")
        result = self.run_driver()
        self.assertEqual(result["provider_requests"], 4)

    def test_wrong_target_observation_still_only_cleans_bound_target(self):
        self.make_driver()
        self.queue_create()
        self.queue({"data": previous.instance(identifier=previous.OTHER)})
        self.queue_terminate()
        self.queue_observe("terminated")
        result = self.run_driver()
        self.assertTrue(result["modeled_terminal"])
        request_bytes = [call.args[0] for call in self.tls.send.call_args_list]
        self.assertNotIn(previous.OTHER.encode(), b"".join(request_bytes))

    def test_service_is_single_use(self):
        self.make_driver(lambda: "stop")
        self.run_driver()
        with patch.object(self.controller, "close", wraps=self.controller.close) as close:
            result = service.run_controller(self.controller, self.control,
                                            clock_ns=lambda: self.now)
        self.assertEqual(result["stop_code"], "service_rejected")
        close.assert_not_called()
        self.dns.assert_not_called()

    def test_wrong_thread_cannot_consume_or_close_controller(self):
        self.make_driver(lambda: "stop")
        reports = []
        thread = threading.Thread(target=lambda: reports.append(service.run_controller(
            self.controller, self.control, clock_ns=lambda: self.now)))
        thread.start()
        thread.join()
        self.assertEqual(reports[0]["stop_code"], "service_rejected")
        self.assertFalse(self.controller.report()["controller_closed"])
        self.dns.assert_not_called()

    def test_invalid_controller_type_does_not_call_it(self):
        controller = Mock()
        result = service.run_controller(controller, Mock(), clock_ns=lambda: 0)
        self.assertEqual(result["stop_code"], "service_rejected")
        self.assertEqual(controller.mock_calls, [])

    def test_already_used_controller_rejects_and_closes_locally(self):
        self.make_driver()
        self.queue_create()
        self.controller.create()
        result = self.run_driver()
        self.assertEqual(result["stop_code"], "service_rejected")
        self.assertEqual(result["provider_requests"], 1)

    def test_clock_regression_stops_before_network(self):
        self.make_driver()
        values = iter((2, 1))
        result = service.run_controller(self.controller, self.control,
                                        clock_ns=lambda: next(values))
        self.assertEqual(result["stop_code"], "clock_rejected")
        self.dns.assert_not_called()

    def test_boolean_clock_rejected(self):
        self.make_driver()
        result = service.run_controller(self.controller, self.control, clock_ns=lambda: True)
        self.assertEqual(result["stop_code"], "clock_rejected")
        self.dns.assert_not_called()

    def test_clock_exception_is_redacted(self):
        self.make_driver()
        result = service.run_controller(self.controller, self.control,
                                        clock_ns=Mock(side_effect=ValueError("SECRET CLOCK")))
        self.assertEqual(result["stop_code"], "clock_rejected")
        self.assertNotIn("SECRET", repr(result))

    def test_control_poll_cap_does_not_write_tick_records(self):
        self.make_driver()
        self.queue_create()
        self.queue_observe()
        with patch.object(service, "MAXIMUM_CONTROL_POLLS", 5):
            result = self.run_driver()
        self.assertEqual(result["stop_code"], "poll_limit")
        self.assertEqual(result["control_polls"], 5)
        self.assertEqual(result["journal_records"], 6)

    def test_action_cap_stops_without_extra_dispatch(self):
        self.make_driver(self.stop_after_observe)
        self.queue_create()
        self.queue_observe()
        with patch.object(service, "MAXIMUM_ACTIONS", 2):
            result = self.run_driver()
        self.assertEqual(result["stop_code"], "action_limit")
        self.assertEqual(result["provider_requests"], 2)

    def test_early_poll_denial_waits_again_instead_of_spin(self):
        self.happy()
        observe = self.controller.observe
        attempts = 0

        def first_denial():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise previous.bridge.ProviderError("poll_too_soon")
            return observe()

        with patch.object(self.controller, "observe", side_effect=first_denial):
            result = self.run_driver()
        self.assertTrue(result["modeled_terminal"])
        self.assertEqual(result["service_actions"], 5)
        self.assertGreaterEqual(self.now, 30 * NS)

    def test_unknown_provider_error_never_retries(self):
        self.make_driver()
        with patch.object(self.controller, "create", side_effect=previous.bridge.ProviderError(
            "intent_rejected",
        )):
            result = self.run_driver()
        self.assertEqual(result["stop_code"], "controller_failed")
        self.assertEqual(result["service_actions"], 1)
        self.dns.assert_not_called()

    def test_close_failure_cannot_report_success(self):
        self.happy()
        with patch.object(self.controller, "close", side_effect=OSError("SECRET CLOSE")):
            result = service.run_controller(self.controller, self.control,
                                            clock_ns=lambda: self.now)
        self.assertEqual(result["status"], "reconciliation_required")
        self.assertFalse(result["local_controller_closed"])
        self.assertNotIn("SECRET", repr(result))

    def test_report_excludes_source_ids_tokens_and_receipts(self):
        self.happy()
        result = self.run_driver()
        for secret in (previous.OWNED, previous.PROTECTED, previous.TOKEN.decode(), "SECRET"):
            self.assertNotIn(secret, repr(result))

    def test_control_blocking_time_is_sampled_after_return(self):
        self.make_driver()
        self.control.poll = lambda _timeout: setattr(self, "now", 14400 * NS) or "continue"
        result = self.run_driver()
        self.assertEqual(result["stop_code"], "hard_deadline")
        self.dns.assert_not_called()

    def test_final_report_validator_rejects_all_unapproved_claims(self):
        self.make_driver(lambda: "stop")
        result = self.run_driver()
        for name in service.FALSE_FIELDS:
            with self.subTest(name=name), self.assertRaises(service.ServiceError):
                service.validate_report({**result, name: True})

    def test_final_report_validator_rejects_extra_fields_and_bad_counts(self):
        self.make_driver(lambda: "stop")
        result = self.run_driver()
        bad = [{**result, "id": previous.OWNED}, {**result, "stop_code": "SECRET"},
               {**result, "status": "success"}, {**result, "reconciliation_required": 1}]
        for name, maximum in service.REPORT_COUNTS.items():
            bad.extend(({**result, name: True}, {**result, name: -1},
                        {**result, name: maximum + 1}))
        for value in bad:
            with self.subTest(value=list(value)), self.assertRaises(service.ServiceError):
                service.validate_report(value)

    def test_final_report_validator_requires_terminal_consistency(self):
        self.make_driver(lambda: "stop")
        result = self.run_driver()
        with self.assertRaises(service.ServiceError):
            service.validate_report({**result, "status": "terminal_modeled"})
        with self.assertRaises(service.ServiceError):
            service.validate_report({**result, "reconciliation_required": False})


if __name__ == "__main__":
    unittest.main()
