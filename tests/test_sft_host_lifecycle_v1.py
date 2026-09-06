"""Synthetic, transport-free lifecycle and whole-host budget decision tests."""

from __future__ import annotations

import builtins
import copy
import hashlib
import importlib
import json
import os
import socket
import subprocess
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_test_sft_host_lifecycle_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
lifecycle = importlib.import_module(PACKAGE + ".posttraining_sft_host_lifecycle_v1")
NS = 1_000_000_000
OWNED = "a" * 32
PROTECTED = "b" * 32
FOREIGN = "c" * 32
PERMISSIONS = (
    "provider_transport_authenticated", "provider_termination_verified", "cloud_billing_stopped",
    "provider_invoice_limit_enforced", "durable_attempt_exclusivity_verified",
    "external_controller_survival_verified", "target_gpu_host_verified", "execution_authorized",
    "real_sft_execution_permitted", "production_eligible", "full_private_integration_passed",
)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def plan(**changes):
    value = {
        "schema_version": lifecycle.PLAN_SCHEMA, "mode": "local_decision_model_only",
        "attempt_id": "c" * 64, "account_scope_sha256": "d" * 64,
        "approval_sha256": "e" * 64, "launch_request_sha256": "f" * 64,
        "protected_instance_ids": [PROTECTED], "region": "us-east-1",
        "instance_type": "gpu_8x_a100_80gb_sxm4", "gpu_count": 8,
        "total_host_hourly_rate_microusd": 22_320_000,
        "maximum_cost_microusd": 150_000_000, "noncompute_reserve_microusd": 10_000_000,
        "maximum_lifecycle_seconds": 14400, "shutdown_reserve_seconds": 600,
        "create_not_before_unix": 999, "create_expires_unix": 2000, "maximum_requests": 8,
    }
    return {**value, **changes}


def model(**changes):
    raw = encode(plan(**changes))
    return lifecycle.LifecycleModel(raw, hashlib.sha256(raw).hexdigest())


def instance(status="active", identifier=OWNED, **changes):
    value = {
        "id": identifier, "status": status, "region": {"name": "us-east-1"},
        "instance_type": {
            "name": "gpu_8x_a100_80gb_sxm4", "price_cents_per_hour": 2232, "specs": {"gpus": 8},
        },
        "name": "SECRET", "ip": "192.0.2.123", "provider_note": "SECRET",
    }
    return {**value, **changes}


def launch(identifier=OWNED):
    return encode({"data": {"instance_ids": [identifier]}})


def observed(status="active", identifier=OWNED, **changes):
    return encode({"data": instance(status, identifier, **changes)})


def terminated(status="terminating", identifier=OWNED):
    return encode({"data": {"terminated_instances": [instance(status, identifier)]}})


def bound(**changes):
    value = model(**changes)
    intent = value.begin_create(now_ns=0, now_unix=1000)
    value.receive(intent.sequence, http_status=200, body=launch(), now_ns=1)
    return value


def active(**changes):
    value = bound(**changes)
    intent = value.begin_observe(now_ns=2)
    value.receive(intent.sequence, http_status=200, body=observed(), now_ns=3)
    return value


class Assertions(unittest.TestCase):
    def assert_code(self, code, operation):
        with self.assertRaises(lifecycle.LifecycleError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)

    def boundaries(self, value):
        report = value.report()
        self.assertEqual(report["mode"], "local_decision_model_only")
        for key in PERMISSIONS:
            self.assertIs(report[key], False, key)
        for key in ("provider_calls_executed", "external_network_bytes", "gpu_operations",
                    "private_artifact_reads", "private_artifact_writes"):
            self.assertEqual(report[key], 0, key)
        raw = json.dumps(report)
        for secret in (OWNED, PROTECTED, FOREIGN, "SECRET", "192.0.2.123", "Traceback",
                       "d" * 64, "e" * 64, "f" * 64, "us-east-1", "gpu_8x_a100_80gb_sxm4"):
            self.assertNotIn(secret, raw)
        self.assertEqual(report["response_capacity_reserved_bytes"],
                         report["request_reservations"] * 65536)
        return report


class PlanTests(Assertions):
    def parse(self, value):
        raw = encode(value)
        return lifecycle.parse_plan(raw, hashlib.sha256(raw).hexdigest())

    def test_fixed_plan_keeps_whole_host_quote_and_copies_protected_ids_to_tuple(self):
        original = plan()
        before = copy.deepcopy(original)
        parsed = self.parse(original)
        self.assertEqual(parsed["protected_instance_ids"], (PROTECTED,))
        self.assertEqual(original, before)
        self.assertEqual(parsed["total_host_hourly_rate_microusd"], 22_320_000)
        self.assertEqual(parsed["maximum_lifecycle_seconds"], 14400)
        self.assertEqual(self.boundaries(model())["state"], "prepared")

    def test_pin_must_be_exact_hash_of_exact_bounded_byte_string(self):
        raw = encode(plan())
        for digest in ("0" * 64, hashlib.sha256(raw).hexdigest().upper(), "x", True, None):
            with self.subTest(digest=digest):
                self.assert_code("plan_rejected",
                                 lambda digest=digest: lifecycle.parse_plan(raw, digest))
        for value in (raw.decode(), bytearray(raw), b"", b" " * 16385):
            digest = hashlib.sha256(value if type(value) is bytes else raw).hexdigest()
            self.assert_code("plan_rejected",
                             lambda value=value, digest=digest: lifecycle.parse_plan(value, digest))

    def test_duplicate_keys_nonfinite_deep_invalid_and_nonobject_json_are_rejected(self):
        raw = encode(plan())
        invalid = (
            b'{"mode":"local_decision_model_only",' + raw[1:],
            b'{"value":NaN}', b'{"value":Infinity}', b"[]", b"null", b"\xff", b"{",
            b"[" * 1200 + b"0" + b"]" * 1200,
        )
        for raw in invalid:
            with self.subTest(length=len(raw)):
                self.assert_code("plan_rejected", lambda raw=raw: lifecycle.parse_plan(
                    raw, hashlib.sha256(raw).hexdigest(),
                ))

    def test_plan_fields_cannot_be_added_or_removed(self):
        extra = plan(SECRET="SECRET")
        self.assert_code("plan_rejected", lambda: self.parse(extra))
        for key in plan():
            incomplete = plan()
            del incomplete[key]
            with self.subTest(missing=key):
                self.assert_code("plan_rejected", lambda value=incomplete: self.parse(value))

    def test_plan_strict_identity_mode_and_label_fields(self):
        cases = {
            "mode": ["live", True], "schema_version": ["wrong", None],
            "attempt_id": ["a" * 63, "A" * 64], "account_scope_sha256": ["x" * 64, True],
            "approval_sha256": [None, "e" * 64 + "\n"], "launch_request_sha256": ["f" * 65],
            "region": ["us east 1", "../SECRET", "a" * 65, "US-EAST-1"],
            "instance_type": ["type/name", "", 1],
        }
        for key, values in cases.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    self.assert_code("plan_rejected",
                                     lambda key=key, value=value: self.parse(plan(**{key: value})))

    def test_protected_ids_are_required_unique_bounded_and_canonical(self):
        for values in ([], [PROTECTED, PROTECTED], ["B" * 32], ["b" * 31], [True],
                       [f"{index:032x}" for index in range(129)]):
            with self.subTest(count=len(values)):
                self.assert_code(
                    "plan_rejected",
                    lambda values=values: self.parse(plan(protected_instance_ids=values)),
                )

    def test_strict_integer_limits_reject_bools_floats_strings_and_out_of_range(self):
        ranges = {
            "gpu_count": (0, 9), "maximum_requests": (3, 65),
            "maximum_lifecycle_seconds": (60, 14401), "shutdown_reserve_seconds": (59, 3601),
            "maximum_cost_microusd": (0, 150000001), "noncompute_reserve_microusd": (-1, 150000001),
            "total_host_hourly_rate_microusd": (0, 150000001),
            "create_not_before_unix": (0, 2**53), "create_expires_unix": (999, 2**53),
        }
        for key, outside in ranges.items():
            for value in (True, False, 1.5, "1000", None, *outside):
                with self.subTest(key=key, value=value):
                    self.assert_code("plan_rejected",
                                     lambda key=key, value=value: self.parse(plan(**{key: value})))

    def test_shutdown_reserve_and_quote_cost_must_fit_entire_lifecycle(self):
        for changes in (
            {"maximum_lifecycle_seconds": 600, "shutdown_reserve_seconds": 600},
            {"maximum_cost_microusd": 99_279_999},
            {"total_host_hourly_rate_microusd": 22_320_001},
            {"noncompute_reserve_microusd": 150_000_000},
        ):
            with self.subTest(changes=changes):
                self.assert_code("plan_rejected",
                                 lambda changes=changes: self.parse(plan(**changes)))
        self.parse(plan(maximum_cost_microusd=99_280_000))

    def test_projected_cost_rounds_minutes_up_and_uses_total_host_rate_once(self):
        expected = ((0, 0), (1, 372000), (60 * NS, 372000), (60 * NS + 1, 744000),
                    (3600 * NS, 22320000), (14400 * NS, 89280000))
        for elapsed, cost in expected:
            self.assertEqual(lifecycle.projected_cost(elapsed, 22320000), cost)
        self.assertEqual(lifecycle.projected_cost(1, 1), 1)
        for elapsed, rate in ((True, 1), (-1, 1), (1.0, 1), (1, True), (1, 0), (1, 150000001)):
            self.assert_code("plan_rejected", lambda elapsed=elapsed, rate=rate:
                             lifecycle.projected_cost(elapsed, rate))


class CreationTests(Assertions):
    def test_create_intent_is_immutable_precharged_and_single_use(self):
        value = model()
        intent = value.begin_create(now_ns=0, now_unix=1000)
        self.assertEqual(intent, ("create", 1, None, 15 * NS, 65536))
        with self.assertRaises(AttributeError):
            intent.kind = "terminate"
        report = self.boundaries(value)
        self.assertEqual(report["create_attempts_modeled"], 1)
        self.assertEqual(report["request_reservations"], 1)
        self.assertTrue(report["pending_request"])
        self.assert_code("state_rejected", lambda: value.begin_create(now_ns=1, now_unix=1000))
        self.assertEqual(value.report()["request_reservations"], 1)

    def test_window_boundaries_are_exact_and_invalid_attempt_stays_closed(self):
        value = model()
        value.begin_create(now_ns=0, now_unix=999)
        for wall in (998, 2000, True, 1000.0, None):
            value = model()
            self.assert_code(
                "authorization_window_closed",
                lambda wall=wall, value=value: value.begin_create(now_ns=0, now_unix=wall),
            )
            self.assert_code("state_rejected",
                             lambda value=value: value.begin_create(now_ns=1, now_unix=1000))
            report = self.boundaries(value)
            self.assertEqual(report["request_reservations"], 0)
            self.assertEqual(report["create_attempts_modeled"], 0)

    def test_only_exact_single_returned_unprotected_id_can_bind(self):
        value = model()
        intent = value.begin_create(now_ns=0, now_unix=1000)
        value.receive(intent.sequence, http_status=200, body=launch(FOREIGN), now_ns=1)
        self.assertEqual(value.begin_observe(now_ns=2).instance_id, FOREIGN)
        self.boundaries(value)
        for body in (launch(PROTECTED), encode({"data": {"instance_ids": [OWNED, FOREIGN]}}),
                     encode({"data": {"instance_ids": []}}), b"SECRET"):
            value = model()
            intent = value.begin_create(now_ns=0, now_unix=1000)
            value.receive(intent.sequence, http_status=200, body=body, now_ns=1)
            report = self.boundaries(value)
            self.assertFalse(report["instance_bound_in_model"])
            self.assertTrue(report["reconciliation_required"])
            self.assert_code("state_rejected", lambda value=value: value.begin_terminate(now_ns=2))
            self.assert_code("state_rejected",
                             lambda value=value: value.begin_create(now_ns=3, now_unix=1000))

    def test_failed_or_ambiguous_create_is_consumed_never_retried(self):
        for action in ("failure", "http", "timeout", "late"):
            value = model()
            intent = value.begin_create(now_ns=0, now_unix=1000)
            if action == "failure":
                value.fail_pending(intent.sequence, now_ns=1)
            elif action == "http":
                value.receive(intent.sequence, http_status=500, body=launch(), now_ns=1)
            elif action == "timeout":
                value.tick(now_ns=intent.deadline_ns)
            else:
                value.receive(intent.sequence, http_status=200, body=launch(),
                              now_ns=intent.deadline_ns)
            report = self.boundaries(value)
            self.assertTrue(report["reconciliation_required"])
            self.assertFalse(report["instance_bound_in_model"])
            self.assertEqual(report["request_reservations"], 1)
            self.assertEqual(report["create_attempts_modeled"], 1)
            self.assert_code(
                "state_rejected", lambda value=value, intent=intent: value.begin_create(
                    now_ns=intent.deadline_ns + 1, now_unix=1000,
                ),
            )

    def test_http_status_must_be_exact_integer_200_even_if_reply_claims_success(self):
        for status in (True, 200.0, "200", 201, 404, 429, 500):
            value = model()
            intent = value.begin_create(now_ns=0, now_unix=1000)
            value.receive(intent.sequence, http_status=status, body=launch(), now_ns=1)
            self.assertEqual(self.boundaries(value)["first_fault"], "request_failed")

    def test_oversized_reply_consumes_reserved_capacity_without_binding_or_refund(self):
        value = model()
        intent = value.begin_create(now_ns=0, now_unix=1000)
        value.receive(intent.sequence, http_status=200, body=b"x" * 65537, now_ns=1)
        report = self.boundaries(value)
        self.assertEqual(report["first_fault"], "response_rejected")
        self.assertEqual(report["responses_consumed"], 1)
        self.assertEqual(report["response_capacity_reserved_bytes"], 65536)
        self.assertFalse(report["instance_bound_in_model"])

    def test_bound_id_without_matching_active_observation_does_not_open_work_window(self):
        value = bound()
        self.assertFalse(self.boundaries(value)["simulated_work_window_open"])
        intent = value.begin_observe(now_ns=2)
        value.receive(intent.sequence, http_status=200, body=observed("booting"), now_ns=3)
        self.assertFalse(self.boundaries(value)["simulated_work_window_open"])
        self.assertTrue(self.boundaries(active())["simulated_work_window_open"])

    def test_identity_region_type_gpu_or_total_price_mismatch_requires_exact_owned_cleanup(self):
        cases = (
            observed(identifier=FOREIGN), observed(region={"name": "different-region"}),
            observed(instance_type={"name": "other", "price_cents_per_hour": 2232,
                                    "specs": {"gpus": 8}}),
            observed(instance_type={"name": "gpu_8x_a100_80gb_sxm4", "price_cents_per_hour": 2232,
                                    "specs": {"gpus": 1}}),
            observed(instance_type={"name": "gpu_8x_a100_80gb_sxm4", "price_cents_per_hour": 279,
                                    "specs": {"gpus": 8}}),
        )
        for body in cases:
            value = bound()
            intent = value.begin_observe(now_ns=2)
            value.receive(intent.sequence, http_status=200, body=body, now_ns=3)
            report = self.boundaries(value)
            self.assertEqual(report["first_fault"], "target_mismatch")
            self.assertFalse(report["simulated_work_window_open"])
            self.assertEqual(value.begin_terminate(now_ns=4).instance_id, OWNED)


class BudgetAndOrderingTests(Assertions):
    def test_lifecycle_clock_includes_provisioning_and_does_not_reset_on_binding(self):
        value = model()
        intent = value.begin_create(now_ns=100 * NS, now_unix=1000)
        value.receive(intent.sequence, http_status=200, body=launch(), now_ns=101 * NS)
        report = value.tick(now_ns=160 * NS + 1)
        self.assertEqual(report["elapsed_lifecycle_ns"], 60 * NS + 1)
        self.assertEqual(report["compute_projection_microusd"], 744000)
        self.assertEqual(report["projection_with_reserve_microusd"], 10744000)
        self.boundaries(value)

    def test_shutdown_reserve_closes_work_and_preserves_one_termination_dispatch(self):
        value = active()
        report = value.tick(now_ns=13800 * NS - 1)
        self.assertTrue(report["simulated_work_window_open"])
        report = value.tick(now_ns=13800 * NS)
        self.assertEqual(report["stop_reason"], "shutdown_reserve_reached")
        self.assertFalse(report["simulated_work_window_open"])
        self.assert_code("termination_required", lambda: value.begin_observe(now_ns=13800 * NS))
        intent = value.begin_terminate(now_ns=13800 * NS + 1)
        self.assertEqual((intent.kind, intent.instance_id), ("terminate", OWNED))
        self.boundaries(value)

    def test_hard_deadline_closes_all_further_intents_without_claiming_billing_stopped(self):
        value = active()
        report = value.tick(now_ns=14400 * NS)
        self.assertTrue(report["reconciliation_required"])
        self.assertTrue(report["lifecycle_deadline_breached"])
        self.assertEqual(report["first_fault"], "lifecycle_deadline_exceeded")
        self.assertEqual(report["compute_projection_microusd"], 89280000)
        self.assert_code("state_rejected", lambda: value.begin_terminate(now_ns=14400 * NS + 1))
        self.assert_code("state_rejected", lambda: value.begin_observe(now_ns=14400 * NS + 2))
        self.boundaries(value)

    def test_two_request_shutdown_floor_is_never_spent_on_another_work_poll(self):
        value = active(maximum_requests=4)
        now = 15 * NS + 2
        self.assert_code("termination_required", lambda: value.begin_observe(now_ns=now))
        self.assertEqual(value.report()["request_reservations"], 2)
        termination = value.begin_terminate(now_ns=now + 1)
        self.assertEqual(termination.sequence, 3)
        value.receive(termination.sequence, http_status=200, body=terminated(), now_ns=now + 2)
        final = value.begin_observe(now_ns=now + 3)
        self.assertEqual(final.sequence, 4)
        value.receive(final.sequence, http_status=200, body=observed("terminated"), now_ns=now + 4)
        report = self.boundaries(value)
        self.assertEqual(report["request_reservations"], 4)
        self.assertTrue(report["terminal_observation_modeled"])

    def test_failure_after_request_floor_consumes_budget_without_refund_or_retry(self):
        value = active(maximum_requests=4)
        value.finish_work(now_ns=4)
        intent = value.begin_terminate(now_ns=5)
        value.fail_pending(intent.sequence, now_ns=6)
        final = value.begin_observe(now_ns=15 * NS + 2)
        value.fail_pending(final.sequence, now_ns=15 * NS + 3)
        self.assert_code("request_budget_exhausted",
                         lambda: value.begin_observe(now_ns=30 * NS + 2))
        report = self.boundaries(value)
        self.assertEqual(report["request_reservations"], 4)
        self.assertEqual(report["termination_attempts_modeled"], 1)
        self.assertTrue(report["reconciliation_required"])

    def test_poll_spacing_uses_monotonic_dispatch_time_and_does_not_precharge_early_poll(self):
        value = active()
        self.assert_code("poll_too_soon", lambda: value.begin_observe(now_ns=15 * NS + 1))
        self.assertEqual(value.report()["request_reservations"], 2)
        intent = value.begin_observe(now_ns=15 * NS + 2)
        self.assertEqual(intent.sequence, 3)
        self.assertEqual(intent.deadline_ns, 30 * NS + 2)

    def test_event_budget_is_global_including_equal_time_ticks(self):
        value = model()
        for _ in range(4096):
            value.tick(now_ns=0)
        self.assert_code("event_budget_exhausted", lambda: value.tick(now_ns=0))
        self.assert_code("event_budget_exhausted",
                         lambda: value.begin_create(now_ns=1, now_unix=1000))
        report = self.boundaries(value)
        self.assertEqual(report["events"], 4096)
        self.assertEqual(report["create_attempts_modeled"], 0)
        self.assertTrue(report["reconciliation_required"])

    def test_invalid_or_backward_monotonic_clocks_quarantine_model(self):
        for now in (True, -1, 1.0, "1", None, 2**63):
            value = model()
            self.assert_code("clock_rejected", lambda now=now, value=value: value.tick(now_ns=now))
            self.assertTrue(self.boundaries(value)["reconciliation_required"])
        value = active()
        self.assert_code("clock_rejected", lambda: value.tick(now_ns=2))
        self.assertFalse(self.boundaries(value)["simulated_work_window_open"])

    def test_create_lifecycle_deadline_addition_cannot_overflow_clock(self):
        value = model()
        self.assert_code("clock_rejected", lambda: value.begin_create(
            now_ns=lifecycle.MAXIMUM_INTEGER, now_unix=1000,
        ))
        self.assertEqual(value.report()["create_attempts_modeled"], 0)
        self.assertEqual(value.report()["request_reservations"], 0)

    def test_out_of_order_reply_quarantine_cannot_be_reopened_by_pending_correct_reply(self):
        value = bound()
        intent = value.begin_observe(now_ns=2)
        self.assert_code("response_out_of_order", lambda: value.receive(
            intent.sequence + 1, http_status=200, body=observed(), now_ns=3,
        ))
        try:
            value.receive(intent.sequence, http_status=200, body=observed(), now_ns=4)
        except lifecycle.LifecycleError:
            pass
        report = self.boundaries(value)
        self.assertTrue(report["reconciliation_required"])
        self.assertFalse(report["simulated_work_window_open"])
        self.assertEqual(report["first_fault"], "response_out_of_order")
        self.assert_code("state_rejected", lambda: value.begin_terminate(now_ns=5))

    def test_backward_clock_quarantine_cannot_be_reopened_by_pending_create_reply(self):
        value = model()
        intent = value.begin_create(now_ns=10, now_unix=1000)
        self.assert_code("clock_rejected", lambda: value.tick(now_ns=9))
        try:
            value.receive(intent.sequence, http_status=200, body=launch(), now_ns=11)
        except lifecycle.LifecycleError:
            pass
        report = self.boundaries(value)
        self.assertTrue(report["reconciliation_required"])
        self.assertFalse(report["instance_bound_in_model"])
        self.assertEqual(report["first_fault"], "clock_rejected")

    def test_reply_sequence_is_exact_strict_integer_and_cannot_be_replayed(self):
        for sequence in (True, 1.0, "1", 0, 65):
            value = model()
            value.begin_create(now_ns=0, now_unix=1000)
            self.assert_code("response_out_of_order", lambda sequence=sequence, value=value:
                             value.receive(
                sequence, http_status=200, body=launch(), now_ns=1,
            ))
            self.boundaries(value)
        value = bound()
        self.assert_code("response_out_of_order", lambda: value.receive(
            1, http_status=200, body=launch(), now_ns=2,
        ))
        self.assertEqual(self.boundaries(value)["responses_consumed"], 1)

    def test_reconciliation_cannot_revive_when_shutdown_reserve_is_later_reached(self):
        for fault in ("response_out_of_order", "clock_rejected"):
            value = bound()
            intent = value.begin_observe(now_ns=2)
            if fault == "response_out_of_order":
                self.assert_code(fault, lambda value=value, intent=intent: value.receive(
                    intent.sequence + 1, http_status=200, body=observed(), now_ns=3,
                ))
            else:
                self.assert_code(fault, lambda value=value: value.tick(now_ns=1))
            report = value.tick(now_ns=13800 * NS)
            self.assertTrue(report["reconciliation_required"])
            self.assertFalse(report["pending_request"])
            self.assertFalse(report["simulated_work_window_open"])
            self.assertEqual(report["first_fault"], fault)
            self.assert_code("state_rejected", lambda value=value: value.begin_create(
                now_ns=13800 * NS + 1, now_unix=1000,
            ))
            self.assert_code("state_rejected",
                             lambda value=value: value.begin_observe(now_ns=13800 * NS + 2))
            self.assert_code("state_rejected",
                             lambda value=value: value.begin_terminate(now_ns=13800 * NS + 3))
            self.assertEqual(self.boundaries(value)["request_reservations"], 2)


class TerminationTests(Assertions):
    def test_acknowledgement_even_terminated_is_not_separate_get_terminal_evidence(self):
        for status in ("terminating", "terminated"):
            value = bound()
            intent = value.begin_terminate(now_ns=2)
            value.receive(intent.sequence, http_status=200, body=terminated(status), now_ns=3)
            report = self.boundaries(value)
            self.assertTrue(report["termination_acknowledgment_modeled"])
            self.assertFalse(report["terminal_observation_modeled"])
            self.assertFalse(report["simulated_work_window_open"])
            self.assertEqual(report["state"], "termination_observing")
            final = value.begin_observe(now_ns=4)
            self.assertEqual(final.instance_id, OWNED)
            value.receive(final.sequence, http_status=200, body=observed("terminated"), now_ns=5)
            self.assertTrue(self.boundaries(value)["terminal_observation_modeled"])

    def test_wrong_id_termination_ack_cannot_authorize_foreign_host_action_or_proof(self):
        value = bound()
        intent = value.begin_terminate(now_ns=2)
        value.receive(intent.sequence, http_status=200, body=terminated(identifier=FOREIGN),
                      now_ns=3)
        report = self.boundaries(value)
        self.assertEqual(report["first_fault"], "response_rejected")
        self.assertFalse(report["termination_acknowledgment_modeled"])
        self.assertFalse(report["terminal_observation_modeled"])
        self.assertEqual(value.begin_observe(now_ns=4).instance_id, OWNED)

    def test_404_preempted_idle_unhealthy_and_nonterminal_status_are_never_terminal_proof(self):
        for status, http in (("terminated", 404), ("preempted", 200), ("idle", 200),
                             ("unhealthy", 200), ("active", 200), ("booting", 200),
                             ("terminating", 200)):
            value = bound()
            intent = value.begin_terminate(now_ns=2)
            value.receive(intent.sequence, http_status=200, body=terminated(), now_ns=3)
            final = value.begin_observe(now_ns=4)
            value.receive(final.sequence, http_status=http, body=observed(status), now_ns=5)
            report = self.boundaries(value)
            self.assertFalse(report["terminal_observation_modeled"], status)
            self.assertFalse(report["simulated_work_window_open"], status)
            self.assertEqual(report["termination_attempts_modeled"], 1)

    def test_termination_post_cannot_be_reissued_before_or_after_ack_timeout_or_failure(self):
        for outcome in ("pending", "ack", "timeout", "failure"):
            value = bound()
            intent = value.begin_terminate(now_ns=2)
            now = 3
            if outcome == "ack":
                value.receive(intent.sequence, http_status=200, body=terminated(), now_ns=now)
            elif outcome == "timeout":
                now = intent.deadline_ns
                value.tick(now_ns=now)
            elif outcome == "failure":
                value.fail_pending(intent.sequence, now_ns=now)
            self.assert_code("state_rejected",
                             lambda value=value, now=now: value.begin_terminate(now_ns=now + 1))
            self.assertEqual(self.boundaries(value)["termination_attempts_modeled"], 1)
            self.assertEqual(value.report()["request_reservations"], 2)

    def test_timeout_still_allows_separate_terminal_get_but_never_another_post(self):
        value = bound()
        intent = value.begin_terminate(now_ns=2)
        value.tick(now_ns=intent.deadline_ns)
        final = value.begin_observe(now_ns=intent.deadline_ns + 1)
        value.receive(final.sequence, http_status=200, body=observed("terminated"),
                      now_ns=intent.deadline_ns + 2)
        report = self.boundaries(value)
        self.assertEqual(report["first_fault"], "request_timed_out")
        self.assertFalse(report["termination_acknowledgment_modeled"])
        self.assertTrue(report["terminal_observation_modeled"])
        self.assertEqual(report["termination_attempts_modeled"], 1)

    def test_failed_work_observation_preserves_owned_cleanup_even_after_create_window_expires(self):
        value = model(create_expires_unix=1001)
        create = value.begin_create(now_ns=0, now_unix=1000)
        value.receive(create.sequence, http_status=200, body=launch(), now_ns=1)
        poll = value.begin_observe(now_ns=2000 * NS)
        value.fail_pending(poll.sequence, now_ns=2000 * NS + 1)
        terminate = value.begin_terminate(now_ns=2000 * NS + 2)
        self.assertEqual(terminate.instance_id, OWNED)
        report = self.boundaries(value)
        self.assertEqual(report["first_fault"], "request_failed")
        self.assertEqual(report["termination_attempts_modeled"], 1)

    def test_work_completion_stops_activity_without_claiming_provider_termination(self):
        value = active()
        value.finish_work(now_ns=4)
        report = self.boundaries(value)
        self.assertEqual(report["stop_reason"], "work_completed")
        self.assertFalse(report["simulated_work_window_open"])
        self.assertFalse(report["terminal_observation_modeled"])
        self.assert_code("termination_required", lambda: value.begin_observe(now_ns=5))
        self.assertEqual(value.begin_terminate(now_ns=6).instance_id, OWNED)

    def test_late_terminal_get_is_not_proof_and_no_intent_deadline_exceeds_host_deadline(self):
        value = bound()
        terminate = value.begin_terminate(now_ns=14380 * NS)
        value.receive(terminate.sequence, http_status=200, body=terminated(), now_ns=14381 * NS)
        final = value.begin_observe(now_ns=14390 * NS)
        self.assertEqual(final.deadline_ns, 14400 * NS)
        value.receive(final.sequence, http_status=200, body=observed("terminated"),
                      now_ns=14400 * NS)
        report = self.boundaries(value)
        self.assertTrue(report["reconciliation_required"])
        self.assertFalse(report["terminal_observation_modeled"])
        self.assertTrue(report["lifecycle_deadline_breached"])

    def test_terminal_modeled_time_freezes_but_never_claims_actual_billing_stopped(self):
        value = bound()
        final = value.begin_observe(now_ns=2)
        value.receive(final.sequence, http_status=200, body=observed("terminated"), now_ns=3)
        before = self.boundaries(value)
        after = value.tick(now_ns=14401 * NS)
        self.assertEqual(after["state"], "terminal_observed")
        self.assertEqual(after["elapsed_lifecycle_ns"], before["elapsed_lifecycle_ns"])
        self.assertEqual(after["compute_projection_microusd"],
                         before["compute_projection_microusd"])
        self.assertFalse(after["lifecycle_deadline_breached"])
        self.assertFalse(after["cloud_billing_stopped"])
        self.assert_code("state_rejected", lambda: value.begin_observe(now_ns=14401 * NS + 1))
        self.assert_code("state_rejected", lambda: value.begin_terminate(now_ns=14401 * NS + 2))
        self.boundaries(value)

    def test_complete_synthetic_lifecycle_performs_no_clock_io_transport_or_host_action(self):
        with (
            patch.object(builtins, "open", side_effect=AssertionError("file I/O")) as opened,
            patch.object(os, "open", side_effect=AssertionError("descriptor I/O")) as os_open,
            patch.object(os, "write", side_effect=AssertionError("host mutation")) as written,
            patch.object(socket, "socket", side_effect=AssertionError("network")) as network,
            patch.object(subprocess, "Popen", side_effect=AssertionError("process")) as process,
            patch.object(time, "monotonic", side_effect=AssertionError("clock read")) as clock,
        ):
            value = active()
            value.finish_work(now_ns=4)
            intent = value.begin_terminate(now_ns=5)
            value.receive(intent.sequence, http_status=200, body=terminated(), now_ns=6)
            final = value.begin_observe(now_ns=15 * NS + 2)
            value.receive(final.sequence, http_status=200, body=observed("terminated"),
                          now_ns=15 * NS + 3)
            report = self.boundaries(value)
            self.assertTrue(report["terminal_observation_modeled"])
            for operation in (opened, os_open, written, network, process, clock):
                operation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
