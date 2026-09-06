"""Transport-free, single-attempt host lifecycle decision model.

Inputs are supplied by a trusted caller, not authenticated provider observations.
Intents are INERT: no HTTP, credentials, clock reads, files, guest access or cloud
actions exist here. This model cannot authorize execution or certify billing.
A future external durable controller must enforce exclusive attempts, authenticate
authority/account/transport, survive guest failure and reconcile ambiguous creates.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from typing import NamedTuple

from . import posttraining_sft_lambda_wire_v1 as wire

PLAN_SCHEMA = "lumi.sft-host-lifecycle-model-plan.v0.1"
REPORT_SCHEMA = "lumi.sft-host-lifecycle-model-report.v0.1"
MAXIMUM_PLAN_BYTES = 16384
MAXIMUM_EVENTS = 4096
RESPONSE_CAPACITY = 65536
REQUEST_TIMEOUT_NS = 15_000_000_000
POLL_INTERVAL_NS = 15_000_000_000
NS = 1_000_000_000
MAXIMUM_INTEGER = 2**63 - 1
HASH = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
IDENTIFIER = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)
LABEL = re.compile(r"[a-z0-9_-]{1,64}\Z", re.ASCII)
PLAN_KEYS = {
    "schema_version", "mode", "attempt_id", "account_scope_sha256", "approval_sha256",
    "launch_request_sha256", "protected_instance_ids", "region", "instance_type",
    "gpu_count", "total_host_hourly_rate_microusd", "maximum_cost_microusd",
    "noncompute_reserve_microusd", "maximum_lifecycle_seconds", "shutdown_reserve_seconds",
    "create_not_before_unix", "create_expires_unix", "maximum_requests",
}
ERROR_CODES = {
    "plan_rejected", "clock_rejected", "event_budget_exhausted", "state_rejected",
    "authorization_window_closed", "request_budget_exhausted", "termination_required",
    "poll_too_soon", "response_rejected", "response_out_of_order", "request_failed",
    "request_timed_out", "protected_instance_rejected", "target_mismatch",
    "host_unhealthy", "terminal_state_unproven", "lifecycle_deadline_exceeded",
    "shutdown_reserve_reached", "work_completed",
}
NONCLAIMS = (
    "provider_transport_authenticated", "provider_termination_verified", "cloud_billing_stopped",
    "provider_invoice_limit_enforced", "durable_attempt_exclusivity_verified",
    "external_controller_survival_verified", "target_gpu_host_verified",
    "execution_authorized", "real_sft_execution_permitted", "production_eligible",
    "full_private_integration_passed",
)


class LifecycleError(ValueError):
    def __init__(self, code):
        self.code = code if type(code) is str and code in ERROR_CODES else "state_rejected"
        super().__init__(self.code)


def need(condition, code="plan_rejected"):
    if not condition:
        raise LifecycleError(code) from None


def integer(value, minimum=0, maximum=MAXIMUM_INTEGER):
    return type(value) is int and minimum <= value <= maximum


def matches(value, pattern):
    return type(value) is str and pattern.fullmatch(value) is not None


def projected_cost(elapsed_ns, rate_microusd):
    """Whole-host quote projection, rounded UP to minute increments and micro-USD.

Provisioning time is conservatively charged from the modeled create dispatch.
This is not the provider's actual health-check/billing clock or final invoice.
"""
    need(integer(elapsed_ns) and integer(rate_microusd, 1, 150_000_000))
    minutes = (elapsed_ns + 60 * NS - 1) // (60 * NS)
    return (minutes * rate_microusd + 59) // 60


def parse_plan(raw, expected_sha256):
    need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_PLAN_BYTES)
    need(matches(expected_sha256, HASH))
    need(hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_sha256))

    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result)
            result[key] = value
        return result

    def constant(_value):
        raise LifecycleError("plan_rejected")

    try:
        plan = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError):
        raise LifecycleError("plan_rejected") from None
    need(type(plan) is dict and set(plan) == PLAN_KEYS)
    need(plan["schema_version"] == PLAN_SCHEMA and plan["mode"] == "local_decision_model_only")
    for key in ("attempt_id", "account_scope_sha256", "approval_sha256", "launch_request_sha256"):
        need(matches(plan[key], HASH))
    for key in ("region", "instance_type"):
        need(matches(plan[key], LABEL))
    protected = plan["protected_instance_ids"]
    need(type(protected) is list and 1 <= len(protected) <= 128)
    need(all(matches(value, IDENTIFIER) for value in protected))
    need(len(set(protected)) == len(protected))
    need(integer(plan["gpu_count"], 1, 8))
    need(integer(plan["maximum_requests"], 4, 64))
    need(integer(plan["maximum_lifecycle_seconds"], 61, 14400))
    need(integer(plan["shutdown_reserve_seconds"], 60, 3600))
    need(plan["shutdown_reserve_seconds"] < plan["maximum_lifecycle_seconds"])
    need(integer(plan["maximum_cost_microusd"], 1, 150_000_000))
    need(integer(plan["noncompute_reserve_microusd"], 0, plan["maximum_cost_microusd"]))
    rate = plan["total_host_hourly_rate_microusd"]
    need(integer(rate, 1, 150_000_000) and rate % 10000 == 0)
    need(projected_cost(plan["maximum_lifecycle_seconds"] * NS, rate)
         + plan["noncompute_reserve_microusd"] <= plan["maximum_cost_microusd"])
    need(integer(plan["create_not_before_unix"], 1, 2**53 - 1))
    need(integer(plan["create_expires_unix"], plan["create_not_before_unix"] + 1, 2**53 - 1))
    plan["protected_instance_ids"] = tuple(protected)
    return plan


class Intent(NamedTuple):
    """Internal, single-use decision token, not an HTTP request or authorization."""

    kind: str
    sequence: int
    instance_id: str | None
    deadline_ns: int
    response_capacity: int


class LifecycleModel:
    """Single-threaded, volatile model. No restart/resume or automatic retry API.

    A failed run retains cleanup obligations for its exact bound instance. Only
    ambiguous ownership, bad clocks or exhausted hard bounds close all intents.
    Even a terminal observation leaves every actual-world admission flag false.
    """

    def __init__(self, plan_bytes, expected_sha256):
        self._plan = parse_plan(plan_bytes, expected_sha256)
        self._state = "prepared"
        self._instance = None
        self._pending = None
        self._start = None
        self._terminal_at = None
        self._last = None
        self._last_poll = None
        self._events = self._requests = self._capacity = self._responses = 0
        self._create_attempts = self._termination_attempts = self._poll_attempts = 0
        self._stop = self._fault = None
        self._active = self._terminal = self._acknowledged = False
        self._deadline_breached = False

    def _fault_once(self, code):
        if self._fault is None:
            self._fault = code
        self._active = False

    def _reject(self, code):
        self._fault_once(code)
        self._state = "reconciliation_required"
        self._pending = None
        raise LifecycleError(code) from None

    def _touch(self, now_ns):
        if not integer(now_ns) or (self._last is not None and now_ns < self._last):
            self._reject("clock_rejected")
        if self._events >= MAXIMUM_EVENTS:
            self._reject("event_budget_exhausted")
        self._events += 1
        self._last = now_ns
        if self._start is not None and not self._terminal:
            elapsed = now_ns - self._start
            if elapsed >= self._plan["maximum_lifecycle_seconds"] * NS:
                self._deadline_breached = True
                self._fault_once("lifecycle_deadline_exceeded")
                self._state = "reconciliation_required"
            elif self._state != "reconciliation_required" and elapsed >= (
                self._plan["maximum_lifecycle_seconds"]
                - self._plan["shutdown_reserve_seconds"]
            ) * NS:
                self._stop = self._stop or "shutdown_reserve_reached"
                self._active = False
                if self._instance is not None and self._termination_attempts == 0:
                    self._state = "termination_due"

    def _reserve(self, kind, now_ns):
        need(self._pending is None, "state_rejected")
        need(self._state not in {"reconciliation_required", "terminal_observed"}, "state_rejected")
        limit = self._plan["maximum_requests"]
        # Before teardown reserve TWO requests: one termination and one terminal GET.
        available = limit - self._requests
        if available <= (2 if kind == "observe" and self._termination_attempts == 0 else 0):
            if self._instance is not None and self._termination_attempts == 0:
                self._stop = self._stop or "request_budget_exhausted"
                self._active = False
                self._state = "termination_due"
                raise LifecycleError("termination_required")
            self._reject("request_budget_exhausted")
        self._requests += 1
        self._capacity += RESPONSE_CAPACITY
        deadline = min(now_ns + REQUEST_TIMEOUT_NS,
                       self._start + self._plan["maximum_lifecycle_seconds"] * NS)
        self._pending = Intent(kind, self._requests, self._instance, deadline, RESPONSE_CAPACITY)
        return self._pending

    def begin_create(self, *, now_ns, now_unix):
        self._touch(now_ns)
        need(self._state == "prepared" and self._create_attempts == 0, "state_rejected")
        if not integer(now_unix, 1, 2**53 - 1) or not (
            self._plan["create_not_before_unix"] <= now_unix < self._plan["create_expires_unix"]
        ):
            self._reject("authorization_window_closed")
        need(now_ns <= MAXIMUM_INTEGER - self._plan["maximum_lifecycle_seconds"] * NS,
             "clock_rejected")
        self._start = now_ns
        self._create_attempts = 1
        self._state = "create_pending"
        return self._reserve("create", now_ns)

    def begin_observe(self, *, now_ns):
        self._touch(now_ns)
        need(self._instance is not None and not self._terminal, "state_rejected")
        if self._state == "termination_due":
            raise LifecycleError("termination_required")
        need(self._state in {"bound", "termination_observing"}, "state_rejected")
        if self._last_poll is not None and now_ns - self._last_poll < POLL_INTERVAL_NS:
            raise LifecycleError("poll_too_soon")
        intent = self._reserve("observe", now_ns)
        self._poll_attempts += 1
        self._last_poll = now_ns
        return intent

    def begin_terminate(self, *, now_ns):
        self._touch(now_ns)
        need(self._instance is not None and self._termination_attempts == 0
             and not self._terminal, "state_rejected")
        need(self._state in {"bound", "termination_due"}, "state_rejected")
        intent = self._reserve("terminate", now_ns)
        self._termination_attempts = 1
        self._active = False
        self._state = "termination_pending"
        return intent

    def finish_work(self, *, now_ns):
        self._touch(now_ns)
        need(self._instance is not None and self._termination_attempts == 0
             and self._state in {"bound", "termination_due"}, "state_rejected")
        self._stop = self._stop or "work_completed"
        self._active = False
        self._state = "termination_due"

    def tick(self, *, now_ns):
        self._touch(now_ns)
        if self._pending is not None and now_ns >= self._pending.deadline_ns:
            pending = self._pending
            self._pending = None
            self._failed(pending, "request_timed_out")
        return self.report()

    def _failed(self, intent, code):
        self._fault_once(code)
        if intent.kind == "create" or self._deadline_breached:
            self._state = "reconciliation_required"
        elif self._termination_attempts:
            self._state = "termination_observing"
        else:
            self._stop = self._stop or code
            self._state = "termination_due"

    def fail_pending(self, sequence, *, now_ns):
        self._touch(now_ns)
        intent = self._consume(sequence)
        self._failed(intent, "request_failed")

    def _consume(self, sequence):
        if not integer(sequence, 1, 64) or self._pending is None or (
            sequence != self._pending.sequence
        ):
            self._reject("response_out_of_order")
        intent = self._pending
        self._pending = None
        return intent

    def receive(self, sequence, *, http_status, body, now_ns):
        """Consume one supplied reply; unknown HTTP status/404 is NEVER terminal proof.

        A late create response is ambiguous, even if it contains a plausible ID.
        No search by name, ownership inference, second create or termination retry.
        """
        self._touch(now_ns)
        intent = self._consume(sequence)
        self._responses += 1
        if now_ns >= intent.deadline_ns:
            self._failed(intent, "request_timed_out")
            return self.report()
        if type(http_status) is not int or http_status != 200:
            self._failed(intent, "request_failed")
            return self.report()
        try:
            if intent.kind == "create":
                identifier = wire.parse_launch(body)
                if identifier in self._plan["protected_instance_ids"]:
                    self._failed(intent, "protected_instance_rejected")
                else:
                    self._instance = identifier
                    self._state = "bound"
            elif intent.kind == "terminate":
                wire.parse_termination(body, self._instance)
                self._acknowledged = True
                self._state = "termination_observing"
            else:
                observation = wire.parse_instance(body)
                if observation["instance_id"] != self._instance:
                    self._failed(intent, "target_mismatch")
                else:
                    self._observe(observation, intent)
        except wire.WireError:
            self._failed(intent, "response_rejected")
        return self.report()

    def _observe(self, observation, intent):
        # Only the direct GET of the exact create-bound ID can model terminal evidence.
        if observation["status"] == "terminated":
            self._terminal = True
            self._terminal_at = self._last
            self._active = False
            self._state = "terminal_observed"
            return
        if observation["status"] in {"preempted", "unhealthy"}:
            self._failed(intent, "host_unhealthy" if observation["status"] == "unhealthy"
                         else "terminal_state_unproven")
            return
        expected = (self._plan["region"], self._plan["instance_type"], self._plan["gpu_count"],
                    self._plan["total_host_hourly_rate_microusd"])
        actual = (observation["region"], observation["instance_type"], observation["gpu_count"],
                  observation["price_cents_per_hour"] * 10000)
        if actual != expected:
            self._failed(intent, "target_mismatch")
        elif observation["status"] == "terminating":
            self._active = False
            self._stop = self._stop or "terminal_state_unproven"
            self._state = ("termination_observing" if self._termination_attempts
                           else "termination_due")
        else:
            self._active = (observation["status"] == "active" and self._fault is None
                            and self._stop is None and self._termination_attempts == 0)
            self._state = "termination_observing" if self._termination_attempts else (
                "termination_due" if self._stop else "bound")

    def report(self):
        """Allowlisted scalar aggregates only: no IDs, approval pins or response fields."""
        last = self._last if self._terminal_at is None else self._terminal_at
        elapsed = 0 if self._start is None else last - self._start
        cost = projected_cost(elapsed, self._plan["total_host_hourly_rate_microusd"])
        return {
            "schema_version": REPORT_SCHEMA,
            "mode": "local_decision_model_only",
            "state": self._state,
            "first_fault": self._fault,
            "stop_reason": self._stop,
            "events": self._events,
            "request_reservations": self._requests,
            "response_capacity_reserved_bytes": self._capacity,
            "responses_consumed": self._responses,
            "create_attempts_modeled": self._create_attempts,
            "termination_attempts_modeled": self._termination_attempts,
            "poll_attempts_modeled": self._poll_attempts,
            "pending_request": self._pending is not None,
            "instance_bound_in_model": self._instance is not None,
            "termination_acknowledgment_modeled": self._acknowledged,
            "terminal_observation_modeled": self._terminal,
            "simulated_work_window_open": self._active,
            "elapsed_lifecycle_ns": elapsed,
            "compute_projection_microusd": cost,
            "noncompute_reserve_microusd": self._plan["noncompute_reserve_microusd"],
            "projection_with_reserve_microusd": cost + self._plan["noncompute_reserve_microusd"],
            "lifecycle_deadline_breached": self._deadline_breached,
            "reconciliation_required": self._state == "reconciliation_required",
            "provider_calls_executed": 0,
            "external_network_bytes": 0,
            "gpu_operations": 0,
            "private_artifact_reads": 0,
            "private_artifact_writes": 0,
            **dict.fromkeys(NONCLAIMS, False),
        }
