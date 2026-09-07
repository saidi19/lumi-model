"""Single-use stop-only driver for an already owned provider controller.

This library creates no controller, process, credential or execution authority.
With an authorized real controller its methods CAN perform provider I/O. The
control endpoint is a trusted stop-only capability, not an operation/target RPC.
Signals must only latch stop in that endpoint. Synchronous resolver, transport,
storage and scheduler delays are not hard wall-time or survival guarantees.
"""

from __future__ import annotations

import os
import threading
import time

from . import posttraining_sft_provider_controller_v1 as provider

NS = 1_000_000_000
SCHEMA = "lumi.sft-controller-service-report.v0.1"
STATUS_CODES = frozenset({"terminal_modeled", "reconciliation_required"})
POLL_INTERVAL_NS = 15 * NS
MAXIMUM_CONTROL_POLLS = 60000
MAXIMUM_ACTIONS = 128
STATES = frozenset({"prepared", "bound", "termination_due", "termination_observing",
                    "terminal_observed", "reconciliation_required"})
STOP_CODES = frozenset({
    "stop_requested", "parent_eof", "signal_requested", "control_fault", "shutdown_cutoff",
    "hard_deadline", "controller_failed", "reconciliation_required", "report_rejected",
    "clock_rejected", "service_rejected", "action_limit", "poll_limit", "terminal_observed",
    "local_close_failed", "unbound_stopped", "bootstrap_rejected", "process_failure",
})
CONTROL_CODES = {"continue": None, "stop": "stop_requested", "eof": "parent_eof",
                 "signal": "signal_requested", "fault": "control_fault"}
BOOL_FIELDS = (
    "controller_failed", "controller_closed", "controller_busy", "pending_request",
    "instance_bound_in_model", "terminal_observation_modeled", "reconciliation_required",
)
COUNT_FIELDS = {
    "request_reservations": 64, "dispatch_calls": 64, "create_attempts_modeled": 1,
    "termination_attempts_modeled": 1, "journal_records_synced": 256,
}
REPORT_COUNTS = {"provider_requests": 64, "control_polls": MAXIMUM_CONTROL_POLLS,
                 "service_actions": MAXIMUM_ACTIONS, "journal_records": 256}
FALSE_FIELDS = ("execution_authorized", "provider_termination_verified", "cloud_billing_stopped",
                "independent_survival_verified", "production_eligible")
REPORT_BOOLEANS = ("modeled_terminal", "reconciliation_required", "local_controller_closed")
REPORT_KEYS = frozenset(("schema_version", "status", "stop_code", *REPORT_COUNTS,
                         *FALSE_FIELDS, *REPORT_BOOLEANS))


class ServiceError(ValueError):
    def __init__(self, code="service_rejected"):
        self.code = code if type(code) is str and code in STOP_CODES else "service_rejected"
        super().__init__(self.code)


def need(condition, code="service_rejected"):
    if not condition:
        raise ServiceError(code) from None


def integer(value, maximum=(1 << 63) - 1):
    return type(value) is int and 0 <= value <= maximum


def validate_report(value):
    """Strict fixed-field projection for an independently framed process result."""
    need(type(value) is dict and set(value) == REPORT_KEYS, "report_rejected")
    need(value["schema_version"] == SCHEMA and type(value["schema_version"]) is str,
         "report_rejected")
    need(type(value["status"]) is str and value["status"] in STATUS_CODES,
         "report_rejected")
    need(type(value["stop_code"]) is str and value["stop_code"] in STOP_CODES,
         "report_rejected")
    unknown = all(value[name] is None for name in REPORT_COUNTS)
    for name, maximum in REPORT_COUNTS.items():
        need(unknown or integer(value[name], maximum), "report_rejected")
    for name in FALSE_FIELDS:
        need(value[name] is False, "report_rejected")
    for name in REPORT_BOOLEANS:
        need(type(value[name]) is bool, "report_rejected")
    if value["status"] == "terminal_modeled":
        need(not unknown and value["modeled_terminal"] and value["local_controller_closed"]
             and not value["reconciliation_required"], "report_rejected")
    else:
        need(value["reconciliation_required"], "report_rejected")
    if unknown:
        need(not value["modeled_terminal"] and not value["local_controller_closed"],
             "report_rejected")
    return dict(value)


def snapshot(controller, maximum_requests):
    """Project and validate only fields used by this driver; discard all extras."""
    value = controller.report()
    need(type(value) is dict, "report_rejected")
    need(type(value.get("state")) is str and value["state"] in STATES, "report_rejected")
    for name in BOOL_FIELDS:
        need(type(value.get(name)) is bool, "report_rejected")
    for name, maximum in COUNT_FIELDS.items():
        need(integer(value.get(name), maximum), "report_rejected")
    need(value["request_reservations"] <= maximum_requests
         and value["dispatch_calls"] <= value["request_reservations"], "report_rejected")
    need(not value["pending_request"] and not value["controller_busy"], "report_rejected")
    state = value["state"]
    need(value["terminal_observation_modeled"] == (state == "terminal_observed"),
         "report_rejected")
    if state == "prepared":
        need(value["request_reservations"] == value["create_attempts_modeled"] == 0
             and not value["instance_bound_in_model"]
             and value["termination_attempts_modeled"] == 0, "report_rejected")
    elif state not in {"reconciliation_required"}:
        need(value["instance_bound_in_model"] and value["create_attempts_modeled"] == 1,
             "report_rejected")
    if state == "termination_observing":
        need(value["termination_attempts_modeled"] == 1, "report_rejected")
    return {name: value[name] for name in ("state", *BOOL_FIELDS, *COUNT_FIELDS)}


def run_controller(controller, control, *, clock_ns=None):
    """Consume one prepared controller; stop closes work, never the cleanup reserve.

    A fresh observation is scheduled at least 15 seconds after completion of the
    preceding observation, including a recoverable early-poll denial. No durable
    tick or fictitious finish_work record is written for local polling/stop.
    Closing releases only local descriptors. Reports cannot prove host shutdown.
    """
    clock = time.monotonic_ns if clock_ns is None else clock_ns
    owner_pid, owner_thread = os.getpid(), threading.current_thread()
    last_clock = None
    started = hard = cutoff = None
    polls = actions = 0
    reason = stop = None
    terminal = closed = False
    failed = False
    owned = claimed = False
    last_snapshot = None
    maximum_requests = 64
    next_observation = 0
    create_started = termination_started = False

    def now():
        nonlocal last_clock
        need(os.getpid() == owner_pid and threading.current_thread() is owner_thread)
        try:
            value = clock()
        except BaseException:
            raise ServiceError("clock_rejected") from None
        need(os.getpid() == owner_pid and threading.current_thread() is owner_thread)
        need(integer(value) and (last_clock is None or value >= last_clock), "clock_rejected")
        last_clock = value
        return value

    def sample():
        nonlocal stop
        value = now()
        need(hard is None or value < hard, "hard_deadline")
        if cutoff is not None and value >= cutoff:
            stop = stop or "shutdown_cutoff"
        return value

    def poll(timeout):
        nonlocal polls, stop
        need(polls < MAXIMUM_CONTROL_POLLS, "poll_limit")
        sample()
        polls += 1
        try:
            result = control.poll(timeout)
        except BaseException:
            result = "fault"
        sample()
        if type(result) is not str or result not in CONTROL_CODES:
            result = "fault"
        stop = stop or CONTROL_CODES[result]

    try:
        need(type(controller) is provider.ProviderController)
        need(controller._pid == owner_pid and controller._thread is owner_thread)
        owned = True
        need(not getattr(controller, "_controller_service_claimed_v1", False))
        controller._controller_service_claimed_v1 = True
        claimed = True
        need(callable(clock) and callable(getattr(control, "poll", None)))
        plan = controller._plan
        need(type(plan) is dict)
        maximum_requests = plan["maximum_requests"]
        life, reserve = plan["maximum_lifecycle_seconds"], plan["shutdown_reserve_seconds"]
        need(integer(maximum_requests, 64) and maximum_requests >= 4)
        need(integer(life, 14400) and life >= 61 and integer(reserve, 3600)
             and 60 <= reserve < life)
        started = now()
        need(started <= (1 << 63) - 1 - life * NS, "clock_rejected")
        hard, cutoff = started + life * NS, started + (life - reserve) * NS
        last_snapshot = snapshot(controller, maximum_requests)
        need(last_snapshot["state"] == "prepared"
             and not last_snapshot["controller_closed"]
             and not last_snapshot["controller_failed"])
        needs_control = True
        while True:
            if needs_control:
                poll(0)
            needs_control = False
            current = sample()
            last_snapshot = snapshot(controller, maximum_requests)
            state = last_snapshot["state"]
            if last_snapshot["controller_failed"] or last_snapshot["controller_closed"]:
                raise ServiceError("controller_failed")
            if last_snapshot["reconciliation_required"] or state == "reconciliation_required":
                raise ServiceError("reconciliation_required")
            if state == "terminal_observed":
                terminal = True
                reason = stop or "terminal_observed"
                break
            if state == "prepared":
                if stop is not None:
                    reason = stop
                    break
                need(not create_started, "report_rejected")
                operation = "create"
            elif state == "termination_due" or (state == "bound" and stop is not None):
                need(not termination_started
                     and last_snapshot["termination_attempts_modeled"] == 0, "report_rejected")
                operation = "terminate"
            elif state in {"bound", "termination_observing"}:
                if current < next_observation:
                    delay = min(0.25, (next_observation - current) / NS,
                                (hard - current) / NS)
                    if stop is None:
                        delay = min(delay, max(0, (cutoff - current) / NS))
                    poll(delay)
                    continue
                operation = "observe"
            else:
                raise ServiceError("report_rejected")
            need(actions < MAXIMUM_ACTIONS, "action_limit")
            sample()
            actions += 1
            needs_control = True
            before = last_snapshot
            if operation == "create":
                create_started = True
            elif operation == "terminate":
                termination_started = True
            error = None
            try:
                getattr(controller, operation)()
            except provider.ProviderError as exc:
                error = exc.code
            except BaseException:
                raise ServiceError("controller_failed") from None
            completed = sample()
            if operation == "observe":
                next_observation = completed + POLL_INTERVAL_NS
            last_snapshot = snapshot(controller, maximum_requests)
            if error is not None:
                need(error in {"exchange_failed", "termination_required", "poll_too_soon"},
                     "controller_failed")
                # The next iteration decides ONLY from the frozen controller state.
                # An unsuccessful reserved sequence is never manually dispatched again.
                if error == "poll_too_soon":
                    need(operation == "observe", "report_rejected")
            denial = error in {"termination_required", "poll_too_soon"}
            need(last_snapshot["request_reservations"]
                 == before["request_reservations"] + (0 if denial else 1), "report_rejected")
            need(all(last_snapshot[name] >= before[name] for name in COUNT_FIELDS),
                 "report_rejected")
            if operation == "create":
                need(not denial and last_snapshot["create_attempts_modeled"] == 1
                     and last_snapshot["state"] in {"bound", "reconciliation_required"},
                     "report_rejected")
            elif operation == "terminate":
                need(not denial and last_snapshot["termination_attempts_modeled"] == 1
                     and last_snapshot["state"] in {"termination_observing",
                                                     "reconciliation_required"},
                     "report_rejected")
            elif error == "termination_required":
                need(last_snapshot["state"] == "termination_due", "report_rejected")
            elif error == "poll_too_soon":
                need(last_snapshot["state"] in {"bound", "termination_observing"},
                     "report_rejected")
    except ServiceError as exc:
        reason, failed = exc.code, True
    except BaseException:
        reason, failed = "service_rejected", True
    finally:
        # A second invocation must not interfere with an already running first one.
        if owned and claimed:
            try:
                controller.close()
                last_snapshot = snapshot(controller, maximum_requests)
                closed = last_snapshot["controller_closed"]
                terminal = last_snapshot["terminal_observation_modeled"]
                if not closed or last_snapshot["controller_failed"]:
                    failed = True
                    reason = reason or "local_close_failed"
                if started is not None:
                    sample()
            except ServiceError as exc:
                failed = True
                reason = exc.code
            except BaseException:
                failed = True
                reason = reason or "local_close_failed"
    reconciliation = failed or not terminal
    # A cancellation before any create has no remote cleanup obligation; the
    # frozen close snapshot still conservatively requires local reconciliation.
    if last_snapshot is not None:
        reconciliation = reconciliation or last_snapshot["reconciliation_required"]
    return validate_report({
        "schema_version": SCHEMA,
        "status": "terminal_modeled" if terminal and not failed else "reconciliation_required",
        "stop_code": reason or "unbound_stopped",
        "modeled_terminal": terminal,
        "reconciliation_required": reconciliation,
        "local_controller_closed": closed,
        "provider_requests": 0 if last_snapshot is None else last_snapshot["request_reservations"],
        "control_polls": polls, "service_actions": actions,
        "journal_records": 0 if last_snapshot is None else last_snapshot["journal_records_synced"],
        "execution_authorized": False, "provider_termination_verified": False,
        "cloud_billing_stopped": False, "independent_survival_verified": False,
        "production_eligible": False,
    })
