"""Exclusive local write-ahead lifecycle decisions; never provider dispatch.

The trusted caller supplies one stable local private-directory descriptor. Every
decision is fsynced before release. Existing attempts are never reopened for
execution, including empty/partial journals. Hash-only evidence cannot reconstruct
targeted teardown after a crash; external authenticated reconciliation is required.
This is not a remote controller, encrypted store or invoice/availability guarantee.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time

from . import posttraining_sft_host_journal_v1 as journal
from . import posttraining_sft_host_lifecycle_v1 as model

SCHEMA = "lumi.sft-durable-host-lifecycle-record.v0.1"
REPORT_SCHEMA = "lumi.sft-durable-host-lifecycle-report.v0.1"
ZERO = "0" * 64
EVENTS = frozenset({"initialized", "begin_create", "begin_observe", "begin_terminate",
                    "receive", "fail_pending", "tick", "finish_work", "close"})
STATES = frozenset({"prepared", "create_pending", "bound", "termination_due",
                    "termination_pending", "termination_observing", "terminal_observed",
                    "reconciliation_required"})
ERROR_CODES = frozenset({"controller_rejected", "controller_closed", "controller_failed",
                         "journal_rejected", "clock_rejected", "durability_deadline_rejected",
                         "journal_evidence_rejected", "authorization_window_closed"})
COUNTERS = {
    "events": 4096, "request_reservations": 64,
    "response_capacity_reserved_bytes": 64 * 65536, "responses_consumed": 64,
    "create_attempts_modeled": 1, "termination_attempts_modeled": 1,
    "poll_attempts_modeled": 64, "elapsed_lifecycle_ns": model.MAXIMUM_INTEGER,
    "compute_projection_microusd": model.MAXIMUM_INTEGER,
}
BOOLEANS = frozenset({"pending_request", "instance_bound_in_model",
                      "termination_acknowledgment_modeled", "terminal_observation_modeled",
                      "simulated_work_window_open", "lifecycle_deadline_breached"})
SNAPSHOT_KEYS = set(COUNTERS) | BOOLEANS | {"state", "first_fault", "stop_reason"}
RECORD_KEYS = {"schema_version", "sequence", "previous_sha256", "plan_sha256", "event",
               "error_code", "intent_kind", "intent_sequence", "intent_deadline_ns",
               "target_sha256", "snapshot", "sha256"}


class ControllerError(ValueError):
    def __init__(self, code):
        self.code = code if type(code) is str and code in ERROR_CODES else "controller_rejected"
        super().__init__(self.code)


def need(condition, code="journal_evidence_rejected"):
    if not condition:
        raise ControllerError(code) from None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _valid_record(record):
    need(type(record) is dict and set(record) == RECORD_KEYS)
    need(record["schema_version"] == SCHEMA and type(record["event"]) is str
         and record["event"] in EVENTS)
    need(model.integer(record["sequence"], 1, 256))
    for key in ("previous_sha256", "plan_sha256", "sha256"):
        need(model.matches(record[key], model.HASH))
    need(record["target_sha256"] is None or model.matches(record["target_sha256"], model.HASH))
    need(record["error_code"] is None or (type(record["error_code"]) is str
                                          and record["error_code"] in model.ERROR_CODES))
    need(record["intent_kind"] is None or (type(record["intent_kind"]) is str
                                           and record["intent_kind"] in {"create", "observe",
                                                                         "terminate"}))
    need(model.integer(record["intent_sequence"], 0, 64))
    need(model.integer(record["intent_deadline_ns"]))
    if record["intent_kind"] is None:
        need(record["intent_sequence"] == record["intent_deadline_ns"] == 0)
    else:
        need(record["event"] == "begin_" + record["intent_kind"]
             and record["intent_sequence"] > 0 and record["intent_deadline_ns"] > 0
             and record["error_code"] is None)
    snapshot = record["snapshot"]
    need(type(snapshot) is dict and set(snapshot) == SNAPSHOT_KEYS)
    need(type(snapshot["state"]) is str and snapshot["state"] in STATES)
    for key in ("first_fault", "stop_reason"):
        need(snapshot[key] is None or (type(snapshot[key]) is str
                                      and snapshot[key] in model.ERROR_CODES))
    for key, maximum in COUNTERS.items():
        need(model.integer(snapshot[key], 0, maximum))
    for key in BOOLEANS:
        need(type(snapshot[key]) is bool)
    need(snapshot["response_capacity_reserved_bytes"] == snapshot["request_reservations"] * 65536)
    need(snapshot["create_attempts_modeled"] + snapshot["termination_attempts_modeled"]
         + snapshot["poll_attempts_modeled"] == snapshot["request_reservations"])
    need(snapshot["responses_consumed"] <= snapshot["request_reservations"])
    need(snapshot["instance_bound_in_model"] == (record["target_sha256"] is not None))
    need(hmac.compare_digest(record["sha256"], digest({
        key: value for key, value in record.items() if key != "sha256"
    })))


def verify_journal(raw, expected_plan_sha256, expected_tail_sha256):
    """Verify a supplied byte string against EXTERNAL plan/tail pins, never replay it.

    This is structural evidence, not approver authentication or a close/dispatch
    receipt. Without an external final-tail pin a removed valid suffix is invisible.
    No path is opened, returned target is reconstructed or execution API supplied.
    """
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result)
            result[key] = value
        return result

    def reject_constant(_value):
        raise ControllerError("journal_evidence_rejected")

    try:
        need(type(raw) is bytes and 0 < len(raw) <= 256 * 4096 and raw.endswith(b"\n"))
        need(model.matches(expected_plan_sha256, model.HASH)
             and model.matches(expected_tail_sha256, model.HASH))
        lines = raw.splitlines(keepends=True)
        need(1 <= len(lines) <= 256)
        previous = ZERO
        last = None
        for sequence, line in enumerate(lines, 1):
            need(len(line) <= 4096 and line.endswith(b"\n"))
            record = json.loads(line.decode("ascii"), object_pairs_hook=pairs,
                                parse_constant=reject_constant)
            _valid_record(record)
            need(line == canonical(record) + b"\n")
            need(record["sequence"] == sequence and record["previous_sha256"] == previous
                 and record["plan_sha256"] == expected_plan_sha256)
            need((record["event"] == "initialized") == (sequence == 1))
            if last is not None:
                need(last["event"] != "close")
                for key in COUNTERS:
                    need(record["snapshot"][key] >= last["snapshot"][key])
                if last["target_sha256"] is not None:
                    need(record["target_sha256"] == last["target_sha256"])
            previous, last = record["sha256"], record
        need(hmac.compare_digest(previous, expected_tail_sha256))
        return {
            "schema_version": "lumi.sft-host-lifecycle-journal-audit.v0.1",
            "structural_chain_verified": True,
            "record_count": len(lines),
            "final_close_record_present": last["event"] == "close",
            "model_terminal_recorded": last["snapshot"]["terminal_observation_modeled"],
            "dispatch_or_resume_authorized": False,
            "provider_termination_verified": False,
            "cloud_billing_stopped": False,
            "execution_authorized": False,
            "production_eligible": False,
        }
    except Exception:
        raise ControllerError("journal_evidence_rejected") from None


class DurableLifecycle:
    """One owning Python thread/process, one filesystem claim, no live transport.

    Clocks are sampled before each model operation and after durable I/O. Injected
    clocks are trusted test/controller dependencies, not independent time proofs.
    A sync that consumes a decision's validity window withholds that decision.
    A storage failure closes ALL wrapper operations, never just the failed call.
    """

    def __init__(self, root_fd, plan_bytes, expected_sha256, *, clock_ns=None, unix_seconds=None):
        self._plan = model.parse_plan(plan_bytes, expected_sha256)
        self._model = model.LifecycleModel(plan_bytes, expected_sha256)
        self._durable_snapshot = self._model.report()
        self._clock_ns = time.monotonic_ns if clock_ns is None else clock_ns
        self._unix_seconds = (lambda: time.time_ns() // model.NS) if unix_seconds is None else (
            unix_seconds)
        need(callable(self._clock_ns) and callable(self._unix_seconds), "controller_rejected")
        self._owner_pid = os.getpid()
        self._owner_thread = threading.current_thread()
        self._plan_sha = expected_sha256
        self._tail = ZERO
        self._records = 0
        self._last_clock = None
        self._failed = None
        self._busy = True
        self._closed = False
        self._journal = None
        try:
            self._clock()
            self._journal = journal.Journal(root_fd, self._plan["attempt_id"])
            self._persist("initialized")
            self._after_sync()
        except BaseException:
            if self._journal is not None:
                self._journal.close()
            raise ControllerError("controller_rejected") from None
        finally:
            self._busy = False

    def _poison(self, code):
        self._failed = self._failed or code
        raise ControllerError(self._failed) from None

    def _clock(self):
        try:
            value = self._clock_ns()
        except Exception:
            self._poison("clock_rejected")
        if self._failed is not None:
            raise ControllerError(self._failed)
        if not model.integer(value) or (self._last_clock is not None and value < self._last_clock):
            self._poison("clock_rejected")
        self._last_clock = value
        return value

    def _enter(self):
        if self._closed:
            raise ControllerError("controller_closed")
        if self._failed is not None:
            raise ControllerError("controller_failed")
        if (self._busy or os.getpid() != self._owner_pid
                or threading.current_thread() is not self._owner_thread):
            self._poison("controller_rejected")
        self._busy = True

    def _persist(self, event, *, intent=None, error_code=None):
        snapshot = self._model.report()
        identifier = self._model._instance
        record = {
            "schema_version": SCHEMA, "sequence": self._records + 1,
            "previous_sha256": self._tail, "plan_sha256": self._plan_sha,
            "event": event, "error_code": error_code,
            "intent_kind": None if intent is None else intent.kind,
            "intent_sequence": 0 if intent is None else intent.sequence,
            "intent_deadline_ns": 0 if intent is None else intent.deadline_ns,
            "target_sha256": (None if identifier is None else
                              hashlib.sha256(identifier.encode("ascii")).hexdigest()),
            "snapshot": {key: snapshot[key] for key in SNAPSHOT_KEYS},
        }
        record["sha256"] = digest(record)
        try:
            _valid_record(record)
            raw = canonical(record) + b"\n"
            self._journal.append(raw)
        except Exception:
            self._poison("journal_rejected")
        if self._failed is not None:
            raise ControllerError(self._failed)
        self._records += 1
        self._tail = record["sha256"]
        self._durable_snapshot = snapshot

    def _after_sync(self, intent=None):
        if intent is not None and intent.kind == "create":
            try:
                wall = self._unix_seconds()
            except Exception:
                self._poison("clock_rejected")
            if not model.integer(wall, 1, 2**53 - 1) or not (
                self._plan["create_not_before_unix"] <= wall < self._plan["create_expires_unix"]
            ):
                self._poison("authorization_window_closed")
        now = self._clock()
        start = self._model._start
        if start is None:
            return
        if now >= start + self._plan["maximum_lifecycle_seconds"] * model.NS:
            self._poison("durability_deadline_rejected")
        if intent is not None and now >= intent.deadline_ns:
            self._poison("durability_deadline_rejected")
        if self._model.report()["simulated_work_window_open"] and now >= start + (
            self._plan["maximum_lifecycle_seconds"] - self._plan["shutdown_reserve_seconds"]
        ) * model.NS:
            self._poison("durability_deadline_rejected")

    def _operate(self, event, sequence=None, **kwargs):
        self._enter()
        try:
            now = self._clock()
            intent = None
            error = None
            try:
                if event == "begin_create":
                    wall = self._unix_seconds()
                    intent = self._model.begin_create(now_ns=now, now_unix=wall)
                elif event in {"begin_observe", "begin_terminate"}:
                    intent = getattr(self._model, event)(now_ns=now)
                elif event in {"receive", "fail_pending"}:
                    getattr(self._model, event)(sequence, now_ns=now, **kwargs)
                else:
                    getattr(self._model, event)(now_ns=now)
            except model.LifecycleError as exc:
                error = exc.code
            self._persist(event, intent=intent, error_code=error)
            self._after_sync(intent)
            if error is not None:
                raise model.LifecycleError(error) from None
        except (model.LifecycleError, ControllerError):
            raise
        except BaseException:
            self._poison("controller_rejected")
        finally:
            self._busy = False
        return self.report() if intent is None else intent

    def begin_create(self):
        return self._operate("begin_create")

    def begin_observe(self):
        return self._operate("begin_observe")

    def begin_terminate(self):
        return self._operate("begin_terminate")

    def receive(self, sequence, *, http_status, body):
        return self._operate("receive", sequence, http_status=http_status, body=body)

    def fail_pending(self, sequence):
        return self._operate("fail_pending", sequence)

    def tick(self):
        return self._operate("tick")

    def finish_work(self):
        return self._operate("finish_work")

    def close(self):
        if self._closed:
            return self.report()
        # Wrong-owner/reentrant close must not race an in-flight reservation.
        if (self._busy or os.getpid() != self._owner_pid
                or threading.current_thread() is not self._owner_thread):
            self._poison("controller_rejected")
        self._busy = True
        try:
            if self._failed is None:
                try:
                    self._model.tick(now_ns=self._clock())
                    self._persist("close")
                    self._after_sync()
                except BaseException:
                    self._failed = self._failed or "controller_rejected"
            closed = self._journal.close()
            if not closed or self._journal.summary()["failed"]:
                self._failed = self._failed or "journal_rejected"
            if self._failed is None:
                try:
                    self._after_sync()
                except BaseException:
                    self._failed = self._failed or "controller_rejected"
            self._closed = True
            return self.report()
        finally:
            self._busy = False

    def report(self):
        result = dict(self._durable_snapshot)
        result.update({
            "schema_version": REPORT_SCHEMA,
            "journal_records_synced": self._records,
            "journal_tail_sha256": self._tail,
            "controller_failed": self._failed is not None,
            "controller_failure_code": self._failed,
            "controller_closed": self._closed,
            "journal": None if self._journal is None else self._journal.summary(),
            "filesystem_claim_scope": "one_trusted_directory_and_attempt_only",
            "crash_replay_or_resume_enabled": False,
            "same_account_global_exclusivity_verified": False,
            "storage_power_loss_durability_verified": False,
            "journal_storage_encryption_verified": False,
            "model_snapshot_is_last_synced_record": self._records > 0,
        })
        if self._failed is not None or self._closed or self._busy:
            result["simulated_work_window_open"] = False
        if self._failed is not None or (
            self._closed and not result["terminal_observation_modeled"]
        ):
            result["reconciliation_required"] = True
        # Frozen model's private-I/O counters are synthetic: don't claim no journal writes.
        result.pop("private_artifact_writes", None)
        result["local_journal_write_capacity_reserved_bytes"] = (
            0 if self._journal is None else self._journal.summary()["capacity_reserved_bytes"])
        return result
