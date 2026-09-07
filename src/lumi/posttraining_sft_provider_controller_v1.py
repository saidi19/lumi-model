"""Explicit single-attempt provider dispatch library, not a launcher or approval.

Calling create/observe/terminate with real dependencies CAN perform provider I/O.
The frozen local-only plan is insufficient: a separately pinned externally issued
MAC authority is required. Supplied keys/pins do not establish human or account
identity. This successor is exercised only with synthetic authority and fake TLS.
No CLI, credential loader, restart, target override, sleep loop or worker launch.
"""

from __future__ import annotations

import os
import threading
import time

from . import posttraining_sft_dispatch_authority_v1 as authority
from . import posttraining_sft_durable_host_lifecycle_v1 as durable
from . import posttraining_sft_lambda_http_v1 as http
from . import posttraining_sft_lambda_https_v1 as net

ERROR_CODES = frozenset({"provider_controller_rejected", "controller_closed", "controller_failed",
                         "clock_rejected", "create_window_closed", "intent_rejected",
                         "durable_failure", "exchange_failed", "poll_too_soon",
                         "termination_required", "transport_summary_rejected"})
MODEL_FIELDS = frozenset({
    "state", "first_fault", "stop_reason", "events", "request_reservations",
    "responses_consumed", "create_attempts_modeled", "termination_attempts_modeled",
    "poll_attempts_modeled", "pending_request", "instance_bound_in_model",
    "termination_acknowledgment_modeled", "terminal_observation_modeled",
    "simulated_work_window_open", "elapsed_lifecycle_ns", "compute_projection_microusd",
    "noncompute_reserve_microusd", "projection_with_reserve_microusd",
    "lifecycle_deadline_breached", "reconciliation_required", "journal_records_synced",
    "journal_tail_sha256", "local_journal_write_capacity_reserved_bytes",
})
NETWORK_COUNTERS = {
    "dns_calls": 1, "tcp_connect_calls": 1, "tls_handshake_calls": 1, "send_calls": 1,
    "recv_calls": 256, "plaintext_response_bytes_received": 81921, "close_calls": 1,
}


class ProviderError(ValueError):
    def __init__(self, code="provider_controller_rejected"):
        self.code = code if type(code) is str and code in ERROR_CODES else (
            "provider_controller_rejected")
        super().__init__(self.code)


def need(condition, code="provider_controller_rejected"):
    if not condition:
        raise ProviderError(code) from None


class ProviderController:
    """One owning process/thread, one exclusive durable attempt, no automatic work.

    A fresh HTTPS object is created for each distinct reserved intent. A failed
    observe may lead to a NEW reserved termination; a failed termination may lead
    only to a NEW reserved observation. A failed sequence is never retried.
    Closing only releases local journal descriptors; it does NOT stop a host.
    """

    def __init__(self, root_fd, plan_bytes, plan_sha256, *, authority_bytes,
                 authority_sha256, authority_key, authority_key_sha256, launch_body,
                 token, clock_ns=None, unix_ns=None):
        self._pid = os.getpid()
        self._thread = threading.current_thread()
        self._busy = True
        self._closed = False
        self._failed = None
        self._last_error = None
        self._last_mono = self._last_wall = None
        self._last_sequence = self._dispatches = 0
        self._network = dict.fromkeys(NETWORK_COUNTERS, 0)
        self._network_complete = True
        self._durable = None
        self._token = self._launch = b""
        self._clock_ns = time.monotonic_ns if clock_ns is None else clock_ns
        self._unix_ns = time.time_ns if unix_ns is None else unix_ns
        failed = False
        try:
            need(callable(self._clock_ns) and callable(self._unix_ns))
            self._plan = authority.verify_authority(
                authority_bytes, authority_sha256, key=authority_key,
                expected_key_sha256=authority_key_sha256, plan_bytes=plan_bytes,
                plan_sha256=plan_sha256, token=token,
            )
            self._launch = http.validate_launch(
                launch_body, self._plan["launch_request_sha256"], self._plan["region"],
                self._plan["instance_type"],
            )
            self._token = token
            self._sample_mono()
            self._create_window()
            # All authority/launch validation precedes even the exclusive file claim.
            self._durable = durable.DurableLifecycle(
                root_fd, plan_bytes, plan_sha256, clock_ns=self._sample_mono,
                unix_seconds=self._wall_seconds,
            )
        except BaseException:
            failed = True
            self._failed = "provider_controller_rejected"
            self._token = self._launch = b""
            if self._durable is not None:
                try:
                    self._durable.close()
                except BaseException:
                    pass
        finally:
            authority_key = authority_bytes = token = launch_body = None
            self._busy = False
        if failed:
            raise ProviderError from None

    def __repr__(self):
        return "<ProviderController: redacted>"

    def _latch(self, code):
        self._failed = self._failed or code
        raise ProviderError(self._failed) from None

    def _owner(self):
        if os.getpid() != self._pid or threading.current_thread() is not self._thread:
            self._latch("provider_controller_rejected")

    def _sample_mono(self):
        self._owner()
        if self._failed is not None:
            raise ProviderError("controller_failed")
        try:
            value = self._clock_ns()
        except BaseException:
            self._latch("clock_rejected")
        self._owner()
        if self._failed is not None:
            raise ProviderError("controller_failed")
        if not durable.model.integer(value) or (
            self._last_mono is not None and value < self._last_mono
        ):
            self._latch("clock_rejected")
        self._last_mono = value
        return value

    def _sample_wall(self):
        self._owner()
        try:
            value = self._unix_ns()
        except BaseException:
            self._latch("clock_rejected")
        self._owner()
        if self._failed is not None:
            raise ProviderError("controller_failed")
        if not durable.model.integer(value, 1) or (
            self._last_wall is not None and value < self._last_wall
        ):
            self._latch("clock_rejected")
        self._last_wall = value
        return value

    def _wall_seconds(self):
        return self._sample_wall() // durable.model.NS

    def _create_window(self):
        wall = self._sample_wall()
        need(self._plan["create_not_before_unix"] * durable.model.NS <= wall
             < self._plan["create_expires_unix"] * durable.model.NS, "create_window_closed")

    def _transport_clock(self, kind, deadline):
        if kind == "create":
            # The transport samples this around DNS, TLS and send as well as entry.
            # A late create is ambiguous, even if its body contains an apparent ID.
            self._create_window()
        # Wall-clock acquisition can itself consume time; sample monotonic LAST.
        now = self._sample_mono()
        need(now < deadline, "intent_rejected")
        return now

    def _enter(self):
        self._owner()
        need(not self._closed, "controller_closed")
        need(self._failed is None, "controller_failed")
        if self._busy:
            self._latch("provider_controller_rejected")
        self._busy = True

    def _intent(self, intent, kind):
        need(type(intent) is durable.model.Intent and intent.kind == kind
             and type(intent.sequence) is int and intent.sequence == self._last_sequence + 1
             and intent.sequence <= self._plan["maximum_requests"]
             and type(intent.response_capacity) is int and intent.response_capacity == 65536
             and durable.model.integer(intent.deadline_ns, 1), "intent_rejected")
        if kind == "create":
            need(intent.instance_id is None, "intent_rejected")
        else:
            need(http.matches(intent.instance_id, http.ID)
                 and intent.instance_id not in self._plan["protected_instance_ids"],
                 "intent_rejected")
        self._last_sequence = intent.sequence

    def _collect_transport(self, transport):
        value = transport.summary()
        need(type(value) is dict, "transport_summary_rejected")
        for name, maximum in NETWORK_COUNTERS.items():
            need(durable.model.integer(value.get(name), 0, maximum),
                 "transport_summary_rejected")
        for name in NETWORK_COUNTERS:
            self._network[name] += value[name]

    def _dispatch(self, kind):
        self._enter()
        intent = transport = prepared = reply = None
        valid_intent = outcome_started = False
        error = None
        try:
            self._sample_mono()
            intent = getattr(self._durable, "begin_" + kind)()
            self._intent(intent, kind)
            valid_intent = True
            if kind == "create":
                prepared = http.prepare_request(
                    kind, token=self._token, launch_body=self._launch,
                    expected_launch_sha256=self._plan["launch_request_sha256"],
                    region=self._plan["region"], instance_type=self._plan["instance_type"],
                )
            else:
                # Only the freshly persisted intent provides a target, never the caller.
                prepared = http.prepare_request(kind, token=self._token,
                                                instance_id=intent.instance_id)
            need(type(prepared) is http.Request, "exchange_failed")
            transport = net.LambdaHttps(clock_ns=lambda: self._transport_clock(
                intent.kind, intent.deadline_ns,
            ))
            self._transport_clock(kind, intent.deadline_ns)
            need(self._dispatches < self._plan["maximum_requests"], "intent_rejected")
            self._dispatches += 1
            reply = transport.exchange(prepared, deadline_ns=intent.deadline_ns)
            need(type(reply) is net.Reply and type(reply.status) is int and reply.status == 200
                 and type(reply.body) is bytes and len(reply.body) <= 65536, "exchange_failed")
            self._sample_mono()
            # Once receipt consumption starts it can NEVER be followed by fail_pending.
            # A persistence failure here is absorbing, not an HTTP retry opportunity.
            outcome_started = True
            self._durable.receive(intent.sequence, http_status=reply.status, body=reply.body)
            self._last_error = None
        except durable.model.LifecycleError as exc:
            error = exc.code if (intent is None and not valid_intent and not outcome_started
                                 and exc.code in {"poll_too_soon", "termination_required"}) else (
                                     "durable_failure")
            if error == "durable_failure":
                self._failed = self._failed or error
        except BaseException:
            error = "exchange_failed"
            if valid_intent and not outcome_started and self._failed is None:
                outcome_started = True
                try:
                    self._durable.fail_pending(intent.sequence)
                except BaseException:
                    self._failed = self._failed or "durable_failure"
            else:
                self._failed = self._failed or "durable_failure"
        finally:
            if transport is not None:
                try:
                    self._collect_transport(transport)
                except BaseException:
                    self._network_complete = False
                    self._failed = self._failed or "transport_summary_rejected"
            if type(prepared) is http.Request:
                # Also clear a compiled credential if expiry prevented exchange.claim().
                prepared._raw = b""
                prepared._used = True
            prepared = reply = intent = transport = None
            self._last_error = error
            self._busy = False
        if self._failed is not None:
            raise ProviderError(self._failed) from None
        if error is not None:
            raise ProviderError(error) from None
        return self.report()

    def create(self):
        return self._dispatch("create")

    def observe(self):
        return self._dispatch("observe")

    def terminate(self):
        return self._dispatch("terminate")

    def _local(self, operation):
        self._enter()
        failed = False
        try:
            getattr(self._durable, operation)()
            self._sample_mono()
        except BaseException:
            self._failed = self._failed or "durable_failure"
            failed = True
        finally:
            self._busy = False
        if failed:
            raise ProviderError(self._failed) from None
        return self.report()

    def finish_work(self):
        return self._local("finish_work")

    def tick(self):
        return self._local("tick")

    def close(self):
        self._owner()
        if self._busy:
            self._latch("provider_controller_rejected")
        if self._closed:
            return self.report()
        self._busy = True
        try:
            result = self._durable.close()
            if result["controller_failed"]:
                self._failed = self._failed or "durable_failure"
        except BaseException:
            self._failed = self._failed or "durable_failure"
        finally:
            self._token = self._launch = b""
            self._closed = True
            self._busy = False
        return self.report()

    def report(self):
        """Last synchronized modeled state and app counters, never a worker admission.

        No clock/I/O occurs here, so a previously open modeled window is a snapshot,
        not a fresh lease. Actual work, physical network and billing remain unproved.
        """
        value = self._durable.report()
        result = {key: value[key] for key in MODEL_FIELDS}
        result.update({
            "schema_version": "lumi.sft-provider-controller-report.v0.1",
            "controller_failed": self._failed is not None or value["controller_failed"],
            "controller_failure_code": self._failed,
            "controller_closed": self._closed,
            "controller_busy": self._busy,
            "last_operation_error": self._last_error,
            "authority_mac_and_bindings_verified": True,
            "dispatch_calls": self._dispatches,
            "application_read_capacity_reserved_bytes": self._dispatches * 81921,
            "application_write_capacity_reserved_bytes": self._dispatches * 8192,
            "network_counters_complete": self._network_complete and not self._busy,
            **{key: count if self._network_complete and not self._busy else None
               for key, count in self._network.items()},
            "same_account_global_exclusivity_verified": False,
            "rights_owner_identity_authenticated": False,
            "provider_account_verified": False,
            "provider_termination_verified": False,
            "cloud_billing_stopped": False,
            "physical_network_byte_cap_enforced": False,
            "hard_wall_time_enforced": False,
            "external_controller_survival_verified": False,
            "execution_authorized": False,
            "real_sft_execution_permitted": False,
            "production_eligible": False,
            "full_private_integration_passed": False,
            "crash_replay_or_resume_enabled": False,
        })
        if self._failed is not None or self._closed or self._busy:
            result["simulated_work_window_open"] = False
        if self._failed is not None or (
            self._closed and not result["terminal_observation_modeled"]
        ):
            result["reconciliation_required"] = True
        return result
