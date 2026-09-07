"""Synthetic provider bridge tests; no real credentials, DNS, sockets or CA store.

Every controller operation runs with DNS, socket creation and TLS construction
globally faked. The only real I/O is an owned temporary 0700 journal directory.
Mocked authority and provider receipts are not live authorization or cloud proof.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_test_sft_provider_controller_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
bridge = importlib.import_module(PACKAGE + ".posttraining_sft_provider_controller_v1")
net = importlib.import_module(PACKAGE + ".posttraining_sft_lambda_https_v1")
durable = importlib.import_module(PACKAGE + ".posttraining_sft_durable_host_lifecycle_v1")
http = net.http
NS = 1_000_000_000
OWNED, PROTECTED, OTHER = "a" * 32, "b" * 32, "9" * 32
ATTEMPT = "c" * 64
TOKEN = b"SYNTHETIC-NOT-A-CREDENTIAL"
KEY = b"SYNTHETIC-KEY-NOT-FOR-REAL-USE!!"
AUTH_SCHEMA = "lumi.sft-lambda-dispatch-authority.v0.1"


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def launch():
    return encode({
        "region_name": "us-east-1", "instance_type_name": "gpu_8x_a100_80gb_sxm4",
        "ssh_key_names": ["Synthetic Key"],
        "image": {"id": "12345678-1234-1234-1234-123456789abc"},
    })


def plan(**changes):
    return {
        "schema_version": durable.model.PLAN_SCHEMA, "mode": "local_decision_model_only",
        "attempt_id": ATTEMPT, "account_scope_sha256": "d" * 64,
        "approval_sha256": "e" * 64, "launch_request_sha256": sha(launch()),
        "protected_instance_ids": [PROTECTED], "region": "us-east-1",
        "instance_type": "gpu_8x_a100_80gb_sxm4", "gpu_count": 8,
        "total_host_hourly_rate_microusd": 22_320_000,
        "maximum_cost_microusd": 150_000_000, "noncompute_reserve_microusd": 10_000_000,
        "maximum_lifecycle_seconds": 14400, "shutdown_reserve_seconds": 600,
        "create_not_before_unix": 999, "create_expires_unix": 2000, "maximum_requests": 8,
        **changes,
    }


def authority(value, raw_plan, **changes):
    unsigned = {
        "schema_version": AUTH_SCHEMA, "mode": "externally_issued_single_attempt_dispatch",
        "provider_host": "cloud.lambda.ai", "plan_sha256": sha(raw_plan),
        "account_scope_sha256": value["account_scope_sha256"],
        "approval_sha256": value["approval_sha256"],
        "launch_request_sha256": value["launch_request_sha256"],
        "credential_sha256": sha(TOKEN),
        "create_not_before_unix": value["create_not_before_unix"],
        "create_expires_unix": value["create_expires_unix"],
        "maximum_requests": value["maximum_requests"],
        "cleanup_scope": "only_fresh_create_bound_id_within_lifecycle_caps",
        **changes,
    }
    mac = hmac.new(KEY, AUTH_SCHEMA.encode("ascii") + b"\0" + encode(unsigned),
                   hashlib.sha256).hexdigest()
    return encode({**unsigned, "mac_sha256": mac})


def instance(status="active", identifier=OWNED):
    return {
        "id": identifier, "status": status, "region": {"name": "us-east-1"},
        "instance_type": {"name": "gpu_8x_a100_80gb_sxm4", "price_cents_per_hour": 2232,
                          "specs": {"gpus": 8}},
        "ip": "192.0.2.123", "name": "SECRET", "jupyter_token": "SECRET",
    }


def framed(value, *, status=b"200 OK"):
    body = value if type(value) is bytes else encode(value)
    return (b"HTTP/1.1 " + status + b"\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(body)).encode("ascii") + b"\r\n\r\n" + body)


class Fixture(unittest.TestCase):
    def setUp(self):
        # Install all network/CA guards before constructing even a rejected controller.
        self.dns = self.enterContext(patch.object(net.socket, "getaddrinfo"))
        self.sockets = self.enterContext(patch.object(net.socket, "socket"))
        self.contexts = self.enterContext(patch.object(net.ssl, "SSLContext"))
        self.default_contexts = self.enterContext(patch.object(
            net.ssl, "create_default_context", side_effect=AssertionError("forbidden TLS factory"),
        ))
        temporary = tempfile.TemporaryDirectory(prefix="lumi-provider-controller-synthetic-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.directory.chmod(0o700)
        self.root_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.addCleanup(os.close, self.root_fd)
        self.path = self.directory / (ATTEMPT + ".lifecycle.jsonl")
        self.now = 0
        self.wall = 1000 * NS
        self.clock = Mock(side_effect=lambda: self.now)
        self.unix = Mock(side_effect=lambda: self.wall)
        self.value = plan()
        self.raw_plan = encode(self.value)
        self.raw_authority = authority(self.value, self.raw_plan)
        self.fragments = []
        self.raw = Mock(name="synthetic_tcp")
        self.tls = Mock(name="synthetic_tls")
        self.context = Mock(name="synthetic_context")
        self.context.verify_flags = 0
        self.raw.get_inheritable.return_value = False
        self.tls.get_inheritable.return_value = False
        self.context.wrap_socket.return_value = self.tls
        self.tls.selected_alpn_protocol.return_value = "http/1.1"
        self.tls.version.return_value = "TLSv1.3"
        self.tls.getpeercert.return_value = {"synthetic": True}
        self.tls.send.side_effect = len
        self.tls.recv.side_effect = self.receive
        self.sockets.return_value = self.raw
        self.contexts.return_value = self.context
        self.dns.return_value = [(net.socket.AF_INET, net.socket.SOCK_STREAM,
                                  net.socket.IPPROTO_TCP, "", ("8.8.8.8", 443))]

    def receive(self, size):
        self.assertTrue(self.fragments, "unexpected receive without synthetic response")
        fragment = self.fragments.pop(0)
        if len(fragment) > size:
            self.fragments.insert(0, fragment[size:])
        return fragment[:size]

    def queue(self, value, **kwargs):
        self.fragments.extend([framed(value, **kwargs), b""])

    def options(self):
        return {
            "authority_bytes": self.raw_authority, "authority_sha256": sha(self.raw_authority),
            "authority_key": KEY, "authority_key_sha256": sha(KEY),
            "launch_body": launch(), "token": TOKEN,
            "clock_ns": self.clock, "unix_ns": self.unix,
        }

    def make(self, **options):
        result = bridge.ProviderController(self.root_fd, self.raw_plan, sha(self.raw_plan),
                                           **{**self.options(), **options})
        self.addCleanup(result.close)
        return result

    def rebind(self, **changes):
        self.value = plan(**changes)
        self.raw_plan = encode(self.value)
        self.raw_authority = authority(self.value, self.raw_plan)

    def bound(self):
        result = self.make()
        self.queue({"data": {"instance_ids": [OWNED]}})
        result.create()
        self.assertEqual(result.report()["state"], "bound")
        return result

    def active(self):
        result = self.bound()
        self.now = 1
        self.queue({"data": instance()})
        result.observe()
        self.assertTrue(result.report()["simulated_work_window_open"])
        return result

    def rejected(self, operation):
        with self.assertRaises(ValueError) as caught:
            operation()
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertNotIn(TOKEN.decode("ascii"), str(caught.exception))
        return caught.exception

    def records(self):
        return [json.loads(line) for line in self.path.read_bytes().splitlines()]

    def last_outcomes(self):
        return [row["event"] for row in self.records()
                if row["event"] in {"receive", "fail_pending"}]

    def boundaries(self, controller):
        report = controller.report()
        encoded = json.dumps(report)
        for secret in (TOKEN.decode("ascii"), KEY.decode("ascii"), OWNED, PROTECTED, ATTEMPT,
                       "d" * 64, "e" * 64, "SECRET", "192.0.2.123", str(self.directory)):
            self.assertNotIn(secret, encoded)
        for key in ("execution_authorized", "real_sft_execution_permitted", "production_eligible",
                    "full_private_integration_passed", "provider_account_verified",
                    "provider_termination_verified", "cloud_billing_stopped",
                    "physical_network_byte_cap_enforced", "hard_wall_time_enforced"):
            self.assertIs(report[key], False, key)
        for key in ("provider_calls_executed", "external_network_bytes", "private_artifact_writes"):
            self.assertNotIn(key, report, key)
        if report["controller_failed"] or report["controller_closed"]:
            self.assertFalse(report["simulated_work_window_open"])
        self.default_contexts.assert_not_called()
        return report


class AuthorityAndOwnershipTests(Fixture):
    def test_constructor_claims_only_private_journal_without_network(self):
        controller = self.make()
        report = self.boundaries(controller)
        self.assertEqual(report["state"], "prepared")
        self.assertEqual(report["request_reservations"], 0)
        self.assertEqual(report["dispatch_calls"], 0)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(len(self.records()), 1)
        self.dns.assert_not_called()
        self.sockets.assert_not_called()
        self.contexts.assert_not_called()

    def test_bad_authority_hash_rejected_before_journal_or_network(self):
        self.rejected(lambda: self.make(authority_sha256="0" * 64))
        self.assertEqual(list(self.directory.iterdir()), [])
        self.dns.assert_not_called()

    def test_forged_mac_rejected_even_when_document_hash_matches(self):
        value = json.loads(self.raw_authority)
        value["mac_sha256"] = "0" * 64
        raw = encode(value)
        self.rejected(lambda: self.make(authority_bytes=raw, authority_sha256=sha(raw)))
        self.assertEqual(list(self.directory.iterdir()), [])
        self.dns.assert_not_called()

    def test_wrong_authority_key_and_credential_pins_rejected_without_journal(self):
        cases = ({"authority_key_sha256": "0" * 64}, {"authority_key": b"x" * 32},
                 {"token": b"SYNTHETIC-DIFFERENT-CREDENTIAL"})
        for options in cases:
            with self.subTest(options=tuple(options)):
                self.rejected(lambda options=options: self.make(**options))
                self.assertEqual(list(self.directory.iterdir()), [])
        self.dns.assert_not_called()

    def test_signed_scope_substitution_does_not_authorize_other_plan(self):
        for changes in ({"account_scope_sha256": "1" * 64}, {"approval_sha256": "1" * 64},
                        {"provider_host": "synthetic.invalid"}, {"maximum_requests": 9},
                        {"plan_sha256": "1" * 64}, {"launch_request_sha256": "1" * 64}):
            with self.subTest(field=next(iter(changes))):
                raw = authority(self.value, self.raw_plan, **changes)
                self.rejected(lambda raw=raw: self.make(authority_bytes=raw,
                                                        authority_sha256=sha(raw)))
                self.assertEqual(list(self.directory.iterdir()), [])
        self.dns.assert_not_called()

    def test_launch_contract_rejected_before_journal_creation(self):
        for body in (b"{}", launch() + b" ", launch().replace(b"Synthetic Key", b"SECRET")):
            with self.subTest(length=len(body)):
                self.rejected(lambda body=body: self.make(launch_body=body))
                self.assertEqual(list(self.directory.iterdir()), [])
        self.dns.assert_not_called()

    def test_invalid_clock_callbacks_or_constructor_window_create_no_journal(self):
        for options in ({"clock_ns": None, "unix_ns": 42}, {"clock_ns": 42},
                        {"clock_ns": lambda: True}, {"unix_ns": lambda: -1},
                        {"unix_ns": lambda: 999 * NS - 1},
                        {"unix_ns": lambda: 2000 * NS}):
            with self.subTest(fields=tuple(options)):
                self.rejected(lambda options=options: self.make(**options))
                self.assertEqual(list(self.directory.iterdir()), [])
        self.dns.assert_not_called()

    def test_existing_attempt_cannot_be_reopened_as_fresh_controller(self):
        controller = self.make()
        controller.close()
        evidence = self.path.read_bytes()
        self.rejected(self.make)
        self.assertEqual(self.path.read_bytes(), evidence)
        self.dns.assert_not_called()

    def test_foreign_thread_cannot_dispatch_or_close_owned_journal(self):
        controller = self.make()
        errors = []

        def wrong_owner():
            try:
                controller.create()
            except ValueError:
                errors.append("rejected")
            self.rejected(controller.close)

        thread = threading.Thread(target=wrong_owner)
        thread.start()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ["rejected"])
        self.assertTrue(self.boundaries(controller)["controller_failed"])
        self.dns.assert_not_called()

    def test_foreign_process_identity_cannot_dispatch(self):
        controller = self.make()
        with patch.object(bridge.os, "getpid", return_value=os.getpid() + 1):
            self.rejected(controller.create)
        self.dns.assert_not_called()
        self.assertTrue(self.boundaries(controller)["controller_failed"])


class DispatchOrderingTests(Fixture):
    def test_slow_wall_sample_after_compile_cannot_release_stale_monotonic_deadline(self):
        controller = self.make()
        original = http.prepare_request

        def slow_wall():
            self.now = 15 * NS
            return self.wall

        def compiled(*args, **kwargs):
            prepared = original(*args, **kwargs)
            self.unix.side_effect = slow_wall
            return prepared

        with patch.object(http, "prepare_request", side_effect=compiled):
            self.rejected(controller.create)
        self.dns.assert_not_called()
        self.assertEqual(self.last_outcomes(), ["fail_pending"])
        self.assertTrue(self.boundaries(controller)["reconciliation_required"])

    def test_slow_wall_sample_after_handshake_cannot_release_stale_send_deadline(self):
        controller = self.make()

        def slow_wall():
            self.now = 15 * NS
            return self.wall

        def handshake():
            self.unix.side_effect = slow_wall

        self.tls.do_handshake.side_effect = handshake
        self.rejected(controller.create)
        self.tls.send.assert_not_called()
        self.tls.close.assert_called_once_with()
        self.assertEqual(self.last_outcomes(), ["fail_pending"])

    def test_logical_denial_during_receipt_is_absorbing_not_recoverable_poll_denial(self):
        controller = self.make()
        self.queue({"data": {"instance_ids": [OWNED]}})
        with patch.object(controller._durable, "receive", side_effect=(
            durable.model.LifecycleError("poll_too_soon")
        )), patch.object(controller._durable, "fail_pending") as failed:
            self.rejected(controller.create)
        failed.assert_not_called()
        self.assertTrue(self.boundaries(controller)["controller_failed"])
        self.rejected(controller.create)
        self.assertEqual(self.dns.call_count, 1)

    def test_logical_denial_after_intent_compilation_is_not_recoverable_state_denial(self):
        controller = self.make()
        with patch.object(http, "prepare_request", side_effect=(
            durable.model.LifecycleError("termination_required")
        )):
            self.rejected(controller.create)
        self.assertTrue(self.boundaries(controller)["controller_failed"])
        self.dns.assert_not_called()

    def test_post_receipt_counter_failure_cannot_release_work_or_double_consume(self):
        controller = self.make()
        self.queue({"data": {"instance_ids": [OWNED]}})
        with patch.object(net.LambdaHttps, "summary", side_effect=ValueError("SECRET")):
            self.rejected(controller.create)
        self.assertEqual(self.last_outcomes(), ["receive"])
        self.assertTrue(self.boundaries(controller)["controller_failed"])
        self.assertFalse(controller.report()["network_counters_complete"])
        self.assertIsNone(controller.report()["dns_calls"])
        self.rejected(controller.terminate)
        self.assertEqual(self.dns.call_count, 1)

    def test_reentry_during_receipt_fsync_withholds_result_without_failure_receipt(self):
        controller = self.make()
        self.queue({"data": {"instance_ids": [OWNED]}})
        original = durable.journal.os.fsync
        calls = 0

        def synced(fd):
            nonlocal calls
            calls += 1
            original(fd)
            if calls == 2:
                self.rejected(controller.terminate)

        with patch.object(durable.journal.os, "fsync", side_effect=synced):
            self.rejected(controller.create)
        self.assertEqual(self.last_outcomes(), ["receive"])
        self.assertTrue(self.boundaries(controller)["controller_failed"])
        self.assertEqual(self.dns.call_count, 1)

    def test_create_reservation_fsynced_before_dns_and_result_fsynced_before_release(self):
        controller = self.make()
        self.queue({"data": {"instance_ids": [OWNED]}})
        events = []
        original_sync = durable.journal.os.fsync

        def synced(fd):
            original_sync(fd)
            events.append("fsync")

        def resolved(*args):
            self.assertEqual(self.records()[-1]["event"], "begin_create")
            snapshot = self.records()[-1]["snapshot"]
            self.assertEqual(snapshot["request_reservations"], 1)
            self.assertTrue(snapshot["pending_request"])
            self.assertEqual(events, ["fsync"])
            events.append("dns")
            return self.dns.return_value

        self.dns.side_effect = resolved
        with patch.object(durable.journal.os, "fsync", side_effect=synced):
            controller.create()
        self.assertEqual(events, ["fsync", "dns", "fsync"])
        self.assertEqual(self.last_outcomes(), ["receive"])
        self.assertEqual(self.boundaries(controller)["dispatch_calls"], 1)

    def test_expired_create_window_precedes_dns(self):
        controller = self.make()
        self.wall = 2000 * NS
        self.rejected(controller.create)
        self.dns.assert_not_called()
        self.assertFalse(self.boundaries(controller)["simulated_work_window_open"])

    def test_expiry_after_reserved_request_compile_blocks_dns(self):
        controller = self.make()
        original = http.prepare_request

        def compiled(*args, **kwargs):
            result = original(*args, **kwargs)
            self.wall = 2000 * NS
            return result

        with patch.object(http, "prepare_request", side_effect=compiled):
            self.rejected(controller.create)
        self.dns.assert_not_called()
        self.assertEqual(self.boundaries(controller)["request_reservations"], 1)
        self.assertEqual(self.last_outcomes(), ["fail_pending"])

    def test_expiry_during_transport_construction_blocks_dns(self):
        controller = self.make()
        original = net.LambdaHttps

        def constructed(**kwargs):
            result = original(**kwargs)
            self.wall = 2000 * NS
            return result

        with patch.object(net, "LambdaHttps", side_effect=constructed):
            self.rejected(controller.create)
        self.dns.assert_not_called()
        self.assertEqual(self.last_outcomes(), ["fail_pending"])

    def test_expiry_inside_dns_blocks_socket_and_send(self):
        controller = self.make()

        def resolved(*args):
            self.wall = 2000 * NS
            return self.dns.return_value

        self.dns.side_effect = resolved
        self.rejected(controller.create)
        self.dns.assert_called_once()
        self.sockets.assert_not_called()
        self.tls.send.assert_not_called()
        self.assertEqual(self.last_outcomes(), ["fail_pending"])

    def test_expiry_inside_handshake_blocks_send_and_closes_once(self):
        controller = self.make()

        def handshake():
            self.wall = 2000 * NS

        self.tls.do_handshake.side_effect = handshake
        self.rejected(controller.create)
        self.tls.send.assert_not_called()
        self.tls.close.assert_called_once_with()
        self.assertEqual(self.last_outcomes(), ["fail_pending"])

    def test_create_compile_failure_consumes_reservation_without_dispatch(self):
        controller = self.make()
        with patch.object(http, "prepare_request", side_effect=http.HttpError("SECRET")):
            self.rejected(controller.create)
        self.assertEqual(self.last_outcomes(), ["fail_pending"])
        self.assertEqual(controller.report()["request_reservations"], 1)
        self.assertEqual(controller.report()["dispatch_calls"], 0)
        self.rejected(controller.create)
        self.dns.assert_not_called()

    def test_invalid_compiler_return_does_not_bypass_busy_cleanup(self):
        controller = self.make()
        with patch.object(http, "prepare_request", return_value=object()):
            self.rejected(controller.create)
        self.assertEqual(self.last_outcomes(), ["fail_pending"])
        self.assertFalse(controller.report()["controller_busy"])
        self.dns.assert_not_called()

    def test_factory_exception_consumes_one_failure_without_dns(self):
        controller = self.make()
        with patch.object(net, "LambdaHttps", side_effect=OSError("SECRET")):
            self.rejected(controller.create)
        self.assertEqual(self.last_outcomes(), ["fail_pending"])
        self.assertEqual(controller.report()["dispatch_calls"], 0)
        self.dns.assert_not_called()

    def test_pre_dispatch_journal_fsync_failure_never_reaches_dns(self):
        controller = self.make()
        with patch.object(durable.journal.os, "fsync", side_effect=OSError("SECRET")):
            self.rejected(controller.create)
        self.dns.assert_not_called()
        report = self.boundaries(controller)
        self.assertTrue(report["controller_failed"])
        self.assertTrue(report["reconciliation_required"])

    def test_create_receipt_fsync_failure_never_double_consumes_or_retries(self):
        controller = self.make()
        self.queue({"data": {"instance_ids": [OWNED]}})
        original = durable.journal.os.fsync
        calls = 0

        def synced(fd):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("SECRET")
            return original(fd)

        with patch.object(durable.journal.os, "fsync", side_effect=synced):
            self.rejected(controller.create)
        report = self.boundaries(controller)
        self.assertTrue(report["controller_failed"])
        self.assertTrue(report["reconciliation_required"])
        self.assertEqual(self.last_outcomes(), ["receive"])
        for action in (controller.create, controller.observe, controller.terminate):
            self.rejected(action)
        self.assertEqual(self.dns.call_count, 1)

    def test_fail_pending_receipt_fsync_failure_is_absorbing_not_repeated(self):
        controller = self.make()
        self.tls.send.side_effect = OSError("SECRET")
        original = controller._durable.fail_pending

        def failed(sequence):
            with patch.object(durable.journal.os, "fsync", side_effect=OSError("SECRET")):
                return original(sequence)

        with patch.object(controller._durable, "fail_pending", side_effect=failed) as consume:
            self.rejected(controller.create)
            consume.assert_called_once_with(1)
            self.rejected(controller.create)
            consume.assert_called_once_with(1)
        self.assertEqual(self.last_outcomes(), ["fail_pending"])
        self.assertTrue(self.boundaries(controller)["controller_failed"])
        self.assertEqual(self.dns.call_count, 1)

    def test_observation_receipt_fsync_sees_closed_work_window_until_return(self):
        controller = self.bound()
        self.queue({"data": instance()})
        original = durable.journal.os.fsync
        observed = []

        def synced(fd):
            observed.append(controller.report()["simulated_work_window_open"])
            return original(fd)

        with patch.object(durable.journal.os, "fsync", side_effect=synced):
            controller.observe()
        self.assertEqual(observed, [False, False])
        self.assertTrue(self.boundaries(controller)["simulated_work_window_open"])

    def test_reentrant_dns_operation_poisoned_without_second_dispatch(self):
        controller = self.make()

        def resolved(*args):
            self.rejected(controller.create)
            return self.dns.return_value

        self.dns.side_effect = resolved
        self.rejected(controller.create)
        self.assertEqual(self.dns.call_count, 1)
        self.tls.send.assert_not_called()
        self.assertTrue(self.boundaries(controller)["controller_failed"])

    def test_bad_clock_before_dispatch_has_no_network(self):
        controller = self.make()
        self.now = -1
        self.rejected(controller.create)
        self.dns.assert_not_called()
        self.assertTrue(self.boundaries(controller)["controller_failed"])

    def test_clock_regression_inside_dns_prevents_send(self):
        self.now = 5
        controller = self.make()

        def resolved(*args):
            self.now = 4
            return self.dns.return_value

        self.dns.side_effect = resolved
        self.rejected(controller.create)
        self.tls.send.assert_not_called()
        self.assertTrue(self.boundaries(controller)["controller_failed"])


class LifecycleAndFaultTests(Fixture):
    def test_successful_four_request_lifecycle_has_exact_owned_paths_and_fresh_transports(self):
        controller = self.make()
        original = net.LambdaHttps
        with patch.object(net, "LambdaHttps", wraps=original) as factories:
            self.queue({"data": {"instance_ids": [OWNED]}})
            controller.create()
            self.queue({"data": instance()})
            controller.observe()
            controller.finish_work()
            self.queue({"data": {"terminated_instances": [instance("terminating")]}})
            controller.terminate()
            self.now = 15 * NS
            self.queue({"data": instance("terminated")})
            controller.observe()
        report = self.boundaries(controller)
        self.assertEqual(report["state"], "terminal_observed")
        self.assertFalse(report["simulated_work_window_open"])
        self.assertEqual(report["dispatch_calls"], 4)
        self.assertEqual(report["request_reservations"], 4)
        self.assertEqual(factories.call_count, 4)
        self.assertEqual(self.last_outcomes(), ["receive"] * 4)
        sent = [call.args[0] for call in self.tls.send.call_args_list]
        self.assertTrue(sent[0].startswith(b"POST /api/v1/instance-operations/launch HTTP/1.1"))
        self.assertTrue(sent[1].startswith(b"GET /api/v1/instances/" + OWNED.encode() + b" "))
        self.assertTrue(sent[2].startswith(b"POST /api/v1/instance-operations/terminate "))
        self.assertEqual(sent[2].split(b"\r\n\r\n")[1], encode({"instance_ids": [OWNED]}))
        self.assertTrue(sent[3].startswith(b"GET /api/v1/instances/" + OWNED.encode() + b" "))
        self.assertNotIn(PROTECTED.encode(), b"".join(sent))
        controller.close()
        self.assertFalse(controller.report()["reconciliation_required"])

    def test_create_send_failure_is_ambiguous_and_never_retried(self):
        controller = self.make()
        self.tls.send.side_effect = OSError("SECRET")
        self.rejected(controller.create)
        self.assertEqual(controller.report()["state"], "reconciliation_required")
        self.assertEqual(self.last_outcomes(), ["fail_pending"])
        for action in (controller.create, controller.observe, controller.terminate):
            self.rejected(action)
        self.assertEqual(self.tls.send.call_count, 1)
        self.assertEqual(self.dns.call_count, 1)

    def test_create_short_send_is_not_resent(self):
        controller = self.make()
        self.tls.send.side_effect = lambda raw: len(raw) - 1
        self.rejected(controller.create)
        self.tls.send.assert_called_once()
        self.assertEqual(self.last_outcomes(), ["fail_pending"])
        self.assertTrue(self.boundaries(controller)["reconciliation_required"])

    def test_malformed_create_body_does_not_become_target_or_transport_retry(self):
        controller = self.make()
        self.queue(b'{"data":{"instance_ids":["SECRET"]}}')
        controller.create()
        self.assertEqual(self.last_outcomes(), ["receive"])
        self.assertEqual(controller.report()["state"], "reconciliation_required")
        self.rejected(controller.terminate)
        self.assertEqual(self.dns.call_count, 1)
        self.boundaries(controller)

    def test_protected_create_id_never_authorizes_terminate(self):
        controller = self.make()
        self.queue({"data": {"instance_ids": [PROTECTED]}})
        controller.create()
        self.assertEqual(controller.report()["state"], "reconciliation_required")
        self.rejected(controller.terminate)
        self.assertEqual(self.dns.call_count, 1)
        self.assertEqual(self.last_outcomes(), ["receive"])

    def test_observe_transport_failure_allows_only_new_reserved_targeted_cleanup(self):
        controller = self.bound()
        self.tls.send.side_effect = OSError("SECRET")
        self.rejected(controller.observe)
        self.assertEqual(controller.report()["state"], "termination_due")
        self.assertFalse(controller.report()["controller_failed"])
        self.tls.send.side_effect = len
        self.queue({"data": {"terminated_instances": [instance("terminating")]}})
        controller.terminate()
        self.assertEqual(controller.report()["request_reservations"], 3)
        self.assertEqual(controller.report()["dispatch_calls"], 3)
        self.assertEqual(self.last_outcomes(), ["receive", "fail_pending", "receive"])
        self.assertEqual(self.dns.call_count, 3)
        sent = self.tls.send.call_args.args[0]
        self.assertEqual(sent.split(b"\r\n\r\n")[1], encode({"instance_ids": [OWNED]}))

    def test_terminate_failure_allows_new_reserved_observe_without_retry(self):
        controller = self.bound()
        self.tls.send.side_effect = OSError("SECRET")
        self.rejected(controller.terminate)
        self.assertEqual(controller.report()["state"], "termination_observing")
        self.assertEqual(self.dns.call_count, 2)
        self.tls.send.side_effect = len
        self.queue({"data": instance("terminated")})
        controller.observe()
        self.assertEqual(controller.report()["state"], "terminal_observed")
        self.assertEqual(self.last_outcomes(), ["receive", "fail_pending", "receive"])
        self.assertEqual(controller.report()["request_reservations"], 3)

    def test_forbidden_second_terminate_is_not_dispatched_and_poisoned(self):
        controller = self.bound()
        self.tls.send.side_effect = OSError("SECRET")
        self.rejected(controller.terminate)
        self.rejected(controller.terminate)
        self.assertEqual(self.dns.call_count, 2)
        self.assertEqual(controller.report()["request_reservations"], 2)
        self.assertTrue(self.boundaries(controller)["controller_failed"])

    def test_termination_ack_with_terminated_status_is_not_final_get_proof(self):
        controller = self.bound()
        self.queue({"data": {"terminated_instances": [instance("terminated")]}})
        controller.terminate()
        self.assertEqual(controller.report()["state"], "termination_observing")
        self.assertFalse(controller.report()["terminal_observation_modeled"])
        self.assertFalse(controller.report()["simulated_work_window_open"])
        self.boundaries(controller)

    def test_target_mismatch_observation_requires_exact_bound_cleanup(self):
        controller = self.bound()
        self.queue({"data": instance(identifier=OTHER)})
        controller.observe()
        self.assertEqual(controller.report()["state"], "termination_due")
        self.queue({"data": {"terminated_instances": [instance("terminating")]}})
        controller.terminate()
        sent = self.tls.send.call_args.args[0]
        self.assertIn(OWNED.encode(), sent)
        self.assertNotIn(OTHER.encode(), sent)

    def test_http_error_is_one_failed_outcome_not_terminal_evidence(self):
        controller = self.bound()
        self.queue(b"{}", status=b"404 Not Found")
        self.rejected(controller.observe)
        self.assertEqual(controller.report()["state"], "termination_due")
        self.assertEqual(self.last_outcomes(), ["receive", "fail_pending"])
        self.assertFalse(controller.report()["terminal_observation_modeled"])

    def test_cleanup_can_continue_after_create_authority_window_expires(self):
        controller = self.bound()
        self.wall = 2001 * NS
        self.queue({"data": {"terminated_instances": [instance("terminating")]}})
        controller.terminate()
        self.queue({"data": instance("terminated")})
        controller.observe()
        self.assertEqual(controller.report()["state"], "terminal_observed")
        self.assertEqual(controller.report()["request_reservations"], 3)

    def test_minimum_request_cap_preserves_two_cleanup_slots_across_fresh_transports(self):
        self.rebind(maximum_requests=4)
        controller = self.active()
        self.now = 15 * NS + 1
        self.rejected(controller.observe)
        self.assertEqual(controller.report()["state"], "termination_due")
        self.assertEqual(self.dns.call_count, 2)
        self.queue({"data": {"terminated_instances": [instance("terminating")]}})
        controller.terminate()
        self.queue({"data": instance("terminated")})
        controller.observe()
        self.assertEqual(controller.report()["request_reservations"], 4)
        self.assertEqual(controller.report()["dispatch_calls"], 4)
        self.assertEqual(self.dns.call_count, 4)
        self.rejected(controller.observe)
        self.assertEqual(self.dns.call_count, 4)

    def test_exhausted_cap_stops_nonterminal_observation_across_fresh_transports(self):
        self.rebind(maximum_requests=4)
        controller = self.active()
        self.queue({"data": {"terminated_instances": [instance("terminating")]}})
        controller.terminate()
        self.now = 15 * NS + 1
        self.queue({"data": instance("terminating")})
        controller.observe()
        self.now = 30 * NS + 1
        self.rejected(controller.observe)
        self.assertEqual(self.dns.call_count, 4)
        self.assertEqual(controller.report()["request_reservations"], 4)
        self.assertEqual(controller.report()["dispatch_calls"], 4)
        self.assertTrue(self.boundaries(controller)["reconciliation_required"])

    def test_lifecycle_deadline_stops_new_dispatch_even_if_create_window_is_open(self):
        self.rebind(maximum_lifecycle_seconds=61, shutdown_reserve_seconds=60)
        controller = self.bound()
        self.now = 61 * NS
        self.rejected(controller.terminate)
        self.assertEqual(self.dns.call_count, 1)
        self.assertTrue(self.boundaries(controller)["reconciliation_required"])

    def test_poll_too_soon_has_no_dispatch_and_later_poll_is_valid(self):
        controller = self.active()
        self.now = 2
        self.rejected(controller.observe)
        self.assertEqual(self.dns.call_count, 2)
        self.assertFalse(controller.report()["controller_failed"])
        self.now = 15 * NS + 1
        self.queue({"data": instance()})
        controller.observe()
        self.assertEqual(self.dns.call_count, 3)

    def test_close_active_requires_reconciliation_without_provider_action(self):
        controller = self.active()
        self.assertTrue(controller.close())
        report = self.boundaries(controller)
        self.assertTrue(report["controller_closed"])
        self.assertTrue(report["reconciliation_required"])
        self.assertEqual(self.dns.call_count, 2)
        self.rejected(controller.observe)
        self.assertEqual(self.dns.call_count, 2)

    def test_tick_and_finish_work_are_local_durable_transitions_not_provider_dispatch(self):
        controller = self.active()
        controller.tick()
        controller.finish_work()
        self.assertEqual(self.dns.call_count, 2)
        self.assertEqual(controller.report()["state"], "termination_due")
        self.assertEqual([r["event"] for r in self.records()][-2:], ["tick", "finish_work"])

    def test_transport_close_failure_records_failure_without_retry(self):
        controller = self.make()
        self.queue({"data": {"instance_ids": [OWNED]}})
        self.tls.close.side_effect = OSError("SECRET")
        self.rejected(controller.create)
        self.assertEqual(self.last_outcomes(), ["fail_pending"])
        self.tls.close.assert_called_once()
        self.rejected(controller.create)
        self.assertEqual(self.dns.call_count, 1)

    def test_close_releases_sensitive_references_and_is_idempotent(self):
        controller = self.make()
        self.assertEqual(repr(controller), "<ProviderController: redacted>")
        controller.close()
        evidence = self.path.read_bytes()
        controller.close()
        self.assertEqual(self.path.read_bytes(), evidence)
        self.assertEqual(controller._token, b"")
        self.assertEqual(controller._launch, b"")
        self.dns.assert_not_called()

    def test_failed_dns_still_retains_full_prepaid_request_capacity(self):
        controller = self.make()
        self.dns.side_effect = OSError("SECRET")
        self.rejected(controller.create)
        report = self.boundaries(controller)
        self.assertEqual(report["dispatch_calls"], 1)
        self.assertEqual(report["dns_calls"], 1)
        self.assertEqual(report["send_calls"], 0)
        self.assertEqual(report["plaintext_response_bytes_received"], 0)
        self.assertEqual(report["application_read_capacity_reserved_bytes"], 81921)
        self.assertEqual(report["application_write_capacity_reserved_bytes"], 8192)

    def test_network_counters_accumulate_without_inherited_zero_network_claims(self):
        controller = self.active()
        report = self.boundaries(controller)
        self.assertEqual(report["dispatch_calls"], 2)
        self.assertEqual(report["dns_calls"], 2)
        self.assertEqual(report["tcp_connect_calls"], 2)
        self.assertEqual(report["tls_handshake_calls"], 2)
        self.assertEqual(report["send_calls"], 2)
        self.assertEqual(report["close_calls"], 2)
        self.assertEqual(report["application_read_capacity_reserved_bytes"], 2 * 81921)
        self.assertEqual(report["application_write_capacity_reserved_bytes"], 2 * 8192)
        self.assertGreater(report["plaintext_response_bytes_received"], 0)


if __name__ == "__main__":
    unittest.main()
