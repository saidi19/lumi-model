"""Synthetic transport faults only: DNS, socket creation and TLS are always faked.

No test opens a network socket, loads a CA store or reads a real API credential.
Mocked handshakes are not live TLS/provider/termination evidence.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_test_sft_lambda_https_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
net = importlib.import_module(PACKAGE + ".posttraining_sft_lambda_https_v1")
http = net.http
TOKEN = b"SYNTHETIC-NOT-A-CREDENTIAL"
IDENTIFIER = "a" * 32
NS = 1_000_000_000


def request():
    return http.prepare_request("observe", token=TOKEN, instance_id=IDENTIFIER)


def response(body=b"{}", *, status=b"200 OK"):
    return (b"HTTP/1.1 " + status + b"\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(body)).encode() + b"\r\n\r\n" + body)


def address(ip="8.8.8.8", port=443):
    # A synthetic resolver result, never contacted; all sockets below are mocks.
    return (net.socket.AF_INET, net.socket.SOCK_STREAM, net.socket.IPPROTO_TCP, "", (ip, port))


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.dns = self.enterContext(patch.object(net.socket, "getaddrinfo"))
        self.sockets = self.enterContext(patch.object(net.socket, "socket"))
        self.contexts = self.enterContext(patch.object(net.ssl, "SSLContext"))
        self.default_contexts = self.enterContext(patch.object(
            net.ssl, "create_default_context", side_effect=AssertionError("unexpected TLS factory"),
        ))
        self.reset_fixture()

    def reset_fixture(self):
        for item in (self.dns, self.sockets, self.contexts):
            item.reset_mock(side_effect=True)
        self.now = 0
        self.clock = Mock(side_effect=lambda: self.now)
        self.dns.return_value = [address()]
        self.raw = Mock(name="synthetic_tcp")
        self.tls = Mock(name="synthetic_tls")
        self.context = Mock(name="synthetic_context")
        self.context.verify_flags = 0
        self.sockets.return_value = self.raw
        self.contexts.return_value = self.context
        self.raw.get_inheritable.return_value = False
        self.tls.get_inheritable.return_value = False
        self.context.wrap_socket.return_value = self.tls
        self.tls.selected_alpn_protocol.return_value = "http/1.1"
        self.tls.version.return_value = "TLSv1.3"
        self.tls.getpeercert.return_value = {"synthetic": True}
        self.tls.send.side_effect = len
        self.fragments = [response(), b""]
        self.tls.recv.side_effect = self.receive
        self.transport = net.LambdaHttps(clock_ns=self.clock)

    def receive(self, size):
        fragment = self.fragments.pop(0)
        if len(fragment) > size:
            self.fragments.insert(0, fragment[size:])
        return fragment[:size]

    def exchange(self, prepared=None, deadline=NS):
        return self.transport.exchange(request() if prepared is None else prepared,
                                       deadline_ns=deadline)

    def rejected(self, prepared=None, deadline=NS):
        with self.assertRaises(net.TransportError) as caught:
            self.exchange(prepared, deadline)
        self.assertEqual(str(caught.exception), "lambda_transport_rejected")
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(self.transport.summary()["failed"])
        self.assertFalse(self.transport.summary()["busy"])
        calls = self.dns.call_count
        with self.assertRaises(net.TransportError):
            self.exchange()
        self.assertEqual(self.dns.call_count, calls)

    def test_constructor_and_summary_do_not_read_credentials_or_perform_io(self):
        self.dns.assert_not_called()
        self.sockets.assert_not_called()
        self.contexts.assert_not_called()
        value = self.transport.summary()
        self.assertEqual(value["exchange_attempts"], 0)
        for key in ("execution_authorized", "durable_controller_integrated",
                    "provider_account_verified", "provider_termination_verified",
                    "cloud_billing_stopped", "production_eligible",
                    "physical_network_byte_cap_enforced", "hard_wall_time_enforced"):
            self.assertIs(value[key], False)

    def test_fixed_dns_tcp_sni_and_certificate_profile_with_single_send_and_close(self):
        result = self.exchange()
        self.assertEqual(result, (200, b"{}"))
        self.assertEqual(repr(result), "<Lambda Reply: redacted>")
        self.dns.assert_called_once_with(http.HOST, 443, net.socket.AF_INET,
                                         net.socket.SOCK_STREAM, net.socket.IPPROTO_TCP)
        self.sockets.assert_called_once_with(net.socket.AF_INET, net.socket.SOCK_STREAM,
                                             net.socket.IPPROTO_TCP)
        self.raw.connect.assert_called_once_with(("8.8.8.8", 443))
        self.contexts.assert_called_once_with(net.ssl.PROTOCOL_TLS_CLIENT)
        self.context.load_default_certs.assert_called_once_with(net.ssl.Purpose.SERVER_AUTH)
        self.default_contexts.assert_not_called()
        self.assertEqual(self.context.minimum_version, net.ssl.TLSVersion.TLSv1_2)
        self.assertIs(self.context.check_hostname, True)
        self.assertEqual(self.context.verify_mode, net.ssl.CERT_REQUIRED)
        self.assertEqual(self.context.verify_flags,
                         net.ssl.VERIFY_X509_STRICT | net.ssl.VERIFY_X509_PARTIAL_CHAIN)
        self.context.set_alpn_protocols.assert_called_once_with(["http/1.1"])
        self.context.wrap_socket.assert_called_once_with(
            self.raw, server_hostname=http.HOST, do_handshake_on_connect=False,
            suppress_ragged_eofs=False,
        )
        self.tls.do_handshake.assert_called_once_with()
        self.assertEqual(self.tls.send.call_count, 1)
        self.tls.close.assert_called_once_with()
        self.raw.close.assert_not_called()  # Ownership transferred into the TLS socket.
        value = self.transport.summary()
        self.assertEqual(value["completed_exchanges"], 1)
        self.assertEqual(value["application_read_capacity_reserved_bytes"], 81921)
        self.assertEqual(value["application_write_capacity_reserved_bytes"], 8192)
        self.assertEqual(value["plaintext_response_bytes_received"], len(response()))

    def test_only_first_address_is_used_no_fallback(self):
        self.dns.return_value = [address(), address("1.1.1.1")]
        self.raw.connect.side_effect = OSError("SECRET")
        self.rejected()
        self.raw.connect.assert_called_once_with(("8.8.8.8", 443))
        self.raw.close.assert_called_once_with()
        self.context.wrap_socket.assert_not_called()

    def test_proxy_environment_does_not_change_request_or_destination(self):
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://synthetic.invalid:9999",
                                     "HTTP_PROXY": "http://synthetic.invalid:9999"}):
            self.exchange()
        sent = self.tls.send.call_args.args[0]
        self.assertTrue(sent.startswith(b"GET /api/v1/instances/"))
        self.assertNotIn(b"CONNECT ", sent)
        self.assertNotIn(b"synthetic.invalid", sent)
        self.raw.connect.assert_called_once_with(("8.8.8.8", 443))

    def test_keylog_environment_is_not_honored_or_opened(self):
        with patch.dict(os.environ, {"SSLKEYLOGFILE": "/synthetic/never-created-keylog"}):
            self.exchange()
        self.default_contexts.assert_not_called()
        self.assertIsNone(self.context.keylog_filename)

    def test_certificate_store_failure_stops_before_socket_creation(self):
        self.context.load_default_certs.side_effect = OSError("SECRET")
        self.rejected()
        self.sockets.assert_not_called()

    def test_request_consumed_and_raw_reference_released_before_dns(self):
        prepared = request()

        def resolve(*args):
            self.assertTrue(prepared._used)
            self.assertEqual(prepared._raw, b"")
            return [address()]

        self.dns.side_effect = resolve
        self.exchange(prepared)
        with self.assertRaises(http.HttpError):
            prepared.claim()

    def test_invalid_or_already_claimed_request_never_reaches_dns(self):
        for prepared in (object(), request()):
            self.reset_fixture()
            if type(prepared) is http.Request:
                prepared.claim()
            self.rejected(prepared)
            self.dns.assert_not_called()

    def test_invalid_deadline_consumes_request_without_io(self):
        for deadline in (None, True, -1, 0, 15 * NS + 1, 2**63):
            self.reset_fixture()
            prepared = request()
            self.rejected(prepared, deadline)
            self.assertTrue(prepared._used)
            self.assertEqual(prepared._raw, b"")
            self.dns.assert_not_called()

    def test_invalid_clock_values_fail_before_request_or_io(self):
        for value in (None, True, -1, 2**63, 0.5):
            self.reset_fixture()
            self.now = value
            self.rejected()
            self.dns.assert_not_called()

    def test_clock_regression_after_dns_blocks_socket_creation(self):
        self.now = 2

        def resolve(*args):
            self.now = 1
            return [address()]

        self.dns.side_effect = resolve
        self.rejected()
        self.sockets.assert_not_called()

    def test_dns_exception_consumes_full_capacity_without_socket(self):
        self.dns.side_effect = OSError("SECRET")
        self.rejected()
        value = self.transport.summary()
        self.assertEqual(value["application_read_capacity_reserved_bytes"], 81921)
        self.assertEqual(value["application_write_capacity_reserved_bytes"], 8192)
        self.assertEqual(value["exchange_attempts"], 1)
        self.sockets.assert_not_called()

    def test_dns_profile_and_nonpublic_addresses_rejected(self):
        cases = (None, (), [], [address()] * 9, [None], [address("127.0.0.1")],
                 [address("10.0.0.1")], [address("169.254.169.254")], [address("192.0.2.1")],
                 [address("224.0.0.1")], [address("0.0.0.0")], [address("::1")],
                 [address("08.8.8.8")], [address(port=True)], [address(port=80)],
                 [(net.socket.AF_INET6, net.socket.SOCK_STREAM, net.socket.IPPROTO_TCP,
                   "", ("::1", 443, 0, 0))])
        for value in cases:
            self.reset_fixture()
            self.dns.return_value = value
            self.rejected()
            self.sockets.assert_not_called()

    def test_context_creation_failure_has_no_socket_to_close(self):
        self.contexts.side_effect = OSError("SECRET")
        self.rejected()
        self.sockets.assert_not_called()
        self.assertEqual(self.transport.summary()["close_calls"], 0)

    def test_socket_creation_failure_has_no_owned_descriptor_to_retry(self):
        self.sockets.side_effect = OSError("SECRET")
        self.rejected()
        self.assertEqual(self.sockets.call_count, 1)
        self.assertEqual(self.transport.summary()["close_calls"], 0)

    def test_raw_socket_inheritance_or_timeout_fault_closes_before_connect(self):
        for fault in ("inheritance", "timeout"):
            self.reset_fixture()
            if fault == "inheritance":
                self.raw.get_inheritable.return_value = True
            else:
                self.raw.settimeout.side_effect = OSError("SECRET")
            self.rejected()
            self.raw.connect.assert_not_called()
            self.raw.close.assert_called_once_with()

    def test_wrap_failure_attempts_raw_cleanup_once_without_handshake(self):
        self.context.wrap_socket.side_effect = net.ssl.SSLError("SECRET")
        self.rejected()
        self.raw.close.assert_called_once_with()
        self.tls.do_handshake.assert_not_called()

    def test_secured_socket_inheritance_rejected_and_closed(self):
        self.tls.get_inheritable.return_value = True
        self.rejected()
        self.tls.do_handshake.assert_not_called()
        self.tls.close.assert_called_once_with()

    def test_handshake_failure_never_sends_or_retries(self):
        for error in (net.ssl.SSLCertVerificationError("SECRET"), TimeoutError("SECRET"),
                      KeyboardInterrupt("SECRET")):
            self.reset_fixture()
            self.tls.do_handshake.side_effect = error
            self.rejected()
            self.tls.do_handshake.assert_called_once_with()
            self.tls.send.assert_not_called()
            self.tls.close.assert_called_once_with()

    def test_bad_alpn_version_or_missing_certificate_blocks_send(self):
        for method, value in (("selected_alpn_protocol", "h2"), ("version", "TLSv1.1"),
                              ("version", None), ("getpeercert", {})):
            self.reset_fixture()
            getattr(self.tls, method).return_value = value
            self.rejected()
            self.tls.send.assert_not_called()

    def test_no_alpn_selection_and_tls12_supported(self):
        self.tls.selected_alpn_protocol.return_value = None
        self.tls.version.return_value = "TLSv1.2"
        self.assertEqual(self.exchange().body, b"{}")

    def test_short_zero_invalid_send_and_send_exception_are_not_retried(self):
        for value in (0, 1, True, None, 999999, "SECRET", OSError("SECRET")):
            self.reset_fixture()
            self.tls.send.side_effect = value if isinstance(value, Exception) else None
            self.tls.send.return_value = value
            self.rejected()
            self.assertEqual(self.tls.send.call_count, 1)
            self.tls.recv.assert_not_called()
            self.tls.close.assert_called_once_with()

    def test_every_byte_can_arrive_as_a_separate_fragment(self):
        wire = response(b'{"synthetic":true}')
        self.fragments = [bytes([byte]) for byte in wire] + [b""]
        self.assertEqual(self.exchange().body, b'{"synthetic":true}')
        self.assertEqual(self.tls.recv.call_count, len(wire) + 1)
        self.assertEqual(self.transport.summary()["application_read_capacity_reserved_bytes"],
                         81921)

    def test_chunked_response_and_maximum_body_are_supported(self):
        for wire, body in ((b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                            b"Transfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n", b"{}"),
                           (response(b"x" * 65536), b"x" * 65536)):
            self.reset_fixture()
            self.fragments = [wire, b""]
            self.assertEqual(self.exchange().body, body)

    def test_framing_success_is_not_json_schema_or_termination_proof(self):
        self.fragments = [response(b"not-json"), b""]
        self.assertEqual(self.exchange().body, b"not-json")
        self.assertFalse(self.transport.summary()["provider_termination_verified"])

    def test_clean_eof_is_required_even_after_complete_body(self):
        self.fragments = [response()]
        count = 0

        def receive(size):
            nonlocal count
            count += 1
            if count == 1:
                return response()
            raise net.ssl.SSLEOFError("SECRET")

        self.tls.recv.side_effect = receive
        self.rejected()
        self.assertEqual(self.tls.recv.call_count, 2)

    def test_redirect_and_error_responses_are_not_followed_or_retried(self):
        for status in (b"301 SECRET", b"307 SECRET", b"401 SECRET", b"429 SECRET",
                       b"500 SECRET"):
            self.reset_fixture()
            self.fragments = [response(status=status)]
            self.rejected()
            self.assertEqual(self.dns.call_count, 1)
            self.assertEqual(self.tls.send.call_count, 1)

    def test_truncation_and_trailing_data_rejected(self):
        for parts in ([response()[:-1], b""], [response(), b"X"], [response() + response()]):
            self.reset_fixture()
            self.fragments = parts
            self.rejected()

    def test_invalid_or_excessive_receive_value_and_exception_rejected(self):
        for value in (None, "SECRET", bytearray(b"{}"), b"x" * 4097, OSError("SECRET")):
            self.reset_fixture()
            self.tls.recv.side_effect = value if isinstance(value, Exception) else None
            self.tls.recv.return_value = value
            self.rejected()
            self.assertEqual(self.tls.recv.call_count, 1)

    def test_receive_call_limit_includes_eof_probe_and_never_retries(self):
        self.fragments = [bytes([byte]) for byte in response(b"x" * 512)] + [b""]
        self.rejected()
        self.assertEqual(self.tls.recv.call_count, 256)
        self.assertEqual(self.transport.summary()["recv_calls"], 256)

    def test_full_receive_reservation_cannot_be_exceeded_even_with_parser_dependency(self):
        # Isolate the transport bound from the stricter HTTP parser's early denials.
        parser = Mock()
        self.tls.recv.side_effect = lambda size: b"x" * size
        with patch.object(http, "ResponseParser", return_value=parser):
            self.rejected()
        self.assertEqual(sum(call.args[0] for call in self.tls.recv.call_args_list), 81921)
        self.assertEqual(self.transport.summary()["plaintext_response_bytes_received"], 81921)
        parser.finish.assert_not_called()
        parser.discard.assert_called_once_with()

    def test_sixty_four_exchanges_cap_prepaid_capacity_and_sixty_fifth_consumed(self):
        for _ in range(64):
            self.fragments = [response(), b""]
            self.exchange()
        prepared = request()
        self.rejected(prepared)
        self.assertTrue(prepared._used)
        value = self.transport.summary()
        self.assertEqual(value["completed_exchanges"], 64)
        self.assertEqual(value["exchange_attempts"], 64)
        self.assertEqual(value["application_read_capacity_reserved_bytes"], 64 * 81921)
        self.assertEqual(value["application_write_capacity_reserved_bytes"], 64 * 8192)
        self.assertEqual(self.dns.call_count, 64)

    def test_expiry_after_each_blocking_stage_withholds_success_and_closes_owned_socket(self):
        for stage in ("dns", "context", "connect", "wrap", "handshake", "send", "recv", "close"):
            self.reset_fixture()

            def expire(*args, current_stage=stage, **kwargs):
                self.now = NS
                return {"dns": [address()], "context": self.context, "wrap": self.tls,
                        "send": len(args[0]) if args and type(args[0]) is bytes else 0,
                        "recv": response()}.get(current_stage)

            method = {"dns": self.dns, "context": self.contexts, "connect": self.raw.connect,
                      "wrap": self.context.wrap_socket, "handshake": self.tls.do_handshake,
                      "send": self.tls.send, "recv": self.tls.recv, "close": self.tls.close}[stage]
            method.side_effect = expire
            self.rejected()
            self.assertEqual(self.transport.summary()["completed_exchanges"], 0)
            if stage in {"dns", "context"}:
                self.sockets.assert_not_called()
            elif stage == "connect":
                self.raw.close.assert_called_once_with()
            else:
                self.tls.close.assert_called_once_with()

    def test_timeout_values_are_refreshed_from_remaining_deadline(self):
        def connect(*args):
            self.now = NS // 2

        self.raw.connect.side_effect = connect
        self.exchange()
        self.raw.settimeout.assert_called_once_with(1.0)
        self.assertTrue(all(call.args == (0.5,) for call in self.tls.settimeout.call_args_list))

    def test_close_failure_withholds_body_and_is_not_retried(self):
        self.tls.close.side_effect = OSError("SECRET")
        self.rejected()
        self.tls.close.assert_called_once_with()
        self.assertEqual(self.transport.summary()["completed_exchanges"], 0)

    def test_parser_cleanup_failure_still_closes_socket_and_withholds_result(self):
        with patch.object(http.ResponseParser, "discard", side_effect=RuntimeError("SECRET")):
            self.rejected()
        self.tls.close.assert_called_once_with()

    def test_reentry_from_clock_and_network_callback_is_sticky(self):
        for stage in ("clock", "dns", "close"):
            self.reset_fixture()

            def reenter(*args, current_stage=stage):
                with self.assertRaises(net.TransportError):
                    self.exchange()
                return [address()] if current_stage == "dns" else 0

            {"clock": self.clock, "dns": self.dns, "close": self.tls.close}[stage].side_effect = (
                reenter)
            self.rejected()
            self.assertEqual(self.transport.summary()["completed_exchanges"], 0)
            if stage == "close":
                self.tls.close.assert_called_once_with()
            else:
                self.sockets.assert_not_called()

    def test_foreign_process_or_thread_cannot_start_transport(self):
        for function, value in (("getpid", os.getpid() + 1),
                                ("current_thread", threading.Thread())):
            self.reset_fixture()
            module = net.os if function == "getpid" else net.threading
            with patch.object(module, function, return_value=value):
                self.rejected()
            self.dns.assert_not_called()

    def test_counters_do_not_contain_token_identifier_response_or_exception_text(self):
        self.fragments = [response(b'{"SECRET":"not for a report"}'), b""]
        result = self.exchange()
        rendered = json.dumps(self.transport.summary()) + repr(result)
        for value in ("SECRET", TOKEN.decode(), IDENTIFIER, "not for a report"):
            self.assertNotIn(value, rendered)
        self.assertIs(self.transport.summary()["execution_authorized"], False)


if __name__ == "__main__":
    unittest.main()
