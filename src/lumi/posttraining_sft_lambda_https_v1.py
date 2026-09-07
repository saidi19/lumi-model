"""Fixed-host, one-shot HTTPS transport library; no CLI or automatic integration.

IMPORTANT: exchange() performs real network I/O when explicitly called with real
dependencies. It must only be called by a separately authorized external controller.
The frozen local-only lifecycle/launch profiles DO NOT authorize calling it. This
source successor is tested with fake DNS/TLS sockets and reads no API credential.

No proxy, redirect, reconnect, second address, resend, compression or raw logging.
DNS/trust-store/kernel/close calls can block beyond deadlines; checks detect that,
not enforce a physical network-byte ceiling or provider invoice limit.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import ssl
import threading
import time
from typing import NamedTuple

from . import posttraining_sft_lambda_http_v1 as http

MAXIMUM_EXCHANGES = 64
MAXIMUM_READ_CALLS = 256
READ_CAPACITY = http.MAXIMUM_WIRE_BYTES + 1
MAXIMUM_DURATION_NS = 15_000_000_000


class TransportError(ValueError):
    def __init__(self, _value=None):
        super().__init__("lambda_transport_rejected")


def need(condition):
    if not condition:
        raise TransportError from None


class Reply(NamedTuple):
    """Internal transient JSON, not a persistence/reporting object or terminal proof."""

    status: int
    body: bytes

    def __repr__(self):
        return "<Lambda Reply: redacted>"


class LambdaHttps:
    def __init__(self, *, clock_ns=None):
        self._clock = time.monotonic_ns if clock_ns is None else clock_ns
        need(callable(self._clock))
        self._pid = os.getpid()
        self._thread = threading.current_thread()
        self._busy = self._failed = False
        self._last = None
        self._deadline = None
        self._attempts = self._dns = self._connects = self._handshakes = 0
        self._writes = self._write_reserved = self._reads = self._read_reserved = 0
        self._received = self._closes = self._completed = 0

    def _now(self):
        need(not self._failed and os.getpid() == self._pid
             and threading.current_thread() is self._thread)
        now = self._clock()
        need(not self._failed and type(now) is int and 0 <= now <= 2**63 - 1
             and (self._last is None or now >= self._last))
        self._last = now
        return now

    def _remaining(self):
        now = self._now()
        need(now < self._deadline)
        return (self._deadline - now) / 1_000_000_000

    def _resolve(self):
        self._remaining()
        self._dns += 1
        entries = socket.getaddrinfo(http.HOST, 443, socket.AF_INET, socket.SOCK_STREAM,
                                     socket.IPPROTO_TCP)
        self._remaining()
        need(type(entries) is list and 1 <= len(entries) <= 8)
        # One selected address, not a loop of connection attempts or family fallback.
        entry = entries[0]
        need(type(entry) is tuple and len(entry) == 5 and entry[:3] == (
            socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP,
        ))
        address = entry[4]
        need(type(address) is tuple and len(address) == 2 and type(address[0]) is str
             and type(address[1]) is int and address[1] == 443)
        parsed = ipaddress.IPv4Address(address[0])
        need(parsed.is_global and not parsed.is_multicast and not parsed.is_reserved
             and str(parsed) == address[0])
        return address

    def exchange(self, request, *, deadline_ns):
        """Consume one compiled request BEFORE I/O; any failure poisons this transport.

        No owner/plan/account approval is authenticated here. Success means framed
        HTTPS data received and owned socket closed, not an instance action proved.
        TLS uses the trusted runtime's default CA store, not caller-supplied CAs.
        """
        if self._busy:
            self._failed = True
            raise TransportError
        self._busy = True
        owned = None
        parser = None
        raw = b""
        response_body = None
        try:
            now = self._now()
            need(type(request) is http.Request)
            raw = request.claim()
            need(type(deadline_ns) is int and now < deadline_ns <= now + MAXIMUM_DURATION_NS
                 and deadline_ns <= 2**63 - 1)
            self._deadline = deadline_ns
            need(self._attempts < MAXIMUM_EXCHANGES)
            self._attempts += 1
            # Whole response capacity is prepaid before DNS, even if nothing is received.
            self._read_reserved += READ_CAPACITY
            self._write_reserved += http.MAXIMUM_REQUEST_BYTES
            address = self._resolve()
            # create_default_context() honors SSLKEYLOGFILE and can open a secret
            # log before returning. Construct the verified client context directly.
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            context.verify_flags |= ssl.VERIFY_X509_STRICT | ssl.VERIFY_X509_PARTIAL_CHAIN
            context.keylog_filename = None
            context.load_default_certs(ssl.Purpose.SERVER_AUTH)
            context.set_alpn_protocols(["http/1.1"])
            need(context.check_hostname is True and context.verify_mode == ssl.CERT_REQUIRED
                 and context.minimum_version >= ssl.TLSVersion.TLSv1_2
                 and context.keylog_filename is None)
            self._remaining()
            owned = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
            need(not owned.get_inheritable())
            owned.settimeout(self._remaining())
            self._connects += 1
            owned.connect(address)
            self._remaining()
            secured = context.wrap_socket(owned, server_hostname=http.HOST,
                                           do_handshake_on_connect=False,
                                           suppress_ragged_eofs=False)
            owned = secured
            need(not owned.get_inheritable())
            owned.settimeout(self._remaining())
            self._handshakes += 1
            owned.do_handshake()
            self._remaining()
            need(owned.selected_alpn_protocol() in (None, "http/1.1")
                 and owned.version() in {"TLSv1.2", "TLSv1.3"}
                 and bool(owned.getpeercert()))
            owned.settimeout(self._remaining())
            self._writes += 1
            expected_written = len(raw)
            written = owned.send(raw)
            raw = b""
            need(type(written) is int and written > 0 and written == expected_written)
            self._remaining()
            parser = http.ResponseParser()
            received = calls = 0
            while True:
                need(calls < MAXIMUM_READ_CALLS and received < READ_CAPACITY)
                # The whole exchange was prepaid before DNS. Short fragments use
                # only the bytes returned within that reservation, never a refund.
                size = min(4096, READ_CAPACITY - received)
                calls += 1
                self._reads += 1
                owned.settimeout(self._remaining())
                chunk = owned.recv(size)
                self._remaining()
                need(type(chunk) is bytes and len(chunk) <= size)
                received += len(chunk)
                self._received += len(chunk)
                if not chunk:
                    response_body = parser.finish()
                    break
                parser.feed(chunk)
            self._remaining()
        except BaseException:
            self._failed = True
        finally:
            raw = b""
            request = None
            if parser is not None:
                try:
                    parser.discard()
                except BaseException:
                    self._failed = True
            if owned is not None:
                self._closes += 1
                try:
                    owned.close()
                except BaseException:
                    self._failed = True
            if not self._failed:
                try:
                    self._remaining()
                except BaseException:
                    self._failed = True
            self._busy = False
        if self._failed:
            response_body = None
            raise TransportError from None
        self._completed += 1
        return Reply(200, response_body)

    def summary(self):
        return {
            "schema_version": "lumi.lambda-https-transport-counters.v0.1",
            "failed": self._failed, "busy": self._busy,
            "exchange_attempts": self._attempts, "completed_exchanges": self._completed,
            "dns_calls": self._dns, "tcp_connect_calls": self._connects,
            "tls_handshake_calls": self._handshakes, "send_calls": self._writes,
            "application_write_capacity_reserved_bytes": self._write_reserved,
            "recv_calls": self._reads, "plaintext_response_bytes_received": self._received,
            "application_read_capacity_reserved_bytes": self._read_reserved,
            "close_calls": self._closes, "physical_network_byte_cap_enforced": False,
            "hard_wall_time_enforced": False, "execution_authorized": False,
            "durable_controller_integrated": False, "provider_account_verified": False,
            "provider_termination_verified": False, "cloud_billing_stopped": False,
            "production_eligible": False,
        }
