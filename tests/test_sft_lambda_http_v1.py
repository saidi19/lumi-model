"""Synthetic pure request/framing fixtures, no network or account access."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_synthetic_lambda_http", ROOT / "src/lumi/posttraining_sft_lambda_http_v1.py",
)
http = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(http)
TOKEN = b"SYNTHETIC-NOT-A-CREDENTIAL"
IDENTIFIER = "a" * 32


def launch(**changes):
    return http.canonical({
        "region_name": "us-east-1", "instance_type_name": "gpu_8x_a100",
        "ssh_key_names": ["Synthetic Key"],
        "image": {"id": "12345678-1234-1234-1234-123456789abc"}, **changes,
    })


def create(body=None):
    body = launch() if body is None else body
    return http.prepare_request("create", token=TOKEN, launch_body=body,
                                expected_launch_sha256=hashlib.sha256(body).hexdigest(),
                                region="us-east-1", instance_type="gpu_8x_a100")


def response(body=b"{}", *, headers=None, status=b"HTTP/1.1 200 OK"):
    headers = (b"Content-Type: application/json\r\nContent-Length: "
               + str(len(body)).encode()) if headers is None else headers
    return status + b"\r\n" + headers + b"\r\n\r\n" + body


class RequestTests(unittest.TestCase):
    def test_fixed_paths_host_auth_body_length_and_redaction(self):
        for operation, method, suffix in (
            ("create", "POST", "instance-operations/launch"),
            ("observe", "GET", "instances/" + IDENTIFIER),
            ("terminate", "POST", "instance-operations/terminate"),
        ):
            request = create() if operation == "create" else http.prepare_request(
                operation, token=TOKEN, instance_id=IDENTIFIER,
            )
            self.assertNotIn(TOKEN.decode(), repr(request))
            raw = request.claim()
            header, body = raw.split(b"\r\n\r\n")
            self.assertTrue(raw.startswith(f"{method} /api/v1/{suffix} HTTP/1.1\r\n".encode()))
            self.assertIn(b"Host: cloud.lambda.ai\r\n", raw)
            self.assertIn(b"Authorization: Bearer " + TOKEN, header)
            self.assertIn(f"Content-Length: {len(body)}".encode(), header)
            self.assertLessEqual(len(raw), 8192)
            if operation == "terminate":
                self.assertEqual(json.loads(body), {"instance_ids": [IDENTIFIER]})
            self.assertEqual(request._raw, b"")

    def test_request_is_consumed_once_and_direct_construction_rejected(self):
        request = create()
        request.claim()
        with self.assertRaises(http.HttpError):
            request.claim()
        with self.assertRaises(http.HttpError):
            http.Request("create", b"arbitrary")

    def test_mutated_request_is_consumed_but_cannot_be_sent(self):
        request = create()
        request._raw = b"bad"
        with self.assertRaises(http.HttpError):
            request.claim()
        self.assertTrue(request._used)
        self.assertEqual(request._raw, b"")
        request = create()
        request._digest = object()
        with self.assertRaises(http.HttpError):
            request.claim()
        self.assertEqual(request._raw, b"")

    def test_header_injection_nonbytes_and_oversized_credentials_rejected(self):
        for token in (None, TOKEN.decode(), b"a", b"a" * 513, b"a\r\nHost: evil", b"a b",
                      b"x" * 8 + b"\x00", b"x" * 8 + b"\xff"):
            with self.subTest(kind=type(token).__name__), self.assertRaises(http.HttpError):
                http.prepare_request("observe", token=token, instance_id=IDENTIFIER)

    def test_operation_target_and_cross_operation_arguments_rejected(self):
        for operation in ("restart", "list", "GET", True, None):
            with self.assertRaises(http.HttpError):
                http.prepare_request(operation, token=TOKEN, instance_id=IDENTIFIER)
        for identifier in (None, "A" * 32, "../x", "a" * 31, [IDENTIFIER], True):
            with self.assertRaises(http.HttpError):
                http.prepare_request("terminate", token=TOKEN, instance_id=identifier)
        with self.assertRaises(http.HttpError):
            http.prepare_request("observe", token=TOKEN, instance_id=IDENTIFIER, region="other")

    def test_launch_exact_pin_region_type_and_no_additional_operations(self):
        with self.assertRaises(http.HttpError):
            http.validate_launch(launch(), "0" * 64, "us-east-1", "gpu_8x_a100")
        for change in ({"region_name": "other"}, {"instance_type_name": "other"},
                       {"user_data": "SECRET"}, {"quantity": 2}, {"file_system_names": []},
                       {"tags": []}, {"image": {"family": "default"}},
                       {"image": {"id": "bad"}}, {"ssh_key_names": []},
                       {"ssh_key_names": ["one", "two"]}, {"ssh_key_names": ["a\nb"]}):
            with self.assertRaises(http.HttpError):
                create(launch(**change))

    def test_launch_duplicate_missing_noncanonical_large_and_invalid_json_rejected(self):
        raw = launch()
        cases = (raw + b" ", b'{"region_name":"other",' + raw[1:], b"{}", b"[]",
                 b"NaN", b"\xff", b"x" * 4097, b"[" * 1500 + b"]" * 1500)
        for body in cases:
            with self.assertRaises(http.HttpError):
                create(body)


class ParserTests(unittest.TestCase):
    def parse(self, raw, width=None):
        parser = http.ResponseParser()
        if width is None:
            parser.feed(raw)
        else:
            for offset in range(0, len(raw), width):
                parser.feed(raw[offset:offset + width])
        return parser.finish()

    def reject(self, raw):
        parser = http.ResponseParser()
        with self.assertRaises(http.HttpError) as caught:
            parser.feed(raw)
            parser.finish()
        self.assertEqual(str(caught.exception), "http_profile_rejected")
        self.assertNotIn("SECRET", repr(caught.exception))
        with self.assertRaises(http.HttpError):
            parser.feed(response())

    def test_length_response_across_all_small_fragment_widths(self):
        body = b'{"synthetic":true}'
        for width in range(1, 32):
            with self.subTest(width=width):
                self.assertEqual(self.parse(response(body), width), body)

    def test_chunked_across_all_small_fragment_widths(self):
        raw = response(b"1\r\n{\r\n1\r\n}\r\n0\r\n\r\n", headers=(
            b"Content-Type: application/json; charset=utf-8\r\nTransfer-Encoding: chunked"))
        for width in range(1, 32):
            with self.subTest(width=width):
                self.assertEqual(self.parse(raw, width), b"{}")

    def test_exact_body_limit_and_header_case_are_supported(self):
        self.assertEqual(len(self.parse(response(b"x" * 65536), 4096)), 65536)
        self.assertEqual(self.parse(response(headers=(
            b"content-type: APPLICATION/JSON\r\ncontent-length: 2\r\n"
            b"Content-Encoding: identity"))), b"{}")

    def test_eof_required_and_reuse_trailing_bytes_rejected(self):
        parser = http.ResponseParser()
        parser.feed(response())
        self.assertEqual(parser.finish(), b"{}")
        with self.assertRaises(http.HttpError):
            parser.finish()
        for suffix in (b"X", b"\r\n", response()):
            self.reject(response() + suffix)
        parser = http.ResponseParser()
        parser.feed(response())
        with self.assertRaises(http.HttpError):
            parser.feed(b"X")

    def test_every_truncated_prefix_rejected(self):
        raw = response(b"{\"x\":1}")
        for index in range(1, len(raw)):
            with self.subTest(index=index):
                self.reject(raw[:index])

    def test_redirect_error_interim_status_and_unsupported_versions_rejected(self):
        for status in (b"HTTP/1.1 301 SECRET", b"HTTP/1.1 307 Redirect", b"HTTP/1.1 100 Continue",
                       b"HTTP/1.1 401 SECRET", b"HTTP/1.1 404 No", b"HTTP/1.1 500 No",
                       b"HTTP/1.0 200 OK", b"HTTP/2 200 OK", b"HTTP/1.1 200"):
            self.reject(response(status=status))

    def test_duplicate_ambiguous_missing_or_invalid_framing_rejected(self):
        prefix = b"Content-Type: application/json\r\n"
        for suffix in (b"", b"Content-Length: 2\r\ncontent-length: 2", b"Content-Length: 02",
                       b"Content-Length: -1", b"Content-Length: 65537", b"Content-Length: 2, 2",
                       b"Content-Length: 2\r\nTransfer-Encoding: chunked",
                       b"Transfer-Encoding: identity", b"Transfer-Encoding: gzip, chunked"):
            self.reject(response(headers=prefix + suffix))

    def test_compression_folding_controls_content_type_header_count_and_size_rejected(self):
        base = b"Content-Type: application/json\r\nContent-Length: 2"
        for extra in (b"\r\nContent-Encoding: gzip", b"\r\n folded: value", b"\r\nX: \x00",
                      b"\r\nX: \xff", b"\r\nX: " + b"x" * 1025,
                      b"".join(f"\r\nX-{index}: x".encode() for index in range(64))):
            self.reject(response(headers=base + extra))
        self.reject(response(headers=b"Content-Type: text/html\r\nContent-Length: 2"))
        self.reject(b"H" * 8193)

    def test_chunk_extensions_trailers_excessive_chunks_and_body_rejected(self):
        headers = b"Content-Type: application/json\r\nTransfer-Encoding: chunked"
        for body in (b"1;x=y\r\nx\r\n0\r\n\r\n", b"0\r\nX: SECRET\r\n\r\n",
                     b"1\r\nxXX0\r\n\r\n", b"+1\r\nx\r\n0\r\n\r\n",
                     b"10001\r\n", b"1\r\nx\r\n" * 129 + b"0\r\n\r\n"):
            self.reject(response(body, headers=headers))
        good = response(b"1\r\nx\r\n" * 128 + b"0\r\n\r\n", headers=headers)
        self.assertEqual(self.parse(good, 7), b"x" * 128)

    def test_feed_types_capacity_and_discard_are_fail_closed(self):
        for value in (None, "SECRET", bytearray(response()), b"", b"x" * 81921):
            self.reject(value)
        parser = http.ResponseParser()
        parser.feed(response(b"SECRET"))
        self.assertNotIn("SECRET", repr(parser))
        parser.discard()
        self.assertEqual(parser._body, b"")
        with self.assertRaises(http.HttpError):
            parser.finish()


if __name__ == "__main__":
    unittest.main()
