"""Synthetic dispatch-contract verification; no real keys, accounts or network."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_test_sft_dispatch_authority_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
authority = importlib.import_module(PACKAGE + ".posttraining_sft_dispatch_authority_v1")
KEY = b"SYNTHETIC-KEY-NOT-REAL-012345678"
TOKEN = b"SYNTHETIC-NOT-A-CREDENTIAL"
MISSING = object()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def plan(**changes):
    value = {
        "schema_version": authority.model.PLAN_SCHEMA, "mode": "local_decision_model_only",
        "attempt_id": "a" * 64, "account_scope_sha256": "b" * 64,
        "approval_sha256": "c" * 64, "launch_request_sha256": "d" * 64,
        "protected_instance_ids": ["e" * 32], "region": "us-east-1",
        "instance_type": "gpu_8x_a100_80gb_sxm4", "gpu_count": 8,
        "total_host_hourly_rate_microusd": 22_320_000,
        "maximum_cost_microusd": 150_000_000, "noncompute_reserve_microusd": 10_000_000,
        "maximum_lifecycle_seconds": 14400, "shutdown_reserve_seconds": 600,
        "create_not_before_unix": 1, "create_expires_unix": 2000, "maximum_requests": 8,
    }
    return {**value, **changes}


def unsigned(plan_value=None, *, token=TOKEN):
    plan_value = plan() if plan_value is None else plan_value
    return {
        "schema_version": authority.SCHEMA,
        "mode": "externally_issued_single_attempt_dispatch",
        "provider_host": "cloud.lambda.ai",
        "plan_sha256": digest(authority.canonical(plan_value)),
        "account_scope_sha256": plan_value["account_scope_sha256"],
        "approval_sha256": plan_value["approval_sha256"],
        "launch_request_sha256": plan_value["launch_request_sha256"],
        "credential_sha256": digest(token),
        "create_not_before_unix": plan_value["create_not_before_unix"],
        "create_expires_unix": plan_value["create_expires_unix"],
        "maximum_requests": plan_value["maximum_requests"],
        "cleanup_scope": "only_fresh_create_bound_id_within_lifecycle_caps",
    }


def signed(value=None, *, key=KEY, domain=authority.DOMAIN):
    value = unsigned() if value is None else value
    mac = hmac.new(key, domain + authority.canonical(value), hashlib.sha256).hexdigest()
    return authority.canonical({**value, "mac_sha256": mac})


class AuthorityTests(unittest.TestCase):
    def verify(self, raw=MISSING, **changes):
        raw = signed() if raw is MISSING else raw
        plan_bytes = authority.canonical(plan())
        arguments = {
            "expected_sha256": digest(raw) if type(raw) is bytes else "f" * 64,
            "key": KEY, "expected_key_sha256": digest(KEY), "plan_bytes": plan_bytes,
            "plan_sha256": digest(plan_bytes), "token": TOKEN,
        }
        return authority.verify_authority(raw, **{**arguments, **changes})

    def reject(self, raw=MISSING, **changes):
        with self.assertRaises(authority.AuthorityError) as caught:
            self.verify(raw, **changes)
        self.assertEqual(str(caught.exception), "dispatch_authority_rejected")
        self.assertEqual(repr(caught.exception), "AuthorityError('dispatch_authority_rejected')")
        self.assertTrue(caught.exception.__suppress_context__)
        return caught.exception

    def test_valid_contract_returns_only_frozen_plan_and_preserves_mode(self):
        self.assertEqual(len(KEY), 32)
        result = self.verify()
        expected = plan()
        expected["protected_instance_ids"] = tuple(expected["protected_instance_ids"])
        self.assertEqual(result, expected)
        self.assertEqual(result["mode"], "local_decision_model_only")
        self.assertNotIn("mac_sha256", result)
        self.assertNotIn("credential_sha256", result)
        self.assertNotIn(KEY.decode(), repr(result))
        self.assertNotIn(TOKEN.decode(), repr(result))

    def test_fresh_parsed_plan_not_shared_across_calls(self):
        first = self.verify()
        first["attempt_id"] = "changed"
        self.assertEqual(self.verify()["attempt_id"], "a" * 64)

    def test_no_issuer_or_authority_claims_are_exported(self):
        self.assertFalse(hasattr(authority, "issue_authority"))
        self.assertFalse(hasattr(authority, "sign_authority"))
        self.assertEqual(authority.DOMAIN, authority.SCHEMA.encode("ascii") + b"\0")
        self.assertNotIn("execution_authorized", self.verify())

    def test_raw_type_and_capacity(self):
        for raw in (b"", b" " * 8193, "SECRET", bytearray(signed()), memoryview(signed()),
                    None, True, 1, [], {}):
            with self.subTest(kind=type(raw).__name__):
                self.reject(raw)

    def test_key_exact_bytes_length_and_external_pin(self):
        for key in (b"", KEY[:-1], KEY + b"x", bytearray(KEY), memoryview(KEY),
                    KEY.decode(), None, True):
            with self.subTest(kind=type(key).__name__):
                self.reject(key=key)
        other = b"x" * 32
        self.reject(key=other)
        self.reject(signed(key=other), key=other)
        self.assertEqual(self.verify(signed(key=other), key=other,
                                     expected_key_sha256=digest(other))["attempt_id"], "a" * 64)

    def test_external_digest_types_shape_case_and_mismatch(self):
        for field in ("expected_sha256", "expected_key_sha256", "plan_sha256"):
            for value in (None, True, 1, b"a" * 64, "a" * 63, "A" * 64, "g" * 64,
                          "f" * 64, "a" * 64 + "\n"):
                with self.subTest(field=field, value_type=type(value).__name__):
                    self.reject(**{field: value})

    def test_all_hash_bindings_reject_even_with_valid_new_mac(self):
        for field in authority.HASH_KEYS - {"mac_sha256"}:
            with self.subTest(field=field):
                self.reject(signed({**unsigned(), field: "f" * 64}))

    def test_hash_types_canonical_lowercase_and_lengths(self):
        for field in authority.HASH_KEYS:
            for value in (None, True, 1, [], {}, "a" * 63, "A" * 64, "g" * 64, "a" * 65):
                with self.subTest(field=field, value_type=type(value).__name__):
                    obj = json.loads(signed())
                    obj[field] = value
                    self.reject(authority.canonical(obj))

    def test_integer_bindings_reject_bool_float_string_bounds_and_mismatch(self):
        for field in authority.INTEGER_KEYS:
            for value in (True, False, 1.0, "1", None, [], {}, -1, 0, 2**63, 1001):
                with self.subTest(field=field, value_type=type(value).__name__):
                    self.reject(signed({**unsigned(), field: value}))
        # True is numerically equal to this valid plan's lower bound, but never accepted.
        self.reject(signed({**unsigned(), "create_not_before_unix": True}))

    def test_fixed_fields_are_exact_strings(self):
        for field in authority.FIXED:
            for value in (None, True, 1, [], {}, "SECRET", authority.FIXED[field] + " "):
                with self.subTest(field=field, value_type=type(value).__name__):
                    self.reject(signed({**unsigned(), field: value}))

    def test_exact_key_set_missing_extra_and_aliases(self):
        obj = json.loads(signed())
        for key in obj:
            with self.subTest(missing=key):
                self.reject(authority.canonical({field: item for field, item in obj.items()
                                                 if field != key}))
        for key in ("instance_id", "cleanup_instance_id", "user_data", "authorization",
                    "execution_authorized", "schemaVersion"):
            with self.subTest(extra=key):
                self.reject(authority.canonical({**obj, key: "SECRET"}))

    def test_mac_requires_domain_key_and_untampered_full_object(self):
        self.reject(signed(domain=b""))
        self.reject(signed(domain=authority.SCHEMA.encode()))
        self.reject(signed(domain=b"different\0"))
        self.reject(signed(key=b"x" * 32))
        obj = json.loads(signed())
        obj["mac_sha256"] = "0" * 64
        self.reject(authority.canonical(obj))

    def test_plan_change_requires_new_authority_and_frozen_plan_validation(self):
        changed = plan(attempt_id="f" * 64)
        raw_plan = authority.canonical(changed)
        self.reject(plan_bytes=raw_plan, plan_sha256=digest(raw_plan))
        self.assertEqual(self.verify(signed(unsigned(changed)), plan_bytes=raw_plan,
                                     plan_sha256=digest(raw_plan))["attempt_id"], "f" * 64)
        for value in (plan(mode="dispatch"), plan(maximum_requests=True),
                      plan(protected_instance_ids=[]), plan(create_expires_unix=1)):
            with self.subTest(change=value):
                raw_plan = authority.canonical(value)
                self.reject(signed(unsigned(value)), plan_bytes=raw_plan,
                            plan_sha256=digest(raw_plan))

    def test_bad_plan_type_hash_raw_and_duplicate(self):
        for raw in (b"", b"{}", b"[", b"x" * 16385, b"\xff", b'{"x":1,"x":2}'):
            with self.subTest(length=len(raw)):
                self.reject(plan_bytes=raw, plan_sha256=digest(raw))
        self.reject(plan_bytes=bytearray(authority.canonical(plan())))

    def test_token_profile_and_credential_hash_binding(self):
        for token in (b"", b"short", b"x" * 513, b"bad\r\nTOKEN", b"space TOKEN",
                      b"line\nTOKEN", b"nul\0TOKEN", b"nonascii\xff", b"=prefixTOKEN",
                      b"validTOKEN===", bytearray(TOKEN), memoryview(TOKEN), TOKEN.decode(),
                      None, True):
            with self.subTest(kind=type(token).__name__):
                self.reject(token=token)
        self.reject(token=b"OTHER-SYNTHETIC-TOKEN")
        for token in (b"12345678", b"A._~+/-z", b"x" * 510 + b"=="):
            with self.subTest(length=len(token)):
                self.assertEqual(self.verify(signed(unsigned(token=token)), token=token)[
                    "maximum_requests"], 8)

    def test_raw_pin_prevents_any_byte_tamper(self):
        raw = signed()
        self.reject(raw + b" ", expected_sha256=digest(raw))
        self.reject(raw.replace(b"cloud.lambda.ai", b"cloud.lambda.io"),
                    expected_sha256=digest(raw))

    def test_noncanonical_json_even_with_correct_raw_pin_and_mac(self):
        obj = json.loads(signed())
        for raw in (signed() + b"\n", b" " + signed(), json.dumps(obj, indent=2).encode(),
                    json.dumps(dict(reversed(list(obj.items()))), separators=(",", ":")).encode(),
                    signed().replace(b"cloud.lambda.ai", b"cloud\\u002elambda.ai")):
            with self.subTest(length=len(raw)):
                self.reject(raw)

    def test_duplicate_keys_at_root_and_nested_depth(self):
        raw = signed()
        self.reject(b'{"schema_version":"SECRET",' + raw[1:])
        self.reject(b'{"extra":{"a":1,"a":2},' + raw[1:])
        self.reject(raw.replace(b'"maximum_requests":8',
                                b'"maximum_requests":8,"maximum_requests":8'))

    def test_nonfinite_invalid_unicode_and_unexpected_top_levels(self):
        for raw in (b"null", b"true", b"1", b'"SECRET"', b"[]", b"{}", b"{", b"\xff",
                    b"\xef\xbb\xbf" + signed(), b'{"maximum_requests":NaN}',
                    b'{"maximum_requests":Infinity}', b'{"maximum_requests":-Infinity}',
                    b'{"maximum_requests":1e999}', b'{"mode":"\\ud800"}'):
            with self.subTest(length=len(raw)):
                self.reject(raw)

    def test_deep_and_oversized_numbers_fail_fixed_without_recursion_leak(self):
        for raw in (b"[" * 2000 + b"0" + b"]" * 2000,
                    b'{"mode":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}",
                    b'{"maximum_requests":' + b"9" * 6000 + b"}"):
            self.reject(raw)

    def test_constant_time_comparisons_cover_key_token_and_mac(self):
        original = authority.hmac.compare_digest
        with patch.object(authority.hmac, "compare_digest", wraps=original) as compare:
            self.verify()
        expected = {digest(KEY), digest(TOKEN), json.loads(signed())["mac_sha256"]}
        compared = {value for call in compare.call_args_list for value in call.args
                    if type(value) is str}
        self.assertLessEqual(expected, compared)

    def test_underlying_error_text_never_appears_in_public_exception(self):
        secret = "SYNTHETIC-SECRET-EXCEPTION-DETAILS"
        for target in ("parse_plan",):
            with patch.object(authority.model, target, side_effect=ValueError(secret)):
                error = self.reject()
                self.assertNotIn(secret, str(error))
                self.assertNotIn(secret, repr(error))
        error = authority.AuthorityError(secret)
        self.assertNotIn(secret, repr(error))

    def test_canonical_is_ascii_deterministic_and_rejects_nonfinite(self):
        self.assertEqual(authority.canonical({"z": "\u00e9", "a": 1}),
                         b'{"a":1,"z":"\\u00e9"}')
        for value in (math.nan, math.inf, -math.inf, object()):
            with self.subTest(kind=type(value).__name__):
                with self.assertRaises(authority.AuthorityError) as caught:
                    authority.canonical(value)
                self.assertEqual(str(caught.exception), "dispatch_authority_rejected")

    def test_verification_performs_no_clock_file_or_network_access(self):
        # Imports are already complete; even ordinary ambient access is forbidden here.
        with patch("builtins.open", side_effect=AssertionError("file_access")), \
                patch("socket.socket", side_effect=AssertionError("socket_access")), \
                patch("socket.getaddrinfo", side_effect=AssertionError("dns_access")), \
                patch("time.monotonic_ns", side_effect=AssertionError("clock_access")), \
                patch("time.time_ns", side_effect=AssertionError("wall_access")):
            self.assertEqual(self.verify()["maximum_requests"], 8)


if __name__ == "__main__":
    unittest.main()
