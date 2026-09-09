"""One-shot sequencing via inert syscall/verifier doubles; no release-tree reads."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_release_gate_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
gate = importlib.import_module(PACKAGE + ".posttraining_sft_release_gate_v1")
IMPORT_ADDITIONS = set(sys.modules) - before_import
ORIGINAL_VERIFY = gate.verify_release
VERIFIER = sys.modules[PACKAGE + ".posttraining_sft_release_verifier_v1"]
RAW = b"SYNTHETIC-MANIFEST-NOT-A-REAL-RELEASE"
PIN = hashlib.sha256(RAW).hexdigest()
LIMITS = {
    "maximum_manifest_bytes": 32768, "maximum_entries": 16,
    "maximum_file_bytes": 4096, "maximum_total_bytes": 16384,
    "maximum_tree_entries": 64, "maximum_content_read_bytes": 32768,
    "maximum_opens": 128, "maximum_runtime_ms": 1000,
}
STATUS_FIELDS = {
    "schema", "state", "code", "verification_attempts", "verification_completed",
    "release_attempts", "release_completed", "real_sft_execution_permitted", "production_eligible",
}
COUNTERS = (
    "verification_attempts", "verification_completed", "release_attempts", "release_completed",
)


def receipt():
    return {
        "schema": gate.VERIFICATION_SCHEMA, "status": "verified", "manifest_sha256": PIN,
        "files_verified": 3, "directories_verified": 6, "content_read_bytes": 64,
        "reserved_read_bytes": 67, "tree_entries_examined": 8, "open_attempts": 24,
        "handles_opened": 24, "handles_closed": 24,
        "exact_tree_matched": True, "root_identity_stable": True,
        **dict.fromkeys(gate.NONCLAIMS, False),
    }


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.os = types.SimpleNamespace(getpid=Mock(return_value=1234),
                                        getppid=Mock(return_value=1233))
        main = object()
        self.threads = types.SimpleNamespace(
            current_thread=Mock(return_value=main), main_thread=Mock(return_value=main),
            active_count=Mock(return_value=1),
        )
        self.clock = types.SimpleNamespace(monotonic_ns=Mock(return_value=100))
        self.release = Mock(return_value=None)
        self.verify = Mock(side_effect=lambda *args, **kwargs: receipt())
        for name, value in (("os", self.os), ("threading", self.threads), ("time", self.clock),
                            ("verify_release", self.verify)):
            selected = patch.object(gate, name, value)
            selected.start()
            self.addCleanup(selected.stop)

    def make(self, **overrides):
        arguments = {
            "root": "/synthetic/no-io/release", "raw_manifest": RAW, "expected_sha256": PIN,
            "limits": dict(LIMITS), "expected_uid": 1000, "expected_gid": 1000,
            "expected_parent_pid": 1233, "deadline_ns": 1000, "release": self.release,
        }
        arguments.update(overrides)
        return gate.VerifiedRelease(**arguments)

    def assert_status(self, selected, state, code, counts):
        result = selected.status()
        self.assertEqual(set(result), STATUS_FIELDS)
        self.assertEqual(result["schema"], gate.SCHEMA)
        self.assertEqual((result["state"], result["code"]), (state, code))
        self.assertIn(result["state"], gate.STATES)
        self.assertIn(result["code"], gate.STATUS_CODES)
        self.assertEqual(tuple(result[key] for key in COUNTERS), counts)
        for key in COUNTERS:
            self.assertIs(type(result[key]), int)
            self.assertIn(result[key], (0, 1))
        self.assertIs(result["real_sft_execution_permitted"], False)
        self.assertIs(result["production_eligible"], False)
        text = json.dumps(result)
        for forbidden in ("/synthetic", "1234", "1233", PIN, RAW.decode(), "PRIVATE-EXCEPTION"):
            self.assertNotIn(forbidden, text)
        return result

    def reject(self, selected, code, counts=(0, 0, 0, 0)):
        with self.assertRaises(gate.ReleaseGateError) as caught:
            selected()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.args, (code,))
        self.assertEqual(str(caught.exception), code)
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assert_status(selected, "failed", code, counts)
        return caught.exception

    def test_import_additions_are_stdlib_and_source_only(self):
        forbidden = {"torch", "numpy", "transformers", "tensorflow", "jax", "cuda", "cupy"}
        self.assertFalse({name.split(".", 1)[0] for name in IMPORT_ADDITIONS} & forbidden)
        self.assertFalse(any("fixture" in name or "_linux_" in name for name in IMPORT_ADDITIONS))

    def test_accepted_verifier_is_imported_directly(self):
        self.assertIs(ORIGINAL_VERIFY, VERIFIER.verify_release)

    def test_api_has_no_caller_supplied_receipt_parameter(self):
        self.assertEqual(set(inspect.signature(gate.VerifiedRelease).parameters), {
            "root", "raw_manifest", "expected_sha256", "limits", "expected_uid", "expected_gid",
            "expected_parent_pid", "deadline_ns", "release",
        })
        self.assertEqual(tuple(inspect.signature(gate.VerifiedRelease.__call__).parameters),
                         ("self",))

    def test_constructor_does_no_verifier_callback_or_filesystem_work(self):
        with patch("builtins.open", side_effect=AssertionError("unexpected filesystem read")):
            selected = self.make()
        self.verify.assert_not_called()
        self.release.assert_not_called()
        self.os.getpid.assert_called_once_with()
        self.os.getppid.assert_not_called()
        self.clock.monotonic_ns.assert_not_called()
        self.assert_status(selected, "ready", "not_started", (0, 0, 0, 0))

    def test_status_is_a_fresh_fixed_aggregate_dict(self):
        selected = self.make()
        status = self.assert_status(selected, "ready", "not_started", (0, 0, 0, 0))
        status.update(state="released", release_completed=22, raw=RAW)
        self.assert_status(selected, "ready", "not_started", (0, 0, 0, 0))

    def test_constructor_copies_limits_before_caller_mutation(self):
        limits = dict(LIMITS)
        selected = self.make(limits=limits)
        limits.clear()
        limits["maximum_opens"] = 1
        selected()
        self.verify.assert_called_once_with(
            "/synthetic/no-io/release", RAW, PIN, limits=LIMITS,
            expected_uid=1000, expected_gid=1000,
        )
        self.assertIsNot(self.verify.call_args.kwargs["limits"], limits)

    def test_constructor_requires_exact_dict_limits(self):
        class DictSubclass(dict):
            pass

        for value in (None, [], (), types.MappingProxyType(LIMITS), DictSubclass(LIMITS), True):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(gate.ReleaseGateError) as caught:
                    self.make(limits=value)
                self.assertEqual(caught.exception.code, "gate_configuration_rejected")
                self.assertIsNone(caught.exception.__context__)
        self.verify.assert_not_called()
        self.release.assert_not_called()

    def test_parent_pid_must_be_an_exact_positive_int(self):
        for value in (0, -1, True, False, 1.0, "1233", None, [], {}):
            with self.subTest(value=value):
                with self.assertRaises(gate.ReleaseGateError) as caught:
                    self.make(expected_parent_pid=value)
                self.assertEqual(str(caught.exception), "gate_configuration_rejected")
        self.verify.assert_not_called()

    def test_deadline_must_be_an_exact_positive_int(self):
        for value in (0, -1, True, False, 1000.0, "1000", None, [], {}):
            with self.subTest(value=value):
                with self.assertRaises(gate.ReleaseGateError) as caught:
                    self.make(deadline_ns=value)
                self.assertEqual(str(caught.exception), "gate_configuration_rejected")
        self.verify.assert_not_called()

    def test_release_must_be_callable(self):
        for value in (None, False, 0, "release", [], {}):
            with self.subTest(value=value):
                with self.assertRaises(gate.ReleaseGateError) as caught:
                    self.make(release=value)
                self.assertEqual(str(caught.exception), "gate_configuration_rejected")

    def test_constructor_owner_pid_must_be_valid(self):
        for value in (0, -1, True, 1.0, None):
            with self.subTest(value=value):
                self.os.getpid.return_value = value
                with self.assertRaises(gate.ReleaseGateError) as caught:
                    self.make()
                self.assertEqual(str(caught.exception), "gate_configuration_rejected")

    def test_constructor_syscall_failure_is_raw_text_free(self):
        self.os.getpid.side_effect = OSError("PRIVATE-EXCEPTION-CONSTRUCTOR")
        with self.assertRaises(gate.ReleaseGateError) as caught:
            self.make()
        self.assertEqual(str(caught.exception), "gate_configuration_rejected")
        self.assertIsNone(caught.exception.__context__)

    def test_content_contract_stays_with_the_verifier(self):
        self.verify.side_effect = gate.ReleaseVerificationError("manifest_hash_mismatch")
        selected = self.make(raw_manifest=b"SYNTHETIC-TAMPER", expected_sha256="0" * 64)
        self.verify.assert_not_called()
        self.reject(selected, "manifest_hash_mismatch", (1, 0, 0, 0))
        self.verify.assert_called_once_with(
            "/synthetic/no-io/release", b"SYNTHETIC-TAMPER", "0" * 64,
            limits=LIMITS, expected_uid=1000, expected_gid=1000,
        )
        self.release.assert_not_called()

    def test_success_orders_verification_before_exactly_one_release(self):
        events = []
        selected = self.make()

        def verify(*args, **kwargs):
            events.append("verify")
            self.assert_status(selected, "verifying", "verification_pending", (1, 0, 0, 0))
            return receipt()

        def release():
            events.append("release")
            self.assert_status(selected, "releasing", "release_pending", (1, 1, 1, 0))

        self.verify.side_effect, self.release.side_effect = verify, release
        self.assertIsNone(selected())
        self.assertEqual(events, ["verify", "release"])
        self.verify.assert_called_once()
        self.release.assert_called_once_with()
        self.assertEqual(self.os.getppid.call_count, 2)
        self.assertEqual(self.clock.monotonic_ns.call_count, 2)
        self.assert_status(selected, "released", "completed", (1, 1, 1, 1))

    def test_second_call_cannot_retry_after_success(self):
        selected = self.make()
        selected()
        self.reject(selected, "gate_already_consumed", (1, 1, 1, 1))
        self.verify.assert_called_once()
        self.release.assert_called_once_with()

    def test_fork_owner_is_rejected_before_verification_and_consumes_adapter(self):
        selected = self.make()
        self.os.getpid.return_value = 4321
        self.reject(selected, "gate_owner_rejected")
        self.os.getpid.return_value = 1234
        with self.assertRaises(gate.ReleaseGateError) as caught:
            selected()
        self.assertEqual(str(caught.exception), "gate_already_consumed")
        self.assert_status(selected, "failed", "gate_owner_rejected", (0, 0, 0, 0))
        self.verify.assert_not_called()
        self.release.assert_not_called()

    def test_owner_change_after_verification_never_releases(self):
        selected = self.make()
        self.os.getpid.side_effect = [1234, 4321]
        self.reject(selected, "gate_owner_rejected", (1, 0, 0, 0))
        self.release.assert_not_called()

    def test_parent_loss_before_verification_never_releases(self):
        selected = self.make()
        self.os.getppid.return_value = 1
        self.reject(selected, "gate_parent_lost")
        self.verify.assert_not_called()
        self.release.assert_not_called()

    def test_parent_loss_immediately_after_verification_never_releases(self):
        selected = self.make()
        self.os.getppid.side_effect = [1233, 1]
        self.reject(selected, "gate_parent_lost", (1, 0, 0, 0))
        self.release.assert_not_called()

    def test_nonmain_thread_before_verification_never_releases(self):
        selected = self.make()
        self.threads.current_thread.return_value = object()
        self.reject(selected, "gate_thread_rejected")
        self.verify.assert_not_called()
        self.release.assert_not_called()

    def test_nonmain_thread_after_verification_never_releases(self):
        selected = self.make()
        self.threads.current_thread.side_effect = [self.threads.main_thread.return_value, object()]
        self.reject(selected, "gate_thread_rejected", (1, 0, 0, 0))
        self.release.assert_not_called()

    def test_other_live_thread_before_verification_never_releases(self):
        selected = self.make()
        self.threads.active_count.return_value = 2
        self.reject(selected, "gate_thread_rejected")
        self.verify.assert_not_called()
        self.release.assert_not_called()

    def test_other_live_thread_after_verification_never_releases(self):
        selected = self.make()
        self.threads.active_count.side_effect = [1, 2]
        self.reject(selected, "gate_thread_rejected", (1, 0, 0, 0))
        self.release.assert_not_called()

    def test_deadline_boundary_is_closed_before_verification(self):
        selected = self.make()
        self.clock.monotonic_ns.return_value = 1000
        self.reject(selected, "gate_deadline_exhausted")
        self.verify.assert_not_called()
        self.release.assert_not_called()

    def test_invalid_clock_values_fail_closed(self):
        for value in (-1, True, 100.0, None):
            with self.subTest(value=value):
                selected = self.make()
                self.clock.monotonic_ns.return_value = value
                self.reject(selected, "gate_deadline_exhausted")
        self.verify.assert_not_called()

    def test_deadline_expiry_immediately_after_verification_never_releases(self):
        selected = self.make()
        self.clock.monotonic_ns.side_effect = [100, 1000]
        self.reject(selected, "gate_deadline_exhausted", (1, 0, 0, 0))
        self.release.assert_not_called()

    def test_guard_exception_before_verification_is_raw_text_free(self):
        selected = self.make()
        self.os.getppid.side_effect = OSError("PRIVATE-EXCEPTION-PARENT")
        self.reject(selected, "gate_internal_failed")
        self.verify.assert_not_called()
        self.release.assert_not_called()

    def test_guard_exception_after_verification_is_raw_text_free(self):
        selected = self.make()
        self.os.getppid.side_effect = [1233, OSError("PRIVATE-EXCEPTION-PARENT")]
        self.reject(selected, "gate_internal_failed", (1, 0, 0, 0))
        self.release.assert_not_called()

    def test_every_verifier_error_code_is_preserved_without_release(self):
        for code in sorted(gate.VERIFIER_ERROR_CODES):
            with self.subTest(code=code):
                self.verify.reset_mock()
                self.verify.side_effect = gate.ReleaseVerificationError(code)
                self.reject(self.make(), code, (1, 0, 0, 0))
                self.verify.assert_called_once()
        self.release.assert_not_called()

    def test_unknown_verifier_exceptions_are_sanitized(self):
        for error in (OSError, RuntimeError, KeyboardInterrupt, SystemExit, BaseException):
            with self.subTest(error=error.__name__):
                self.verify.side_effect = error("PRIVATE-EXCEPTION-VERIFIER")
                self.reject(self.make(), "gate_verification_failed", (1, 0, 0, 0))
        self.release.assert_not_called()

    def test_error_subclass_cannot_inject_a_status_code(self):
        class ErrorSubclass(gate.ReleaseVerificationError):
            pass

        self.verify.side_effect = ErrorSubclass("manifest_hash_mismatch")
        self.reject(self.make(), "gate_verification_failed", (1, 0, 0, 0))
        self.release.assert_not_called()

    def test_mutated_verifier_error_code_cannot_leak_into_status(self):
        for code in ("PRIVATE-EXCEPTION-CODE", [], None, 1):
            with self.subTest(code=code):
                error = gate.ReleaseVerificationError("digest_mismatch")
                error.code = code
                self.verify.side_effect = error
                self.reject(self.make(), "gate_internal_failed", (1, 0, 0, 0))
        self.release.assert_not_called()

    def test_missing_verifier_error_code_cannot_leak_an_attribute_error(self):
        error = gate.ReleaseVerificationError("digest_mismatch")
        del error.code
        self.verify.side_effect = error
        self.reject(self.make(), "gate_internal_failed", (1, 0, 0, 0))

    def test_receipt_must_be_an_exact_dict_with_exact_fields(self):
        class DictSubclass(dict):
            pass

        missing = receipt()
        del missing["root_identity_stable"]
        for value in (None, True, [], DictSubclass(receipt()), missing,
                      receipt() | {"raw": "PRIVATE-EXCEPTION-RECEIPT"}):
            with self.subTest(value=type(value).__name__):
                self.verify.side_effect = None
                self.verify.return_value = value
                self.reject(self.make(), "gate_receipt_rejected", (1, 0, 0, 0))
        self.release.assert_not_called()

    def test_receipt_requires_verified_schema_matching_pin_and_true_guards(self):
        for key, value in (("schema", "other"), ("status", "failed"),
                           ("manifest_sha256", "0" * 64), ("manifest_sha256", None),
                           ("exact_tree_matched", False), ("exact_tree_matched", 1),
                           ("root_identity_stable", False), ("root_identity_stable", 1)):
            with self.subTest(key=key, value=value):
                self.verify.side_effect = None
                self.verify.return_value = receipt() | {key: value}
                self.reject(self.make(), "gate_receipt_rejected", (1, 0, 0, 0))
        self.release.assert_not_called()

    def test_every_nonclaim_must_remain_exactly_false(self):
        for key in gate.NONCLAIMS:
            for value in (True, 0, None, "false"):
                with self.subTest(key=key, value=value):
                    self.verify.side_effect = None
                    self.verify.return_value = receipt() | {key: value}
                    self.reject(self.make(), "gate_receipt_rejected", (1, 0, 0, 0))
        self.release.assert_not_called()

    def test_receipt_counters_require_exact_nonnegative_ints(self):
        for key in gate._VERIFICATION_COUNTS:
            for value in (True, -1, 1.0, "1", None):
                with self.subTest(key=key, value=value):
                    self.verify.side_effect = None
                    self.verify.return_value = receipt() | {key: value}
                    self.reject(self.make(), "gate_receipt_rejected", (1, 0, 0, 0))
        self.release.assert_not_called()

    def test_receipt_requires_balanced_handles_and_possible_open_attempts(self):
        for change in ({"handles_closed": 23}, {"handles_closed": 25}, {"open_attempts": 23}):
            with self.subTest(change=change):
                self.verify.side_effect = None
                self.verify.return_value = receipt() | change
                self.reject(self.make(), "gate_receipt_rejected", (1, 0, 0, 0))
        self.release.assert_not_called()

    def test_release_non_none_returns_are_rejected_without_retry(self):
        for value in (False, 0, "", [], {}, receipt()):
            with self.subTest(value=type(value).__name__):
                self.release.reset_mock()
                self.release.return_value = value
                selected = self.make()
                self.reject(selected, "gate_release_return_rejected", (1, 1, 1, 0))
                self.release.assert_called_once_with()

    def test_release_exception_text_never_escapes(self):
        for error in (OSError, RuntimeError, KeyboardInterrupt, SystemExit, BaseException):
            with self.subTest(error=error.__name__):
                self.release.reset_mock()
                self.release.side_effect = error("PRIVATE-EXCEPTION-RELEASE")
                self.reject(self.make(), "gate_release_failed", (1, 1, 1, 0))
                self.release.assert_called_once_with()

    def test_verifier_reentry_fails_closed_without_release(self):
        selected = self.make()
        self.verify.side_effect = lambda *args, **kwargs: selected()
        self.reject(selected, "gate_already_consumed", (1, 0, 0, 0))
        self.verify.assert_called_once()
        self.release.assert_not_called()

    def test_swallowed_verifier_reentry_cannot_restore_success(self):
        selected = self.make()

        def reenter(*args, **kwargs):
            with self.assertRaises(gate.ReleaseGateError):
                selected()
            return receipt()

        self.verify.side_effect = reenter
        self.reject(selected, "gate_already_consumed", (1, 0, 0, 0))
        self.verify.assert_called_once()
        self.release.assert_not_called()

    def test_release_reentry_cannot_repeat_either_stage(self):
        selected = self.make()
        self.release.side_effect = selected
        self.reject(selected, "gate_already_consumed", (1, 1, 1, 0))
        self.verify.assert_called_once()
        self.release.assert_called_once_with()

    def test_swallowed_release_reentry_cannot_restore_success(self):
        selected = self.make()

        def reenter():
            with self.assertRaises(gate.ReleaseGateError):
                selected()

        self.release.side_effect = reenter
        self.reject(selected, "gate_already_consumed", (1, 1, 1, 0))
        self.verify.assert_called_once()
        self.release.assert_called_once_with()

    def test_initial_verifier_failure_is_retained_after_second_call(self):
        selected = self.make()
        self.verify.side_effect = gate.ReleaseVerificationError("digest_mismatch")
        self.reject(selected, "digest_mismatch", (1, 0, 0, 0))
        self.verify.side_effect = None
        self.verify.return_value = receipt()
        with self.assertRaises(gate.ReleaseGateError) as caught:
            selected()
        self.assertEqual(str(caught.exception), "gate_already_consumed")
        self.assert_status(selected, "failed", "digest_mismatch", (1, 0, 0, 0))
        self.verify.assert_called_once()
        self.release.assert_not_called()

    def test_unknown_error_constructor_codes_are_sanitized(self):
        for code in (None, True, 1, [], "PRIVATE-EXCEPTION-CODE"):
            with self.subTest(code=code):
                error = gate.ReleaseGateError(code)
                self.assertEqual(error.code, "gate_internal_failed")
                self.assertEqual(str(error), "gate_internal_failed")


if __name__ == "__main__":
    unittest.main()
