"""One-shot, pre-ML release verification and a trusted release callback.

This adapter never accepts a caller-supplied verification receipt. It invokes
the accepted verifier once using construction-time inputs, checks the owning
process, parent and cooperative deadline, and only then calls a trusted release
callback. It does not authenticate a caller, freeze a verified tree, supervise
arbitrary Python code, or grant permission for real SFT or production use.

Use inside a trusted, single-threaded bootstrap. The caller supplies the expected
manifest pin, owner IDs, parent PID, deadline and bounded limits independently.
The deadline cannot interrupt a blocked verifier syscall or release callback;
an independent supervisor still owns hard termination and execution policy.
"""

from __future__ import annotations

import os
import threading
import time

from .posttraining_sft_release_verifier_v1 import (
    ERROR_CODES as VERIFIER_ERROR_CODES,
)
from .posttraining_sft_release_verifier_v1 import (
    NONCLAIMS,
    ReleaseVerificationError,
    verify_release,
)
from .posttraining_sft_release_verifier_v1 import (
    SCHEMA as VERIFICATION_SCHEMA,
)

SCHEMA = "lumi.sft-release-gate.v0.1"
ERROR_CODES = VERIFIER_ERROR_CODES | frozenset({
    "gate_configuration_rejected", "gate_already_consumed", "gate_owner_rejected",
    "gate_thread_rejected", "gate_parent_lost", "gate_deadline_exhausted",
    "gate_receipt_rejected", "gate_verification_failed", "gate_release_failed",
    "gate_release_return_rejected", "gate_internal_failed",
})
STATUS_CODES = ERROR_CODES | frozenset({
    "not_started", "verification_pending", "release_pending", "completed",
})
STATES = frozenset({"ready", "verifying", "releasing", "released", "failed"})
_VERIFICATION_COUNTS = (
    "files_verified", "directories_verified", "content_read_bytes", "reserved_read_bytes",
    "tree_entries_examined", "open_attempts", "handles_opened", "handles_closed",
)
_VERIFICATION_FIELDS = frozenset({
    "schema", "status", "manifest_sha256", "exact_tree_matched", "root_identity_stable",
    *_VERIFICATION_COUNTS, *NONCLAIMS,
})


class ReleaseGateError(ValueError):
    """Only allowlisted codes; never include input, identity or exception text."""

    def __init__(self, code="gate_internal_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "gate_internal_failed"
        super().__init__(self.code)


def _need(condition, code):
    if not condition:
        raise ReleaseGateError(code) from None


def _safe_code(code):
    return code if type(code) is str and code in ERROR_CODES else "gate_internal_failed"


class VerifiedRelease:
    """Internal one-shot adapter, not an adversarial Python object capability.

    Construction performs no filesystem reads. Manifest/schema/hash and content
    checks remain the verifier's contract; limits are copied from an exact dict.
    A completed counter means that stage returned and its checks were accepted.
    Calling again, including reentry, fails closed and never retries either stage.
    """

    __slots__ = (
        "_root", "_raw_manifest", "_expected_sha256", "_limits", "_expected_uid",
        "_expected_gid", "_expected_parent_pid", "_deadline_ns", "_release", "_owner_pid",
        "_consumed", "_state", "_code", "_failure", "_verification_attempts",
        "_verification_completed", "_release_attempts", "_release_completed",
    )

    def __init__(self, *, root, raw_manifest, expected_sha256, limits, expected_uid,
                 expected_gid, expected_parent_pid, deadline_ns, release):
        failure = False
        try:
            _need(type(limits) is dict, "gate_configuration_rejected")
            _need(type(expected_parent_pid) is int and expected_parent_pid > 0
                  and type(deadline_ns) is int and deadline_ns > 0 and callable(release),
                  "gate_configuration_rejected")
            owner_pid = os.getpid()
            _need(type(owner_pid) is int and owner_pid > 0, "gate_configuration_rejected")
            self._root, self._raw_manifest = root, raw_manifest
            self._expected_sha256, self._limits = expected_sha256, dict(limits)
            self._expected_uid, self._expected_gid = expected_uid, expected_gid
            self._expected_parent_pid, self._deadline_ns = expected_parent_pid, deadline_ns
            self._release, self._owner_pid = release, owner_pid
            self._consumed, self._failure = False, None
            self._state, self._code = "ready", "not_started"
            self._verification_attempts = self._verification_completed = 0
            self._release_attempts = self._release_completed = 0
        except BaseException:
            failure = True
        if failure:
            raise ReleaseGateError("gate_configuration_rejected") from None

    def status(self):
        """Fresh aggregate-only status; no paths, pins, IDs, receipts or raw errors."""
        return {
            "schema": SCHEMA, "state": self._state, "code": self._code,
            "verification_attempts": self._verification_attempts,
            "verification_completed": self._verification_completed,
            "release_attempts": self._release_attempts,
            "release_completed": self._release_completed,
            "real_sft_execution_permitted": False, "production_eligible": False,
        }

    def _fail(self, code):
        if self._failure is None:
            self._failure = _safe_code(code)
        self._state, self._code = "failed", self._failure

    def _guard(self):
        _need(os.getpid() == self._owner_pid, "gate_owner_rejected")
        _need(threading.current_thread() is threading.main_thread()
              and threading.active_count() == 1, "gate_thread_rejected")
        _need(os.getppid() == self._expected_parent_pid, "gate_parent_lost")
        now = time.monotonic_ns()
        _need(type(now) is int and 0 <= now < self._deadline_ns, "gate_deadline_exhausted")

    def _receipt(self, result):
        _need(type(result) is dict and set(result) == _VERIFICATION_FIELDS,
              "gate_receipt_rejected")
        _need(type(result["schema"]) is str and result["schema"] == VERIFICATION_SCHEMA
              and type(result["status"]) is str and result["status"] == "verified"
              and type(result["manifest_sha256"]) is str
              and result["manifest_sha256"] == self._expected_sha256
              and result["exact_tree_matched"] is True
              and result["root_identity_stable"] is True
              and all(result[key] is False for key in NONCLAIMS)
              and all(type(result[key]) is int and result[key] >= 0
                      for key in _VERIFICATION_COUNTS)
              and result["handles_opened"] == result["handles_closed"]
              and result["handles_opened"] <= result["open_attempts"],
              "gate_receipt_rejected")

    def __call__(self):
        if self._consumed:
            self._fail("gate_already_consumed")
            raise ReleaseGateError("gate_already_consumed") from None
        self._consumed = True  # Consume before checks or any verifier/callback work.
        stage, failure = "guard", None
        try:
            self._state, self._code = "verifying", "verification_pending"
            self._guard()
            self._verification_attempts += 1
            stage = "verification"
            result = verify_release(
                self._root, self._raw_manifest, self._expected_sha256,
                limits=dict(self._limits), expected_uid=self._expected_uid,
                expected_gid=self._expected_gid,
            )
            stage = "guard"
            self._guard()  # Immediately after the verifier; never release after parent loss/expiry.
            _need(self._state == "verifying", "gate_already_consumed")
            self._receipt(result)
            self._verification_completed += 1
            self._state, self._code = "releasing", "release_pending"
            self._release_attempts += 1
            stage = "release"
            result = self._release()
            stage = "guard"
            _need(self._state == "releasing", "gate_already_consumed")
            _need(result is None, "gate_release_return_rejected")
            self._release_completed += 1
            self._state, self._code = "released", "completed"
        except BaseException as error:
            if self._failure is not None:
                failure = self._failure
            elif stage == "verification":
                failure = (_safe_code(getattr(error, "code", None))
                           if type(error) is ReleaseVerificationError
                           else "gate_verification_failed")
            elif stage == "release":
                failure = "gate_release_failed"
            else:
                failure = (_safe_code(getattr(error, "code", None))
                           if type(error) is ReleaseGateError
                           else "gate_internal_failed")
        if failure is not None:
            self._fail(failure)
            raise ReleaseGateError(failure) from None
