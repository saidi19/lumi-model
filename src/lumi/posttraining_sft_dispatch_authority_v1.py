"""Pure, pinned HMAC dispatch-contract verification; no issuer or credential I/O.

The external caller supplies the key, its trusted digest, authority bytes and pins.
Verification proves consistency with that supplied key, NOT rights-owner approval,
provider account identity, global attempt exclusivity or permission to run SFT.
The frozen local-only model is unchanged. No secret is read, printed or persisted.
Clearing Python references would not establish secure memory erasure.
"""

from __future__ import annotations

import hashlib
import hmac
import json

from . import posttraining_sft_host_lifecycle_v1 as model
from . import posttraining_sft_lambda_http_v1 as http

SCHEMA = "lumi.sft-lambda-dispatch-authority.v0.1"
DOMAIN = SCHEMA.encode("ascii") + b"\0"
MAXIMUM_BYTES = 8192
HASH_KEYS = frozenset({"plan_sha256", "account_scope_sha256", "approval_sha256",
                       "launch_request_sha256", "credential_sha256", "mac_sha256"})
INTEGER_KEYS = frozenset({"create_not_before_unix", "create_expires_unix", "maximum_requests"})
FIXED = {
    "schema_version": SCHEMA,
    "mode": "externally_issued_single_attempt_dispatch",
    "provider_host": "cloud.lambda.ai",
    "cleanup_scope": "only_fresh_create_bound_id_within_lifecycle_caps",
}
KEYS = HASH_KEYS | INTEGER_KEYS | FIXED.keys()


class AuthorityError(ValueError):
    def __init__(self, _value=None):
        super().__init__("dispatch_authority_rejected")


def need(condition):
    if not condition:
        raise AuthorityError from None


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except Exception:
        raise AuthorityError from None


def _pairs(items):
    value = {}
    for key, item in items:
        need(key not in value)
        value[key] = item
    return value


def _constant(_value):
    raise AuthorityError from None


def verify_authority(raw, expected_sha256, *, key, expected_key_sha256,
                     plan_bytes, plan_sha256, token):
    """Return only the bound frozen plan, never the key/token/authority object.

This function does not evaluate current time or dispatch a request. The controller
must enforce fresh create-window checks, durable reservations and lifecycle caps;
cleanup is restricted to its exact newly create-bound target, not arbitrary IDs.
Trusted external pins are inputs, not independently authenticated identities.
"""
    try:
        need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_BYTES)
        need(model.matches(expected_sha256, model.HASH)
             and model.matches(expected_key_sha256, model.HASH))
        need(type(key) is bytes and len(key) == 32)
        need(hmac.compare_digest(hashlib.sha256(key).hexdigest(), expected_key_sha256))
        need(hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_sha256))
        need(type(token) is bytes and len(token) <= 512
             and http.TOKEN.fullmatch(token) is not None)
        plan = model.parse_plan(plan_bytes, plan_sha256)
        value = json.loads(raw.decode("ascii"), object_pairs_hook=_pairs,
                           parse_constant=_constant)
        need(type(value) is dict and value.keys() == KEYS)
        for field, expected in FIXED.items():
            need(type(value[field]) is str and value[field] == expected)
        for field in HASH_KEYS:
            need(model.matches(value[field], model.HASH))
        for field in INTEGER_KEYS:
            need(type(value[field]) is int and value[field] == plan[field])
        need(hmac.compare_digest(value["plan_sha256"], plan_sha256))
        for field in ("account_scope_sha256", "approval_sha256", "launch_request_sha256"):
            need(hmac.compare_digest(value[field], plan[field]))
        need(hmac.compare_digest(value["credential_sha256"], hashlib.sha256(token).hexdigest()))
        need(canonical(value) == raw)
        unsigned = {field: item for field, item in value.items() if field != "mac_sha256"}
        expected_mac = hmac.new(key, DOMAIN + canonical(unsigned), hashlib.sha256).hexdigest()
        need(hmac.compare_digest(value["mac_sha256"], expected_mac))
        return plan
    except Exception:
        raise AuthorityError from None
