"""Stdlib-only admission for ONE fixed synthetic CPU integration, never a GPU permit.

This request describes a local fixture test, not owner approval, a production
contract, a real-data permission or an authenticated resource-supervisor receipt.
The fixed task cannot accept model/data paths, credentials, hardware overrides
or arbitrary worker arguments. The real entrypoint must check its actual runtime
before adding ML paths and must enforce its separate owned-process watchdog.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import sys
from types import MappingProxyType

MAXIMUM_REQUEST_BYTES = 4096
SCHEMA = "lumi.sft-offline-cpu-request.v0.1"
ERROR_CODES = frozenset({"offline_cli_rejected", "pre_ml_runtime_required", "request_rejected"})
ML_ROOTS = frozenset({"torch", "transformers", "numpy", "safetensors", "tokenizers",
                      "cuda", "cupy", "tensorflow", "jax"})
_REQUEST = MappingProxyType({
    "schema_version": SCHEMA,
    "scope": "synthetic_cpu_sft_only",
    "operation": "ddp_two_step_fresh_resume",
    "device": "cpu", "backend": "gloo", "world_size": 2,
    "precision": "fp32", "attention_implementation": "eager",
    "record_limit": 4, "target_steps": 2,
    "per_rank_batch_size": 1, "gradient_accumulation_steps": 1,
    "maximum_runtime_seconds": 180, "maximum_private_write_bytes": 67_108_864,
    "production_eligible": False, "real_model_allowed": False,
    "external_network_allowed": False, "inference_allowed": False,
})


class AdmissionError(ValueError):
    def __init__(self, code="request_rejected"):
        self.code = code if type(code) is str and code in ERROR_CODES else "request_rejected"
        super().__init__(self.code)


def need(condition, code="request_rejected"):
    if not condition:
        raise AdmissionError(code) from None


def fixed_request():
    """Return a fresh option snapshot, not an execution/approval capability."""
    return dict(_REQUEST)


def canonical(value):
    failed = False
    result = None
    try:
        result = json.dumps(value, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=True, allow_nan=False).encode("ascii")
    except BaseException:
        failed = True
    if failed:
        raise AdmissionError from None
    return result


def digest(raw):
    need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_REQUEST_BYTES)
    return hashlib.sha256(raw).hexdigest()


def validate_cli(argv):
    """No paths, GPU execution switch, target choices or arbitrary command API."""
    need(type(argv) is list and len(argv) == 1
         and type(argv[0]) is str and argv[0] == "--run-synthetic-cpu", "offline_cli_rejected")


def _flags(no_site, no_bytecode):
    need(type(no_site) is int and no_site == 1
         and type(no_bytecode) in (bool, int) and no_bytecode == 1, "pre_ml_runtime_required")


def pre_ml_runtime_check():
    """Check the ACTUAL interpreter before runtime discovery/ML/fixtures/spawn.

    Call independently in the stdlib parent and every fresh worker interpreter.
    This is a sequencing assertion in trusted source, not a general code sandbox.
    """
    _flags(sys.flags.no_site, sys.dont_write_bytecode)
    need(not any(name.split(".", 1)[0] in ML_ROOTS for name in tuple(sys.modules)),
         "pre_ml_runtime_required")


def validate_request(raw_bytes, expected_sha256, *, no_site, no_bytecode):
    """Pure exact request check; supplied flags/pin are not externally authenticated.

    The real entrypoint separately calls pre_ml_runtime_check(). Ordinary unit
    tests can pass explicit observed/synthetic flag values without claiming the
    pytest process is an isolated interpreter. No file, model or runtime is read.
    """
    _flags(no_site, no_bytecode)
    failed = False
    result = None
    try:
        need(type(raw_bytes) is bytes and 0 < len(raw_bytes) <= MAXIMUM_REQUEST_BYTES)
        need(type(expected_sha256) is str
             and re.fullmatch(r"[a-f0-9]{64}", expected_sha256) is not None)
        need(hmac.compare_digest(digest(raw_bytes), expected_sha256))

        def pairs(items):
            value = {}
            for key, item in items:
                need(key not in value)
                value[key] = item
            return value

        def constant(_value):
            raise AdmissionError from None

        value = json.loads(raw_bytes.decode("ascii"), object_pairs_hook=pairs,
                           parse_constant=constant)
        need(type(value) is dict and value.keys() == _REQUEST.keys())
        for name, expected in _REQUEST.items():
            need(type(value[name]) is type(expected) and value[name] == expected)
        need(canonical(value) == raw_bytes)
        result = dict(value)
    except BaseException:
        failed = True
    if failed:
        raise AdmissionError from None
    return result
