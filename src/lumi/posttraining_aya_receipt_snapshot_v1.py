"""Project one unchanged Aya receipt test onto authenticated in-memory bytes.

The caller must independently authenticate the binding manifest. Only selected
original declarations are compiled; imports, candidate readers and finalizers
are excluded. Receipt JSON is parsed only after every input is authenticated.
This checks receipt integrity/nonpromotion, not human authorship or decisions.
No filesystem access, module registration or execution authorization occurs.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import types
from collections.abc import Mapping

from .posttraining_aya_plan_snapshot_v1 import _compile, _select
from .posttraining_safety_replay_snapshot_v1 import Snapshot

MODULE_NAME = "test_aya_bengali_owner_review"
TEST_NAME = "test_actual_owner_receipt_is_complete_and_nonpromoting"
SOURCE_PATHS = {
    "receipt": "evidence/lumi-4b-sft-aya-bengali-owner-review-v0.1.json",
    "review_implementation": "src/lumi/posttraining_aya_owner_review.py",
    "hash_helper_source": "src/lumi/posttraining_oasst1_full_run.py",
    "test_source": "tests/test_aya_bengali_owner_review.py",
}
INPUT_ROLES = frozenset(SOURCE_PATHS)
MAXIMUM_RECEIPT_BYTES = 8192
MAXIMUM_SOURCE_BYTES = 65536
MAXIMUM_TOTAL_BYTES = 131072
_ERROR = "invalid Aya receipt source projection"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_AUTHORIZATION_FLAGS = frozenset({
    "ai_or_automated_owner_approval_authorized",
    "external_api_spend_authorized",
    "network_access_authorized",
    "paid_gpu_authorized",
    "production_dataset_promotion_authorized",
    "production_dataset_seal_authorized",
    "production_sft_authorized",
})
_RECEIPT_FIELDS = frozenset({
    "accepted_records", "all_sampled_rows_accepted", "authorization", "bindings",
    "owner_review_claim", "production_eligible", "production_promotions",
    "raw_training_text_retained", "rejected_record_ids", "rejected_records",
    "remaining_blockers", "reviewed_at", "reviewer_alias", "sample_records",
    "schema_version", "status", "receipt_sha256",
})


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError(_ERROR)


def _authenticate(payloads: dict, bindings: dict, repository_root: str) -> tuple[dict, str]:
    _require(type(repository_root) is str and 1 < len(repository_root) <= 4096)
    _require(repository_root.startswith("/") and not repository_root.startswith("//"))
    _require(posixpath.normpath(repository_root) == repository_root)
    _require("\\" not in repository_root)
    _require(all(ord(char) >= 32 and ord(char) != 127 for char in repository_root))
    _require(not any(part.startswith("~") for part in repository_root.split("/")))
    _require(type(payloads) is dict and type(bindings) is dict)
    _require(set(payloads) == set(bindings) == INPUT_ROLES)
    blobs = {}
    total = 0
    internal_sha256 = None
    for role in sorted(INPUT_ROLES):
        binding, payload = bindings[role], payloads[role]
        _require(type(binding) is dict)
        binding = dict(binding)
        fields = {"path", "bytes", "sha256"}
        if role == "receipt":
            fields.add("internal_sha256")
            internal_sha256 = binding.get("internal_sha256")
            _require(type(internal_sha256) is str and _SHA256.fullmatch(internal_sha256))
        _require(set(binding) == fields)
        _require(type(binding["path"]) is str and binding["path"] == SOURCE_PATHS[role])
        maximum = MAXIMUM_RECEIPT_BYTES if role == "receipt" else MAXIMUM_SOURCE_BYTES
        _require(type(binding["bytes"]) is int and 0 < binding["bytes"] <= maximum)
        _require(type(payload) is bytes and len(payload) == binding["bytes"])
        _require(type(binding["sha256"]) is str and _SHA256.fullmatch(binding["sha256"]))
        total += len(payload)
        _require(total <= MAXIMUM_TOTAL_BYTES)
        _require(hashlib.sha256(payload).hexdigest() == binding["sha256"])
        blobs[role] = payload
    return blobs, internal_sha256


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result)
        result[key] = value
    return result


def _reject_constant(_value: str):
    raise ValueError(_ERROR)


def _check_receipt(raw: bytes, internal_sha256: str) -> None:
    receipt = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )
    _require(type(receipt) is dict and set(receipt) == _RECEIPT_FIELDS)
    _require(receipt["receipt_sha256"] == internal_sha256)
    authorization = receipt["authorization"]
    _require(
        type(authorization) is dict and set(authorization) == _AUTHORIZATION_FLAGS | {"purpose"}
    )
    _require(all(authorization[field] is False for field in _AUTHORIZATION_FLAGS))
    _require(authorization["purpose"] == "private_human_owner_quality_spot_check_only")
    _require(receipt["production_eligible"] is False)
    _require(
        type(receipt["production_promotions"]) is int and receipt["production_promotions"] == 0
    )
    _require(receipt["raw_training_text_retained"] is False)


def build_test_module(payloads: dict, bindings: dict, repository_root: str):
    """Return an unregistered exact-test module and a one-receipt read-only view.

    Source bytes and receipt file/internal digests must be independently pinned
    by the caller. Construction checks duplicate-free JSON and closed operation
    flags before the original test invokes its unchanged receipt validator.
    The caller supplies pytest and fixed-output lifecycle accounting separately.
    Construction consumes zero virtual opens; the exact test consumes one.
    """
    try:
        blobs, internal_sha256 = _authenticate(payloads, bindings, repository_root)
        _check_receipt(blobs["receipt"], internal_sha256)
        review = _select(
            blobs["review_implementation"],
            frozenset({"RECEIPT_SCHEMA_VERSION", "_SHA256"}),
            frozenset({"_require_sha256", "validate_owner_review_receipt"}),
        )
        helper = _select(
            blobs["hash_helper_source"], frozenset(),
            frozenset({"canonical_json_bytes", "sha256_value"}),
        )
        test = _select(blobs["test_source"], frozenset(), frozenset({TEST_NAME}))
        receipt_path = repository_root + "/" + SOURCE_PATHS["receipt"]
        snapshot = Snapshot(
            {receipt_path: blobs["receipt"]},
            repository_root=repository_root,
            maximum_virtual_opens=1,
            maximum_logical_read_bytes=len(blobs["receipt"]),
            maximum_gzip_bytes=1,
        )
        module = types.ModuleType(MODULE_NAME)
        module.__file__ = repository_root + "/" + SOURCE_PATHS["test_source"]
        module.__package__ = ""
        module.__dict__.update({
            "json": json, "re": re, "hashlib": hashlib, "Mapping": Mapping,
            "Path": snapshot.Path, "RECEIPT": snapshot.Path(receipt_path),
        })
        _compile(review, repository_root + "/" + SOURCE_PATHS["review_implementation"],
                 module.__dict__)
        _compile(helper, repository_root + "/" + SOURCE_PATHS["hash_helper_source"],
                 module.__dict__)
        _compile(test, module.__file__, module.__dict__)
        _require({name for name in vars(module) if name.startswith("test_")} == {TEST_NAME})
        return module, snapshot
    except Exception:
        raise ValueError(_ERROR) from None
