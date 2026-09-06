"""Project the unchanged Aya sample test onto authenticated in-memory bytes.

The caller independently authenticates the binding manifest and authorizes any
physical reads. Only selected original declarations are compiled; no historical
imports, owner decisions, receipt validator or finalizer are executed. Candidate
and report bodies are parsed only by the original test and loader. The caller
must suppress historical exceptions and emit only aggregate results: this test
necessarily holds private candidate rows in process memory while it executes.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import types
from collections import defaultdict
from collections.abc import Mapping

from .posttraining_aya_plan_snapshot_v1 import _compile, _select
from .posttraining_safety_replay_snapshot_v1 import Snapshot

MODULE_NAME = "test_aya_bengali_owner_review"
TEST_NAME = "test_actual_plan_rebuilds_exact_sample"
PLAN_BINDING_ROLES = frozenset({
    "accepted_records", "attribution_manifest", "materialization_report",
    "owner_review_packet", "receipt_finalizer", "review_implementation",
})
SOURCE_PATHS = {
    "plan": "configs/lumi-4b-sft-aya-bengali-owner-review-v0.1.json",
    "review_implementation": "src/lumi/posttraining_aya_owner_review.py",
    "receipt_finalizer": "scripts/finalize_aya_owner_review.py",
    "materialization_report": "evidence/lumi-4b-sft-aya-bengali-materialization-v0.1.json",
    "accepted_records": (
        "../lumi-artifacts/4b-sft-aya-bengali-materialization-v0.1/"
        "6d7c20b19f6b/attempt-001/accepted-records.jsonl"
    ),
    "attribution_manifest": (
        "../lumi-artifacts/4b-sft-aya-bengali-materialization-v0.1/"
        "6d7c20b19f6b/attempt-001/attribution-manifest.jsonl"
    ),
    "owner_review_packet": (
        "../lumi-artifacts/4b-sft-aya-bengali-materialization-v0.1/"
        "6d7c20b19f6b/attempt-001/owner-review-32.html"
    ),
    "file_hash_source": "src/lumi/posttraining_first_party.py",
    "sample_helper_source": "src/lumi/posttraining_oasst1_full_run.py",
    "test_source": "tests/test_aya_bengali_owner_review.py",
}
INPUT_ROLES = frozenset(SOURCE_PATHS)
CODE_ROLES = frozenset({
    "review_implementation", "receipt_finalizer", "file_hash_source",
    "sample_helper_source", "test_source",
})
AYA_CONSTANTS = frozenset({
    "PLAN_SCHEMA_VERSION", "EXPECTED_AUTHORIZATION", "EXPECTED_CANDIDATE",
    "EXPECTED_LIMITS", "EXPECTED_OUTPUT", "EXPECTED_BINDINGS", "_SHA256",
})
AYA_FUNCTIONS = frozenset({
    "_load_json", "_require_sha256", "resolve_plan_binding",
    "load_owner_review_plan", "load_exact_owner_sample",
})
MAXIMUM_TOTAL_BYTES = 8 * 1024 * 1024
MAXIMUM_CANDIDATE_BYTES = 6_000_000
MAXIMUM_PRIVATE_BYTES = 2 * 1024 * 1024
MAXIMUM_SOURCE_BYTES = 65_536
MAXIMUM_VIRTUAL_OPENS = 9
_ERROR = "invalid Aya sample source projection"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError(_ERROR)


def _authenticate(payloads: dict, bindings: dict, repository_root: str) -> tuple[dict, dict]:
    _require(type(repository_root) is str and 1 < len(repository_root) <= 4096)
    _require(repository_root.startswith("/") and not repository_root.startswith("//"))
    _require(posixpath.normpath(repository_root) == repository_root)
    _require("\\" not in repository_root)
    _require(all(ord(char) >= 32 and ord(char) != 127 for char in repository_root))
    _require(not any(part.startswith("~") for part in repository_root.split("/")))
    _require(type(payloads) is dict and type(bindings) is dict)
    _require(set(payloads) == set(bindings) == INPUT_ROLES)
    blobs, checked = {}, {}
    total = 0
    for role in sorted(INPUT_ROLES):
        binding, payload = bindings[role], payloads[role]
        _require(type(binding) is dict and set(binding) == {"path", "bytes", "sha256"})
        binding = dict(binding)
        _require(type(binding["path"]) is str and binding["path"] == SOURCE_PATHS[role])
        maximum = (
            MAXIMUM_SOURCE_BYTES if role in CODE_ROLES
            else MAXIMUM_CANDIDATE_BYTES if role == "accepted_records"
            else MAXIMUM_PRIVATE_BYTES
        )
        _require(type(binding["bytes"]) is int and 0 < binding["bytes"] <= maximum)
        _require(type(payload) is bytes and len(payload) == binding["bytes"])
        _require(type(binding["sha256"]) is str and _SHA256.fullmatch(binding["sha256"]))
        total += len(payload)
        _require(total <= MAXIMUM_TOTAL_BYTES)
        _require(hashlib.sha256(payload).hexdigest() == binding["sha256"])
        blobs[role], checked[role] = payload, binding
    return blobs, checked


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result)
        result[key] = value
    return result


def _reject_constant(_value: str):
    raise ValueError(_ERROR)


def _check_plan(raw: bytes, bindings: dict) -> None:
    """Bind the input plan only; do not parse report, packet or candidate here."""
    plan = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )
    _require(type(plan) is dict)
    observed = plan.get("bindings")
    _require(type(observed) is dict and set(observed) == PLAN_BINDING_ROLES)
    for role in PLAN_BINDING_ROLES:
        _require(observed[role] == {
            "path": bindings[role]["path"], "sha256": bindings[role]["sha256"],
        })


def build_test_module(payloads: dict, bindings: dict, repository_root: str):
    """Return an unregistered original-test module and seven-blob snapshot.

    All ten file bindings are authenticated before any AST or JSON parsing.
    Construction uses zero virtual opens. The original test makes nine: plan,
    six binding hashes, report JSON and candidate JSONL. The file hash function
    is selected from its actual first-party import origin, not the identical
    helper in the OASST1 module. No owner approval is implied by this projection.
    """
    try:
        blobs, checked = _authenticate(payloads, bindings, repository_root)
        _check_plan(blobs["plan"], checked)
        review = _select(blobs["review_implementation"], AYA_CONSTANTS, AYA_FUNCTIONS)
        file_hash = _select(
            blobs["file_hash_source"], frozenset(), frozenset({"file_sha256"}),
        )
        sample_helpers = _select(
            blobs["sample_helper_source"], frozenset(),
            frozenset({"canonical_json_bytes", "sha256_value", "deterministic_owner_sample"}),
        )
        test = _select(blobs["test_source"], frozenset(), frozenset({TEST_NAME}))
        view_roles = PLAN_BINDING_ROLES | {"plan"}
        logical_bytes = (
            sum(checked[role]["bytes"] for role in view_roles)
            + checked["accepted_records"]["bytes"]
            + checked["materialization_report"]["bytes"]
        )
        snapshot = Snapshot(
            {
                posixpath.normpath(repository_root + "/" + SOURCE_PATHS[role]): blobs[role]
                for role in view_roles
            },
            repository_root=repository_root,
            maximum_virtual_opens=MAXIMUM_VIRTUAL_OPENS,
            maximum_logical_read_bytes=logical_bytes,
            maximum_gzip_bytes=1,
        )
        module = types.ModuleType(MODULE_NAME)
        module.__file__ = repository_root + "/" + SOURCE_PATHS["test_source"]
        module.__package__ = ""
        module.__dict__.update({
            "json": json, "re": re, "hashlib": hashlib, "Mapping": Mapping,
            "defaultdict": defaultdict, "Path": snapshot.Path,
            "ROOT": snapshot.Path(repository_root),
            "PLAN": snapshot.Path(repository_root) / SOURCE_PATHS["plan"],
        })
        for role, nodes in (
            ("review_implementation", review), ("file_hash_source", file_hash),
            ("sample_helper_source", sample_helpers), ("test_source", test),
        ):
            _compile(nodes, repository_root + "/" + SOURCE_PATHS[role], module.__dict__)
        _require(module.EXPECTED_BINDINGS == PLAN_BINDING_ROLES)
        _require({name for name in vars(module) if name.startswith("test_")} == {TEST_NAME})
        return module, snapshot
    except Exception:
        raise ValueError(_ERROR) from None
