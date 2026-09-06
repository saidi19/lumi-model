"""Project one unchanged Aya prerequisite test onto authenticated input bytes.

The caller authenticates the input manifest and separately authorizes physical
reads, including any exact historical source-copy retrieval. Bindings here use
only original logical paths. No source-registry implementation or promotion
runner is imported or compiled. Historical authorization fields are verified as
data, not executed. Caller-owned capture must suppress historical exceptions and
private attribution or receipt content; this module returns no aggregate claim.
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

MODULE_NAME = "test_aya_source_promotion"
TEST_NAME = "test_exact_reviewed_aya_prerequisites_pass"
PLAN_BINDING_ROLES = frozenset({
    "attribution_manifest", "data_contract", "materialization_report",
    "owner_review_receipt", "predecessor_registry", "promotion_implementation",
    "promotion_runner", "source_registry_implementation",
})
SOURCE_PATHS = {
    "plan": "configs/lumi-4b-sft-aya-source-promotion-v0.1.json",
    "attribution_manifest": (
        "../lumi-artifacts/4b-sft-aya-bengali-materialization-v0.1/"
        "6d7c20b19f6b/attempt-001/attribution-manifest.jsonl"
    ),
    "data_contract": "configs/lumi-4b-sft-data-contract-v0.1.json",
    "materialization_report": "evidence/lumi-4b-sft-aya-bengali-materialization-v0.1.json",
    "owner_review_receipt": "evidence/lumi-4b-sft-aya-bengali-owner-review-v0.1.json",
    "predecessor_registry": "configs/lumi-4b-sft-source-registry-v0.3.json",
    "promotion_implementation": "src/lumi/posttraining_aya_source_promotion.py",
    "promotion_runner": "scripts/promote_aya_source.py",
    "source_registry_implementation": "src/lumi/posttraining_sources.py",
    "hash_helper_source": "src/lumi/posttraining_first_party.py",
    "test_source": "tests/test_aya_source_promotion.py",
}
INPUT_ROLES = frozenset(SOURCE_PATHS)
CODE_ROLES = frozenset({
    "promotion_implementation", "promotion_runner", "source_registry_implementation",
    "hash_helper_source", "test_source",
})
AYA_CONSTANTS = frozenset({
    "PLAN_SCHEMA_VERSION", "EXPECTED_AUTHORIZATION", "EXPECTED_LIMITS", "EXPECTED_OUTPUT",
    "EXPECTED_CURATED_METRICS", "EXPECTED_REQUIRED_TASKS", "EXPECTED_SOURCE_BEFORE",
    "EXPECTED_SOURCE_AFTER", "EXPECTED_BINDINGS", "_SHA256",
})
AYA_FUNCTIONS = frozenset({
    "sha256_value", "_load_json", "_require_sha256", "_verify_internal_seal",
    "resolve_promotion_binding", "load_source_promotion_plan",
    "verify_source_promotion_prerequisites",
})
MAXIMUM_TOTAL_BYTES = 4 * 1024 * 1024
MAXIMUM_METADATA_BYTES = 2 * 1024 * 1024
MAXIMUM_SOURCE_BYTES = 65_536
MAXIMUM_VIRTUAL_OPENS = 12
_ERROR = "invalid Aya prerequisite source projection"
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
        maximum = MAXIMUM_SOURCE_BYTES if role in CODE_ROLES else MAXIMUM_METADATA_BYTES
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
    """Match the input plan only; private prerequisites remain opaque here."""
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
    """Return an unregistered original-test module and nine-blob snapshot.

    All eleven inputs authenticate before any AST or JSON parsing. Construction
    uses zero virtual opens. The original test consumes twelve: plan, eight
    binding hashes, attribution JSONL, materialization report and owner receipt.
    Exact historical source-registry bytes are hash-only at their original path;
    physical-copy paths and approval to retrieve them belong to the caller.
    """
    try:
        blobs, checked = _authenticate(payloads, bindings, repository_root)
        _check_plan(blobs["plan"], checked)
        promotion = _select(blobs["promotion_implementation"], AYA_CONSTANTS, AYA_FUNCTIONS)
        helpers = _select(
            blobs["hash_helper_source"], frozenset(),
            frozenset({"canonical_json_bytes", "file_sha256"}),
        )
        test = _select(
            blobs["test_source"], frozenset(), frozenset({"_load", "_prerequisites", TEST_NAME}),
        )
        view_roles = PLAN_BINDING_ROLES | {"plan"}
        logical_bytes = (
            sum(checked[role]["bytes"] for role in view_roles)
            + sum(checked[role]["bytes"] for role in (
                "attribution_manifest", "materialization_report", "owner_review_receipt",
            ))
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
            "Path": snapshot.Path, "ROOT": snapshot.Path(repository_root),
            "PLAN": snapshot.Path(repository_root) / SOURCE_PATHS["plan"],
        })
        for role, nodes in (
            ("promotion_implementation", promotion), ("hash_helper_source", helpers),
            ("test_source", test),
        ):
            _compile(nodes, repository_root + "/" + SOURCE_PATHS[role], module.__dict__)
        _require(module.EXPECTED_BINDINGS == PLAN_BINDING_ROLES)
        _require({name for name in vars(module) if name.startswith("test_")} == {TEST_NAME})
        return module, snapshot
    except Exception:
        raise ValueError(_ERROR) from None
