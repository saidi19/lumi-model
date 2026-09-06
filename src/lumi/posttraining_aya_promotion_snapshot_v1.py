"""Read-only projection of two unchanged Aya successor/receipt comparisons.

All original inputs are caller-authenticated immutable bytes. Historical registry
declarations execute only in an isolated namespace with a snapshot-backed Path;
no original module imports, publisher, registry writes or model stack is used.
The caller must capture original exceptions and expose only bounded aggregates.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import posixpath
import re
import sys
import types
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from .posttraining_aya_plan_snapshot_v1 import _compile
from .posttraining_safety_replay_snapshot_v1 import Snapshot

MODULE_NAME = "test_aya_source_promotion"
TEST_NAMES = (
    "test_successor_changes_only_the_existing_aya_entry",
    "test_promotion_receipt_is_reproducible_raw_free_and_nonproduction",
)
CLASS_MODULE_NAMES = (
    "_lumi_aya_promotion_projection_policy",
    "_lumi_aya_promotion_projection_registry",
)
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
    "policy_source": "src/lumi/posttraining_data.py",
    "successor_registry": "configs/lumi-4b-sft-source-registry-v0.4.json",
    "promotion_receipt": "evidence/lumi-4b-sft-aya-source-promotion-v0.1.json",
}
INPUT_ROLES = frozenset(SOURCE_PATHS)
CODE_ROLES = frozenset({
    "promotion_implementation", "promotion_runner", "source_registry_implementation",
    "hash_helper_source", "test_source", "policy_source",
})
POLICY_CONSTANTS = frozenset({
    "SFT_SCHEMA_VERSION", "_SHA256", "_LEGACY_SOURCE_REGISTRY_BINDING",
    "_PRIVATE_RESEARCH_SOURCE_REGISTRY_BINDING", "_PRIVATE_RESEARCH_INCLUDED_SOURCE_IDS",
    "_PRIVATE_RESEARCH_EXCLUDED_SOURCE_IDS", "_PRIVATE_RESEARCH_AI_SOURCE_CONTRACTS",
    "_PRIVATE_RESEARCH_MIX_BINDINGS", "_PRIVATE_RESEARCH_EVALUATION_SPLIT_PROHIBITIONS",
    "_PRIVATE_RESEARCH_BENCHMARK_CONTRACT", "_PRIVATE_RESEARCH_PRESERVED_POLICY_SECTIONS",
    "_PRIVATE_RESEARCH_NOTES", "_LEARNING_CANDIDATE_KEYS",
})
POLICY_FUNCTIONS = frozenset({
    "_valid_sha256", "_string_set", "_strict_json_equal", "_validate_learning_candidate",
    "load_sft_policy",
})
REGISTRY_CONSTANTS = frozenset({
    "SOURCE_REGISTRY_SCHEMA_VERSION", "SOURCE_STATUSES", "SOURCE_ACCESS", "SOURCE_ORIGINS",
    "_GIT_SHA", "_SHA256", "_SOURCE_ID",
})
REGISTRY_FUNCTIONS = frozenset({
    "_canonical_json", "_sha256", "_file_sha256", "_required_string", "_string_tuple",
    "load_sft_source_registry", "build_sft_source_readiness_report",
})
PROMOTION_CONSTANTS = frozenset({
    "PLAN_SCHEMA_VERSION", "RECEIPT_SCHEMA_VERSION", "EXPECTED_AUTHORIZATION", "EXPECTED_LIMITS",
    "EXPECTED_OUTPUT", "EXPECTED_CURATED_METRICS", "EXPECTED_REQUIRED_TASKS",
    "EXPECTED_SOURCE_BEFORE", "EXPECTED_SOURCE_AFTER", "EXPECTED_BINDINGS", "_SHA256",
})
PROMOTION_FUNCTIONS = frozenset({
    "sha256_value", "json_file_bytes", "_load_json", "_require_sha256", "_verify_internal_seal",
    "resolve_promotion_binding", "load_source_promotion_plan",
    "verify_source_promotion_prerequisites", "load_policy_and_predecessor",
    "build_successor_registry", "validate_successor_registry", "build_source_promotion_receipt",
    "verify_source_promotion_receipt",
})
MAXIMUM_TOTAL_BYTES = 4 * 1024 * 1024
MAXIMUM_METADATA_BYTES = 2 * 1024 * 1024
MAXIMUM_SOURCE_BYTES = 65_536
MAXIMUM_VIRTUAL_OPENS = 35
_ERROR = "invalid Aya promotion source projection"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CLASS_METHODS = {
    "SFTPolicy": {"from_mapping": "classmethod"},
    "SFTSource": {"from_mapping": "classmethod"},
    "SFTSourceRegistry": {"by_id": "property", "record_reasons": None},
}


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError(_ERROR)


def _constant_expression(node: ast.AST, constants: frozenset[str]) -> bool:
    if isinstance(node, ast.Constant):
        return type(node.value) in {str, int, float, bool, type(None)}
    if isinstance(node, ast.Name):
        return node.id in constants
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_constant_expression(item, constants) for item in node.elts)
    if isinstance(node, ast.Dict):
        return (
            all(key is not None and _constant_expression(key, constants) for key in node.keys)
            and all(_constant_expression(value, constants) for value in node.values)
        )
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        if isinstance(node.func, ast.Name) and node.func.id == "frozenset":
            return (
                isinstance(node.args[0], ast.Set) and _constant_expression(node.args[0], constants)
            )
        return (
            isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "re" and node.func.attr == "compile"
            and isinstance(node.args[0], ast.Constant) and type(node.args[0].value) is str
        )
    return False


def _check_function(node: ast.FunctionDef, decorator: str | None = None, *, method=False) -> None:
    _require(not getattr(node, "type_params", []) and not node.args.defaults)
    if decorator is None:
        _require(not node.decorator_list)
    else:
        _require(
            len(node.decorator_list) == 1 and isinstance(node.decorator_list[0], ast.Name)
            and node.decorator_list[0].id == decorator
        )
    permitted = {} if method else {
        "_string_tuple": {"allow_empty": False},
        "load_sft_source_registry": {"expected_sha256": None},
    }.get(node.name, {})
    observed = {}
    for arg, value in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
        if value is not None:
            _require(arg.arg in permitted and isinstance(value, ast.Constant))
            _require(value.value is permitted[arg.arg])
            observed[arg.arg] = value.value
    _require(set(observed) == set(permitted))


def _check_class(node: ast.ClassDef) -> None:
    _require(node.name in _CLASS_METHODS)
    _require(not node.bases and not node.keywords and not getattr(node, "type_params", []))
    _require(len(node.decorator_list) == 1)
    decorator = node.decorator_list[0]
    _require(
        isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name)
        and decorator.func.id == "dataclass" and not decorator.args
        and len(decorator.keywords) == 1 and decorator.keywords[0].arg == "frozen"
        and isinstance(decorator.keywords[0].value, ast.Constant)
        and decorator.keywords[0].value.value is True
    )
    members, methods = set(), set()
    for item in node.body:
        if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
            _require(type(item.value.value) is str)
            continue
        if isinstance(item, ast.AnnAssign):
            _require(isinstance(item.target, ast.Name) and item.simple == 1)
            name = item.target.id
            _require(item.value is None or (
                name == "learning_candidate" and node.name == "SFTPolicy"
                and isinstance(item.value, ast.Constant) and item.value.value is None
            ))
        elif isinstance(item, ast.FunctionDef):
            name = item.name
            _require(name in _CLASS_METHODS[node.name])
            _check_function(item, _CLASS_METHODS[node.name][name], method=True)
            methods.add(name)
        else:
            _require(False)
        _require(name not in members)
        members.add(name)
    _require(methods == set(_CLASS_METHODS[node.name]))


def _select(source: bytes, constants: frozenset[str], functions: frozenset[str],
            classes: frozenset[str] = frozenset()) -> list[ast.stmt]:
    selected, found = [], set()
    for node in ast.parse(source).body:
        name = None
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if set(names) & constants:
                _require(len(node.targets) == 1 and len(names) == 1)
                name = names[0]
                _require(name in constants and _constant_expression(node.value, constants))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in functions:
            _require(isinstance(node, ast.FunctionDef))
            _check_function(node)
            name = node.name
        elif isinstance(node, ast.ClassDef) and node.name in classes:
            _check_class(node)
            name = node.name
        if name is not None:
            _require(name not in found)
            found.add(name)
            selected.append(copy.deepcopy(node))
    _require(found == constants | functions | classes)
    return selected


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


def _module(name: str, role: str, root: str, values: dict) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__, module.__package__ = root + "/" + SOURCE_PATHS[role], ""
    module.__dict__.update(values)
    return module


def _class_modules(nodes: dict, snapshot: Snapshot, root: str) -> tuple:
    _require(not any(
        name == expected or name.startswith(expected + ".")
        for name in sys.modules for expected in CLASS_MODULE_NAMES
    ))
    modules = (
        _module(CLASS_MODULE_NAMES[0], "policy_source", root, {
            "json": json, "re": re, "Mapping": Mapping, "dataclass": dataclass,
            "Path": snapshot.Path,
        }),
        _module(CLASS_MODULE_NAMES[1], "source_registry_implementation", root, {
            "json": json, "re": re, "Mapping": Mapping, "dataclass": dataclass,
            "Path": snapshot.Path, "hashlib": hashlib, "Counter": Counter,
        }),
    )
    installed = {}
    clean = True
    try:
        for module, role in zip(modules, ("policy_source", "source_registry_implementation"),
                                strict=True):
            installed[module.__name__] = module
            sys.modules[module.__name__] = module
            _compile(nodes[role], module.__file__, module.__dict__)
    finally:
        for name, module in reversed(tuple(installed.items())):
            if sys.modules.get(name) is module:
                del sys.modules[name]
            else:
                clean = False
    _require(clean)
    return modules


def build_test_module(payloads: dict, bindings: dict, repository_root: str):
    """Return an unregistered two-test module and an eleven-blob, 35-open view."""
    try:
        blobs, checked = _authenticate(payloads, bindings, repository_root)
        _check_plan(blobs["plan"], checked)
        nodes = {
            "policy_source": _select(
                blobs["policy_source"], POLICY_CONSTANTS, POLICY_FUNCTIONS,
                frozenset({"SFTPolicy"}),
            ),
            "source_registry_implementation": _select(
                blobs["source_registry_implementation"], REGISTRY_CONSTANTS, REGISTRY_FUNCTIONS,
                frozenset({"SFTSource", "SFTSourceRegistry"}),
            ),
            "hash_helper_source": _select(
                blobs["hash_helper_source"], frozenset(),
                frozenset({"canonical_json_bytes", "file_sha256"}),
            ),
            "promotion_implementation": _select(
                blobs["promotion_implementation"], PROMOTION_CONSTANTS, PROMOTION_FUNCTIONS,
            ),
            "test_source": _select(
                blobs["test_source"], frozenset(),
                frozenset({"_load", "_prerequisites", *TEST_NAMES}),
            ),
        }
        view_roles = PLAN_BINDING_ROLES | {"plan", "successor_registry", "promotion_receipt"}
        logical_bytes = (
            3 * checked["plan"]["bytes"]
            + 2 * sum(checked[role]["bytes"] for role in PLAN_BINDING_ROLES)
            + 2 * checked["data_contract"]["bytes"]
            + 7 * checked["predecessor_registry"]["bytes"]
            + 3 * checked["successor_registry"]["bytes"]
            + checked["promotion_receipt"]["bytes"]
            + sum(checked[role]["bytes"] for role in (
                "attribution_manifest", "materialization_report", "owner_review_receipt",
            ))
        )
        snapshot = Snapshot(
            {
                posixpath.normpath(repository_root + "/" + SOURCE_PATHS[role]): blobs[role]
                for role in view_roles
            },
            repository_root=repository_root, maximum_virtual_opens=MAXIMUM_VIRTUAL_OPENS,
            maximum_logical_read_bytes=logical_bytes, maximum_gzip_bytes=1,
        )
        policy, registry = _class_modules(nodes, snapshot, repository_root)
        helpers = _module("_lumi_aya_promotion_projection_hash_helpers", "hash_helper_source",
                          repository_root, {
                              "json": json, "hashlib": hashlib, "Path": snapshot.Path,
                          })
        _compile(nodes["hash_helper_source"], helpers.__file__, helpers.__dict__)
        promotion = _module("_lumi_aya_promotion_projection_promotion", "promotion_implementation",
                            repository_root, {
                                "copy": copy, "hashlib": hashlib, "json": json, "re": re,
                                "Mapping": Mapping, "Path": snapshot.Path,
                                "SFTPolicy": policy.SFTPolicy,
                                "load_sft_policy": policy.load_sft_policy,
                                "SFTSource": registry.SFTSource,
                                "SFTSourceRegistry": registry.SFTSourceRegistry,
                                "load_sft_source_registry": registry.load_sft_source_registry,
                                "build_sft_source_readiness_report": (
                                    registry.build_sft_source_readiness_report
                                ),
                                "canonical_json_bytes": helpers.canonical_json_bytes,
                                "file_sha256": helpers.file_sha256,
                            })
        _compile(nodes["promotion_implementation"], promotion.__file__, promotion.__dict__)
        _require(promotion.EXPECTED_BINDINGS == PLAN_BINDING_ROLES)
        module = _module(MODULE_NAME, "test_source", repository_root, {
            "hashlib": hashlib, "json": json, "Path": snapshot.Path,
            "ROOT": snapshot.Path(repository_root),
            "PLAN": snapshot.Path(repository_root) / SOURCE_PATHS["plan"],
            "PREDECESSOR": snapshot.Path(repository_root) / SOURCE_PATHS["predecessor_registry"],
            "SUCCESSOR": snapshot.Path(repository_root) / SOURCE_PATHS["successor_registry"],
            "RECEIPT": snapshot.Path(repository_root) / SOURCE_PATHS["promotion_receipt"],
            "file_sha256": helpers.file_sha256,
            "EXPECTED_SOURCE_AFTER": promotion.EXPECTED_SOURCE_AFTER,
            **{name: getattr(promotion, name) for name in (
                "build_source_promotion_receipt", "build_successor_registry", "json_file_bytes",
                "load_policy_and_predecessor", "load_source_promotion_plan",
                "resolve_promotion_binding", "validate_successor_registry",
                "verify_source_promotion_prerequisites", "verify_source_promotion_receipt",
            )},
        })
        _compile(nodes["test_source"], module.__file__, module.__dict__)
        _require({name for name in vars(module) if name.startswith("test_")} == set(TEST_NAMES))
        return module, snapshot
    except Exception:
        raise ValueError(_ERROR) from None
