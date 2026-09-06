"""Project one original Aya test onto authenticated immutable input bytes.

This is not a normal checkout import: only the selected original constants,
loader helpers, hash helper and test body are compiled. All original imports and
unselected statements/functions are excluded. The returned module is not added
to sys.modules. Its original test must be run with caller-owned output capture:
unchanged historical exceptions can contain paths. Construction failures here
are fixed and raw-free. No filesystem access or operational imports occur here.
"""

from __future__ import annotations
import __future__

import ast
import copy
import hashlib
import json
import posixpath
import re
import types
from collections.abc import Mapping
from pathlib import PurePosixPath

from .posttraining_safety_replay_snapshot_v1 import Snapshot

MODULE_NAME = "test_aya_bengali_repair_preflight"
TEST_NAME = "test_aya_repair_plan_is_exact_local_and_non_authorizing"
PLAN_BINDING_ROLES = frozenset(
    {
        "benchmark_registry",
        "candidate_source_registry",
        "chat_template",
        "data_contract",
        "filter_implementation",
        "format_implementation",
        "input_audit_report",
        "input_inventory",
        "missing_mix_proposal",
        "reconciliation_receipt",
        "repair_implementation",
        "repair_runner",
        "source_registry",
        "tokenizer_config",
        "tokenizer_json",
    }
)
INPUT_ROLES = PLAN_BINDING_ROLES | {"plan", "hash_helper_source", "test_source"}
AYA_CONSTANTS = frozenset(
    {
        "PLAN_SCHEMA_VERSION",
        "SOURCE_ID",
        "DATASET_ID",
        "SOURCE_REVISION",
        "SOURCE_LICENSE",
        "EXPECTED_STATUS",
        "EXPECTED_AUTHORIZATION",
        "EXPECTED_LIMITS",
        "EXPECTED_SOURCE",
        "EXPECTED_POLICY",
        "EXPECTED_OUTPUT",
        "EXPECTED_BINDING_NAMES",
        "_SHA256",
    }
)
AYA_FUNCTIONS = frozenset(
    {
        "_load_json",
        "_require_sha256",
        "resolve_workspace_binding",
        "load_aya_repair_plan",
    }
)
SOURCE_PATHS = {
    "plan": "configs/lumi-4b-sft-aya-bengali-repair-preflight-v0.1.json",
    "repair_implementation": "src/lumi/posttraining_aya_repair.py",
    "hash_helper_source": "src/lumi/posttraining_oasst1_full_run.py",
    "test_source": "tests/test_aya_bengali_repair_preflight.py",
}
MAXIMUM_TOTAL_BYTES = 32 * 1024 * 1024
MAXIMUM_SOURCE_BYTES = 1024 * 1024
_ERROR = "invalid Aya plan source projection"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError(_ERROR)


def _authenticate(payloads: dict, bindings: dict, repository_root: str) -> dict[str, bytes]:
    _require(type(repository_root) is str and repository_root.startswith("/"))
    _require(repository_root != "/" and not repository_root.startswith("//"))
    _require(posixpath.normpath(repository_root) == repository_root)
    _require(type(payloads) is dict and type(bindings) is dict)
    _require(set(payloads) == set(bindings) == INPUT_ROLES)
    workspace = PurePosixPath(repository_root).parent
    blobs = {}
    total = 0
    for role in sorted(INPUT_ROLES):
        binding, payload = bindings[role], payloads[role]
        _require(
            type(binding) is dict
            and set(binding)
            in (
                {"path", "bytes", "sha256"},
                {"path", "bytes", "sha256", "git_revision"},
            )
        )
        path = binding["path"]
        _require(type(path) is str and bool(path) and len(path) <= 4096)
        _require(not path.startswith("/") and "\\" not in path)
        _require(all(ord(char) >= 32 and ord(char) != 127 for char in path))
        absolute = posixpath.normpath(repository_root + "/" + path)
        _require(PurePosixPath(absolute).is_relative_to(workspace))
        _require(absolute not in blobs)
        if role in SOURCE_PATHS:
            _require(path == SOURCE_PATHS[role])
        if "git_revision" in binding:
            revision = binding["git_revision"]
            _require(type(revision) is str and re.fullmatch(r"[0-9a-f]{40}", revision) is not None)
        _require(type(binding["bytes"]) is int and binding["bytes"] > 0)
        _require(type(payload) is bytes and len(payload) == binding["bytes"])
        _require(
            type(binding["sha256"]) is str and _SHA256.fullmatch(binding["sha256"]) is not None
        )
        total += len(payload)
        _require(total <= MAXIMUM_TOTAL_BYTES)
        _require(hashlib.sha256(payload).hexdigest() == binding["sha256"])
        if role in SOURCE_PATHS:
            _require(len(payload) <= MAXIMUM_SOURCE_BYTES)
        blobs[absolute] = payload
    return blobs


def _constant_expression(node: ast.AST) -> bool:
    """Permit literal constants plus the two constructors used by Aya itself."""
    if isinstance(node, ast.Constant):
        return type(node.value) in {str, int, float, bool, type(None)}
    if isinstance(node, ast.Name):
        return node.id in AYA_CONSTANTS
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_constant_expression(value) for value in node.elts)
    if isinstance(node, ast.Dict):
        return all(key is not None and _constant_expression(key) for key in node.keys) and all(
            _constant_expression(value) for value in node.values
        )
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        if isinstance(node.func, ast.Name) and node.func.id == "frozenset":
            return isinstance(node.args[0], ast.Set) and _constant_expression(node.args[0])
        return (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "re"
            and node.func.attr == "compile"
            and isinstance(node.args[0], ast.Constant)
            and type(node.args[0].value) is str
        )
    return False


def _select(source: bytes, constants: frozenset[str], functions: frozenset[str]) -> list[ast.stmt]:
    selected = []
    found = set()
    for node in ast.parse(source).body:
        name = None
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if set(names) & constants:
                _require(len(node.targets) == 1 and len(names) == 1)
                name = names[0]
                _require(name in constants and _constant_expression(node.value))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in functions:
            name = node.name
            _require(isinstance(node, ast.FunctionDef) and not node.decorator_list)
            _require(
                not node.args.defaults and all(value is None for value in node.args.kw_defaults)
            )
            _require(not getattr(node, "type_params", []))
        if name is not None:
            _require(name not in found)
            found.add(name)
            selected.append(copy.deepcopy(node))
    _require(found == constants | functions)
    return selected


def _compile(nodes: list[ast.stmt], filename: str, namespace: dict) -> None:
    tree = ast.Module(body=nodes, type_ignores=[])
    code = compile(
        ast.fix_missing_locations(tree),
        filename,
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, namespace)


def build_test_module(payloads: dict, bindings: dict, repository_root: str):
    """Return an unregistered original-test projection and its immutable snapshot.

    Caller bindings authenticate all 18 roles before any source parsing. The
    caller must authenticate that binding manifest independently. Inputs are
    memory-only; embedded inventory paths are never traversed. A preexisting
    sys.modules entry is neither overwritten nor removed by this function.
    """
    try:
        blobs = _authenticate(payloads, bindings, repository_root)
        aya = _select(payloads["repair_implementation"], AYA_CONSTANTS, AYA_FUNCTIONS)
        helper = _select(payloads["hash_helper_source"], frozenset(), frozenset({"file_sha256"}))
        test = _select(payloads["test_source"], frozenset(), frozenset({TEST_NAME}))
        snapshot = Snapshot(blobs, repository_root=repository_root)
        module = types.ModuleType(MODULE_NAME)
        module.__file__ = repository_root + "/" + SOURCE_PATHS["test_source"]
        module.__package__ = ""
        module.__dict__.update(
            {
                "hashlib": hashlib,
                "json": json,
                "re": re,
                "Mapping": Mapping,
                "Path": snapshot.Path,
                "ROOT": snapshot.Path(repository_root),
                "PLAN": snapshot.Path(repository_root) / SOURCE_PATHS["plan"],
            }
        )
        _compile(
            aya, repository_root + "/" + SOURCE_PATHS["repair_implementation"], module.__dict__
        )
        _compile(
            helper, repository_root + "/" + SOURCE_PATHS["hash_helper_source"], module.__dict__
        )
        _compile(test, module.__file__, module.__dict__)
        _require(module.EXPECTED_BINDING_NAMES == PLAN_BINDING_ROLES)
        _require({name for name in module.__dict__ if name.startswith("test_")} == {TEST_NAME})
        return module, snapshot
    except Exception:
        raise ValueError(_ERROR) from None
