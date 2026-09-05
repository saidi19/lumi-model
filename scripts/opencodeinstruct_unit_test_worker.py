#!/usr/bin/env python3
"""Isolated, raw-text-free verifier for OpenCodeInstruct Python solutions."""

from __future__ import annotations

import ast
import builtins
import collections
import copy
import dataclasses
import decimal
import enum
import fractions
import functools
import hashlib
import heapq
import itertools
import json
import math
import operator
import random
import re
import resource
import signal
import statistics
import string
import sys
import types
import typing
import unicodedata
from typing import Any

MAX_AST_NODES = 50_000
MAX_CODE_CHARACTERS = 24_000
MAX_TEST_CHARACTERS = 8_000
MAX_TOTAL_TEST_CHARACTERS = 32_000
MAX_STRING_CONSTANT_CHARACTERS = 16_384
MAX_RANGE_ITEMS = 2_000_000
RECORD_TIMEOUT_SECONDS = 0.75

_ALLOWED_MODULES = {
    "__future__": __import__("__future__"),
    "collections": collections,
    "copy": copy,
    "dataclasses": dataclasses,
    "decimal": decimal,
    "enum": enum,
    "fractions": fractions,
    "functools": functools,
    "hashlib": hashlib,
    "heapq": heapq,
    "itertools": itertools,
    "json": json,
    "math": math,
    "operator": operator,
    "random": random,
    "re": re,
    "statistics": statistics,
    "string": string,
    "types": types,
    "typing": typing,
    "unicodedata": unicodedata,
}

_BLOCKED_CALL_NAMES = {
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "quit",
    "setattr",
    "vars",
    "__import__",
}
_BLOCKED_ATTRIBUTES = {
    "bind",
    "chmod",
    "chown",
    "connect",
    "execv",
    "fork",
    "kill",
    "listen",
    "makedirs",
    "mkdir",
    "open",
    "popen",
    "recv",
    "remove",
    "rename",
    "rmdir",
    "send",
    "spawn",
    "system",
    "unlink",
    "walk",
    "write_text",
    "write_bytes",
}
_FORBIDDEN_AST_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.Global,
    ast.Nonlocal,
)


class _RecordTimeout(Exception):
    pass


def _alarm_handler(_signum: int, _frame: Any) -> None:
    raise _RecordTimeout("record_timeout")


def _set_process_limits() -> None:
    limits = (
        (resource.RLIMIT_CPU, (1_200, 1_200)),
        (resource.RLIMIT_FSIZE, (0, 0)),
        (resource.RLIMIT_NOFILE, (32, 32)),
    )
    for name, value in limits:
        try:
            resource.setrlimit(name, value)
        except (OSError, ValueError):
            pass
    try:
        resource.setrlimit(resource.RLIMIT_AS, (2_147_483_648, 2_147_483_648))
    except (OSError, ValueError):
        pass


def _audit_hook(event: str, _args: tuple[Any, ...]) -> None:
    blocked_prefixes = (
        "ctypes.",
        "os.system",
        "os.spawn",
        "socket.",
        "subprocess.",
    )
    if event == "open" or event.startswith(blocked_prefixes):
        raise PermissionError("sandbox_audit_block")


def _validate_identifier(value: str) -> None:
    if "__" in value:
        raise ValueError("dunder_identifier")


def _validate_tree(tree: ast.AST) -> None:
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise ValueError("ast_node_limit")
    for node in nodes:
        if isinstance(node, _FORBIDDEN_AST_NODES):
            raise ValueError(f"forbidden_ast_node:{type(node).__name__}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots = (
                [alias.name.split(".", 1)[0] for alias in node.names]
                if isinstance(node, ast.Import)
                else [(node.module or "").split(".", 1)[0]]
            )
            if any(root not in _ALLOWED_MODULES for root in roots):
                raise ValueError("forbidden_import")
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError("relative_import")
        elif isinstance(node, ast.Name):
            _validate_identifier(node.id)
        elif isinstance(node, ast.Attribute):
            _validate_identifier(node.attr)
            if node.attr.casefold() in _BLOCKED_ATTRIBUTES:
                raise ValueError("blocked_attribute")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALL_NAMES:
                raise ValueError("blocked_call")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr.casefold() in _BLOCKED_ATTRIBUTES
            ):
                raise ValueError("blocked_attribute_call")
        elif isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, str) and len(value) > MAX_STRING_CONSTANT_CHARACTERS:
                raise ValueError("string_constant_limit")
            if isinstance(value, int) and abs(value) > 10**100:
                raise ValueError("integer_constant_limit")


def _bounded_range(*args: int) -> range:
    value = range(*args)
    if len(value) > MAX_RANGE_ITEMS:
        raise ValueError("range_limit")
    return value


def _safe_import(
    name: str,
    _globals: Any = None,
    _locals: Any = None,
    fromlist: tuple[str, ...] | list[str] = (),
    level: int = 0,
) -> Any:
    if level:
        raise ImportError("relative_import")
    root = name.split(".", 1)[0]
    module = _ALLOWED_MODULES.get(root)
    if module is None:
        raise ImportError("forbidden_import")
    if name != root:
        current: Any = module
        for part in name.split(".")[1:]:
            if "__" in part or not hasattr(current, part):
                raise ImportError("forbidden_submodule")
            current = getattr(current, part)
        return current if fromlist else module
    return module


def _safe_builtins() -> dict[str, Any]:
    names = {
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "BaseException",
        "Exception",
        "IndexError",
        "KeyError",
        "LookupError",
        "NotImplementedError",
        "OverflowError",
        "RuntimeError",
        "StopIteration",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
        "__build_class__",
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytes",
        "callable",
        "chr",
        "classmethod",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "oct",
        "ord",
        "pow",
        "print",
        "property",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "zip",
    }
    safe = {name: getattr(builtins, name) for name in names}
    safe["range"] = _bounded_range
    safe["__import__"] = _safe_import
    safe["print"] = lambda *args, **kwargs: None
    return safe


def verify_record(code: Any, tests: Any) -> tuple[bool, str]:
    if not isinstance(code, str) or not 1 <= len(code) <= MAX_CODE_CHARACTERS:
        return False, "code_length"
    if (
        not isinstance(tests, list)
        or len(tests) != 10
        or any(not isinstance(test, str) or not test.strip() for test in tests)
    ):
        return False, "tests_schema"
    if any(len(test) > MAX_TEST_CHARACTERS for test in tests):
        return False, "test_length"
    if sum(map(len, tests)) > MAX_TOTAL_TEST_CHARACTERS:
        return False, "total_test_length"
    try:
        code_tree = ast.parse(code, mode="exec")
        _validate_tree(code_tree)
        test_trees = [ast.parse(test, mode="exec") for test in tests]
        for tree in test_trees:
            _validate_tree(tree)
    except (SyntaxError, ValueError) as error:
        return False, str(error).split(":", 1)[0]

    random.seed(0)
    environment = {
        "__builtins__": _safe_builtins(),
        "__name__": "__lumi_candidate__",
    }
    try:
        signal.setitimer(signal.ITIMER_REAL, RECORD_TIMEOUT_SECONDS)
        exec(compile(code_tree, "<candidate>", "exec"), environment, environment)
        for test_tree in test_trees:
            exec(compile(test_tree, "<published-test>", "exec"), environment, environment)
    except _RecordTimeout:
        return False, "record_timeout"
    except AssertionError:
        return False, "assertion_failed"
    except Exception as error:
        return False, f"execution_{type(error).__name__}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return True, "verified"


def main() -> int:
    _set_process_limits()
    signal.signal(signal.SIGALRM, _alarm_handler)
    sys.addaudithook(_audit_hook)
    for line in sys.stdin:
        if not line.strip():
            continue
        record_id: Any = None
        code_sha256: str | None = None
        tests_sha256: str | None = None
        try:
            row = json.loads(line)
            record_id = row.get("record_id")
            code = row.get("code")
            tests = row.get("unit_tests")
            code_sha256 = (
                hashlib.sha256(code.encode("utf-8")).hexdigest()
                if isinstance(code, str)
                else None
            )
            tests_sha256 = hashlib.sha256(
                json.dumps(
                    tests,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            verified, reason = verify_record(code, tests)
        except Exception as error:
            verified, reason = False, f"worker_input_{type(error).__name__}"
        result = {
            "code_sha256": code_sha256,
            "reason": reason,
            "record_id": record_id,
            "tests_sha256": tests_sha256,
            "verified": verified,
        }
        sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
