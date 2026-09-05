#!/usr/bin/env python3
"""Isolated, raw-text-free verifier worker for pinned DMath solution programs."""

from __future__ import annotations

import ast
import itertools
import json
import math
import resource
import signal
import sys
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any

MAX_AST_NODES = 12_000
MAX_SOURCE_CHARACTERS = 32_768
MAX_OUTPUT_CHARACTERS = 16_384
MAX_ITERABLE_ITEMS = 1_000_000
RECORD_TIMEOUT_SECONDS = 1.0

_FORBIDDEN_NODES = (
    ast.Assert,
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.FunctionDef,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Match,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)
_ALLOWED_NAME_CALLS = {
    "abs",
    "all",
    "dict",
    "enumerate",
    "eval",
    "float",
    "int",
    "len",
    "list",
    "map",
    "print",
    "range",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "zip",
}
_ALLOWED_METHODS = {
    "append",
    "copy",
    "format",
    "get",
    "index",
    "isdigit",
    "items",
    "join",
    "keys",
    "pop",
    "replace",
    "reverse",
    "sort",
    "split",
}
_ALLOWED_MODULE_CALLS = {
    "itertools": {"permutations", "product"},
    "math": {"gcd", "isclose", "sqrt"},
}
_ALLOWED_FORMAT_LITERALS = {"{:.2f}"}


class _RecordTimeout(Exception):
    pass


def _alarm_handler(_signum: int, _frame: Any) -> None:
    raise _RecordTimeout("record_timeout")


def _set_process_limits() -> None:
    limits = (
        (resource.RLIMIT_CPU, (600, 600)),
        (resource.RLIMIT_FSIZE, (1_048_576, 1_048_576)),
        (resource.RLIMIT_NOFILE, (32, 32)),
    )
    for name, value in limits:
        try:
            resource.setrlimit(name, value)
        except (OSError, ValueError):
            pass
    try:
        resource.setrlimit(resource.RLIMIT_AS, (1_073_741_824, 1_073_741_824))
    except (OSError, ValueError):
        pass


def _validate_identifier(value: str) -> None:
    if value.startswith("_") or "__" in value:
        raise ValueError("private_identifier")


def _validate_call(node: ast.Call) -> None:
    function = node.func
    if isinstance(function, ast.Name):
        if function.id not in _ALLOWED_NAME_CALLS:
            raise ValueError("call_not_allowed")
        return
    if not isinstance(function, ast.Attribute):
        raise ValueError("dynamic_call_not_allowed")
    _validate_identifier(function.attr)
    if isinstance(function.value, ast.Name) and function.value.id in _ALLOWED_MODULE_CALLS:
        if function.attr not in _ALLOWED_MODULE_CALLS[function.value.id]:
            raise ValueError("module_call_not_allowed")
        return
    if function.attr not in _ALLOWED_METHODS:
        raise ValueError("method_call_not_allowed")
    if function.attr in {"format", "join"}:
        if not isinstance(function.value, ast.Constant) or not isinstance(
            function.value.value, str
        ):
            raise ValueError("dynamic_string_method_not_allowed")
        if function.attr == "format" and function.value.value not in _ALLOWED_FORMAT_LITERALS:
            raise ValueError("format_literal_not_allowed")


def validate_solution_tree(tree: ast.AST) -> None:
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise ValueError("ast_node_limit")
    for node in nodes:
        if isinstance(node, _FORBIDDEN_NODES):
            raise ValueError(f"forbidden_ast_node:{type(node).__name__}")
        if isinstance(node, ast.Name):
            _validate_identifier(node.id)
        elif isinstance(node, ast.Attribute):
            _validate_identifier(node.attr)
        elif isinstance(node, ast.Call):
            _validate_call(node)
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (str, int, float, bool, type(None))):
                raise ValueError("constant_type_not_allowed")
            if isinstance(node.value, str) and len(node.value) > 10_000:
                raise ValueError("string_constant_limit")
            if isinstance(node.value, int) and abs(node.value) > 10**18:
                raise ValueError("integer_constant_limit")


def _bounded_range(*args: int) -> range:
    value = range(*args)
    if len(value) > MAX_ITERABLE_ITEMS:
        raise ValueError("range_limit")
    return value


def _bounded_permutations(values: Iterable[Any], length: int | None = None) -> Any:
    materialized = tuple(values)
    chosen = len(materialized) if length is None else int(length)
    if chosen < 0 or chosen > len(materialized):
        return iter(())
    if math.perm(len(materialized), chosen) > MAX_ITERABLE_ITEMS:
        raise ValueError("permutations_limit")
    return itertools.permutations(materialized, chosen)


def _bounded_product(*values: Iterable[Any], repeat: int = 1) -> Any:
    materialized = [tuple(value) for value in values]
    size = math.prod(len(value) for value in materialized) ** int(repeat)
    if size > MAX_ITERABLE_ITEMS:
        raise ValueError("product_limit")
    return itertools.product(*materialized, repeat=repeat)


def _numeric_eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _numeric_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _numeric_eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _numeric_eval(node.left)
        right = _numeric_eval(node.right)
        operations: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b,
            ast.Pow: lambda a, b: a**b,
        }
        operation = operations.get(type(node.op))
        if operation is None:
            raise ValueError("numeric_eval_operator")
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("numeric_eval_exponent")
        result = operation(left, right)
        if isinstance(result, (int, float)) and abs(result) > 10**100:
            raise ValueError("numeric_eval_magnitude")
        return result
    raise ValueError("numeric_eval_expression")


def _safe_eval(value: Any) -> int | float:
    text = str(value)
    if len(text) > 128:
        raise ValueError("numeric_eval_length")
    return _numeric_eval(ast.parse(text, mode="eval"))


def _normalize_answer(value: Any) -> str:
    return " ".join(str(value).strip().split())


def verify_record(code: str, expected_answer: str) -> tuple[bool, str]:
    if not isinstance(code, str) or len(code) > MAX_SOURCE_CHARACTERS:
        return False, "source_length"
    try:
        tree = ast.parse(code, mode="exec")
        validate_solution_tree(tree)
    except (SyntaxError, ValueError) as error:
        return False, str(error)

    output: list[str] = []

    def safe_print(*values: Any, sep: str = " ", end: str = "\n") -> None:
        if sep != " " or end != "\n":
            raise ValueError("print_options")
        rendered = sep.join(str(value) for value in values)
        if sum(len(item) for item in output) + len(rendered) > MAX_OUTPUT_CHARACTERS:
            raise ValueError("output_limit")
        output.append(rendered)

    safe_builtins = {
        "abs": abs,
        "all": all,
        "dict": dict,
        "enumerate": enumerate,
        "eval": _safe_eval,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "print": safe_print,
        "range": _bounded_range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "zip": zip,
    }
    environment = {
        "__builtins__": safe_builtins,
        "itertools": SimpleNamespace(
            permutations=_bounded_permutations,
            product=_bounded_product,
        ),
        "math": SimpleNamespace(
            gcd=math.gcd,
            isclose=math.isclose,
            sqrt=math.sqrt,
        ),
    }
    try:
        signal.setitimer(signal.ITIMER_REAL, RECORD_TIMEOUT_SECONDS)
        exec(compile(tree, "<dmath-solution>", "exec"), environment, environment)
    except _RecordTimeout:
        return False, "record_timeout"
    except Exception as error:
        return False, f"execution:{type(error).__name__}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    if len(output) != 1:
        return False, "print_count"
    if _normalize_answer(output[0]) != _normalize_answer(expected_answer):
        return False, "answer_mismatch"
    return True, "verified"


def main() -> int:
    _set_process_limits()
    signal.signal(signal.SIGALRM, _alarm_handler)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            record_id = row["record_id"]
            verified, reason = verify_record(row["solution_code_en"], row["answer_en"])
            result = {
                "reason": reason,
                "record_id": record_id,
                "verified": verified,
            }
        except Exception as error:
            result = {
                "reason": f"worker_input:{type(error).__name__}",
                "record_id": None,
                "verified": False,
            }
        sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
