from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/dmath_verifier_worker.py"
SPEC = importlib.util.spec_from_file_location("dmath_verifier_worker", WORKER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_simple_arithmetic_program_is_verified() -> None:
    code = "a = 9\nb = 6\nc = a * b + 5\nprint(int(c))"
    assert MODULE.verify_record(code, "59") == (True, "verified")


def test_numeric_eval_is_replaced_with_safe_parser() -> None:
    code = 'value = "3/2"\nvalue = eval(value)\nprint(float(value))'
    assert MODULE.verify_record(code, "1.5") == (True, "verified")


@pytest.mark.parametrize(
    "code",
    [
        "import os\nprint(1)",
        "print((1).__class__)",
        'print(open("/etc/passwd").read())',
        'print("{x.__class__}".format_map({"x": 1}))',
    ],
)
def test_unsafe_programs_fail_closed(code: str) -> None:
    verified, _reason = MODULE.verify_record(code, "1")
    assert verified is False
