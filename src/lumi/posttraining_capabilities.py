"""Deterministic basic-capability gates for the future Lumi SFT checkpoint."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASIC_CAPABILITY_SCHEMA_VERSION = "lumi.basic-capability-gate.v0.1"
EXPECTED_AUTHORIZATION = {
    "model_inference_authorized": False,
    "paid_gpu_authorized": False,
    "production_release_authorized": False,
    "purpose": "post_sft_basic_capability_gate_definition_only",
    "raw_prompt_rows_in_git": False,
}
_INTEGER_TOKEN = re.compile(r"(?<!\w)-?(?:\d{1,3}(?:,\d{3})+|\d+)(?!\w)")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class ArithmeticCase:
    case_id: str
    expected: int
    left: int
    operation: str
    right: int


def load_basic_capability_plan(path: str | Path) -> dict[str, Any]:
    """Validate the metadata-only post-SFT basic capability gate."""

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("basic capability plan must be an object")
    if value.get("schema_version") != BASIC_CAPABILITY_SCHEMA_VERSION:
        raise ValueError("unsupported basic capability gate schema")
    if value.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("basic capability authorization boundary changed")
    if value.get("status") != "gate_defined_not_executed":
        raise ValueError("basic capability gate status changed")

    arithmetic = value.get("arithmetic")
    if not isinstance(arithmetic, Mapping):
        raise ValueError("basic capability arithmetic plan is missing")
    seed = arithmetic.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("basic capability arithmetic seed is invalid")
    operations = arithmetic.get("operations")
    expected_operations = {"addition", "exact_division", "multiplication", "subtraction"}
    if not isinstance(operations, Mapping) or set(operations) != expected_operations:
        raise ValueError("basic capability arithmetic operation set changed")
    total = 0
    for name, config in operations.items():
        if not isinstance(config, Mapping):
            raise ValueError(f"basic capability operation is invalid: {name}")
        total += _positive_int(config.get("cases"), f"{name}.cases")
    if total != arithmetic.get("total_cases") or total != 256:
        raise ValueError("basic capability arithmetic case bound changed")
    thresholds = arithmetic.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("basic capability thresholds are missing")
    if thresholds.get("calculator_backed_required_pass_rate") != 1.0:
        raise ValueError("calculator-backed arithmetic must require a perfect pass rate")

    suites = value.get("interaction_suites")
    if not isinstance(suites, list) or {suite.get("suite_id") for suite in suites} != {
        "greeting_and_small_talk",
        "typo_recovery",
    }:
        raise ValueError("basic interaction suite coverage changed")
    if value.get("reporting") != {
        "raw_model_responses_retained": False,
        "response_sha256_required": True,
        "results_bound_to_exact_sft_checkpoint": True,
    }:
        raise ValueError("basic capability reporting boundary changed")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _bounded_int(config: Mapping[str, Any], field: str) -> int:
    value = config.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def generate_arithmetic_cases(plan: Mapping[str, Any]) -> list[ArithmeticCase]:
    """Generate exact, deterministic integer arithmetic without storing prompt rows."""

    arithmetic = plan["arithmetic"]
    generator = random.Random(arithmetic["seed"])
    result: list[ArithmeticCase] = []
    for operation, config in arithmetic["operations"].items():
        target = config["cases"]
        seen: set[tuple[int, int]] = set()
        while len(seen) < target:
            if operation == "exact_division":
                divisor = generator.randint(
                    _bounded_int(config, "minimum_divisor"),
                    _bounded_int(config, "maximum_divisor"),
                )
                quotient = generator.randint(
                    _bounded_int(config, "minimum_quotient"),
                    _bounded_int(config, "maximum_quotient"),
                )
                left, right, expected = divisor * quotient, divisor, quotient
            else:
                minimum = _bounded_int(config, "minimum_operand")
                maximum = _bounded_int(config, "maximum_operand")
                left = generator.randint(minimum, maximum)
                right = generator.randint(minimum, maximum)
                if operation == "addition":
                    expected = left + right
                elif operation == "subtraction":
                    expected = left - right
                elif operation == "multiplication":
                    expected = left * right
                else:  # pragma: no cover - guarded by plan validation
                    raise AssertionError(f"unsupported arithmetic operation: {operation}")
            if (left, right) in seen:
                continue
            seen.add((left, right))
            result.append(
                ArithmeticCase(
                    case_id=f"basic-{operation}-{len(seen) - 1:03d}",
                    expected=expected,
                    left=left,
                    operation=operation,
                    right=right,
                )
            )
    return result


def parse_single_integer_answer(response: str) -> int | None:
    """Accept one integer in a concise natural response and reject ambiguous outputs."""

    if not isinstance(response, str):
        raise TypeError("arithmetic response must be text")
    matches = _INTEGER_TOKEN.findall(response)
    if len(matches) != 1:
        return None
    return int(matches[0].replace(",", ""))


def score_arithmetic_responses(
    cases: Sequence[ArithmeticCase],
    responses: Mapping[str, str],
) -> dict[str, Any]:
    """Score exact arithmetic while retaining hashes rather than model response text."""

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("arithmetic case IDs must be unique")
    if set(responses) != set(case_ids):
        raise ValueError("arithmetic responses must exactly match case IDs")

    by_operation: dict[str, list[bool]] = defaultdict(list)
    results = []
    for case in cases:
        response = responses[case.case_id]
        parsed = parse_single_integer_answer(response)
        passed = parsed == case.expected
        by_operation[case.operation].append(passed)
        results.append(
            {
                "case_id": case.case_id,
                "operation": case.operation,
                "passed": passed,
                "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            }
        )
    operations = {
        operation: {
            "cases": len(outcomes),
            "pass_rate": sum(outcomes) / len(outcomes),
            "passed": sum(outcomes),
        }
        for operation, outcomes in sorted(by_operation.items())
    }
    report = {
        "case_results": results,
        "cases": len(cases),
        "operations": operations,
        "pass_rate": sum(item["passed"] for item in results) / len(results),
        "raw_responses_retained": False,
        "schema_version": "lumi.basic-arithmetic-report.v0.1",
    }
    report["report_sha256"] = _fingerprint(report)
    return report
