"""Deterministic post-training benchmark and safety evaluation primitives."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

EVALUATION_PLAN_SCHEMA_VERSION = "lumi.posttraining-eval.v0.1"
SAFETY_CASE_SCHEMA_VERSION = "lumi.safety-case.v0.1"
FINAL_BASE_TRAINING_TOKENS = 80_000_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
SUPPORTED_CHECKS = frozenset(
    {
        "contains_all",
        "contains_any",
        "max_characters",
        "min_characters",
        "not_contains_any",
        "regex",
    }
)


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
class EvaluationCase:
    case_id: str
    checks: tuple[Mapping[str, Any], ...]
    messages: tuple[Mapping[str, str], ...]
    suite: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvaluationCase:
        if value.get("schema_version") != SAFETY_CASE_SCHEMA_VERSION:
            raise ValueError("unsupported safety case schema")
        case_id = value.get("case_id")
        suite = value.get("suite")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("case_id is required")
        if not isinstance(suite, str) or not suite:
            raise ValueError("suite is required")
        raw_messages = value.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError(f"{case_id}: messages must be non-empty")
        messages = []
        for message in raw_messages:
            if not isinstance(message, Mapping):
                raise ValueError(f"{case_id}: message must be an object")
            role = message.get("role")
            content = message.get("content")
            if role not in {"developer", "system", "user"}:
                raise ValueError(f"{case_id}: test messages may not prefill assistant output")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{case_id}: message content is empty")
            messages.append({"content": content, "role": role})
        if messages[-1]["role"] != "user":
            raise ValueError(f"{case_id}: final message must be user")

        raw_checks = value.get("checks")
        if not isinstance(raw_checks, list) or not raw_checks:
            raise ValueError(f"{case_id}: checks must be non-empty")
        checks: list[Mapping[str, Any]] = []
        for check in raw_checks:
            if not isinstance(check, Mapping) or check.get("type") not in SUPPORTED_CHECKS:
                raise ValueError(f"{case_id}: unsupported check")
            _validate_check(case_id, check)
            checks.append(dict(check))
        return cls(
            case_id=case_id,
            checks=tuple(checks),
            messages=tuple(messages),
            suite=suite,
        )


def _validate_check(case_id: str, check: Mapping[str, Any]) -> None:
    check_type = check["type"]
    if check_type in {"contains_all", "contains_any", "not_contains_any"}:
        values = check.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{case_id}: {check_type} requires values")
        if any(not isinstance(item, str) or not item for item in values):
            raise ValueError(f"{case_id}: {check_type} values must be strings")
    elif check_type in {"max_characters", "min_characters"}:
        value = check.get("value")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{case_id}: {check_type} requires a non-negative integer")
    elif check_type == "regex":
        pattern = check.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"{case_id}: regex requires a pattern")
        re.compile(pattern)


def validate_evaluation_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an evaluation plan and return a plain immutable-ready copy."""

    if value.get("schema_version") != EVALUATION_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation plan schema")
    baseline = value.get("base_baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("evaluation plan needs the sealed base baseline")
    if baseline.get("binding") != "configs/lumi-4b-final-base-binding-v0.1.json":
        raise ValueError("evaluation base binding changed")
    if baseline.get("status") != "sealed":
        raise ValueError("base baseline is not sealed")
    if _SHA256.fullmatch(str(baseline.get("frozen_evaluation_report_sha256", ""))) is None:
        raise ValueError("base frozen evaluation SHA-256 is invalid")
    suites = value.get("suites")
    if not isinstance(suites, Mapping) or not suites:
        raise ValueError("evaluation plan suites must be non-empty")
    for suite, config in suites.items():
        if not isinstance(suite, str) or not suite or not isinstance(config, Mapping):
            raise ValueError("invalid suite configuration")
        minimum = config.get("minimum_pass_rate")
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, (int, float))
            or not 0 <= float(minimum) <= 1
        ):
            raise ValueError(f"{suite}: minimum_pass_rate must be between 0 and 1")
        if not isinstance(config.get("required"), bool):
            raise ValueError(f"{suite}: required must be boolean")
    adapters = value.get("benchmark_adapters")
    if not isinstance(adapters, list) or not adapters:
        raise ValueError("benchmark_adapters must be non-empty")
    adapter_ids = [item.get("id") for item in adapters if isinstance(item, Mapping)]
    if len(adapter_ids) != len(adapters) or any(
        not isinstance(item, str) or not item for item in adapter_ids
    ):
        raise ValueError("every benchmark adapter needs an id")
    if len(set(adapter_ids)) != len(adapter_ids):
        raise ValueError("benchmark adapter ids must be unique")
    return json.loads(json.dumps(value))


def validate_model_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Require exact model provenance before scoring can become release evidence."""

    if not isinstance(value.get("model_id"), str) or not value["model_id"]:
        raise ValueError("model binding requires model_id")
    if value.get("checkpoint_stage") not in {"base", "sft"}:
        raise ValueError("model binding checkpoint_stage must be base or sft")
    if value.get("base_training_tokens") != FINAL_BASE_TRAINING_TOKENS:
        raise ValueError("model binding must descend from the exact 80B base boundary")
    if _GIT_SHA.fullmatch(str(value.get("source_commit", ""))) is None:
        raise ValueError("model binding source_commit must be an exact Git SHA")
    for field in ("checkpoint_manifest_sha256", "tokenizer_manifest_sha256"):
        if _SHA256.fullmatch(str(value.get(field, ""))) is None:
            raise ValueError(f"model binding {field} must be a SHA-256")
    return json.loads(json.dumps(value))


def _evaluate_check(check: Mapping[str, Any], response: str) -> bool:
    folded = response.casefold()
    check_type = check["type"]
    if check_type == "contains_all":
        return all(value.casefold() in folded for value in check["values"])
    if check_type == "contains_any":
        return any(value.casefold() in folded for value in check["values"])
    if check_type == "not_contains_any":
        return all(value.casefold() not in folded for value in check["values"])
    if check_type == "max_characters":
        return len(response) <= check["value"]
    if check_type == "min_characters":
        return len(response) >= check["value"]
    if check_type == "regex":
        return re.search(check["pattern"], response, re.IGNORECASE | re.MULTILINE) is not None
    raise AssertionError(f"unhandled check: {check_type}")


def score_response(case: EvaluationCase, response: str) -> dict[str, Any]:
    """Score one response without storing response text in the result."""

    if not isinstance(response, str):
        raise TypeError("response must be text")
    checks = [
        {
            "passed": _evaluate_check(check, response),
            "type": check["type"],
        }
        for check in case.checks
    ]
    return {
        "case_id": case.case_id,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
        "response_characters": len(response),
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "suite": case.suite,
    }


def build_evaluation_report(
    *,
    cases: list[EvaluationCase],
    responses: Mapping[str, str],
    plan: Mapping[str, Any],
    model_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate case results into a thresholded, hash-bound report."""

    validated_plan = validate_evaluation_plan(plan)
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("evaluation case IDs must be unique")
    if set(responses) != set(case_ids):
        raise ValueError("responses must exactly match evaluation case IDs")
    validated_binding = validate_model_binding(model_binding)

    configured_suites = validated_plan["suites"]
    if any(case.suite not in configured_suites for case in cases):
        raise ValueError("evaluation case references an unconfigured suite")

    results = [score_response(case, responses[case.case_id]) for case in cases]
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_suite[result["suite"]].append(result)

    suites = {}
    required_passed = True
    for name, config in configured_suites.items():
        suite_results = by_suite.get(name, [])
        passed_cases = sum(item["passed"] for item in suite_results)
        pass_rate = passed_cases / len(suite_results) if suite_results else 0.0
        passed = bool(suite_results) and pass_rate >= float(config["minimum_pass_rate"])
        suites[name] = {
            "cases": len(suite_results),
            "minimum_pass_rate": float(config["minimum_pass_rate"]),
            "pass_rate": pass_rate,
            "passed": passed,
            "required": bool(config["required"]),
        }
        if config["required"] and not passed:
            required_passed = False

    report = {
        "case_results": results,
        "model_binding": validated_binding,
        "model_binding_sha256": _fingerprint(validated_binding),
        "passed": required_passed,
        "plan_sha256": _fingerprint(validated_plan),
        "raw_responses_retained": False,
        "schema_version": EVALUATION_PLAN_SCHEMA_VERSION,
        "suites": suites,
    }
    report["report_sha256"] = _fingerprint(report)
    return report
