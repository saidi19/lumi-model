"""Safety and structural helpers for Lumi tokenizer decision gates."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


def reserved_token_literals(contract: Mapping[str, Any]) -> dict[str, str]:
    """Return the exact reserved strings after validating the contract shape."""

    specifications = contract.get("special_tokens")
    if not isinstance(specifications, Mapping) or not specifications:
        raise ValueError("tokenizer contract must define special_tokens")
    result = {}
    for name, specification in specifications.items():
        if not isinstance(name, str) or not isinstance(specification, Mapping):
            raise ValueError("invalid tokenizer special-token specification")
        token = specification.get("token")
        if not isinstance(token, str) or not token:
            raise ValueError(f"special token {name!r} must be nonempty text")
        result[name] = token
    if len(result.values()) != len(set(result.values())):
        raise ValueError("tokenizer special-token strings must be unique")
    return result


def count_reserved_literals(text: str, literals: Mapping[str, str]) -> dict[str, int]:
    """Count exact control-token literals without retaining surrounding text."""

    return {
        name: count
        for name, token in literals.items()
        if (count := text.count(token)) > 0
    }


def maximum_digit_span(
    pretokenized_pieces: list[tuple[str, tuple[int, int]]],
    source_text: str | None = None,
) -> int:
    """Return the longest Unicode digit run contained by one pre-tokenized piece.

    Byte-level pre-tokenization rewrites non-ASCII digits into visible byte symbols, so
    callers that have the original text should pass it and let offsets recover the
    source characters.
    """

    return max(
        (
            len(match.group(0))
            for piece, (start, end) in pretokenized_pieces
            for match in re.finditer(
                r"\d+", source_text[start:end] if source_text is not None else piece
            )
        ),
        default=0,
    )


def encode_untrusted_text(tokenizer: Any, text: str) -> list[int]:
    """Encode data text while forcing reserved literals through ordinary BPE pieces."""

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        split_special_tokens=True,
    )
    reserved_ids = set(tokenizer.all_special_ids)
    collisions = sorted(reserved_ids.intersection(token_ids))
    if collisions:
        raise ValueError(f"untrusted text emitted reserved token IDs: {collisions}")
    return token_ids


def _fraction_change(candidate: int, baseline: int) -> float:
    if baseline < 1:
        raise ValueError("baseline token count must be positive")
    return candidate / baseline - 1.0


def evaluate_tokenizer_gate(
    config: Mapping[str, Any],
    comparison: Mapping[str, Any],
    reserved_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the pinned final-tokenizer gate and return a decision report."""

    baseline_config = config["baseline"]
    candidate_config = config["candidate"]
    decision_config = config["decision"]
    sample_config = config["sample"]
    baseline_label = baseline_config["label"]
    candidate_label = candidate_config["label"]
    tokenizers = comparison["tokenizers"]
    baseline = tokenizers[baseline_label]
    candidate = tokenizers[candidate_label]
    conditions: list[dict[str, Any]] = []

    def record(name: str, passed: bool, observed: Any, expected: Any) -> None:
        conditions.append(
            {
                "expected": expected,
                "name": name,
                "observed": observed,
                "passed": bool(passed),
            }
        )

    for field in ("documents", "output_sha256", "text_utf8_bytes"):
        observed = comparison["sample"][field]
        expected = sample_config[field]
        record(f"sample.{field}", observed == expected, observed, expected)
    record(
        "evaluation.full_sample",
        comparison["evaluation"]["documents"] == sample_config["documents"]
        and comparison["evaluation"]["max_documents"] is None,
        comparison["evaluation"],
        {"documents": sample_config["documents"], "max_documents": None},
    )
    record(
        "baseline.tokenizer_json_sha256",
        baseline["artifact"]["tokenizer_json_sha256"]
        == baseline_config["tokenizer_json_sha256"],
        baseline["artifact"]["tokenizer_json_sha256"],
        baseline_config["tokenizer_json_sha256"],
    )
    for label, result, specification in (
        (baseline_label, baseline, baseline_config),
        (candidate_label, candidate, candidate_config),
    ):
        record(
            f"{label}.vocab_size",
            result["artifact"]["vocab_size"] == specification["vocab_size"],
            result["artifact"]["vocab_size"],
            specification["vocab_size"],
        )

    baseline_overall = baseline["overall"]
    candidate_overall = candidate["overall"]
    identical_overall_input = all(
        baseline_overall[field] == candidate_overall[field]
        for field in ("documents", "text_utf8_bytes", "words")
    )
    record(
        "comparison.identical_overall_input",
        identical_overall_input,
        {
            "baseline": {
                field: baseline_overall[field]
                for field in ("documents", "text_utf8_bytes", "words")
            },
            "candidate": {
                field: candidate_overall[field]
                for field in ("documents", "text_utf8_bytes", "words")
            },
        },
        "identical",
    )
    overall_change = _fraction_change(
        candidate_overall["tokens"], baseline_overall["tokens"]
    )
    overall_limit = decision_config["overall_token_increase_max_fraction"]
    record(
        "efficiency.overall_token_change",
        overall_change <= overall_limit,
        overall_change,
        {"max_fraction": overall_limit},
    )

    baseline_categories = baseline["categories"]
    candidate_categories = candidate["categories"]
    record(
        "comparison.identical_category_set",
        set(baseline_categories) == set(candidate_categories),
        sorted(candidate_categories),
        sorted(baseline_categories),
    )
    category_changes = {}
    category_limit = decision_config["domain_token_increase_max_fraction"]
    for category in sorted(set(baseline_categories).intersection(candidate_categories)):
        baseline_category = baseline_categories[category]
        candidate_category = candidate_categories[category]
        identical_input = all(
            baseline_category[field] == candidate_category[field]
            for field in ("documents", "text_utf8_bytes", "words")
        )
        record(
            f"comparison.identical_category_input.{category}",
            identical_input,
            {
                "baseline_documents": baseline_category["documents"],
                "candidate_documents": candidate_category["documents"],
            },
            "identical",
        )
        change = _fraction_change(
            candidate_category["tokens"], baseline_category["tokens"]
        )
        category_changes[category] = change
        record(
            f"efficiency.category_token_change.{category}",
            change <= category_limit,
            change,
            {"max_fraction": category_limit},
        )

    baseline_probes = baseline["probes"]
    candidate_probes = candidate["probes"]
    record(
        "comparison.identical_probe_set",
        set(baseline_probes) == set(candidate_probes),
        sorted(candidate_probes),
        sorted(baseline_probes),
    )
    probe_changes = {}
    probe_limit = decision_config["general_probe_token_increase_max"]
    for probe in decision_config["general_probes"]:
        baseline_probe = baseline_probes[probe]
        candidate_probe = candidate_probes[probe]
        increase = candidate_probe["tokens"] - baseline_probe["tokens"]
        probe_changes[probe] = increase
        record(
            f"efficiency.general_probe_token_increase.{probe}",
            candidate_probe["text_utf8_bytes"] == baseline_probe["text_utf8_bytes"]
            and increase <= probe_limit,
            increase,
            {"max_tokens": probe_limit},
        )

    digit_limit = candidate_config["digit_chunk_size"]
    candidate_diagnostics = candidate["pretokenizer_diagnostics"]
    for probe in decision_config["digit_boundary_probes"]:
        observed = candidate_diagnostics[probe]["max_consecutive_digits_per_piece"]
        record(
            f"structure.digit_boundary.{probe}",
            observed <= digit_limit,
            observed,
            {"max_digits": digit_limit},
        )

    audit_sample = reserved_audit["sample"]
    record(
        "security.reserved_literal_audit",
        reserved_audit["status"] == "pass"
        and reserved_audit["total_literal_occurrences"] == 0
        and reserved_audit["affected_documents"] == 0
        and reserved_audit["documents_scanned"] == sample_config["documents"]
        and audit_sample["output_sha256"] == sample_config["output_sha256"]
        and audit_sample["text_utf8_bytes"] == sample_config["text_utf8_bytes"],
        {
            "affected_documents": reserved_audit["affected_documents"],
            "documents_scanned": reserved_audit["documents_scanned"],
            "status": reserved_audit["status"],
            "total_literal_occurrences": reserved_audit[
                "total_literal_occurrences"
            ],
        },
        {
            "affected_documents": 0,
            "documents_scanned": sample_config["documents"],
            "status": "pass",
            "total_literal_occurrences": 0,
        },
    )

    passed = all(condition["passed"] for condition in conditions)
    return {
        "artifacts": {
            "baseline_tokenizer_json_sha256": baseline["artifact"][
                "tokenizer_json_sha256"
            ],
            "candidate_tokenizer_json_sha256": candidate["artifact"][
                "tokenizer_json_sha256"
            ],
        },
        "conditions": conditions,
        "decision": (
            decision_config["structural_pass_action"]
            if passed
            else decision_config["failure_action"]
        ),
        "metrics": {
            "category_token_change_fraction": category_changes,
            "general_probe_token_increase": probe_changes,
            "overall_token_change_fraction": overall_change,
        },
        "schema_version": 1,
        "status": "pass" if passed else "fail",
    }
