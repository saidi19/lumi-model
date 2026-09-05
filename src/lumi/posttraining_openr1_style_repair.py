"""Fail-closed visible-final extraction for the curated OpenR1 candidate."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_openr1_full_run import file_sha256, sha256_value
from .posttraining_owner_review import build_owner_review_html

PLAN_SCHEMA_VERSION = "lumi.sft-openr1-visible-final-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-openr1-visible-final-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-openr1-visible-final-verification.v0.1"
EXPECTED_STATUS = "authorized_local_visible_final_repair_no_promotion"
EXPECTED_AUTHORIZATION = {
    "dataset_promotion_authorized": False,
    "external_api_spend_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "purpose": "existing_openr1_candidate_visible_final_style_repair_only",
    "source_registry_promotion_authorized": False,
}
EXPECTED_LIMITS = {
    "context_window": 4_096,
    "maximum_records": 2_688,
    "maximum_runtime_minutes": 60,
    "maximum_source_supervised_tokens": 6_599_974,
    "maximum_visible_answer_characters": 3_000,
    "minimum_visible_answer_characters": 20,
    "owner_sample_records": 32,
}
EXPECTED_POLICY = {
    "assistant_target": "post_think_visible_final_only",
    "exact_think_pairs_required": 1,
    "hidden_reasoning_is_sft_target": False,
    "hidden_reasoning_stored_separately_private": True,
    "latex_braces_must_balance": True,
    "latex_delimiters_normalized_to_markdown_math": True,
    "owner_decisions_from_raw_trace_review_carried_forward": False,
    "reference_solution_accessed": False,
    "visible_think_tags_allowed": False,
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)


class VisibleFinalRepairRejection(ValueError):
    """A content-level style failure that should be quarantined, not repaired."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def resolve_binding(repository_root: Path, binding: Mapping[str, Any]) -> Path:
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("OpenR1 style-repair binding path must be non-empty")
    path = (repository_root / relative).resolve()
    if not path.is_relative_to(repository_root.parent.resolve()):
        raise ValueError("OpenR1 style-repair binding escaped workspace")
    if not path.is_file():
        raise FileNotFoundError(f"OpenR1 style-repair binding is missing: {relative}")
    expected = _require_sha256(binding.get("sha256"), f"binding {relative}")
    if file_sha256(path) != expected:
        raise ValueError(f"OpenR1 style-repair binding changed: {relative}")
    return path


def load_style_repair_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported OpenR1 style-repair plan")
    if plan.get("status") != EXPECTED_STATUS:
        raise ValueError("OpenR1 style-repair status changed")
    if plan.get("observed_at") != "2026-08-06":
        raise ValueError("OpenR1 style-repair date changed")
    for field, expected in (
        ("authorization", EXPECTED_AUTHORIZATION),
        ("limits", EXPECTED_LIMITS),
        ("policy", EXPECTED_POLICY),
    ):
        if plan.get(field) != expected:
            raise ValueError(f"OpenR1 style-repair {field} changed")
    bindings = plan.get("bindings")
    required = {
        "attribution_manifest",
        "benchmark_registry",
        "chat_template",
        "curation_report",
        "curation_verification",
        "data_contract",
        "input_candidate",
        "input_owner_sample",
        "repair_implementation",
        "repair_runner",
        "repair_verifier",
        "source_registry",
        "tokenizer_config",
        "tokenizer_json",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != required:
        raise ValueError("OpenR1 style-repair binding inventory changed")
    for binding in bindings.values():
        if not isinstance(binding, Mapping):
            raise ValueError("OpenR1 style-repair binding must be an object")
        resolve_binding(root, binding)
    output = plan.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("OpenR1 style-repair output is missing")
    if output.get("production_promotion_permitted") is not False:
        raise ValueError("OpenR1 style-repair promotion is not authorized")
    if output.get("raw_training_text_in_git") is not False:
        raise ValueError("OpenR1 style-repair text may not enter Git")
    private_root = output.get("private_artifact_root")
    report_path = output.get("raw_text_free_report_path")
    verification_path = output.get("raw_text_free_verification_path")
    if not isinstance(private_root, str) or not private_root:
        raise ValueError("OpenR1 style-repair private root is missing")
    if not isinstance(report_path, str) or not report_path.startswith("evidence/"):
        raise ValueError("OpenR1 style-repair report must stay under evidence/")
    if not isinstance(verification_path, str) or not verification_path.startswith(
        "evidence/"
    ):
        raise ValueError("OpenR1 style-repair verification must stay under evidence/")
    private_resolved = (root / private_root).resolve()
    if private_resolved.is_relative_to(root):
        raise ValueError("OpenR1 repaired private text must stay outside Git")
    if not private_resolved.is_relative_to(root.parent):
        raise ValueError("OpenR1 repaired artifacts escaped workspace")
    return plan


def split_think_response(text: str) -> tuple[str, str]:
    """Split one exact reasoning/final pair and reject ambiguous tag structure."""

    if len(_THINK_OPEN.findall(text)) != 1 or len(_THINK_CLOSE.findall(text)) != 1:
        raise VisibleFinalRepairRejection("response must contain exactly one think pair")
    open_match = _THINK_OPEN.search(text)
    close_match = _THINK_CLOSE.search(text)
    assert open_match is not None and close_match is not None
    if text[: open_match.start()].strip():
        raise VisibleFinalRepairRejection(
            "response contains visible content before think tag"
        )
    if close_match.start() <= open_match.end():
        raise VisibleFinalRepairRejection("response think tags are out of order")
    reasoning = text[open_match.end() : close_match.start()].strip()
    final = text[close_match.end() :].strip()
    if not reasoning:
        raise VisibleFinalRepairRejection("response hidden reasoning is empty")
    if not final:
        raise VisibleFinalRepairRejection("response visible final is empty")
    if _THINK_OPEN.search(final) or _THINK_CLOSE.search(final):
        raise VisibleFinalRepairRejection("visible final retained a think tag")
    return reasoning, final


def _balanced_latex_braces(text: str) -> bool:
    depth = 0
    escaped = False
    for character in text:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _balanced_math_delimiters(text: str) -> bool:
    if text.count("\\(") != text.count("\\)"):
        return False
    if text.count("\\[") != text.count("\\]"):
        return False
    return True


def normalize_markdown_math(text: str) -> str:
    """Convert TeX delimiters to the Markdown math form used by the product UI."""

    if not _balanced_latex_braces(text):
        raise VisibleFinalRepairRejection(
            "visible final has unbalanced LaTeX braces"
        )
    if not _balanced_math_delimiters(text):
        raise VisibleFinalRepairRejection(
            "visible final has unbalanced LaTeX delimiters"
        )
    normalized = text.replace("\\[", "\n$$\n").replace("\\]", "\n$$\n")
    normalized = normalized.replace("\\(", "$").replace("\\)", "$")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    if not _balanced_latex_braces(normalized):
        raise VisibleFinalRepairRejection(
            "normalized final has unbalanced LaTeX braces"
        )
    if normalized.count("$$") % 2:
        raise VisibleFinalRepairRejection(
            "normalized final has unbalanced display math"
        )
    inline_probe = normalized.replace("$$", "")
    if len(re.findall(r"(?<!\\)\$", inline_probe)) % 2:
        raise VisibleFinalRepairRejection(
            "normalized final has unbalanced inline math"
        )
    return normalized


def repair_record(
    record: Mapping[str, Any],
    attribution: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Create one visible-only SFT record and separate private reasoning row."""

    messages = record.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or [message.get("role") for message in messages] != ["user", "assistant"]
    ):
        raise ValueError("OpenR1 source record message shape changed")
    old_record_id = record.get("record_id")
    if not isinstance(old_record_id, str) or not old_record_id:
        raise ValueError("OpenR1 source record ID changed")
    reasoning, final = split_think_response(str(messages[1]["content"]))
    visible_final = normalize_markdown_math(final)
    if not (
        EXPECTED_LIMITS["minimum_visible_answer_characters"]
        <= len(visible_final)
        <= EXPECTED_LIMITS["maximum_visible_answer_characters"]
    ):
        raise VisibleFinalRepairRejection(
            "OpenR1 visible final length is outside bounds"
        )
    new_messages = [
        {"content": str(messages[0]["content"]), "role": "user"},
        {"content": visible_final, "role": "assistant"},
    ]
    transform = {
        "adapter": "openr1_post_think_visible_final_v0.1",
        "hidden_reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
        "parent_record_id": old_record_id,
        "visible_final_sha256": hashlib.sha256(visible_final.encode()).hexdigest(),
    }
    new_record_id = f"openr1_visible_final-{sha256_value(transform)[:24]}"
    repaired = json.loads(json.dumps(record, ensure_ascii=False))
    repaired["messages"] = new_messages
    repaired["record_id"] = new_record_id
    repaired["provenance"] = {
        "consent_basis": record["provenance"]["consent_basis"],
        "content_sha256": sha256_value(new_messages),
        "transform_sha256": sha256_value(transform),
    }
    repaired["quality"] = {
        **dict(record["quality"]),
        "review_method": "openr1_visible_final_style_gate_v0.1",
    }
    repaired["style_provenance"] = {
        "hidden_reasoning_in_assistant_target": False,
        "latex_delimiters": "markdown_math_dollar_v0.1",
        "parent_record_id": old_record_id,
        "visible_think_tags": False,
    }
    hidden = {
        "content": reasoning,
        "content_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
        "parent_record_id": old_record_id,
        "reasoning_id": f"openr1_hidden_reasoning-{sha256_value(transform)[:24]}",
        "sft_target": False,
        "visible_record_id": new_record_id,
    }
    repaired_attribution = dict(attribution)
    repaired_attribution["parent_record_id"] = old_record_id
    repaired_attribution["record_id"] = new_record_id
    repaired_attribution["style_transform_sha256"] = sha256_value(transform)
    repaired_attribution["visible_final_sha256"] = transform["visible_final_sha256"]
    return repaired, hidden, repaired_attribution


def select_visible_final_review_sample(
    source_sample: Sequence[Mapping[str, Any]],
    repaired_by_parent: Mapping[str, Mapping[str, Any]],
    repaired_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Retain surviving source-sample rows and fill rejected slots deterministically."""

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for source_row in source_sample:
        repaired = repaired_by_parent.get(str(source_row.get("record_id")))
        if repaired is None:
            continue
        copied = json.loads(json.dumps(repaired, ensure_ascii=False))
        selected.append(copied)
        selected_ids.add(str(copied["record_id"]))
    retained = len(selected)
    if retained > EXPECTED_LIMITS["owner_sample_records"]:
        raise ValueError("OpenR1 source owner sample exceeded its bound")
    for repaired in sorted(repaired_records, key=lambda row: str(row["record_id"])):
        if len(selected) == EXPECTED_LIMITS["owner_sample_records"]:
            break
        record_id = str(repaired["record_id"])
        if record_id in selected_ids:
            continue
        copied = json.loads(json.dumps(repaired, ensure_ascii=False))
        selected.append(copied)
        selected_ids.add(record_id)
    if len(selected) != EXPECTED_LIMITS["owner_sample_records"]:
        raise ValueError("OpenR1 repaired owner sample could not be filled")
    return selected, retained


def style_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    lengths: list[int] = []
    for record in records:
        messages = record["messages"]
        final = str(messages[-1]["content"])
        counts["records"] += 1
        counts["visible_think_tags"] += bool(
            _THINK_OPEN.search(final) or _THINK_CLOSE.search(final)
        )
        counts["unbalanced_latex_braces"] += not _balanced_latex_braces(final)
        counts["unbalanced_display_math"] += final.count("$$") % 2 != 0
        counts["legacy_tex_delimiters"] += bool(
            re.search(r"\\\(|\\\)|\\\[|\\\]", final)
        )
        counts["boxed_math_records"] += "\\boxed" in final
        counts["markdown_heading_records"] += bool(re.search(r"(?m)^#{1,6}\s", final))
        lengths.append(len(final))
    ordered = sorted(lengths)
    return {
        "boxed_math_records": counts["boxed_math_records"],
        "legacy_tex_delimiter_records": counts["legacy_tex_delimiters"],
        "markdown_heading_records": counts["markdown_heading_records"],
        "records": counts["records"],
        "unbalanced_display_math_records": counts["unbalanced_display_math"],
        "unbalanced_latex_brace_records": counts["unbalanced_latex_braces"],
        "visible_answer_characters": {
            "maximum": max(ordered, default=0),
            "median": ordered[len(ordered) // 2] if ordered else 0,
            "minimum": min(ordered, default=0),
            "p95": ordered[max(0, int(len(ordered) * 0.95) - 1)] if ordered else 0,
        },
        "visible_think_tag_records": counts["visible_think_tags"],
    }


def build_visible_final_review_html(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    payload = [
        {
            "language_mode": "english",
            "messages": [dict(message) for message in row["messages"]],
            "record_id": row["record_id"],
            "skill_id": "reasoning_math_science_visible_final",
            "split": row["split"],
        }
        for row in rows
    ]
    sample_sha256 = sha256_value(payload)
    html = build_owner_review_html(payload, sample_sha256=sample_sha256)
    replacements = {
        "<title>Lumi owner review</title>": (
            "<title>Lumi OpenR1 visible-final review</title>"
        ),
        "<h1>Lumi owner spot-check</h1>": (
            "<h1>Lumi OpenR1 final-answer review</h1>"
        ),
        (
            f"<p>Review all {len(payload)} rows. Check correctness, naturalness, "
            "language, context, and safety.</p>"
        ): (
            f"<p>Review all {len(payload)} visible final answers. Hidden reasoning "
            "has been removed. Check correctness, clarity, concision, naturalness, "
            "safety, and whether the answer is suitable for Lumi.</p>"
        ),
        (
            "These rows are AI-generated and owner-reviewed. Approving them does not "
            "make them human-written."
        ): (
            "Only the visible answer is shown. Dollar-delimited text is exact "
            "Markdown/TeX math syntax and will be rendered by the product UI; this "
            "offline page intentionally loads no external JavaScript. Approval is "
            "still non-production and does not authorize SFT."
        ),
        "lumi-owner-review-decisions-v0.3.json": (
            "lumi-openr1-visible-final-owner-decisions-v0.1.json"
        ),
    }
    for old, new in replacements.items():
        if old not in html:
            raise ValueError(f"OpenR1 visible-final review template drifted: {old}")
        html = html.replace(old, new)
    return html, sample_sha256


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite JSON artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite JSONL artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite text artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
