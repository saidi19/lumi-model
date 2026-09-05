"""Private materialization of exact locally reusable Lumi SFT subsets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_existing_candidate_salvage_preflight_v1 import (
    AYA_MIT_FAMILY_SHA256,
    PureByteLevelBPETokenizer,
    _load_jsonl,
    _measure,
    _owner_ids,
    _select_nemotron,
    _selected_metrics,
    build_report as build_preflight_report,
    file_sha256,
    load_plan as load_preflight_plan,
)
from .posttraining_first_party import canonical_json_bytes as compact_json_bytes
from .posttraining_format import load_chat_template
from .posttraining_owner_review import build_owner_review_html

PLAN_SCHEMA_VERSION = "lumi.sft-existing-candidate-salvage-materialization-plan.v0.1"
REPORT_SCHEMA_VERSION = (
    "lumi.sft-existing-candidate-salvage-materialization-report.v0.1"
)
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-existing-candidate-salvage-materialization-verification.v0.1"
)
MANIFEST_SCHEMA_VERSION = (
    "lumi.sft-existing-candidate-salvage-private-manifest.v0.1"
)

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "download_authorized": False,
    "external_api_authorized": False,
    "gpu_authorized": False,
    "inference_authorized": False,
    "network_authorized": False,
    "predecessor_mutation_authorized": False,
    "private_subset_materialization_authorized": True,
    "production_promotion_authorized": False,
    "remote_access_authorized": False,
    "sft_training_authorized": False,
}


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_value(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(compact_json_bytes(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    plan = _load_json(Path(path).resolve())
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("salvage materialization plan schema changed")
    if plan.get("status") != "authorized_private_local_salvage_materialization":
        raise ValueError("salvage materialization is not authorized")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("salvage materialization authorization widened")
    bindings = plan.get("bindings")
    if not isinstance(bindings, dict) or len(bindings) != 7:
        raise ValueError("salvage materialization bindings changed")
    for name, binding in bindings.items():
        if not isinstance(binding, dict):
            raise ValueError(f"invalid binding: {name}")
        resolved = (root / str(binding.get("path"))).resolve()
        if not resolved.is_relative_to(workspace):
            raise ValueError(f"binding escaped workspace: {name}")
        if not resolved.is_file():
            raise ValueError(f"bound file is missing: {name}")
        if file_sha256(resolved) != binding.get("sha256"):
            raise ValueError(f"binding hash mismatch: {name}")
    output_root = (root / str(plan["output"]["private_root"])).resolve()
    if not output_root.is_relative_to(root / "artifacts"):
        raise ValueError("private materialization output escaped repository artifacts")
    return plan


def _write_lines(path: Path, measured: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for item in measured:
            handle.write(item["line"])
    metrics = _selected_metrics(measured)
    if file_sha256(path) != metrics["candidate_sha256"]:
        raise ValueError("materialized candidate hash differs from projection")
    return {**metrics, "bytes": path.stat().st_size, "path": path.name}


def _review_rows(
    aya_measured: list[dict[str, Any]], prior_owner_ids: set[str]
) -> list[dict[str, Any]]:
    eligible = [
        item
        for item in aya_measured
        if str(item["record"]["record_id"]) not in prior_owner_ids
    ]
    ranked = sorted(
        eligible,
        key=lambda item: hashlib.sha256(
            (
                "lumi-aya-mit-owner-review-delta-v0.1\0"
                + str(item["record"]["record_id"])
            ).encode()
        ).hexdigest(),
    )[:6]
    return [
        {
            "language_mode": "bengali_script",
            "messages": item["record"]["messages"],
            "record_id": item["record"]["record_id"],
            "skill_id": item["record"]["task_category"],
            "split": item["record"]["split"],
        }
        for item in ranked
    ]


def materialize(
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved = {
        name: (root / binding["path"]).resolve()
        for name, binding in plan["bindings"].items()
    }
    preflight_plan = load_preflight_plan(resolved["preflight_plan"], root)
    preflight_report = _load_json(resolved["preflight_report"])
    expected_preflight = build_preflight_report(
        preflight_plan, file_sha256(resolved["preflight_plan"]), root
    )
    if preflight_report != expected_preflight:
        raise ValueError("bound salvage preflight report is not reproducible")
    preflight_verification = _load_json(resolved["preflight_verification"])
    if preflight_verification.get("status") != (
        "PASS_local_salvage_subsets_projected_materialization_not_run"
    ):
        raise ValueError("bound salvage preflight verification did not pass")

    preflight_bindings = preflight_plan["bindings"]
    tokenizer_path = (root / preflight_bindings["tokenizer"]["path"]).resolve()
    template_path = (root / preflight_bindings["chat_template"]["path"]).resolve()
    tokenizer = PureByteLevelBPETokenizer(tokenizer_path)
    template = load_chat_template(template_path)
    prior_owner_ids = _owner_ids(
        (root / preflight_bindings["owner_review_sample"]["path"]).resolve()
    )

    aya_rows = _load_jsonl(
        (root / preflight_bindings["aya_rights_candidate"]["path"]).resolve()
    )
    aya_mit_rows = [
        item
        for item in aya_rows
        if item[0]["provenance"].get("source_dataset_name_sha256")
        == AYA_MIT_FAMILY_SHA256
    ]
    aya_measured, _ = _measure(aya_mit_rows, tokenizer, template)

    nemotron_rows = _load_jsonl(
        (root / preflight_bindings["nemotron_candidate"]["path"]).resolve()
    )
    nemotron_measured, _ = _measure(nemotron_rows, tokenizer, template)
    nemotron_selected, _ = _select_nemotron(nemotron_measured, prior_owner_ids)

    aya_path = output / "aya-mit-candidate.jsonl"
    nemotron_path = output / "nemotron-tool-subset.jsonl"
    aya_metrics = _write_lines(aya_path, aya_measured)
    nemotron_metrics = _write_lines(nemotron_path, nemotron_selected)
    projected = preflight_report["projected_materialization"]
    for metrics, expected in (
        (aya_metrics, projected["aya_mit"]),
        (nemotron_metrics, projected["nemotron_tool_subset"]),
    ):
        if metrics["candidate_sha256"] != expected["candidate_sha256"]:
            raise ValueError("materialized candidate does not match preflight hash")
        if metrics["by_split"] != expected["by_split"]:
            raise ValueError("materialized candidate token ledger changed")

    sample_rows = _review_rows(aya_measured, prior_owner_ids)
    sample_sha256 = hashlib.sha256(compact_json_bytes(sample_rows)).hexdigest()
    sample_path = output / "aya-mit-owner-review-delta-6.json"
    sample_path.write_bytes(canonical_json_bytes({"rows": sample_rows}))
    review_html = build_owner_review_html(sample_rows, sample_sha256=sample_sha256)
    review_path = output / "aya-mit-owner-review-delta-6.html"
    review_path.write_text(review_html, encoding="utf-8")

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "private_nonproduction_salvage_subsets_materialized",
        "plan_file_sha256": plan_file_sha256,
        "preflight_report_file_sha256": plan["bindings"]["preflight_report"]["sha256"],
        "candidates": {
            "aya_mit": aya_metrics,
            "nemotron_tool_subset": nemotron_metrics,
            "banglabridge_reuse": {
                "path": preflight_bindings["banglabridge_candidate"]["path"],
                "sha256": preflight_bindings["banglabridge_candidate"]["sha256"],
            },
            "banglatlit_auxiliary_reuse": {
                "path": preflight_bindings["banglatlit_candidate"]["path"],
                "sha256": preflight_bindings["banglatlit_candidate"]["sha256"],
            },
        },
        "owner_review_delta": {
            "prior_aya_mit_reviewed_rows": 2,
            "new_sample_records": len(sample_rows),
            "sample_sha256": sample_sha256,
            "sample_file_sha256": file_sha256(sample_path),
            "review_html_sha256": file_sha256(review_path),
            "status": "owner_review_pending",
        },
        "exclusions": {
            "aya_wikisplit_cc_by_sa": "excluded_unresolved_share_alike",
            "aya_wikilingua_noncommercial": "permanently_excluded",
            "retired_gpt_oss_closure": "permanently_excluded",
        },
        "checks": {
            "candidate_lines_preserved_byte_exact": True,
            "new_inference_used": False,
            "network_or_gpu_used": False,
            "predecessor_candidates_mutated": False,
            "registry_or_dataset_seal_changed": False,
        },
        "production_eligible": False,
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    manifest_path = output.parent / "private-manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "materialized_private_nonproduction_salvage_owner_review_pending",
        "plan_file_sha256": plan_file_sha256,
        "preflight_report_file_sha256": plan["bindings"]["preflight_report"]["sha256"],
        "output": {
            "aya_mit_candidate_file_sha256": aya_metrics["candidate_sha256"],
            "aya_mit_records": aya_metrics["records"],
            "aya_mit_train_supervised_tokens": aya_metrics["by_split"]["train"]["supervised_tokens"],
            "nemotron_candidate_file_sha256": nemotron_metrics["candidate_sha256"],
            "nemotron_records": nemotron_metrics["records"],
            "nemotron_train_supervised_tokens": nemotron_metrics["by_split"]["train"]["supervised_tokens"],
            "manifest_file_sha256": file_sha256(manifest_path),
            "manifest_internal_sha256": manifest["manifest_sha256"],
        },
        "owner_review_delta": {
            "sample_records": len(sample_rows),
            "sample_sha256": sample_sha256,
            "sample_file_sha256": file_sha256(sample_path),
            "review_html_sha256": file_sha256(review_path),
            "status": "pending",
        },
        "checks": manifest["checks"],
        "raw_training_text_in_report": False,
        "production_eligible": False,
    }
    report["report_sha256"] = sha256_value(report)
    return report


def build_verification(
    report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256")
    if sha256_value(unsigned) != internal:
        raise ValueError("salvage materialization report hash does not verify")
    if report["owner_review_delta"]["sample_records"] != 6:
        raise ValueError("Aya owner review delta size changed")
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_private_salvage_materialized_owner_review_pending",
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": internal,
        "checks": {
            "aya_mit_candidate_reproduced_from_bound_parent": True,
            "nemotron_subset_reproduced_from_bound_parent": True,
            "preflight_hash_and_token_ledgers_matched": True,
            "prior_owner_review_membership_preserved": True,
            "six_row_aya_owner_review_delta_created": True,
            "unresolved_and_retired_sources_excluded": True,
            "no_predecessor_mutation_registry_seal_download_inference_gpu_or_sft": True,
            "report_is_raw_text_free": True,
        },
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification

