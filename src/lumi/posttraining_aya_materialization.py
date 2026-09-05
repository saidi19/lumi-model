"""Private, non-production materialization for the reviewed Aya repair projection."""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_aya_repair import (
    DATASET_ID,
    SOURCE_ID,
    SOURCE_LICENSE,
    SOURCE_REVISION,
    resolve_workspace_binding,
    validate_report_seal,
)
from .posttraining_oasst1_full_run import file_sha256, sha256_value

PLAN_SCHEMA_VERSION = "lumi.sft-aya-bengali-materialization-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-aya-bengali-materialization-report.v0.1"
ATTRIBUTION_SCHEMA_VERSION = "lumi.sft-aya-bengali-attribution.v0.1"
DECISIONS_SCHEMA_VERSION = "lumi.sft-aya-bengali-owner-review-decisions.v0.1"

EXPECTED_OWNER_STATEMENT = (
    "I approve one bounded CPU-only materialization of the exact Aya Bengali repair "
    "candidate projected by receipt "
    "6d7c20b19f6b8154c58f51c59521d1e0045dbcbe3e06baa10ac1766c0f6800b2: "
    "maximum 1,478 records / 577,453 supervised tokens and 60 minutes. It may "
    "write the private candidate, attribution manifest, and deterministic 32-row "
    "owner-review packet outside Git. No network/API, GPU, source-registry "
    "promotion, dataset seal, SFT training, unrelated data access, or external "
    "distribution."
)
EXPECTED_STATUS = "authorized_private_candidate_materialization"
EXPECTED_AUTHORIZATION = {
    "candidate_materialization_authorized": True,
    "dataset_promotion_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "purpose": "exact_aya_bengali_private_candidate_and_owner_review_materialization",
    "source_registry_promotion_authorized": False,
    "unrelated_source_access_authorized": False,
}
EXPECTED_LIMITS = {
    "exact_private_artifacts": 3,
    "maximum_candidate_records": 1_478,
    "maximum_runtime_minutes": 60,
    "maximum_supervised_tokens": 577_453,
    "owner_sample_records": 32,
    "source_page_objects": 23,
    "source_rows": 1_534,
}
EXPECTED_PREFLIGHT = {
    "attribution_ledger_sha256": (
        "7a306b76db27f9bbdc59d79d1b12cf7fb1ae1410821357ac9efe7316c4aafd61"
    ),
    "candidate_records": 1_478,
    "file_sha256": "9fa37c0462e20be5849cf68dbed668dd956984cb252b028226b68d0d15cdfddb",
    "internal_report_sha256": ("6d7c20b19f6b8154c58f51c59521d1e0045dbcbe3e06baa10ac1766c0f6800b2"),
    "record_ids_sha256": ("7dcd6759adfbd8158ddb87d218debe465701658011f27c2c23bd48b9ddc1c5c5"),
    "supervised_tokens": 577_453,
    "token_ledger_sha256": ("1e733a10120d437e307c066b2d7a6be3c64aa5cd47e79f57e2b6a91a0299234d"),
}
EXPECTED_OUTPUT = {
    "attribution_manifest_filename": "attribution-manifest.jsonl",
    "candidate_filename": "accepted-records.jsonl",
    "owner_review_filename": "owner-review-32.html",
    "private_artifact_root": (
        "../lumi-artifacts/4b-sft-aya-bengali-materialization-v0.1/6d7c20b19f6b/attempt-001"
    ),
    "raw_text_free_report_path": ("evidence/lumi-4b-sft-aya-bengali-materialization-v0.1.json"),
    "raw_training_text_in_git": False,
}
EXPECTED_BINDING_NAMES = frozenset(
    {
        "benchmark_registry",
        "chat_template",
        "data_contract",
        "filter_implementation",
        "format_implementation",
        "input_audit_report",
        "input_inventory",
        "materialization_implementation",
        "materialization_runner",
        "preflight_plan",
        "preflight_report",
        "reconciliation_receipt",
        "repair_implementation",
        "source_registry",
        "tokenizer_config",
        "tokenizer_json",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def load_aya_materialization_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load the exact owner-authorized, non-production materialization plan."""

    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported Aya materialization plan")
    if plan.get("status") != EXPECTED_STATUS:
        raise ValueError("Aya materialization status changed")
    if plan.get("observed_at") != "2026-08-06":
        raise ValueError("Aya materialization observation date changed")
    for field, expected in {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "output": EXPECTED_OUTPUT,
        "preflight": EXPECTED_PREFLIGHT,
    }.items():
        if plan.get(field) != expected:
            raise ValueError(f"Aya materialization {field} changed")

    owner = plan.get("owner_authorization")
    if not isinstance(owner, Mapping) or owner.get("statement") != EXPECTED_OWNER_STATEMENT:
        raise ValueError("Aya materialization owner authorization changed")
    statement_sha256 = hashlib.sha256(EXPECTED_OWNER_STATEMENT.encode("utf-8")).hexdigest()
    if owner.get("sha256") != statement_sha256:
        raise ValueError("Aya materialization owner authorization hash changed")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDING_NAMES:
        raise ValueError("Aya materialization binding inventory changed")
    for name in sorted(EXPECTED_BINDING_NAMES):
        binding = bindings[name]
        if not isinstance(binding, Mapping):
            raise ValueError(f"Aya materialization binding is invalid: {name}")
        resolve_workspace_binding(repository_root, binding)

    root = Path(repository_root).resolve()
    private_root = (root / EXPECTED_OUTPUT["private_artifact_root"]).resolve()
    if private_root.is_relative_to(root):
        raise ValueError("Aya private materialization root entered the Git repository")
    if not private_root.is_relative_to(root.parent):
        raise ValueError("Aya private materialization root escaped the workspace")
    return plan


def verify_exact_preflight(report: Mapping[str, Any]) -> None:
    """Require the exact sealed projection approved by the owner."""

    validate_report_seal(report)
    candidate = report.get("candidate")
    token_measurement = report.get("token_measurement")
    if not isinstance(candidate, Mapping) or not isinstance(token_measurement, Mapping):
        raise ValueError("Aya preflight metrics are missing")
    observed = {
        "attribution_ledger_sha256": candidate.get("attribution_ledger_sha256"),
        "candidate_records": candidate.get("records"),
        "internal_report_sha256": report.get("report_sha256"),
        "record_ids_sha256": candidate.get("record_ids_sha256"),
        "supervised_tokens": candidate.get("supervised_tokens"),
        "token_ledger_sha256": token_measurement.get("token_ledger_sha256"),
    }
    expected = {key: value for key, value in EXPECTED_PREFLIGHT.items() if key != "file_sha256"}
    if observed != expected:
        raise ValueError("Aya materialization preflight projection changed")
    if candidate.get("materialized") is not False:
        raise ValueError("Aya preflight unexpectedly materialized a candidate")


def load_exact_aya_pages(
    inventory_path: Path,
    *,
    expected_objects: int,
    expected_rows: int,
) -> list[dict[str, Any]]:
    """Load only the exact hash-bound local page set from the prior audit."""

    inventory = _load_json(inventory_path)
    pages = inventory.get("page_inventory")
    if (
        inventory.get("rows") != expected_rows
        or not isinstance(pages, list)
        or len(pages) != expected_objects
    ):
        raise ValueError("Aya materialization input inventory changed")
    raw_root = inventory_path.parent / "raw"
    wrapped_rows: list[dict[str, Any]] = []
    expected_offset = 0
    for number, item in enumerate(pages):
        if not isinstance(item, Mapping):
            raise ValueError("Aya materialization page inventory item is invalid")
        filename = f"aya-bengali-page-{number:03d}.json"
        if item.get("filename") != filename or item.get("offset") != expected_offset:
            raise ValueError("Aya materialization page sequence changed")
        page_path = raw_root / filename
        if file_sha256(page_path) != item.get("sha256"):
            raise ValueError(f"Aya materialization page hash changed: {filename}")
        if page_path.stat().st_size != item.get("size_bytes"):
            raise ValueError(f"Aya materialization page size changed: {filename}")
        payload = _load_json(page_path)
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != item.get("rows"):
            raise ValueError(f"Aya materialization page row count changed: {filename}")
        wrapped_rows.extend(rows)
        expected_offset += len(rows)
    if len(wrapped_rows) != expected_rows:
        raise ValueError("Aya materialization page set is incomplete")
    return wrapped_rows


def build_attribution_manifest(
    final_records: Sequence[Mapping[str, Any]],
    *,
    candidates_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build a raw-text-free content-addressed attribution row per candidate."""

    manifest = []
    for record in final_records:
        record_id = str(record["record_id"])
        candidate = candidates_by_id.get(record_id)
        if not isinstance(candidate, Mapping):
            raise ValueError(f"Aya final record lost source attribution: {record_id}")
        source = record.get("source")
        provenance = record.get("provenance")
        if not isinstance(source, Mapping) or not isinstance(provenance, Mapping):
            raise ValueError(f"Aya final record metadata is invalid: {record_id}")
        manifest.append(
            {
                "annotation_type": candidate["annotation_type"],
                "annotator_sha256": candidate["annotator_sha256"],
                "content_sha256": candidate["content_sha256"],
                "dataset_id": DATASET_ID,
                "document_id": source["document_id"],
                "full_payload_sha256": candidate["full_payload_sha256"],
                "license_id": SOURCE_LICENSE,
                "record_id": record_id,
                "revision": SOURCE_REVISION,
                "schema_version": ATTRIBUTION_SCHEMA_VERSION,
                "source_id": SOURCE_ID,
                "transform_sha256": provenance["transform_sha256"],
                "uri": source["uri"],
            }
        )
    manifest.sort(key=lambda row: row["record_id"])
    minimal = [
        {
            "annotator_sha256": row["annotator_sha256"],
            "record_id": row["record_id"],
            "revision": row["revision"],
            "source_id": row["source_id"],
        }
        for row in manifest
    ]
    if sha256_value(minimal) != EXPECTED_PREFLIGHT["attribution_ledger_sha256"]:
        raise ValueError("Aya attribution ledger no longer matches the approved preflight")
    return manifest


def render_owner_review_html(
    sample: Sequence[Mapping[str, Any]],
    *,
    attribution_file_sha256: str,
    candidate_file_sha256: str,
    sample_record_ids_sha256: str,
) -> str:
    """Render a self-contained private review UI with fail-closed JSON export."""

    _require_sha256(attribution_file_sha256, "attribution_file_sha256")
    _require_sha256(candidate_file_sha256, "candidate_file_sha256")
    _require_sha256(sample_record_ids_sha256, "sample_record_ids_sha256")
    cards = []
    record_ids = []
    for index, record in enumerate(sample, start=1):
        record_id = str(record["record_id"])
        record_ids.append(record_id)
        messages = record.get("messages")
        if not isinstance(messages, Sequence) or len(messages) != 2:
            raise ValueError("Aya owner review sample message shape changed")
        prompt = html.escape(str(messages[0]["content"]))
        response = html.escape(str(messages[1]["content"]))
        safe_id = html.escape(record_id, quote=True)
        cards.append(
            f"""
            <article class="card" data-record-id="{safe_id}">
              <header><span class="index">{index:02d}</span><code>{safe_id}</code></header>
              <h3>User</h3><pre>{prompt}</pre>
              <h3>Lumi answer candidate</h3><pre>{response}</pre>
              <div class="decision">
                <button type="button" data-value="accept">Accept</button>
                <button type="button" data-value="reject">Reject</button>
                <textarea placeholder="Optional reviewer note"></textarea>
              </div>
            </article>"""
        )
    ids_json = json.dumps(record_ids, ensure_ascii=False)
    bindings_json = json.dumps(
        {
            "attribution_file_sha256": attribution_file_sha256,
            "candidate_file_sha256": candidate_file_sha256,
            "preflight_receipt_sha256": EXPECTED_PREFLIGHT["internal_report_sha256"],
            "sample_record_ids_sha256": sample_record_ids_sha256,
        },
        sort_keys=True,
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lumi Aya Bengali owner review</title>
  <style>
    :root {{ color-scheme: light dark; font-family: Inter, system-ui, sans-serif; }}
    body {{ margin: 0 auto; max-width: 980px; padding: 24px; line-height: 1.5; }}
    .toolbar {{ position: sticky; top: 0; z-index: 3; padding: 16px; border: 1px solid #8885;
      border-radius: 14px; background: Canvas; display: flex; gap: 12px; align-items: center;
      flex-wrap: wrap; }}
    .toolbar input {{ padding: 9px; min-width: 220px; }}
    .toolbar button {{ padding: 10px 16px; font-weight: 700; }}
    .card {{ margin: 20px 0; padding: 20px; border: 1px solid #8885; border-radius: 14px; }}
    .card header {{ display: flex; gap: 12px; align-items: center; }}
    .index {{ font-size: 1.35rem; font-weight: 800; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; padding: 14px; border-radius: 10px;
      background: color-mix(in srgb, CanvasText 7%, Canvas); }}
    .decision {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .decision button {{ padding: 12px; font-weight: 750; border-radius: 9px; }}
    .decision textarea {{ grid-column: 1 / -1; min-height: 64px; padding: 10px; }}
    .card[data-decision="accept"] button[data-value="accept"] {{
      background: #1d8f51; color: white;
    }}
    .card[data-decision="reject"] button[data-value="reject"] {{
      background: #b63434; color: white;
    }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Lumi Aya Bengali — 32-row owner review</h1>
  <p>Review quality, instruction compliance, language, correctness, and formatting.
  No choice is preselected. Export is blocked until all 32 rows have a decision.</p>
  <section class="toolbar">
    <strong id="progress">0 / {len(sample)} reviewed</strong>
    <input id="reviewer" placeholder="Private reviewer alias" value="owner-01">
    <button id="export" type="button">Download decisions JSON</button>
  </section>
  {"".join(cards)}
  <script>
    const recordIds = {ids_json};
    const bindings = {bindings_json};
    const decisions = new Map();
    const progress = document.getElementById('progress');
    function updateProgress() {{
      progress.textContent = `${{decisions.size}} / ${{recordIds.length}} reviewed`;
    }}
    document.querySelectorAll('.card').forEach(card => {{
      card.querySelectorAll('button[data-value]').forEach(button => {{
        button.addEventListener('click', () => {{
          const value = button.dataset.value;
          card.dataset.decision = value;
          decisions.set(card.dataset.recordId, value);
          updateProgress();
        }});
      }});
    }});
    document.getElementById('export').addEventListener('click', () => {{
      if (decisions.size !== recordIds.length) {{
        alert(`Review all ${{recordIds.length}} rows before exporting.`);
        return;
      }}
      const reviewer = document.getElementById('reviewer').value.trim();
      if (!reviewer) {{ alert('Enter a private reviewer alias.'); return; }}
      const rows = recordIds.map(recordId => {{
        const card = document.querySelector(`[data-record-id="${{recordId}}"]`);
        return {{ decision: decisions.get(recordId), note: card.querySelector('textarea').value,
          record_id: recordId }};
      }});
      const payload = {{ bindings, records: rows, reviewer_alias: reviewer,
        schema_version: '{DECISIONS_SCHEMA_VERSION}' }};
      const blob = new Blob(
        [JSON.stringify(payload, null, 2) + '\\n'],
        {{type: 'application/json'}}
      );
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'lumi-aya-owner-review-decisions-v0.1.json';
      link.click();
      URL.revokeObjectURL(link.href);
    }});
  </script>
</body>
</html>
"""


def build_materialization_report(
    *,
    artifacts: Sequence[Mapping[str, Any]],
    candidate_records: int,
    owner_sample_record_ids_sha256: str,
    owner_sample_records: int,
    plan_file_sha256: str,
    supervised_tokens: int,
) -> dict[str, Any]:
    """Build the raw-text-free receipt for exactly three private artifacts."""

    _require_sha256(plan_file_sha256, "plan_file_sha256")
    _require_sha256(owner_sample_record_ids_sha256, "owner_sample_record_ids_sha256")
    if candidate_records != EXPECTED_LIMITS["maximum_candidate_records"]:
        raise ValueError("Aya materialized record count differs from approval")
    if supervised_tokens != EXPECTED_LIMITS["maximum_supervised_tokens"]:
        raise ValueError("Aya materialized token count differs from approval")
    if owner_sample_records != EXPECTED_LIMITS["owner_sample_records"]:
        raise ValueError("Aya materialized owner sample count differs from approval")
    if len(artifacts) != EXPECTED_LIMITS["exact_private_artifacts"]:
        raise ValueError("Aya private artifact count differs from approval")
    normalized_artifacts = []
    for artifact in artifacts:
        digest = _require_sha256(artifact.get("sha256"), "artifact.sha256")
        if not isinstance(artifact.get("bytes"), int) or artifact["bytes"] <= 0:
            raise ValueError("Aya materialization artifact byte count is invalid")
        normalized_artifacts.append({**dict(artifact), "sha256": digest})

    report: dict[str, Any] = {
        "artifacts": sorted(normalized_artifacts, key=lambda item: item["name"]),
        "authorization": dict(EXPECTED_AUTHORIZATION),
        "candidate": {
            "materialized": True,
            "records": candidate_records,
            "supervised_tokens": supervised_tokens,
        },
        "observed_at": "2026-08-06",
        "owner_authorization_sha256": hashlib.sha256(
            EXPECTED_OWNER_STATEMENT.encode("utf-8")
        ).hexdigest(),
        "owner_review": {
            "complete": False,
            "record_count": owner_sample_records,
            "sample_record_ids_sha256": owner_sample_record_ids_sha256,
            "status": "private_packet_ready_owner_decisions_pending",
        },
        "plan_file_sha256": plan_file_sha256,
        "preflight_receipt_sha256": EXPECTED_PREFLIGHT["internal_report_sha256"],
        "production": {
            "candidate_registry_promotion_authorized": False,
            "dataset_seal_authorized": False,
            "production_eligible": False,
            "sft_training_authorized": False,
        },
        "raw_training_text_in_report": False,
        "remaining_blockers": [
            "deterministic_owner_quality_sample_not_reviewed",
            "candidate_registry_promotion_not_authorized",
            "cross_source_deduplication_and_full_10m_to_30m_mix_not_complete",
            "production_dataset_seal_not_authorized",
            "production_sft_not_authorized",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": {
            "dataset_id": DATASET_ID,
            "license_id": SOURCE_LICENSE,
            "revision": SOURCE_REVISION,
            "source_id": SOURCE_ID,
        },
        "status": "private_candidate_materialized_owner_review_pending",
    }
    report["report_sha256"] = sha256_value(report)
    return report


def validate_materialization_report(report: Mapping[str, Any]) -> None:
    """Verify the receipt seal and every fail-closed production flag."""

    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("Aya materialization report schema changed")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    if (
        sha256_value({key: value for key, value in report.items() if key != "report_sha256"})
        != seal
    ):
        raise ValueError("Aya materialization report seal is invalid")
    if report.get("raw_training_text_in_report") is not False:
        raise ValueError("Aya materialization report retained raw text")
    production = report.get("production")
    if not isinstance(production, Mapping) or any(production.values()):
        raise ValueError("Aya materialization crossed a production boundary")
