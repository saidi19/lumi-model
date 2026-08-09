# ruff: noqa: E501
"""Private, browser-based owner review for the Lumi AI-assisted SFT pilot."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_first_party import canonical_json_bytes

DECISIONS_SCHEMA_VERSION = "lumi.ai-assisted-owner-decisions.v0.3"
RECEIPT_SCHEMA_VERSION = "lumi.ai-assisted-owner-review-receipt.v0.3"
_UTC_TIMESTAMP = re.compile(
    r"^20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T"
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z$"
)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _safe_embedded_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def build_owner_review_html(
    sample_rows: list[Mapping[str, Any]],
    *,
    sample_sha256: str,
) -> str:
    """Build a self-contained private review page without network dependencies."""

    if not sample_rows:
        raise ValueError("owner review requires at least one row")
    if _fingerprint(sample_rows) != sample_sha256:
        raise ValueError("owner review sample SHA-256 changed")
    record_ids = [row.get("record_id") for row in sample_rows]
    if any(not isinstance(record_id, str) for record_id in record_ids):
        raise ValueError("owner review sample has an invalid record ID")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("owner review sample record IDs are not unique")

    html = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lumi owner review</title>
  <style>
    :root { color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #0c111b; color: #edf2ff; }
    main { width: min(960px, calc(100% - 28px)); margin: 24px auto 80px; }
    .panel { background: #151d2b; border: 1px solid #2c3950; border-radius: 18px; padding: 20px; box-shadow: 0 16px 50px #0005; }
    .top { display: grid; gap: 14px; grid-template-columns: 1fr auto; align-items: end; }
    h1 { margin: 0 0 6px; font-size: clamp(24px, 5vw, 40px); }
    p { color: #b8c3d9; line-height: 1.5; }
    label { color: #b8c3d9; font-size: 14px; }
    input, textarea { width: 100%; margin-top: 6px; border: 1px solid #3b4962; border-radius: 10px; padding: 11px; background: #0e1522; color: #f7f9ff; font: inherit; }
    .progress { height: 10px; background: #273247; border-radius: 999px; overflow: hidden; }
    .progress > div { height: 100%; width: 0; background: linear-gradient(90deg,#6e7cff,#53d6a6); transition: width .2s; }
    .meta { display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0; }
    .tag { background: #263249; color: #d9e3fa; border-radius: 999px; padding: 6px 10px; font-size: 13px; }
    .message { border-radius: 14px; padding: 14px 16px; margin: 10px 0; white-space: pre-wrap; line-height: 1.5; }
    .message strong { display: block; margin-bottom: 8px; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; opacity: .7; }
    .user { background: #24395c; }
    .assistant { background: #173f37; }
    .tool { background: #4a3420; font-family: ui-monospace, monospace; }
    .tool_result { background: #382d4d; font-family: ui-monospace, monospace; }
    .actions { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-top: 18px; }
    button { border: 0; border-radius: 12px; padding: 13px 16px; font: inherit; font-size: 15px; font-weight: 700; cursor: pointer; }
    button:hover { filter: brightness(1.08); }
    .accept { background: #43c993; color: #071711; }
    .reject { background: #ff6e7d; color: #23070a; }
    .nav { display: flex; gap: 10px; margin-top: 12px; }
    .nav button { flex: 1; background: #2b3850; color: #edf2ff; }
    .download { width: 100%; margin-top: 18px; background: #7787ff; color: white; }
    .status { text-align: right; font-weight: 700; }
    .decision { min-height: 22px; margin-top: 12px; font-weight: 700; }
    .honesty { border-left: 4px solid #f0bc52; padding-left: 12px; color: #f5d896; }
    @media (max-width: 640px) { .top { grid-template-columns: 1fr; } .status { text-align: left; } .actions { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <section class="panel">
    <div class="top">
      <div>
        <h1>Lumi owner spot-check</h1>
        <p>Review all __ROW_COUNT__ rows. Check correctness, naturalness, language, context, and safety.</p>
      </div>
      <div class="status" id="status">0 / __ROW_COUNT__ reviewed</div>
    </div>
    <p class="honesty">These rows are AI-generated and owner-reviewed. Approving them does not make them human-written.</p>
    <label>Private reviewer alias
      <input id="reviewer" autocomplete="off" placeholder="Example: owner-afnan">
    </label>
    <div class="progress" aria-label="review progress"><div id="bar"></div></div>
    <div class="meta" id="meta"></div>
    <div id="messages"></div>
    <label>Optional note for this row
      <textarea id="note" rows="3" placeholder="Why you accepted/rejected it, or what should change"></textarea>
    </label>
    <div class="decision" id="decision"></div>
    <div class="actions">
      <button class="accept" id="accept">Accept this row (A)</button>
      <button class="reject" id="reject">Reject this row (R)</button>
    </div>
    <div class="nav">
      <button id="previous">Previous</button>
      <button id="next">Next</button>
    </div>
    <button class="download" id="download">Download signed decisions JSON</button>
  </section>
</main>
<script>
const rows = __ROWS_JSON__;
const sampleSha256 = "__SAMPLE_SHA256__";
const storageKey = `lumi-owner-review-${sampleSha256}`;
let state = JSON.parse(localStorage.getItem(storageKey) || '{"index":0,"decisions":{},"reviewer":""}');
const byId = id => document.getElementById(id);
const save = () => localStorage.setItem(storageKey, JSON.stringify(state));
const escapeHtml = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function render() {
  state.index = Math.max(0, Math.min(rows.length - 1, state.index));
  const row = rows[state.index];
  const decision = state.decisions[row.record_id];
  const reviewed = Object.keys(state.decisions).length;
  byId('status').textContent = `${reviewed} / ${rows.length} reviewed`;
  byId('bar').style.width = `${reviewed / rows.length * 100}%`;
  byId('reviewer').value = state.reviewer || '';
  byId('meta').innerHTML = [
    `${state.index + 1} of ${rows.length}`,
    row.skill_id,
    row.language_mode,
    row.split,
    row.record_id
  ].map(item => `<span class="tag">${escapeHtml(item)}</span>`).join('');
  byId('messages').innerHTML = row.messages.map(message =>
    `<div class="message ${escapeHtml(message.role)}"><strong>${escapeHtml(message.role)}</strong>${escapeHtml(message.content)}</div>`
  ).join('');
  byId('note').value = decision?.note || '';
  byId('decision').textContent = decision ? (decision.accepted ? 'Accepted ✓' : 'Rejected ✕') : 'Not reviewed yet';
  byId('decision').style.color = decision ? (decision.accepted ? '#58e3ad' : '#ff8793') : '#b8c3d9';
}

function decide(accepted) {
  const row = rows[state.index];
  state.decisions[row.record_id] = {accepted, note: byId('note').value.trim()};
  save();
  if (state.index < rows.length - 1) state.index += 1;
  save();
  render();
}

byId('reviewer').addEventListener('input', event => { state.reviewer = event.target.value.trim(); save(); });
byId('note').addEventListener('input', event => {
  const row = rows[state.index];
  if (state.decisions[row.record_id]) {
    state.decisions[row.record_id].note = event.target.value;
    save();
  }
});
byId('accept').onclick = () => decide(true);
byId('reject').onclick = () => decide(false);
byId('previous').onclick = () => { state.index -= 1; save(); render(); };
byId('next').onclick = () => { state.index += 1; save(); render(); };
document.addEventListener('keydown', event => {
  if (event.target.matches('input,textarea')) return;
  if (event.key.toLowerCase() === 'a') decide(true);
  if (event.key.toLowerCase() === 'r') decide(false);
});
byId('download').onclick = () => {
  if (!state.reviewer) return alert('Enter your private reviewer alias first.');
  if (Object.keys(state.decisions).length !== rows.length) return alert(`Review all ${rows.length} rows first.`);
  const decisions = rows.map(row => ({record_id: row.record_id, ...state.decisions[row.record_id]}));
  const payload = {
    schema_version: 'lumi.ai-assisted-owner-decisions.v0.3',
    sample_sha256: sampleSha256,
    reviewer_id: state.reviewer,
    reviewed_at: new Date().toISOString().replace(/\.\d{3}Z$/, 'Z'),
    decisions
  };
  const blob = new Blob([JSON.stringify(payload, null, 2) + '\n'], {type: 'application/json'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'lumi-owner-review-decisions-v0.3.json';
  link.click();
  URL.revokeObjectURL(link.href);
};
render();
</script>
</body>
</html>
"""
    return (
        html.replace("__ROWS_JSON__", _safe_embedded_json(sample_rows))
        .replace("__SAMPLE_SHA256__", sample_sha256)
        .replace("__ROW_COUNT__", str(len(sample_rows)))
    )


def validate_owner_decisions(
    value: Mapping[str, Any],
    *,
    sample_rows: list[Mapping[str, Any]],
    sample_sha256: str,
) -> dict[str, Any]:
    """Validate a browser-exported decision file and return a raw-text-free receipt."""

    if value.get("schema_version") != DECISIONS_SCHEMA_VERSION:
        raise ValueError("unsupported owner decision schema")
    if value.get("sample_sha256") != sample_sha256:
        raise ValueError("owner decision sample SHA-256 changed")
    if _fingerprint(sample_rows) != sample_sha256:
        raise ValueError("owner review sample content changed")
    reviewer = value.get("reviewer_id")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("owner reviewer alias is required")
    if reviewer in {
        "openai-codex-current-session",
        "lumi-deterministic-ai-assisted-guard-v0.3",
    }:
        raise ValueError("an AI generator or automated guard cannot claim owner review")
    reviewed_at = value.get("reviewed_at")
    if not isinstance(reviewed_at, str) or _UTC_TIMESTAMP.fullmatch(reviewed_at) is None:
        raise ValueError("owner reviewed_at must be UTC ISO-8601 without fractions")

    decisions = value.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(sample_rows):
        raise ValueError("owner decisions must cover all sampled rows")
    expected_ids = {str(row["record_id"]) for row in sample_rows}
    observed_ids: set[str] = set()
    accepted_ids: list[str] = []
    rejected_ids: list[str] = []
    for decision in decisions:
        if not isinstance(decision, Mapping) or set(decision) != {
            "accepted",
            "note",
            "record_id",
        }:
            raise ValueError("owner decision fields changed")
        record_id = decision.get("record_id")
        if not isinstance(record_id, str) or record_id not in expected_ids:
            raise ValueError("owner decision contains an unknown record ID")
        if record_id in observed_ids:
            raise ValueError("owner decision contains a duplicate record ID")
        observed_ids.add(record_id)
        accepted = decision.get("accepted")
        if not isinstance(accepted, bool):
            raise ValueError("owner decision accepted value must be boolean")
        note = decision.get("note")
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError("owner decision note is invalid")
        (accepted_ids if accepted else rejected_ids).append(record_id)
    if observed_ids != expected_ids:
        raise ValueError("owner decisions do not match the sampled record IDs")

    full_candidate_review = len(sample_rows) == 320
    review_scope = "full_320_row_candidate" if full_candidate_review else "sample"
    if rejected_ids:
        status = "owner_review_rejected_rows_require_repair"
    elif full_candidate_review:
        status = "owner_full_320_row_review_passed_pilot_still_not_production_dataset"
    else:
        status = "owner_spot_check_passed_pilot_still_not_production_dataset"
    receipt: dict[str, Any] = {
        "accepted_records": len(accepted_ids),
        "all_sampled_rows_accepted": not rejected_ids,
        "authorization": {
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
        },
        "decision_file_sha256": _fingerprint(value),
        "owner_review_claim": "ai_generated_plus_owner_reviewed_not_human_written",
        "production_eligible": False,
        "rejected_record_ids": sorted(rejected_ids),
        "rejected_records": len(rejected_ids),
        "reviewed_at": reviewed_at,
        "reviewer_id": reviewer,
        "review_scope": review_scope,
        "sample_records": len(sample_rows),
        "sample_sha256": sample_sha256,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": status,
    }
    receipt["receipt_sha256"] = _fingerprint(receipt)
    return receipt


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    values = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at line {line_number}")
            values.append(value)
    return values
