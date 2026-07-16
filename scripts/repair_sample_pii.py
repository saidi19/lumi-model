#!/usr/bin/env python3
"""Repair residual PII in a sealed sample while preserving a safe validation checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumi.data_pipeline import (
    PII_DETECTOR_VERSION,
    _atomic_json_write,
    _canonical_json_bytes,
    _category_targets,
    _manifest_from_run,
    _scan_records,
    _sha256_bytes,
    _sha256_file,
    _validation_fingerprint,
    detect_pii,
    filter_document,
)
from lumi.data_quality import DataQualityGate, QualityEvaluation
from lumi.development_config import compose_development_config

REPOSITORY_ROOT = Path(__file__).parents[1]
DEFAULT_PLAN = REPOSITORY_ROOT / "configs" / "tokenizer-sample-v0.3.json"
SOURCE_PII_DETECTOR = "contextual-v4"


@dataclass(frozen=True)
class PendingRepair:
    added_redactions: dict[str, int]
    document_id: str
    line_number: int
    new_content_sha256: str
    new_evaluation: QualityEvaluation
    new_text_bytes: int
    old_content_sha256: str
    old_evaluation: QualityEvaluation
    old_text_bytes: int


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _run_fingerprint(config_sha256: str, target_bytes: int) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "config_sha256": config_sha256,
                "pii_detector": PII_DETECTOR_VERSION,
                "target_bytes": target_bytes,
            }
        )
    )


def _json_document_sha256(value: dict[str, Any]) -> str:
    serialized = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return _sha256_bytes(serialized)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _open_checkpoint(
    output: Path, manifest: dict[str, Any]
) -> tuple[sqlite3.Connection, dict[str, Any]]:
    path = output / "validation-state.sqlite3"
    if not path.is_file():
        raise FileNotFoundError(f"validation checkpoint is missing: {path}")
    connection = sqlite3.connect(path)
    row = connection.execute(
        "SELECT fingerprint, status, next_offset, line_number, statistics, updated_at "
        "FROM validation_progress WHERE id = 1"
    ).fetchone()
    if row is None:
        connection.close()
        raise ValueError("validation checkpoint has no progress row")
    checkpoint = {
        "fingerprint": str(row[0]),
        "status": str(row[1]),
        "next_offset": int(row[2]),
        "line_number": int(row[3]),
        "statistics": json.loads(row[4]),
        "updated_at": str(row[5]),
    }
    if checkpoint["fingerprint"] != _validation_fingerprint(manifest):
        connection.close()
        raise ValueError("validation checkpoint fingerprint does not match the old manifest")
    if checkpoint["status"] != "validating":
        connection.close()
        raise ValueError("repair requires an incomplete validating checkpoint")
    document_count = int(
        connection.execute("SELECT COUNT(*) FROM validation_document_ids").fetchone()[0]
    )
    content_count = int(
        connection.execute("SELECT COUNT(*) FROM validation_content_hashes").fetchone()[0]
    )
    quality_count = int(
        connection.execute("SELECT COUNT(*) FROM validation_near_duplicate_signatures").fetchone()[
            0
        ]
    )
    if {document_count, content_count, quality_count} != {checkpoint["line_number"]}:
        connection.close()
        raise ValueError("validation checkpoint tables do not match its line number")
    return connection, checkpoint


def _validate_audit(
    audit: dict[str, Any],
    *,
    checkpoint: dict[str, Any],
    manifest: dict[str, Any],
    sample_path: Path,
) -> dict[int, dict[str, Any]]:
    if audit.get("pii_detector") != PII_DETECTOR_VERSION:
        raise ValueError("audit was not produced by the current PII detector")
    if audit.get("sample_file_bytes") != sample_path.stat().st_size:
        raise ValueError("audit sample size does not match the current sample")
    if audit.get("start_line") != checkpoint["line_number"]:
        raise ValueError("audit start line does not match the validation checkpoint")
    if audit.get("start_offset") != checkpoint["next_offset"]:
        raise ValueError("audit start offset does not match the validation checkpoint")
    if audit.get("scanned_bytes") != sample_path.stat().st_size - checkpoint["next_offset"]:
        raise ValueError("audit did not reach the end of the sample")
    if (
        int(audit.get("scanned_records", -1)) + checkpoint["line_number"]
        != manifest["statistics"]["documents"]
    ):
        raise ValueError("audit record count does not reach the manifest document count")
    matches = audit.get("matches")
    if not isinstance(matches, list) or not matches:
        raise ValueError("repair requires at least one audited residual PII record")
    by_line: dict[int, dict[str, Any]] = {}
    for match in matches:
        if not isinstance(match, dict):
            raise ValueError("audit match entries must be objects")
        line_number = int(match.get("line_number", 0))
        if line_number <= checkpoint["line_number"]:
            raise ValueError("audit would invalidate the committed validation prefix")
        if line_number in by_line:
            raise ValueError(f"audit repeats sample line {line_number}")
        by_line[line_number] = match
    if dict(sorted(Counter(item["kind"] for item in matches).items())) != audit.get("kinds"):
        raise ValueError("audit kind totals do not match its residual records")
    return by_line


def _repair_record(
    record: dict[str, Any],
    *,
    line_number: int,
    audit_match: dict[str, Any],
    filters: dict[str, Any],
    quality_gate: DataQualityGate,
) -> tuple[dict[str, Any], PendingRepair]:
    document_id = str(record["id"])
    if document_id != audit_match.get("document_id"):
        raise ValueError(f"audit document ID mismatch at line {line_number}")
    old_text = record["text"]
    if not isinstance(old_text, str):
        raise ValueError(f"sample text is not a string at line {line_number}")
    old_pii = detect_pii(old_text)
    if old_pii != audit_match.get("kind"):
        raise ValueError(f"audited PII kind no longer matches line {line_number}")

    category = record["provenance"]["category"]
    old_evaluation = quality_gate.evaluate(old_text, category, exclude_document_id=document_id)
    if old_evaluation.rejection_reason is not None:
        raise ValueError(
            f"existing quality gate {old_evaluation.rejection_reason} failed at line {line_number}"
        )
    if old_evaluation.metadata != record.get("quality"):
        raise ValueError(f"existing quality metadata mismatch at line {line_number}")

    filtered = filter_document(old_text, filters)
    if filtered.rejection_reason is not None or filtered.text is None:
        raise ValueError(f"PII repair rejected line {line_number}: {filtered.rejection_reason}")
    if filtered.text == old_text or detect_pii(filtered.text) is not None:
        raise ValueError(f"PII repair did not produce clean text at line {line_number}")
    new_evaluation = quality_gate.evaluate(filtered.text, category, exclude_document_id=document_id)
    if new_evaluation.rejection_reason is not None or new_evaluation.metadata is None:
        raise ValueError(
            f"repaired quality gate {new_evaluation.rejection_reason} failed at line {line_number}"
        )

    repaired = deepcopy(record)
    new_text_bytes = filtered.text.encode("utf-8")
    repaired["text"] = filtered.text
    repaired["content_sha256"] = _sha256_bytes(new_text_bytes)
    repaired["quality"] = new_evaluation.metadata
    transformations = repaired.setdefault("transformations", {})
    redactions = Counter(transformations.get("pii_redactions", {}))
    redactions.update(filtered.redactions)
    transformations["pii_redactions"] = dict(sorted(redactions.items()))
    pending = PendingRepair(
        added_redactions=dict(sorted(filtered.redactions.items())),
        document_id=document_id,
        line_number=line_number,
        new_content_sha256=repaired["content_sha256"],
        new_evaluation=new_evaluation,
        new_text_bytes=len(new_text_bytes),
        old_content_sha256=str(record["content_sha256"]),
        old_evaluation=old_evaluation,
        old_text_bytes=len(old_text.encode("utf-8")),
    )
    return repaired, pending


def _rewrite_sample(
    *,
    sample_path: Path,
    temporary_path: Path,
    matches: dict[int, dict[str, Any]],
    checkpoint: dict[str, Any],
    filters: dict[str, Any],
    quality_gate: DataQualityGate,
) -> list[PendingRepair]:
    pending: list[PendingRepair] = []
    seen_matches: set[int] = set()
    line_number = 0
    with sample_path.open("rb") as source, temporary_path.open("xb") as destination:
        while raw_line := source.readline():
            line_number += 1
            audit_match = matches.get(line_number)
            if audit_match is None:
                output_line = raw_line
            else:
                try:
                    record = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(f"invalid JSON at repair line {line_number}") from error
                repaired, repair = _repair_record(
                    record,
                    line_number=line_number,
                    audit_match=audit_match,
                    filters=filters,
                    quality_gate=quality_gate,
                )
                output_line = (
                    json.dumps(
                        repaired,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
                pending.append(repair)
                seen_matches.add(line_number)
            destination.write(output_line)
            if line_number == checkpoint["line_number"]:
                if source.tell() != checkpoint["next_offset"]:
                    raise ValueError("source checkpoint offset changed before repair")
                if destination.tell() != checkpoint["next_offset"]:
                    raise ValueError("repair changed bytes inside the validated prefix")
            if line_number % 100_000 == 0:
                print(f"rewrite line={line_number:,}", flush=True)
        destination.flush()
        os.fsync(destination.fileno())
    if seen_matches != set(matches):
        missing = sorted(set(matches) - seen_matches)
        raise ValueError(f"audited repair lines were not found: {missing}")
    return pending


def _restore_file_from_hard_link(backup: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".restore.tmp")
    if temporary.exists():
        temporary.unlink()
    os.link(backup, temporary)
    os.replace(temporary, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="atomically install the prepared repair and update durable indexes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    sample_path = output / "sample.jsonl"
    manifest_path = output / "manifest.json"
    state_path = output / "run-state.json"
    quality_path = output / "build-quality-state.sqlite3"
    for required in (sample_path, manifest_path, state_path, quality_path, args.audit):
        if not required.is_file():
            raise FileNotFoundError(f"required repair input is missing: {required}")

    manifest = _load_json(manifest_path)
    state = _load_json(state_path)
    audit = _load_json(args.audit)
    if manifest.get("implementation", {}).get("pii_detector") != SOURCE_PII_DETECTOR:
        raise ValueError("sample is not sealed with the expected source PII detector")
    if state.get("implementation", {}).get("pii_detector") != SOURCE_PII_DETECTOR:
        raise ValueError("run state is not using the expected source PII detector")
    if state.get("status") != "failed" or state.get("failure", {}).get("type") != "ValueError":
        raise ValueError("repair requires the safely failed validation run state")
    if not str(state.get("failure", {}).get("message", "")).startswith("PII pattern "):
        raise ValueError("repair requires a residual-PII validation failure")

    config = compose_development_config(args.plan, repository_root=REPOSITORY_ROOT)
    config_sha256 = _sha256_bytes(_canonical_json_bytes(config))
    if config_sha256 != manifest.get("config_sha256") or config_sha256 != state.get(
        "config_sha256"
    ):
        raise ValueError("repair plan does not match the sample config hash")
    if sample_path.stat().st_size != manifest["output"]["file_bytes"]:
        raise ValueError("sample byte size does not match the old manifest")
    print("verifying old sample SHA-256", flush=True)
    old_sample_sha256 = _sha256_file(sample_path)
    if old_sample_sha256 != manifest["output"]["sha256"]:
        raise ValueError("sample SHA-256 does not match the old manifest")

    validation_connection, checkpoint = _open_checkpoint(output, manifest)
    matches = _validate_audit(
        audit,
        checkpoint=checkpoint,
        manifest=manifest,
        sample_path=sample_path,
    )
    quality_gate = DataQualityGate(
        manifest["quality"],
        REPOSITORY_ROOT,
        near_duplicate_path=quality_path,
    )
    if quality_gate.near_duplicate.count() != manifest["statistics"]["documents"]:
        quality_gate.close()
        validation_connection.close()
        raise ValueError("build quality index does not cover the complete sample")

    temporary_sample = output / "sample.jsonl.contextual-v5.tmp"
    if temporary_sample.exists():
        quality_gate.close()
        validation_connection.close()
        raise FileExistsError(f"temporary repair output already exists: {temporary_sample}")
    pending: list[PendingRepair] = []
    try:
        print(f"rewriting {len(matches)} audited records", flush=True)
        pending = _rewrite_sample(
            sample_path=sample_path,
            temporary_path=temporary_sample,
            matches=matches,
            checkpoint=checkpoint,
            filters=manifest["filters"],
            quality_gate=quality_gate,
        )
        print("scanning repaired sample statistics and uniqueness", flush=True)
        stats = _scan_records(temporary_sample, config)
        if stats["documents"] != manifest["statistics"]["documents"]:
            raise ValueError("repair changed the sample document count")

        target_bytes = int(state["target_uncompressed_text_bytes"])
        new_run_fingerprint = _run_fingerprint(config_sha256, target_bytes)
        new_state = deepcopy(state)
        new_state.pop("failure", None)
        new_state["implementation"] = {"pii_detector": PII_DETECTOR_VERSION}
        new_state["language_labels"] = dict(sorted(stats["language_labels"].items()))
        new_state["pii_redactions"] = dict(sorted(stats["pii_redactions"].items()))
        new_state["progress"] = {
            "documents": int(stats["documents"]),
            "text_utf8_bytes": int(stats["text_bytes"]),
        }
        new_state["run_fingerprint"] = new_run_fingerprint
        new_state["status"] = "validating"
        new_state["updated_at"] = _utc_now()
        new_manifest = _manifest_from_run(
            config=config,
            config_sha256=config_sha256,
            run_fingerprint=new_run_fingerprint,
            sample_path=temporary_sample,
            targets=_category_targets(config, target_bytes),
            stats=stats,
            state=new_state,
        )
        new_manifest["output"]["file"] = sample_path.name
        migration = {
            "from_pii_detector": SOURCE_PII_DETECTOR,
            "from_sample_sha256": old_sample_sha256,
            "reason": "fixed-point contextual PII redaction after resumable validation audit",
            "repaired_documents": len(pending),
            "to_pii_detector": PII_DETECTOR_VERSION,
            "to_sample_sha256": new_manifest["output"]["sha256"],
            "validation_checkpoint_line": checkpoint["line_number"],
            "verified_at": _utc_now(),
        }
        new_state.setdefault("migrations", []).append(migration)
        new_state["quality_repairs"] = {
            "documents": len(pending),
            "status": "complete",
        }
        new_state["manifest_sha256"] = _json_document_sha256(new_manifest)
        new_validation_fingerprint = _validation_fingerprint(new_manifest)
        repair_summary = {
            "added_redactions": dict(
                sorted(
                    sum(
                        (Counter(repair.added_redactions) for repair in pending),
                        Counter(),
                    ).items()
                )
            ),
            "from_pii_detector": SOURCE_PII_DETECTOR,
            "from_sample_sha256": old_sample_sha256,
            "new_file_bytes": new_manifest["output"]["file_bytes"],
            "new_text_utf8_bytes": new_manifest["output"]["text_utf8_bytes"],
            "old_file_bytes": manifest["output"]["file_bytes"],
            "old_text_utf8_bytes": manifest["output"]["text_utf8_bytes"],
            "repaired_records": [
                {
                    "document_id": repair.document_id,
                    "line_number": repair.line_number,
                    "new_content_sha256": repair.new_content_sha256,
                    "old_content_sha256": repair.old_content_sha256,
                    "text_byte_delta": repair.new_text_bytes - repair.old_text_bytes,
                }
                for repair in pending
            ],
            "status": "prepared",
            "to_pii_detector": PII_DETECTOR_VERSION,
            "to_sample_sha256": new_manifest["output"]["sha256"],
            "validation_checkpoint_line": checkpoint["line_number"],
        }
        if not args.apply:
            print("PASS: residual-PII repair dry run is compatible")
            print(json.dumps(repair_summary, indent=2, sort_keys=True))
            temporary_sample.unlink()
            return

        backups = {
            "audit": output / "pii-audit.pre-contextual-v5.json",
            "manifest": output / "manifest.pre-contextual-v5.json",
            "sample": output / "sample.pre-contextual-v5.jsonl",
            "state": output / "run-state.pre-contextual-v5.json",
        }
        report_path = output / "pii-repair-contextual-v5.json"
        if report_path.exists() or any(path.exists() for path in backups.values()):
            raise FileExistsError("contextual-v5 repair backup or report already exists")
        shutil.copy2(args.audit, backups["audit"])
        shutil.copy2(manifest_path, backups["manifest"])
        shutil.copy2(state_path, backups["state"])
        os.link(sample_path, backups["sample"])
        _atomic_json_write(report_path, repair_summary)

        quality_committed = False
        validation_committed = False
        cutover_started = False
        try:
            for repair in pending:
                quality_gate.replace(repair.document_id, repair.new_evaluation)
            if quality_gate.near_duplicate.count() != stats["documents"]:
                raise ValueError("quality index count changed during repair")
            os.replace(temporary_sample, sample_path)
            cutover_started = True
            _atomic_json_write(manifest_path, new_manifest)
            _atomic_json_write(state_path, new_state)
            validation_connection.execute(
                "UPDATE validation_progress SET fingerprint = ?, updated_at = ? WHERE id = 1",
                (new_validation_fingerprint, _utc_now()),
            )
            validation_connection.commit()
            validation_committed = True
            quality_gate.flush()
            quality_committed = True
            repair_summary["status"] = "applied"
            repair_summary["completed_at"] = _utc_now()
            repair_summary["backups"] = {name: str(path) for name, path in sorted(backups.items())}
            _atomic_json_write(report_path, repair_summary)
        except BaseException:
            quality_gate.near_duplicate.connection.rollback()
            validation_connection.rollback()
            if quality_committed:
                for repair in pending:
                    quality_gate.replace(repair.document_id, repair.old_evaluation)
                quality_gate.flush()
            if validation_committed:
                validation_connection.execute(
                    "UPDATE validation_progress SET fingerprint = ?, updated_at = ? WHERE id = 1",
                    (checkpoint["fingerprint"], _utc_now()),
                )
                validation_connection.commit()
            if cutover_started:
                _restore_file_from_hard_link(backups["sample"], sample_path)
                shutil.copy2(backups["manifest"], manifest_path)
                shutil.copy2(backups["state"], state_path)
            repair_summary["status"] = "rolled_back"
            repair_summary["completed_at"] = _utc_now()
            _atomic_json_write(report_path, repair_summary)
            raise

        print("PASS: residual-PII repair applied")
        print(json.dumps(repair_summary, indent=2, sort_keys=True))
    finally:
        if temporary_sample.exists():
            temporary_sample.unlink()
        quality_gate.close()
        validation_connection.close()


if __name__ == "__main__":
    main()
