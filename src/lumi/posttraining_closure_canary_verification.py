"""Independent deterministic verification for the Lumi SFT closure canary."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .data_quality import BenchmarkRegistry
from .posttraining_closure_canary_curation import (
    BENCHMARK_SHA256,
    CURRENT_ACCEPTED_TRAIN_TOKENS,
    MAXIMUM_AUXILIARY_TRAIN_TOKENS,
    MAXIMUM_GENERAL_TRAIN_TOKENS,
    MAXIMUM_PRIVATE_WRITES,
    MAXIMUM_TRAIN_TOKENS,
    PLAN_SHA256,
    PYARROW_VERSION,
    PYARROW_WHEEL_SHA256,
    RAW_OBJECTS,
    SOURCE_SPECS,
    TEN_MILLION_TOKENS,
    TOKENIZER_SHA256,
    VALIDATION_DENOMINATOR,
    _load_reservoirs,
    _measure_reservoirs,
    _owner_sample,
    _select_records,
    _summarize_lengths,
    _verify_raw_objects,
    file_sha256,
    load_direct_tokenizer,
    sha256_value,
)
from .posttraining_closure_canary_curation import (
    SCHEMA_VERSION as CURATION_SCHEMA_VERSION,
)
from .posttraining_data import load_sft_policy
from .posttraining_format import load_chat_template

SCHEMA_VERSION = "lumi.sft-closure-canary-curation-verification.v0.1"
OVERLAY_PLAN_FILE_SHA256 = (
    "298e42f4a3f0c23331892278406a1b7cdd84f676a7b4864e61ecf9ce02545561"
)
OVERLAY_PLAN_INTERNAL_SHA256 = (
    "4990004157a8fef8c0c85afe0da9ed768078a09205c58cf7adee86868b6c5b9b"
)
PYARROW_WHEEL_BYTES = 35_844_549
MAXIMUM_OVERLAY_PRIVATE_WRITES = 268_435_456


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_internal_sha(value: Mapping[str, Any], field: str) -> None:
    claimed = value.get(field)
    unsigned = {key: item for key, item in value.items() if key != field}
    _require(claimed == sha256_value(unsigned), f"{field} mismatch")


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    _require(candidate.is_relative_to(root), "private manifest path escaped root")
    return candidate


def _record_identity(record: Mapping[str, Any]) -> tuple[str, str, str]:
    source = record["source"]
    return (
        str(source["source_id"]),
        str(record["split"]),
        str(record["record_id"]),
    )


def _verify_record_binding(record: Mapping[str, Any]) -> None:
    source = record["source"]
    source_id = str(source["source_id"])
    matching = [spec for spec in SOURCE_SPECS.values() if spec["source_id"] == source_id]
    _require(len(matching) == 1, "candidate source_id is not bound")
    spec = matching[0]
    for key in ("dataset_id", "license_id", "revision", "uri"):
        _require(source.get(key) == spec[key], f"candidate source {key} changed")
    document_id = str(source["document_id"])
    document_digest = hashlib.sha256(document_id.encode()).hexdigest()[:24]
    expected_id = f"closure-{source_id}-{document_digest}"
    _require(record["record_id"] == expected_id, "candidate record_id changed")
    split_digest = hashlib.sha256(
        f"closure-split-v0.1\0{expected_id}".encode()
    ).digest()
    expected_split = (
        "validation"
        if int.from_bytes(split_digest[:8], "big") % VALIDATION_DENOMINATOR == 0
        else "train"
    )
    _require(record["split"] == expected_split, "candidate split changed")
    expected_content = sha256_value(
        {
            "language": record["language"],
            "messages": record["messages"],
            "task_category": record["task_category"],
        }
    )
    _require(
        record["provenance"]["content_sha256"] == expected_content,
        "candidate content binding changed",
    )


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def verify_curation(
    *,
    repository_root: str | Path,
    raw_root: str | Path,
    private_root: str | Path,
    curation_report_path: str | Path,
    plan_path: str | Path,
    tokenizer_path: str | Path,
    chat_template_path: str | Path,
    data_contract_path: str | Path,
    data_quality_path: str | Path,
    overlay_plan_path: str | Path,
    overlay_root: str | Path,
    wheel_path: str | Path,
) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    raw = Path(raw_root).resolve()
    private = Path(private_root).resolve()
    report_path = Path(curation_report_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("schema_version") == CURATION_SCHEMA_VERSION, "report schema")
    _verify_internal_sha(report, "report_sha256")

    plan = Path(plan_path).resolve()
    tokenizer_file = Path(tokenizer_path).resolve()
    quality_file = Path(data_quality_path).resolve()
    template_file = Path(chat_template_path).resolve()
    contract_file = Path(data_contract_path).resolve()
    _require(file_sha256(plan) == PLAN_SHA256, "acquisition plan changed")
    _require(file_sha256(tokenizer_file) == TOKENIZER_SHA256, "tokenizer changed")
    bindings = report["bindings"]
    _require(file_sha256(quality_file) == bindings["data_quality_sha256"], "quality")
    _require(file_sha256(template_file) == bindings["chat_template_sha256"], "template")
    _require(file_sha256(contract_file) == bindings["data_contract_sha256"], "contract")

    overlay_plan_file = Path(overlay_plan_path).resolve()
    overlay_plan = json.loads(overlay_plan_file.read_text(encoding="utf-8"))
    _require(
        file_sha256(overlay_plan_file) == OVERLAY_PLAN_FILE_SHA256,
        "overlay plan file changed",
    )
    _verify_internal_sha(overlay_plan, "plan_sha256")
    _require(
        overlay_plan["plan_sha256"] == OVERLAY_PLAN_INTERNAL_SHA256,
        "overlay plan internal binding changed",
    )
    wheel = Path(wheel_path).resolve()
    _require(wheel.stat().st_size == PYARROW_WHEEL_BYTES, "PyArrow wheel size changed")
    _require(file_sha256(wheel) == PYARROW_WHEEL_SHA256, "PyArrow wheel changed")
    overlay = Path(overlay_root).resolve()
    overlay_bytes = _directory_bytes(overlay) + wheel.stat().st_size
    _require(overlay_bytes <= MAXIMUM_OVERLAY_PRIVATE_WRITES, "overlay write ceiling")
    import pyarrow

    _require(pyarrow.__version__ == PYARROW_VERSION, "PyArrow version changed")

    raw_inventory = _verify_raw_objects(raw)
    _require(raw_inventory == report["raw_object_inventory"], "raw inventory replay")
    raw_bytes = sum(value[0] for value in RAW_OBJECTS.values())

    quality = json.loads(quality_file.read_text(encoding="utf-8"))
    _require(
        quality["benchmark_decontamination"]["registry_sha256"] == BENCHMARK_SHA256,
        "benchmark registry config changed",
    )
    tokenizer = load_direct_tokenizer(tokenizer_file)
    template = load_chat_template(template_file)
    policy = load_sft_policy(contract_file)
    benchmark = BenchmarkRegistry(quality["benchmark_decontamination"], repository)

    reservoirs, source_gate = _load_reservoirs(raw)
    _require(source_gate == report["source_gate"], "source-gate replay mismatch")
    measured, machine_gate = _measure_reservoirs(
        reservoirs,
        policy=policy,
        benchmark=benchmark,
        tokenizer=tokenizer,
        template=template,
    )
    _require(machine_gate == report["machine_gate"], "machine-gate replay mismatch")
    replay_selected, replay_dedup = _select_records(measured)
    _require(replay_dedup == report["deduplication"], "selection replay mismatch")
    replay_by_id = {str(item.record["record_id"]): item for item in replay_selected}
    _require(len(replay_by_id) == len(replay_selected), "replay record_id collision")

    manifest_path = private / "private-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        file_sha256(manifest_path) == bindings["private_manifest_file_sha256"],
        "private manifest file changed",
    )
    _verify_internal_sha(manifest, "manifest_sha256")
    _require(
        manifest["manifest_sha256"] == bindings["private_manifest_internal_sha256"],
        "private manifest internal binding changed",
    )

    observed_ids: set[str] = set()
    observed_by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    observed_source_train = Counter()
    token_ledger: list[dict[str, Any]] = []
    inventories: list[dict[str, Any]] = []
    auxiliary_train_tokens = 0
    for expected_inventory in manifest["candidate_files"]:
        candidate_path = _safe_child(private, str(expected_inventory["file"]))
        _require(candidate_path.is_file(), "candidate file missing")
        _require(
            candidate_path.stat().st_size == expected_inventory["bytes"],
            "candidate byte count changed",
        )
        _require(file_sha256(candidate_path) == expected_inventory["sha256"], "candidate hash")
        source_id = str(expected_inventory["source_id"])
        source_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
        records = 0
        with candidate_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"candidate JSON invalid at line {line_number}"
                    ) from error
                record_id = str(record["record_id"])
                _require(record_id not in observed_ids, "candidate record_id duplicated")
                observed_ids.add(record_id)
                _require(record_id in replay_by_id, "candidate absent from deterministic replay")
                replay = replay_by_id[record_id]
                _require(record == replay.record, "candidate record bytes changed semantically")
                _require(record["source"]["source_id"] == source_id, "candidate source")
                _verify_record_binding(record)
                split = str(record["split"])
                source_split[split]["records"] += 1
                source_split[split]["serialized_tokens"] += replay.serialized_tokens
                source_split[split]["supervised_tokens"] += replay.supervised_tokens
                observed_by_split[split]["records"] += 1
                observed_by_split[split]["serialized_tokens"] += replay.serialized_tokens
                observed_by_split[split]["supervised_tokens"] += replay.supervised_tokens
                if split == "train":
                    observed_source_train[source_id] += replay.supervised_tokens
                    if record["provenance"].get("auxiliary_transliteration_only") is True:
                        auxiliary_train_tokens += replay.supervised_tokens
                token_ledger.append(
                    {
                        "record_id": record_id,
                        "serialized_tokens": replay.serialized_tokens,
                        "split": split,
                        "supervised_tokens": replay.supervised_tokens,
                    }
                )
                records += 1
        inventory = {
            "bytes": candidate_path.stat().st_size,
            "file": str(candidate_path.relative_to(private)),
            "records": records,
            "sha256": file_sha256(candidate_path),
            "source_id": source_id,
            "splits": {key: dict(value) for key, value in sorted(source_split.items())},
        }
        _require(inventory == expected_inventory, "candidate inventory mismatch")
        inventories.append(inventory)

    _require(observed_ids == set(replay_by_id), "candidate set differs from replay")
    _require(inventories == manifest["candidate_files"], "manifest inventory mismatch")
    ledger = {
        "entries": len(token_ledger),
        "ledger_sha256": sha256_value(
            sorted(token_ledger, key=lambda value: value["record_id"])
        ),
    }
    _require(ledger == manifest["token_ledger"], "token ledger mismatch")

    candidate = report["candidate"]
    by_split = {key: dict(value) for key, value in sorted(observed_by_split.items())}
    _require(by_split == candidate["by_split"], "split totals mismatch")
    train_tokens = observed_by_split["train"]["supervised_tokens"]
    validation_tokens = observed_by_split["validation"]["supervised_tokens"]
    general_train_tokens = train_tokens - auxiliary_train_tokens
    _require(train_tokens == candidate["train_supervised_tokens"], "train tokens")
    _require(validation_tokens == candidate["validation_supervised_tokens"], "validation")
    _require(
        general_train_tokens == candidate["general_train_supervised_tokens"],
        "general train tokens",
    )
    _require(
        auxiliary_train_tokens == candidate["auxiliary_train_supervised_tokens"],
        "auxiliary train tokens",
    )
    _require(
        dict(sorted(observed_source_train.items()))
        == candidate["source_train_supervised_tokens"],
        "source totals",
    )
    _require(len(replay_selected) == candidate["records"], "candidate record total")
    _require(_summarize_lengths(replay_selected) == candidate["serialized_length"], "lengths")
    _require(train_tokens <= MAXIMUM_TRAIN_TOKENS, "train ceiling")
    _require(general_train_tokens <= MAXIMUM_GENERAL_TRAIN_TOKENS, "general ceiling")
    _require(auxiliary_train_tokens <= MAXIMUM_AUXILIARY_TRAIN_TOKENS, "aux ceiling")

    projected_total = CURRENT_ACCEPTED_TRAIN_TOKENS + train_tokens
    _require(
        projected_total == report["closure"]["projected_train_supervised_tokens"],
        "projection",
    )
    _require(projected_total >= TEN_MILLION_TOKENS, "10M floor not reached")

    expected_sample = _owner_sample(replay_selected)
    review = manifest["owner_review"]
    sample_path = _safe_child(private, str(review["sample_file"]))
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    _require(sample == expected_sample, "owner sample replay mismatch")
    _require(sha256_value(sample) == review["sample_sha256"], "owner sample hash")
    _require(len(sample) == 32 and review["owner_review_status"] == "pending", "owner review state")
    html_path = _safe_child(private, str(review["html_file"]))
    _require(file_sha256(html_path) == review["html_sha256"], "owner HTML hash")

    private_bytes = _directory_bytes(private)
    _require(
        private_bytes == report["private_writes"]["candidate_and_evidence_bytes"],
        "private byte total",
    )
    _require(private_bytes + raw_bytes <= MAXIMUM_PRIVATE_WRITES, "private write ceiling")

    verification: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_private_nonproduction_owner_review_pending",
        "bindings": {
            "curation_report_file_sha256": file_sha256(report_path),
            "curation_report_internal_sha256": report["report_sha256"],
            "private_manifest_file_sha256": file_sha256(manifest_path),
            "private_manifest_internal_sha256": manifest["manifest_sha256"],
            "overlay_plan_file_sha256": file_sha256(overlay_plan_file),
            "overlay_plan_internal_sha256": overlay_plan["plan_sha256"],
            "pyarrow_wheel_sha256": file_sha256(wheel),
            "tokenizer_sha256": file_sha256(tokenizer_file),
            "benchmark_registry_sha256": benchmark.sha256,
        },
        "checks": {
            "candidate_files_rehashed": len(inventories),
            "candidate_records_replayed": len(replay_selected),
            "deterministic_selection_replayed": True,
            "machine_gate_replayed": True,
            "owner_review_pending": True,
            "raw_objects_rehashed": len(raw_inventory),
            "source_gate_replayed": True,
            "ten_million_floor_reached_by_projection": True,
        },
        "limits": {
            "auxiliary_train_supervised_tokens": auxiliary_train_tokens,
            "general_train_supervised_tokens": general_train_tokens,
            "overlay_and_wheel_bytes": overlay_bytes,
            "private_and_raw_bytes": private_bytes + raw_bytes,
            "train_supervised_tokens": train_tokens,
            "validation_supervised_tokens": validation_tokens,
        },
        "prohibitions_preserved": {
            "candidate_registry_change": True,
            "dataset_seal": True,
            "external_api": True,
            "external_distribution": True,
            "gpu": True,
            "production_promotion": True,
            "sft_training": True,
        },
        "raw_training_text_in_verification": False,
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


def write_verification(path: str | Path, value: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = ["SCHEMA_VERSION", "verify_curation", "write_verification"]
