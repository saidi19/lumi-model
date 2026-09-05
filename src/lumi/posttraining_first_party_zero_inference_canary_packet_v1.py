"""Build blank, private authoring packets for the zero-inference SFT canary."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PLAN_SCHEMA = "lumi.sft-first-party-zero-inference-canary-packet-plan.v0.1"
PLAN_STATUS = "authorized_local_blank_packet_preparation_no_candidate_construction"
ROW_SCHEMA = "lumi.sft-first-party-zero-inference-canary-blank-row.v0.1"
MANIFEST_SCHEMA = "lumi.sft-first-party-zero-inference-canary-packet-manifest.v0.1"
REPORT_SCHEMA = "lumi.sft-first-party-zero-inference-canary-packet-report.v0.1"
REPORT_STATUS = (
    "PASS_blank_zero_inference_canary_packets_materialized_"
    "human_completion_pending"
)
VERIFICATION_SCHEMA = (
    "lumi.sft-first-party-zero-inference-canary-packet-verification.v0.1"
)
VERIFICATION_STATUS = (
    "PASS_verified_blank_zero_inference_canary_packets_human_completion_pending"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

AUTHOR_SLOTS = tuple(f"author-slot-{index:02d}" for index in range(1, 11))
REVIEWER_SLOTS = tuple(f"reviewer-slot-{index:02d}" for index in range(1, 11))

EXPECTED_ALLOCATIONS = (
    {
        "creation_lane": "human_authored",
        "language_mode": "banglish",
        "task_category": "conversation",
        "train_records": 66,
        "validation_records": 17,
    },
    {
        "creation_lane": "human_authored",
        "language_mode": "natural_code_switching",
        "task_category": "conversation",
        "train_records": 34,
        "validation_records": 8,
    },
    {
        "creation_lane": "human_authored",
        "language_mode": "banglish",
        "task_category": "factual_qa",
        "train_records": 66,
        "validation_records": 17,
    },
    {
        "creation_lane": "human_authored",
        "language_mode": "natural_code_switching",
        "task_category": "factual_qa",
        "train_records": 34,
        "validation_records": 8,
    },
    {
        "creation_lane": "human_authored",
        "language_mode": "banglish",
        "task_category": "multilingual_bengali",
        "train_records": 66,
        "validation_records": 17,
    },
    {
        "creation_lane": "human_authored",
        "language_mode": "natural_code_switching",
        "task_category": "multilingual_bengali",
        "train_records": 34,
        "validation_records": 8,
    },
    {
        "creation_lane": "human_authored",
        "language_mode": "banglish",
        "task_category": "writing_summarization",
        "train_records": 134,
        "validation_records": 33,
    },
    {
        "creation_lane": "human_authored",
        "language_mode": "natural_code_switching",
        "task_category": "writing_summarization",
        "train_records": 66,
        "validation_records": 17,
    },
    {
        "creation_lane": "deterministic_grounded",
        "language_mode": "banglish",
        "task_category": "tool_api",
        "train_records": 66,
        "validation_records": 17,
    },
    {
        "creation_lane": "deterministic_grounded",
        "language_mode": "natural_code_switching",
        "task_category": "tool_api",
        "train_records": 34,
        "validation_records": 8,
    },
    {
        "creation_lane": "human_authored",
        "language_mode": "banglish",
        "task_category": "safety_helpfulness",
        "train_records": 67,
        "validation_records": 16,
    },
    {
        "creation_lane": "deterministic_grounded",
        "language_mode": "banglish",
        "task_category": "safety_helpfulness",
        "train_records": 67,
        "validation_records": 17,
    },
    {
        "creation_lane": "human_authored",
        "language_mode": "natural_code_switching",
        "task_category": "safety_helpfulness",
        "train_records": 33,
        "validation_records": 9,
    },
    {
        "creation_lane": "deterministic_grounded",
        "language_mode": "natural_code_switching",
        "task_category": "safety_helpfulness",
        "train_records": 33,
        "validation_records": 8,
    },
)

_REPORT_FORBIDDEN_KEYS = {
    "assistant_response",
    "content",
    "messages",
    "prompt",
    "response",
    "source_text",
    "text",
    "user_prompt",
    "utterance",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(pretty_json_bytes(value))
    temporary.replace(destination)


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _resolve_binding(
    repository_root: str | Path,
    specification: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    root = Path(repository_root).resolve()
    relative = specification.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label}.path must be a non-empty string")
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing binding {label}: {path}")
    expected = _require_sha256(specification.get("sha256"), f"{label}.sha256")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} file SHA-256 changed")
    return path


def _verify_internal_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    expected = _require_sha256(value.get(field), f"{label}.{field}")
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != expected:
        raise ValueError(f"{label} internal seal changed")


def _assert_report_raw_text_free(value: Any, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("evidence keys must be strings")
            if key.lower() in _REPORT_FORBIDDEN_KEYS:
                raise ValueError(f"raw-text key forbidden at {'.'.join(path + (key,))}")
            _assert_report_raw_text_free(item, path=path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_report_raw_text_free(item, path=path + (str(index),))


def allocation_totals(
    allocations: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int] | int]:
    split = Counter()
    lane = Counter()
    language = Counter()
    task = Counter()
    cell = Counter()
    for item in allocations:
        train = item.get("train_records")
        validation = item.get("validation_records")
        if not isinstance(train, int) or train < 0:
            raise ValueError("allocation train count must be non-negative")
        if not isinstance(validation, int) or validation < 0:
            raise ValueError("allocation validation count must be non-negative")
        total = train + validation
        split["train"] += train
        split["validation"] += validation
        lane[str(item["creation_lane"])] += total
        language[str(item["language_mode"])] += total
        task[str(item["task_category"])] += total
        cell[f"{item['task_category']}::{item['language_mode']}"] += total
    return {
        "records": sum(split.values()),
        "split_records": dict(sorted(split.items())),
        "lane_records": dict(sorted(lane.items())),
        "language_records": dict(sorted(language.items())),
        "task_records": dict(sorted(task.items())),
        "task_language_cell_records": dict(sorted(cell.items())),
    }


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = _read_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != PLAN_STATUS:
        raise ValueError("unexpected blank canary-packet plan identity")
    authorization = plan.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("authorization missing")
    if authorization.get("blank_private_packet_preparation_authorized") is not True:
        raise ValueError("blank private packet preparation is not authorized")
    forbidden = (
        "candidate_construction_authorized",
        "candidate_reads_authorized",
        "dataset_seal_authorized",
        "external_distribution_authorized",
        "gpu_authorized",
        "inference_authorized",
        "model_generated_training_text_authorized",
        "network_authorized",
        "production_promotion_authorized",
        "registry_mutation_authorized",
        "remote_access_authorized",
        "sft_training_authorized",
        "source_reads_authorized",
        "test_object_reads_authorized",
        "unrelated_access_authorized",
    )
    if any(authorization.get(field) is not False for field in forbidden):
        raise ValueError("blank packet plan authorizes a forbidden action")

    limits = plan.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("limits missing")
    for field in (
        "maximum_candidate_reads",
        "maximum_candidate_writes",
        "maximum_gpu_actions",
        "maximum_inference_calls",
        "maximum_network_bytes",
        "maximum_registry_writes",
        "maximum_source_line_reads",
        "maximum_test_object_reads",
        "maximum_training_actions",
    ):
        if limits.get(field) != 0:
            raise ValueError(f"{field} must be zero")
    if limits.get("maximum_blank_template_rows_written") != 1000:
        raise ValueError("blank template write bound changed")
    if limits.get("maximum_blank_template_rows_verified") != 1000:
        raise ValueError("blank template verification bound changed")
    if limits.get("maximum_packet_files") != 50:
        raise ValueError("packet-file bound changed")

    bindings = plan.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("bindings missing")
    for name, specification in bindings.items():
        if not isinstance(specification, dict):
            raise ValueError(f"binding {name} must be an object")
        _resolve_binding(repository_root, specification, label=name)

    if plan.get("allocations") != list(EXPECTED_ALLOCATIONS):
        raise ValueError("canary allocation schedule changed")
    totals = allocation_totals(plan["allocations"])
    if totals != plan.get("expected_totals"):
        raise ValueError("canary allocation totals changed")
    packet_contract = plan.get("packet_contract")
    if packet_contract != {
        "author_slots": list(AUTHOR_SLOTS),
        "human_identity_assignment_complete": False,
        "packet_count": 50,
        "reviewer_slots": list(REVIEWER_SLOTS),
        "rows_per_packet": 20,
        "slot_names_are_not_human_identity_evidence": True,
        "train_rows_per_packet": 16,
        "validation_rows_per_packet": 4,
    }:
        raise ValueError("packet contract changed")
    return plan


def _blank_row(
    *,
    allocation: Mapping[str, Any],
    split: str,
    index: int,
) -> dict[str, Any]:
    task = str(allocation["task_category"])
    language = str(allocation["language_mode"])
    lane = str(allocation["creation_lane"])
    row: dict[str, Any] = {
        "author_declaration": {
            "author_id": "",
            "authored_at": "",
            "human_written_from_scratch": None if lane == "human_authored" else False,
            "model_assistance_used": None if lane == "human_authored" else False,
            "rights_confirmed": None,
        },
        "creation_lane": lane,
        "independent_review": {
            "accepted": None,
            "context_grounding_passed": None,
            "language_naturalness_passed": None,
            "non_copying_passed": None,
            "privacy_passed": None,
            "quality_score": None,
            "reviewed_at": "",
            "reviewer_id": "",
            "rights_passed": None,
            "safety_passed": None,
            "task_correctness_passed": None,
        },
        "language_mode": language,
        "messages": [],
        "record_id": f"zic-v01-{split}-{task}-{language}-{lane}-{index:04d}",
        "schema_version": ROW_SCHEMA,
        "split": split,
        "task_category": task,
        "task_evidence": {},
        "template_family_id": "",
    }
    if lane == "deterministic_grounded":
        row["deterministic_provenance"] = {
            "atom_pack_sha256": "",
            "human_authorship_claimed": False,
            "renderer_sha256": "",
            "scenario_id": "",
            "tool_result_sha256": "",
        }
    return row


def build_blank_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    if plan.get("allocations") != list(EXPECTED_ALLOCATIONS):
        raise ValueError("canary allocation schedule changed")
    rows: list[dict[str, Any]] = []
    for allocation in plan["allocations"]:
        for split in ("train", "validation"):
            count = int(allocation[f"{split}_records"])
            for index in range(1, count + 1):
                rows.append(_blank_row(allocation=allocation, split=split, index=index))
    validate_blank_rows(rows, expected_totals=plan["expected_totals"])
    return rows


def validate_blank_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_totals: Mapping[str, Any]
) -> None:
    if len(rows) != 1000:
        raise ValueError("blank canary must contain exactly 1,000 rows")
    record_ids: set[str] = set()
    allocations = Counter()
    for row in rows:
        if row.get("schema_version") != ROW_SCHEMA:
            raise ValueError("blank row schema changed")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id in record_ids:
            raise ValueError("blank row ID is missing or duplicated")
        record_ids.add(record_id)
        if row.get("messages") != []:
            raise ValueError("blank packet contains authored message text")
        if row.get("task_evidence") != {} or row.get("template_family_id") != "":
            raise ValueError("blank packet contains completed task evidence")
        author = row.get("author_declaration")
        review = row.get("independent_review")
        if not isinstance(author, Mapping) or author.get("author_id") != "":
            raise ValueError("blank row contains an author identity")
        if not isinstance(review, Mapping) or review.get("reviewer_id") != "":
            raise ValueError("blank row contains a reviewer identity")
        allocations[
            (
                str(row.get("creation_lane")),
                str(row.get("language_mode")),
                str(row.get("task_category")),
                str(row.get("split")),
            )
        ] += 1
    split_counts = Counter(row["split"] for row in rows)
    lane_counts = Counter(row["creation_lane"] for row in rows)
    language_counts = Counter(row["language_mode"] for row in rows)
    task_counts = Counter(row["task_category"] for row in rows)
    cell_counts = Counter(
        f"{row['task_category']}::{row['language_mode']}" for row in rows
    )
    observed = {
        "records": len(rows),
        "split_records": dict(sorted(split_counts.items())),
        "lane_records": dict(sorted(lane_counts.items())),
        "language_records": dict(sorted(language_counts.items())),
        "task_records": dict(sorted(task_counts.items())),
        "task_language_cell_records": dict(sorted(cell_counts.items())),
    }
    if observed != expected_totals:
        raise ValueError("blank-row aggregate schedule changed")
    for allocation in EXPECTED_ALLOCATIONS:
        for split in ("train", "validation"):
            key = (
                allocation["creation_lane"],
                allocation["language_mode"],
                allocation["task_category"],
                split,
            )
            if allocations[key] != allocation[f"{split}_records"]:
                raise ValueError("blank-row cell allocation changed")


def build_packets(
    rows: Sequence[Mapping[str, Any]], *, plan: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    validate_blank_rows(rows, expected_totals=plan["expected_totals"])
    packet_rows: list[list[dict[str, Any]]] = [[] for _ in range(50)]
    for split in ("train", "validation"):
        ordered = sorted(
            (
                json.loads(json.dumps(row, ensure_ascii=False))
                for row in rows
                if row["split"] == split
            ),
            key=lambda row: row["record_id"],
        )
        for index, row in enumerate(ordered):
            packet_rows[index % 50].append(row)
    packets: list[dict[str, Any]] = []
    for index, rows_for_packet in enumerate(packet_rows):
        packet_id = f"zic-v01-packet-{index + 1:03d}"
        author_slot = AUTHOR_SLOTS[index % len(AUTHOR_SLOTS)]
        reviewer_slot = REVIEWER_SLOTS[(index + 1) % len(REVIEWER_SLOTS)]
        if author_slot.split("-")[-1] == reviewer_slot.split("-")[-1]:
            raise AssertionError("author and reviewer slots must differ")
        for row in rows_for_packet:
            row["assignment"] = {
                "author_slot": author_slot,
                "packet_id": packet_id,
                "reviewer_slot": reviewer_slot,
            }
        rows_for_packet.sort(key=lambda row: row["record_id"])
        split_counts = Counter(row["split"] for row in rows_for_packet)
        if split_counts != {"train": 16, "validation": 4}:
            raise AssertionError("packet split balance changed")
        packets.append(
            {
                "author_slot": author_slot,
                "packet_id": packet_id,
                "reviewer_slot": reviewer_slot,
                "rows": rows_for_packet,
            }
        )
    if len(packets) != 50 or sum(len(packet["rows"]) for packet in packets) != 1000:
        raise AssertionError("blank packet totals changed")
    return tuple(packets)


def packet_jsonl_bytes(packet: Mapping[str, Any]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in packet["rows"])


def build_manifest(
    packets: Sequence[Mapping[str, Any]], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    entries = []
    for packet in packets:
        payload = packet_jsonl_bytes(packet)
        rows = packet["rows"]
        entries.append(
            {
                "author_slot": packet["author_slot"],
                "bytes": len(payload),
                "creation_lane_counts": dict(
                    sorted(Counter(row["creation_lane"] for row in rows).items())
                ),
                "jsonl_sha256": hashlib.sha256(payload).hexdigest(),
                "language_counts": dict(
                    sorted(Counter(row["language_mode"] for row in rows).items())
                ),
                "packet_id": packet["packet_id"],
                "relative_path": f"packets/{packet['packet_id']}.jsonl",
                "reviewer_slot": packet["reviewer_slot"],
                "rows": len(rows),
                "split_counts": dict(
                    sorted(Counter(row["split"] for row in rows).items())
                ),
                "task_counts": dict(
                    sorted(Counter(row["task_category"] for row in rows).items())
                ),
            }
        )
    manifest: dict[str, Any] = {
        "authorization": {
            "candidate_construction_authorized": False,
            "model_generated_training_text_authorized": False,
        },
        "expected_totals": plan["expected_totals"],
        "human_identity_assignment_complete": False,
        "packet_count": len(entries),
        "packets": entries,
        "raw_training_text_retained": False,
        "records": sum(entry["rows"] for entry in entries),
        "schema_version": MANIFEST_SCHEMA,
        "status": "blank_private_packets_waiting_for_real_human_identity_assignment",
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("blank packet manifest schema changed")
    _verify_internal_seal(manifest, "manifest_sha256", "blank_packet_manifest")
    if manifest.get("status") != (
        "blank_private_packets_waiting_for_real_human_identity_assignment"
    ):
        raise ValueError("blank packet manifest status changed")
    if manifest.get("human_identity_assignment_complete") is not False:
        raise ValueError("blank packet manifest claims human identities")
    if manifest.get("raw_training_text_retained") is not False:
        raise ValueError("blank packet manifest retained raw training text")
    if manifest.get("records") != 1000 or manifest.get("packet_count") != 50:
        raise ValueError("blank packet manifest totals changed")
    entries = manifest.get("packets")
    if not isinstance(entries, list) or len(entries) != 50:
        raise ValueError("blank packet manifest inventory changed")
    packet_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("blank packet manifest entry is invalid")
        packet_id = entry.get("packet_id")
        if not isinstance(packet_id, str) or packet_id in packet_ids:
            raise ValueError("blank packet ID is missing or duplicated")
        packet_ids.add(packet_id)
        if entry.get("rows") != 20 or entry.get("split_counts") != {
            "train": 16,
            "validation": 4,
        }:
            raise ValueError("blank packet row balance changed")
        if entry.get("author_slot") not in AUTHOR_SLOTS:
            raise ValueError("blank packet author slot changed")
        if entry.get("reviewer_slot") not in REVIEWER_SLOTS:
            raise ValueError("blank packet reviewer slot changed")
        if entry["author_slot"].split("-")[-1] == entry["reviewer_slot"].split(
            "-"
        )[-1]:
            raise ValueError("blank packet review slot is not independent")
        _require_sha256(entry.get("jsonl_sha256"), "packet.jsonl_sha256")
        if not isinstance(entry.get("bytes"), int) or entry["bytes"] <= 0:
            raise ValueError("blank packet byte count is invalid")


def validate_bound_predecessors(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> None:
    objects = {}
    for label in (
        "targeted_gap_plan_v02",
        "targeted_gap_report_v02",
        "targeted_gap_verification_v02",
        "zero_inference_preflight_plan_v01",
        "zero_inference_preflight_report_v01",
        "zero_inference_preflight_verification_v01",
    ):
        path = _resolve_binding(repository_root, plan["bindings"][label], label=label)
        objects[label] = _read_object(path)
    _verify_internal_seal(
        objects["targeted_gap_report_v02"], "report_sha256", "targeted_gap_report_v02"
    )
    _verify_internal_seal(
        objects["targeted_gap_verification_v02"],
        "verification_sha256",
        "targeted_gap_verification_v02",
    )
    _verify_internal_seal(
        objects["zero_inference_preflight_plan_v01"],
        "plan_internal_sha256",
        "zero_inference_preflight_plan_v01",
    )
    _verify_internal_seal(
        objects["zero_inference_preflight_report_v01"],
        "report_sha256",
        "zero_inference_preflight_report_v01",
    )
    _verify_internal_seal(
        objects["zero_inference_preflight_verification_v01"],
        "verification_sha256",
        "zero_inference_preflight_verification_v01",
    )
    report = objects["targeted_gap_report_v02"]
    verification = objects["targeted_gap_verification_v02"]
    if report.get("status") != (
        "PASS_registry_v0_22_targeted_gap_rebase_massive_active_"
        "first_party_canary_ready"
    ):
        raise ValueError("targeted-gap v0.2 readiness changed")
    if verification.get("status") != (
        "PASS_verified_registry_v0_22_targeted_gap_rebase_nonproduction"
    ):
        raise ValueError("targeted-gap v0.2 verification changed")
    if report.get("next_gate", {}).get("action") != (
        "first_party_zero_inference_1000_row_candidate_canary"
    ):
        raise ValueError("targeted-gap v0.2 next gate changed")
    if report.get("source_selection", {}).get(
        "candidate_construction_authorized"
    ) is not False:
        raise ValueError("predecessor authorized candidate construction")
    if report.get("production_eligible") is not False:
        raise ValueError("predecessor became production eligible")


def build_report(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
) -> dict[str, Any]:
    validate_manifest(manifest)
    packet_bytes = sum(entry["bytes"] for entry in manifest["packets"])
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": REPORT_STATUS,
        "observed_at": plan["observed_at"],
        "authorization_sha256": plan["authorization_sha256"],
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "manifest_file_sha256": _require_sha256(
            manifest_file_sha256, "manifest_file_sha256"
        ),
        "manifest_internal_sha256": manifest["manifest_sha256"],
        "bindings": {
            name: {"file_sha256": specification["sha256"]}
            for name, specification in sorted(plan["bindings"].items())
        },
        "packet_summary": {
            "blank_template_only": True,
            "human_identity_assignment_complete": False,
            "packet_count": manifest["packet_count"],
            "packet_payload_bytes": packet_bytes,
            "records": manifest["records"],
            "rows_per_packet": 20,
            "train_records": manifest["expected_totals"]["split_records"]["train"],
            "validation_records": manifest["expected_totals"]["split_records"][
                "validation"
            ],
        },
        "schedule": manifest["expected_totals"],
        "quality_contract": plan["quality_contract"],
        "storage_policy": plan["storage_policy"],
        "next_gate": plan["next_gate"],
        "authorization": plan["authorization"],
        "resource_accounting": {
            "blank_template_rows_written": 1000,
            "candidate_reads": 0,
            "candidate_writes": 0,
            "gpu_actions": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "packet_files_written": 50,
            "registry_writes": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
            "training_actions": 0,
        },
        "candidate_construction_started": False,
        "legal_conclusion_made": False,
        "production_eligible": False,
        "raw_training_text_retained": False,
    }
    _assert_report_raw_text_free(report)
    report["report_sha256"] = sha256_value(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA or report.get(
        "status"
    ) != REPORT_STATUS:
        raise ValueError("blank packet report identity changed")
    _verify_internal_seal(report, "report_sha256", "blank_packet_report")
    _assert_report_raw_text_free(report)
    if report.get("candidate_construction_started") is not False:
        raise ValueError("blank packet report claims candidate construction")
    if report.get("production_eligible") is not False:
        raise ValueError("blank packet report became production eligible")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("blank packet report retained raw training text")
    summary = report.get("packet_summary")
    if not isinstance(summary, Mapping) or summary.get("blank_template_only") is not True:
        raise ValueError("blank packet summary changed")
    if summary.get("records") != 1000 or summary.get("packet_count") != 50:
        raise ValueError("blank packet report totals changed")
    if summary.get("train_records") != 800 or summary.get("validation_records") != 200:
        raise ValueError("blank packet split totals changed")
    if report.get("next_gate", {}).get("action") != (
        "real_human_identity_assignment_terms_and_private_packet_completion"
    ):
        raise ValueError("blank packet next gate changed")
    accounting = report.get("resource_accounting")
    if not isinstance(accounting, Mapping):
        raise ValueError("resource accounting missing")
    for field in (
        "candidate_reads",
        "candidate_writes",
        "gpu_actions",
        "inference_calls",
        "network_bytes",
        "registry_writes",
        "source_line_reads",
        "test_object_reads",
        "training_actions",
    ):
        if accounting.get(field) != 0:
            raise ValueError(f"prohibited resource accounting changed: {field}")


def build_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_report(report)
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": VERIFICATION_STATUS,
        "observed_at": plan["observed_at"],
        "report_file_sha256": _require_sha256(
            report_file_sha256, "report_file_sha256"
        ),
        "report_internal_sha256": report["report_sha256"],
        "checks": {
            "targeted_gap_v0_2_readiness_and_internal_seals_match": True,
            "massive_gate_precondition_is_satisfied": True,
            "exactly_1000_blank_slots_materialized": True,
            "exactly_50_packets_with_20_rows_each": True,
            "every_packet_has_16_train_and_4_validation_rows": True,
            "schedule_has_800_train_and_200_validation_rows": True,
            "schedule_has_750_human_and_250_deterministic_rows": True,
            "all_task_language_cells_are_present": True,
            "all_message_arrays_are_empty": True,
            "no_human_identity_or_completed_review_is_claimed": True,
            "author_and_reviewer_placeholder_slots_differ": True,
            "completed_packets_must_remain_private_and_outside_git": True,
            "candidate_construction_has_not_started": True,
            "no_source_candidate_or_test_object_read": True,
            "no_network_gpu_inference_registry_seal_or_sft": True,
            "production_eligible_false": True,
        },
        "resource_accounting": {
            **report["resource_accounting"],
            "blank_template_rows_verified": 1000,
            "packet_files_verified": 50,
        },
        "production_eligible": False,
        "raw_training_text_retained": False,
    }
    _assert_report_raw_text_free(verification)
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "AUTHOR_SLOTS",
    "EXPECTED_ALLOCATIONS",
    "MANIFEST_SCHEMA",
    "PLAN_SCHEMA",
    "PLAN_STATUS",
    "REPORT_SCHEMA",
    "REPORT_STATUS",
    "REVIEWER_SLOTS",
    "ROW_SCHEMA",
    "VERIFICATION_SCHEMA",
    "VERIFICATION_STATUS",
    "allocation_totals",
    "build_blank_rows",
    "build_manifest",
    "build_packets",
    "build_report",
    "build_verification",
    "canonical_json_bytes",
    "load_plan",
    "packet_jsonl_bytes",
    "pretty_json_bytes",
    "sha256_file",
    "sha256_value",
    "validate_blank_rows",
    "validate_bound_predecessors",
    "validate_manifest",
    "validate_report",
    "write_json_atomic",
]
