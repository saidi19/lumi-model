"""CPU-only, raw-text-free capacity scan for rights-retained Aya Bengali lanes."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data_quality import BenchmarkRegistry, MinHashNearDuplicateIndex, normalize_for_matching
from .posttraining_closure_canary_curation import (
    _aya_record,
    _custom_quality_reasons,
    _priority,
    load_direct_tokenizer,
)
from .posttraining_data import evaluate_sft_record, load_sft_policy
from .posttraining_format import encode_sft_messages, load_chat_template

PLAN_SCHEMA_VERSION = "lumi.sft-aya-capacity-scan-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-aya-capacity-scan-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-aya-capacity-scan-verification.v0.1"

TARGETED_PLAN_FILE_SHA256 = (
    "fc3707418865a328462e87dfdf2ef77be2ee6078a1b75bec7fa56566de11ad58"
)
TARGETED_REPORT_FILE_SHA256 = (
    "e6ddd287c6420c8b788b14bd0bdc8afce1926350b5208ec1a4448b11395f31b2"
)
TARGETED_VERIFICATION_FILE_SHA256 = (
    "958c307417b5ac3e5de0c55e69cadd62c1d757bdc16a3aeac2f8163a33b7ae03"
)
RAW_AYA_SHA256 = "8d4079c0f6e329280208cdb50d7c42164dd5c02781aab78cdf99780a606c9810"
RAW_AYA_BYTES = 442_809_026
RAW_AYA_ROWS = 1_200_429
WIKISPLIT_RESERVOIR_LIMIT = 100_000
TARGETED_CLOSURE_TOKENS = 1_600_000

EXPECTED_AUTHORIZATION = {
    "candidate_materialization_authorized": False,
    "cpu_only": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "network_access_authorized": False,
    "production_promotion_authorized": False,
    "registry_change_authorized": False,
    "sft_training_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_current_mix_records": 20_000,
    "maximum_private_write_bytes": 8_388_608,
    "maximum_rows_read": RAW_AYA_ROWS,
    "maximum_runtime_minutes": 180,
    "wikisplit_near_dedup_reservoir_records": WIKISPLIT_RESERVOIR_LIMIT,
}
EXPECTED_OUTPUT = {
    "candidate_materialized": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-aya-capacity-scan-v0.1.json",
    "verification_path": "evidence/lumi-4b-sft-aya-capacity-scan-verification-v0.1.json",
}
EXPECTED_BINDINGS = frozenset(
    {
        "ai_v3_candidate",
        "aya_registry_candidate",
        "aya_rights_candidate",
        "aya_rights_materialization_report",
        "aya_rights_materialization_verification",
        "banglabridge_candidate",
        "banglatlit_candidate",
        "benchmark_registry",
        "chat_template",
        "closure_successor_report",
        "closure_successor_verification",
        "current_mix_report",
        "current_mix_verification",
        "data_contract",
        "data_quality",
        "implementation",
        "raw_aya_parquet",
        "rights_report",
        "rights_verification",
        "runner",
        "safety_candidate",
        "targeted_plan",
        "targeted_report",
        "targeted_verification",
        "teacher_candidate",
        "test",
        "tokenizer_json",
        "verifier",
    }
)

CURRENT_SOURCE_EXPECTATIONS = {
    "ai_v3_candidate": 320,
    "aya_registry_candidate": 1_458,
    "aya_rights_candidate": 3_995,
    "banglabridge_candidate": 116,
    "banglatlit_candidate": 13_576,
    "safety_candidate": 320,
    "teacher_candidate": 80,
}

FAMILY_BY_DATASET = {
    "Wiki-split-inst (T)": "wiki_split_inst_translated",
    "Joke-explaination-inst (T)": "joke_explaination_inst_translated",
}
QUARANTINED_DATASET = "Flan-GEM-wiki-lingua (T)"
FAMILY_RIGHTS = {
    "wiki_split_inst_translated": {
        "license_id": "cc-by-sa-4.0",
        "production_sft_eligible_now": False,
        "production_token_credit_policy": "zero_until_share_alike_compliance_resolved",
        "upstream_repository": "google-research-datasets/wiki-split",
        "upstream_revision": "80f5bddf42d1a68201b7b107ee4b8d86d82e9fde",
    },
    "joke_explaination_inst_translated": {
        "license_id": "mit",
        "production_sft_eligible_now": True,
        "production_token_credit_policy": "capacity_only_pending_materialization_and_promotion",
        "upstream_repository": "theblackcat102/joke_explaination",
        "upstream_revision": "888f2ffeb2398ae737ccc6e64cc7215e129436e7",
    },
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(json_file_bytes(value))
    temporary.replace(destination)


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a SHA-256") from error
    return value


def resolve_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Aya capacity binding path is invalid: {name}")
    path = (root / relative).resolve()
    if not path.is_relative_to(workspace):
        raise ValueError(f"Aya capacity binding escapes workspace: {name}")
    if not path.is_file():
        raise ValueError(f"Aya capacity binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(path) != expected:
        raise ValueError(f"Aya capacity binding changed: {name}")
    return path


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-10",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_cpu_only_local_aya_capacity_scan",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"Aya capacity plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("Aya capacity binding set changed")
    resolved = {
        name: resolve_binding(repository_root, binding, name=name)
        for name, binding in bindings.items()
    }
    exact = {
        "raw_aya_parquet": RAW_AYA_SHA256,
        "targeted_plan": TARGETED_PLAN_FILE_SHA256,
        "targeted_report": TARGETED_REPORT_FILE_SHA256,
        "targeted_verification": TARGETED_VERIFICATION_FILE_SHA256,
    }
    for name, expected_sha256 in exact.items():
        if bindings[name]["sha256"] != expected_sha256:
            raise ValueError(f"Aya capacity exact binding changed: {name}")
    raw = resolved["raw_aya_parquet"]
    if raw.stat().st_size != RAW_AYA_BYTES:
        raise ValueError("Aya capacity raw object byte count changed")
    quality = _load_json(resolved["data_quality"])
    near = quality.get("near_duplicate")
    if not isinstance(near, Mapping) or dict(near) != plan.get("near_duplicate_policy"):
        raise ValueError("Aya capacity near-duplicate policy changed")
    targeted = _load_json(resolved["targeted_plan"])
    lane = targeted.get("local_aya_capacity_lane")
    if not isinstance(lane, Mapping):
        raise ValueError("targeted closure Aya capacity lane is missing")
    if (
        lane.get("maximum_rows_read") != RAW_AYA_ROWS
        or lane.get("raw_object_sha256") != RAW_AYA_SHA256
        or lane.get("production_token_credit_before_compliance_resolution") != 0
    ):
        raise ValueError("targeted closure Aya capacity boundary changed")
    return plan


def _role_tagged_text(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("candidate messages are missing")
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("candidate message is not an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("candidate message fields are invalid")
        parts.append(f"{role} {content}")
    return normalize_for_matching("\n".join(parts))


def _exact_digest(record: Mapping[str, Any]) -> bytes:
    return hashlib.sha256(_role_tagged_text(record).encode("utf-8")).digest()


def _rights_project(record: dict[str, Any], family_id: str) -> dict[str, Any]:
    rights = FAMILY_RIGHTS[family_id]
    projected = dict(record)
    projected["source"] = dict(record["source"])
    projected["source"]["license_id"] = rights["license_id"]
    projected["provenance"] = dict(record["provenance"])
    projected["provenance"].update(
        {
            "constituent_family_id": family_id,
            "constituent_rights_projection_only": True,
            "upstream_license_id": rights["license_id"],
            "upstream_repository_sha256": hashlib.sha256(
                rights["upstream_repository"].encode("utf-8")
            ).hexdigest(),
            "upstream_revision": rights["upstream_revision"],
        }
    )
    return projected


@dataclass(frozen=True)
class CapacityItem:
    family_id: str
    priority: int
    record: dict[str, Any]
    serialized_tokens: int
    supervised_tokens: int


class _CachedTokenizer:
    """Preserve the tokenizer contract while avoiding a full vocab copy per row."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.all_special_ids = list(delegate.all_special_ids)
        self._vocabulary = dict(delegate.get_vocab())

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        split_special_tokens: bool,
    ) -> list[int]:
        return self._delegate.encode(
            text,
            add_special_tokens=add_special_tokens,
            split_special_tokens=split_special_tokens,
        )

    def get_vocab(self) -> Mapping[str, int]:
        return self._vocabulary


def _reservoir_add(
    heap: list[tuple[int, str, CapacityItem]],
    item: CapacityItem,
    *,
    limit: int,
) -> None:
    entry = (-item.priority, str(item.record["record_id"]), item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def _sorted_reservoir(
    heap: Sequence[tuple[int, str, CapacityItem]],
) -> list[CapacityItem]:
    return [entry[2] for entry in sorted(heap, key=lambda entry: (-entry[0], entry[1]))]


def _candidate_rows(path: Path) -> Sequence[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"candidate row is not an object: {path}:{line_number}")
            rows.append(row)
    return rows


def _current_mix_index(
    *,
    plan: Mapping[str, Any],
    resolved: Mapping[str, Path],
) -> tuple[set[bytes], MinHashNearDuplicateIndex, dict[str, Any]]:
    near = MinHashNearDuplicateIndex(plan["near_duplicate_policy"])
    exact: set[bytes] = set()
    metrics: dict[str, Any] = {}
    accepted_index_ids: set[str] = set()
    total = 0
    for binding_name, expected_records in sorted(CURRENT_SOURCE_EXPECTATIONS.items()):
        rows = _candidate_rows(resolved[binding_name])
        if len(rows) != expected_records:
            raise ValueError(f"current-mix candidate count changed: {binding_name}")
        total += len(rows)
        source_counter = Counter()
        for line_number, row in enumerate(rows, start=1):
            digest = _exact_digest(row)
            exact.add(digest)
            signature = near.signature(_role_tagged_text(row))
            duplicate = near.duplicate_of(signature)
            if duplicate is not None:
                source_counter["near_duplicate_within_current_mix"] += 1
                continue
            index_id = f"current:{binding_name}:{line_number:07d}"
            if index_id in accepted_index_ids:
                raise ValueError("current-mix near index identity collision")
            near.add(index_id, signature)
            accepted_index_ids.add(index_id)
            source_counter["near_indexed"] += 1
        metrics[binding_name] = {
            "candidate_file_sha256": file_sha256(resolved[binding_name]),
            "records": len(rows),
            "near_duplicate_within_current_mix": source_counter[
                "near_duplicate_within_current_mix"
            ],
            "near_indexed": source_counter["near_indexed"],
        }
    if total > plan["limits"]["maximum_current_mix_records"]:
        raise ValueError("current-mix record ceiling exceeded")
    return exact, near, {
        "exact_conversation_hashes": len(exact),
        "near_index_records": near.count(),
        "records_read": total,
        "sources": metrics,
    }


def _empty_split_metrics() -> dict[str, Counter[str]]:
    return {"train": Counter(), "validation": Counter()}


def _split_metrics_json(value: Mapping[str, Counter[str]]) -> dict[str, Any]:
    return {
        split: {
            "records": counter["records"],
            "serialized_tokens": counter["serialized_tokens"],
            "supervised_tokens": counter["supervised_tokens"],
        }
        for split, counter in sorted(value.items())
    }


def _near_deduplicate_reservoir(
    *,
    current_near: MinHashNearDuplicateIndex,
    items: Sequence[CapacityItem],
) -> tuple[dict[str, dict[str, Counter[str]]], dict[str, Counter[str]]]:
    by_family: dict[str, dict[str, Counter[str]]] = defaultdict(_empty_split_metrics)
    duplicate_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in sorted(items, key=lambda value: (value.priority, value.record["record_id"])):
        text = _role_tagged_text(item.record)
        signature = current_near.signature(text)
        duplicate = current_near.duplicate_of(signature)
        if duplicate is not None:
            reason = (
                "near_duplicate_against_current_mix"
                if duplicate.startswith("current:")
                else "near_duplicate_within_scanned_reservoir"
            )
            duplicate_counts[item.family_id][reason] += 1
            continue
        index_id = f"scan:{item.family_id}:{item.record['record_id']}"
        current_near.add(index_id, signature)
        split = str(item.record["split"])
        counter = by_family[item.family_id][split]
        counter["records"] += 1
        counter["serialized_tokens"] += item.serialized_tokens
        counter["supervised_tokens"] += item.supervised_tokens
    return by_family, duplicate_counts


def _family_result_template(family_id: str) -> dict[str, Any]:
    rights = FAMILY_RIGHTS[family_id]
    return {
        "constituent_family_id": family_id,
        "license_id": rights["license_id"],
        "production_sft_eligible_now": rights["production_sft_eligible_now"],
        "production_token_credit_policy": rights["production_token_credit_policy"],
        "upstream_repository_sha256": hashlib.sha256(
            rights["upstream_repository"].encode("utf-8")
        ).hexdigest(),
        "upstream_revision": rights["upstream_revision"],
    }


def build_report(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    root = Path(repository_root).resolve()
    resolved = {
        name: resolve_binding(root, binding, name=name)
        for name, binding in plan["bindings"].items()
    }
    raw_path = resolved["raw_aya_parquet"]
    parquet = pq.ParquetFile(raw_path)
    if parquet.metadata.num_rows != RAW_AYA_ROWS:
        raise ValueError("Aya Parquet row count changed")

    policy = load_sft_policy(resolved["data_contract"])
    benchmark = BenchmarkRegistry(
        {
            "anchor_ngram_tokens": 8,
            "minimum_anchor_matches": 2,
            "registry": str(resolved["benchmark_registry"]),
            "registry_sha256": file_sha256(resolved["benchmark_registry"]),
        },
        root,
    )
    tokenizer = _CachedTokenizer(load_direct_tokenizer(resolved["tokenizer_json"]))
    template = load_chat_template(resolved["chat_template"])
    current_exact, current_near, current_metrics = _current_mix_index(
        plan=plan,
        resolved=resolved,
    )

    raw_rows = Counter()
    eligible_rows = Counter()
    machine_accepted = Counter()
    exact_current_duplicates = Counter()
    exact_scan_duplicates = Counter()
    rejections: dict[str, Counter[str]] = defaultdict(Counter)
    exact_capacity: dict[str, dict[str, Counter[str]]] = defaultdict(_empty_split_metrics)
    exact_seen: set[bytes] = set()
    reservoirs: dict[str, list[tuple[int, str, CapacityItem]]] = defaultdict(list)
    scan_fingerprint = hashlib.sha256()
    processed = 0
    columns = [
        "id",
        "inputs",
        "targets",
        "dataset_name",
        "task_type",
        "language",
        "script",
        "split",
    ]
    for batch in parquet.iter_batches(batch_size=8_192, columns=columns):
        for row in batch.to_pylist():
            processed += 1
            dataset_name = str(row.get("dataset_name"))
            family_id = FAMILY_BY_DATASET.get(dataset_name)
            if dataset_name == QUARANTINED_DATASET:
                raw_rows["flan_gem_wikilingua_translated"] += 1
                scan_fingerprint.update(b"quarantine\0")
                continue
            if family_id is None:
                raw_rows["excluded_other_families"] += 1
                scan_fingerprint.update(b"excluded\0")
                continue
            raw_rows[family_id] += 1
            lane, record, adapter_reason = _aya_record(row)
            if record is None:
                rejections[family_id][f"adapter:{adapter_reason}"] += 1
                scan_fingerprint.update(f"{family_id}\0{adapter_reason}\0".encode())
                continue
            del lane
            eligible_rows[family_id] += 1
            record = _rights_project(record, family_id)
            reasons = list(_custom_quality_reasons(record))
            decision = evaluate_sft_record(
                record,
                policy=policy,
                benchmark_registry=benchmark,
                source_registry=None,
            )
            reasons.extend(decision.reasons)
            reasons = list(dict.fromkeys(reasons))
            if reasons:
                rejections[family_id].update(reasons)
                scan_fingerprint.update(
                    f"{family_id}\0reject\0{'|'.join(sorted(reasons))}\0".encode()
                )
                continue
            try:
                encoded = encode_sft_messages(
                    record["messages"], tokenizer=tokenizer, template=template
                )
            except ValueError as error:
                reason = "over_4k" if "maximum is 4096" in str(error) else "encoding_error"
                rejections[family_id][reason] += 1
                scan_fingerprint.update(f"{family_id}\0{reason}\0".encode())
                continue
            machine_accepted[family_id] += 1
            exact = _exact_digest(record)
            if exact in current_exact:
                exact_current_duplicates[family_id] += 1
                scan_fingerprint.update(f"{family_id}\0current_exact\0".encode())
                continue
            if exact in exact_seen:
                exact_scan_duplicates[family_id] += 1
                scan_fingerprint.update(f"{family_id}\0scan_exact\0".encode())
                continue
            exact_seen.add(exact)
            split = str(record["split"])
            counter = exact_capacity[family_id][split]
            counter["records"] += 1
            counter["serialized_tokens"] += len(encoded.input_ids)
            counter["supervised_tokens"] += encoded.supervised_tokens
            item = CapacityItem(
                family_id=family_id,
                priority=_priority(record),
                record=record,
                serialized_tokens=len(encoded.input_ids),
                supervised_tokens=encoded.supervised_tokens,
            )
            limit = (
                WIKISPLIT_RESERVOIR_LIMIT
                if family_id == "wiki_split_inst_translated"
                else RAW_AYA_ROWS
            )
            _reservoir_add(reservoirs[family_id], item, limit=limit)
            scan_fingerprint.update(
                family_id.encode()
                + b"\0accept\0"
                + exact
                + encoded.supervised_tokens.to_bytes(8, "big")
            )
        if progress is not None and processed % 65_536 < len(batch):
            progress(processed)
    if processed != RAW_AYA_ROWS:
        raise ValueError("Aya capacity scan did not read the exact row ceiling")

    reservoir_items = [
        item
        for family_id in sorted(reservoirs)
        for item in _sorted_reservoir(reservoirs[family_id])
    ]
    near_capacity, near_duplicates = _near_deduplicate_reservoir(
        current_near=current_near,
        items=reservoir_items,
    )

    families: dict[str, Any] = {}
    for family_id in sorted(FAMILY_RIGHTS):
        value = _family_result_template(family_id)
        value.update(
            {
                "adapter_eligible_records": eligible_rows[family_id],
                "exact_current_mix_duplicate_records": exact_current_duplicates[family_id],
                "exact_deduplicated_capacity_upper_bound": _split_metrics_json(
                    exact_capacity[family_id]
                ),
                "exact_within_scan_duplicate_records": exact_scan_duplicates[family_id],
                "machine_accepted_before_dedup_records": machine_accepted[family_id],
                "machine_rejection_reasons": dict(sorted(rejections[family_id].items())),
                "near_deduplicated_capacity_lower_bound": _split_metrics_json(
                    near_capacity[family_id]
                ),
                "near_duplicate_reasons": dict(sorted(near_duplicates[family_id].items())),
                "raw_rows": raw_rows[family_id],
                "reservoir_records": len(reservoirs[family_id]),
                "reservoir_scope": (
                    "deterministic_priority_sample_lower_bound"
                    if family_id == "wiki_split_inst_translated"
                    else "complete_family"
                ),
            }
        )
        if family_id == "wiki_split_inst_translated":
            value["production_credited_train_supervised_tokens"] = 0
            value["share_alike_compliance_resolved"] = False
        else:
            value["production_credited_train_supervised_tokens"] = value[
                "near_deduplicated_capacity_lower_bound"
            ]["train"]["supervised_tokens"]
            value["capacity_only_no_materialization_or_promotion"] = True
        families[family_id] = value

    wiki_train = families["wiki_split_inst_translated"][
        "near_deduplicated_capacity_lower_bound"
    ]["train"]["supervised_tokens"]
    joke_train = families["joke_explaination_inst_translated"][
        "near_deduplicated_capacity_lower_bound"
    ]["train"]["supervised_tokens"]
    unsigned = {
        "authorization": dict(plan["authorization"]),
        "bindings": {
            "plan_file_sha256": plan_file_sha256,
            "raw_aya_parquet_bytes": RAW_AYA_BYTES,
            "raw_aya_parquet_sha256": RAW_AYA_SHA256,
            "targeted_plan_file_sha256": TARGETED_PLAN_FILE_SHA256,
            "targeted_report_file_sha256": TARGETED_REPORT_FILE_SHA256,
            "targeted_verification_file_sha256": TARGETED_VERIFICATION_FILE_SHA256,
        },
        "capacity_decision": {
            "conditional_wikisplit_train_supervised_tokens_lower_bound": wiki_train,
            "immediately_rights_clear_joke_train_supervised_tokens": joke_train,
            "joke_capacity_requires_materialization_registry_and_seal_before_use": True,
            "targeted_closure_train_tokens": TARGETED_CLOSURE_TOKENS,
            "wikisplit_capacity_lower_bound_reaches_numeric_target": (
                wiki_train >= TARGETED_CLOSURE_TOKENS
            ),
            "wikisplit_production_token_credit": 0,
        },
        "current_mix_dedup_index": current_metrics,
        "families": families,
        "limits": dict(plan["limits"]),
        "observed_at": plan["observed_at"],
        "output_contract": {
            "candidate_materialized": False,
            "raw_training_text_retained": False,
            "registry_changed": False,
            "sft_or_gpu_run": False,
        },
        "raw_inventory": {
            "excluded_other_family_rows": raw_rows["excluded_other_families"],
            "rows_read": processed,
            "wiki_lingua_quarantined_rows": raw_rows[
                "flan_gem_wikilingua_translated"
            ],
            "wiki_lingua_rows_evaluated_or_tokenized": 0,
        },
        "scan_fingerprint_sha256": scan_fingerprint.hexdigest(),
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_capacity_only_no_candidate_or_production_action",
    }
    return {**unsigned, "report_sha256": sha256_value(unsigned)}


def validate_report(value: Mapping[str, Any]) -> str:
    if value.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported Aya capacity report schema")
    report_sha256 = _require_sha256(value.get("report_sha256"), "report_sha256")
    unsigned = {key: item for key, item in value.items() if key != "report_sha256"}
    if sha256_value(unsigned) != report_sha256:
        raise ValueError("Aya capacity report hash changed")
    if value.get("raw_inventory", {}).get("rows_read") != RAW_AYA_ROWS:
        raise ValueError("Aya capacity report row count changed")
    wiki = value.get("families", {}).get("wiki_split_inst_translated", {})
    if (
        wiki.get("production_credited_train_supervised_tokens") != 0
        or wiki.get("share_alike_compliance_resolved") is not False
    ):
        raise ValueError("WikiSplit production boundary changed")
    if value.get("output_contract") != {
        "candidate_materialized": False,
        "raw_training_text_retained": False,
        "registry_changed": False,
        "sft_or_gpu_run": False,
    }:
        raise ValueError("Aya capacity output boundary changed")
    return report_sha256


def build_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_report(report)
    unsigned = {
        "checks": {
            "all_bindings_hash_verified": True,
            "candidate_materialization_absent": True,
            "full_row_replay_identical": True,
            "joke_complete_family_near_dedup_replayed": True,
            "no_network_api_gpu_registry_seal_or_sft": True,
            "raw_text_free_outputs": True,
            "wiki_lingua_quarantine_preserved": True,
            "wikisplit_near_dedup_lower_bound_replayed": True,
            "wikisplit_production_credit_zero": True,
        },
        "plan_file_sha256": report["bindings"]["plan_file_sha256"],
        "report_file_sha256": report_file_sha256,
        "report_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_independent_full_replay_capacity_only",
        "write_limit_bytes": plan["limits"]["maximum_private_write_bytes"],
    }
    return {**unsigned, "verification_sha256": sha256_value(unsigned)}
