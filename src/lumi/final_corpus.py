"""Fail-closed planning and selection contracts for Lumi's final 80B corpus."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCHEMA_VERSION = 1
PLAN_GATE = "lumi_final_corpus_80b_plan_v0.1"
SELECTION_RESULT_GATE = "lumi_final_corpus_80b_selection_result_v0.1"
SHARD_MANIFEST_GATE = "lumi_final_corpus_80b_shard_manifest_v0.1"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{label} is outside its allowed range")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _resolve_pinned(
    root: Path, pin: Mapping[str, Any], label: str
) -> tuple[Path, str]:
    relative = pin.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError(f"{label} path must be repository-relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escaped the repository") from error
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {relative}")
    digest = _sha256_file(path)
    if digest != _valid_sha256(pin.get("sha256"), f"{label} SHA-256"):
        raise ValueError(f"{label} file changed")
    return path, digest


def _valid_s3_uri(value: Any, *, bucket: str, namespace: str, label: str) -> str:
    parsed = urlparse(str(value))
    if parsed.scheme != "s3" or parsed.netloc != bucket:
        raise ValueError(f"{label} left the final-corpus bucket")
    prefix = f"/{namespace}/"
    if not parsed.path.startswith(prefix) or "/../" in parsed.path:
        raise ValueError(f"{label} left the final-corpus namespace")
    return str(value)


def final_selection_rank(
    seed_sha256: str,
    *,
    category: str,
    component: str,
    document_id: str,
    content_sha256: str,
) -> str:
    """Return the cross-runtime rank used to sample accepted documents."""

    seed = _valid_sha256(seed_sha256, "selection seed")
    content = _valid_sha256(content_sha256, "content SHA-256")
    fields = (seed, category, component, document_id, content)
    if any(not isinstance(value, str) or not value for value in fields):
        raise ValueError("final selection rank fields must be non-empty strings")
    return _sha256_bytes("\0".join(fields).encode())


def final_corpus_plan_fingerprint(plan: Mapping[str, Any]) -> str:
    material = dict(plan)
    material.pop("plan_fingerprint", None)
    return _sha256_bytes(_canonical_json_bytes(material))


def build_final_corpus_plan(
    contract_path: Path,
    *,
    repository_root: Path,
    source_commit: str,
) -> dict[str, Any]:
    """Validate every upstream pin and derive the exact 80B selection plan."""

    root = repository_root.resolve()
    contract = _read_json(contract_path)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported final-corpus contract schema")
    if contract.get("gate") != PLAN_GATE:
        raise ValueError("unexpected final-corpus contract gate")
    if not isinstance(source_commit, str) or re.fullmatch(
        r"[0-9a-f]{40}", source_commit
    ) is None:
        raise ValueError("final-corpus source commit must be a full Git SHA")

    authorization = contract.get("authorization")
    expected_permissions = {
        "full_4b_training": False,
        "model_training": False,
        "paid_glue_canary": False,
        "paid_glue_selection": False,
        "paid_glue_sharding": False,
        "paid_gpu_compute": False,
    }
    if authorization != expected_permissions:
        raise ValueError("final-corpus planning must not authorize paid work or training")

    inputs = contract.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("final-corpus input pins are missing")
    expected_inputs = {
        "baseline",
        "benchmark_registry",
        "data_mix",
        "dedup_result",
        "filter_policy",
        "frozen_tokenizer",
        "training_exclusion_ledger",
    }
    if set(inputs) != expected_inputs:
        raise ValueError("final-corpus input pin set changed")

    baseline_path, baseline_file_sha = _resolve_pinned(
        root, inputs["baseline"], "baseline"
    )
    benchmark_path, benchmark_sha = _resolve_pinned(
        root, inputs["benchmark_registry"], "benchmark registry"
    )
    data_mix_path, data_mix_sha = _resolve_pinned(root, inputs["data_mix"], "data mix")
    filter_path, filter_sha = _resolve_pinned(
        root, inputs["filter_policy"], "filter policy"
    )
    dedup_pin = inputs["dedup_result"]
    if not isinstance(dedup_pin, Mapping):
        raise ValueError("dedup-result pin is invalid")
    dedup_path, dedup_file_sha = _resolve_pinned(
        root,
        {"path": dedup_pin.get("path"), "sha256": dedup_pin.get("file_sha256")},
        "dedup result",
    )
    baseline = _read_json(baseline_path)
    data_mix = _read_json(data_mix_path)
    filter_policy = _read_json(filter_path)
    dedup_result = _read_json(dedup_path)
    dedup_canonical_sha = _sha256_bytes(_canonical_json_bytes(dedup_result))
    if dedup_canonical_sha != _valid_sha256(
        dedup_pin.get("canonical_sha256"), "dedup canonical SHA-256"
    ):
        raise ValueError("dedup-result canonical content changed")

    if dedup_result.get("schema_version") != 1 or dedup_result.get("mode") != "production":
        raise ValueError("final corpus requires the production dedup result")
    if dedup_result.get("gate") != "lumi_glue_production_global_dedup_result_v0.1":
        raise ValueError("final corpus received an unexpected dedup result")
    if dedup_result.get("status") != "pass_category_floors" or dedup_result.get(
        "total_deficit_tokens"
    ) != 0:
        raise ValueError("dedup result has not passed every category floor")
    for name, expected in {
        "release_fingerprint": "2efbadc7360971e33e86c52e787c15fc15045a75fed65f5e16eaafbdfa3a4bb5",
        "authorization_sha256": "e83391e490c29918625129965a5e4836946d3074085e8108ecdf790c31717746",
        "input_index_sha256": "2cd4537a86d6e48e2fa079c1088fef167d856229e208a45284fab216f207d266",
        "map_checkpoint_sha256": "90ccdacf5b8bff23c87061ec68bbce0d0994f811cdfdccaf3abd09d0d01be59b",
    }.items():
        if dedup_result.get(name) != expected:
            raise ValueError(f"dedup-result {name} changed")

    limits = contract.get("limits")
    expected_limits = {
        "candidate_floor_tokens": 120_000_000_000,
        "final_training_tokens": 80_000_000_000,
        "maximum_document_boundary_overshoot_tokens_per_category": 131_072,
        "packed_shard_target_tokens": 500_000_000,
    }
    if limits != expected_limits:
        raise ValueError("final-corpus limits differ from the sealed roadmap")
    if filter_policy.get("filters", {}).get("max_segment_utf8_bytes") != expected_limits[
        "maximum_document_boundary_overshoot_tokens_per_category"
    ]:
        raise ValueError("final-corpus document overshoot cap changed from the segment cap")

    expected_external = {
        "frozen_tokenizer": {
            "bytes": 4_647_195,
            "s3_key": (
                "lumi-corpus/v0.1/control/bootstrap/artifacts/frozen_tokenizer/"
                "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee.json"
            ),
            "sha256": "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee",
        },
        "training_exclusion_ledger": {
            "bytes": 12_476_739,
            "entries": 35_889,
            "s3_key": (
                "lumi-corpus/v0.1/control/bootstrap/artifacts/training_exclusion_ledger/"
                "709027bb95f3e0f956b22a79d4b29f21490ef0b360ba10f2faca7d5db3d06c62.jsonl"
            ),
            "sha256": "709027bb95f3e0f956b22a79d4b29f21490ef0b360ba10f2faca7d5db3d06c62",
        },
    }
    for name, expected in expected_external.items():
        if inputs.get(name) != expected:
            raise ValueError(f"final-corpus external input changed: {name}")
    if benchmark_sha != "be02bdbd7f2c0e807c86cf9678eff86d711aaa34ffdc8e6ebb687a26220ffcbb":
        raise ValueError("final-corpus benchmark registry changed")

    selection = contract.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("final-corpus selection policy is missing")
    if selection.get("algorithm") != "sha256_hash_ranked_document_boundary_v0.1":
        raise ValueError("final-corpus selection algorithm changed")
    if selection.get("preserve_immutable_prefix") is not True:
        raise ValueError("final-corpus immutable prefix must be retained")
    selection_seed = _valid_sha256(selection.get("seed_sha256"), "selection seed")

    if data_mix.get("version") != "0.1" or data_mix.get("status") != "candidate":
        raise ValueError("final-corpus data mix changed")
    mixture = data_mix.get("mixture")
    if not isinstance(mixture, list) or len(mixture) != 7:
        raise ValueError("final-corpus data mix must contain seven categories")
    targets: dict[str, int] = {}
    weights: dict[str, float] = {}
    for row in mixture:
        if not isinstance(row, Mapping):
            raise ValueError("final-corpus mixture row is invalid")
        category = str(row.get("category"))
        if category in targets:
            raise ValueError("final-corpus mixture repeats a category")
        targets[category] = _positive_int(
            row.get("target_tokens_at_80b"), "final category target"
        )
        weights[category] = float(row.get("weight"))
    if sum(targets.values()) != expected_limits["final_training_tokens"]:
        raise ValueError("final-corpus category targets do not sum to 80B")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("final-corpus mixture weights do not sum to one")

    category_results = dedup_result.get("categories")
    if not isinstance(category_results, list):
        raise ValueError("dedup-result categories are missing")
    observed: dict[str, Mapping[str, Any]] = {}
    for row in category_results:
        if not isinstance(row, Mapping):
            raise ValueError("dedup-result category row is invalid")
        category = str(row.get("category"))
        if category in observed:
            raise ValueError("dedup result repeats a category")
        observed[category] = row
    if set(observed) != set(targets):
        raise ValueError("dedup-result category set differs from the 80B mix")

    baseline_totals = baseline.get("totals")
    if not isinstance(baseline_totals, Mapping):
        raise ValueError("verified baseline totals are missing")
    baseline_tokens_total = _positive_int(
        baseline_totals.get("accepted_tokens"), "baseline accepted tokens"
    )
    baseline_documents_total = _positive_int(
        baseline_totals.get("accepted_documents"), "baseline accepted documents"
    )
    dedup_metrics = dedup_result.get("deduplication")
    if not isinstance(dedup_metrics, Mapping):
        raise ValueError("dedup-result metrics are missing")
    if int(dedup_metrics.get("anchor_documents", -1)) != baseline_documents_total:
        raise ValueError("dedup-result anchor count differs from the verified baseline")

    accepted = dedup_result.get("accepted_candidates")
    if not isinstance(accepted, Mapping):
        raise ValueError("dedup-result accepted-candidate inventory is missing")
    accepted_tokens = _positive_int(accepted.get("tokens"), "accepted candidate tokens")
    accepted_documents = _positive_int(
        accepted.get("documents"), "accepted candidate documents"
    )
    _valid_sha256(accepted.get("objects_sha256"), "accepted inventory SHA-256")
    accepted_uri = urlparse(str(accepted.get("s3_uri")))
    if (
        accepted_uri.scheme != "s3"
        or accepted_uri.netloc != str(baseline.get("bucket"))
        or not accepted_uri.path.startswith(
            "/lumi-corpus/v0.1/deduplicated/glue-production/"
        )
    ):
        raise ValueError("accepted candidate inventory left its verified S3 boundary")
    if _positive_int(accepted.get("object_count"), "accepted object count") != 2772:
        raise ValueError("accepted candidate object count changed")
    if _positive_int(accepted.get("object_bytes"), "accepted object bytes") != 507_737_434_761:
        raise ValueError("accepted candidate byte count changed")

    category_plan = []
    baseline_tokens_seen = 0
    baseline_documents_seen = 0
    accepted_tokens_seen = 0
    accepted_documents_seen = 0
    final_floor_tokens = 0
    new_packed_shards = 0
    overshoot_cap = expected_limits[
        "maximum_document_boundary_overshoot_tokens_per_category"
    ]
    packed_target = expected_limits["packed_shard_target_tokens"]
    for mix_row in mixture:
        category = str(mix_row["category"])
        row = observed[category]
        target = targets[category]
        floor = _positive_int(row.get("floor_tokens"), "dedup category floor")
        if floor != target * 3 // 2:
            raise ValueError("dedup category floor is not 1.5 times its final target")
        selected_tokens = _positive_int(
            row.get("selected_tokens"), "dedup selected tokens", allow_zero=True
        )
        selected_documents = _positive_int(
            row.get("selected_documents"), "dedup selected documents", allow_zero=True
        )
        final_tokens = _positive_int(row.get("final_tokens"), "dedup final tokens")
        final_documents = _positive_int(
            row.get("final_documents"), "dedup final documents"
        )
        baseline_tokens = final_tokens - selected_tokens
        baseline_documents = final_documents - selected_documents
        if baseline_tokens < 0 or baseline_documents < 0:
            raise ValueError("dedup category baseline accounting is negative")
        requested_candidate_tokens = target - baseline_tokens
        if requested_candidate_tokens <= 0:
            raise ValueError("verified baseline unexpectedly exceeds a final category target")
        if selected_tokens < requested_candidate_tokens:
            raise ValueError("deduplicated category cannot satisfy its final target")
        if final_tokens < floor or int(row.get("deficit_tokens", -1)) != 0:
            raise ValueError("deduplicated category floor is incomplete")
        category_plan.append(
            {
                "available_candidate_documents": selected_documents,
                "available_candidate_tokens": selected_tokens,
                "baseline_documents": baseline_documents,
                "baseline_tokens": baseline_tokens,
                "category": category,
                "maximum_selected_candidate_tokens": (
                    requested_candidate_tokens + overshoot_cap
                ),
                "mixture_weight": weights[category],
                "new_packed_shards": math.ceil(requested_candidate_tokens / packed_target),
                "requested_candidate_tokens": requested_candidate_tokens,
                "target_final_tokens": target,
            }
        )
        baseline_tokens_seen += baseline_tokens
        baseline_documents_seen += baseline_documents
        accepted_tokens_seen += selected_tokens
        accepted_documents_seen += selected_documents
        final_floor_tokens += floor
        new_packed_shards += math.ceil(requested_candidate_tokens / packed_target)

    if baseline_tokens_seen != baseline_tokens_total:
        raise ValueError("category baseline tokens do not match the verified prefix")
    if baseline_documents_seen != baseline_documents_total:
        raise ValueError("category baseline documents do not match the verified prefix")
    if accepted_tokens_seen != accepted_tokens or accepted_documents_seen != accepted_documents:
        raise ValueError("dedup accepted-candidate totals do not match category accounting")
    if final_floor_tokens != expected_limits["candidate_floor_tokens"]:
        raise ValueError("dedup category floors do not sum to 120B")

    baseline_shards = baseline.get("sealed_shards")
    partial = baseline.get("partial_prefix")
    if not isinstance(baseline_shards, list) or len(baseline_shards) != 2:
        raise ValueError("verified baseline sealed-shard set changed")
    if not isinstance(partial, Mapping) or int(partial.get("accepted_tokens", -1)) <= 0:
        raise ValueError("verified baseline partial shard is missing")
    preserved_baseline_shards = len(baseline_shards) + 1

    output = contract.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("final-corpus output policy is missing")
    bucket = str(output.get("bucket"))
    namespace = str(output.get("namespace"))
    if bucket != baseline.get("bucket") or namespace != "lumi-corpus/v0.1/final-80b":
        raise ValueError("final-corpus output boundary changed")
    if any(
        output.get(name) is not True
        for name in (
            "preserve_selected_parquet",
            "publish_packed_uint16_shards",
            "publish_selection_ledger",
        )
    ):
        raise ValueError("final-corpus required output was disabled")

    if contract.get("canary") != {
        "maximum_dpu": 8,
        "maximum_runtime_minutes": 30,
        "maximum_usd": "1.76",
        "requires_new_owner_approval": True,
    }:
        raise ValueError("final-corpus canary boundary changed")
    if contract.get("production") != {
        "maximum_dpu": 100,
        "maximum_runtime_minutes": 1440,
        "requires_new_owner_approval": True,
    }:
        raise ValueError("final-corpus production boundary changed")

    plan = {
        "authorization": dict(authorization),
        "canary": dict(contract["canary"]),
        "categories": category_plan,
        "gate": PLAN_GATE,
        "inputs": {
            "baseline": {
                "file_sha256": baseline_file_sha,
                "path": str(baseline_path.relative_to(root)),
            },
            "benchmark_registry": {
                "file_sha256": benchmark_sha,
                "path": str(benchmark_path.relative_to(root)),
            },
            "data_mix": {
                "file_sha256": data_mix_sha,
                "path": str(data_mix_path.relative_to(root)),
            },
            "dedup_result": {
                "accepted_candidates": dict(accepted),
                "canonical_sha256": dedup_canonical_sha,
                "file_sha256": dedup_file_sha,
                "path": str(dedup_path.relative_to(root)),
            },
            "filter_policy": {
                "file_sha256": filter_sha,
                "path": str(filter_path.relative_to(root)),
            },
            "frozen_tokenizer": dict(inputs["frozen_tokenizer"]),
            "training_exclusion_ledger": dict(inputs["training_exclusion_ledger"]),
        },
        "limits": {
            **dict(limits),
            "maximum_final_selected_tokens": (
                expected_limits["final_training_tokens"] + len(targets) * overshoot_cap
            ),
            "training_token_limit": expected_limits["final_training_tokens"],
        },
        "output": {
            **dict(output),
            "expected_new_packed_shards": new_packed_shards,
            "expected_total_packed_shards": new_packed_shards + preserved_baseline_shards,
            "preserved_baseline_shards": preserved_baseline_shards,
        },
        "production": dict(contract["production"]),
        "schema_version": SCHEMA_VERSION,
        "selection": {
            **dict(selection),
            "rank_fields": [
                "seed_sha256",
                "category",
                "component",
                "document_id",
                "content_sha256",
            ],
            "seed_sha256": selection_seed,
        },
        "source_commit": source_commit,
        "status": "ready_for_bounded_glue_canary_approval",
    }
    plan["plan_fingerprint"] = final_corpus_plan_fingerprint(plan)
    return plan


def select_hash_ranked_token_prefix(
    candidates: Any,
    *,
    seed_sha256: str,
    target_tokens: int,
    maximum_overshoot_tokens: int,
    partitions: int | None = None,
) -> dict[str, Any]:
    """Select a deterministic category sample without a single-partition window."""

    from pyspark.sql import functions as F

    seed = _valid_sha256(seed_sha256, "selection seed")
    target = _positive_int(target_tokens, "selection target")
    overshoot_cap = _positive_int(
        maximum_overshoot_tokens, "selection overshoot cap", allow_zero=True
    )
    required = {
        "category",
        "component",
        "content_sha256",
        "document_id",
        "segment_index",
        "source_line",
        "source_ordinal",
        "token_count",
    }
    if not required.issubset(candidates.columns):
        raise ValueError("final selection input schema is missing required columns")
    if partitions is None:
        partitions = int(candidates.sparkSession.conf.get("spark.sql.shuffle.partitions"))
    if not isinstance(partitions, int) or isinstance(partitions, bool) or partitions < 1:
        raise ValueError("final selection partition count must be positive")
    rank_column = "_lumi_selection_rank"
    if rank_column in candidates.columns:
        raise ValueError("final selection input already contains the private rank column")
    material = F.concat_ws(
        "\0",
        F.lit(seed),
        F.col("category"),
        F.col("component"),
        F.col("document_id"),
        F.col("content_sha256"),
    )
    ranked = candidates.withColumn(rank_column, F.sha2(material, 256))
    order = [
        rank_column,
        "content_sha256",
        "document_id",
        "source_ordinal",
        "source_line",
        "segment_index",
    ]
    ordered = (
        ranked.repartitionByRange(partitions, *[F.col(name) for name in order])
        .sortWithinPartitions(*[F.col(name).asc() for name in order])
        .persist()
    )

    def summarize(index: int, rows: Any) -> Any:
        documents = 0
        tokens = 0
        for row in rows:
            token_count = int(row["token_count"])
            if token_count <= 0:
                raise ValueError("final selection row has no tokens")
            documents += 1
            tokens += token_count
        yield index, documents, tokens

    summaries = sorted(ordered.rdd.mapPartitionsWithIndex(summarize).collect())
    available_documents = sum(int(row[1]) for row in summaries)
    available_tokens = sum(int(row[2]) for row in summaries)
    if available_documents == 0 or available_tokens < target:
        ordered.unpersist()
        raise ValueError("final selection category cannot reach its target")
    before_tokens = 0
    cutoff_partition = summaries[-1][0]
    for partition, _, tokens in summaries:
        if before_tokens + int(tokens) >= target:
            cutoff_partition = int(partition)
            break
        before_tokens += int(tokens)
    cutoff_target = target - before_tokens

    def select(index: int, rows: Any) -> Any:
        if index < cutoff_partition:
            yield from rows
            return
        if index > cutoff_partition:
            return
        selected_tokens = 0
        for row in rows:
            if selected_tokens >= cutoff_target:
                break
            token_count = int(row["token_count"])
            if token_count <= 0:
                raise ValueError("final selection row has no tokens")
            yield row
            selected_tokens += token_count

    selected = candidates.sparkSession.createDataFrame(
        ordered.rdd.mapPartitionsWithIndex(select), schema=ordered.schema
    ).persist()
    totals = selected.agg(
        F.count(F.lit(1)).alias("documents"), F.sum("token_count").alias("tokens")
    ).first()
    selected_documents = int(totals["documents"] or 0)
    selected_tokens = int(totals["tokens"] or 0)
    overshoot = selected_tokens - target
    if selected_documents <= 0 or selected_tokens < target:
        selected.unpersist()
        ordered.unpersist()
        raise RuntimeError("final selection stopped before its target")
    if overshoot > overshoot_cap:
        selected.unpersist()
        ordered.unpersist()
        raise RuntimeError("final selection exceeded its document-boundary overshoot cap")
    last = selected.orderBy(*[F.col(name).desc() for name in order]).first()
    assert last is not None
    cutoff = {name: last[name] for name in order}
    ordered.unpersist()
    return {
        "available_documents": available_documents,
        "available_tokens": available_tokens,
        "cutoff": cutoff,
        "overshoot_tokens": overshoot,
        "selected": selected,
        "selected_documents": selected_documents,
        "selected_tokens": selected_tokens,
    }


def final_selection_result_fingerprint(result: Mapping[str, Any]) -> str:
    material = dict(result)
    material.pop("result_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(material))


def validate_final_selection_result(
    result: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate exact input binding, category targets, and selected inventories."""

    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported final selection result schema")
    if result.get("gate") != SELECTION_RESULT_GATE or result.get("status") != "complete":
        raise ValueError("final selection result is incomplete")
    if result.get("plan_fingerprint") != plan.get("plan_fingerprint"):
        raise ValueError("final selection result targets another plan")
    expected_dedup_sha = plan["inputs"]["dedup_result"]["canonical_sha256"]
    if result.get("dedup_result_sha256") != expected_dedup_sha:
        raise ValueError("final selection dedup-result binding changed")
    if result.get("selection_seed_sha256") != plan["selection"]["seed_sha256"]:
        raise ValueError("final selection seed changed")
    if result.get("training_token_limit") != plan["limits"]["training_token_limit"]:
        raise ValueError("final selection training token limit changed")
    rows = result.get("categories")
    if not isinstance(rows, list):
        raise ValueError("final selection category results are missing")
    expected = {str(row["category"]): row for row in plan["categories"]}
    observed: dict[str, Mapping[str, Any]] = {}
    total_documents = 0
    total_tokens = 0
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("category")) not in expected:
            raise ValueError("final selection category row is invalid")
        category = str(row["category"])
        if category in observed:
            raise ValueError("final selection repeats a category")
        observed[category] = row
        planned = expected[category]
        selected_tokens = _positive_int(
            row.get("selected_candidate_tokens"), "selected candidate tokens"
        )
        selected_documents = _positive_int(
            row.get("selected_candidate_documents"), "selected candidate documents"
        )
        if selected_tokens < int(planned["requested_candidate_tokens"]):
            raise ValueError("final selection category stopped below target")
        if selected_tokens > int(planned["maximum_selected_candidate_tokens"]):
            raise ValueError("final selection category exceeded its overshoot cap")
        if selected_documents > int(planned["available_candidate_documents"]):
            raise ValueError("final selection category selected too many documents")
        if int(row.get("available_candidate_tokens", -1)) != int(
            planned["available_candidate_tokens"]
        ):
            raise ValueError("final selection available tokens changed")
        if int(row.get("available_candidate_documents", -1)) != int(
            planned["available_candidate_documents"]
        ):
            raise ValueError("final selection available documents changed")
        final_tokens = int(planned["baseline_tokens"]) + selected_tokens
        final_documents = int(planned["baseline_documents"]) + selected_documents
        if int(row.get("final_tokens", -1)) != final_tokens:
            raise ValueError("final selection category token conservation failed")
        if int(row.get("final_documents", -1)) != final_documents:
            raise ValueError("final selection category document conservation failed")
        if final_tokens < int(planned["target_final_tokens"]):
            raise ValueError("final selection category remains below its 80B mix target")
        cutoff = row.get("cutoff")
        if not isinstance(cutoff, Mapping) or set(cutoff) != {
            "_lumi_selection_rank",
            "content_sha256",
            "document_id",
            "segment_index",
            "source_line",
            "source_ordinal",
        }:
            raise ValueError("final selection category cutoff is missing")
        _valid_sha256(cutoff.get("_lumi_selection_rank"), "selection cutoff rank")
        _valid_sha256(cutoff.get("content_sha256"), "selection cutoff content")
        if not isinstance(cutoff.get("document_id"), str) or not cutoff["document_id"]:
            raise ValueError("final selection cutoff document ID is invalid")
        for name in ("segment_index", "source_line", "source_ordinal"):
            _positive_int(
                cutoff.get(name),
                f"selection cutoff {name}",
                allow_zero=name != "source_line",
            )
        inventory = row.get("selected_inventory")
        if not isinstance(inventory, Mapping):
            raise ValueError("final selection category inventory is missing")
        _valid_sha256(inventory.get("objects_sha256"), "selection inventory SHA-256")
        _positive_int(inventory.get("object_count"), "selection object count")
        _positive_int(inventory.get("object_bytes"), "selection object bytes")
        _valid_s3_uri(
            inventory.get("s3_uri"),
            bucket=str(plan["output"]["bucket"]),
            namespace=str(plan["output"]["namespace"]),
            label="selection inventory",
        )
        total_documents += final_documents
        total_tokens += final_tokens
    if set(observed) != set(expected):
        raise ValueError("final selection is missing one or more categories")
    if int(result.get("final_documents", -1)) != total_documents:
        raise ValueError("final selection total document conservation failed")
    if int(result.get("final_tokens", -1)) != total_tokens:
        raise ValueError("final selection total token conservation failed")
    if total_tokens < int(plan["limits"]["final_training_tokens"]):
        raise ValueError("final selection is below 80B")
    if total_tokens > int(plan["limits"]["maximum_final_selected_tokens"]):
        raise ValueError("final selection exceeds its aggregate overshoot cap")
    expected_result_sha = final_selection_result_fingerprint(result)
    if result.get("result_sha256") != expected_result_sha:
        raise ValueError("final selection result fingerprint changed")
    return dict(result)


def validate_final_shard_manifest(
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    selection_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal every selected token to one immutable packed uint16 shard."""

    validate_final_selection_result(selection_result, plan)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported final shard manifest schema")
    if manifest.get("gate") != SHARD_MANIFEST_GATE or manifest.get("status") != "sealed":
        raise ValueError("final shard manifest is not sealed")
    if manifest.get("plan_fingerprint") != plan.get("plan_fingerprint"):
        raise ValueError("final shard manifest targets another plan")
    if manifest.get("selection_result_sha256") != selection_result.get("result_sha256"):
        raise ValueError("final shard manifest selection binding changed")
    if manifest.get("training_token_limit") != plan["limits"]["training_token_limit"]:
        raise ValueError("final shard training token limit changed")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("final shard manifest contains no shards")
    if len(shards) != int(plan["output"]["expected_total_packed_shards"]):
        raise ValueError("final shard count changed")
    expected_categories = {
        str(row["category"]): int(row["final_tokens"])
        for row in selection_result["categories"]
    }
    category_tokens = {category: 0 for category in expected_categories}
    object_keys: set[str] = set()
    object_hashes: set[str] = set()
    total_tokens = 0
    total_documents = 0
    maximum_tokens = int(plan["limits"]["packed_shard_target_tokens"]) + int(
        plan["limits"]["maximum_document_boundary_overshoot_tokens_per_category"]
    )
    for expected_sequence, shard in enumerate(shards):
        if not isinstance(shard, Mapping):
            raise ValueError("final shard row is invalid")
        if int(shard.get("sequence", -1)) != expected_sequence:
            raise ValueError("final shard sequence is not contiguous")
        category = str(shard.get("category"))
        if category not in expected_categories:
            raise ValueError("final shard category is invalid")
        tokens = _positive_int(shard.get("tokens"), "final shard tokens")
        documents = _positive_int(shard.get("documents"), "final shard documents")
        if tokens > maximum_tokens:
            raise ValueError("final shard exceeds its document-boundary overshoot cap")
        if _positive_int(shard.get("object_bytes"), "final shard bytes") != tokens * 2:
            raise ValueError("final shard uint16 byte conservation failed")
        digest = _valid_sha256(shard.get("object_sha256"), "final shard SHA-256")
        key = _valid_s3_uri(
            shard.get("s3_uri"),
            bucket=str(plan["output"]["bucket"]),
            namespace=str(plan["output"]["namespace"]),
            label="final shard",
        )
        if key in object_keys or digest in object_hashes:
            raise ValueError("final shard manifest repeats an object")
        object_keys.add(key)
        object_hashes.add(digest)
        category_tokens[category] += tokens
        total_tokens += tokens
        total_documents += documents
    if category_tokens != expected_categories:
        raise ValueError("final shard category token conservation failed")
    if total_tokens != int(selection_result["final_tokens"]):
        raise ValueError("final shard total token conservation failed")
    if total_documents != int(selection_result["final_documents"]):
        raise ValueError("final shard total document conservation failed")
    if int(manifest.get("tokens", -1)) != total_tokens or int(
        manifest.get("documents", -1)
    ) != total_documents:
        raise ValueError("final shard manifest totals changed")
    return dict(manifest)
