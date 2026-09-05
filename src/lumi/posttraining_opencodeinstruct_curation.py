"""Bounded CPU-only curation for one pinned OpenCodeInstruct shard."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

from .data_quality import (
    BenchmarkRegistry,
    MinHashNearDuplicateIndex,
    normalize_for_matching,
    validate_quality_config,
)
from .posttraining_data import SFTPolicy, load_sft_policy
from .posttraining_format import (
    ChatTemplate,
    ChatTokenizer,
    encode_sft_messages,
    load_chat_template,
)
from .posttraining_owner_review import build_owner_review_html

PLAN_SCHEMA_VERSION = "lumi.sft-opencodeinstruct-single-shard-run.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-opencodeinstruct-single-shard-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-opencodeinstruct-single-shard-verification.v0.1"
)

SOURCE_ID = "nvidia_opencodeinstruct"
SOURCE_DATASET_ID = "nvidia/OpenCodeInstruct"
SOURCE_REVISION = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
SOURCE_LICENSE = "cc-by-4.0"
README_PATH = "README.md"
README_BYTES = 4_377
README_GIT_BLOB_SHA1 = "8226aea38e6986a3a243ea161739ed3ad45ac600"
SHARD_PATH = "data/train-00007-of-00050.parquet"
SHARD_BYTES = 113_867_035
SHARD_SHA256 = "2839de0d6c4d1691710dbc63f43cd816e7c02a943f8e2c07087bbf5fa26172ce"
SOURCE_ROWS = 100_000
OWNER_AUTHORIZATION_STATEMENT = (
    "I approve one bounded CPU-only OpenCodeInstruct single-shard acquisition and "
    "curation at revision 8f3ba5bafe4d6e8db46082cf7ae6741bc370604d, limited to "
    "README.md (4,377 bytes) and data/train-00007-of-00050.parquet (113,867,035 "
    "bytes; SHA-256 2839de0d6c4d1691710dbc63f43cd816e7c02a943f8e2c07087bbf5fa26172ce), "
    "maximum 128 MiB network / 512 MiB decoded / 180 minutes / 2,300,000 retained "
    "train supervised tokens. It may perform schema validation, published unit-test "
    "verification, quality filtering, PII/secret/malware filtering, exact and near "
    "deduplication, benchmark decontamination, 4K-fit, frozen-tokenizer measurement, "
    "attribution, and create a deterministic owner-review packet. No external API, "
    "GPU, registry promotion, dataset seal, SFT training, unrelated data, or external "
    "distribution."
)

EXPECTED_COLUMNS = (
    "id",
    "input",
    "output",
    "domain",
    "generation_algorithm",
    "llm_judgement",
    "unit_tests",
    "tests_execution_status",
    "average_test_score",
)
JUDGEMENT_CRITERIA = (
    "requirement_conformance",
    "logical_correctness",
    "edge_case_consideration",
)
REQUIRED_CODING_BENCHMARKS = {
    "humaneval",
    "mbpp",
    "livecodebench",
    "bigcodebench",
}

EXPECTED_AUTHORIZATION = {
    "candidate_registry_promotion_authorized": False,
    "cpu_only_curation_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": True,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "unrelated_dataset_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_decoded_bytes": 536_870_912,
    "maximum_network_bytes": 134_217_728,
    "maximum_retained_train_supervised_tokens": 2_300_000,
    "maximum_runtime_minutes": 180,
    "maximum_source_objects": 2,
    "maximum_source_rows": SOURCE_ROWS,
    "owner_sample_records": 32,
    "target_train_supervised_tokens": 2_100_000,
    "validation_fraction": 0.05,
}
EXPECTED_POLICY = {
    "average_test_score_required": 1.0,
    "benchmark_decontamination": {
        "active_registry_required": True,
        "required_coding_benchmarks": sorted(REQUIRED_CODING_BENCHMARKS),
    },
    "exact_duplicate_scope": "normalized_full_conversation",
    "four_k_fit_required": True,
    "judgement_minimum_each": 4,
    "judgement_required_fields": list(JUDGEMENT_CRITERIA),
    "local_unit_test_sandbox_required": True,
    "near_duplicate": {
        "comparison_unit": "role_tagged_full_conversation",
        "threshold": 0.9,
    },
    "published_test_count": 10,
    "published_test_status_required": "pass",
    "split_assignment": "sha256_modulo_20_validation_v0.1",
}
EXPECTED_OUTPUT = {
    "private_root": (
        "../lumi-artifacts/4b-sft-opencodeinstruct-single-shard-v0.1/"
        "8f3ba5bafe4d/attempt-001"
    ),
    "raw_text_free_report_path": (
        "evidence/lumi-4b-sft-opencodeinstruct-single-shard-v0.1.json"
    ),
    "raw_text_free_verification_path": (
        "evidence/lumi-4b-sft-opencodeinstruct-single-shard-verification-v0.1.json"
    ),
    "raw_training_text_in_git": False,
}
EXPECTED_BINDING_PATHS = {
    "benchmark_registry": "data/benchmark-signatures-v0.1.jsonl.gz",
    "bulk_source_shortlist": "configs/lumi-4b-sft-bulk-source-shortlist-v0.1.json",
    "chat_template": "configs/lumi-4b-chat-template-v0.1.json",
    "curation_implementation": "src/lumi/posttraining_opencodeinstruct_curation.py",
    "curation_runner": "scripts/run_opencodeinstruct_curation.py",
    "curation_verifier": "scripts/verify_opencodeinstruct_curation.py",
    "data_contract": "configs/lumi-4b-sft-data-contract-v0.1.json",
    "data_quality_implementation": "src/lumi/data_quality.py",
    "filtering_implementation": "src/lumi/posttraining_data.py",
    "owner_review_implementation": "src/lumi/posttraining_owner_review.py",
    "quality_config": "configs/data-quality-v0.1.json",
    "tokenizer_config": (
        "../lumi-artifacts/4b-segment2-throughput-gate-v0.1/8bedfbc868e8/"
        "staging/input/tokenizer/tokenizer_config.json"
    ),
    "tokenizer_json": (
        "../lumi-artifacts/4b-segment2-throughput-gate-v0.1/8bedfbc868e8/"
        "staging/input/tokenizer/tokenizer.json"
    ),
    "unit_test_worker": "scripts/opencodeinstruct_unit_test_worker.py",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_MALWARE_PATTERN = re.compile(
    r"\b(?:backdoor|botnet|credential\s*steal(?:er|ing)?|cryptominer|ddos|exfiltrat|"
    r"keylogger|malware|phishing|ransomware|reverse\s+shell|rootkit|"
    r"shellcode|spyware)\b",
    re.IGNORECASE,
)
_PYTHON_FENCE = "```python"
_FENCE = "```"


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


def git_blob_sha1(path: str | Path) -> str:
    value = Path(path).read_bytes()
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()  # noqa: S324 - Git object identity.


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
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
        raise ValueError(f"invalid OpenCodeInstruct binding path: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"OpenCodeInstruct binding escapes workspace: {name}")
    if not resolved.is_file():
        raise FileNotFoundError(f"OpenCodeInstruct binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"OpenCodeInstruct binding changed: {name}")
    return resolved


def load_opencodeinstruct_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-07",
        "owner_authorization_statement": OWNER_AUTHORIZATION_STATEMENT,
        "output": EXPECTED_OUTPUT,
        "policy": EXPECTED_POLICY,
        "schema_version": PLAN_SCHEMA_VERSION,
        "source": {
            "dataset_id": SOURCE_DATASET_ID,
            "declared_license_id": SOURCE_LICENSE,
            "objects": [
                {
                    "git_blob_sha1": README_GIT_BLOB_SHA1,
                    "path": README_PATH,
                    "size_bytes": README_BYTES,
                },
                {
                    "path": SHARD_PATH,
                    "sha256": SHARD_SHA256,
                    "size_bytes": SHARD_BYTES,
                },
            ],
            "repository_url": (
                "https://huggingface.co/datasets/nvidia/OpenCodeInstruct"
            ),
            "revision": SOURCE_REVISION,
            "source_id": SOURCE_ID,
            "source_rows": SOURCE_ROWS,
            "source_split": "train",
        },
        "status": "authorized_cpu_only_single_shard_curation_no_promotion",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"OpenCodeInstruct plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(EXPECTED_BINDING_PATHS):
        raise ValueError("OpenCodeInstruct binding set changed")
    for name, expected_path in EXPECTED_BINDING_PATHS.items():
        binding = bindings[name]
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise ValueError(f"OpenCodeInstruct binding path changed: {name}")
        resolve_binding(repository_root, binding, name=name)
    return plan


def verify_raw_objects(raw_root: str | Path) -> dict[str, Any]:
    root = Path(raw_root).resolve()
    readme = root / README_PATH
    shard = root / SHARD_PATH
    observed = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if observed != {README_PATH, SHARD_PATH}:
        raise ValueError("OpenCodeInstruct raw object inventory changed")
    if readme.stat().st_size != README_BYTES or git_blob_sha1(readme) != README_GIT_BLOB_SHA1:
        raise ValueError("OpenCodeInstruct README identity changed")
    if shard.stat().st_size != SHARD_BYTES or file_sha256(shard) != SHARD_SHA256:
        raise ValueError("OpenCodeInstruct shard identity changed")
    return {
        "network_bytes": README_BYTES + SHARD_BYTES,
        "objects": [
            {
                "bytes": README_BYTES,
                "git_blob_sha1": README_GIT_BLOB_SHA1,
                "path": README_PATH,
            },
            {"bytes": SHARD_BYTES, "path": SHARD_PATH, "sha256": SHARD_SHA256},
        ],
    }


def _parse_structured(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return None


def strip_python_fence(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip().casefold() != _PYTHON_FENCE:
        return None
    if lines[-1].strip() != _FENCE:
        return None
    code = "\n".join(lines[1:-1]).strip()
    return code or None


def _judgement_scores(value: Any) -> dict[str, int] | None:
    parsed = _parse_structured(value)
    if not isinstance(parsed, Mapping) or set(parsed) != set(JUDGEMENT_CRITERIA):
        return None
    scores: dict[str, int] = {}
    for criterion in JUDGEMENT_CRITERIA:
        item = parsed.get(criterion)
        if not isinstance(item, Mapping) or set(item) != {"justification", "score"}:
            return None
        score = item.get("score")
        justification = item.get("justification")
        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 1 <= score <= 5
            or not isinstance(justification, str)
            or not justification.strip()
        ):
            return None
        scores[criterion] = score
    return scores


def _published_tests(row: Mapping[str, Any]) -> tuple[list[str], list[str]] | None:
    tests = _parse_structured(row.get("unit_tests"))
    statuses = _parse_structured(row.get("tests_execution_status"))
    if (
        not isinstance(tests, list)
        or not isinstance(statuses, list)
        or len(tests) != EXPECTED_POLICY["published_test_count"]
        or len(statuses) != len(tests)
        or any(not isinstance(test, str) or not test.strip() for test in tests)
        or any(not isinstance(status, str) for status in statuses)
    ):
        return None
    return tests, statuses


def _row_envelope(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if set(row) != set(EXPECTED_COLUMNS):
        return None, "schema"
    record_id = row.get("id")
    prompt = row.get("input")
    response = row.get("output")
    if not all(isinstance(item, str) and item.strip() for item in (record_id, prompt, response)):
        return None, "required_text"
    if row.get("domain") != "algorithmic" or row.get("generation_algorithm") != "self-instruct":
        return None, "domain_or_generation_algorithm"
    if row.get("average_test_score") != EXPECTED_POLICY["average_test_score_required"]:
        return None, "published_average_test_score"
    scores = _judgement_scores(row.get("llm_judgement"))
    if scores is None:
        return None, "judgement_schema"
    if min(scores.values()) < EXPECTED_POLICY["judgement_minimum_each"]:
        return None, "judgement_below_floor"
    published = _published_tests(row)
    if published is None:
        return None, "published_tests_schema"
    tests, statuses = published
    if any(status.strip().casefold() != "pass" for status in statuses):
        return None, "published_test_failed"
    code = strip_python_fence(response)
    if code is None:
        return None, "python_fence"
    combined = f"{prompt}\n{response}"
    if _MALWARE_PATTERN.search(combined):
        return None, "unsafe_code_intent_pattern"

    messages = [
        {"content": prompt.strip(), "role": "user"},
        {"content": response.strip(), "role": "assistant"},
    ]
    content_sha256 = sha256_value(messages)
    transform_sha256 = sha256_value(
        {
            "messages": messages,
            "transform": "opencodeinstruct_one_turn_projection_v0.1",
        }
    )
    quality_score = round(sum(scores.values()) / (5 * len(scores)), 6)
    record = {
        "language": "en",
        "messages": messages,
        "provenance": {
            "consent_basis": "source_level_open_license",
            "content_sha256": content_sha256,
            "transform_sha256": transform_sha256,
        },
        "quality": {
            "pii_scan": "passed",
            "review_method": (
                "published_tests_plus_local_sandbox_plus_judgement_and_lumi_filters_v0.1"
            ),
            "review_status": "accepted",
            "score": quality_score,
            "secret_scan": "passed",
        },
        "record_id": f"opencode-{record_id}",
        "safety": {
            "policy_version": "lumi-sft-code-safety-pattern-and-sandbox-v0.1",
            "risk_labels": [],
        },
        "schema_version": "lumi.sft.v0.1",
        "source": {
            "dataset_id": SOURCE_DATASET_ID,
            "document_id": record_id,
            "license_id": SOURCE_LICENSE,
            "revision": SOURCE_REVISION,
            "source_id": SOURCE_ID,
            "uri": (
                f"https://huggingface.co/datasets/{SOURCE_DATASET_ID}/blob/"
                f"{SOURCE_REVISION}/{SHARD_PATH}#row-{record_id}"
            ),
        },
        "split": "train",
        "task_category": "coding",
    }
    tests_sha256 = sha256_value(tests)
    code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
    attribution = {
        "code_sha256": code_sha256,
        "content_sha256": content_sha256,
        "domain": row["domain"],
        "generation_algorithm": row["generation_algorithm"],
        "judgement_scores": scores,
        "output_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "published_average_test_score": 1.0,
        "published_status_sha256": sha256_value([status.casefold() for status in statuses]),
        "record_id": record["record_id"],
        "source_object_path": SHARD_PATH,
        "source_object_sha256": SHARD_SHA256,
        "source_record_id": record_id,
        "source_revision": SOURCE_REVISION,
        "tests_sha256": tests_sha256,
        "transform_sha256": transform_sha256,
    }
    return {
        "attribution": attribution,
        "code": code,
        "record": record,
        "unit_tests": tests,
    }, "eligible"


def load_source_envelopes(
    shard_path: str | Path,
    *,
    maximum_decoded_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    path = Path(shard_path)
    if path.stat().st_size != SHARD_BYTES or file_sha256(path) != SHARD_SHA256:
        raise ValueError("OpenCodeInstruct shard changed before decode")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != SOURCE_ROWS or parquet.schema_arrow.names != list(
        EXPECTED_COLUMNS
    ):
        raise ValueError("OpenCodeInstruct parquet metadata changed")
    envelopes: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    decoded_bytes = 0
    rows_seen = 0
    source_ids: set[str] = set()
    for batch in parquet.iter_batches(batch_size=2_048, columns=list(EXPECTED_COLUMNS)):
        decoded_bytes += batch.nbytes
        if decoded_bytes > maximum_decoded_bytes:
            raise ValueError("OpenCodeInstruct decoded byte limit exceeded")
        for row in batch.to_pylist():
            rows_seen += 1
            source_id = row.get("id")
            if not isinstance(source_id, str) or source_id in source_ids:
                rejection_counts["duplicate_or_invalid_source_id"] += 1
                continue
            source_ids.add(source_id)
            envelope, reason = _row_envelope(row)
            if envelope is None:
                rejection_counts[reason] += 1
            else:
                envelopes.append(envelope)
    if rows_seen != SOURCE_ROWS or len(source_ids) != SOURCE_ROWS:
        raise ValueError("OpenCodeInstruct source row identity changed")
    return envelopes, {
        "decoded_bytes": decoded_bytes,
        "eligible_before_local_verification": len(envelopes),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "rows": rows_seen,
    }


def run_local_unit_test_verifier(
    envelopes: Sequence[Mapping[str, Any]],
    *,
    worker_path: str | Path,
    python_executable: str = sys.executable,
    timeout_seconds: int = 1_200,
) -> tuple[list[dict[str, Any]], set[str], dict[str, int]]:
    payload = []
    expected: dict[str, tuple[str, str]] = {}
    for envelope in envelopes:
        attribution = envelope["attribution"]
        record_id = str(envelope["record"]["record_id"])
        expected[record_id] = (
            str(attribution["code_sha256"]),
            str(attribution["tests_sha256"]),
        )
        payload.append(
            json.dumps(
                {
                    "code": envelope["code"],
                    "record_id": record_id,
                    "unit_tests": envelope["unit_tests"],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    completed = subprocess.run(
        [python_executable, "-I", "-S", str(Path(worker_path).resolve())],
        check=False,
        capture_output=True,
        input=("\n".join(payload) + "\n").encode("utf-8"),
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "OpenCodeInstruct unit-test worker failed raw-text-free: "
            f"returncode={completed.returncode} stderr_sha256="
            f"{hashlib.sha256(completed.stderr).hexdigest()}"
        )
    results = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    if len(results) != len(envelopes):
        raise ValueError("OpenCodeInstruct verifier result count changed")
    ids = [result.get("record_id") for result in results]
    if set(ids) != set(expected) or len(ids) != len(set(ids)):
        raise ValueError("OpenCodeInstruct verifier IDs changed")
    passed: set[str] = set()
    reasons: Counter[str] = Counter()
    ledger: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: str(item["record_id"])):
        record_id = str(result["record_id"])
        code_sha256, tests_sha256 = expected[record_id]
        if result.get("code_sha256") != code_sha256 or result.get("tests_sha256") != tests_sha256:
            raise ValueError("OpenCodeInstruct verifier content identity changed")
        verified = result.get("verified") is True
        reason = result.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError("OpenCodeInstruct verifier reason changed")
        if verified:
            if reason != "verified":
                raise ValueError("OpenCodeInstruct verifier pass reason changed")
            passed.add(record_id)
        reasons[reason] += 1
        ledger.append(
            {
                "code_sha256": code_sha256,
                "reason": reason,
                "record_id": record_id,
                "tests_sha256": tests_sha256,
                "verified": verified,
            }
        )
    return ledger, passed, dict(sorted(reasons.items()))


def apply_conversation_near_dedup(
    records: Sequence[Mapping[str, Any]],
    *,
    near_duplicate_config: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config = dict(near_duplicate_config)
    config["threshold"] = EXPECTED_POLICY["near_duplicate"]["threshold"]
    index = MinHashNearDuplicateIndex(config)
    accepted: list[Mapping[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    ordered = sorted(
        records,
        key=lambda row: (
            -float(row["quality"]["score"]),
            hashlib.sha256(str(row["record_id"]).encode()).hexdigest(),
        ),
    )
    for record in ordered:
        record_id = str(record["record_id"])
        text = normalize_for_matching(
            "\n".join(
                f"{message['role']} {message['content']}" for message in record["messages"]
            )
        )
        signature = index.signature(text)
        duplicate_of = index.duplicate_of(signature)
        accepted_row = duplicate_of is None
        if accepted_row:
            index.add(record_id, signature)
            accepted.append(record)
        ledger.append(
            {
                "accepted": accepted_row,
                "duplicate_of": duplicate_of,
                "record_id": record_id,
                "signature_sha256": index.fingerprint(signature),
            }
        )
    return accepted, ledger, {
        "accepted": len(accepted),
        "algorithm": "minhash_lsh_with_exact_jaccard_confirmation_v0.1",
        "comparison_unit": "role_tagged_full_conversation",
        "rejected": len(records) - len(accepted),
        "threshold": config["threshold"],
    }


def _nearest_rank(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def measure_and_select(
    records: Sequence[Mapping[str, Any]],
    *,
    maximum_train_supervised_tokens: int,
    target_train_supervised_tokens: int,
    validation_fraction: float,
    template: ChatTemplate,
    tokenizer: ChatTokenizer,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    measured: list[tuple[dict[str, Any], int, int]] = []
    overlength = 0
    for source_record in records:
        record = dict(source_record)
        digest = hashlib.sha256(
            f"opencode-validation-v0.1\0{record['record_id']}".encode()
        ).digest()
        record["split"] = (
            "validation" if int.from_bytes(digest[:8], "big") % 20 == 0 else "train"
        )
        try:
            encoded = encode_sft_messages(
                record["messages"], tokenizer=tokenizer, template=template
            )
        except ValueError as error:
            if "maximum is 4096" not in str(error):
                raise
            overlength += 1
            continue
        measured.append((record, len(encoded.input_ids), encoded.supervised_tokens))
    by_split: defaultdict[str, list[tuple[dict[str, Any], int, int]]] = defaultdict(list)
    for item in measured:
        by_split[str(item[0]["split"])].append(item)
    for items in by_split.values():
        items.sort(
            key=lambda item: (
                -float(item[0]["quality"]["score"]),
                hashlib.sha256(str(item[0]["record_id"]).encode()).hexdigest(),
            )
        )

    selected_train: list[tuple[dict[str, Any], int, int]] = []
    train_tokens = 0
    for item in by_split["train"]:
        if train_tokens >= target_train_supervised_tokens:
            break
        if train_tokens + item[2] > maximum_train_supervised_tokens:
            continue
        selected_train.append(item)
        train_tokens += item[2]
    if train_tokens < target_train_supervised_tokens:
        raise ValueError("OpenCodeInstruct candidate did not reach the target train tokens")

    validation_target = math.ceil(
        train_tokens * validation_fraction / (1.0 - validation_fraction)
    )
    selected_validation: list[tuple[dict[str, Any], int, int]] = []
    validation_tokens = 0
    for item in by_split["validation"]:
        if validation_tokens >= validation_target:
            break
        selected_validation.append(item)
        validation_tokens += item[2]
    if validation_tokens < validation_target:
        raise ValueError("OpenCodeInstruct candidate did not reach validation target")

    selected = selected_train + selected_validation
    selected.sort(key=lambda item: (str(item[0]["split"]), str(item[0]["record_id"])))
    token_ledger = [
        {
            "record_id": item[0]["record_id"],
            "serialized_tokens": item[1],
            "split": item[0]["split"],
            "supervised_tokens": item[2],
        }
        for item in selected
    ]
    lengths = [item[1] for item in selected]
    supervised = [item[2] for item in selected]
    metrics_by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record, serialized_tokens, supervised_tokens in selected:
        bucket = metrics_by_split[str(record["split"])]
        bucket["records"] += 1
        bucket["serialized_tokens"] += serialized_tokens
        bucket["supervised_tokens"] += supervised_tokens
        bucket["masked_tokens"] += serialized_tokens - supervised_tokens
    metrics = {
        "accepted_records": len(selected),
        "assistant_only_loss_verified": True,
        "by_split": {key: dict(value) for key, value in sorted(metrics_by_split.items())},
        "candidate_4k_fit_records": len(measured),
        "context_window": template.context_window,
        "overlength_rejected": overlength,
        "serialized_length": {
            "maximum": max(lengths, default=0),
            "mean": mean(lengths) if lengths else 0,
            "median": median(lengths) if lengths else 0,
            "minimum": min(lengths, default=0),
            "p95": _nearest_rank(lengths, 0.95),
        },
        "serialized_tokens": sum(lengths),
        "supervised_tokens": sum(supervised),
        "token_ledger_sha256": sha256_value(token_ledger),
        "validation_target_supervised_tokens": validation_target,
    }
    return [item[0] for item in selected], metrics, token_ledger


def deterministic_owner_sample(
    records: Sequence[Mapping[str, Any]],
    *,
    maximum_records: int,
) -> list[dict[str, Any]]:
    buckets: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[str(record["split"])].append(record)
    for rows in buckets.values():
        rows.sort(
            key=lambda row: hashlib.sha256(str(row["record_id"]).encode()).hexdigest()
        )
    selected: list[dict[str, Any]] = []
    while len(selected) < maximum_records and any(buckets.values()):
        for split in ("train", "validation"):
            if buckets[split] and len(selected) < maximum_records:
                row = buckets[split].pop(0)
                selected.append(
                    {
                        "language_mode": "english",
                        "messages": row["messages"],
                        "record_id": row["record_id"],
                        "skill_id": "coding_algorithmic",
                        "split": row["split"],
                    }
                )
    return selected


def build_opencode_owner_review_html(
    sample_rows: list[Mapping[str, Any]],
) -> tuple[str, str]:
    sample_sha256 = sha256_value(sample_rows)
    html = build_owner_review_html(sample_rows, sample_sha256=sample_sha256)
    replacements = {
        "Lumi owner review": "Lumi OpenCodeInstruct owner review",
        "Lumi owner spot-check": "Lumi OpenCodeInstruct coding spot-check",
        (
            "These rows are AI-generated and owner-reviewed. Approving them does not "
            "make them human-written."
        ): (
            "These are upstream synthetic coding examples that passed published tests, "
            "a local fail-closed sandbox, and Lumi filters. Check whether each prompt is "
            "clear and each answer is correct, useful, safe, and well formatted. Approval "
            "does not authorize registry promotion, dataset sealing, or SFT."
        ),
        "lumi.ai-assisted-owner-decisions.v0.3": (
            "lumi.opencodeinstruct-owner-decisions.v0.1"
        ),
        "lumi-owner-review-decisions-v0.3.json": (
            "lumi-opencodeinstruct-owner-decisions-v0.1.json"
        ),
    }
    for old, new in replacements.items():
        if old not in html:
            raise ValueError(f"OpenCodeInstruct owner-review marker missing: {old[:48]}")
        html = html.replace(old, new)
    html = html.replace(
        "Check correctness, naturalness, language, context, and safety.",
        "Check prompt clarity, code correctness, usefulness, formatting, and safety.",
    )
    return html, sample_sha256


def load_runtime_components(
    plan: Mapping[str, Any],
    *,
    repository_root: Path,
) -> tuple[SFTPolicy, BenchmarkRegistry, Mapping[str, Any], ChatTemplate, ChatTokenizer]:
    from transformers import AutoTokenizer

    bindings = plan["bindings"]
    contract_path = resolve_binding(
        repository_root, bindings["data_contract"], name="data_contract"
    )
    benchmark_path = resolve_binding(
        repository_root, bindings["benchmark_registry"], name="benchmark_registry"
    )
    quality_path = resolve_binding(
        repository_root, bindings["quality_config"], name="quality_config"
    )
    template_path = resolve_binding(
        repository_root, bindings["chat_template"], name="chat_template"
    )
    tokenizer_path = resolve_binding(
        repository_root, bindings["tokenizer_json"], name="tokenizer_json"
    )
    tokenizer_config = resolve_binding(
        repository_root, bindings["tokenizer_config"], name="tokenizer_config"
    )
    if tokenizer_path.parent != tokenizer_config.parent:
        raise ValueError("frozen Lumi tokenizer files are not colocated")
    policy = load_sft_policy(contract_path)
    benchmark = BenchmarkRegistry(policy.benchmark_decontamination, repository_root)
    if benchmark.path.resolve() != benchmark_path:
        raise ValueError("OpenCodeInstruct benchmark registry resolved elsewhere")
    quality = _load_json(quality_path)
    validate_quality_config(quality)
    template = load_chat_template(template_path)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path.parent, local_files_only=True)
    return policy, benchmark, dict(quality["near_duplicate"]), template, tokenizer


def artifact_inventory(
    paths: Mapping[str, Path],
    *,
    raw_text_artifacts: Iterable[str],
) -> dict[str, Any]:
    raw = set(raw_text_artifacts)
    objects = [
        {
            "bytes": path.stat().st_size,
            "contains_raw_training_text": name in raw,
            "name": name,
            "path": path.name,
            "sha256": file_sha256(path),
        }
        for name, path in sorted(paths.items())
    ]
    return {"objects": objects, "total_bytes": sum(item["bytes"] for item in objects)}


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite OpenCodeInstruct JSON: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite OpenCodeInstruct JSONL: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite OpenCodeInstruct text: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_raw_text_free_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported OpenCodeInstruct curation report")
    if report.get("status") != "curated_nonproduction_owner_review_pending":
        raise ValueError("OpenCodeInstruct curation status changed")
    if report.get("raw_training_text_present") is not False:
        raise ValueError("OpenCodeInstruct report contains raw training text")
    production = report.get("production")
    if production != {
        "candidate_registry_promotions": 0,
        "dataset_sealed": False,
        "production_eligible": False,
        "production_sft_authorized": False,
    }:
        raise ValueError("OpenCodeInstruct production boundary changed")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("OpenCodeInstruct curation report seal mismatch")


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "SHARD_PATH",
    "VERIFICATION_SCHEMA_VERSION",
    "apply_conversation_near_dedup",
    "artifact_inventory",
    "build_opencode_owner_review_html",
    "deterministic_owner_sample",
    "file_sha256",
    "load_opencodeinstruct_plan",
    "load_runtime_components",
    "load_source_envelopes",
    "measure_and_select",
    "run_local_unit_test_verifier",
    "sha256_value",
    "validate_raw_text_free_report",
    "verify_raw_objects",
    "write_json_atomic",
    "write_jsonl_atomic",
    "write_text_atomic",
]
