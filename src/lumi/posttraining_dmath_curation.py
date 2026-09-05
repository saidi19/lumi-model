"""Fail-closed, CPU-only curation for the pinned DMath train source."""

from __future__ import annotations

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
from typing import Any

from .data_quality import BenchmarkRegistry, matching_tokens, validate_quality_config
from .posttraining_data import SFTPolicy, load_sft_policy
from .posttraining_format import ChatTemplate, ChatTokenizer, load_chat_template
from .posttraining_owner_review import build_owner_review_html

PLAN_SCHEMA_VERSION = "lumi.sft-dmath-curation-run.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-dmath-curation-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-dmath-curation-verification.v0.1"

SOURCE_ID = "dmath"
SOURCE_DATASET_ID = "JiwooKimAR/dmath"
SOURCE_REVISION = "149a7403619ac1bd6ed10f50694ad82699d5cc03"
SOURCE_LICENSE = "mit"
SOURCE_OBJECT_PATH = "dataset/train.json"
SOURCE_OBJECT_SHA256 = "8c18e591d4e81072107887c1297dcc5971757e548b7d79069e28f01ac85703d8"
SOURCE_ROWS = 7_943
EXPECTED_SOURCE_FIELDS = frozenset(
    {
        "answer_en",
        "answer_ko",
        "category",
        "question_en",
        "question_ko",
        "solution_abst_en",
        "solution_abst_ko",
        "solution_code_en",
        "solution_code_ko",
    }
)

EXPECTED_AUTHORIZATION = {
    "candidate_registry_promotion_authorized": False,
    "cpu_only_curation_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_spend_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "test_split_access_authorized": False,
    "unrelated_dataset_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_decoded_bytes": 67_108_864,
    "maximum_retained_records": SOURCE_ROWS,
    "maximum_runtime_minutes": 120,
    "maximum_source_rows": SOURCE_ROWS,
    "owner_sample_records": 32,
    "validation_fraction": 0.05,
}
EXPECTED_OUTPUT = {
    "private_root": ("../lumi-artifacts/4b-sft-dmath-acquisition-v0.1/aaba91c90ea8/attempt-001"),
    "raw_text_free_report_path": "evidence/lumi-4b-sft-dmath-curation-v0.1.json",
    "raw_text_free_verification_path": (
        "evidence/lumi-4b-sft-dmath-curation-verification-v0.1.json"
    ),
    "raw_training_text_in_git": False,
}
EXPECTED_POLICY = {
    "assistant_response_template": "The answer is {answer_en}.",
    "benchmark_decontamination_required": True,
    "cross_split_exact_and_prompt_near_dedup_required": True,
    "hidden_solution_fields_training_target_permitted": False,
    "near_duplicate": {
        "algorithm": "exact_token_shingle_jaccard_inverted_index_v0.1",
        "shingle_tokens": 5,
        "threshold": 0.9,
    },
    "source_registry_check": "deferred_until_separate_promotion",
    "target_fields": ["question_en", "answer_en"],
    "validation_assignment": "category_stratified_sha256_v0.1",
    "verifier_fields": ["solution_abst_en", "solution_code_en"],
}
EXPECTED_BINDING_PATHS = {
    "acquisition_plan": "configs/lumi-4b-sft-dmath-acquisition-v0.1.json",
    "acquisition_report": "evidence/lumi-4b-sft-dmath-acquisition-v0.1.json",
    "benchmark_registry": "data/benchmark-signatures-v0.1.jsonl.gz",
    "chat_template": "configs/lumi-4b-chat-template-v0.1.json",
    "curation_implementation": "src/lumi/posttraining_dmath_curation.py",
    "curation_runner": "scripts/run_dmath_curation.py",
    "curation_verifier": "scripts/verify_dmath_curation.py",
    "data_contract": "configs/lumi-4b-sft-data-contract-v0.1.json",
    "dedup_implementation": "src/lumi/data_quality.py",
    "encoding_implementation": "src/lumi/posttraining_format.py",
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
    "verifier_worker": "scripts/dmath_verifier_worker.py",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_CATEGORY = re.compile(r"[^a-z0-9]+")


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
        raise ValueError(f"DMath curation binding path is invalid: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"DMath curation binding escapes workspace: {name}")
    if not resolved.is_file():
        raise FileNotFoundError(f"DMath curation binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"DMath curation binding changed: {name}")
    return resolved


def load_dmath_curation_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load the exact local-only curation plan and verify every static binding."""

    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-06",
        "output": EXPECTED_OUTPUT,
        "policy": EXPECTED_POLICY,
        "schema_version": PLAN_SCHEMA_VERSION,
        "source": {
            "dataset_id": SOURCE_DATASET_ID,
            "declared_license_id": SOURCE_LICENSE,
            "object_path": SOURCE_OBJECT_PATH,
            "object_sha256": SOURCE_OBJECT_SHA256,
            "repository_url": "https://github.com/JiwooKimAR/dmath",
            "revision": SOURCE_REVISION,
            "source_id": SOURCE_ID,
            "source_rows": SOURCE_ROWS,
            "source_split": "train",
        },
        "status": "authorized_cpu_only_dmath_curation_no_promotion",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"DMath curation plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(EXPECTED_BINDING_PATHS):
        raise ValueError("DMath curation binding set changed")
    for name, expected_path in EXPECTED_BINDING_PATHS.items():
        binding = bindings[name]
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise ValueError(f"DMath curation binding path changed: {name}")
        resolve_binding(repository_root, binding, name=name)

    acquisition = _load_json(
        resolve_binding(repository_root, bindings["acquisition_report"], name="acquisition_report")
    )
    if acquisition.get("report_sha256") != (
        "73756b72b0b0734ab50bf177c6c04b58c7e67dd68d0629a198669da184d0d7a9"
    ):
        raise ValueError("DMath acquisition receipt changed")
    if acquisition.get("test_split_accessed") is not False:
        raise ValueError("DMath official test split may not be accessed")
    return plan


def load_source_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate the pinned object-by-ID DMath train JSON."""

    source_path = Path(path)
    if source_path.stat().st_size > EXPECTED_LIMITS["maximum_decoded_bytes"]:
        raise ValueError("DMath source exceeds decoded byte limit")
    if file_sha256(source_path) != SOURCE_OBJECT_SHA256:
        raise ValueError("DMath source object SHA-256 changed")
    value = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("DMath train root must remain object-by-ID")
    expected_ids = [str(index) for index in range(1, SOURCE_ROWS + 1)]
    if list(value) != expected_ids:
        raise ValueError("DMath train record IDs changed")
    rows: dict[str, dict[str, Any]] = {}
    for record_id, raw_row in value.items():
        if not isinstance(raw_row, dict) or frozenset(raw_row) != EXPECTED_SOURCE_FIELDS:
            raise ValueError(f"DMath row schema changed: {record_id}")
        for field in ("question_en", "answer_en", "solution_abst_en", "solution_code_en"):
            if not isinstance(raw_row[field], str) or not raw_row[field].strip():
                raise ValueError(f"DMath required field is empty: {record_id}:{field}")
        if not isinstance(raw_row["category"], str) or not raw_row["category"].strip():
            raise ValueError(f"DMath category is empty: {record_id}")
        rows[record_id] = raw_row
    return rows


def run_isolated_verifier(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    worker_path: str | Path,
    python_executable: str = sys.executable,
    timeout_seconds: int = 900,
) -> tuple[list[dict[str, Any]], set[str], dict[str, int]]:
    """Run source programs in the isolated verifier and retain hashes only."""

    inputs = []
    for record_id, row in rows.items():
        inputs.append(
            json.dumps(
                {
                    "answer_en": row["answer_en"],
                    "record_id": record_id,
                    "solution_code_en": row["solution_code_en"],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    completed = subprocess.run(
        [python_executable, "-I", "-S", str(Path(worker_path).resolve())],
        check=False,
        capture_output=True,
        input=("\n".join(inputs) + "\n").encode("utf-8"),
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "DMath verifier worker failed without exposing source text: "
            f"returncode={completed.returncode} stderr_sha256="
            f"{hashlib.sha256(completed.stderr).hexdigest()}"
        )
    results = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    if len(results) != len(rows):
        raise ValueError("DMath verifier result count changed")
    expected_ids = set(rows)
    observed_ids = {result.get("record_id") for result in results}
    if observed_ids != expected_ids:
        raise ValueError("DMath verifier record IDs changed")

    passed: set[str] = set()
    reason_counts: Counter[str] = Counter()
    ledger: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: int(str(item["record_id"]))):
        record_id = str(result["record_id"])
        row = rows[record_id]
        verified = result.get("verified") is True
        reason = result.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError("DMath verifier emitted an invalid reason")
        if verified and reason != "verified":
            raise ValueError("DMath verifier pass reason changed")
        if verified:
            passed.add(record_id)
        reason_counts[reason] += 1
        ledger.append(
            {
                "abstract_expression_sha256": hashlib.sha256(
                    row["solution_abst_en"].encode("utf-8")
                ).hexdigest(),
                "answer_sha256": hashlib.sha256(row["answer_en"].encode("utf-8")).hexdigest(),
                "python_solution_sha256": hashlib.sha256(
                    row["solution_code_en"].encode("utf-8")
                ).hexdigest(),
                "reason": reason,
                "record_id": f"dmath-train-{int(record_id):06d}",
                "verified": verified,
            }
        )
    return ledger, passed, dict(sorted(reason_counts.items()))


def _category_slug(category: str) -> str:
    return _SAFE_CATEGORY.sub("_", category.casefold()).strip("_")


def build_candidate_records(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    verified_source_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project verified final-answer supervision without solution-trace leakage."""

    records: list[dict[str, Any]] = []
    attributions: list[dict[str, Any]] = []
    for source_id in sorted(verified_source_ids, key=int):
        row = rows[source_id]
        question = row["question_en"].strip()
        answer = row["answer_en"].strip()
        response = f"The answer is {answer}."
        record_id = f"dmath-train-{int(source_id):06d}"
        source_content = {
            "answer_en": answer,
            "question_en": question,
        }
        messages = [
            {"content": question, "role": "user"},
            {"content": response, "role": "assistant"},
        ]
        content_sha256 = sha256_value(source_content)
        transform_sha256 = sha256_value(
            {
                "messages": messages,
                "transform": "dmath_final_answer_projection_v0.1",
            }
        )
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
                "review_method": "dmath_fail_closed_program_verifier_plus_lumi_filters_v0.1",
                "review_status": "accepted",
                "score": 0.95,
                "secret_scan": "passed",
            },
            "record_id": record_id,
            "safety": {
                "policy_version": "lumi-sft-basic-pattern-scan-v0.1",
                "risk_labels": [],
            },
            "schema_version": "lumi.sft.v0.1",
            "source": {
                "dataset_id": SOURCE_DATASET_ID,
                "document_id": source_id,
                "license_id": SOURCE_LICENSE,
                "revision": SOURCE_REVISION,
                "source_id": SOURCE_ID,
                "uri": (
                    f"https://github.com/JiwooKimAR/dmath/blob/{SOURCE_REVISION}/"
                    f"{SOURCE_OBJECT_PATH}#record-{source_id}"
                ),
            },
            "split": "train",
            "task_category": "reasoning_math_science",
        }
        attribution = {
            "abstract_expression_sha256": hashlib.sha256(
                row["solution_abst_en"].encode("utf-8")
            ).hexdigest(),
            "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            "content_sha256": content_sha256,
            "excluded_training_target_fields": ["solution_abst_en", "solution_code_en"],
            "python_solution_sha256": hashlib.sha256(
                row["solution_code_en"].encode("utf-8")
            ).hexdigest(),
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "record_id": record_id,
            "source_object_path": SOURCE_OBJECT_PATH,
            "source_object_sha256": SOURCE_OBJECT_SHA256,
            "source_record_id": source_id,
            "target_fields": ["question_en", "answer_en"],
            "transform_sha256": transform_sha256,
            "upstream_category": row["category"],
            "verifier_status": "passed",
        }
        records.append(record)
        attributions.append(attribution)
    return records, attributions


def apply_prompt_near_dedup(
    records: Sequence[Mapping[str, Any]],
    *,
    near_duplicate_config: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Retain one representative using exact token-shingle Jaccard comparisons."""

    threshold = float(near_duplicate_config["threshold"])
    shingle_tokens = int(near_duplicate_config["shingle_tokens"])
    accepted_signatures: dict[str, frozenset[str]] = {}
    inverted: defaultdict[str, set[str]] = defaultdict(set)
    accepted: list[Mapping[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: str(item["record_id"])):
        prompt = record["messages"][0]["content"]
        tokens = matching_tokens(prompt)
        if len(tokens) < shingle_tokens:
            shingles = frozenset({" ".join(tokens)})
        else:
            shingles = frozenset(
                " ".join(tokens[index : index + shingle_tokens])
                for index in range(len(tokens) - shingle_tokens + 1)
            )
        candidate_ids: set[str] = set()
        for shingle in shingles:
            candidate_ids.update(inverted[shingle])
        duplicate_of = None
        for candidate_id in sorted(candidate_ids):
            candidate = accepted_signatures[candidate_id]
            union_size = len(shingles | candidate)
            similarity = len(shingles & candidate) / union_size if union_size else 1.0
            if similarity >= threshold:
                duplicate_of = candidate_id
                break
        accepted_row = duplicate_of is None
        if accepted_row:
            record_id = str(record["record_id"])
            accepted_signatures[record_id] = shingles
            for shingle in shingles:
                inverted[shingle].add(record_id)
            accepted.append(record)
        ledger.append(
            {
                "accepted": accepted_row,
                "duplicate_of": duplicate_of,
                "record_id": record["record_id"],
                "signature_sha256": sha256_value(sorted(shingles)),
            }
        )
    metrics = {
        "accepted": len(accepted),
        "algorithm": "exact_token_shingle_jaccard_inverted_index_v0.1",
        "cross_split_scope": "global_before_split_assignment",
        "rejected": len(records) - len(accepted),
        "shingle_tokens": shingle_tokens,
        "threshold": threshold,
    }
    return accepted, ledger, metrics


def assign_validation_split(
    records: Sequence[Mapping[str, Any]],
    *,
    attribution_by_id: Mapping[str, Mapping[str, Any]],
    validation_fraction: float,
) -> list[dict[str, Any]]:
    """Assign an exact, category-stratified internal validation split."""

    if not 0 < validation_fraction < 0.5:
        raise ValueError("DMath validation fraction is outside the safe range")
    buckets: defaultdict[str, list[str]] = defaultdict(list)
    for record in records:
        record_id = str(record["record_id"])
        category = str(attribution_by_id[record_id]["upstream_category"])
        buckets[category].append(record_id)
    validation_ids: set[str] = set()
    for category, record_ids in sorted(buckets.items()):
        ordered = sorted(
            record_ids,
            key=lambda value: hashlib.sha256(
                f"dmath-validation-v0.1\0{category}\0{value}".encode()
            ).hexdigest(),
        )
        count = max(1, math.floor(len(ordered) * validation_fraction + 0.5))
        validation_ids.update(ordered[:count])
    assigned = []
    for record in records:
        updated = dict(record)
        updated["split"] = "validation" if str(record["record_id"]) in validation_ids else "train"
        assigned.append(updated)
    return assigned


def deterministic_owner_sample(
    records: Sequence[Mapping[str, Any]],
    *,
    attribution_by_id: Mapping[str, Mapping[str, Any]],
    maximum_records: int,
) -> list[dict[str, Any]]:
    """Build a split/category-interleaved private review payload."""

    buckets: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        record_id = str(record["record_id"])
        category = str(attribution_by_id[record_id]["upstream_category"])
        buckets[(str(record["split"]), category)].append(record)
    for key, rows in buckets.items():
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"dmath-owner-review-v0.1\0{key[0]}\0{key[1]}\0{row['record_id']}".encode()
            ).hexdigest()
        )
    selected: list[Mapping[str, Any]] = []
    offset = 0
    keys = sorted(buckets)
    target = min(maximum_records, len(records))
    while len(selected) < target:
        added = False
        for key in keys:
            if offset < len(buckets[key]):
                selected.append(buckets[key][offset])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        offset += 1
    payload = []
    for record in selected:
        record_id = str(record["record_id"])
        category = str(attribution_by_id[record_id]["upstream_category"])
        payload.append(
            {
                **dict(record),
                "language_mode": "en",
                "skill_id": f"dmath_{_category_slug(category)}",
                "source_category": category,
            }
        )
    return payload


def build_dmath_owner_review_html(
    sample_rows: list[Mapping[str, Any]],
) -> tuple[str, str]:
    """Build a self-contained review page with DMath-specific honesty text."""

    sample_sha256 = sha256_value(sample_rows)
    html = build_owner_review_html(sample_rows, sample_sha256=sample_sha256)
    replacements = {
        "<title>Lumi owner review</title>": "<title>Lumi DMath owner review</title>",
        "<h1>Lumi owner spot-check</h1>": "<h1>Lumi DMath quality spot-check</h1>",
        (
            f"<p>Review all {len(sample_rows)} rows. Check correctness, naturalness, "
            "language, context, and safety.</p>"
        ): (
            f"<p>Review all {len(sample_rows)} rows. Accept only when the question is "
            "clear and the concise final answer is correct, natural, safe, and suitable "
            "for Lumi math instruction tuning.</p>"
        ),
        (
            "These rows are AI-generated and owner-reviewed. Approving them does not "
            "make them human-written."
        ): (
            "These are upstream DMath examples selected by deterministic machine gates. "
            "Only question_en and answer_en are proposed training targets; hidden abstract "
            "and Python solutions are verifier evidence only. Owner approval does not "
            "authorize source-registry promotion, dataset sealing, or SFT."
        ),
        "lumi.ai-assisted-owner-decisions.v0.3": "lumi.dmath-owner-decisions.v0.1",
        "lumi-owner-review-decisions-v0.3.json": "lumi-dmath-owner-decisions-v0.1.json",
    }
    for old, new in replacements.items():
        if old not in html:
            raise ValueError(f"DMath owner-review HTML template marker missing: {old[:40]}")
        html = html.replace(old, new)
    return html, sample_sha256


def load_runtime_components(
    plan: Mapping[str, Any],
    *,
    repository_root: Path,
) -> tuple[SFTPolicy, BenchmarkRegistry, Mapping[str, Any], ChatTemplate, ChatTokenizer]:
    """Load exact offline filters, dedup policy, chat template, and tokenizer."""

    from transformers import AutoTokenizer

    bindings = plan["bindings"]
    contract_path = resolve_binding(
        repository_root, bindings["data_contract"], name="data_contract"
    )
    benchmark_path = resolve_binding(
        repository_root,
        bindings["benchmark_registry"],
        name="benchmark_registry",
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
        repository_root,
        bindings["tokenizer_config"],
        name="tokenizer_config",
    )
    if tokenizer_path.parent != tokenizer_config.parent:
        raise ValueError("frozen Lumi tokenizer files are not colocated")
    policy = load_sft_policy(contract_path)
    benchmark = BenchmarkRegistry(policy.benchmark_decontamination, repository_root)
    if benchmark.path.resolve() != benchmark_path:
        raise ValueError("DMath benchmark registry resolved to another path")
    quality = _load_json(quality_path)
    validate_quality_config(quality)
    near_duplicate = dict(quality["near_duplicate"])
    near_duplicate["threshold"] = 0.9
    template = load_chat_template(template_path)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path.parent, local_files_only=True)
    return policy, benchmark, near_duplicate, template, tokenizer


def artifact_inventory(
    paths: Mapping[str, Path],
    *,
    raw_text_artifacts: Iterable[str],
) -> dict[str, Any]:
    raw_names = set(raw_text_artifacts)
    objects = []
    for name, path in sorted(paths.items()):
        objects.append(
            {
                "bytes": path.stat().st_size,
                "contains_raw_training_text": name in raw_names,
                "name": name,
                "path": path.name,
                "sha256": file_sha256(path),
            }
        )
    return {
        "objects": objects,
        "total_bytes": sum(item["bytes"] for item in objects),
    }


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite DMath JSON artifact: {path}")
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
        raise FileExistsError(f"refusing to overwrite DMath JSONL artifact: {path}")
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
        raise FileExistsError(f"refusing to overwrite DMath text artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_raw_text_free_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported DMath curation report")
    if report.get("status") != "curated_nonproduction_owner_review_pending":
        raise ValueError("DMath curation status changed")
    if report.get("raw_training_text_present") is not False:
        raise ValueError("DMath report contains raw training text")
    production = report.get("production")
    if not isinstance(production, Mapping) or production != {
        "candidate_registry_promotions": 0,
        "dataset_sealed": False,
        "production_eligible": False,
        "production_sft_authorized": False,
    }:
        raise ValueError("DMath production boundary changed")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("DMath curation report seal mismatch")


def count_by_split(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record["split"]) for record in records).items()))
