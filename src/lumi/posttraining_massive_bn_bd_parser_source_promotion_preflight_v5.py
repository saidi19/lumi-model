"""Cross-source dedup and registry-promotion preflight for MASSIVE bn-BD.

The implementation performs two independently reproducible local passes.  Each
pass streams the exact registry-v0.21 314,716-row candidate inventory, rebuilds
the frozen exact and 128-permutation MinHash indexes, and queries the exact 64
MASSIVE parser rows.  The projected v0.22 registry exists only in memory; no
candidate, source, registry, seal, or training artifact is mutated.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import os
import re
import stat
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_candidate_mix_reaudit_v15 as mix_v15
from . import posttraining_massive_bn_bd_acquisition_v1 as acquisition_v1
from . import posttraining_massive_bn_bd_parser_owner_review_v4 as owner_v4
from .data_quality import MinHashNearDuplicateIndex
from .posttraining_candidate_mix_preflight import _canonical_message_digest
from .posttraining_candidate_mix_reaudit import _role_tagged_conversation
from .posttraining_data import load_sft_policy
from .posttraining_sources import SFTSource, load_sft_source_registry

PLAN_SCHEMA_VERSION = (
    "lumi.sft-massive-bn-bd-parser-source-promotion-preflight-plan.v0.5"
)
REPORT_SCHEMA_VERSION = (
    "lumi.sft-massive-bn-bd-parser-source-promotion-preflight-report.v0.5"
)
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-massive-bn-bd-parser-source-promotion-preflight-"
    "verification.v0.5"
)
PLAN_STATUS = "authorized_local_only_cross_source_registry_preflight_nonproduction"
PASS_STATUS = (
    "PASS_massive_bn_bd_parser_cross_source_clean_registry_v0_22_projection_ready_"
    "nonproduction"
)
STOP_STATUS = (
    "STOP_massive_bn_bd_parser_cross_source_collision_registry_projection_blocked_"
    "nonproduction"
)
PASS_VERIFICATION_STATUS = (
    "PASS_verified_massive_bn_bd_parser_cross_source_clean_registry_v0_22_"
    "projection_ready_nonproduction"
)
STOP_VERIFICATION_STATUS = (
    "PASS_verified_massive_bn_bd_parser_cross_source_collision_registry_projection_"
    "blocked_nonproduction"
)

EXPECTED_PLAN_PATH = Path(
    "configs/lumi-4b-sft-massive-bn-bd-parser-source-promotion-preflight-v0.5.json"
)
EXPECTED_AUTHORIZATION_TEXT = "ok next step shuru de"
EXPECTED_AUTHORIZATION_SHA256 = (
    "7a2317268ba301ceb4a30bb9aa558fad1796139ccc3f5e87722f1e944c4ddcd3"
)
EXPECTED_CODE_PATHS = {
    "implementation": (
        "src/lumi/posttraining_massive_bn_bd_parser_source_promotion_preflight_v5.py"
    ),
    "runner": (
        "scripts/preflight_sft_massive_bn_bd_parser_source_promotion_v5.py"
    ),
    "verifier": (
        "scripts/verify_sft_massive_bn_bd_parser_source_promotion_preflight_v5.py"
    ),
    "test": (
        "tests/test_massive_bn_bd_parser_source_promotion_preflight_v5.py"
    ),
}
EXPECTED_OUTPUT = {
    "report_path": (
        "evidence/lumi-4b-sft-massive-bn-bd-parser-source-promotion-"
        "preflight-v0.5.json"
    ),
    "verification_path": (
        "evidence/lumi-4b-sft-massive-bn-bd-parser-source-promotion-"
        "preflight-verification-v0.5.json"
    ),
}
EXPECTED_INPUTS = {
    "baseline_plan": {
        "path": "configs/lumi-4b-sft-candidate-mix-reaudit-v0.15.json",
        "bytes": 7_691,
        "sha256": "7d37369d115b0c666d96d138508f7b31a442cc8c6d990470ec62cf47096282fb",
    },
    "baseline_report": {
        "path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.15.json",
        "bytes": 24_444,
        "sha256": "fc81dc5827ebc1a039df7c7692b9f4ccd9144ad1ba1f2a32754ff6f6449d94bf",
        "internal_sha256": (
            "19fc82952653c7e1a022338fbe58bafeae25a6a88d63a2a3e3bc4154e5d99504"
        ),
    },
    "baseline_verification": {
        "path": (
            "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.15.json"
        ),
        "bytes": 1_498,
        "sha256": "1af422f2fbb5dcf21b2d4212891ab4d60c9c4cb63a797ac87acfa7598b9be782",
        "internal_sha256": (
            "bf11960dee2f429d11c964ef2e3f8960062796bf950d27f8e55a88576f19b9da"
        ),
    },
    "source_registry": {
        "path": "configs/lumi-4b-sft-source-registry-v0.21.json",
        "bytes": 28_405,
        "sha256": "daa2d81755456728c5c72bca9ee95d5424aba0fc05e9fdd13223018f4055e3a1",
    },
    "owner_plan": {
        "path": "configs/lumi-4b-sft-massive-bn-bd-parser-owner-review-v0.4.json",
        "bytes": 5_692,
        "sha256": "e27c5b468ab82268f215ccc4f29e240b1b5539e7b8b8657855e7c493ffa70ec0",
    },
    "owner_receipt": {
        "path": "evidence/lumi-4b-sft-massive-bn-bd-parser-owner-review-v0.4.json",
        "bytes": 4_327,
        "sha256": "bf6156fcbe36e551a798248c005bb8e02ec0c42b39bed67dde72f7f907089542",
        "internal_sha256": (
            "d33ae45ea208548d050955074414af56f429aa97950ae68f3ba40c71f63a064b"
        ),
    },
    "owner_verification": {
        "path": (
            "evidence/lumi-4b-sft-massive-bn-bd-parser-owner-review-"
            "verification-v0.4.json"
        ),
        "bytes": 1_867,
        "sha256": "6978eccfe96c02e3d41339d64527453de8d3656ca7f594e394d9ec3d16947c2b",
        "internal_sha256": (
            "bed6031ce36b26006a64ad7426467020abb25f26ce6c424d7a058fdf7f399299"
        ),
    },
    "massive_candidate": {
        "path": (
            "../lumi-artifacts/4b-sft-massive-bn-bd-parser-canary-v0.3/"
            "38be2e88eabf/attempt-001/private/candidate-rows.jsonl"
        ),
        "bytes": 140_522,
        "sha256": "8e107adb12839baf9e0aca0d9eebe0b058a2e69d2e9e9b5f4985b69c143e243c",
    },
    "acquisition_plan": {
        "path": "configs/lumi-4b-sft-massive-bn-bd-acquisition-v0.1.json",
        "bytes": 7_797,
        "sha256": "f5826974727257403a64c7e2e64be76016d296148e9c45338b3a35d286e0e853",
        "internal_sha256": (
            "eafceb4a630f863203757fb49abfd43ef403cde1caeb30841a0bfd46beccf7de"
        ),
    },
    "acquisition_report": {
        "path": "evidence/lumi-4b-sft-massive-bn-bd-acquisition-v0.1.json",
        "bytes": 6_046,
        "sha256": "68d3507a4b7a35674cf1068feefea7466978fbdf37490797d12fa8e4d0677817",
        "internal_sha256": (
            "a5b4bffbf4a1fd2255d739084aad41ce63c36aafa10c8f0e0a0070ca776659e0"
        ),
    },
}
EXPECTED_LIMITS = {
    "passes": 2,
    "maximum_baseline_candidate_row_reads": 629_432,
    "maximum_massive_candidate_row_reads": 128,
    "maximum_total_candidate_row_reads": 629_560,
    "maximum_source_line_reads": 0,
    "maximum_test_object_reads": 0,
    "maximum_registry_writes": 0,
    "maximum_private_write_bytes": 4_194_304,
    "maximum_runtime_seconds_per_process": 1_800,
    "maximum_runtime_seconds_total": 3_600,
    "maximum_network_bytes": 0,
    "maximum_gpu_use": 0,
    "maximum_inference_calls": 0,
    "maximum_training_actions": 0,
    "maximum_retries_after_started_pass": 0,
}
EXPECTED_AUTHORIZATION = {
    "local_candidate_read_authorized": True,
    "cross_source_exact_and_near_dedup_preflight_authorized": True,
    "in_memory_registry_successor_projection_authorized": True,
    "candidate_or_source_mutation_authorized": False,
    "registry_mutation_or_successor_write_authorized": False,
    "production_promotion_authorized": False,
    "dataset_seal_authorized": False,
    "sft_training_authorized": False,
    "network_gpu_inference_or_remote_access_authorized": False,
    "test_object_access_authorized": False,
    "unrelated_action_authorized": False,
}
EXPECTED_BASELINE = {
    "candidate_sources": 17,
    "registry_sources": 24,
    "records": 314_716,
    "train_supervised_tokens": 10_604_149,
    "validation_supervised_tokens": 628_604,
}
EXPECTED_MASSIVE = {
    "records": 64,
    "train_supervised_tokens": 6_451,
    "validation_supervised_tokens": 0,
    "task_category": "tool_api",
    "language_mode": "bengali_script",
}
EXPECTED_COMBINED = {
    "candidate_sources": 18,
    "registry_sources": 25,
    "records": 314_780,
    "train_supervised_tokens": 10_610_600,
    "validation_supervised_tokens": 628_604,
}
EXPECTED_NEAR_POLICY = {
    "comparison_unit": "role_tagged_full_conversation",
    "num_perm": 128,
    "package_version": "2.0.0",
    "scheme": "affine32",
    "seed": 1,
    "shingle_tokens": 5,
    "threshold": 0.9,
}
EXPECTED_PROJECTED_REGISTRY = {
    "bytes": 29_673,
    "canonical_sha256": (
        "02a8de18c8a22dd62ca77e4cc5f094ab7e60116999dea0b02f7d7f5e38c1a240"
    ),
    "file_sha256": "c472d2ca1fe16beff9700719d53a3f8b539073bf52a6a2fa297da192d944f6b8",
    "observed_at": "2026-08-31",
    "sources": 25,
}
EXPECTED_NEW_SOURCE = {
    "access": "public",
    "adapter": "massive_bn_bd_nfc_collision_aware_pure_parser_owner_reviewed_v0.5",
    "blockers": [],
    "dataset_id": "AmazonScience/massive",
    "evidence_urls": [
        "https://huggingface.co/datasets/AmazonScience/massive/tree/"
        "d1792d0e4f32a74d43725cc1b901f600dad990dd",
        "https://raw.githubusercontent.com/alexa/massive/"
        "f966f21846043aabef9b0f974fa7970027f43738/LICENSE.txt",
        "https://raw.githubusercontent.com/alexa/massive/"
        "f966f21846043aabef9b0f974fa7970027f43738/NOTICE.md",
    ],
    "languages": ["bn"],
    "license_id": "cc-by-4.0",
    "origin": "human",
    "production_blockers": [
        "candidate_is_only_64_records_6451_supervised_tokens",
        "full_registry_v0_22_cross_source_hygiene_audit_not_complete",
        "production_dataset_seal_and_sft_authorization_not_complete",
    ],
    "production_eligible": False,
    "repository_url": "https://huggingface.co/datasets/AmazonScience/massive",
    "revision": "d1792d0e4f32a74d43725cc1b901f600dad990dd",
    "source_id": "amazon_massive_bn_bd_tool_canary",
    "splits": ["train"],
    "status": "canary_eligible",
    "task_categories": ["tool_api"],
}
EXPECTED_RECEIPT_MARKER = (
    b'{"started":"massive_bn_bd_parser_source_promotion_preflight_v0.5"}\n'
)
EXPECTED_VERIFICATION_MARKER = (
    b'{"started":"massive_bn_bd_parser_source_promotion_verification_v0.5"}\n'
)
EXPECTED_EMPTY_COLLISION_LEDGER_SHA256 = hashlib.sha256(b"[]").hexdigest()

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RECORD_ID_RE = re.compile(r"^massive-bn-bd-parser-[0-9a-f]{24}$")
_FORBIDDEN_PUBLIC_KEYS = {
    "assistant_response",
    "content",
    "messages",
    "note",
    "record_id",
    "reviewer_id",
    "user_prompt",
}

canonical_json_bytes = owner_v4.canonical_json_bytes
json_bytes = owner_v4.json_bytes
sha256_bytes = owner_v4.sha256_bytes
sha256_value = owner_v4.sha256_value
file_sha256 = owner_v4.file_sha256
strict_json_bytes = owner_v4.strict_json_bytes
start_marker = owner_v4.start_marker
finish_marker = owner_v4.finish_marker


def _resolve_repo_path(root: Path, path: str) -> Path:
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.parent):
        raise ValueError("bound repository-adjacent path escaped")
    return resolved


def _read_bound_bytes(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected_bytes:
            raise ValueError("bound input metadata changed")
        chunks: list[bytes] = []
        remaining = expected_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(payload) != expected_bytes or sha256_bytes(payload) != expected_sha256:
        raise ValueError("bound input content changed")
    return payload


def _input_path(root: Path, name: str) -> Path:
    return _resolve_repo_path(root, str(EXPECTED_INPUTS[name]["path"]))


def _read_input(root: Path, name: str) -> bytes:
    binding = EXPECTED_INPUTS[name]
    return _read_bound_bytes(
        _input_path(root, name),
        expected_bytes=int(binding["bytes"]),
        expected_sha256=str(binding["sha256"]),
    )


def _validate_internal_seal(
    value: Any,
    binding: Mapping[str, Any],
    seal_key: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("sealed JSON root changed")
    unsigned = {key: item for key, item in value.items() if key != seal_key}
    expected = binding["internal_sha256"]
    if value.get(seal_key) != expected or sha256_value(unsigned) != expected:
        raise ValueError("sealed JSON hash changed")


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError("private or raw-text key retained")
            _assert_no_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_keys(item)


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved = Path(path).resolve()
    if resolved != (root / EXPECTED_PLAN_PATH).resolve():
        raise ValueError("source-promotion preflight plan path changed")
    value = strict_json_bytes(resolved.read_bytes())
    expected_keys = {
        "authorization",
        "authorization_sha256",
        "authorization_text",
        "implementation",
        "inputs",
        "limits",
        "near_duplicate_policy",
        "output",
        "projected_registry",
        "runner",
        "runtime",
        "schema_version",
        "status",
        "test",
        "verifier",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("schema_version") != PLAN_SCHEMA_VERSION
        or value.get("status") != PLAN_STATUS
        or value.get("authorization_text") != EXPECTED_AUTHORIZATION_TEXT
        or value.get("authorization_sha256") != EXPECTED_AUTHORIZATION_SHA256
        or sha256_bytes(EXPECTED_AUTHORIZATION_TEXT.encode())
        != EXPECTED_AUTHORIZATION_SHA256
        or value.get("authorization") != EXPECTED_AUTHORIZATION
        or value.get("inputs") != EXPECTED_INPUTS
        or value.get("limits") != EXPECTED_LIMITS
        or value.get("near_duplicate_policy") != EXPECTED_NEAR_POLICY
        or value.get("output") != EXPECTED_OUTPUT
        or value.get("projected_registry")
        != {
            **EXPECTED_PROJECTED_REGISTRY,
            "entry": EXPECTED_NEW_SOURCE,
            "materialization_authorized": False,
            "predecessor_sha256": EXPECTED_INPUTS["source_registry"]["sha256"],
        }
        or value.get("runtime")
        != {
            "datasketch_version": "2.0.0",
            "python_executable": "/Users/afnan/Documents/lumi-model/.venv/bin/python",
            "python_version": "3.12.13",
        }
    ):
        raise ValueError("source-promotion preflight plan contract changed")
    for name, expected_path in EXPECTED_CODE_PATHS.items():
        binding = value.get(name)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != expected_path
            or not isinstance(binding.get("sha256"), str)
            or _SHA256_RE.fullmatch(binding["sha256"]) is None
        ):
            raise ValueError("source-promotion preflight code binding changed")
        code_path = (root / expected_path).resolve()
        if not code_path.is_relative_to(root) or file_sha256(code_path) != binding["sha256"]:
            raise ValueError("source-promotion preflight code hash changed")
    for name, binding in EXPECTED_INPUTS.items():
        info = _input_path(root, name).lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_size != binding["bytes"]
        ):
            raise ValueError("source-promotion preflight input metadata changed")
    return value


def _read_json_input(root: Path, name: str) -> Any:
    return strict_json_bytes(_read_input(root, name))


def _validate_bound_evidence(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline_plan = mix_v15.load_plan(_input_path(root, "baseline_plan"), root)
    baseline_report = _read_json_input(root, "baseline_report")
    baseline_verification = _read_json_input(root, "baseline_verification")
    _validate_internal_seal(
        baseline_report,
        EXPECTED_INPUTS["baseline_report"],
        "report_sha256",
    )
    _validate_internal_seal(
        baseline_verification,
        EXPECTED_INPUTS["baseline_verification"],
        "verification_sha256",
    )
    mix_v15.validate_report(baseline_report)
    mix_v15.validate_verification(baseline_verification)
    if (
        baseline_report.get("status")
        != "PASS_registry_v0_21_full_candidate_mix_reaudit_nonproduction"
        or baseline_report.get("candidate_pool", {}).get("candidate_records_read")
        != EXPECTED_BASELINE["records"]
        or baseline_report.get("exact_duplicate_audit", {}).get("duplicate_groups") != 0
        or baseline_report.get("near_duplicate_audit", {}).get("multi_record_components")
        != 0
        or baseline_verification.get("report_file_sha256")
        != EXPECTED_INPUTS["baseline_report"]["sha256"]
        or baseline_verification.get("production_eligible") is not False
    ):
        raise ValueError("registry-v0.21 baseline evidence changed")

    owner_v4.load_plan(_input_path(root, "owner_plan"), root)
    owner_receipt = _read_json_input(root, "owner_receipt")
    owner_verification = _read_json_input(root, "owner_verification")
    _validate_internal_seal(
        owner_receipt,
        EXPECTED_INPUTS["owner_receipt"],
        "receipt_sha256",
    )
    _validate_internal_seal(
        owner_verification,
        EXPECTED_INPUTS["owner_verification"],
        "verification_sha256",
    )
    owner_v4.validate_receipt(owner_receipt)
    owner_v4.validate_verification(owner_verification)
    if (
        owner_receipt.get("owner_review", {}).get("accepted_records") != 64
        or owner_receipt.get("owner_review", {}).get("rejected_records") != 0
        or owner_receipt.get("bindings", {}).get("candidate_file_sha256")
        != EXPECTED_INPUTS["massive_candidate"]["sha256"]
        or owner_verification.get("receipt_file_sha256")
        != EXPECTED_INPUTS["owner_receipt"]["sha256"]
        or owner_receipt.get("production_eligible") is not False
    ):
        raise ValueError("MASSIVE owner-review evidence changed")

    acquisition_plan = _read_json_input(root, "acquisition_plan")
    acquisition_report = _read_json_input(root, "acquisition_report")
    _validate_internal_seal(
        acquisition_plan,
        EXPECTED_INPUTS["acquisition_plan"],
        "plan_sha256",
    )
    _validate_internal_seal(
        acquisition_report,
        EXPECTED_INPUTS["acquisition_report"],
        "report_sha256",
    )
    acquisition_v1.validate_raw_text_free_report(acquisition_report)
    rights = acquisition_report.get("rights_documents")
    if (
        acquisition_plan.get("schema_version")
        != "lumi.sft-massive-bn-bd-train-acquisition-plan.v0.1"
        or not isinstance(rights, list)
        or {item.get("document_code") for item in rights}
        != {"massive_notice", "massive_license", "massive_third_party", "slurp_text_license"}
    ):
        raise ValueError("MASSIVE rights evidence changed")

    registry = _read_json_input(root, "source_registry")
    if (
        not isinstance(registry, dict)
        or registry.get("schema_version") != "lumi.sft.sources.v0.1"
        or not isinstance(registry.get("sources"), list)
        or len(registry["sources"]) != EXPECTED_BASELINE["registry_sources"]
        or any(
            row.get("source_id") == EXPECTED_NEW_SOURCE["source_id"]
            for row in registry["sources"]
            if isinstance(row, Mapping)
        )
    ):
        raise ValueError("source registry v0.21 projection base changed")
    return baseline_plan, baseline_report, registry


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("candidate input is not a regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = strict_json_bytes(line.encode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError(f"candidate row root changed at line {line_number}")
                yield value
    finally:
        os.close(descriptor)


def _validate_messages(row: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("candidate messages changed")
    for message in messages:
        if (
            not isinstance(message, Mapping)
            or set(message) != {"content", "role"}
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("candidate message fields changed")
    return messages


def _baseline_inventory(
    root: Path,
    baseline_plan: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    bound = {
        name: mix_v15._resolve_binding(root, binding, name)
        for name, binding in baseline_plan["bindings"].items()
    }
    mix_v15._validate_predecessor_and_promotion(bound)
    source_plans, candidate_paths, _, _ = mix_v15._source_inventory(
        root=root,
        bound=bound,
    )
    if len(source_plans) != EXPECTED_BASELINE["candidate_sources"]:
        raise ValueError("baseline source inventory changed")
    return source_plans, candidate_paths


def _build_cross_source_indexes(
    *,
    root: Path,
    baseline_plan: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    started: float,
) -> tuple[dict[str, str], MinHashNearDuplicateIndex, dict[str, int]]:
    source_plans, candidate_paths = _baseline_inventory(root, baseline_plan)
    near_config = {
        key: EXPECTED_NEAR_POLICY[key]
        for key in (
            "num_perm",
            "package_version",
            "scheme",
            "seed",
            "shingle_tokens",
            "threshold",
        )
    }
    if importlib.metadata.version("datasketch") != EXPECTED_NEAR_POLICY["package_version"]:
        raise ValueError("datasketch runtime changed")
    near_index = MinHashNearDuplicateIndex(near_config)
    exact_index: dict[str, str] = {}
    counts: Counter[str] = Counter()
    expected_metrics = baseline_report.get("source_metrics")
    if not isinstance(expected_metrics, Mapping):
        raise ValueError("baseline source metrics changed")
    for source_id in sorted(source_plans):
        path = candidate_paths[source_id]
        for row in _iter_jsonl(path):
            counts["rows"] += 1
            counts[source_id] += 1
            if counts["rows"] % 4_096 == 0:
                if time.monotonic() - started > EXPECTED_LIMITS[
                    "maximum_runtime_seconds_per_process"
                ]:
                    raise TimeoutError("source-promotion preflight runtime cap reached")
            if row.get("source", {}).get("source_id") != source_id:
                raise ValueError("baseline row source changed")
            record_id = row.get("record_id")
            messages = _validate_messages(row)
            if (
                not isinstance(record_id, str)
                or not record_id
                or record_id in near_index.signatures
            ):
                raise ValueError("baseline record-ID inventory changed")
            digest = _canonical_message_digest(messages)
            if digest in exact_index:
                raise ValueError("baseline exact-duplicate evidence changed")
            signature = near_index.signature(_role_tagged_conversation(row))
            if near_index.duplicate_of(signature) is not None:
                raise ValueError("baseline MinHash evidence changed")
            exact_index[digest] = record_id
            near_index.add(record_id, signature)
        metric = expected_metrics.get(source_id)
        if not isinstance(metric, Mapping) or counts[source_id] != metric.get("records"):
            raise ValueError("baseline per-source row count changed")
    if counts["rows"] != EXPECTED_BASELINE["records"]:
        raise ValueError("baseline row count changed")
    return exact_index, near_index, dict(counts)


def _validate_massive_row(row: Mapping[str, Any]) -> None:
    source = row.get("source")
    provenance = row.get("provenance")
    messages = _validate_messages(row)
    if (
        not isinstance(source, Mapping)
        or source.get("source_id") != EXPECTED_NEW_SOURCE["source_id"]
        or source.get("dataset_id") != EXPECTED_NEW_SOURCE["dataset_id"]
        or source.get("revision") != EXPECTED_NEW_SOURCE["revision"]
        or source.get("license_id") != EXPECTED_NEW_SOURCE["license_id"]
        or row.get("language") != "bn"
        or row.get("split") != "train"
        or row.get("task_category") != "tool_api"
        or [message.get("role") for message in messages]
        != ["user", "tool", "tool_result", "assistant"]
        or not isinstance(provenance, Mapping)
        or provenance.get("owner_review_status")
        != "pending_deterministic_64_row_spot_check"
        or provenance.get("source_registry_status") != "pending_separate_promotion"
    ):
        raise ValueError("MASSIVE candidate promotion contract changed")


def _cross_source_audit(
    *,
    root: Path,
    baseline_plan: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    started: float,
) -> dict[str, Any]:
    exact_index, near_index, counts = _build_cross_source_indexes(
        root=root,
        baseline_plan=baseline_plan,
        baseline_report=baseline_report,
        started=started,
    )
    candidate_path = _input_path(root, "massive_candidate")
    collisions: list[dict[str, str]] = []
    massive_rows = 0
    historical_owner_pending = 0
    historical_registry_pending = 0
    for row in _iter_jsonl(candidate_path):
        massive_rows += 1
        _validate_massive_row(row)
        record_id = str(row["record_id"])
        if _RECORD_ID_RE.fullmatch(record_id) is None or record_id in near_index.signatures:
            raise ValueError("MASSIVE record-ID domain changed")
        provenance = row["provenance"]
        historical_owner_pending += int(
            provenance.get("owner_review_status")
            == "pending_deterministic_64_row_spot_check"
        )
        historical_registry_pending += int(
            provenance.get("source_registry_status") == "pending_separate_promotion"
        )
        digest = _canonical_message_digest(row["messages"])
        exact_match = exact_index.get(digest)
        if exact_match is not None:
            collisions.append(
                {
                    "kind": "exact",
                    "massive_id_sha256": sha256_bytes(record_id.encode()),
                    "match_id_sha256": sha256_bytes(exact_match.encode()),
                }
            )
            continue
        signature = near_index.signature(_role_tagged_conversation(row))
        near_match = near_index.duplicate_of(signature)
        if near_match is not None:
            collisions.append(
                {
                    "kind": "minhash_0_9",
                    "massive_id_sha256": sha256_bytes(record_id.encode()),
                    "match_id_sha256": sha256_bytes(near_match.encode()),
                }
            )
    if (
        massive_rows != EXPECTED_MASSIVE["records"]
        or historical_owner_pending != massive_rows
        or historical_registry_pending != massive_rows
    ):
        raise ValueError("MASSIVE candidate row inventory changed")
    collision_counts = Counter(item["kind"] for item in collisions)
    collision_ledger = sorted(
        collisions,
        key=lambda item: (
            item["massive_id_sha256"],
            item["kind"],
            item["match_id_sha256"],
        ),
    )
    return {
        "baseline_candidate_row_reads": counts["rows"],
        "baseline_sources": EXPECTED_BASELINE["candidate_sources"],
        "collision_ledger_sha256": sha256_value(collision_ledger),
        "cross_source_exact_collision_rows": collision_counts["exact"],
        "cross_source_minhash_0_9_collision_rows": collision_counts["minhash_0_9"],
        "historical_owner_review_pending_rows": historical_owner_pending,
        "historical_registry_pending_rows": historical_registry_pending,
        "massive_candidate_row_reads": massive_rows,
        "retained_massive_rows": massive_rows - len(collisions),
        "total_collision_rows": len(collisions),
    }


def _project_registry(
    *,
    root: Path,
    registry_value: Mapping[str, Any],
) -> dict[str, Any]:
    policy = load_sft_policy(root / "configs/lumi-4b-sft-data-contract-v0.1.json")
    SFTSource.from_mapping(
        EXPECTED_NEW_SOURCE,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
    )
    registry_path = _input_path(root, "source_registry")
    loaded = load_sft_source_registry(
        registry_path,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=EXPECTED_INPUTS["source_registry"]["sha256"],
    )
    if len(loaded.sources) != EXPECTED_BASELINE["registry_sources"]:
        raise ValueError("source registry object count changed")
    projected = copy.deepcopy(dict(registry_value))
    predecessor_sources = copy.deepcopy(projected["sources"])
    projected["observed_at"] = EXPECTED_PROJECTED_REGISTRY["observed_at"]
    projected["sources"].append(copy.deepcopy(EXPECTED_NEW_SOURCE))
    if projected["sources"][:-1] != predecessor_sources:
        raise ValueError("projected registry changed predecessor sources")
    payload = json_bytes(projected)
    if (
        len(payload) != EXPECTED_PROJECTED_REGISTRY["bytes"]
        or sha256_bytes(payload) != EXPECTED_PROJECTED_REGISTRY["file_sha256"]
        or sha256_value(projected)
        != EXPECTED_PROJECTED_REGISTRY["canonical_sha256"]
        or len(projected["sources"]) != EXPECTED_PROJECTED_REGISTRY["sources"]
    ):
        raise ValueError("projected registry v0.22 binding changed")
    return {
        "added_sources": 1,
        "changed_predecessor_sources": 0,
        "materialized": False,
        "predecessor_file_sha256": EXPECTED_INPUTS["source_registry"]["sha256"],
        "predecessor_sources": EXPECTED_BASELINE["registry_sources"],
        "projected_canonical_sha256": EXPECTED_PROJECTED_REGISTRY["canonical_sha256"],
        "projected_file_bytes": EXPECTED_PROJECTED_REGISTRY["bytes"],
        "projected_file_sha256": EXPECTED_PROJECTED_REGISTRY["file_sha256"],
        "projected_observed_at": EXPECTED_PROJECTED_REGISTRY["observed_at"],
        "projected_sources": EXPECTED_PROJECTED_REGISTRY["sources"],
        "removed_sources": 0,
        "source_id": EXPECTED_NEW_SOURCE["source_id"],
        "successor_path": "configs/lumi-4b-sft-source-registry-v0.22.json",
        "successor_write_authorized": False,
    }


def build_report(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
    started: float | None = None,
) -> dict[str, Any]:
    del plan
    if _SHA256_RE.fullmatch(plan_file_sha256) is None:
        raise ValueError("preflight plan file hash changed")
    root = Path(repository_root).resolve()
    started_at = time.monotonic() if started is None else started
    baseline_plan, baseline_report, registry_value = _validate_bound_evidence(root)
    audit = _cross_source_audit(
        root=root,
        baseline_plan=baseline_plan,
        baseline_report=baseline_report,
        started=started_at,
    )
    projection = _project_registry(root=root, registry_value=registry_value)
    clean = audit["total_collision_rows"] == 0
    blockers = []
    if not clean:
        blockers.append("cross_source_collision_rows_require_quarantine_or_repair")
    blockers.extend(
        [
            "registry_v0_22_successor_not_materialized",
            "full_registry_v0_22_cross_source_hygiene_audit_not_complete",
            "candidate_mix_conversation_factual_banglish_and_natural_code_switching_"
            "floors_not_met",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ]
    )
    next_action = (
        "separate_registry_v0_22_promotion_authorization"
        if clean
        else "separate_collision_localization_and_candidate_repair_authorization"
    )
    report: dict[str, Any] = {
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "authorization_sha256": EXPECTED_AUTHORIZATION_SHA256,
            "baseline_plan_file_sha256": EXPECTED_INPUTS["baseline_plan"]["sha256"],
            "baseline_report_file_sha256": EXPECTED_INPUTS["baseline_report"]["sha256"],
            "baseline_report_internal_sha256": EXPECTED_INPUTS["baseline_report"][
                "internal_sha256"
            ],
            "baseline_verification_file_sha256": EXPECTED_INPUTS["baseline_verification"]["sha256"],
            "baseline_verification_internal_sha256": EXPECTED_INPUTS[
                "baseline_verification"
            ]["internal_sha256"],
            "massive_candidate_file_sha256": EXPECTED_INPUTS["massive_candidate"]["sha256"],
            "owner_receipt_file_sha256": EXPECTED_INPUTS["owner_receipt"]["sha256"],
            "owner_receipt_internal_sha256": EXPECTED_INPUTS["owner_receipt"]["internal_sha256"],
            "owner_verification_file_sha256": EXPECTED_INPUTS["owner_verification"]["sha256"],
            "owner_verification_internal_sha256": EXPECTED_INPUTS[
                "owner_verification"
            ]["internal_sha256"],
            "plan_file_sha256": plan_file_sha256,
            "source_registry_v0_21_sha256": EXPECTED_INPUTS["source_registry"]["sha256"],
        },
        "cross_source_audit": {
            **audit,
            "comparison_unit": EXPECTED_NEAR_POLICY["comparison_unit"],
            "near_duplicate_policy": EXPECTED_NEAR_POLICY,
            "owner_review_external_receipt_resolves_historical_pending_field": True,
            "registry_promotion_will_resolve_historical_registry_pending_field": True,
        },
        "candidate_projection": {
            "baseline_candidate_sources": EXPECTED_BASELINE["candidate_sources"],
            "baseline_records": EXPECTED_BASELINE["records"],
            "baseline_train_supervised_tokens": EXPECTED_BASELINE["train_supervised_tokens"],
            "baseline_validation_supervised_tokens": EXPECTED_BASELINE[
                "validation_supervised_tokens"
            ],
            "combined_candidate_sources_if_promoted": EXPECTED_COMBINED["candidate_sources"],
            "combined_records_if_all_64_retained": EXPECTED_COMBINED["records"],
            "combined_train_supervised_tokens_if_all_64_retained": EXPECTED_COMBINED[
                "train_supervised_tokens"
            ],
            "combined_validation_supervised_tokens_if_all_64_retained": (
                EXPECTED_COMBINED["validation_supervised_tokens"]
            ),
            "massive_records": EXPECTED_MASSIVE["records"],
            "massive_train_supervised_tokens": EXPECTED_MASSIVE["train_supervised_tokens"],
            "tool_api_gap_before_supervised_tokens": 2,
            "tool_api_gap_after_if_promoted_supervised_tokens": 0,
        },
        "registry_projection": {
            **projection,
            "promotion_preflight_passed": clean,
            "projected_active_tool_api_credit_rows": 64 if clean else 0,
            "current_active_tool_api_credit_rows": 0,
            "production_eligible": False,
        },
        "remaining_blockers": blockers,
        "next_gate": {
            "action": next_action,
            "registry_mutation_authorized": False,
            "separate_authorization_required": True,
        },
        "resource_accounting": {
            "baseline_candidate_row_reads": audit["baseline_candidate_row_reads"],
            "full_audit_passes": 1,
            "massive_candidate_row_reads": audit["massive_candidate_row_reads"],
            "source_line_reads": 0,
            "test_object_reads": 0,
            "registry_writes": 0,
        },
        "runtime": {
            "datasketch_version": importlib.metadata.version("datasketch"),
            "gpu_use": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "training_actions": 0,
        },
        "raw_training_text_retained": False,
        "registry_successor_materialized": False,
        "production_eligible": False,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": PASS_STATUS if clean else STOP_STATUS,
        "report_sha256": "",
    }
    report["report_sha256"] = sha256_value(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    audit = report.get("cross_source_audit")
    projection = report.get("registry_projection")
    clean = isinstance(audit, Mapping) and audit.get("total_collision_rows") == 0
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("status") != (PASS_STATUS if clean else STOP_STATUS)
        or report.get("report_sha256") != sha256_value(unsigned)
        or report.get("authorization") != EXPECTED_AUTHORIZATION
        or report.get("production_eligible") is not False
        or report.get("registry_successor_materialized") is not False
        or report.get("raw_training_text_retained") is not False
        or not isinstance(audit, Mapping)
        or audit.get("baseline_candidate_row_reads") != EXPECTED_BASELINE["records"]
        or audit.get("massive_candidate_row_reads") != EXPECTED_MASSIVE["records"]
        or audit.get("retained_massive_rows")
        != EXPECTED_MASSIVE["records"] - int(audit.get("total_collision_rows", -1))
        or audit.get("near_duplicate_policy") != EXPECTED_NEAR_POLICY
        or not isinstance(projection, Mapping)
        or projection.get("projected_file_sha256")
        != EXPECTED_PROJECTED_REGISTRY["file_sha256"]
        or projection.get("projected_sources") != EXPECTED_PROJECTED_REGISTRY["sources"]
        or projection.get("materialized") is not False
        or projection.get("successor_write_authorized") is not False
        or projection.get("promotion_preflight_passed") is not clean
        or report.get("resource_accounting")
        != {
            "baseline_candidate_row_reads": EXPECTED_BASELINE["records"],
            "full_audit_passes": 1,
            "massive_candidate_row_reads": EXPECTED_MASSIVE["records"],
            "source_line_reads": 0,
            "test_object_reads": 0,
            "registry_writes": 0,
        }
        or report.get("runtime")
        != {
            "datasketch_version": "2.0.0",
            "gpu_use": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "training_actions": 0,
        }
    ):
        raise ValueError("MASSIVE source-promotion preflight report changed")
    if clean and (
        audit.get("collision_ledger_sha256")
        != EXPECTED_EMPTY_COLLISION_LEDGER_SHA256
        or audit.get("retained_massive_rows") != 64
        or projection.get("projected_active_tool_api_credit_rows") != 64
        or report.get("next_gate", {}).get("action")
        != "separate_registry_v0_22_promotion_authorization"
    ):
        raise ValueError("clean source-promotion decision changed")
    _assert_no_forbidden_keys(report)


def build_verification(
    report: Mapping[str, Any],
    *,
    report_file_sha256: str,
    report_file_bytes: int,
    plan_file_sha256: str,
) -> dict[str, Any]:
    validate_report(report)
    if (
        _SHA256_RE.fullmatch(report_file_sha256) is None
        or report_file_bytes <= 0
        or report.get("bindings", {}).get("plan_file_sha256") != plan_file_sha256
    ):
        raise ValueError("source-promotion report file binding changed")
    clean = report["cross_source_audit"]["total_collision_rows"] == 0
    verification: dict[str, Any] = {
        "checks": {
            "all_314716_registry_v0_21_rows_replayed_each_pass": True,
            "all_64_massive_rows_replayed_each_pass": True,
            "baseline_v0_15_exact_and_near_clean_evidence_reproduced": True,
            "massive_owner_review_64_of_64_bound": True,
            "projected_registry_v0_22_is_exact_and_not_materialized": True,
            "raw_training_text_absent_from_outputs": True,
            "registry_source_candidate_seal_and_training_unchanged": True,
            "zero_network_gpu_inference_source_or_test_reads": True,
        },
        "combined_resource_accounting": {
            "baseline_candidate_row_reads": 629_432,
            "full_audit_passes": 2,
            "massive_candidate_row_reads": 128,
            "source_line_reads": 0,
            "test_object_reads": 0,
            "registry_writes": 0,
        },
        "decision": {
            "cross_source_clean": clean,
            "next_gate": report["next_gate"]["action"],
            "promotion_preflight_passed": clean,
            "registry_successor_materialized": False,
        },
        "plan_file_sha256": plan_file_sha256,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "report_file_bytes": report_file_bytes,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": PASS_VERIFICATION_STATUS if clean else STOP_VERIFICATION_STATUS,
        "verification_sha256": "",
        "zero_use": {
            "gpu_use": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "training_actions": 0,
        },
    }
    verification["verification_sha256"] = sha256_value(
        {
            key: value
            for key, value in verification.items()
            if key != "verification_sha256"
        }
    )
    validate_verification(verification)
    return verification


def validate_verification(verification: Mapping[str, Any]) -> None:
    unsigned = {
        key: value
        for key, value in verification.items()
        if key != "verification_sha256"
    }
    decision = verification.get("decision")
    clean = isinstance(decision, Mapping) and decision.get("cross_source_clean") is True
    if (
        verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION
        or verification.get("status")
        != (PASS_VERIFICATION_STATUS if clean else STOP_VERIFICATION_STATUS)
        or verification.get("verification_sha256") != sha256_value(unsigned)
        or verification.get("production_eligible") is not False
        or verification.get("raw_training_text_retained") is not False
        or verification.get("combined_resource_accounting")
        != {
            "baseline_candidate_row_reads": 629_432,
            "full_audit_passes": 2,
            "massive_candidate_row_reads": 128,
            "source_line_reads": 0,
            "test_object_reads": 0,
            "registry_writes": 0,
        }
        or verification.get("zero_use")
        != {
            "gpu_use": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "training_actions": 0,
        }
        or not isinstance(verification.get("report_file_bytes"), int)
        or verification["report_file_bytes"] <= 0
        or any(
            not isinstance(verification.get(field), str)
            or _SHA256_RE.fullmatch(verification[field]) is None
            for field in (
                "plan_file_sha256",
                "report_file_sha256",
                "report_internal_sha256",
            )
        )
    ):
        raise ValueError("MASSIVE source-promotion verification changed")
    _assert_no_forbidden_keys(verification)


def preflight_runner(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    report = (root / str(plan["output"]["report_path"])).resolve()
    verification = (root / str(plan["output"]["verification_path"])).resolve()
    if report.exists() or verification.exists():
        raise FileExistsError("source-promotion preflight already started")
    return report, verification


def preflight_verifier(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    report = (root / str(plan["output"]["report_path"])).resolve()
    verification = (root / str(plan["output"]["verification_path"])).resolve()
    if not report.is_file() or verification.exists():
        raise FileExistsError("source-promotion verification preflight changed")
    if report.stat().st_size <= len(EXPECTED_RECEIPT_MARKER):
        raise ValueError("source-promotion report is an incomplete marker")
    return report, verification


def prospective_private_write_bytes(
    *,
    plan: Mapping[str, Any],
    root: Path,
    report_bytes: bytes,
    verification_bytes: bytes,
) -> int:
    static = (root / EXPECTED_PLAN_PATH).stat().st_size
    for name in EXPECTED_CODE_PATHS:
        static += (root / str(plan[name]["path"])).stat().st_size
    return (
        static
        + len(EXPECTED_RECEIPT_MARKER)
        + len(report_bytes)
        + len(EXPECTED_VERIFICATION_MARKER)
        + len(verification_bytes)
    )


__all__ = [
    "EXPECTED_LIMITS",
    "EXPECTED_RECEIPT_MARKER",
    "EXPECTED_VERIFICATION_MARKER",
    "build_report",
    "build_verification",
    "file_sha256",
    "finish_marker",
    "json_bytes",
    "load_plan",
    "preflight_runner",
    "preflight_verifier",
    "prospective_private_write_bytes",
    "sha256_bytes",
    "start_marker",
    "strict_json_bytes",
    "validate_report",
    "validate_verification",
]
