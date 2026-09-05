"""Metadata-only source preflight for coding benchmark signatures."""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_opencodeinstruct_curation import file_sha256, sha256_value

PLAN_SCHEMA_VERSION = "lumi.code-benchmark-signature-source-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.code-benchmark-signature-source-preflight.v0.1"
ACQUISITION_PLAN_SCHEMA_VERSION = (
    "lumi.code-benchmark-signature-acquisition-plan.v0.1"
)
VERIFICATION_SCHEMA_VERSION = (
    "lumi.code-benchmark-signature-source-preflight-verification.v0.1"
)

OWNER_AUTHORIZATION_STATEMENT = (
    "I approve one bounded CPU-only LiveCodeBench and BigCodeBench benchmark-signature "
    "source preflight, maximum 60 minutes / 16 MiB network, limited to official "
    "repositories, dataset cards, LICENSE/README files, pinned revision discovery, "
    "and exact object inventory. It may audit rights and provenance and produce a "
    "hash-only decontamination acquisition plan. No benchmark bulk-data or solution "
    "download, external API, GPU, candidate-registry promotion, dataset seal, SFT "
    "training, unrelated data, or external distribution."
)
EXPECTED_AUTHORIZATION = {
    "benchmark_bulk_data_authorized": False,
    "candidate_registry_promotion_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "metadata_only_network_authorized": True,
    "sft_training_authorized": False,
    "signature_registry_promotion_authorized": False,
    "solution_or_test_download_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_network_bytes": 16_777_216,
    "maximum_runtime_minutes": 60,
}
EXPECTED_CAPTURED_BYTES = 135_458

LIVE_GITHUB_REVISION = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
LIVE_GITHUB_TREE = "65790b9fc909976194ff54c84f673395c2a6c85d"
LIVE_HF_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
BIG_GITHUB_REVISION = "09dd993f46c3fbf3a799465bb96d524edcb0b199"
BIG_GITHUB_TREE = "26e669cf5283ac6c314fb3e47b9cb71c09aa23fa"
BIG_HF_REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"

METADATA_BINDINGS = {
    "gh_big_head",
    "gh_big_license",
    "gh_big_readme",
    "gh_big_repo",
    "gh_big_tree",
    "gh_live_head",
    "gh_live_license",
    "gh_live_readme",
    "gh_live_repo",
    "gh_live_tree",
    "hf_big_readme",
    "hf_big_repo",
    "hf_big_tree",
    "hf_live_readme",
    "hf_live_repo",
    "hf_live_tree",
}
LOCAL_BINDINGS = {
    "active_benchmark_registry",
    "benchmark_builder",
    "data_quality_config",
    "implementation",
    "runner",
    "verifier",
}
EXPECTED_BINDINGS = METADATA_BINDINGS | LOCAL_BINDINGS

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _read_json(path: str | Path) -> dict[str, Any]:
    value = _read_json_value(path)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_json_value(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_binding(
    repository_root: str | Path,
    specification: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    relative = specification.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid code-benchmark preflight binding: {name}")
    path = (root / relative).resolve()
    if not path.is_relative_to(workspace):
        raise ValueError(f"code-benchmark preflight binding escapes workspace: {name}")
    if not path.is_file():
        raise FileNotFoundError(f"code-benchmark preflight binding missing: {name}")
    expected = _require_sha256(specification.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(path) != expected:
        raise ValueError(f"code-benchmark preflight binding changed: {name}")
    if name in METADATA_BINDINGS:
        expected_bytes = specification.get("bytes")
        if not isinstance(expected_bytes, int) or path.stat().st_size != expected_bytes:
            raise ValueError(f"code-benchmark metadata size changed: {name}")
        url = specification.get("url")
        if not isinstance(url, str) or not url.startswith(
            ("https://api.github.com/", "https://huggingface.co/", "https://raw.githubusercontent.com/")
        ):
            raise ValueError(f"code-benchmark metadata URL changed: {name}")
    return path


def load_preflight_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _read_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported code-benchmark preflight plan")
    if plan.get("observed_at") != "2026-08-07":
        raise ValueError("code-benchmark preflight observation date changed")
    if plan.get("owner_authorization_statement") != OWNER_AUTHORIZATION_STATEMENT:
        raise ValueError("code-benchmark preflight owner authorization changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("code-benchmark preflight authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("code-benchmark preflight limits changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("code-benchmark preflight binding inventory changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid code-benchmark binding object: {name}")
        resolve_binding(repository_root, specification, name=name)
    captured = sum(int(bindings[name]["bytes"]) for name in METADATA_BINDINGS)
    if captured != EXPECTED_CAPTURED_BYTES or captured > EXPECTED_LIMITS[
        "maximum_network_bytes"
    ]:
        raise ValueError("code-benchmark captured metadata bytes changed")
    output = plan.get("output")
    if not isinstance(output, Mapping) or set(output) != {
        "acquisition_plan_path",
        "raw_benchmark_text_in_git",
        "report_path",
        "verification_path",
    }:
        raise ValueError("code-benchmark preflight output inventory changed")
    if output.get("raw_benchmark_text_in_git") is not False:
        raise ValueError("code-benchmark preflight cannot retain benchmark text in Git")
    return plan


def _json_binding(
    plan: Mapping[str, Any], repository_root: str | Path, name: str
) -> Any:
    return _read_json_value(
        resolve_binding(repository_root, plan["bindings"][name], name=name)
    )


def _text_binding(
    plan: Mapping[str, Any], repository_root: str | Path, name: str
) -> str:
    return resolve_binding(
        repository_root, plan["bindings"][name], name=name
    ).read_text(encoding="utf-8")


def _hf_inventory(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    siblings = value.get("siblings")
    if not isinstance(siblings, list):
        raise ValueError("Hugging Face object inventory is missing")
    inventory: list[dict[str, Any]] = []
    for sibling in siblings:
        if not isinstance(sibling, Mapping):
            raise ValueError("invalid Hugging Face inventory row")
        item: dict[str, Any] = {
            "bytes": sibling.get("size"),
            "git_blob_sha1": sibling.get("blobId"),
            "path": sibling.get("rfilename"),
        }
        lfs = sibling.get("lfs")
        if isinstance(lfs, Mapping):
            item["lfs_sha256"] = lfs.get("sha256")
        if not isinstance(item["bytes"], int) or not isinstance(item["path"], str):
            raise ValueError("invalid Hugging Face object inventory fields")
        inventory.append(item)
    return sorted(inventory, key=lambda item: item["path"])


def _metadata_inventory(
    plan: Mapping[str, Any], repository_root: str | Path
) -> list[dict[str, Any]]:
    root = Path(repository_root).resolve()
    inventory = []
    for name in sorted(METADATA_BINDINGS):
        specification = plan["bindings"][name]
        path = resolve_binding(root, specification, name=name)
        inventory.append(
            {
                "bytes": path.stat().st_size,
                "name": name,
                "sha256": file_sha256(path),
                "url": specification["url"],
            }
        )
    return inventory


def _active_registry_metadata(path: Path) -> dict[str, Any]:
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    if not isinstance(first, Mapping) or not isinstance(first.get("_meta"), Mapping):
        raise ValueError("active benchmark registry metadata is invalid")
    return dict(first["_meta"])


def build_preflight_outputs(
    plan: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    gh_live_repo = _json_binding(plan, root, "gh_live_repo")
    gh_live_head = _json_binding(plan, root, "gh_live_head")
    gh_live_tree = _json_binding(plan, root, "gh_live_tree")
    gh_big_repo = _json_binding(plan, root, "gh_big_repo")
    gh_big_head = _json_binding(plan, root, "gh_big_head")
    gh_big_tree = _json_binding(plan, root, "gh_big_tree")
    hf_live_repo = _json_binding(plan, root, "hf_live_repo")
    hf_big_repo = _json_binding(plan, root, "hf_big_repo")
    hf_live_tree = _json_binding(plan, root, "hf_live_tree")
    hf_big_tree = _json_binding(plan, root, "hf_big_tree")
    hf_live_readme = _text_binding(plan, root, "hf_live_readme")
    hf_big_readme = _text_binding(plan, root, "hf_big_readme")
    gh_live_license = _text_binding(plan, root, "gh_live_license")
    gh_big_license = _text_binding(plan, root, "gh_big_license")

    if gh_live_repo.get("full_name") != "LiveCodeBench/LiveCodeBench":
        raise ValueError("official LiveCodeBench GitHub repository changed")
    if gh_live_head.get("sha") != LIVE_GITHUB_REVISION:
        raise ValueError("LiveCodeBench GitHub revision changed")
    if gh_live_head.get("commit", {}).get("tree", {}).get("sha") != LIVE_GITHUB_TREE:
        raise ValueError("LiveCodeBench GitHub tree changed")
    if gh_live_tree.get("sha") != LIVE_GITHUB_TREE or gh_live_tree.get("truncated"):
        raise ValueError("LiveCodeBench GitHub inventory is incomplete")
    if gh_live_repo.get("license", {}).get("spdx_id") != "MIT":
        raise ValueError("LiveCodeBench software license changed")
    if not gh_live_license.startswith("MIT License\n"):
        raise ValueError("LiveCodeBench license text changed")

    if gh_big_repo.get("full_name") != "bigcode-project/bigcodebench":
        raise ValueError("official BigCodeBench GitHub repository changed")
    if gh_big_head.get("sha") != BIG_GITHUB_REVISION:
        raise ValueError("BigCodeBench GitHub revision changed")
    if gh_big_head.get("commit", {}).get("tree", {}).get("sha") != BIG_GITHUB_TREE:
        raise ValueError("BigCodeBench GitHub tree changed")
    if gh_big_tree.get("sha") != BIG_GITHUB_TREE or gh_big_tree.get("truncated"):
        raise ValueError("BigCodeBench GitHub inventory is incomplete")
    if gh_big_repo.get("license", {}).get("spdx_id") != "Apache-2.0":
        raise ValueError("BigCodeBench software license changed")
    if "Apache License\n                           Version 2.0" not in gh_big_license:
        raise ValueError("BigCodeBench license text changed")

    if hf_live_repo.get("id") != "livecodebench/code_generation_lite":
        raise ValueError("official LiveCodeBench dataset repository changed")
    if hf_live_repo.get("sha") != LIVE_HF_REVISION:
        raise ValueError("LiveCodeBench dataset revision changed")
    if hf_live_repo.get("cardData", {}).get("license") != "cc":
        raise ValueError("LiveCodeBench dataset license declaration changed")
    if not isinstance(hf_live_tree, list) or len(hf_live_tree) != 11:
        raise ValueError("LiveCodeBench dataset object inventory changed")
    if "LeetCode, AtCoder, and Codeforces" not in hf_live_readme:
        raise ValueError("LiveCodeBench source provenance declaration changed")

    if hf_big_repo.get("id") != "bigcode/bigcodebench":
        raise ValueError("official BigCodeBench dataset repository changed")
    if hf_big_repo.get("sha") != BIG_HF_REVISION:
        raise ValueError("BigCodeBench dataset revision changed")
    if hf_big_repo.get("cardData", {}).get("license") != "apache-2.0":
        raise ValueError("BigCodeBench dataset license declaration changed")
    if not isinstance(hf_big_tree, list) or len(hf_big_tree) != 10:
        raise ValueError("BigCodeBench dataset object inventory changed")
    if "originally sourced from GPT-4-0613" not in hf_big_readme:
        raise ValueError("BigCodeBench source provenance declaration changed")

    live_inventory = _hf_inventory(hf_live_repo)
    big_inventory = _hf_inventory(hf_big_repo)
    live_v6 = next(item for item in live_inventory if item["path"] == "test6.jsonl")
    big_v014 = next(
        item
        for item in big_inventory
        if item["path"] == "data/v0.1.4-00000-of-00001.parquet"
    )
    if live_v6 != {
        "bytes": 134_303_240,
        "git_blob_sha1": "f862a72b62f61fb9592351329bd4cd0d2d468efa",
        "lfs_sha256": "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5",
        "path": "test6.jsonl",
    }:
        raise ValueError("LiveCodeBench release-v6 object changed")
    if big_v014 != {
        "bytes": 2_362_110,
        "git_blob_sha1": "ef4634c67d030dd64fd23f5304afdebed11cc6ed",
        "lfs_sha256": "d9a4965821c9507ebdfb551c288656b2d5fe553234f5183044333ca8a4018267",
        "path": "data/v0.1.4-00000-of-00001.parquet",
    }:
        raise ValueError("BigCodeBench v0.1.4 object changed")

    registry_path = resolve_binding(
        root,
        plan["bindings"]["active_benchmark_registry"],
        name="active_benchmark_registry",
    )
    registry_metadata = _active_registry_metadata(registry_path)
    if registry_metadata.get("entries") != 46_999:
        raise ValueError("active benchmark registry entry count changed")
    metadata_inventory = _metadata_inventory(plan, root)

    acquisition_plan: dict[str, Any] = {
        "active_registry": {
            "entries": 46_999,
            "path": "data/benchmark-signatures-v0.1.jsonl.gz",
            "sha256": plan["bindings"]["active_benchmark_registry"]["sha256"],
        },
        "authorization": {
            "acquisition_authorized": False,
            "active_registry_mutation_authorized": False,
            "gpu_or_sft_authorized": False,
        },
        "bigcodebench": {
            "decision": "eligible_for_separate_private_prompt_only_hash_acquisition",
            "dataset_revision": BIG_HF_REVISION,
            "declared_license": "apache-2.0",
            "expected_examples": 1_140,
            "maximum_new_signatures": 3_420,
            "network_object": big_v014,
            "prompt_fields": ["code_prompt", "complete_prompt", "instruct_prompt"],
            "prohibited_signature_fields": [
                "canonical_solution",
                "doc_struct",
                "entry_point",
                "libs",
                "test",
            ],
            "provenance": "GPT-4-0613 with ODEX/StackOverflow seeds plus expert and LLM annotation",
        },
        "livecodebench": {
            "decision": "blocked_pending_dataset_content_license_clarification",
            "dataset_revision": LIVE_HF_REVISION,
            "declared_dataset_license": "cc_non_spdx_ambiguous",
            "github_software_license": "MIT",
            "proposed_release_object_not_authorized": live_v6,
            "provenance": "competition problems from LeetCode, AtCoder, and Codeforces",
            "reason": (
                "MIT covers the repository software, while the dataset card only says cc "
                "and does not resolve upstream contest-problem rights"
            ),
        },
        "next_authorization_statement": (
            "I approve one bounded CPU-only BigCodeBench v0.1.4 prompt-signature "
            "acquisition at Hugging Face revision "
            "b74c0d0bf70d2c0bc459be537895cca163007f1a, limited to exact object "
            "data/v0.1.4-00000-of-00001.parquet (2,362,110 bytes; SHA-256 "
            "d9a4965821c9507ebdfb551c288656b2d5fe553234f5183044333ca8a4018267), "
            "maximum 4 MiB network / 32 MiB decoded / 60 minutes. It may read only "
            "task_id, complete_prompt, instruct_prompt, and code_prompt, create at most "
            "3,420 private prompt signatures, and build a raw-text-free candidate "
            "successor preserving all 46,999 existing registry entries. It must not "
            "retain or hash canonical_solution, test, doc_struct, entry_point, or libs. "
            "No external API, GPU, active-registry mutation, candidate-source promotion, "
            "dataset seal, SFT training, LiveCodeBench data, unrelated data, or external "
            "distribution."
        ),
        "observed_at": "2026-08-07",
        "registry_successor_requirements": {
            "active_registry_unchanged": True,
            "all_existing_entries_preserved": True,
            "candidate_path": "data/benchmark-signatures-v0.2.candidate.jsonl.gz",
            "raw_benchmark_text_retained": False,
            "requires_independent_replay": True,
        },
        "schema_version": ACQUISITION_PLAN_SCHEMA_VERSION,
        "status": "proposed_not_authorized_livecodebench_rights_blocked",
    }
    acquisition_plan["plan_sha256"] = sha256_value(acquisition_plan)

    report: dict[str, Any] = {
        "authorization": EXPECTED_AUTHORIZATION,
        "benchmark_data_downloaded": False,
        "captured_metadata": {
            "objects": metadata_inventory,
            "payload_bytes": sum(item["bytes"] for item in metadata_inventory),
            "response_objects": len(metadata_inventory),
        },
        "decisions": {
            "bigcodebench": "eligible_for_separate_private_prompt_only_hash_acquisition",
            "livecodebench": "blocked_pending_dataset_content_license_clarification",
        },
        "inventories": {
            "bigcodebench": {
                "dataset_objects": big_inventory,
                "dataset_revision": BIG_HF_REVISION,
                "github_commit": BIG_GITHUB_REVISION,
                "github_repository_archived": gh_big_repo.get("archived"),
                "total_dataset_object_bytes": sum(item["bytes"] for item in big_inventory),
            },
            "livecodebench": {
                "dataset_objects": live_inventory,
                "dataset_revision": LIVE_HF_REVISION,
                "github_commit": LIVE_GITHUB_REVISION,
                "github_repository_archived": gh_live_repo.get("archived"),
                "total_dataset_object_bytes": sum(item["bytes"] for item in live_inventory),
            },
        },
        "observed_at": "2026-08-07",
        "production_eligible": False,
        "raw_benchmark_text_in_git": False,
        "remaining_blockers": [
            "bigcodebench_prompt_signature_acquisition_not_authorized",
            "livecodebench_dataset_content_license_unresolved",
            "active_benchmark_registry_successor_not_built_or_promoted",
            "opencodeinstruct_candidate_registry_promotion_not_authorized",
            "dataset_seal_and_sft_not_authorized",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "signatures_created": 0,
        "solutions_or_tests_downloaded": False,
        "status": "verified_metadata_preflight_bigcodebench_ready_livecodebench_blocked",
    }
    report["report_sha256"] = sha256_value(report)
    return report, acquisition_plan


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported code-benchmark preflight report")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("code-benchmark preflight report seal is invalid")
    if report.get("benchmark_data_downloaded") is not False:
        raise ValueError("code-benchmark preflight downloaded benchmark data")
    if report.get("solutions_or_tests_downloaded") is not False:
        raise ValueError("code-benchmark preflight downloaded solutions or tests")
    if report.get("signatures_created") != 0:
        raise ValueError("code-benchmark preflight crossed signature-build boundary")
    if report.get("production_eligible") is not False:
        raise ValueError("code-benchmark preflight crossed production boundary")


def validate_acquisition_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != ACQUISITION_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported code-benchmark acquisition plan")
    seal = _require_sha256(plan.get("plan_sha256"), "plan_sha256")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("code-benchmark acquisition plan seal is invalid")
    if plan.get("authorization", {}).get("acquisition_authorized") is not False:
        raise ValueError("code-benchmark acquisition plan crossed authorization boundary")
    if plan.get("livecodebench", {}).get("decision") != (
        "blocked_pending_dataset_content_license_clarification"
    ):
        raise ValueError("LiveCodeBench rights blocker disappeared")


__all__ = [
    "ACQUISITION_PLAN_SCHEMA_VERSION",
    "EXPECTED_AUTHORIZATION",
    "EXPECTED_CAPTURED_BYTES",
    "EXPECTED_LIMITS",
    "OWNER_AUTHORIZATION_STATEMENT",
    "PLAN_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "VERIFICATION_SCHEMA_VERSION",
    "build_preflight_outputs",
    "load_preflight_plan",
    "resolve_binding",
    "validate_acquisition_plan",
    "validate_report",
]
