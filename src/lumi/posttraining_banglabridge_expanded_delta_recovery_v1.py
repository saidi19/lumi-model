"""Recover and audit the authorized 445-row BanglaBridge expansion."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import posttraining_banglabridge_hygiene_repair as prior_repair
from . import posttraining_candidate_mix_reaudit as v2
from . import posttraining_candidate_mix_reaudit_v14 as v14
from .posttraining_candidate_mix_preflight import (
    EXPECTED_LANGUAGE_MODES,
    EXPECTED_TASKS,
    _canonical_message_digest,
    _duplicate_snapshot,
    _load_json,
    _load_jsonl,
    _targets,
    file_sha256,
    sha256_value,
)
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_first_party import canonical_json_bytes
from .posttraining_format import (
    encode_sft_messages,
    load_chat_template,
    verify_chat_tokenizer,
)
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
)
from .posttraining_owner_review import (
    DECISIONS_SCHEMA_VERSION,
    build_owner_review_html,
)
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-banglabridge-expanded-delta-recovery-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-banglabridge-expanded-delta-recovery-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-banglabridge-expanded-delta-recovery-verification.v0.1"
)
STATUS = "PASS_private_banglabridge_445_row_expanded_delta_owner_review_pending"
VERIFICATION_STATUS = (
    "PASS_verified_private_banglabridge_445_row_expanded_delta_owner_review_pending"
)
EXPECTED_VERIFICATION_CHECKS = {
    "all_314716_rows_replayed_in_independent_second_pass": True,
    "all_445_delta_rows_and_608_source_rows_reconstructed": True,
    "candidate_delta_ledger_and_owner_packet_bytes_reproduced": True,
    "cross_source_exact_split_and_minhash_0_9_results_reproduced": True,
    "generic_owner_decision_schema_preserved": True,
    "no_network_gpu_inference_registry_seal_sft_or_production_action": True,
    "no_raw_training_text_in_report_or_verification": True,
    "owner_review_remains_pending": True,
    "packet_file_hash_domain_reproduced": True,
    "predecessors_and_source_remain_unmodified": True,
    "static_only_html_customization_verified": True,
    "unicode_content_hash_domain_reproduced": True,
}
SOURCE_ID = "banglabridge_instructions_closure_canary"
SOURCE_FAMILY = "banglabridge_family"

EXPECTED_AUTHORIZATION_TEXT_SHA256 = (
    "dc943fde2aaab4a6506d5aef20ab272c8c4ed727298b9481ac2e450d1e736613"
)
SHA256_PATTERN = r"^[0-9a-f]{64}$"
EXPECTED_OUTPUT_NAMES = frozenset(
    {"candidate", "delta_ledger", "owner_review_html", "owner_review_packet"}
)
EXPECTED_RUNTIME_RECOVERY = {
    "failed_attempt": {
        "candidate_input_rows_parsed": 561,
        "full_candidate_audit_passes_completed": 0,
        "full_candidate_audit_row_reads": 0,
        "source_input_rows_parsed": 608,
        "success_output_files_written": 0,
    },
    "predecessor_bindings": {
        "failed_implementation_sha256": (
            "2c94b25d6b068348cbcab120f409e54a871b089686bc3c2c0e7f51316824d9fa"
        ),
        "failed_plan_sha256": (
            "0dc8bd59cd8b73f87372ee81ee8e6cf555fb1679291d81e5e52cf3b1fecfd3c6"
        ),
        "fixed_pre_successor_implementation_sha256": (
            "d55d120ec2cf45fc1a00ef146d3849729f4a9df181544f242c8a997a4b499b90"
        ),
        "fixed_pre_successor_plan_sha256": (
            "3955db4141b10a81f3964615950ef63c61b65ef2ee15c3388ef5185ed2d45c0d"
        ),
        "original_authorization_text_sha256": (
            "422a8bb02815d6be95c2a497adfd9d935c4240d03df526616fceb9488cb0630b"
        ),
        "runner_sha256": (
            "2d01b3f3ad5066d129f9871317782824a1bbd8673961ec99590f51d47f6c9794"
        ),
        "test_sha256": (
            "61d3aa0f13772df1b80cee8065c360694810dd654cb377f4f9050224c63fec01"
        ),
        "verifier_sha256": (
            "5e5a1d85ae98787bfc9f3aceefb78f681569e773672c94a77cb947bc84429133"
        ),
    },
    "repair": "preserve_fixed_historical_validation_id_exclusion_only",
}
EXPECTED_AUTHORIZATION = {
    "candidate_materialization_authorized": True,
    "dataset_seal_authorized": False,
    "download_authorized": False,
    "external_api_authorized": False,
    "full_local_reaudit_authorized": True,
    "gpu_authorized": False,
    "inference_authorized": False,
    "network_authorized": False,
    "owner_review_packet_authorized": True,
    "predecessor_mutation_authorized": False,
    "production_promotion_authorized": False,
    "registry_mutation_authorized": False,
    "remote_access_authorized": False,
    "sft_training_authorized": False,
    "source_mutation_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_candidate_records_per_audit_pass": 314_716,
    "maximum_full_audit_passes": 2,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 16 * 1024 * 1024,
    "maximum_runtime_minutes": 120,
    "maximum_source_row_reads": 1_216,
    "maximum_total_candidate_row_reads": 630_554,
    "maximum_total_candidate_audit_row_reads": 629_432,
}
EXPECTED_NEAR_DUPLICATE_POLICY = dict(v14.EXPECTED_NEAR_DUPLICATE_POLICY)

EXPECTED_BINDINGS = frozenset(
    {
        "benchmark_registry",
        "chat_template",
        "data_contract",
        "failure_report",
        "failure_report_v02",
        "failure_verification",
        "failure_verification_v02",
        "implementation",
        "included_candidate",
        "original_candidate",
        "original_owner_receipt",
        "original_owner_sample",
        "original_owner_verification",
        "prior_repair_plan",
        "prior_repair_report",
        "prior_repair_verification",
        "rights_preflight",
        "rights_preflight_verification",
        "runner",
        "source_readme",
        "source_registry",
        "source_rows",
        "test",
        "tokenizer_json",
        "v14_plan",
        "v14_report",
        "v14_verification",
        "verifier",
    }
)
EXPECTED_INPUT_SHA256 = {
    "benchmark_registry": "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e",
    "chat_template": "bc7853a9277feddae2fd3db786c6a12f4fa3602a135a0408c38d4390f33bec12",
    "data_contract": "9a283d9333d29deb7b516fa45eb4eec50232be9925f4ee4aa3a8f624213b2f06",
    "failure_report": "02ec3848a80c59225c90a1d9e942bf59beaf5fe2c9e6e9d02b45bebbc6cb0941",
    "failure_verification": (
        "7ed4d36c20aacc362f7ca75ac39ea86c2dc56c6cb4716638cbd2c4de0ed3b5fc"
    ),
    "failure_report_v02": (
        "b5c2433736fc8a99588ef17d7853a14c5586cf0e39694604a17948852d0bbac5"
    ),
    "failure_verification_v02": (
        "e77c6ce26d5491f7ac1af1cdf3e45de9e0652c54156df6be77e67221f3e1c44a"
    ),
    "included_candidate": "e32a8b74202c3bc40674da326c2998e2600a486a335524eb79e245e99061ccba",
    "original_candidate": "b75c3517be8dba9295d36f93e3275f06d1287da7e2fe5b60fdfadbc4f27e9900",
    "original_owner_receipt": "f42cf8bacfb70c9d5e09d411512f78a04bf974a45163be76cba815d3cdb26b09",
    "original_owner_sample": "8349eac20f3370922fda1a94229aadd4a7d621374d63717bddbf5510c1e190aa",
    "original_owner_verification": (
        "956e1dea53a2164befd2d9456136357f27147eb116e19b7ecda7ca0bfdc5ca92"
    ),
    "prior_repair_plan": "b455248a0a6b49050349fc4395eb165198e7b4eb15b9b33308d75b25591b355e",
    "prior_repair_report": "0530f983eea5c56e7ed543924852f8d221b9e851c6038c71b58716cf04785eef",
    "prior_repair_verification": "69da46f43e0cafd6cd28319ae0398d1720dd06b5307d14c8d5c85ba32a8037ce",
    "rights_preflight": "9f725bc8761e8f6f5fccea20e81b36a3c72a465084a8a68c422ad9cf2f44ef3c",
    "rights_preflight_verification": (
        "edc13f19b67de6147e104f3848e0005c3da9772b8e214fe0fe592b7aa7b560ec"
    ),
    "source_readme": "f97019bf4d561e5c399418181347eec9448984fc34e181f13fca53ac1a7b5850",
    "source_registry": "ce5de72f3cd308aaf99b35c1316236a316959f87beed6bd9876c2e28be0a2e7b",
    "source_rows": "313e5efb38760d8de09050162b6d8c5a2db76b9107cb2c24d828012113a7a73d",
    "tokenizer_json": "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee",
    "v14_plan": "3b4ef584b1ca8196db5451ff101e8993eaeb0d6d708cb76d75d95304f1be02dd",
    "v14_report": "73a5c39e30fb8e64336211b33ae3bcd0a1a59849302d834ff260877d94e9c120",
    "v14_verification": "2ba639e39e26af3e06f184d0f5ab2775c208b24d8ffa755c3b6e6bb9c4f518dd",
}

EXPECTED_ORIGINAL_ROWS = 561
EXPECTED_INCLUDED_ROWS = 116
EXPECTED_DELTA_ROWS = 445
EXPECTED_SOURCE_ROWS = 608
EXPECTED_SUCCESSOR_SHA256 = (
    "5cf0b5f4ad4029f7a9e25b49ac7785f43607fb3196baf8c9f5dd272da0551058"
)
EXPECTED_SUCCESSOR_BYTES = 1_967_432
EXPECTED_DELTA_SHA256 = (
    "bf6617ed8e0a7a95710c1c046d99d0954cdbd2136574f15ea9909aebc5fecce2"
)
EXPECTED_DELTA_BYTES = 1_554_396
EXPECTED_SUCCESSOR_TRAIN_ROWS = 544
EXPECTED_SUCCESSOR_VALIDATION_ROWS = 17
EXPECTED_SUCCESSOR_TRAIN_TOKENS = 250_441
EXPECTED_SUCCESSOR_VALIDATION_TOKENS = 4_960
EXPECTED_DELTA_TRAIN_TOKENS = 200_685
EXPECTED_DELTA_LANGUAGE_TOKENS = {
    "bengali_script": 191_691,
    "natural_code_switching": 8_994,
}
EXPECTED_MAXIMUM_SERIALIZED_TOKENS = 3_872
EXPECTED_AUDIT_RECORDS = 314_716
EXPECTED_AUDIT_TRAIN_TOKENS = 10_604_149
EXPECTED_AUDIT_VALIDATION_TOKENS = 628_604
EXPECTED_LANGUAGE_GAPS = {
    "banglish": 772_114,
    "bengali_script": 0,
    "english": 0,
    "natural_code_switching": 365_420,
}
EXPECTED_TASK_GAPS = {
    "coding": 0,
    "conversation": 5_934,
    "factual_qa": 114_306,
    "multilingual_bengali": 0,
    "reasoning_math_science": 0,
    "safety_helpfulness": 0,
    "tool_api": 2,
    "writing_summarization": 0,
}
EXPECTED_PROVENANCE_FIELDS = (
    "source_adapted_instruction_sha256",
    "source_adapted_output_sha256",
    "source_original_instruction_sha256",
    "source_original_output_sha256",
)


def _check_runtime(started_at: float) -> None:
    per_pass_seconds = (
        EXPECTED_LIMITS["maximum_runtime_minutes"]
        * 60
        / EXPECTED_LIMITS["maximum_full_audit_passes"]
    )
    if time.monotonic() - started_at > per_pass_seconds:
        raise TimeoutError("expanded-delta local runtime cap exceeded")


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
        for row in rows
    )


def _resolve_binding(root: Path, binding: Mapping[str, Any], name: str) -> Path:
    path = (root / str(binding.get("path"))).resolve()
    if not path.is_relative_to(root.parent.resolve()) or not path.is_file():
        raise ValueError(f"invalid or missing expanded-delta binding: {name}")
    if file_sha256(path) != binding.get("sha256"):
        raise ValueError(f"expanded-delta binding hash mismatch: {name}")
    return path


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(Path(path).resolve())
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "expected": {
            "audit_records": EXPECTED_AUDIT_RECORDS,
            "delta_records": EXPECTED_DELTA_ROWS,
            "delta_train_supervised_tokens": EXPECTED_DELTA_TRAIN_TOKENS,
            "owner_review_minimum_accepts": 29,
            "owner_review_records": 32,
            "successor_records": EXPECTED_ORIGINAL_ROWS,
            "successor_train_supervised_tokens": EXPECTED_SUCCESSOR_TRAIN_TOKENS,
            "successor_validation_supervised_tokens": (
                EXPECTED_SUCCESSOR_VALIDATION_TOKENS
            ),
        },
        "limits": EXPECTED_LIMITS,
        "near_duplicate_policy": EXPECTED_NEAR_DUPLICATE_POLICY,
        "observed_at": "2026-08-13",
        "output": {
            "candidate_path": (
                "artifacts/lumi-sft-banglabridge-expanded-delta-v0.1/"
                "attempt-001/private/banglabridge-candidate-561.jsonl"
            ),
            "delta_ledger_path": (
                "artifacts/lumi-sft-banglabridge-expanded-delta-v0.1/"
                "attempt-001/private/delta-ledger-445.jsonl"
            ),
            "owner_review_html_path": (
                "artifacts/lumi-sft-banglabridge-expanded-delta-v0.1/"
                "attempt-001/private/owner-review-32.html"
            ),
            "owner_review_packet_path": (
                "artifacts/lumi-sft-banglabridge-expanded-delta-v0.1/"
                "attempt-001/private/owner-review-32.json"
            ),
            "report_path": (
                "evidence/lumi-4b-sft-banglabridge-expanded-delta-recovery-v0.1.json"
            ),
            "verification_path": (
                "evidence/lumi-4b-sft-banglabridge-expanded-delta-recovery-"
                "verification-v0.1.json"
            ),
        },
        "schema_version": PLAN_SCHEMA_VERSION,
        "runtime_recovery": EXPECTED_RUNTIME_RECOVERY,
        "status": (
            "authorized_private_local_expanded_delta_owner_packet_correctness_"
            "recovery_no_promotion"
        ),
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"expanded-delta plan changed: {field}")
    original_authorization_text = plan.get("authorization_text")
    authorization_text = plan.get("runtime_recovery_authorization_text")
    owner_packet_authorization_text = plan.get(
        "owner_packet_correctness_authorization_text"
    )
    if (
        not isinstance(original_authorization_text, str)
        or hashlib.sha256(original_authorization_text.encode("utf-8")).hexdigest()
        != EXPECTED_RUNTIME_RECOVERY["predecessor_bindings"][
            "original_authorization_text_sha256"
        ]
        or plan.get("authorization_text_sha256")
        != EXPECTED_RUNTIME_RECOVERY["predecessor_bindings"][
            "original_authorization_text_sha256"
        ]
        or not isinstance(authorization_text, str)
        or hashlib.sha256(authorization_text.encode("utf-8")).hexdigest()
        != "5396120ef6f39e5d21b4700fd85d855dd1abc39524fac5f22c2c2746fa340d87"
        or plan.get("runtime_recovery_authorization_text_sha256")
        != "5396120ef6f39e5d21b4700fd85d855dd1abc39524fac5f22c2c2746fa340d87"
        or not isinstance(owner_packet_authorization_text, str)
        or hashlib.sha256(owner_packet_authorization_text.encode("utf-8")).hexdigest()
        != EXPECTED_AUTHORIZATION_TEXT_SHA256
        or plan.get("owner_packet_correctness_authorization_text_sha256")
        != EXPECTED_AUTHORIZATION_TEXT_SHA256
    ):
        raise ValueError("expanded-delta exact authorization text changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("expanded-delta binding inventory changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid expanded-delta binding: {name}")
        _resolve_binding(root, binding, name)
    for name, expected_sha256 in EXPECTED_INPUT_SHA256.items():
        if bindings[name].get("sha256") != expected_sha256:
            raise ValueError(f"expanded-delta exact input binding changed: {name}")
    if bindings["runner"].get("sha256") != EXPECTED_RUNTIME_RECOVERY[
        "predecessor_bindings"
    ]["runner_sha256"]:
        raise ValueError("expanded-delta frozen runner changed")
    return plan


def _bound_paths(
    plan: Mapping[str, Any], root: Path
) -> dict[str, Path]:
    return {
        name: _resolve_binding(root, binding, name)
        for name, binding in plan["bindings"].items()
    }


def _validate_evidence(bound: Mapping[str, Path], root: Path) -> None:
    failure_report = _load_json(bound["failure_report"])
    failure_verification = _load_json(bound["failure_verification"])
    failure_unsigned = dict(failure_report)
    failure_internal = failure_unsigned.pop("failure_report_sha256", None)
    verification_unsigned = dict(failure_verification)
    verification_internal = verification_unsigned.pop("verification_sha256", None)
    if (
        failure_report.get("status")
        != "FAIL_CLOSED_pre_full_audit_delta_inventory_bug_no_outputs"
        or failure_internal
        != "1a081525ce93c8d20a8d2676e4fefa5b4e00ec50d846e460a891ce4d9a2ce394"
        or failure_internal != sha256_value(failure_unsigned)
        or failure_report.get("consumption")
        != {
            "authorized_success_output_files_written": 0,
            "candidate_input_rows_parsed": 561,
            "full_candidate_audit_passes_completed": 0,
            "full_candidate_audit_row_reads": 0,
            "source_input_rows_parsed": 608,
        }
        or failure_verification.get("status")
        != "PASS_verified_fail_closed_pre_full_audit_no_outputs"
        or failure_verification.get("failure_report_file_sha256")
        != EXPECTED_INPUT_SHA256["failure_report"]
        or verification_internal
        != "9bae637ca532ca0e61ba6e08c14b53f30169257b51ed7492900ff03335d7f551"
        or verification_internal != sha256_value(verification_unsigned)
    ):
        raise ValueError("expanded-delta runtime-recovery evidence changed")

    failure_v02 = _load_json(bound["failure_report_v02"])
    verification_v02 = _load_json(bound["failure_verification_v02"])
    failure_v02_unsigned = dict(failure_v02)
    failure_v02_internal = failure_v02_unsigned.pop("failure_report_sha256", None)
    verification_v02_unsigned = dict(verification_v02)
    verification_v02_internal = verification_v02_unsigned.pop(
        "verification_sha256", None
    )
    if (
        failure_v02.get("status")
        != "FAIL_CLOSED_post_full_audit_owner_packet_hash_domain_mismatch_no_outputs"
        or failure_v02_internal
        != "3423821031506cef5eee7790cae77d762c72586b899ab56916c6d0f2dd6e6a82"
        or failure_v02_internal != sha256_value(failure_v02_unsigned)
        or failure_v02.get("attempt_consumption")
        != {
            "authorized_success_output_files_written": 0,
            "candidate_input_rows_parsed": 561,
            "full_candidate_audit_passes_completed": 1,
            "full_candidate_audit_row_reads": 314_716,
            "runtime_seconds": 155.32,
            "source_input_rows_parsed": 608,
        }
        or verification_v02.get("status")
        != "PASS_verified_fail_closed_post_full_audit_no_outputs"
        or verification_v02.get("failure_report_file_sha256")
        != EXPECTED_INPUT_SHA256["failure_report_v02"]
        or verification_v02_internal
        != "d238a5b1497cd39f8dba3b0a8f16dce7d28100a19494af24445a040224ed926a"
        or verification_v02_internal != sha256_value(verification_v02_unsigned)
    ):
        raise ValueError("expanded-delta post-audit failure evidence changed")

    v14_plan = v14.load_plan(bound["v14_plan"], root)
    v14_report = _load_json(bound["v14_report"])
    v14_verification = _load_json(bound["v14_verification"])
    v14.validate_report(v14_report)
    if (
        v14_report.get("candidate_pool", {}).get("candidate_records_read") != 314_271
        or v14_report.get("candidate_pool", {}).get("accepted_train_supervised_tokens")
        != 10_403_464
        or v14_report.get("candidate_pool", {}).get(
            "accepted_validation_supervised_tokens"
        )
        != EXPECTED_AUDIT_VALIDATION_TOKENS
        or v14_verification.get("status")
        != "PASS_verified_registry_v0_20_full_candidate_mix_reaudit_nonproduction"
        or v14_verification.get("report_file_sha256")
        != EXPECTED_INPUT_SHA256["v14_report"]
        or v14_verification.get("report_internal_sha256")
        != v14_report.get("report_sha256")
    ):
        raise ValueError("expanded-delta v0.14 evidence changed")
    if file_sha256(bound["v14_plan"]) != EXPECTED_INPUT_SHA256["v14_plan"]:
        raise ValueError("expanded-delta v0.14 plan changed")
    if v14_plan["bindings"]["source_registry"]["sha256"] != EXPECTED_INPUT_SHA256[
        "source_registry"
    ]:
        raise ValueError("expanded-delta v0.14 registry binding changed")

    prior = _load_json(bound["prior_repair_report"])
    prior_verification = _load_json(bound["prior_repair_verification"])
    if (
        prior.get("status") != "repaired_private_nonproduction_candidate_registry_unchanged"
        or prior.get("output", {}).get("candidate_file_sha256")
        != EXPECTED_INPUT_SHA256["included_candidate"]
        or prior.get("output", {}).get("records") != EXPECTED_INCLUDED_ROWS
        or prior.get("selection", {}).get("fit_skipped_train_records")
        != EXPECTED_DELTA_ROWS
        or prior_verification.get("status")
        != "verified_private_nonproduction_banglabridge_hygiene_repair"
        or prior_verification.get("output_candidate_file_sha256")
        != EXPECTED_INPUT_SHA256["included_candidate"]
    ):
        raise ValueError("expanded-delta predecessor repair evidence changed")

    rights = _load_json(bound["rights_preflight"])
    rights_verification = _load_json(bound["rights_preflight_verification"])
    selected = rights.get("selected_sources")
    bridge = [
        row
        for row in selected or []
        if row.get("source_id") == "subhajitmahata84_banglabridge_instructions"
    ]
    if (
        rights.get("status") != "verified_shortlist_ready_payload_and_sft_not_authorized"
        or len(bridge) != 1
        or bridge[0].get("declared_license") != "cc-by-4.0"
        or bridge[0].get("revision")
        != "b5ace9b2e56702cfd3aa5fe2e8c3870f67f57300"
        or rights_verification.get("status") != "PASS"
        or rights_verification.get("report_file_sha256")
        != EXPECTED_INPUT_SHA256["rights_preflight"]
        or not all(rights_verification.get("checks", {}).values())
    ):
        raise ValueError("expanded-delta rights evidence changed")


def _language_mode(row: Mapping[str, Any]) -> str:
    language = row.get("language")
    if language == "bn":
        return "bengali_script"
    if language == "multilingual":
        return "natural_code_switching"
    raise ValueError("expanded-delta language mode changed")


def _owner_sample(delta_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = {
        "bengali_script": [],
        "natural_code_switching": [],
    }
    for row in delta_rows:
        buckets[_language_mode(row)].append(row)
    for mode, rows in buckets.items():
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"banglabridge-expanded-delta-owner-v0.1\0{mode}\0{row['record_id']}".encode()
            ).hexdigest()
        )
        if len(rows) < 16:
            raise ValueError(f"expanded-delta owner-review bucket too small: {mode}")
    selected: list[dict[str, Any]] = []
    for index in range(16):
        for mode in ("bengali_script", "natural_code_switching"):
            row = buckets[mode][index]
            selected.append(
                {
                    "language_mode": mode,
                    "messages": row["messages"],
                    "record_id": row["record_id"],
                    "skill_id": "multilingual_bengali",
                    "split": row["split"],
                }
            )
    return selected


def _owner_html(sample: list[Mapping[str, Any]], sample_sha256: str) -> str:
    page = build_owner_review_html(sample, sample_sha256=sample_sha256)
    replacements = {
        "<title>Lumi owner review</title>": (
            "<title>Lumi BanglaBridge expanded-delta owner review</title>"
        ),
        "<h1>Lumi owner spot-check</h1>": (
            "<h1>Lumi BanglaBridge 445-row recovery spot-check</h1>"
        ),
        (
            '<p class="honesty">These rows are AI-generated and owner-reviewed. '
            "Approving them does not make them human-written.</p>"
        ): (
            '<p class="honesty">These rows are source-derived or deterministically '
            "transformed under the bound BanglaBridge CC-BY-4.0 provenance; row-level "
            "human authorship is not established. Check language, quality, factuality, "
            "safety, and naturalness. Approval does not authorize registry change, "
            "dataset sealing, or SFT.</p>"
        ),
    }
    for old, new in replacements.items():
        if page.count(old) != 1:
            raise ValueError(f"expanded-delta owner HTML marker missing: {old[:48]}")
        page = page.replace(old, new, 1)
    if (
        f"schema_version: '{DECISIONS_SCHEMA_VERSION}'" not in page
        or "link.download = 'lumi-owner-review-decisions-v0.3.json'" not in page
        or f'const sampleSha256 = "{sample_sha256}";' not in page
    ):
        raise ValueError("expanded-delta owner HTML contract changed")
    return page


def _owner_artifacts(
    sample: list[dict[str, Any]],
) -> tuple[bytes, bytes, str, str]:
    packet_bytes = json_bytes(sample)
    content_sha256 = hashlib.sha256(canonical_json_bytes(sample)).hexdigest()
    file_sha256 = hashlib.sha256(packet_bytes).hexdigest()
    if content_sha256 == file_sha256 or json.loads(packet_bytes) != sample:
        raise ValueError("expanded-delta owner packet hash domains collapsed")
    html_bytes = _owner_html(sample, content_sha256).encode("utf-8")
    if DECISIONS_SCHEMA_VERSION.encode() not in html_bytes:
        raise ValueError("expanded-delta generic owner decision schema changed")
    return packet_bytes, html_bytes, content_sha256, file_sha256


def validate_owner_preflight_artifacts(
    *,
    report: Mapping[str, Any],
    owner_packet_bytes: bytes,
    owner_html_bytes: bytes,
    delta_ledger_bytes: bytes,
) -> None:
    """Validate every private owner artifact before the independent audit begins."""

    validate_report(report)
    try:
        owner_packet = json.loads(owner_packet_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("expanded-delta owner packet is invalid JSON") from error
    if not isinstance(owner_packet, list) or len(owner_packet) != 32:
        raise ValueError("expanded-delta owner packet row inventory changed")
    expected_packet_keys = {
        "language_mode",
        "messages",
        "record_id",
        "skill_id",
        "split",
    }
    packet_ids: list[str] = []
    for row in owner_packet:
        if not isinstance(row, Mapping) or set(row) != expected_packet_keys:
            raise ValueError("expanded-delta owner packet row schema changed")
        record_id = row.get("record_id")
        messages = row.get("messages")
        if (
            not isinstance(record_id, str)
            or not record_id
            or row.get("split") != "train"
            or row.get("skill_id") != "multilingual_bengali"
            or row.get("language_mode")
            not in {"bengali_script", "natural_code_switching"}
            or not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(message, Mapping)
                or set(message) != {"content", "role"}
                or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
                for message in messages
            )
        ):
            raise ValueError("expanded-delta owner packet row changed")
        packet_ids.append(record_id)
    if (
        len(set(packet_ids)) != 32
        or Counter(str(row["language_mode"]) for row in owner_packet)
        != {"bengali_script": 16, "natural_code_switching": 16}
    ):
        raise ValueError("expanded-delta owner packet allocation changed")

    try:
        ledger = [
            json.loads(line)
            for line in delta_ledger_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("expanded-delta ledger is invalid JSONL") from error
    expected_ledger_keys = {
        "language_mode",
        "messages_sha256",
        "owner_review_status",
        "record_id",
        "serialized_tokens",
        "split",
        "supervised_tokens",
    }
    ledger_ids: list[str] = []
    for row in ledger:
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_ledger_keys
            or not isinstance(row.get("record_id"), str)
            or not row["record_id"]
            or row.get("split") != "train"
            or row.get("owner_review_status")
            != "pending_deterministic_spot_check"
            or row.get("language_mode")
            not in {"bengali_script", "natural_code_switching"}
            or not isinstance(row.get("messages_sha256"), str)
            or re.fullmatch(SHA256_PATTERN, row["messages_sha256"]) is None
            or not isinstance(row.get("serialized_tokens"), int)
            or row["serialized_tokens"] <= 0
            or not isinstance(row.get("supervised_tokens"), int)
            or row["supervised_tokens"] <= 0
        ):
            raise ValueError("expanded-delta ledger row changed")
        ledger_ids.append(str(row["record_id"]))
    if (
        len(ledger) != EXPECTED_DELTA_ROWS
        or len(set(ledger_ids)) != EXPECTED_DELTA_ROWS
        or not set(packet_ids).issubset(ledger_ids)
        or delta_ledger_bytes != jsonl_bytes(ledger)
    ):
        raise ValueError("expanded-delta owner packet is not delta-only")

    (
        expected_packet_bytes,
        expected_html_bytes,
        packet_content_sha256,
        packet_file_sha256,
    ) = _owner_artifacts(owner_packet)
    owner = report["owner_review"]
    outputs = report["outputs"]
    if (
        owner_packet_bytes != expected_packet_bytes
        or owner_html_bytes != expected_html_bytes
        or packet_content_sha256 != owner["packet_content_sha256"]
        or packet_file_sha256 != owner["packet_file_sha256"]
        or len(owner_packet_bytes) != owner["packet_bytes"]
        or hashlib.sha256(delta_ledger_bytes).hexdigest()
        != outputs["delta_ledger"]["sha256"]
        or len(delta_ledger_bytes) != outputs["delta_ledger"]["bytes"]
        or hashlib.sha256(owner_packet_bytes).hexdigest()
        != outputs["owner_review_packet"]["sha256"]
        or len(owner_packet_bytes) != outputs["owner_review_packet"]["bytes"]
        or hashlib.sha256(owner_html_bytes).hexdigest()
        != outputs["owner_review_html"]["sha256"]
        or len(owner_html_bytes) != outputs["owner_review_html"]["bytes"]
    ):
        raise ValueError("expanded-delta owner artifact preflight changed")


def _materialize_successor(
    *, bound: Mapping[str, Path], root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    original_rows = prior_repair._read_jsonl(bound["original_candidate"])
    source_rows = prior_repair._read_jsonl(bound["source_rows"])
    if (
        len(original_rows) != EXPECTED_ORIGINAL_ROWS
        or len(source_rows) != EXPECTED_SOURCE_ROWS
    ):
        raise ValueError("expanded-delta input row inventory changed")

    original_by_id: dict[str, dict[str, Any]] = {}
    for row in original_rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id in original_by_id:
            raise ValueError("expanded-delta original record IDs changed")
        original_by_id[record_id] = row
    repaired = [prior_repair._repair_row(row, source_rows) for row in original_rows]
    repaired.sort(key=lambda row: (str(row["split"]), str(row["record_id"])))
    repaired_by_id = {str(row["record_id"]): row for row in repaired}

    for row in repaired:
        original = original_by_id[str(row["record_id"])]
        stripped = json.loads(json.dumps(row, ensure_ascii=False))
        provenance = stripped.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("expanded-delta repaired provenance missing")
        for field in EXPECTED_PROVENANCE_FIELDS:
            value = provenance.pop(field, None)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError("expanded-delta provenance hash changed")
        if stripped != original:
            raise ValueError("expanded-delta changed a field beyond four provenance hashes")
        if row.get("provenance", {}).get("owner_review_status") != (
            "pending_deterministic_spot_check"
        ):
            raise ValueError("expanded-delta owner-review status changed")

    sample = json.loads(bound["original_owner_sample"].read_text(encoding="utf-8"))
    receipt = _load_json(bound["original_owner_receipt"])
    owner_verification = _load_json(bound["original_owner_verification"])
    reviewed_ids, owner_review = prior_repair._owner_reviewed_rows(
        original_by_id, sample, receipt, owner_verification
    )

    prior_plan = prior_repair.load_repair_plan(
        bound["prior_repair_plan"], repository_root=root
    )
    prior_metrics, prior_machine_gate = prior_repair._audit_and_measure(
        repaired,
        repository_root=root,
        bindings=prior_plan["bindings"],
    )
    if (
        prior_machine_gate["machine_rejected_records"] != 0
        or prior_machine_gate["benchmark_matches"] != 0
    ):
        raise ValueError("expanded-delta predecessor machine gate changed")
    reviewed_train = sorted(
        (
            repaired_by_id[record_id]
            for record_id in reviewed_ids
            if repaired_by_id[record_id]["split"] == "train"
        ),
        key=lambda row: str(row["record_id"]),
    )
    prior_selected_train = list(reviewed_train)
    included_ids = {str(row["record_id"]) for row in prior_selected_train}
    prior_train_tokens = sum(
        prior_metrics[str(row["record_id"])]["supervised_tokens"]
        for row in prior_selected_train
    )
    prior_ceiling = int(prior_plan["limits"]["maximum_train_supervised_tokens"])
    for row in sorted(
        (
            row
            for row in repaired
            if row["split"] == "train" and str(row["record_id"]) not in included_ids
        ),
        key=lambda row: prior_repair._priority(str(row["record_id"])),
    ):
        tokens = prior_metrics[str(row["record_id"])]["supervised_tokens"]
        if prior_train_tokens + tokens > prior_ceiling:
            continue
        prior_selected_train.append(row)
        included_ids.add(str(row["record_id"]))
        prior_train_tokens += tokens
    prior_validation = sorted(
        (row for row in repaired if row["split"] == "validation"),
        key=lambda row: str(row["record_id"]),
    )
    included_ids.update(str(row["record_id"]) for row in prior_validation)
    prior_selected = sorted(
        prior_selected_train + prior_validation,
        key=lambda row: (str(row["split"]), str(row["record_id"])),
    )
    prior_payload = jsonl_bytes(prior_selected)
    if (
        len(prior_selected) != EXPECTED_INCLUDED_ROWS
        or len(prior_payload) != 413_036
        or hashlib.sha256(prior_payload).hexdigest()
        != EXPECTED_INPUT_SHA256["included_candidate"]
    ):
        raise ValueError("expanded-delta 116-row predecessor subset changed")

    delta_rows = [row for row in repaired if str(row["record_id"]) not in included_ids]
    if len(delta_rows) != EXPECTED_DELTA_ROWS or any(
        row.get("split") != "train"
        or row.get("task_category") != "multilingual_bengali"
        for row in delta_rows
    ):
        raise ValueError("expanded-delta recovered row inventory changed")
    if len(reviewed_ids) != 8 or not reviewed_ids.issubset(included_ids):
        raise ValueError("expanded-delta historical owner-review rows changed")
    if reviewed_ids & {str(row["record_id"]) for row in delta_rows}:
        raise ValueError("expanded-delta incorrectly reused historical owner review")

    candidate_payload = jsonl_bytes(repaired)
    delta_payload = jsonl_bytes(delta_rows)
    if (
        len(candidate_payload) != EXPECTED_SUCCESSOR_BYTES
        or hashlib.sha256(candidate_payload).hexdigest() != EXPECTED_SUCCESSOR_SHA256
        or len(delta_payload) != EXPECTED_DELTA_BYTES
        or hashlib.sha256(delta_payload).hexdigest() != EXPECTED_DELTA_SHA256
    ):
        raise ValueError("expanded-delta prospective byte binding changed")
    return repaired, delta_rows, owner_review


def _measure_successor(
    *,
    rows: Sequence[Mapping[str, Any]],
    delta_ids: frozenset[str],
    tokenizer: Any,
    template: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_tokens = 0
    validation_tokens = 0
    train_rows = 0
    validation_rows = 0
    delta_tokens = 0
    delta_language: Counter[str] = Counter()
    maximum = 0
    ledger: list[dict[str, Any]] = []
    for row in rows:
        encoded = encode_sft_messages(row["messages"], tokenizer=tokenizer, template=template)
        serialized = len(encoded.input_ids)
        supervised = encoded.supervised_tokens
        maximum = max(maximum, serialized)
        if row["split"] == "train":
            train_rows += 1
            train_tokens += supervised
        else:
            validation_rows += 1
            validation_tokens += supervised
        record_id = str(row["record_id"])
        if record_id in delta_ids:
            delta_tokens += supervised
            delta_language[_language_mode(row)] += supervised
            ledger.append(
                {
                    "language_mode": _language_mode(row),
                    "messages_sha256": sha256_value(row["messages"]),
                    "owner_review_status": "pending_deterministic_spot_check",
                    "record_id": record_id,
                    "serialized_tokens": serialized,
                    "split": row["split"],
                    "supervised_tokens": supervised,
                }
            )
    observed = {
        "delta_language_train_supervised_tokens": dict(sorted(delta_language.items())),
        "delta_train_supervised_tokens": delta_tokens,
        "maximum_serialized_tokens": maximum,
        "train_records": train_rows,
        "train_supervised_tokens": train_tokens,
        "validation_records": validation_rows,
        "validation_supervised_tokens": validation_tokens,
    }
    expected = {
        "delta_language_train_supervised_tokens": EXPECTED_DELTA_LANGUAGE_TOKENS,
        "delta_train_supervised_tokens": EXPECTED_DELTA_TRAIN_TOKENS,
        "maximum_serialized_tokens": EXPECTED_MAXIMUM_SERIALIZED_TOKENS,
        "train_records": EXPECTED_SUCCESSOR_TRAIN_ROWS,
        "train_supervised_tokens": EXPECTED_SUCCESSOR_TRAIN_TOKENS,
        "validation_records": EXPECTED_SUCCESSOR_VALIDATION_ROWS,
        "validation_supervised_tokens": EXPECTED_SUCCESSOR_VALIDATION_TOKENS,
    }
    if observed != expected or len(ledger) != EXPECTED_DELTA_ROWS:
        raise ValueError("expanded-delta token ledger changed")
    return ledger, observed


def _audit_successor(
    *,
    plan: Mapping[str, Any],
    bound: Mapping[str, Path],
    root: Path,
    successor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    v14_plan = v14.load_plan(bound["v14_plan"], root)
    v14_bound = {
        name: v14._resolve_binding(root, binding, name)
        for name, binding in v14_plan["bindings"].items()
    }
    v14._validate_predecessor_and_promotion(v14_bound)
    source_plans, candidate_paths, effective, old_resolved = v14._source_inventory(
        root=root, bound=v14_bound
    )
    if (
        SOURCE_ID not in source_plans
        or file_sha256(candidate_paths[SOURCE_ID])
        != EXPECTED_INPUT_SHA256["included_candidate"]
        or source_plans[SOURCE_ID].get("family") != SOURCE_FAMILY
        or source_plans[SOURCE_ID].get("language_policy")
        != "banglabridge_language_field"
    ):
        raise ValueError("expanded-delta v0.14 BanglaBridge source changed")

    base_policy = load_sft_policy(bound["data_contract"])
    audit_policy = replace(
        base_policy,
        benchmark_decontamination={
            "anchor_ngram_tokens": 8,
            "minimum_anchor_matches": 2,
            "registry": "data/benchmark-signatures-v0.3.jsonl.gz",
            "registry_sha256": EXPECTED_INPUT_SHA256["benchmark_registry"],
        },
    )
    benchmark = v2.BenchmarkRegistry(audit_policy.benchmark_decontamination, root)
    registry = load_sft_source_registry(
        bound["source_registry"],
        allowed_licenses=audit_policy.allowed_licenses,
        allowed_languages=audit_policy.allowed_languages,
        allowed_splits=audit_policy.allowed_splits,
        allowed_tasks=frozenset(audit_policy.task_mix),
        expected_sha256=EXPECTED_INPUT_SHA256["source_registry"],
    )
    if len(registry.by_id) != 24:
        raise ValueError("expanded-delta registry inventory changed")
    excluded_source_ids = frozenset(registry.by_id) - frozenset(source_plans)
    if excluded_source_ids != v14.EXPECTED_EXCLUDED_SOURCE_IDS:
        raise ValueError("expanded-delta excluded-source inventory changed")
    exclusions = v2._verify_exclusions(
        plan=effective,
        resolved=old_resolved,
        registry=registry,
    )
    template = load_chat_template(bound["chat_template"])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = PureByteLevelBPETokenizer(bound["tokenizer_json"])
    verify_chat_tokenizer(tokenizer, template)

    all_rows: list[dict[str, Any]] = []
    source_metrics: dict[str, dict[str, Any]] = {}
    for source_id in sorted(source_plans):
        source_plan = source_plans[source_id]
        registry_source = registry.by_id.get(source_id)
        if (
            registry_source is None
            or registry_source.revision != source_plan["revision"]
            or registry_source.status not in {"canary_eligible", "candidate_eligible"}
        ):
            raise ValueError(f"expanded-delta registry source mismatch: {source_id}")
        if source_id == SOURCE_ID:
            rows = successor_rows
            candidate_sha256 = EXPECTED_SUCCESSOR_SHA256
        else:
            rows = _load_jsonl(candidate_paths[source_id])
            candidate_sha256 = file_sha256(candidate_paths[source_id])
        if source_id == v14.BILLSUM_SOURCE["source_id"]:
            v14._validate_billsum_rows(rows)
        metrics = v14.v13.v18._source_metrics(
            rows=rows,
            source_id=source_id,
            family=str(source_plan["family"]),
            language_policy=str(source_plan["language_policy"]),
            candidate_sha256=candidate_sha256,
            tokenizer=tokenizer,
            template=template,
        )
        source_metrics[source_id] = metrics
        all_rows.extend(rows)

    input_train, input_validation, *_ = v14.v13.v18._aggregate_metrics(source_metrics)
    if (
        len(all_rows) != EXPECTED_AUDIT_RECORDS
        or input_train != EXPECTED_AUDIT_TRAIN_TOKENS
        or input_validation != EXPECTED_AUDIT_VALIDATION_TOKENS
    ):
        raise ValueError("expanded-delta full audit input totals changed")

    accepted, generic = filter_sft_records(
        all_rows,
        policy=audit_policy,
        benchmark_registry=benchmark,
        source_registry=registry,
    )
    if generic["rejected"] or len(accepted) != EXPECTED_AUDIT_RECORDS:
        raise ValueError(
            f"expanded-delta generic gate rejected rows: {generic['rejection_counts']}"
        )
    accepted_metrics = v14.v13.v18._accepted_metrics(
        rows=accepted,
        source_plans=source_plans,
        tokenizer=tokenizer,
        template=template,
    )
    duplicate_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in accepted:
        duplicate_index[_canonical_message_digest(row["messages"])].append(
            (row["source"]["source_id"], row["split"])
        )
    exact = _duplicate_snapshot(duplicate_index)
    if exact["duplicate_groups"] or exact["train_validation_overlap_groups"]:
        raise ValueError("expanded-delta exact duplicate or split overlap found")
    near = v2._near_duplicate_audit(accepted, policy=plan["near_duplicate_policy"])
    if near["multi_record_components"] or near["train_validation_components"]:
        raise ValueError("expanded-delta cross-source near duplicate or split overlap found")

    targets = _targets(_load_json(old_resolved["missing_mix_proposal"]))
    train_tokens = int(accepted_metrics["train_supervised_tokens"])
    task_behavior = Counter(accepted_metrics["task_train_supervised_tokens_behavior_safe"])
    language_behavior = Counter(
        accepted_metrics["language_train_supervised_tokens_behavior_safe"]
    )
    task_gaps = {
        task: max(0, int(targets["task_minimums"][task]) - task_behavior[task])
        for task in EXPECTED_TASKS
    }
    language_gaps = {
        mode: max(
            0, int(targets["language_minimums"][mode]) - language_behavior[mode]
        )
        for mode in EXPECTED_LANGUAGE_MODES
    }
    if task_gaps != EXPECTED_TASK_GAPS or language_gaps != EXPECTED_LANGUAGE_GAPS:
        raise ValueError("expanded-delta residual gap projection changed")
    family_tokens = accepted_metrics["family_train_supervised_tokens"]
    family_shares = {
        family: round(int(tokens) / train_tokens, 8)
        for family, tokens in sorted(family_tokens.items())
    }
    family_cap_violations = [
        family
        for family, share in family_shares.items()
        if share > float(targets["source_cap"])
    ]
    if family_cap_violations:
        raise ValueError("expanded-delta source-family cap violation")
    family_dilution_gap = max(
        (
            max(0, math.ceil(int(tokens) / float(targets["source_cap"]) - train_tokens))
            for tokens in family_tokens.values()
        ),
        default=0,
    )
    strict_gap = max(
        max(0, int(targets["total"]["minimum"]) - train_tokens),
        sum(task_gaps.values()),
        sum(language_gaps.values()),
        family_dilution_gap,
    )
    if strict_gap != 1_137_534:
        raise ValueError("expanded-delta strict residual gap changed")
    generic_summary = {key: value for key, value in generic.items() if key != "decisions"}
    return {
        "accepted_records": len(accepted),
        "accepted_train_supervised_tokens": train_tokens,
        "accepted_validation_supervised_tokens": accepted_metrics[
            "validation_supervised_tokens"
        ],
        "candidate_audit_row_reads": len(all_rows),
        "exact_duplicate_audit": exact,
        "excluded_source_gate": {
            "candidate_rows_from_excluded_sources": 0,
            "predecessor_evidence_replayed": exclusions,
            "registry_sources_not_in_candidate_inventory": sorted(excluded_source_ids),
        },
        "family_cap_dilution_gap": family_dilution_gap,
        "family_cap_violations": family_cap_violations,
        "family_shares": family_shares,
        "generic_sft_validation": generic_summary,
        "language_floor_gaps_behavior_safe": language_gaps,
        "near_duplicate_audit": near,
        "source_metrics": source_metrics,
        "strict_targeted_gap_lower_bound": strict_gap,
        "task_floor_gaps_behavior_safe": task_gaps,
        "token_metrics": accepted_metrics,
    }


def build_recovery(
    *, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path
) -> tuple[dict[str, bytes], dict[str, Any]]:
    started_at = time.monotonic()
    root = Path(repository_root).resolve()
    bound = _bound_paths(plan, root)
    _validate_evidence(bound, root)
    successor_rows, delta_rows, historical_owner_review = _materialize_successor(
        bound=bound, root=root
    )
    _check_runtime(started_at)
    delta_ids = frozenset(str(row["record_id"]) for row in delta_rows)
    owner_packet = _owner_sample(delta_rows)
    if (
        len(owner_packet) != 32
        or {str(row["record_id"]) for row in owner_packet}.issubset(delta_ids) is False
        or Counter(str(row["language_mode"]) for row in owner_packet)
        != {"bengali_script": 16, "natural_code_switching": 16}
        or any(
            row.get("split") != "train"
            or row.get("skill_id") != "multilingual_bengali"
            for row in owner_packet
        )
    ):
        raise ValueError("expanded-delta owner packet changed")
    (
        owner_packet_bytes,
        owner_html_bytes,
        packet_content_sha256,
        packet_file_sha256,
    ) = _owner_artifacts(owner_packet)
    bridge_custom_rejections: Counter[str] = Counter()
    for row in successor_rows:
        bridge_custom_rejections.update(prior_repair._custom_quality_reasons(row))
    if bridge_custom_rejections:
        raise ValueError(
            f"expanded-delta BanglaBridge custom gate rejected rows: "
            f"{dict(bridge_custom_rejections)}"
        )
    template = load_chat_template(bound["chat_template"])
    tokenizer = PureByteLevelBPETokenizer(bound["tokenizer_json"])
    verify_chat_tokenizer(tokenizer, template)
    ledger, materialization = _measure_successor(
        rows=successor_rows,
        delta_ids=delta_ids,
        tokenizer=tokenizer,
        template=template,
    )
    _check_runtime(started_at)
    audit = _audit_successor(
        plan=plan,
        bound=bound,
        root=root,
        successor_rows=successor_rows,
    )
    _check_runtime(started_at)
    artifacts = {
        "candidate": jsonl_bytes(successor_rows),
        "delta_ledger": jsonl_bytes(ledger),
        "owner_review_html": owner_html_bytes,
        "owner_review_packet": owner_packet_bytes,
    }
    artifact_bindings = {
        name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in sorted(artifacts.items())
    }
    report: dict[str, Any] = {
        "audit": audit,
        "authorization": dict(plan["authorization"]),
        "bindings": {
            name: value["sha256"] for name, value in sorted(plan["bindings"].items())
        },
        "checks": {
            "all_561_rows_have_four_bound_provenance_hashes": True,
            "all_445_delta_rows_owner_review_pending": True,
            "candidate_and_delta_prospective_hashes_match": True,
            "cross_source_minhash_0_9_passed": True,
            "banglabridge_language_punctuation_and_safety_gates_passed": True,
            "blanket_row_level_human_authorship_claimed": False,
            "no_exact_duplicate_or_split_overlap": True,
            "no_network_gpu_inference_registry_seal_sft_or_production_action": True,
            "predecessors_and_source_preserved": True,
            "schema_rights_provenance_secret_pii_benchmark_and_4k_gates_passed": True,
        },
        "delta": {
            "bytes": EXPECTED_DELTA_BYTES,
            "language_train_supervised_tokens": EXPECTED_DELTA_LANGUAGE_TOKENS,
            "records": EXPECTED_DELTA_ROWS,
            "sha256": EXPECTED_DELTA_SHA256,
            "task": "multilingual_bengali",
            "train_supervised_tokens": EXPECTED_DELTA_TRAIN_TOKENS,
        },
        "evaluation_prohibition": v14.EXPECTED_EVALUATION_PROHIBITION,
        "materialization": materialization,
        "observed_at": plan["observed_at"],
        "outputs": artifact_bindings,
        "owner_review": {
            "all_recovered_rows_status": "pending_deterministic_spot_check",
            "historical_review_records_preserved": historical_owner_review[
                "bridge_sample_records"
            ],
            "historical_review_reused_for_delta": False,
            "minimum_accepts_required_for_later_promotion": 29,
            "packet_bytes": len(owner_packet_bytes),
            "packet_content_sha256": packet_content_sha256,
            "packet_file_sha256": packet_file_sha256,
            "packet_records": 32,
            "status": "pending_human_owner_decision",
        },
        "plan_file_sha256": plan_file_sha256,
        "production_eligible": False,
        "runtime_recovery": dict(plan["runtime_recovery"]),
        "raw_text_boundary": {
            "candidate_contains_raw_training_text": True,
            "delta_ledger_contains_raw_training_text": False,
            "owner_review_html_contains_raw_training_text": True,
            "owner_review_packet_contains_raw_training_text": True,
            "report_contains_raw_training_text": False,
            "tests_or_logs_may_contain_raw_training_text": False,
            "verification_contains_raw_training_text": False,
        },
        "resource_accounting": {
            "candidate_audit_passes_completed": 1,
            "candidate_audit_row_reads": EXPECTED_AUDIT_RECORDS,
            "candidate_reconstruction_row_reads": EXPECTED_ORIGINAL_ROWS,
            "candidate_total_row_reads": EXPECTED_AUDIT_RECORDS
            + EXPECTED_ORIGINAL_ROWS,
            "source_row_reads": EXPECTED_SOURCE_ROWS,
        },
        "report_sha256": "",
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": STATUS,
        "successor": {
            "bytes": EXPECTED_SUCCESSOR_BYTES,
            "records": EXPECTED_ORIGINAL_ROWS,
            "sha256": EXPECTED_SUCCESSOR_SHA256,
            "train_records": EXPECTED_SUCCESSOR_TRAIN_ROWS,
            "train_supervised_tokens": EXPECTED_SUCCESSOR_TRAIN_TOKENS,
            "validation_records": EXPECTED_SUCCESSOR_VALIDATION_ROWS,
            "validation_supervised_tokens": EXPECTED_SUCCESSOR_VALIDATION_TOKENS,
        },
    }
    report.pop("report_sha256")
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return artifacts, report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported expanded-delta report schema")
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("expanded-delta report internal hash mismatch")
    audit = report.get("audit", {})
    successor = report.get("successor", {})
    delta = report.get("delta", {})
    materialization = report.get("materialization", {})
    owner = report.get("owner_review", {})
    resources = report.get("resource_accounting", {})
    outputs = report.get("outputs", {})
    exact = audit.get("exact_duplicate_audit", {})
    near = audit.get("near_duplicate_audit", {})
    generic = audit.get("generic_sft_validation", {})
    excluded = audit.get("excluded_source_gate", {})
    source_metrics = audit.get("source_metrics", {})
    token_metrics = audit.get("token_metrics", {})
    expected_top_level = {
        "audit",
        "authorization",
        "bindings",
        "checks",
        "delta",
        "evaluation_prohibition",
        "materialization",
        "observed_at",
        "outputs",
        "owner_review",
        "plan_file_sha256",
        "production_eligible",
        "raw_text_boundary",
        "report_sha256",
        "resource_accounting",
        "runtime_recovery",
        "schema_version",
        "status",
        "successor",
    }
    expected_checks = {
        "all_561_rows_have_four_bound_provenance_hashes": True,
        "all_445_delta_rows_owner_review_pending": True,
        "banglabridge_language_punctuation_and_safety_gates_passed": True,
        "blanket_row_level_human_authorship_claimed": False,
        "candidate_and_delta_prospective_hashes_match": True,
        "cross_source_minhash_0_9_passed": True,
        "no_exact_duplicate_or_split_overlap": True,
        "no_network_gpu_inference_registry_seal_sft_or_production_action": True,
        "predecessors_and_source_preserved": True,
        "schema_rights_provenance_secret_pii_benchmark_and_4k_gates_passed": True,
    }
    output_shapes_valid = (
        isinstance(outputs, Mapping)
        and set(outputs) == EXPECTED_OUTPUT_NAMES
        and all(
            isinstance(value, Mapping)
            and set(value) == {"bytes", "sha256"}
            and isinstance(value.get("bytes"), int)
            and value["bytes"] > 0
            and isinstance(value.get("sha256"), str)
            and re.fullmatch(SHA256_PATTERN, value["sha256"]) is not None
            for value in outputs.values()
        )
        and sum(int(value["bytes"]) for value in outputs.values())
        <= EXPECTED_LIMITS["maximum_private_write_bytes"]
    )
    expected_successor = {
        "bytes": EXPECTED_SUCCESSOR_BYTES,
        "records": EXPECTED_ORIGINAL_ROWS,
        "sha256": EXPECTED_SUCCESSOR_SHA256,
        "train_records": EXPECTED_SUCCESSOR_TRAIN_ROWS,
        "train_supervised_tokens": EXPECTED_SUCCESSOR_TRAIN_TOKENS,
        "validation_records": EXPECTED_SUCCESSOR_VALIDATION_ROWS,
        "validation_supervised_tokens": EXPECTED_SUCCESSOR_VALIDATION_TOKENS,
    }
    expected_delta = {
        "bytes": EXPECTED_DELTA_BYTES,
        "language_train_supervised_tokens": EXPECTED_DELTA_LANGUAGE_TOKENS,
        "records": EXPECTED_DELTA_ROWS,
        "sha256": EXPECTED_DELTA_SHA256,
        "task": "multilingual_bengali",
        "train_supervised_tokens": EXPECTED_DELTA_TRAIN_TOKENS,
    }
    expected_materialization = {
        "delta_language_train_supervised_tokens": EXPECTED_DELTA_LANGUAGE_TOKENS,
        "delta_train_supervised_tokens": EXPECTED_DELTA_TRAIN_TOKENS,
        "maximum_serialized_tokens": EXPECTED_MAXIMUM_SERIALIZED_TOKENS,
        "train_records": EXPECTED_SUCCESSOR_TRAIN_ROWS,
        "train_supervised_tokens": EXPECTED_SUCCESSOR_TRAIN_TOKENS,
        "validation_records": EXPECTED_SUCCESSOR_VALIDATION_ROWS,
        "validation_supervised_tokens": EXPECTED_SUCCESSOR_VALIDATION_TOKENS,
    }
    expected_exact = {
        "cross_source_duplicate_groups": 0,
        "duplicate_extra_records": 0,
        "duplicate_groups": 0,
        "train_validation_overlap_groups": 0,
    }
    expected_near = {
        "comparison_unit": EXPECTED_NEAR_DUPLICATE_POLICY["comparison_unit"],
        "component_ledger_sha256": sha256_value([]),
        "components": EXPECTED_AUDIT_RECORDS,
        "cross_source_components": 0,
        "duplicate_extra_records": 0,
        "edge_count": 0,
        "largest_component_records": 1,
        "multi_record_components": 0,
        "per_source_multi_record_components": {},
        "records": EXPECTED_AUDIT_RECORDS,
        "size_histogram": {},
        "train_validation_components": 0,
        "train_validation_records": 0,
    }
    expected_resources = {
        "candidate_audit_passes_completed": 1,
        "candidate_audit_row_reads": EXPECTED_AUDIT_RECORDS,
        "candidate_reconstruction_row_reads": EXPECTED_ORIGINAL_ROWS,
        "candidate_total_row_reads": EXPECTED_AUDIT_RECORDS
        + EXPECTED_ORIGINAL_ROWS,
        "source_row_reads": EXPECTED_SOURCE_ROWS,
    }
    expected_audit_keys = {
        "accepted_records",
        "accepted_train_supervised_tokens",
        "accepted_validation_supervised_tokens",
        "candidate_audit_row_reads",
        "exact_duplicate_audit",
        "excluded_source_gate",
        "family_cap_dilution_gap",
        "family_cap_violations",
        "family_shares",
        "generic_sft_validation",
        "language_floor_gaps_behavior_safe",
        "near_duplicate_audit",
        "source_metrics",
        "strict_targeted_gap_lower_bound",
        "task_floor_gaps_behavior_safe",
        "token_metrics",
    }
    if (
        set(report) != expected_top_level
        or report.get("status") != STATUS
        or report.get("observed_at") != "2026-08-13"
        or report.get("production_eligible") is not False
        or report.get("authorization") != EXPECTED_AUTHORIZATION
        or report.get("checks") != expected_checks
        or report.get("runtime_recovery") != EXPECTED_RUNTIME_RECOVERY
        or not isinstance(report.get("bindings"), Mapping)
        or set(report["bindings"]) != EXPECTED_BINDINGS
        or any(
            not isinstance(value, str)
            or re.fullmatch(SHA256_PATTERN, value) is None
            for value in report["bindings"].values()
        )
        or not isinstance(report.get("plan_file_sha256"), str)
        or re.fullmatch(SHA256_PATTERN, report["plan_file_sha256"]) is None
        or not output_shapes_valid
        or report.get("raw_text_boundary")
        != {
            "candidate_contains_raw_training_text": True,
            "delta_ledger_contains_raw_training_text": False,
            "owner_review_html_contains_raw_training_text": True,
            "owner_review_packet_contains_raw_training_text": True,
            "report_contains_raw_training_text": False,
            "tests_or_logs_may_contain_raw_training_text": False,
            "verification_contains_raw_training_text": False,
        }
        or report.get("evaluation_prohibition") != v14.EXPECTED_EVALUATION_PROHIBITION
        or successor != expected_successor
        or delta != expected_delta
        or materialization != expected_materialization
        or outputs.get("candidate", {}).get("sha256") != EXPECTED_SUCCESSOR_SHA256
        or outputs.get("candidate", {}).get("bytes") != EXPECTED_SUCCESSOR_BYTES
        or outputs.get("owner_review_packet", {}).get("sha256")
        != owner.get("packet_file_sha256")
        or outputs.get("owner_review_packet", {}).get("bytes")
        != owner.get("packet_bytes")
        or not isinstance(owner.get("packet_content_sha256"), str)
        or re.fullmatch(SHA256_PATTERN, owner["packet_content_sha256"]) is None
        or owner.get("packet_content_sha256") == owner.get("packet_file_sha256")
        or not isinstance(audit, Mapping)
        or set(audit) != expected_audit_keys
        or audit.get("accepted_records") != EXPECTED_AUDIT_RECORDS
        or audit.get("accepted_train_supervised_tokens")
        != EXPECTED_AUDIT_TRAIN_TOKENS
        or audit.get("accepted_validation_supervised_tokens")
        != EXPECTED_AUDIT_VALIDATION_TOKENS
        or audit.get("candidate_audit_row_reads") != EXPECTED_AUDIT_RECORDS
        or not isinstance(generic, Mapping)
        or set(generic)
        != {
            "accepted",
            "accepted_dataset_sha256",
            "raw_text_retained",
            "rejected",
            "rejection_counts",
            "report_sha256",
            "schema_version",
            "total",
        }
        or generic.get("accepted") != EXPECTED_AUDIT_RECORDS
        or generic.get("total") != EXPECTED_AUDIT_RECORDS
        or generic.get("rejected") != 0
        or generic.get("rejection_counts") != {}
        or generic.get("raw_text_retained") is not False
        or any(
            not isinstance(generic.get(name), str)
            or re.fullmatch(SHA256_PATTERN, generic[name]) is None
            for name in ("accepted_dataset_sha256", "report_sha256")
        )
        or exact != expected_exact
        or near != expected_near
        or audit.get("family_cap_violations") != []
        or audit.get("family_cap_dilution_gap") != 0
        or not isinstance(audit.get("family_shares"), Mapping)
        or audit["family_shares"].get(SOURCE_FAMILY)
        != round(EXPECTED_SUCCESSOR_TRAIN_TOKENS / EXPECTED_AUDIT_TRAIN_TOKENS, 8)
        or any(float(share) > 0.35 for share in audit["family_shares"].values())
        or not isinstance(excluded, Mapping)
        or set(excluded)
        != {
            "candidate_rows_from_excluded_sources",
            "predecessor_evidence_replayed",
            "registry_sources_not_in_candidate_inventory",
        }
        or excluded.get("candidate_rows_from_excluded_sources") != 0
        or excluded.get("registry_sources_not_in_candidate_inventory")
        != sorted(v14.EXPECTED_EXCLUDED_SOURCE_IDS)
        or not isinstance(excluded.get("predecessor_evidence_replayed"), list)
        or not isinstance(source_metrics, Mapping)
        or len(source_metrics) != 17
        or SOURCE_ID not in source_metrics
        or source_metrics[SOURCE_ID].get("candidate_file_sha256")
        != EXPECTED_SUCCESSOR_SHA256
        or not isinstance(token_metrics, Mapping)
        or token_metrics.get("train_supervised_tokens")
        != EXPECTED_AUDIT_TRAIN_TOKENS
        or token_metrics.get("validation_supervised_tokens")
        != EXPECTED_AUDIT_VALIDATION_TOKENS
        or audit.get("strict_targeted_gap_lower_bound") != 1_137_534
        or audit.get("language_floor_gaps_behavior_safe") != EXPECTED_LANGUAGE_GAPS
        or audit.get("task_floor_gaps_behavior_safe") != EXPECTED_TASK_GAPS
        or owner.get("packet_records") != 32
        or set(owner)
        != {
            "all_recovered_rows_status",
            "historical_review_records_preserved",
            "historical_review_reused_for_delta",
            "minimum_accepts_required_for_later_promotion",
            "packet_bytes",
            "packet_content_sha256",
            "packet_file_sha256",
            "packet_records",
            "status",
        }
        or owner.get("all_recovered_rows_status")
        != "pending_deterministic_spot_check"
        or owner.get("historical_review_records_preserved") != 8
        or owner.get("historical_review_reused_for_delta") is not False
        or owner.get("minimum_accepts_required_for_later_promotion") != 29
        or owner.get("status") != "pending_human_owner_decision"
        or resources != expected_resources
    ):
        raise ValueError("expanded-delta report boundary or audit result changed")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"', '"user_prompt"'):
        if forbidden in serialized:
            raise ValueError("expanded-delta report retained raw training text")


def build_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
    artifact_file_sha256: Mapping[str, str],
) -> dict[str, Any]:
    validate_report(report)
    expected = {
        name: value["sha256"] for name, value in report["outputs"].items()
    }
    if dict(artifact_file_sha256) != expected:
        raise ValueError("expanded-delta materialized artifact hash changed")
    verification: dict[str, Any] = {
        "artifact_file_sha256": dict(sorted(artifact_file_sha256.items())),
        "checks": EXPECTED_VERIFICATION_CHECKS,
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "raw_training_text_in_verification": False,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "owner_packet_content_sha256": report["owner_review"][
            "packet_content_sha256"
        ],
        "owner_packet_file_sha256": report["owner_review"]["packet_file_sha256"],
        "owner_review_html_file_sha256": report["outputs"]["owner_review_html"][
            "sha256"
        ],
        "resource_accounting": {
            "candidate_audit_passes_completed": 2,
            "candidate_audit_row_reads": 2 * EXPECTED_AUDIT_RECORDS,
            "candidate_reconstruction_row_reads": 2 * EXPECTED_ORIGINAL_ROWS,
            "candidate_total_row_reads": 2
            * (EXPECTED_AUDIT_RECORDS + EXPECTED_ORIGINAL_ROWS),
            "source_row_reads": 2 * EXPECTED_SOURCE_ROWS,
        },
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
    }
    verification["verification_sha256"] = sha256_value(verification)
    validate_verification(verification)
    return verification


def validate_verification(verification: Mapping[str, Any]) -> None:
    unsigned = dict(verification)
    internal = unsigned.pop("verification_sha256", None)
    artifact_hashes = verification.get("artifact_file_sha256", {})
    resources = verification.get("resource_accounting", {})
    expected_keys = {
        "artifact_file_sha256",
        "checks",
        "observed_at",
        "owner_packet_content_sha256",
        "owner_packet_file_sha256",
        "owner_review_html_file_sha256",
        "production_eligible",
        "raw_training_text_in_verification",
        "report_file_sha256",
        "report_internal_sha256",
        "resource_accounting",
        "schema_version",
        "status",
        "verification_sha256",
    }
    if (
        set(verification) != expected_keys
        or internal != sha256_value(unsigned)
        or verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION
        or verification.get("observed_at") != "2026-08-13"
        or verification.get("status") != VERIFICATION_STATUS
        or verification.get("checks") != EXPECTED_VERIFICATION_CHECKS
        or verification.get("production_eligible") is not False
        or verification.get("raw_training_text_in_verification") is not False
        or not isinstance(artifact_hashes, Mapping)
        or set(artifact_hashes) != EXPECTED_OUTPUT_NAMES
        or any(
            not isinstance(value, str)
            or re.fullmatch(SHA256_PATTERN, value) is None
            for value in artifact_hashes.values()
        )
        or verification.get("owner_packet_file_sha256")
        != artifact_hashes.get("owner_review_packet")
        or verification.get("owner_review_html_file_sha256")
        != artifact_hashes.get("owner_review_html")
        or verification.get("owner_packet_content_sha256")
        == verification.get("owner_packet_file_sha256")
        or any(
            not isinstance(verification.get(name), str)
            or re.fullmatch(SHA256_PATTERN, verification[name]) is None
            for name in (
                "owner_packet_content_sha256",
                "owner_packet_file_sha256",
                "owner_review_html_file_sha256",
                "report_file_sha256",
                "report_internal_sha256",
            )
        )
        or resources
        != {
            "candidate_audit_passes_completed": 2,
            "candidate_audit_row_reads": 2 * EXPECTED_AUDIT_RECORDS,
            "candidate_reconstruction_row_reads": 2 * EXPECTED_ORIGINAL_ROWS,
            "candidate_total_row_reads": 2
            * (EXPECTED_AUDIT_RECORDS + EXPECTED_ORIGINAL_ROWS),
            "source_row_reads": 2 * EXPECTED_SOURCE_ROWS,
        }
    ):
        raise ValueError("expanded-delta verification boundary changed")
    serialized = json.dumps(verification, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"', '"user_prompt"'):
        if forbidden in serialized:
            raise ValueError("expanded-delta verification retained raw training text")


__all__ = [
    "EXPECTED_AUDIT_RECORDS",
    "EXPECTED_DELTA_ROWS",
    "EXPECTED_DELTA_TRAIN_TOKENS",
    "EXPECTED_LIMITS",
    "EXPECTED_NEAR_DUPLICATE_POLICY",
    "EXPECTED_SUCCESSOR_SHA256",
    "EXPECTED_SUCCESSOR_TRAIN_TOKENS",
    "build_recovery",
    "build_verification",
    "json_bytes",
    "jsonl_bytes",
    "load_plan",
    "_owner_artifacts",
    "validate_owner_preflight_artifacts",
    "validate_report",
    "validate_verification",
]
