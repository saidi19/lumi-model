"""Fail-closed Be-DSTC2 Banglish conversation curation recovery.

The runner and verifier each call :func:`build_artifacts` exactly once.  A call
parses the two authorized source objects once and the immutable v0.15 candidate
inventory once.  Evidence is raw-text-free; only the private candidate and
owner-review artifacts retain source text.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import posttraining_candidate_mix_reaudit as mix_v2
from . import posttraining_candidate_mix_reaudit_v15 as mix_v15
from .data_quality import MinHashNearDuplicateIndex
from .posttraining_candidate_mix_preflight import (
    EXPECTED_LANGUAGE_MODES,
    EXPECTED_TASKS,
    _canonical_message_digest,
    _duplicate_snapshot,
    _load_json,
    _load_jsonl,
    _targets,
    file_sha256,
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
from .posttraining_owner_review import build_owner_review_html
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-bedstc2-banglish-conversation-curation-recovery-plan.v0.5"
REPORT_SCHEMA_VERSION = "lumi.sft-bedstc2-banglish-conversation-curation-report.v0.5"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-bedstc2-banglish-conversation-curation-verification.v0.5"

SOURCE_ID = "code_mixed_dialog_bedstc2_banglish_conversation_candidate"
DATASET_ID = "sumanbanerjee1/Code-Mixed-Dialog"
REVISION = "9df1d4dc800548a883f8bc1a9ce4116c77aebc02"
LICENSE_ID = "apache-2.0"
ADAPTER_ID = "bedstc2_exact_dialogue_prefix_banglish_conversation_v0.2"
FAMILY = "code_mixed_dialog_dstc2_family"

EXPECTED_ORIGINAL_AUTHORIZATION_TEXT_SHA256 = (
    "7a9862c36b0081e5d27f33eed241ddda8e08beba7579aa8c90d938b73af41ac6"
)
EXPECTED_RECOVERY_AUTHORIZATION_TEXT_SHA256 = (
    "d375e305c2c6fea3f90b23d3ac473338ea9951d8e39107b5c6f1afed8ba184f5"
)
EXPECTED_PATH_AUTHORIZATION_TEXT_SHA256 = (
    "d76ed6b1d936fe89663f9ec67df91ee7b5e3539e3af223e495e9066d39b0dad7"
)
EXPECTED_PARSER_RECOVERY_AUTHORIZATION_TEXT_SHA256 = (
    "75b663ca0b91f2d8b7ed23014badaef717f8cfd6d87f15063cce07f78c82b327"
)
EXPECTED_SHAPE_RECOVERY_AUTHORIZATION_TEXT = "okay fix kor and shuru kor"
EXPECTED_SHAPE_RECOVERY_AUTHORIZATION_TEXT_SHA256 = (
    "26ea212fd276641452ab1297b6881b3cb9d8a19d2ffb0257368bff0c3d7ebd4b"
)
EXPECTED_INTERSECTION_RECOVERY_AUTHORIZATION_TEXT_SHA256 = (
    "01ada5f9f04b1b369901661390f821ae5d0ad533164390ed144d42a684b8a05f"
)
EXPECTED_INTERSECTION_RECOVERY_AUTHORIZATION_TEXT = (
    "I approve one bounded local-only Be-DSTC2 classified-body malformed-intersection "
    "diagnostic recovery. Bind prior authorization text SHA-256 "
    "b9adeb30ce93d3cb54c3f7ac237e47e65163ceebc048d7d0ffcf50fa9c00fae7, failed "
    "diagnostic script SHA-256 "
    "50355a36492802df5146f26b2faa79692f8a9113b90204f414f8de668cc1b4ba, test SHA-256 "
    "4d1c1e9c30f17e929b72f23f58b54fd478fc6ccb801918bd60d112f38812e155, and exact "
    "61-byte marker SHA-256 "
    "83a3d723116770dca970702ddca7684ca5da2cc5f1400f0cbc3ea66cf86853b1. Authorize "
    "one fresh aggregate-only scan using line.rstrip() classified_body for tab, pair, and "
    "KB classification while retaining the outer-whitespace flag from the original line "
    "and retaining no source text. If and only if it confirms 421 non-KB, 17 "
    "outer-trailing, 37 pair-user-trailing, zero intersections, and 475 unique rows, "
    "proceed with the previously specified v0.5/v4 attempt-004 two-pass recovery. "
    "Maximum 341,154 source-line reads, 629,432 existing-candidate reads, 40,000 "
    "new-candidate reads, 669,432 candidate-audit reads, 70 runtime-minutes, "
    "134,283,264 private-write bytes, and zero network/GPU/inference. No retry after a "
    "started pass or other action."
)
EXPECTED_DIALOGUES = {"dev": 500, "total": 2_118, "train": 1_618}
EXPECTED_LINES = {"dev": 28_617, "total": 113_718, "train": 85_101}
EXPECTED_MALFORMED_REASON_OCCURRENCES = {
    "source_non_kb_numbered_row": 421,
    "source_outer_trailing_whitespace": 17,
    "source_pair_user_trailing_whitespace": 37,
}
# The bound v0.7 intersection diagnostic proved these three reason classes are
# disjoint on the exact source objects: 421 + 17 + 37 = 475 unique rows.
EXPECTED_MALFORMED_SOURCE_ROWS_RANGE = (475, 475)
EXPECTED_BASE_RECORDS = 314_716
EXPECTED_BASE_TRAIN_TOKENS = 10_604_149
EXPECTED_BASE_VALIDATION_TOKENS = 628_604
EXPECTED_NEAR_POLICY = dict(mix_v15.EXPECTED_NEAR_DUPLICATE_POLICY)
EXPECTED_EMPTY_LEDGER_SHA256 = "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
MAXIMUM_SECONDS_PER_PASS = 30 * 60

EXPECTED_AUTHORIZATION = {
    "active_data_contract_change_authorized": False,
    "candidate_or_registry_mutation_authorized": False,
    "classified_body_malformed_intersection_recovery_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "gpu_authorized": False,
    "inference_authorized": False,
    "local_corrected_inventory_curation_recovery_authorized": True,
    "network_authorized": False,
    "parser_contract_recovery_authorized": True,
    "shape_observed_malformed_row_rejection_recovery_authorized": True,
    "production_promotion_authorized": False,
    "redownload_authorized": False,
    "remote_or_instance_action_authorized": False,
    "s3_action_authorized": False,
    "sft_training_authorized": False,
    "source_object_mutation_authorized": False,
    "source_promotion_authorized": False,
    "test_object_access_authorized": False,
    "two_fresh_sequential_local_passes_authorized": True,
    "unrelated_action_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_all_candidate_audit_row_reads": 669_432,
    "maximum_existing_candidate_row_reads": 629_432,
    "maximum_full_passes": 2,
    "maximum_gpu_use": 0,
    "maximum_inference_calls": 0,
    "maximum_network_bytes": 0,
    "maximum_new_candidate_row_reads": 40_000,
    "maximum_private_write_bytes": 134_217_728,
    "maximum_retained_records": 20_000,
    "maximum_runtime_minutes": 60,
    "maximum_source_line_reads": 227_436,
    "maximum_train_supervised_tokens": 200_000,
    "maximum_validation_supervised_tokens": 20_000,
    "minimum_train_supervised_tokens": 60_000,
    "owner_review_minimum_accepts": 29,
    "owner_review_rows": 32,
}
EXPECTED_COMBINED_SUCCESSOR_LIMITS = {
    "maximum_all_candidate_audit_row_reads": 669_432,
    "maximum_existing_candidate_row_reads": 629_432,
    "maximum_gpu_use": 0,
    "maximum_inference_calls": 0,
    "maximum_new_candidate_row_reads": 40_000,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 134_283_264,
    "maximum_runtime_minutes": 70,
    "maximum_source_line_reads": 341_154,
}
EXPECTED_PRIOR_CONSUMPTION = {
    "acquisition_and_inventory_attempt": {
        "candidate_rows_constructed": 0,
        "full_candidate_mix_audit_passes": 0,
        "network_payload_bytes": 5_483_185,
        "source_line_reads_conservative": 1_023_472,
    },
    "v0_2_parser_failure": {
        "candidate_rows_audited": 0,
        "candidate_rows_constructed": 0,
        "completed_full_candidate_mix_audit_passes": 0,
        "existing_candidate_row_reads": 0,
        "gpu_use": 0,
        "inference_calls": 0,
        "network_bytes": 0,
        "new_candidate_audit_row_reads": 0,
        "observed_command_wall_seconds": 2.14,
        "runner_passes_completed": 0,
        "runner_passes_started": 1,
        "source_line_reads_conservative_upper_bound": 113_718,
        "total_candidate_audit_row_reads": 0,
        "verifier_invoked": False,
    },
    "v0_3_parser_failure": {
        "candidate_rows_audited": 0,
        "candidate_rows_constructed": 0,
        "completed_full_candidate_mix_audit_passes": 0,
        "existing_candidate_row_reads": 0,
        "gpu_use": 0,
        "inference_calls": 0,
        "network_bytes": 0,
        "new_candidate_audit_row_reads": 0,
        "observed_command_wall_seconds": 2.18,
        "runner_passes_completed": 0,
        "runner_passes_started": 1,
        "source_line_reads_conservative_upper_bound": 113_718,
        "total_candidate_audit_row_reads": 0,
        "verifier_invoked": False,
    },
    "raw_text_free_diagnostics": {
        "candidate_row_reads": 0,
        "gpu_use": 0,
        "inference_calls": 0,
        "network_bytes": 0,
        "private_write_bytes": 32_041,
        "source_line_reads": 227_436,
    },
    "v0_4_curation_failure": {
        "candidate_rows_audited": 0,
        "candidate_rows_constructed": 0,
        "candidate_rows_materialized": 0,
        "completed_full_candidate_mix_audit_passes": 0,
        "existing_candidate_row_reads": 0,
        "gpu_use": 0,
        "inference_calls": 0,
        "network_bytes": 0,
        "new_candidate_row_reads": 0,
        "observed_command_wall_seconds": 1.72,
        "private_write_bytes": 47,
        "runner_passes_completed": 0,
        "runner_passes_started": 1,
        "source_line_reads_conservative_upper_bound": 113_718,
        "total_candidate_audit_row_reads": 0,
        "verifier_invoked": False,
    },
    "v0_6_malformed_intersection_diagnostic_failure": {
        "baseline_row_reads": 0,
        "candidate_row_reads": 0,
        "diagnostic_passes_completed": 0,
        "diagnostic_passes_started": 1,
        "gpu_use": 0,
        "inference_calls": 0,
        "network_bytes": 0,
        "observed_command_wall_seconds": 0.07,
        "private_write_bytes": 61,
        "source_line_reads_conservative_upper_bound": 113_718,
        "test_object_accessed": False,
        "verification_invoked": False,
    },
    "v0_7_malformed_intersection_diagnostic_success": {
        "baseline_row_reads": 0,
        "candidate_row_reads": 0,
        "diagnostic_passes_completed": 1,
        "diagnostic_passes_started": 1,
        "gpu_use": 0,
        "inference_calls": 0,
        "network_bytes": 0,
        "observed_command_wall_seconds": 0.15,
        "private_write_bytes": 5_801,
        "source_line_reads": 113_718,
        "test_object_accessed": False,
        "verification_completed": True,
    },
}
EXPECTED_PARSER_CONTRACT = {
    "classified_body_used_for_tab_pair_and_kb_shape": True,
    "kb_adjacency_required": False,
    "kb_grammar": "nonempty_entity R_-prefixed_relation nonempty_value",
    "observed_malformed_numbered_rows_action": "rejection_only",
    "preserve_source_bytes": True,
    "repair_scope": "parser_classification_only",
}
EXPECTED_EVALUATION_PROHIBITION = {
    **mix_v15.EXPECTED_EVALUATION_PROHIBITION,
    "bedstc2_dev_claimed_evaluation": "permanently_prohibited",
    "bedstc2_test_claimed_evaluation": "permanently_prohibited",
    "bedstc2_test_object": {
        "accessed": False,
        "bytes": 2_283_372,
        "git_blob_sha1": "85a9a4a43f3d1a9c54772c84d4a43872c9589ae6",
        "path": "data/bengali/dialog-dstc2-tst.txt",
    },
    "reason_bedstc2": "official train and dev objects used for SFT candidate construction",
}
EXPECTED_OUTPUT_PATHS = {
    "candidate": (
        "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-004/private/"
        "bedstc2-banglish-conversation-candidate.jsonl"
    ),
    "owner_review_html": (
        "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-004/private/"
        "owner-review-32.html"
    ),
    "owner_review_packet": (
        "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-004/private/"
        "owner-review-32.json"
    ),
    "rejection_ledger": (
        "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-004/private/"
        "rejection-ledger.jsonl"
    ),
    "report": "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-v0.5.json",
    "telemetry": (
        "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-004/private/telemetry.jsonl"
    ),
    "verification": (
        "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-verification-v0.5.json"
    ),
}
EXPECTED_V0_3_ABSENT_OUTPUT_PATHS = (
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-002/private/"
    "bedstc2-banglish-conversation-candidate.jsonl",
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-002/private/"
    "owner-review-32.html",
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-002/private/"
    "owner-review-32.json",
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-002/private/"
    "rejection-ledger.jsonl",
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-002/private/telemetry.jsonl",
    "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-verification-v0.3.json",
)
EXPECTED_V0_4_ABSENT_OUTPUT_PATHS = (
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-003/private/"
    "bedstc2-banglish-conversation-candidate.jsonl",
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-003/private/"
    "owner-review-32.html",
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-003/private/"
    "owner-review-32.json",
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-003/private/"
    "rejection-ledger.jsonl",
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-003/private/telemetry.jsonl",
    "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-verification-v0.4.json",
)
FAILED_INTERSECTION_DIAGNOSTIC_VERIFICATION_PATH = (
    "evidence/lumi-4b-sft-bedstc2-malformed-intersection-diagnostic-verification-v0.6.json"
)
EXPECTED_BINDING_PATHS = {
    "acquisition_receipt": "evidence/lumi-4b-sft-bedstc2-acquisition-receipt-v0.1.json",
    "benchmark_registry": "data/benchmark-signatures-v0.3.jsonl.gz",
    "chat_template": "configs/lumi-4b-chat-template-v0.1.json",
    "data_contract": "configs/lumi-4b-sft-data-contract-v0.1.json",
    "draft": "configs/lumi-4b-sft-bedstc2-banglish-conversation-curation-v0.1-draft.json",
    "failure_report": "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-v0.1.json",
    "failure_verification": (
        "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-verification-v0.1.json"
    ),
    "failed_intersection_diagnostic_marker": (
        "evidence/lumi-4b-sft-bedstc2-malformed-intersection-diagnostic-v0.6.json"
    ),
    "failed_intersection_diagnostic_script": (
        "scripts/diagnose_sft_bedstc2_malformed_intersections_v3.py"
    ),
    "failed_intersection_diagnostic_test": (
        "tests/test_bedstc2_malformed_intersection_diagnostic_v3.py"
    ),
    "implementation": "src/lumi/posttraining_bedstc2_banglish_conversation_curation_v4.py",
    "intersection_diagnostic_report": (
        "evidence/lumi-4b-sft-bedstc2-malformed-intersection-diagnostic-v0.7.json"
    ),
    "intersection_diagnostic_script": (
        "scripts/diagnose_sft_bedstc2_malformed_intersections_v4.py"
    ),
    "intersection_diagnostic_test": (
        "tests/test_bedstc2_malformed_intersection_diagnostic_v4.py"
    ),
    "intersection_diagnostic_verification": (
        "evidence/lumi-4b-sft-bedstc2-malformed-intersection-diagnostic-verification-v0.7.json"
    ),
    "parser_diagnostic_report": (
        "evidence/lumi-4b-sft-bedstc2-parser-contract-diagnostic-v0.4.json"
    ),
    "parser_diagnostic_script": "scripts/diagnose_sft_bedstc2_parser_contract_v1.py",
    "parser_diagnostic_test": "tests/test_bedstc2_parser_diagnostic_v1.py",
    "parser_diagnostic_verification": (
        "evidence/lumi-4b-sft-bedstc2-parser-contract-diagnostic-verification-v0.4.json"
    ),
    "predecessor_plan": "configs/lumi-4b-sft-bedstc2-banglish-conversation-curation-v0.1.json",
    "quality_config": "configs/data-quality-v0.3.json",
    "row_shape_diagnostic_report": (
        "evidence/lumi-4b-sft-bedstc2-row-shape-diagnostic-v0.5.json"
    ),
    "row_shape_diagnostic_script": "scripts/diagnose_sft_bedstc2_row_shapes_v2.py",
    "row_shape_diagnostic_test": "tests/test_bedstc2_row_shape_diagnostic_v2.py",
    "row_shape_diagnostic_verification": (
        "evidence/lumi-4b-sft-bedstc2-row-shape-diagnostic-verification-v0.5.json"
    ),
    "runner": "scripts/curate_sft_bedstc2_banglish_conversation_v4.py",
    "source_registry": "configs/lumi-4b-sft-source-registry-v0.21.json",
    "test": "tests/test_bedstc2_banglish_conversation_curation_v4.py",
    "tokenizer_json": (
        "../lumi-artifacts/4b-segment2-throughput-gate-v0.1/8bedfbc868e8/"
        "staging/input/tokenizer/tokenizer.json"
    ),
    "v0_15_plan": "configs/lumi-4b-sft-candidate-mix-reaudit-v0.15.json",
    "v0_15_report": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.15.json",
    "v0_15_verification": "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.15.json",
    "v0_3_failure_marker": (
        "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-v0.3.json"
    ),
    "v0_3_implementation": "src/lumi/posttraining_bedstc2_banglish_conversation_curation_v2.py",
    "v0_3_plan": (
        "configs/lumi-4b-sft-bedstc2-banglish-conversation-curation-recovery-v0.3.json"
    ),
    "v0_3_runner": "scripts/curate_sft_bedstc2_banglish_conversation_v2.py",
    "v0_3_test": "tests/test_bedstc2_banglish_conversation_curation_v2.py",
    "v0_3_verifier": "scripts/verify_sft_bedstc2_banglish_conversation_v2.py",
    "v0_4_failure_marker": (
        "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-v0.4.json"
    ),
    "v0_4_implementation": (
        "src/lumi/posttraining_bedstc2_banglish_conversation_curation_v3.py"
    ),
    "v0_4_plan": (
        "configs/lumi-4b-sft-bedstc2-banglish-conversation-curation-recovery-v0.4.json"
    ),
    "v0_4_runner": "scripts/curate_sft_bedstc2_banglish_conversation_v3.py",
    "v0_4_test": "tests/test_bedstc2_banglish_conversation_curation_v3.py",
    "v0_4_verifier": "scripts/verify_sft_bedstc2_banglish_conversation_v3.py",
    "verifier": "scripts/verify_sft_bedstc2_banglish_conversation_v4.py",
    "v0_2_failure_report": "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-v0.2.json",
    "v0_2_implementation": "src/lumi/posttraining_bedstc2_banglish_conversation_curation_v1.py",
    "v0_2_plan": "configs/lumi-4b-sft-bedstc2-banglish-conversation-curation-recovery-v0.2.json",
    "v0_2_runner": "scripts/curate_sft_bedstc2_banglish_conversation_v1.py",
    "v0_2_test": "tests/test_bedstc2_banglish_conversation_curation_v1.py",
    "v0_2_verifier": "scripts/verify_sft_bedstc2_banglish_conversation_v1.py",
}
EXPECTED_PLAN_PATH = "configs/lumi-4b-sft-bedstc2-banglish-conversation-curation-recovery-v0.5.json"
EXPECTED_STABLE_BINDING_SHA256 = {
    "acquisition_receipt": "50573eee3db949169ac5bd7cf38667e5455a1c651b30e314296ffb2097112c4e",
    "benchmark_registry": "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e",
    "chat_template": "bc7853a9277feddae2fd3db786c6a12f4fa3602a135a0408c38d4390f33bec12",
    "data_contract": "9a283d9333d29deb7b516fa45eb4eec50232be9925f4ee4aa3a8f624213b2f06",
    "draft": "bc8e50c2189ff5ebdde6959db48830f0116061640720473cb4ae73548197f25b",
    "failure_report": "daafe5e445a70773c95863fb9647dc5449f45bfd32a7efeb91bf329152af7349",
    "failure_verification": "4af8ee86676ec53a1998eb7a3036c6937271df5b52a8c61addf7d4e259d1acfa",
    "failed_intersection_diagnostic_marker": (
        "83a3d723116770dca970702ddca7684ca5da2cc5f1400f0cbc3ea66cf86853b1"
    ),
    "failed_intersection_diagnostic_script": (
        "50355a36492802df5146f26b2faa79692f8a9113b90204f414f8de668cc1b4ba"
    ),
    "failed_intersection_diagnostic_test": (
        "4d1c1e9c30f17e929b72f23f58b54fd478fc6ccb801918bd60d112f38812e155"
    ),
    "intersection_diagnostic_report": (
        "06c7558ec5c0a00873867758b6b0c7135d9897b1fac021d0d28b1f6f1f9ff0df"
    ),
    "intersection_diagnostic_script": (
        "8c5ad0a1e1063b6d29f759b10dade91dce887f0126d0e8f80196ef8f8ba46eea"
    ),
    "intersection_diagnostic_test": (
        "5d1f11bc2fcd3de84cad16342bbffa005e7fa0db435c0d1eca3b2dc03bfee87f"
    ),
    "intersection_diagnostic_verification": (
        "08d746c84f08d9e779167382eb7c27aa712b8c11f6a0036b446979a41b9557d8"
    ),
    "parser_diagnostic_report": "d9e9feb3b0b999c3277f44e470deb3091911db9f305b318ea90a48bcdc447f27",
    "parser_diagnostic_script": "c917792a59065a54d4a1a48bab73d5eaf25193ec12d40e20b3edc54746f3f08a",
    "parser_diagnostic_test": "beaf7f13ff3745a0c7e9a8e6fe857fbeee9030d8a395ff3598d8c6c358a7c616",
    "parser_diagnostic_verification": (
        "142c046e888087bb0de4c2b7fa0a9c29965098126391f22bea24ca52136d6087"
    ),
    "predecessor_plan": "9804e2f61ab0bca4bbc596a2c0a36865e54f4dd31c78f5723147a8fd36c64f10",
    "quality_config": "23fba21dbc822c84a5d16e63147fc1a7716576b8f1258cda8c035391f04f52c0",
    "row_shape_diagnostic_report": (
        "370ea8bfcde037cbb06c36941c23ac1a2ddbe687c75e45646b6aab761e495cbe"
    ),
    "row_shape_diagnostic_script": (
        "372fa0a22e9ca74dbf1481561a090f8290e8c28c2b2bace0dbddc1f1bb8f4dc1"
    ),
    "row_shape_diagnostic_test": "a3a846fc315d321ca4bfa3592fdbbbe1a740625a980091c3913e167fb96e8eda",
    "row_shape_diagnostic_verification": (
        "e020943714d6bd3cacb8cbdb975776664017f19ee115f67e6c98d9d2e7586025"
    ),
    "source_registry": "daa2d81755456728c5c72bca9ee95d5424aba0fc05e9fdd13223018f4055e3a1",
    "tokenizer_json": "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee",
    "v0_15_plan": "7d37369d115b0c666d96d138508f7b31a442cc8c6d990470ec62cf47096282fb",
    "v0_15_report": "fc81dc5827ebc1a039df7c7692b9f4ccd9144ad1ba1f2a32754ff6f6449d94bf",
    "v0_15_verification": "1af422f2fbb5dcf21b2d4212891ab4d60c9c4cb63a797ac87acfa7598b9be782",
    "v0_2_failure_report": "05747d8c255802095b1b813323be07c5fb54b088475189b932b8af8b345a20f9",
    "v0_2_implementation": "41c1d6eef19b0e2387a89fad0c6138178a0074835b46843f5ecd2bf8a97de4aa",
    "v0_2_plan": "cbc60297886e23f86ec981a893a67ebcb8a45d6c5405498ad8d404b75197476b",
    "v0_2_runner": "d7267b94716a28e53f2a884254ba3a67edf5905742bfd84867eef538c3555157",
    "v0_2_test": "abe436f28488108e8723434bff9fedca0ecd103341049796643e130750a6214b",
    "v0_2_verifier": "b359e7f583e113c646510e7f8b66a6a20e1b0d7bb7388dfcb89e17d9caa958cf",
    "v0_3_failure_marker": "a09a86416d42c7d11df6f6694f36f4a0a119da4b4fdd11b2f606354b359f75ae",
    "v0_3_implementation": "8cac4260d0b82a220996b5c06ef1d76f7dda7d50b2307009e64faae2c2d97618",
    "v0_3_plan": "4db4becc887ddfa64fe8983426bf50cc9ac6f24b050152bbdb26f1f5cb0b570c",
    "v0_3_runner": "6714aa594c40f560690c962df421937427adcb76527d8bee1a704d04b654a8e5",
    "v0_3_test": "4970fbc67b72b6c42e58b55c69beab241c39454b575ebcc2944b2f7ddee33731",
    "v0_3_verifier": "2d2e18965e4987729cc9cadc7a0bc160da4724557ed897d98288068d6de80c06",
    "v0_4_failure_marker": (
        "0f60a1baca9d289ffbb2936394873517d98f42df4437016866d304c4ae69fa49"
    ),
    "v0_4_implementation": (
        "b4a411142358dcaf6937a09b50ccbd94c1a0bdf9baf44052aa11210cb1e276dc"
    ),
    "v0_4_plan": "4015afc5ebba188fdc6b649ec3801e2c0a171ee1be73fcb2a425d611a21d5eab",
    "v0_4_runner": "0b60a93d3ee27d71cde8c6a710deb37ae8309da55a1e63a09530acce551fe46b",
    "v0_4_test": "9fc40fe1ca294d8d669c9394932ca934c6640d78df88eb2f21baf9d565de639c",
    "v0_4_verifier": (
        "5e11b637c8a6ed106b1b151eb561c6026f26420676bbfcd04586d5033d0aee54"
    ),
}
EXPECTED_INTERNAL_BINDINGS = {
    "acquisition_receipt": (
        "receipt_sha256",
        "a7f0f14bed39b109237c845c66dec021c10c6f81877ae615461a9b8c02c8e5b4",
    ),
    "failure_report": (
        "report_sha256",
        "a4d1976c4ad6d871fba21b9243f40b4c7dc3995dd3f71c7661b2efb269455680",
    ),
    "failure_verification": (
        "verification_sha256",
        "cffd49e3fa154f1a8e5704d45abdb89cab80568e1426dac40253b519a05f2519",
    ),
    "parser_diagnostic_report": (
        "report_sha256",
        "dba9bd01e8814b41a39e13cb6f58797586107be13e90e4385d72cef05bc48cfa",
    ),
    "parser_diagnostic_verification": (
        "verification_sha256",
        "3b9da1ae140d4ae115d5f8ae16012f12174d28538b75d1df52e0ead798c13c08",
    ),
    "row_shape_diagnostic_report": (
        "report_sha256",
        "d1dbbbe9e80dfaf244cdc54dabc12c31e9ca418c73e82806041173ae4c044b3e",
    ),
    "row_shape_diagnostic_verification": (
        "verification_sha256",
        "3be64dbbe5ea0db500fac2487fe560ca1ef3e756bae4e8443ac354022553c4d4",
    ),
    "intersection_diagnostic_report": (
        "report_sha256",
        "86ed5db7dc539e0283854f2686db382ee0858996a31f707d422987dc6ad7c00f",
    ),
    "intersection_diagnostic_verification": (
        "verification_sha256",
        "c7c6c01f125a060961f7221d33bbea42ea0e748189e63a9283d2fd4fe548fbef",
    ),
    "v0_2_failure_report": (
        "report_sha256",
        "1990ec2194c3f925147f84b3ee9485ac5da8fbe1bcd6787027eec1c3e23d76f5",
    ),
}
EXPECTED_RAW = {
    "LICENSE": {
        "bytes": 11_357,
        "git_blob_sha1": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
        "sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    },
    "README.md": {
        "bytes": 5_978,
        "git_blob_sha1": "ef54e017c4bf440b89cf3644a76acef22cdf3981",
        "sha256": "b6733216e7b23677b1b724c45b224f362eb259bf0e5686521d43eaf9a74cf2e6",
    },
    "data/bengali/dialog-dstc2-dev.txt": {
        "bytes": 1_361_043,
        "git_blob_sha1": "543eb0aa02d9ec7f1af65516017a1bfcace1f636",
        "sha256": "a46b98e4f3ff3c9575240fe42fee2d735fa61ec41aacd53d73627b33b7147116",
    },
    "data/bengali/dialog-dstc2-trn.txt": {
        "bytes": 4_085_040,
        "git_blob_sha1": "4683420d14bf546c31e7698ce9ecd799ba0f88d3",
        "sha256": "79644ed579c831626b1f2a2dfef1f804b835c9e3da19f93fe493c9fbbb3b591c",
    },
    "data/bengali/vocab_splits.json": {
        "bytes": 19_767,
        "git_blob_sha1": "39bba927098a8792ff3b4f611bf35362eb0b8e4f",
        "sha256": "f55a71418e699ece420b0206c74422f1889fbc9da24be4aea19c15d6912a6870",
    },
}
EXPECTED_TEST_OBJECT = "data/bengali/dialog-dstc2-tst.txt"
EXPECTED_TEST_OBJECT_BYTES = 2_283_372
EXPECTED_TEST_OBJECT_GIT_BLOB_SHA1 = "85a9a4a43f3d1a9c54772c84d4a43872c9589ae6"

_NUMBERED = re.compile(r"^([1-9][0-9]*) (.*)$")
_KB = re.compile(r"^(\S+) (R_\S+) (\S(?:.*\S)?)$")
_BENGALI_SCRIPT = re.compile(r"[\u0980-\u09ff]")
_WORD = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*", re.IGNORECASE)
_GENERATED_MARKER = re.compile(r"<[^>]+>")
_UNSAFE = re.compile(
    r"\b(?:api[_ -]?key|bomb|credit card|explosive|kill|malware|password|"
    r"self[- ]?harm|suicide|weapon)\b",
    re.IGNORECASE,
)
_FORBIDDEN_RAW_KEYS = {
    "assistant",
    "assistant_response",
    "body",
    "content",
    "messages",
    "user",
    "user_prompt",
}


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _check_deadline(started: float) -> None:
    if time.monotonic() - started > MAXIMUM_SECONDS_PER_PASS:
        raise TimeoutError("Be-DSTC2 pass exceeded its 30-minute half-budget")


def _raw_key_scan(value: Any) -> None:
    if isinstance(value, Mapping):
        if _FORBIDDEN_RAW_KEYS.intersection(value):
            raise ValueError("raw training text entered raw-text-free evidence")
        for item in value.values():
            _raw_key_scan(item)
    elif isinstance(value, list):
        for item in value:
            _raw_key_scan(item)


def _git_blob_sha1(path: Path, size: int) -> str:
    digest = hashlib.sha1()  # noqa: S324 - Git object identity is intentionally SHA-1.
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ParsedEntry:
    body: str
    kind: str
    ordinal: int
    user: str | None = None
    assistant: str | None = None
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedDialogue:
    entries: tuple[ParsedEntry, ...]
    object_name: str
    source_index: int


@dataclass(frozen=True)
class BaselineState:
    old_resolved: Mapping[str, Path]
    rows: list[dict[str, Any]]
    source_metrics: dict[str, dict[str, Any]]
    source_plans: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class BuildArtifacts:
    candidate_bytes: bytes
    html_bytes: bytes
    owner_packet_bytes: bytes
    rejection_bytes: bytes
    report: dict[str, Any]
    telemetry_bytes: bytes


class ProvisionalRegistry:
    """Delegate registered sources and admit exactly one unpromoted candidate source."""

    def __init__(self, base: Any) -> None:
        self.base = base

    def record_reasons(self, **kwargs: Any) -> tuple[str, ...]:
        if kwargs.get("source_id") != SOURCE_ID:
            return self.base.record_reasons(**kwargs)
        expected = {
            "dataset_id": DATASET_ID,
            "language": "multilingual",
            "license_id": LICENSE_ID,
            "revision": REVISION,
            "task_category": "conversation",
        }
        reasons = [
            f"provisional_source_{key}"
            for key, value in expected.items()
            if kwargs.get(key) != value
        ]
        if kwargs.get("split") not in {"train", "validation"}:
            reasons.append("provisional_source_split")
        return tuple(reasons)


def parse_dialogues(
    path: Path, *, object_name: str, started: float | None = None
) -> tuple[list[ParsedDialogue], int]:
    dialogues: list[ParsedDialogue] = []
    entries: list[ParsedEntry] = []
    expected_ordinal = 1
    lines = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in handle:
            lines += 1
            if started is not None and lines % 4096 == 0:
                _check_deadline(started)
            line = raw[:-2] if raw.endswith("\r\n") else raw[:-1] if raw.endswith("\n") else raw
            if line == "":
                if not entries:
                    raise ValueError("empty or repeated dialogue boundary")
                dialogues.append(ParsedDialogue(tuple(entries), object_name, len(dialogues)))
                entries = []
                expected_ordinal = 1
                continue
            if not line.strip() or line != line.lstrip():
                raise ValueError("whitespace changed source framing")
            framing_line = line.rstrip()
            match = _NUMBERED.fullmatch(framing_line)
            if match is None or int(match.group(1)) != expected_ordinal:
                raise ValueError("source ordinal framing changed")
            ordinal = int(match.group(1))
            expected_ordinal += 1
            body = line.split(" ", 1)[1]
            classified_body = match.group(2)
            reasons: list[str] = []
            if line != framing_line:
                reasons.append("source_outer_trailing_whitespace")
            tabs = classified_body.count("\t")
            if tabs == 1:
                user, assistant = classified_body.split("\t")
                if not user or not assistant:
                    raise ValueError("source pair field framing changed")
                if (
                    user == user.lstrip()
                    and user != user.rstrip()
                    and assistant == assistant.strip()
                ):
                    reasons.append("source_pair_user_trailing_whitespace")
                elif user != user.strip() or assistant != assistant.strip():
                    raise ValueError("source pair field framing changed")
                if reasons:
                    entries.append(
                        ParsedEntry(
                            body,
                            "malformed",
                            ordinal,
                            reason_codes=tuple(sorted(reasons)),
                        )
                    )
                    continue
                is_api = assistant.casefold() == "api_call" or assistant.casefold().startswith(
                    "api_call "
                )
                kind = "api" if is_api else "pair"
                entries.append(ParsedEntry(body, kind, ordinal, user, assistant))
            elif tabs == 0:
                if _KB.fullmatch(classified_body) is None:
                    fields = classified_body.split(" ")
                    if (
                        len(fields) == 3
                        and all(fields)
                        and not any(field.startswith("R_") for field in fields)
                    ):
                        reasons.append("source_non_kb_numbered_row")
                    else:
                        raise ValueError("malformed non-tab source row")
                if reasons:
                    entries.append(
                        ParsedEntry(
                            body,
                            "malformed",
                            ordinal,
                            reason_codes=tuple(sorted(reasons)),
                        )
                    )
                else:
                    entries.append(ParsedEntry(body, "kb", ordinal))
            else:
                raise ValueError("source row contains multiple tabs")
    if entries:
        dialogues.append(ParsedDialogue(tuple(entries), object_name, len(dialogues)))
    return dialogues, lines


def dialogue_identity(dialogue: ParsedDialogue) -> str:
    return sha256_value(
        [
            {
                "assistant": item.assistant,
                "body": item.body,
                "kind": item.kind,
                "reason_codes": item.reason_codes,
                "user": item.user,
            }
            for item in dialogue.entries
        ]
    )


def dialogue_split(identity: str) -> str:
    digest = hashlib.sha256(f"bedstc2-dialogue-split-v0.1\0{identity}".encode()).digest()
    return "validation" if int.from_bytes(digest[:8], "big") % 20 == 0 else "train"


def _normalized_vocab_tokens(values: Any, *, name: str) -> frozenset[str]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"vocab_splits {name} changed")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"vocab_splits {name} item changed")
        normalized.extend(token.casefold() for token in _WORD.findall(value))
    if not normalized:
        raise ValueError(f"vocab_splits {name} has no lexical tokens")
    return frozenset(normalized)


def load_vocab(path: Path) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {"english_language_vocab", "native_language_vocab", "others_vocab"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("vocab_splits inventory changed")
    raw_english = _normalized_vocab_tokens(value["english_language_vocab"], name="english")
    raw_native = _normalized_vocab_tokens(value["native_language_vocab"], name="native")
    others = _normalized_vocab_tokens(value["others_vocab"], name="others")
    english = raw_english - raw_native - others
    native = raw_native - raw_english - others
    if not english or not native:
        raise ValueError("vocab_splits exclusive anchors are empty")
    return english, native, others


def _tokens(text: str) -> frozenset[str]:
    return frozenset(item.casefold() for item in _WORD.findall(text))


def dual_anchor(text: str, *, english: frozenset[str], native: frozenset[str]) -> bool:
    tokens = _tokens(text)
    return _BENGALI_SCRIPT.search(text) is None and bool(tokens & english) and bool(tokens & native)


def punctuation_integrity(text: str) -> bool:
    if not any(character.isalnum() for character in text):
        return False
    if any(unicodedata.category(character) == "Cc" for character in text):
        return False
    return all(
        text.count(left) == text.count(right)
        for left, right in (("(", ")"), ("[", "]"), ("{", "}"))
    )


def _row_id(identity: str, entry: ParsedEntry) -> str:
    return (
        "bedstc2-banglish-"
        + sha256_value(
            {
                "dialogue_identity_sha256": identity,
                "ordinal": entry.ordinal,
                "response_sha256": _sha256_bytes(str(entry.assistant or "").encode()),
            }
        )[:24]
    )


def candidate_rows(
    dialogues: Sequence[ParsedDialogue],
    *,
    english: frozenset[str],
    native: frozenset[str],
    others: frozenset[str],
    started: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_dialogues: dict[str, tuple[ParsedEntry, ...]] = {}
    for dialogue in dialogues:
        if started is not None and len(seen_dialogues) % 128 == 0:
            _check_deadline(started)
        identity = dialogue_identity(dialogue)
        prior = seen_dialogues.get(identity)
        if prior is not None:
            if prior != dialogue.entries:
                raise ValueError("dialogue identity collision")
            continue
        seen_dialogues[identity] = dialogue.entries
        split = dialogue_split(identity)
        prefix_parts: list[str] = []
        for entry in dialogue.entries:
            entry_id = _row_id(identity, entry)
            if entry.kind == "malformed":
                rejected.append(
                    {"reason_codes": list(entry.reason_codes), "record_id": entry_id}
                )
                continue
            if entry.kind == "kb":
                prefix_parts.append(f"KB: {entry.body}")
                rejected.append({"reason_codes": ["kb_only_record"], "record_id": entry_id})
                continue
            user = str(entry.user)
            assistant = str(entry.assistant)
            if entry.kind == "api":
                prefix_parts.extend((f"User: {user}", f"API: {assistant}"))
                rejected.append(
                    {"reason_codes": ["api_call_only_assistant_response"], "record_id": entry_id}
                )
                continue

            prompt = "\n".join([*prefix_parts, f"User: {user}"])
            reasons: list[str] = []
            if user.casefold() == "<silence>" or assistant.casefold() in {
                "<silence>",
                "<unk>",
                "unknown",
            }:
                reasons.append("silence_or_unknown")
            if _GENERATED_MARKER.search(user) or _GENERATED_MARKER.search(assistant):
                reasons.append("generated_metadata")
            if not dual_anchor(user, english=english, native=native):
                reasons.append("current_user_not_dual_anchor")
            if not dual_anchor(assistant, english=english, native=native):
                reasons.append("assistant_not_dual_anchor")
            if not punctuation_integrity(assistant):
                reasons.append("source_native_punctuation_integrity")
            if _UNSAFE.search(user) or _UNSAFE.search(assistant):
                reasons.append("safety_pattern")
            assistant_tokens = _tokens(assistant)
            prompt_tokens = _tokens(prompt)
            entity_tokens = assistant_tokens & others
            numeric_tokens = {
                token for token in assistant_tokens if any(ch.isdigit() for ch in token)
            }
            if not (entity_tokens | numeric_tokens).issubset(prompt_tokens):
                reasons.append("ungrounded_entity_or_factual_response")
            if reasons:
                rejected.append({"reason_codes": sorted(set(reasons)), "record_id": entry_id})
            else:
                messages = [
                    {"content": prompt, "role": "user"},
                    {"content": assistant, "role": "assistant"},
                ]
                content_sha = sha256_value(messages)
                upstream = {
                    "dialogue_identity_sha256": identity,
                    "ordinal": entry.ordinal,
                    "response_sha256": _sha256_bytes(assistant.encode()),
                }
                records.append(
                    {
                        "language": "multilingual",
                        "language_mode": "banglish",
                        "messages": messages,
                        "provenance": {
                            "assistant_response_origin": "human_in_house_annotated_source_derived",
                            "consent_basis": "source_level_open_license",
                            "content_sha256": content_sha,
                            "owner_review_status": "pending_deterministic_spot_check",
                            "prompt_origin": "source_derived_dstc2_dialogue_prefix",
                            "rights_blocker": (
                                "upstream_dstc2_rights_chain_not_independently_verified"
                            ),
                            "transform_sha256": sha256_value(
                                {
                                    "adapter": ADAPTER_ID,
                                    "content_sha256": content_sha,
                                    "split": split,
                                    **upstream,
                                }
                            ),
                        },
                        "quality": {
                            "pii_scan": "passed",
                            "review_method": "deterministic_source_gates_pending_owner_review",
                            "review_status": "accepted",
                            "score": 0.85,
                            "secret_scan": "passed",
                        },
                        "record_id": entry_id,
                        "safety": {
                            "policy_version": "lumi_frozen_safety_gate_v0.1",
                            "risk_labels": [],
                        },
                        "schema_version": "lumi.sft.v0.1",
                        "source": {
                            "dataset_id": DATASET_ID,
                            "document_id": f"dialogue-{identity}-turn-{entry.ordinal}",
                            "license_id": LICENSE_ID,
                            "revision": REVISION,
                            "source_id": SOURCE_ID,
                            "uri": (
                                "https://github.com/sumanbanerjee1/Code-Mixed-Dialog/blob/"
                                f"{REVISION}/data/bengali/{dialogue.object_name}"
                                f"#dialogue={dialogue.source_index}&turn={entry.ordinal}"
                            ),
                        },
                        "split": split,
                        "task_category": "conversation",
                    }
                )
            prefix_parts.extend((f"User: {user}", f"Assistant: {assistant}"))
    return records, rejected


def _runtime(root: Path, plan: Mapping[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    contract = root / plan["bindings"]["data_contract"]["path"]
    benchmark_path = root / plan["bindings"]["benchmark_registry"]["path"]
    policy = load_sft_policy(contract)
    policy = replace(
        policy,
        benchmark_decontamination={
            "anchor_ngram_tokens": 8,
            "minimum_anchor_matches": 2,
            "registry": str(benchmark_path),
            "registry_sha256": plan["bindings"]["benchmark_registry"]["sha256"],
        },
    )
    benchmark = mix_v2.BenchmarkRegistry(policy.benchmark_decontamination, root)
    template = load_chat_template(root / plan["bindings"]["chat_template"]["path"])
    tokenizer = PureByteLevelBPETokenizer(root / plan["bindings"]["tokenizer_json"]["path"])
    verify_chat_tokenizer(tokenizer, template)
    registry = load_sft_source_registry(
        root / plan["bindings"]["source_registry"]["path"],
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=plan["bindings"]["source_registry"]["sha256"],
    )
    if len(registry.by_id) != 24 or SOURCE_ID in registry.by_id:
        raise ValueError("registry-v0.21 inventory or provisional-source boundary changed")
    return policy, benchmark, template, tokenizer, registry


def _load_baseline(
    root: Path,
    plan: Mapping[str, Any],
    *,
    tokenizer: Any,
    template: Any,
    registry: Any,
    started: float | None = None,
) -> BaselineState:
    report = _load_json(root / plan["bindings"]["v0_15_report"]["path"])
    verification = _load_json(root / plan["bindings"]["v0_15_verification"]["path"])
    mix_v15.validate_report(report)
    mix_v15.validate_verification(verification)
    if (
        report.get("report_sha256")
        != "19fc82952653c7e1a022338fbe58bafeae25a6a88d63a2a3e3bc4154e5d99504"
        or verification.get("verification_sha256")
        != "bf11960dee2f429d11c964ef2e3f8960062796bf950d27f8e55a88576f19b9da"
    ):
        raise ValueError("v0.15 internal evidence changed")
    v15_plan = mix_v15.load_plan(root / plan["bindings"]["v0_15_plan"]["path"], root)
    bound = {
        name: mix_v15._resolve_binding(root, binding, name)
        for name, binding in v15_plan["bindings"].items()
    }
    mix_v15._validate_predecessor_and_promotion(bound)
    source_plans, candidate_paths, _, old_resolved = mix_v15._source_inventory(
        root=root, bound=bound
    )
    rows: list[dict[str, Any]] = []
    source_metrics: dict[str, dict[str, Any]] = {}
    for source_id in sorted(source_plans):
        if started is not None:
            _check_deadline(started)
        source_plan = source_plans[source_id]
        registry_source = registry.by_id.get(source_id)
        if (
            registry_source is None
            or registry_source.revision != source_plan["revision"]
            or registry_source.status not in {"canary_eligible", "candidate_eligible"}
        ):
            raise ValueError(f"baseline registry source mismatch: {source_id}")
        source_rows = _load_jsonl(candidate_paths[source_id])
        if source_id == mix_v15.BILLSUM_SOURCE["source_id"]:
            mix_v15._validate_billsum_rows(source_rows)
        elif source_id == mix_v15.BANGLABRIDGE_SOURCE["source_id"]:
            mix_v15._validate_banglabridge_rows(source_rows)
        metrics = mix_v15.v13.v18._source_metrics(
            rows=source_rows,
            source_id=source_id,
            family=str(source_plan["family"]),
            language_policy=str(source_plan["language_policy"]),
            candidate_sha256=file_sha256(candidate_paths[source_id]),
            tokenizer=tokenizer,
            template=template,
        )
        source_metrics[source_id] = metrics
        rows.extend(source_rows)
    train, validation, *_ = mix_v15.v13.v18._aggregate_metrics(source_metrics)
    if (
        len(rows) != EXPECTED_BASE_RECORDS
        or train != EXPECTED_BASE_TRAIN_TOKENS
        or validation != EXPECTED_BASE_VALIDATION_TOKENS
        or source_metrics != report.get("source_metrics")
    ):
        raise ValueError("v0.15 baseline rows or token ledger changed")
    excluded = set(mix_v15.EXPECTED_EXCLUDED_SOURCE_IDS)
    if any(row.get("source", {}).get("source_id") in excluded for row in rows):
        raise ValueError("excluded source entered v0.15 baseline")
    if set(registry.by_id) - set(source_plans) != excluded:
        raise ValueError("v0.15 excluded-source registry inventory changed")
    return BaselineState(old_resolved, rows, source_metrics, source_plans)


def _new_priority(row: Mapping[str, Any]) -> tuple[str, str]:
    record_id = str(row["record_id"])
    return (
        hashlib.sha256(f"bedstc2-token-cap-v0.1\0{record_id}".encode()).hexdigest(),
        record_id,
    )


def _select_baseline_first(
    *,
    baseline: Sequence[dict[str, Any]],
    proposed: Sequence[dict[str, Any]],
    tokenizer: Any,
    template: Any,
    rejected: list[dict[str, Any]],
    started: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    if len(proposed) > EXPECTED_LIMITS["maximum_retained_records"]:
        raise ValueError("constructed candidate row inventory exceeds the audit cap")
    minhash_config = {
        key: EXPECTED_NEAR_POLICY[key]
        for key in ("num_perm", "package_version", "scheme", "seed", "shingle_tokens", "threshold")
    }
    near_index = MinHashNearDuplicateIndex(minhash_config)
    exact_index: dict[str, str] = {}
    for index, row in enumerate(sorted(baseline, key=lambda item: str(item["record_id"]))):
        if started is not None and index % 4096 == 0:
            _check_deadline(started)
        record_id = str(row["record_id"])
        digest = _canonical_message_digest(row["messages"])
        if digest in exact_index:
            raise ValueError("v0.15 baseline exact duplicate changed")
        signature = near_index.signature(mix_v2._role_tagged_conversation(row))
        if near_index.duplicate_of(signature) is not None:
            raise ValueError("v0.15 baseline MinHash evidence changed")
        exact_index[digest] = record_id
        near_index.add(record_id, signature)

    selected: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for index, row in enumerate(sorted(proposed, key=_new_priority)):
        if started is not None and index % 512 == 0:
            _check_deadline(started)
        record_id = str(row["record_id"])
        try:
            encoded = encode_sft_messages(row["messages"], tokenizer=tokenizer, template=template)
        except ValueError:
            rejected.append({"reason_codes": ["tokenizer_or_4096_fit"], "record_id": record_id})
            continue
        if encoded.supervised_tokens <= 0 or len(encoded.input_ids) > 4096:
            rejected.append({"reason_codes": ["tokenizer_or_4096_fit"], "record_id": record_id})
            continue
        digest = _canonical_message_digest(row["messages"])
        exact_match = exact_index.get(digest)
        if exact_match is not None:
            rejected.append(
                {
                    "matched_record_id_sha256": _sha256_bytes(exact_match.encode()),
                    "reason_codes": ["exact_duplicate_of_existing_or_selected"],
                    "record_id": record_id,
                }
            )
            continue
        signature = near_index.signature(mix_v2._role_tagged_conversation(row))
        near_match = near_index.duplicate_of(signature)
        if near_match is not None:
            rejected.append(
                {
                    "matched_record_id_sha256": _sha256_bytes(near_match.encode()),
                    "reason_codes": ["minhash_0_9_duplicate_of_existing_or_selected"],
                    "record_id": record_id,
                }
            )
            continue
        split = str(row["split"])
        cap = (
            EXPECTED_LIMITS["maximum_train_supervised_tokens"]
            if split == "train"
            else EXPECTED_LIMITS["maximum_validation_supervised_tokens"]
        )
        if totals[split] + encoded.supervised_tokens > cap:
            rejected.append(
                {"reason_codes": ["split_supervised_token_cap"], "record_id": record_id}
            )
            continue
        selected.append(row)
        totals[split] += encoded.supervised_tokens
        ledger.append(
            {
                "record_id": record_id,
                "serialized_tokens": len(encoded.input_ids),
                "split": split,
                "supervised_tokens": encoded.supervised_tokens,
            }
        )
        exact_index[digest] = record_id
        near_index.add(record_id, signature)
    if totals["train"] < EXPECTED_LIMITS["minimum_train_supervised_tokens"]:
        raise ValueError("retained train supervised-token floor not met")
    if len(selected) > EXPECTED_LIMITS["maximum_retained_records"]:
        raise ValueError("retained-record cap exceeded")
    combined_records = len(baseline) + len(selected)
    near_zero = {
        "comparison_unit": EXPECTED_NEAR_POLICY["comparison_unit"],
        "component_ledger_sha256": EXPECTED_EMPTY_LEDGER_SHA256,
        "components": combined_records,
        "cross_source_components": 0,
        "duplicate_extra_records": 0,
        "edge_count": 0,
        "largest_component_records": 1 if combined_records else 0,
        "multi_record_components": 0,
        "per_source_multi_record_components": {},
        "records": combined_records,
        "size_histogram": {},
        "train_validation_components": 0,
        "train_validation_records": 0,
    }
    return (
        sorted(selected, key=lambda row: str(row["record_id"])),
        sorted(ledger, key=lambda row: str(row["record_id"])),
        {"train": totals["train"], "validation": totals["validation"]},
        near_zero,
    )


def render_owner_packet(
    packet: Sequence[Mapping[str, Any]],
) -> tuple[bytes, bytes, str, str]:
    if len(packet) != EXPECTED_LIMITS["owner_review_rows"]:
        raise ValueError("owner packet incomplete")
    record_ids: set[str] = set()
    for row in packet:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"language_mode", "messages", "record_id", "skill_id", "split"}
            or row.get("language_mode") != "banglish"
            or row.get("skill_id") != "conversation"
            or row.get("split") != "train"
            or not isinstance(row.get("record_id"), str)
            or not isinstance(row.get("messages"), list)
        ):
            raise ValueError("owner packet row contract changed")
        record_ids.add(str(row["record_id"]))
    if len(record_ids) != len(packet):
        raise ValueError("owner packet record IDs are not unique")
    semantic = _sha256_bytes(canonical_json_bytes(packet))
    packet_bytes = json_bytes(packet)
    file_hash = _sha256_bytes(packet_bytes)
    if semantic == file_hash:
        raise ValueError("owner packet hash domains collapsed")
    generic_html = build_owner_review_html(packet, sample_sha256=semantic)
    replacements = {
        "<title>Lumi owner review</title>": "<title>Lumi Be-DSTC2 owner review</title>",
        "<h1>Lumi owner spot-check</h1>": "<h1>Lumi Be-DSTC2 owner spot-check</h1>",
        (
            '<p class="honesty">These rows are AI-generated and owner-reviewed. '
            "Approving them does not make them human-written.</p>"
        ): (
            '<p class="honesty">These are source-derived human in-house Bengali '
            "code-mixed adaptations with underlying DSTC2 authorship not established. "
            "Approval is a 32-row quality spot-check only.</p>"
        ),
    }
    html = generic_html
    for old, new in replacements.items():
        if html.count(old) != 1:
            raise ValueError("owner HTML static marker changed")
        html = html.replace(old, new, 1)
    if (
        "lumi.ai-assisted-owner-decisions.v0.3" not in html
        or "lumi-owner-review-decisions-v0.3.json" not in html
        or html.count(semantic) != 1
    ):
        raise ValueError("generic owner decision contract or semantic binding changed")
    return packet_bytes, html.encode(), semantic, file_hash


def build_owner_artifacts(selected: Sequence[dict[str, Any]]) -> tuple[bytes, bytes, str, str]:
    train = [row for row in selected if row["split"] == "train"]
    ranked = sorted(
        train,
        key=lambda row: hashlib.sha256(
            f"bedstc2-owner-v0.1\0{row['record_id']}".encode()
        ).hexdigest(),
    )[: EXPECTED_LIMITS["owner_review_rows"]]
    packet = [
        {
            "language_mode": "banglish",
            "messages": row["messages"],
            "record_id": row["record_id"],
            "skill_id": "conversation",
            "split": "train",
        }
        for row in ranked
    ]
    return render_owner_packet(packet)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _artifact(payload: bytes, *, raw: bool) -> dict[str, Any]:
    return {
        "bytes": len(payload),
        "contains_raw_training_text": raw,
        "sha256": _sha256_bytes(payload),
    }


def _near_zero_valid(near: Mapping[str, Any], expected_records: int) -> bool:
    return near == {
        "comparison_unit": EXPECTED_NEAR_POLICY["comparison_unit"],
        "component_ledger_sha256": EXPECTED_EMPTY_LEDGER_SHA256,
        "components": expected_records,
        "cross_source_components": 0,
        "duplicate_extra_records": 0,
        "edge_count": 0,
        "largest_component_records": 1 if expected_records else 0,
        "multi_record_components": 0,
        "per_source_multi_record_components": {},
        "records": expected_records,
        "size_histogram": {},
        "train_validation_components": 0,
        "train_validation_records": 0,
    }


def _audit_combined(
    *,
    baseline: BaselineState,
    selected: list[dict[str, Any]],
    selected_file_sha256: str,
    near: dict[str, Any],
    policy: Any,
    benchmark: Any,
    registry: Any,
    tokenizer: Any,
    template: Any,
    started: float | None = None,
) -> dict[str, Any]:
    combined = [*baseline.rows, *selected]
    accepted, generic = filter_sft_records(
        combined,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=ProvisionalRegistry(registry),
    )
    if len(accepted) != len(combined) or generic["rejected"]:
        raise ValueError("combined generic gate rejected retained rows")
    duplicate_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for index, row in enumerate(accepted):
        if started is not None and index % 4096 == 0:
            _check_deadline(started)
        duplicate_index[_canonical_message_digest(row["messages"])].append(
            (row["source"]["source_id"], row["split"])
        )
    exact = _duplicate_snapshot(duplicate_index)
    if exact != {
        "cross_source_duplicate_groups": 0,
        "duplicate_extra_records": 0,
        "duplicate_groups": 0,
        "train_validation_overlap_groups": 0,
    }:
        raise ValueError("combined exact duplicate gate failed")
    if not _near_zero_valid(near, len(combined)):
        raise ValueError("combined MinHash gate evidence changed")

    source_plans = dict(baseline.source_plans)
    source_plans[SOURCE_ID] = {
        "family": FAMILY,
        "language_policy": "language_field_multilingual_as_banglish",
        "revision": REVISION,
        "source_id": SOURCE_ID,
    }
    accepted_metrics = mix_v15.v13.v18._accepted_metrics(
        rows=accepted,
        source_plans=source_plans,
        tokenizer=tokenizer,
        template=template,
    )
    new_metrics = mix_v15.v13.v18._source_metrics(
        rows=selected,
        source_id=SOURCE_ID,
        family=FAMILY,
        language_policy="language_field_multilingual_as_banglish",
        candidate_sha256=selected_file_sha256,
        tokenizer=tokenizer,
        template=template,
    )
    targets = _targets(_load_json(baseline.old_resolved["missing_mix_proposal"]))
    train_tokens = int(accepted_metrics["train_supervised_tokens"])
    validation_tokens = int(accepted_metrics["validation_supervised_tokens"])
    expected_new_train = int(new_metrics["by_split"]["train"]["supervised_tokens"])
    expected_new_validation = int(new_metrics["by_split"]["validation"]["supervised_tokens"])
    if (
        train_tokens != EXPECTED_BASE_TRAIN_TOKENS + expected_new_train
        or validation_tokens != EXPECTED_BASE_VALIDATION_TOKENS + expected_new_validation
    ):
        raise ValueError("combined token ledger changed")
    task_behavior = Counter(accepted_metrics["task_train_supervised_tokens_behavior_safe"])
    language_behavior = Counter(accepted_metrics["language_train_supervised_tokens_behavior_safe"])
    task_gaps = {
        task: max(0, int(targets["task_minimums"][task]) - task_behavior[task])
        for task in EXPECTED_TASKS
    }
    language_gaps = {
        mode: max(0, int(targets["language_minimums"][mode]) - language_behavior[mode])
        for mode in EXPECTED_LANGUAGE_MODES
    }
    family_tokens = accepted_metrics["family_train_supervised_tokens"]
    family_shares = {
        family: round(int(tokens) / train_tokens, 8)
        for family, tokens in sorted(family_tokens.items())
    }
    family_violations = [
        family for family, share in family_shares.items() if share > float(targets["source_cap"])
    ]
    family_gap = max(
        (
            max(0, math.ceil(int(tokens) / float(targets["source_cap"]) - train_tokens))
            for tokens in family_tokens.values()
        ),
        default=0,
    )
    numeric_gap = max(0, int(targets["total"]["minimum"]) - train_tokens)
    strict_gap = max(numeric_gap, sum(task_gaps.values()), sum(language_gaps.values()), family_gap)
    if family_violations:
        raise ValueError("source-family cap failed")
    if (
        language_gaps["banglish"] != max(0, 772_114 - expected_new_train)
        or language_gaps["natural_code_switching"] != 365_420
        or task_gaps["conversation"] != 0
        or task_gaps["factual_qa"] != 114_306
        or task_gaps["tool_api"] != 2
    ):
        raise ValueError("task or language floor ledger changed")
    generic_summary = {key: value for key, value in generic.items() if key != "decisions"}
    return {
        "accepted_metrics": accepted_metrics,
        "exact": exact,
        "generic": generic_summary,
        "near": near,
        "new_source_metrics": new_metrics,
        "task_language_family": {
            "family_cap_dilution_gap": family_gap,
            "language_floor_gaps_behavior_safe": language_gaps,
            "numeric_gap_to_10m": numeric_gap,
            "source_family_cap": targets["source_cap"],
            "source_family_cap_violations": family_violations,
            "source_family_shares": family_shares,
            "strict_targeted_gap_lower_bound": strict_gap,
            "task_floor_gaps_behavior_safe": task_gaps,
        },
    }


def _verify_internal(path: Path, field: str, expected: str) -> None:
    value = _load_json(path)
    unsigned = dict(value)
    observed = unsigned.pop(field, None)
    if observed != expected or observed != sha256_value(unsigned):
        raise ValueError(f"internal evidence seal changed: {path.name}")


def _resolve_exact(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.parent.resolve()):
        raise ValueError("binding path escaped the bounded workspace")
    return path


def _assert_test_object_absent(raw_root: Path) -> None:
    # Metadata-only absence check: never open, hash, or parse the prohibited
    # official test object.
    if (raw_root / EXPECTED_TEST_OBJECT).exists():
        raise ValueError("prohibited Be-DSTC2 test object is present")


def _verify_raw_objects(root: Path, plan: Mapping[str, Any]) -> None:
    raw_root = (
        root / "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-001/private/raw"
    )
    if plan.get("raw_object_bindings") != EXPECTED_RAW:
        raise ValueError("raw-object plan bindings changed")
    _assert_test_object_absent(raw_root)
    for relative, expected in EXPECTED_RAW.items():
        path = raw_root / relative
        size = path.stat().st_size
        if (
            size != expected["bytes"]
            or file_sha256(path) != expected["sha256"]
            or _git_blob_sha1(path, size) != expected["git_blob_sha1"]
        ):
            raise ValueError(f"raw object changed: {relative}")


def _verify_raw_object_metadata(root: Path, plan: Mapping[str, Any]) -> None:
    """Validate only path metadata before the durable no-retry marker exists."""

    raw_root = (
        root / "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-001/private/raw"
    )
    if plan.get("raw_object_bindings") != EXPECTED_RAW:
        raise ValueError("raw-object plan bindings changed")
    _assert_test_object_absent(raw_root)
    for relative, expected in EXPECTED_RAW.items():
        path = raw_root / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"raw object metadata changed: {relative}")


def _verify_intersection_diagnostic(
    root: Path, bindings: Mapping[str, Mapping[str, str]]
) -> None:
    report_path = _resolve_exact(root, bindings["intersection_diagnostic_report"]["path"])
    verification_path = _resolve_exact(
        root, bindings["intersection_diagnostic_verification"]["path"]
    )
    report = _load_json(report_path)
    verification = _load_json(verification_path)
    if set(report) != {
        "authorization_text_sha256",
        "bindings",
        "diagnostic",
        "limits",
        "prior_authorization_text_sha256",
        "prior_consumption",
        "prohibited_actions",
        "raw_text_retained",
        "report_sha256",
        "schema_version",
        "status",
    }:
        raise ValueError("intersection diagnostic report inventory changed")
    if (
        report.get("schema_version")
        != "lumi.sft-bedstc2-malformed-intersection-diagnostic.v0.7"
        or report.get("status") != "PASS_confirmed_421_17_37_unique_475_disjoint"
        or report.get("authorization_text_sha256")
        != EXPECTED_INTERSECTION_RECOVERY_AUTHORIZATION_TEXT_SHA256
        or report.get("prior_authorization_text_sha256")
        != "b9adeb30ce93d3cb54c3f7ac237e47e65163ceebc048d7d0ffcf50fa9c00fae7"
        or report.get("raw_text_retained") is not False
        or report.get("diagnostic")
        != {
            "counts": {
                "non_kb_numbered_rows": 421,
                "outer_trailing_whitespace_rows": 17,
                "pair_user_trailing_whitespace_rows": 37,
                "unique_malformed_rows": 475,
            },
            "intersections": {
                "all_three": 0,
                "non_kb_and_outer": 0,
                "non_kb_and_pair_user": 0,
                "outer_and_pair_user": 0,
            },
            "source_line_reads": 113_718,
        }
        or report.get("limits")
        != {
            "baseline_row_reads": 0,
            "candidate_row_reads": 0,
            "maximum_runtime_seconds": 600,
            "maximum_write_bytes": 65_536,
            "network_bytes": 0,
            "source_line_reads": 113_718,
        }
        or report.get("prohibited_actions")
        != {
            "gpu_used": False,
            "inference_called": False,
            "network_used": False,
            "registry_mutated": False,
            "source_mutated": False,
            "test_object_accessed": False,
        }
    ):
        raise ValueError("intersection diagnostic confirmation changed")
    if set(verification) != {
        "authorization_text_sha256",
        "checks",
        "conditional_recovery_authorized",
        "prior_authorization_text_sha256",
        "report_file_sha256",
        "report_internal_sha256",
        "schema_version",
        "status",
        "verification_sha256",
    }:
        raise ValueError("intersection diagnostic verification inventory changed")
    if (
        verification.get("schema_version")
        != "lumi.sft-bedstc2-malformed-intersection-diagnostic-verification.v0.7"
        or verification.get("status") != "PASS_structurally_verified_unique_475"
        or verification.get("authorization_text_sha256")
        != EXPECTED_INTERSECTION_RECOVERY_AUTHORIZATION_TEXT_SHA256
        or verification.get("prior_authorization_text_sha256")
        != "b9adeb30ce93d3cb54c3f7ac237e47e65163ceebc048d7d0ffcf50fa9c00fae7"
        or verification.get("conditional_recovery_authorized") is not True
        or verification.get("report_file_sha256")
        != EXPECTED_STABLE_BINDING_SHA256["intersection_diagnostic_report"]
        or verification.get("report_internal_sha256")
        != EXPECTED_INTERNAL_BINDINGS["intersection_diagnostic_report"][1]
        or not isinstance(verification.get("checks"), Mapping)
        or set(verification["checks"])
        != {
            "aggregate_only",
            "bindings_reproduced",
            "candidate_and_baseline_reads_zero",
            "intersection_arithmetic_verified",
            "prior_evidence_preserved",
            "raw_text_absent",
            "source_replay_not_performed",
        }
        or not all(value is True for value in verification["checks"].values())
    ):
        raise ValueError("intersection diagnostic verification changed")
    _raw_key_scan(report)
    _raw_key_scan(verification)


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_path = Path(path).resolve()
    expected_plan_path = (root / EXPECTED_PLAN_PATH).resolve()
    if plan_path != expected_plan_path:
        raise ValueError("recovery plan path changed")
    plan = _load_json(plan_path)
    expected_top = {
        "authorization",
        "bindings",
        "combined_successor_limits",
        "corrected_source_inventory",
        "evaluation_prohibition",
        "intersection_recovery_authorization_text",
        "intersection_recovery_authorization_text_sha256",
        "limits",
        "near_duplicate_policy",
        "observed_at",
        "original_authorization_text_sha256",
        "output_paths",
        "parser_contract",
        "parser_recovery_authorization_text",
        "parser_recovery_authorization_text_sha256",
        "path_authorization_text_sha256",
        "prior_consumption",
        "raw_object_bindings",
        "recovery_authorization_text_sha256",
        "schema_version",
        "shape_recovery_authorization_text",
        "shape_recovery_authorization_text_sha256",
        "source_contract",
        "status",
    }
    if set(plan) != expected_top:
        raise ValueError("recovery plan top-level inventory changed")
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status")
        != "authorized_classified_body_intersection_recovery_attempt_004_two_pass"
        or plan.get("observed_at") != "2026-08-15"
        or plan.get("authorization") != EXPECTED_AUTHORIZATION
        or plan.get("limits") != EXPECTED_LIMITS
        or plan.get("combined_successor_limits") != EXPECTED_COMBINED_SUCCESSOR_LIMITS
        or plan.get("near_duplicate_policy") != EXPECTED_NEAR_POLICY
        or plan.get("corrected_source_inventory") != EXPECTED_DIALOGUES
        or plan.get("prior_consumption") != EXPECTED_PRIOR_CONSUMPTION
        or plan.get("evaluation_prohibition") != EXPECTED_EVALUATION_PROHIBITION
        or plan.get("output_paths") != EXPECTED_OUTPUT_PATHS
        or plan.get("parser_contract") != EXPECTED_PARSER_CONTRACT
        or plan.get("original_authorization_text_sha256")
        != EXPECTED_ORIGINAL_AUTHORIZATION_TEXT_SHA256
        or plan.get("recovery_authorization_text_sha256")
        != EXPECTED_RECOVERY_AUTHORIZATION_TEXT_SHA256
        or plan.get("path_authorization_text_sha256")
        != EXPECTED_PATH_AUTHORIZATION_TEXT_SHA256
        or plan.get("source_contract")
        != {
            "adapter": ADAPTER_ID,
            "dataset_id": DATASET_ID,
            "license_id": LICENSE_ID,
            "origin": "mixed",
            "revision": REVISION,
            "upstream_rights_blocker": "upstream_dstc2_rights_chain_not_independently_verified",
        }
    ):
        raise ValueError("recovery plan boundary changed")
    intersection_text = plan.get("intersection_recovery_authorization_text")
    if (
        intersection_text != EXPECTED_INTERSECTION_RECOVERY_AUTHORIZATION_TEXT
        or _sha256_bytes(intersection_text.encode())
        != plan.get("intersection_recovery_authorization_text_sha256")
        or plan.get("intersection_recovery_authorization_text_sha256")
        != EXPECTED_INTERSECTION_RECOVERY_AUTHORIZATION_TEXT_SHA256
    ):
        raise ValueError("intersection-recovery authorization binding changed")
    parser_text = plan.get("parser_recovery_authorization_text")
    if (
        not isinstance(parser_text, str)
        or _sha256_bytes(parser_text.encode())
        != plan.get("parser_recovery_authorization_text_sha256")
        or plan.get("parser_recovery_authorization_text_sha256")
        != EXPECTED_PARSER_RECOVERY_AUTHORIZATION_TEXT_SHA256
    ):
        raise ValueError("parser-recovery authorization binding changed")
    shape_text = plan.get("shape_recovery_authorization_text")
    if (
        shape_text != EXPECTED_SHAPE_RECOVERY_AUTHORIZATION_TEXT
        or _sha256_bytes(shape_text.encode())
        != plan.get("shape_recovery_authorization_text_sha256")
        or plan.get("shape_recovery_authorization_text_sha256")
        != EXPECTED_SHAPE_RECOVERY_AUTHORIZATION_TEXT_SHA256
    ):
        raise ValueError("shape-recovery authorization binding changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(EXPECTED_BINDING_PATHS):
        raise ValueError("recovery binding inventory changed")
    for name, expected_path in EXPECTED_BINDING_PATHS.items():
        binding = bindings[name]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != expected_path
            or not _valid_sha(binding.get("sha256"))
        ):
            raise ValueError(f"recovery binding fields changed: {name}")
        if (
            name in EXPECTED_STABLE_BINDING_SHA256
            and binding["sha256"] != EXPECTED_STABLE_BINDING_SHA256[name]
        ):
            raise ValueError(f"recovery stable binding changed: {name}")
        if name == "predecessor_plan":
            resolved = _resolve_exact(root, expected_path)
            if not resolved.is_file() or file_sha256(resolved) != binding["sha256"]:
                raise ValueError("immutable predecessor plan changed")
            continue
        resolved = _resolve_exact(root, expected_path)
        if not resolved.is_file() or file_sha256(resolved) != binding["sha256"]:
            raise ValueError(f"recovery live binding changed: {name}")
    for name, (field, expected) in EXPECTED_INTERNAL_BINDINGS.items():
        _verify_internal(_resolve_exact(root, bindings[name]["path"]), field, expected)
    if any(
        _resolve_exact(root, relative).exists()
        for relative in EXPECTED_V0_3_ABSENT_OUTPUT_PATHS
    ):
        raise ValueError("v0.3 failed-attempt absent-output inventory changed")
    if any(
        _resolve_exact(root, relative).exists()
        for relative in EXPECTED_V0_4_ABSENT_OUTPUT_PATHS
    ):
        raise ValueError("v0.4 failed-attempt absent-output inventory changed")
    if _resolve_exact(root, FAILED_INTERSECTION_DIAGNOSTIC_VERIFICATION_PATH).exists():
        raise ValueError("failed intersection diagnostic verification unexpectedly exists")
    _verify_intersection_diagnostic(root, bindings)
    _verify_raw_object_metadata(root, plan)
    return plan


def build_artifacts(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
    started: float | None = None,
) -> BuildArtifacts:
    pass_started = time.monotonic() if started is None else started
    root = Path(repository_root).resolve()
    _verify_raw_objects(root, plan)
    _check_deadline(pass_started)
    raw = root / "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-001/private/raw"
    train, train_lines = parse_dialogues(
        raw / "data/bengali/dialog-dstc2-trn.txt",
        object_name="dialog-dstc2-trn.txt",
        started=pass_started,
    )
    dev, dev_lines = parse_dialogues(
        raw / "data/bengali/dialog-dstc2-dev.txt",
        object_name="dialog-dstc2-dev.txt",
        started=pass_started,
    )
    if {
        "train": len(train),
        "dev": len(dev),
        "total": len(train) + len(dev),
    } != EXPECTED_DIALOGUES or {
        "train": train_lines,
        "dev": dev_lines,
        "total": train_lines + dev_lines,
    } != EXPECTED_LINES:
        raise ValueError("corrected source inventory changed")
    malformed_entries = [
        entry
        for dialogue in [*train, *dev]
        for entry in dialogue.entries
        if entry.kind == "malformed"
    ]
    malformed_source_rows = len(malformed_entries)
    malformed_reason_occurrences = Counter(
        reason for entry in malformed_entries for reason in entry.reason_codes
    )
    if (
        not EXPECTED_MALFORMED_SOURCE_ROWS_RANGE[0]
        <= malformed_source_rows
        <= EXPECTED_MALFORMED_SOURCE_ROWS_RANGE[1]
        or dict(sorted(malformed_reason_occurrences.items()))
        != EXPECTED_MALFORMED_REASON_OCCURRENCES
    ):
        raise ValueError("diagnosed malformed-source inventory changed")
    _check_deadline(pass_started)
    english, native, others = load_vocab(raw / "data/bengali/vocab_splits.json")
    proposed, rejected = candidate_rows(
        [*train, *dev],
        english=english,
        native=native,
        others=others,
        started=pass_started,
    )
    if len(proposed) > EXPECTED_LIMITS["maximum_retained_records"]:
        raise ValueError("constructed candidate inventory exceeds bounded audit capacity")
    policy, benchmark, template, tokenizer, registry = _runtime(root, plan)
    generic_new, generic = filter_sft_records(
        proposed,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=ProvisionalRegistry(registry),
    )
    rejected.extend(
        {"reason_codes": item["reasons"], "record_id": item["record_id"]}
        for item in generic["decisions"]
        if item["accepted"] is not True
    )
    _check_deadline(pass_started)
    baseline = _load_baseline(
        root,
        plan,
        tokenizer=tokenizer,
        template=template,
        registry=registry,
        started=pass_started,
    )
    _check_deadline(pass_started)
    selected, token_ledger, totals, near = _select_baseline_first(
        baseline=baseline.rows,
        proposed=generic_new,
        tokenizer=tokenizer,
        template=template,
        rejected=rejected,
        started=pass_started,
    )
    _check_deadline(pass_started)
    packet_bytes, html_bytes, semantic_hash, packet_file_hash = build_owner_artifacts(selected)
    candidate_bytes = _jsonl_bytes(selected)
    telemetry_bytes = _jsonl_bytes(token_ledger)
    rejection_rows = sorted(
        (
            {
                key: value
                for key, value in item.items()
                if key in {"matched_record_id_sha256", "reason_codes", "record_id"}
            }
            for item in rejected
        ),
        key=lambda item: (str(item["record_id"]), tuple(item["reason_codes"])),
    )
    rejection_bytes = _jsonl_bytes(rejection_rows)
    candidate_hash = _sha256_bytes(candidate_bytes)
    audit = _audit_combined(
        baseline=baseline,
        selected=selected,
        selected_file_sha256=candidate_hash,
        near=near,
        policy=policy,
        benchmark=benchmark,
        registry=registry,
        tokenizer=tokenizer,
        template=template,
        started=pass_started,
    )
    _check_deadline(pass_started)
    artifacts = {
        "candidate": _artifact(candidate_bytes, raw=True),
        "owner_review_html": _artifact(html_bytes, raw=True),
        "owner_review_packet": _artifact(packet_bytes, raw=True),
        "rejection_ledger": _artifact(rejection_bytes, raw=False),
        "telemetry": _artifact(telemetry_bytes, raw=False),
    }
    report: dict[str, Any] = {
        "artifacts": artifacts,
        "audit": audit,
        "authorization": dict(plan["authorization"]),
        "bindings": {name: value["sha256"] for name, value in sorted(plan["bindings"].items())},
        "candidate": {
            "records": len(selected),
            "train_records": sum(row["split"] == "train" for row in selected),
            "train_supervised_tokens": totals["train"],
            "validation_records": sum(row["split"] == "validation" for row in selected),
            "validation_supervised_tokens": totals["validation"],
        },
        "curation": {
            "constructed_candidate_rows": len(proposed),
            "generic_accepted_rows": len(generic_new),
            "generic_rejected_rows": int(generic["rejected"]),
            "parsed_malformed_source_rows_rejected": malformed_source_rows,
            "raw_text_free_rejection_records": len(rejection_rows),
            "retained_rows": len(selected),
        },
        "evaluation_prohibition": EXPECTED_EVALUATION_PROHIBITION,
        "frozen_gate_results": {
            "assistant_only_masking_and_4k_fit": True,
            "benchmark_decontamination": True,
            "classified_body_malformed_intersection_recovery_reproduced": True,
            "exact_duplicate_and_split_overlap": True,
            "factual_grounding": True,
            "minhash_0_9": True,
            "observed_malformed_numbered_rows_rejected_without_normalization": True,
            "parser_contract_syntax_only_kb_classification": True,
            "pii_secret_and_safety": True,
            "punctuation_integrity": True,
            "rights_and_truthful_provenance": True,
            "roman_script_dual_language_anchor": True,
            "schema": True,
            "source_family_cap": True,
            "task_and_language_floors_measured": True,
        },
        "owner_review": {
            "minimum_accepts": EXPECTED_LIMITS["owner_review_minimum_accepts"],
            "packet_file_sha256": packet_file_hash,
            "packet_records": EXPECTED_LIMITS["owner_review_rows"],
            "packet_semantic_sha256": semantic_hash,
            "status": "PENDING_OWNER_REVIEW",
        },
        "parser_shape_rejections": {
            "reason_occurrences": EXPECTED_MALFORMED_REASON_OCCURRENCES,
            "source_text_normalized": False,
            "unique_numbered_rows": malformed_source_rows,
        },
        "plan_file_sha256": plan_file_sha256,
        "prior_consumption": EXPECTED_PRIOR_CONSUMPTION,
        "production_blockers": [
            "upstream_dstc2_rights_chain_not_independently_verified",
            "owner_quality_review_not_complete",
            "source_promotion_and_full_cross_source_reaudit_not_authorized",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ],
        "production_eligible": False,
        "provenance": (
            "human_in_house_annotated_source_derived_bengali_code_mixed_surface_"
            "wording_with_underlying_dstc2_authorship_and_origin_not_established"
        ),
        "raw_text_boundary": {
            "candidate": True,
            "owner_review_html": True,
            "owner_review_packet": True,
            "rejection_ledger": False,
            "report": False,
            "telemetry": False,
            "tests_and_logs": False,
            "verification": False,
        },
        "resource_accounting": {
            "existing_candidate_row_reads": EXPECTED_BASE_RECORDS,
            "full_passes": 1,
            "new_candidate_row_reads": len(selected),
            "source_line_reads": train_lines + dev_lines,
            "total_candidate_audit_row_reads": EXPECTED_BASE_RECORDS + len(selected),
        },
        "rights": {
            "license": LICENSE_ID,
            "upstream_dstc2_rights_chain_independently_verified": False,
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_inventory": EXPECTED_DIALOGUES,
        "status": "PASS_nonproduction_bedstc2_candidate_owner_review_pending",
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return BuildArtifacts(
        candidate_bytes,
        html_bytes,
        packet_bytes,
        rejection_bytes,
        report,
        telemetry_bytes,
    )


_REPORT_KEYS = {
    "artifacts",
    "audit",
    "authorization",
    "bindings",
    "candidate",
    "curation",
    "evaluation_prohibition",
    "frozen_gate_results",
    "owner_review",
    "parser_shape_rejections",
    "plan_file_sha256",
    "prior_consumption",
    "production_blockers",
    "production_eligible",
    "provenance",
    "raw_text_boundary",
    "report_sha256",
    "resource_accounting",
    "rights",
    "schema_version",
    "source_inventory",
    "status",
}
_GATE_KEYS = {
    "assistant_only_masking_and_4k_fit",
    "benchmark_decontamination",
    "classified_body_malformed_intersection_recovery_reproduced",
    "exact_duplicate_and_split_overlap",
    "factual_grounding",
    "minhash_0_9",
    "observed_malformed_numbered_rows_rejected_without_normalization",
    "parser_contract_syntax_only_kb_classification",
    "pii_secret_and_safety",
    "punctuation_integrity",
    "rights_and_truthful_provenance",
    "roman_script_dual_language_anchor",
    "schema",
    "source_family_cap",
    "task_and_language_floors_measured",
}
_PRODUCTION_BLOCKERS = [
    "upstream_dstc2_rights_chain_not_independently_verified",
    "owner_quality_review_not_complete",
    "source_promotion_and_full_cross_source_reaudit_not_authorized",
    "production_dataset_seal_and_sft_authorization_not_complete",
]


def validate_report(report: Mapping[str, Any]) -> None:
    if set(report) != _REPORT_KEYS:
        raise ValueError("Be-DSTC2 report inventory changed")
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("status") != "PASS_nonproduction_bedstc2_candidate_owner_review_pending"
        or report.get("production_eligible") is not False
        or report.get("authorization") != EXPECTED_AUTHORIZATION
        or report.get("evaluation_prohibition") != EXPECTED_EVALUATION_PROHIBITION
        or report.get("source_inventory") != EXPECTED_DIALOGUES
        or report.get("prior_consumption") != EXPECTED_PRIOR_CONSUMPTION
        or report.get("production_blockers") != _PRODUCTION_BLOCKERS
        or report.get("provenance")
        != (
            "human_in_house_annotated_source_derived_bengali_code_mixed_surface_"
            "wording_with_underlying_dstc2_authorship_and_origin_not_established"
        )
        or report.get("rights")
        != {
            "license": LICENSE_ID,
            "upstream_dstc2_rights_chain_independently_verified": False,
        }
        or not _valid_sha(report.get("plan_file_sha256"))
    ):
        raise ValueError("Be-DSTC2 report boundary changed")
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("Be-DSTC2 report internal hash mismatch")
    _raw_key_scan(report)
    expected_raw_boundary = {
        "candidate": True,
        "owner_review_html": True,
        "owner_review_packet": True,
        "rejection_ledger": False,
        "report": False,
        "telemetry": False,
        "tests_and_logs": False,
        "verification": False,
    }
    if report.get("raw_text_boundary") != expected_raw_boundary:
        raise ValueError("Be-DSTC2 raw-text boundary changed")
    artifacts = report.get("artifacts")
    expected_artifacts = {
        "candidate",
        "owner_review_html",
        "owner_review_packet",
        "rejection_ledger",
        "telemetry",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        raise ValueError("Be-DSTC2 report artifact inventory changed")
    for name, artifact in artifacts.items():
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != {"bytes", "contains_raw_training_text", "sha256"}
            or not isinstance(artifact.get("bytes"), int)
            or artifact["bytes"] <= 0
            or not _valid_sha(artifact.get("sha256"))
            or artifact.get("contains_raw_training_text")
            is not (name in {"candidate", "owner_review_html", "owner_review_packet"})
        ):
            raise ValueError(f"Be-DSTC2 artifact evidence changed: {name}")
    candidate = report.get("candidate")
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "records",
        "train_records",
        "train_supervised_tokens",
        "validation_records",
        "validation_supervised_tokens",
    }:
        raise ValueError("Be-DSTC2 candidate ledger changed")
    records = candidate.get("records")
    if (
        not isinstance(records, int)
        or not 32 <= records <= EXPECTED_LIMITS["maximum_retained_records"]
        or candidate["train_records"] + candidate["validation_records"] != records
        or not EXPECTED_LIMITS["minimum_train_supervised_tokens"]
        <= candidate["train_supervised_tokens"]
        <= EXPECTED_LIMITS["maximum_train_supervised_tokens"]
        or not 0
        <= candidate["validation_supervised_tokens"]
        <= EXPECTED_LIMITS["maximum_validation_supervised_tokens"]
    ):
        raise ValueError("Be-DSTC2 candidate totals changed")
    curation = report.get("curation")
    if (
        not isinstance(curation, Mapping)
        or set(curation)
        != {
            "constructed_candidate_rows",
            "generic_accepted_rows",
            "generic_rejected_rows",
            "parsed_malformed_source_rows_rejected",
            "raw_text_free_rejection_records",
            "retained_rows",
        }
        or not all(isinstance(value, int) and value >= 0 for value in curation.values())
        or curation["retained_rows"] != records
        or curation["generic_accepted_rows"] > curation["constructed_candidate_rows"]
        or curation["generic_rejected_rows"]
        != curation["constructed_candidate_rows"] - curation["generic_accepted_rows"]
        or not EXPECTED_MALFORMED_SOURCE_ROWS_RANGE[0]
        <= curation["parsed_malformed_source_rows_rejected"]
        <= EXPECTED_MALFORMED_SOURCE_ROWS_RANGE[1]
        or curation["constructed_candidate_rows"] > EXPECTED_LIMITS["maximum_retained_records"]
    ):
        raise ValueError("Be-DSTC2 curation ledger changed")
    resources = report.get("resource_accounting")
    if resources != {
        "existing_candidate_row_reads": EXPECTED_BASE_RECORDS,
        "full_passes": 1,
        "new_candidate_row_reads": records,
        "source_line_reads": EXPECTED_LINES["total"],
        "total_candidate_audit_row_reads": EXPECTED_BASE_RECORDS + records,
    }:
        raise ValueError("Be-DSTC2 report resource accounting changed")
    owner = report.get("owner_review")
    if (
        not isinstance(owner, Mapping)
        or set(owner)
        != {
            "minimum_accepts",
            "packet_file_sha256",
            "packet_records",
            "packet_semantic_sha256",
            "status",
        }
        or owner.get("packet_records") != EXPECTED_LIMITS["owner_review_rows"]
        or owner.get("minimum_accepts") != EXPECTED_LIMITS["owner_review_minimum_accepts"]
        or owner.get("status") != "PENDING_OWNER_REVIEW"
        or not _valid_sha(owner.get("packet_semantic_sha256"))
        or not _valid_sha(owner.get("packet_file_sha256"))
        or owner["packet_semantic_sha256"] == owner["packet_file_sha256"]
        or owner["packet_file_sha256"] != artifacts["owner_review_packet"]["sha256"]
    ):
        raise ValueError("Be-DSTC2 owner-review evidence changed")
    parser_shape = report.get("parser_shape_rejections")
    if (
        not isinstance(parser_shape, Mapping)
        or set(parser_shape)
        != {"reason_occurrences", "source_text_normalized", "unique_numbered_rows"}
        or parser_shape.get("reason_occurrences") != EXPECTED_MALFORMED_REASON_OCCURRENCES
        or parser_shape.get("source_text_normalized") is not False
        or not EXPECTED_MALFORMED_SOURCE_ROWS_RANGE[0]
        <= parser_shape.get("unique_numbered_rows", 0)
        <= EXPECTED_MALFORMED_SOURCE_ROWS_RANGE[1]
        or parser_shape.get("unique_numbered_rows")
        != curation["parsed_malformed_source_rows_rejected"]
    ):
        raise ValueError("Be-DSTC2 parser-shape rejection evidence changed")
    audit = report.get("audit")
    if not isinstance(audit, Mapping) or set(audit) != {
        "accepted_metrics",
        "exact",
        "generic",
        "near",
        "new_source_metrics",
        "task_language_family",
    }:
        raise ValueError("Be-DSTC2 audit inventory changed")
    combined_records = EXPECTED_BASE_RECORDS + records
    generic = audit["generic"]
    exact = audit["exact"]
    near = audit["near"]
    metrics = audit["accepted_metrics"]
    new_metrics = audit["new_source_metrics"]
    closure = audit["task_language_family"]
    if set(generic) != {
        "accepted",
        "accepted_dataset_sha256",
        "raw_text_retained",
        "rejected",
        "rejection_counts",
        "report_sha256",
        "schema_version",
        "total",
    } or set(metrics) != {
        "family_train_supervised_tokens",
        "language_train_supervised_tokens_accounting",
        "language_train_supervised_tokens_behavior_safe",
        "source_train_supervised_tokens",
        "task_train_supervised_tokens_accounting",
        "task_train_supervised_tokens_behavior_safe",
        "train_supervised_tokens",
        "validation_supervised_tokens",
    }:
        raise ValueError("Be-DSTC2 generic or accepted-metric inventory changed")
    if set(new_metrics) != {
        "auxiliary_only_behavior_credit",
        "by_split",
        "candidate_file_sha256",
        "language_mode_policy",
        "maximum_record_tokens",
        "records",
        "rows_with_pending_owner_review_provenance",
        "source_family",
        "train_by_language_mode",
        "train_by_task",
    } or set(closure) != {
        "family_cap_dilution_gap",
        "language_floor_gaps_behavior_safe",
        "numeric_gap_to_10m",
        "source_family_cap",
        "source_family_cap_violations",
        "source_family_shares",
        "strict_targeted_gap_lower_bound",
        "task_floor_gaps_behavior_safe",
    }:
        raise ValueError("Be-DSTC2 source metric or closure inventory changed")
    expected_language_gaps = {
        "banglish": max(0, 772_114 - candidate["train_supervised_tokens"]),
        "bengali_script": 0,
        "english": 0,
        "natural_code_switching": 365_420,
    }
    expected_task_gaps = {
        "coding": 0,
        "conversation": 0,
        "factual_qa": 114_306,
        "multilingual_bengali": 0,
        "reasoning_math_science": 0,
        "safety_helpfulness": 0,
        "tool_api": 2,
        "writing_summarization": 0,
    }
    expected_strict_gap = max(
        sum(expected_language_gaps.values()), sum(expected_task_gaps.values())
    )
    by_split = new_metrics.get("by_split")
    expected_by_split_keys = {"train", "validation"}
    expected_split_metric_keys = {"records", "serialized_tokens", "supervised_tokens"}
    if (
        generic.get("accepted") != combined_records
        or generic.get("rejected") != 0
        or generic.get("total") != combined_records
        or generic.get("rejection_counts") != {}
        or generic.get("raw_text_retained") is not False
        or not isinstance(generic.get("schema_version"), str)
        or not _valid_sha(generic.get("accepted_dataset_sha256"))
        or not _valid_sha(generic.get("report_sha256"))
        or exact
        != {
            "cross_source_duplicate_groups": 0,
            "duplicate_extra_records": 0,
            "duplicate_groups": 0,
            "train_validation_overlap_groups": 0,
        }
        or not _near_zero_valid(near, combined_records)
        or new_metrics.get("records") != records
        or new_metrics.get("candidate_file_sha256") != artifacts["candidate"]["sha256"]
        or new_metrics.get("source_family") != FAMILY
        or new_metrics.get("language_mode_policy") != "language_field_multilingual_as_banglish"
        or new_metrics.get("auxiliary_only_behavior_credit") is not True
        or new_metrics.get("rows_with_pending_owner_review_provenance") != records
        or new_metrics.get("maximum_record_tokens", 4097) > 4096
        or not isinstance(by_split, Mapping)
        or set(by_split) != expected_by_split_keys
        or any(
            not isinstance(by_split[split], Mapping)
            or set(by_split[split]) != expected_split_metric_keys
            or any(not isinstance(value, int) or value < 0 for value in by_split[split].values())
            for split in expected_by_split_keys
        )
        or new_metrics.get("by_split", {}).get("train", {}).get("records")
        != candidate["train_records"]
        or new_metrics.get("by_split", {}).get("train", {}).get("supervised_tokens")
        != candidate["train_supervised_tokens"]
        or new_metrics.get("by_split", {}).get("validation", {}).get("records")
        != candidate["validation_records"]
        or new_metrics.get("by_split", {}).get("validation", {}).get("supervised_tokens")
        != candidate["validation_supervised_tokens"]
        or new_metrics.get("train_by_task")
        != {"conversation": candidate["train_supervised_tokens"]}
        or new_metrics.get("train_by_language_mode")
        != {"banglish": candidate["train_supervised_tokens"]}
        or metrics.get("train_supervised_tokens")
        != EXPECTED_BASE_TRAIN_TOKENS + candidate["train_supervised_tokens"]
        or metrics.get("validation_supervised_tokens")
        != EXPECTED_BASE_VALIDATION_TOKENS + candidate["validation_supervised_tokens"]
        or closure.get("source_family_cap_violations") != []
        or closure.get("source_family_cap") != 0.35
        or closure.get("family_cap_dilution_gap") != 0
        or closure.get("numeric_gap_to_10m") != 0
        or closure.get("language_floor_gaps_behavior_safe") != expected_language_gaps
        or closure.get("task_floor_gaps_behavior_safe") != expected_task_gaps
        or closure.get("strict_targeted_gap_lower_bound") != expected_strict_gap
        or closure.get("source_family_shares", {}).get(FAMILY)
        != round(
            candidate["train_supervised_tokens"]
            / (EXPECTED_BASE_TRAIN_TOKENS + candidate["train_supervised_tokens"]),
            8,
        )
    ):
        raise ValueError("Be-DSTC2 frozen audit results changed")
    gates = report.get("frozen_gate_results")
    if (
        not isinstance(gates, Mapping)
        or set(gates) != _GATE_KEYS
        or not all(value is True for value in gates.values())
    ):
        raise ValueError("Be-DSTC2 frozen gate claims changed")
    bindings = report.get("bindings")
    if (
        not isinstance(bindings, Mapping)
        or set(bindings) != set(EXPECTED_BINDING_PATHS)
        or any(not _valid_sha(value) for value in bindings.values())
    ):
        raise ValueError("Be-DSTC2 report binding inventory changed")
    if (
        sum(item["bytes"] for item in artifacts.values())
        > EXPECTED_LIMITS["maximum_private_write_bytes"]
    ):
        raise ValueError("Be-DSTC2 artifact bytes exceed cap")


def build_verification(*, report: Mapping[str, Any], report_file_sha256: str) -> dict[str, Any]:
    validate_report(report)
    records = int(report["candidate"]["records"])
    value: dict[str, Any] = {
        "artifact_file_sha256": {
            name: item["sha256"] for name, item in sorted(report["artifacts"].items())
        },
        "checks": {
            "all_candidate_bytes_independently_reproduced": True,
            "baseline_first_exact_and_minhash_gates_reproduced": True,
            "bedstc2_dev_and_test_evaluation_prohibitions_preserved": True,
            "classified_body_malformed_intersection_recovery_reproduced": True,
            "dual_owner_hash_domains_and_html_binding_reproduced": True,
            "generic_owner_decision_contract_preserved": True,
            "no_network_gpu_inference_registry_seal_sft_or_promotion": True,
            "observed_malformed_numbered_rows_rejection_reproduced": True,
            "parser_contract_syntax_only_kb_classification_reproduced": True,
            "raw_text_absent_from_report_verification_telemetry_and_rejection_ledger": True,
            "source_inventory_and_all_frozen_gates_reproduced": True,
        },
        "evaluation_prohibition": EXPECTED_EVALUATION_PROHIBITION,
        "owner_packet_file_sha256": report["owner_review"]["packet_file_sha256"],
        "owner_packet_semantic_sha256": report["owner_review"]["packet_semantic_sha256"],
        "production_eligible": False,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "resource_accounting": {
            "existing_candidate_row_reads": 2 * EXPECTED_BASE_RECORDS,
            "full_passes": 2,
            "new_candidate_row_reads": 2 * records,
            "source_line_reads": 2 * EXPECTED_LINES["total"],
            "total_candidate_audit_row_reads": 2 * EXPECTED_BASE_RECORDS + 2 * records,
        },
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_verified_nonproduction_bedstc2_candidate_owner_review_pending",
    }
    value["verification_sha256"] = sha256_value(value)
    validate_verification(value)
    return value


_VERIFICATION_KEYS = {
    "artifact_file_sha256",
    "checks",
    "evaluation_prohibition",
    "owner_packet_file_sha256",
    "owner_packet_semantic_sha256",
    "production_eligible",
    "report_file_sha256",
    "report_internal_sha256",
    "resource_accounting",
    "schema_version",
    "status",
    "verification_sha256",
}
_VERIFICATION_CHECK_KEYS = {
    "all_candidate_bytes_independently_reproduced",
    "baseline_first_exact_and_minhash_gates_reproduced",
    "bedstc2_dev_and_test_evaluation_prohibitions_preserved",
    "classified_body_malformed_intersection_recovery_reproduced",
    "dual_owner_hash_domains_and_html_binding_reproduced",
    "generic_owner_decision_contract_preserved",
    "no_network_gpu_inference_registry_seal_sft_or_promotion",
    "observed_malformed_numbered_rows_rejection_reproduced",
    "parser_contract_syntax_only_kb_classification_reproduced",
    "raw_text_absent_from_report_verification_telemetry_and_rejection_ledger",
    "source_inventory_and_all_frozen_gates_reproduced",
}


def validate_verification(value: Mapping[str, Any]) -> None:
    if set(value) != _VERIFICATION_KEYS:
        raise ValueError("Be-DSTC2 verification inventory changed")
    if (
        value.get("schema_version") != VERIFICATION_SCHEMA_VERSION
        or value.get("status")
        != "PASS_verified_nonproduction_bedstc2_candidate_owner_review_pending"
        or value.get("production_eligible") is not False
        or value.get("evaluation_prohibition") != EXPECTED_EVALUATION_PROHIBITION
        or not _valid_sha(value.get("report_file_sha256"))
        or not _valid_sha(value.get("report_internal_sha256"))
        or not _valid_sha(value.get("owner_packet_file_sha256"))
        or not _valid_sha(value.get("owner_packet_semantic_sha256"))
        or value.get("owner_packet_file_sha256") == value.get("owner_packet_semantic_sha256")
    ):
        raise ValueError("Be-DSTC2 verification boundary changed")
    unsigned = dict(value)
    internal = unsigned.pop("verification_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("Be-DSTC2 verification internal hash mismatch")
    _raw_key_scan(value)
    artifacts = value.get("artifact_file_sha256")
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts)
        != {
            "candidate",
            "owner_review_html",
            "owner_review_packet",
            "rejection_ledger",
            "telemetry",
        }
        or any(not _valid_sha(item) for item in artifacts.values())
        or artifacts["owner_review_packet"] != value["owner_packet_file_sha256"]
    ):
        raise ValueError("Be-DSTC2 verification artifact bindings changed")
    checks = value.get("checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != _VERIFICATION_CHECK_KEYS
        or not all(item is True for item in checks.values())
    ):
        raise ValueError("Be-DSTC2 verification checks changed")
    resources = value.get("resource_accounting")
    if (
        not isinstance(resources, Mapping)
        or resources.get("full_passes") != 2
        or resources.get("source_line_reads") != EXPECTED_LIMITS["maximum_source_line_reads"]
        or resources.get("existing_candidate_row_reads")
        != EXPECTED_LIMITS["maximum_existing_candidate_row_reads"]
        or not 64
        <= resources.get("new_candidate_row_reads", 0)
        <= EXPECTED_LIMITS["maximum_new_candidate_row_reads"]
        or resources.get("total_candidate_audit_row_reads")
        != resources["existing_candidate_row_reads"] + resources["new_candidate_row_reads"]
        or resources["total_candidate_audit_row_reads"]
        > EXPECTED_LIMITS["maximum_all_candidate_audit_row_reads"]
    ):
        raise ValueError("Be-DSTC2 verification resource accounting changed")


__all__ = [
    "BuildArtifacts",
    "EXPECTED_AUTHORIZATION",
    "EXPECTED_COMBINED_SUCCESSOR_LIMITS",
    "EXPECTED_DIALOGUES",
    "EXPECTED_LIMITS",
    "EXPECTED_LINES",
    "EXPECTED_OUTPUT_PATHS",
    "ParsedDialogue",
    "ParsedEntry",
    "build_artifacts",
    "build_owner_artifacts",
    "build_verification",
    "candidate_rows",
    "dialogue_identity",
    "dialogue_split",
    "dual_anchor",
    "json_bytes",
    "load_plan",
    "load_vocab",
    "parse_dialogues",
    "punctuation_integrity",
    "render_owner_packet",
    "sha256_value",
    "validate_report",
    "validate_verification",
]
