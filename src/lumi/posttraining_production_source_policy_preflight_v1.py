"""Raw-text-free production source-policy and data-contract preflight."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import posttraining_candidate_mix_reaudit_v18 as v18
from .posttraining_data import SFTPolicy, load_sft_policy
from .posttraining_sources import SFTSource, load_sft_source_registry


PLAN_SCHEMA_VERSION = "lumi.sft-production-source-policy-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-production-source-policy-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-production-source-policy-preflight-verification.v0.1"
)
PLAN_STATUS = "authorized_local_metadata_only_policy_preflight"
REPORT_STATUS = "PASS_policy_preflight_decision_required_nonproduction"
VERIFICATION_STATUS = "PASS_verified_policy_preflight_no_mutation"

EXPECTED_AUTHORIZATION = {
    "active_data_contract_change_authorized": False,
    "candidate_row_access_authorized": False,
    "dataset_seal_authorized": False,
    "gpu_authorized": False,
    "inference_authorized": False,
    "network_authorized": False,
    "policy_preflight_authorized": True,
    "production_promotion_authorized": False,
    "registry_mutation_authorized": False,
    "remote_access_authorized": False,
    "sft_training_authorized": False,
    "source_row_access_authorized": False,
    "test_object_access_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_candidate_row_reads": 0,
    "maximum_factual_passes": 2,
    "maximum_gpu_use": 0,
    "maximum_inference_calls": 0,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 1_048_576,
    "maximum_registry_or_data_contract_writes": 0,
    "maximum_retries_after_started_pass": 0,
    "maximum_runtime_minutes": 15,
    "maximum_source_row_reads": 0,
    "maximum_test_object_reads": 0,
    "maximum_training_actions": 0,
}
EXPECTED_OUTPUT = {
    "report_path": "evidence/lumi-4b-sft-production-source-policy-preflight-v0.1.json",
    "verification_path": (
        "evidence/lumi-4b-sft-production-source-policy-preflight-verification-v0.1.json"
    ),
}
EXPECTED_BINDING_NAMES = frozenset(
    {
        "candidate_mix_plan",
        "candidate_mix_report",
        "candidate_mix_verification",
        "data_contract",
        "implementation",
        "policy_loader",
        "production_seal",
        "registry_loader",
        "runner",
        "source_registry",
        "test",
        "verifier",
    }
)
EXPECTED_EXTERNAL_BINDINGS = {
    "candidate_mix_plan": (
        "configs/lumi-4b-sft-candidate-mix-reaudit-v0.18.json",
        "14e737fa9768c4527eb2663d5fda32d38121459591de44c83f899066034acb26",
    ),
    "candidate_mix_report": (
        "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.18.json",
        "3fa0502a3d9c064224a1c11fc1c8429c0d5ca13d7dd1be8d42d539a681a92000",
    ),
    "candidate_mix_verification": (
        "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.18.json",
        "43db682ea4f28577f0d919df6160d9fc7339e56df8469737abc0569201365651",
    ),
    "data_contract": (
        "configs/lumi-4b-sft-data-contract-v0.1.json",
        "9a283d9333d29deb7b516fa45eb4eec50232be9925f4ee4aa3a8f624213b2f06",
    ),
    "policy_loader": (
        "src/lumi/posttraining_data.py",
        "df8297e753e97028999c6bcd203edc2a00a02147f7d9ecb2255569c480363970",
    ),
    "production_seal": (
        "src/lumi/posttraining_seal.py",
        "52b9d93ceaf756497b7ccc4805b51d76e22490269a61cd644ce5a1b12e946155",
    ),
    "registry_loader": (
        "src/lumi/posttraining_sources.py",
        "10aa8e195bcab2613f3941f22f5303fc1cc8264403b4d7f20d04230bfcb40e50",
    ),
    "source_registry": (
        "configs/lumi-4b-sft-source-registry-v0.24.json",
        "a7a2daec58bc08e5a8e11ecff79cea3a5db90d9b423eb6a55cba28559d74a6bb",
    ),
}
EXPECTED_INVENTORY = {
    "accepted_dataset_sha256": (
        "1a60f602648f45480dc7827d0cbe686d0e17fb3b28e476dbb7ef21986e4fd37e"
    ),
    "included_sources": 20,
    "input_records": 329_200,
    "production_eligible_included_sources": 0,
    "registry_sources": 27,
    "train_records": 309_643,
    "train_supervised_tokens": 11_855_217,
    "validation_records": 19_557,
    "validation_supervised_tokens": 939_293,
}
EXPECTED_REGISTRY_STATUS_COUNTS = {
    "blocked": 4,
    "canary_eligible": 14,
    "candidate_eligible": 8,
    "planned": 1,
}
EXPECTED_INCLUDED_STATUS_COUNTS = {
    "canary_eligible": 13,
    "candidate_eligible": 7,
}
EXPECTED_INCLUDED_SOURCE_IDS = (
    "allenai_prosocial_dialog_safety_candidate",
    "amazon_massive_bn_bd_tool_canary",
    "banglabridge_instructions_closure_canary",
    "banglatlit_auxiliary_closure_canary",
    "coherelabs_aya_bengali_closure_canary",
    "coherelabs_aya_dataset",
    "deepmind_mathematics_dataset",
    "dmath",
    "fiscalnote_billsum_writing_candidate",
    "lumi_ai_assisted_gap_closure_canary_v0_1",
    "lumi_ai_assisted_gap_closure_scaled_v0_2",
    "lumi_ai_assisted_sft_v0_3",
    "lumi_first_party_gpt_oss_teacher_canary_v0_1",
    "lumi_first_party_multilingual_safety_tool_v0_6",
    "lumi_first_party_teacher_canary_v0_1",
    "lumi_safety_tool_sft_v0_1",
    "nvidia_nemotron_post_training_tool_canary",
    "nvidia_opencodeinstruct",
    "openassistant_oasst1",
    "openassistant_oasst2",
)
EXPECTED_EXCLUDED_SOURCE_IDS = (
    "bigcode_self_oss_instruct_50k",
    "huggingfaceh4_ultrachat_200k",
    "lumi_ai_assisted_sft_v0_1",
    "lumi_ai_assisted_sft_v0_2",
    "lumi_owned_sft_v0_1",
    "openr1_math_220k",
    "salesforce_xlam_function_calling_60k",
)
EXPECTED_PENDING_OWNER = {
    "banglabridge_instructions_closure_canary": {
        "records": 561,
        "train_supervised_tokens": 250_441,
    },
    "banglatlit_auxiliary_closure_canary": {
        "records": 13_576,
        "train_supervised_tokens": 300_000,
    },
    "coherelabs_aya_bengali_closure_canary": {
        "records": 673,
        "train_supervised_tokens": 225_756,
    },
    "fiscalnote_billsum_writing_candidate": {
        "records": 4_433,
        "train_supervised_tokens": 849_991,
    },
    "nvidia_nemotron_post_training_tool_canary": {
        "records": 1_228,
        "train_supervised_tokens": 770_663,
    },
}
EXPECTED_MACHINE_ASSISTED_NO_HUMAN_REVIEW = {
    "lumi_ai_assisted_gap_closure_canary_v0_1": {
        "human_review_completed_rows": 0,
        "model_assisted_rows": 1_000,
        "train_supervised_tokens": 86_712,
    },
    "lumi_ai_assisted_gap_closure_scaled_v0_2": {
        "human_review_completed_rows": 0,
        "model_assisted_rows": 13_420,
        "train_supervised_tokens": 1_157_905,
    },
}
EXPECTED_CRITICAL_BLOCKERS = {
    "fiscalnote_billsum_writing_candidate": (
        "source_provided_legislative_summary_authorship_not_established",
    ),
    "lumi_ai_assisted_gap_closure_scaled_v0_2": (
        "candidate_is_machine_only_ai_assisted_scaled_lane_pending_full_mix_reaudit",
        "full_registry_v0_24_cross_source_hygiene_and_floor_reaudit_not_complete",
        "production_data_contract_dataset_seal_and_sft_authorization_not_complete",
    ),
}

SUCCESSOR_AUTHORIZATION_TEXT = (
    "I approve one bounded local-only Lumi private-research non-production SFT data-contract "
    "v0.2 successor. Bind source registry v0.24 SHA-256 "
    "a7a2daec58bc08e5a8e11ecff79cea3a5db90d9b423eb6a55cba28559d74a6bb, "
    "candidate-mix v0.18 report SHA-256 "
    "3fa0502a3d9c064224a1c11fc1c8429c0d5ca13d7dd1be8d42d539a681a92000, "
    "verification SHA-256 "
    "43db682ea4f28577f0d919df6160d9fc7339e56df8469737abc0569201365651, "
    "and data contract v0.1 SHA-256 "
    "9a283d9333d29deb7b516fa45eb4eec50232be9925f4ee4aa3a8f624213b2f06. "
    "Create a versioned v0.2 contract and minimal backward-compatible policy-loader tests "
    "that bind the exact 20 audited candidate sources and exact seven exclusions. Permit "
    "truthfully labelled AI-assisted first-party rows only for a private non-production "
    "learning-candidate seal without human-authorship or completed-human-review claims. "
    "Preserve the SFT schema, allowed licenses/languages/splits, benchmark signature hash, "
    "4K reject policy, minimum quality 0.75, task weights, assistant/tool-only labels, "
    "single-conversation packing, all source provenance, and every evaluation exclusion. "
    "Keep every registry source production_eligible=false and do not create registry v0.25, "
    "read or mutate candidate rows, seal a dataset, run CPU/GPU/inference/training, access "
    "network/remote/test objects, or perform unrelated actions. Maximum two factual passes, "
    "15 local runtime-minutes, 4 MiB private writes, and zero retries after a started pass."
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RAW_TEXT_KEYS = frozenset(
    {
        "assistant_response",
        "content",
        "messages",
        "prompt",
        "raw_text",
        "response",
        "text",
        "user_prompt",
    }
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("policy-preflight binding must be a JSON object")
    return value


def _resolve_binding(root: Path, binding: Mapping[str, Any], name: str) -> Path:
    relative = binding.get("path")
    expected = binding.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise ValueError(f"invalid policy-preflight binding: {name}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"missing policy-preflight binding: {name}")
    if file_sha256(path) != expected:
        raise ValueError(f"policy-preflight binding changed: {name}")
    return path


def _assert_raw_text_free(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _RAW_TEXT_KEYS:
                raise ValueError(f"raw candidate text key forbidden at {path}")
            _assert_raw_text_free(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_raw_text_free(item, f"{path}[{index}]")


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("policy-preflight output already exists")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError("stale policy-preflight temporary exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_object(Path(path))
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported policy-preflight plan")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("policy-preflight plan is not authorized")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("policy-preflight authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("policy-preflight limits changed")
    if plan.get("expected") != EXPECTED_INVENTORY:
        raise ValueError("policy-preflight expected inventory changed")
    if plan.get("output") != EXPECTED_OUTPUT:
        raise ValueError("policy-preflight output boundary changed")
    text = plan.get("authorization_text")
    if (
        not isinstance(text, str)
        or plan.get("authorization_bytes") != len(text.encode("utf-8"))
        or plan.get("authorization_sha256")
        != hashlib.sha256(text.encode("utf-8")).hexdigest()
    ):
        raise ValueError("policy-preflight authorization text binding changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDING_NAMES:
        raise ValueError("policy-preflight binding set changed")
    for name, (expected_path, expected_sha) in EXPECTED_EXTERNAL_BINDINGS.items():
        if bindings.get(name) != {"path": expected_path, "sha256": expected_sha}:
            raise ValueError(f"policy-preflight external binding changed: {name}")
    for name, binding in bindings.items():
        _resolve_binding(root, binding, name)
    return plan


def _source_probe_is_blocked(
    *, source_value: Mapping[str, Any], policy: SFTPolicy
) -> bool:
    probe = copy.deepcopy(dict(source_value))
    probe["status"] = "reviewed"
    probe["production_eligible"] = True
    probe["production_blockers"] = []
    try:
        SFTSource.from_mapping(
            probe,
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
        )
    except ValueError:
        return True
    return False


def _policy_successor_path_is_blocked(contract: Mapping[str, Any]) -> bool:
    probe = copy.deepcopy(dict(contract))
    probe["source_registry"] = dict(probe["source_registry"])
    probe["source_registry"]["path"] = EXPECTED_EXTERNAL_BINDINGS["source_registry"][0]
    probe["source_registry"]["sha256"] = EXPECTED_EXTERNAL_BINDINGS["source_registry"][1]
    try:
        SFTPolicy.from_mapping(probe)
    except ValueError:
        return True
    return False


def build_report(
    *, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved = {
        name: _resolve_binding(root, binding, name)
        for name, binding in plan["bindings"].items()
    }
    mix_plan = _load_object(resolved["candidate_mix_plan"])
    mix_report = _load_object(resolved["candidate_mix_report"])
    mix_verification = _load_object(resolved["candidate_mix_verification"])
    contract = _load_object(resolved["data_contract"])
    registry_value = _load_object(resolved["source_registry"])

    v18.validate_report(mix_report)
    v18.validate_verification(mix_verification)
    if (
        mix_plan.get("schema_version")
        != "lumi.sft-candidate-mix-reaudit-plan.v0.18"
        or mix_report.get("report_sha256")
        != "7bce1bd1bcaf5b804ba5f8da5cb2614836fe8a398028079fae498131d310858e"
        or mix_verification.get("verification_sha256")
        != "162f2e8d803426921c1457ae9f5127e978be13120b54f13301cb47d53d6176cc"
        or mix_verification.get("report_file_sha256")
        != EXPECTED_EXTERNAL_BINDINGS["candidate_mix_report"][1]
        or mix_verification.get("resource_accounting", {}).get("candidate_row_reads")
        != 658_400
    ):
        raise ValueError("candidate-mix v0.18 evidence changed")

    policy = load_sft_policy(resolved["data_contract"])
    registry = load_sft_source_registry(
        resolved["source_registry"],
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=EXPECTED_EXTERNAL_BINDINGS["source_registry"][1],
    )
    included = tuple(sorted(mix_report["source_metrics"]))
    excluded = tuple(
        mix_report["excluded_source_gate"]["registry_sources_not_in_candidate_inventory"]
    )
    if included != EXPECTED_INCLUDED_SOURCE_IDS or excluded != EXPECTED_EXCLUDED_SOURCE_IDS:
        raise ValueError("policy-preflight source inventory changed")
    if set(included) | set(excluded) != set(registry.by_id):
        raise ValueError("policy-preflight registry partition changed")

    status_counts = dict(sorted(Counter(source.status for source in registry.sources).items()))
    included_status_counts = dict(
        sorted(Counter(registry.by_id[source_id].status for source_id in included).items())
    )
    production_eligible = sorted(
        source.source_id for source in registry.sources if source.production_eligible
    )
    reviewed = sorted(source.source_id for source in registry.sources if source.status == "reviewed")
    if (
        len(registry.sources) != EXPECTED_INVENTORY["registry_sources"]
        or status_counts != EXPECTED_REGISTRY_STATUS_COUNTS
        or included_status_counts != EXPECTED_INCLUDED_STATUS_COUNTS
        or production_eligible
        or reviewed
    ):
        raise ValueError("policy-preflight registry eligibility changed")

    pool = mix_report["candidate_pool"]
    generic = mix_report["generic_sft_validation"]
    exact = mix_report["exact_duplicate_audit"]
    near = mix_report["near_duplicate_audit"]
    closure = mix_report["closure"]
    train_records = sum(
        int(metrics["by_split"]["train"]["records"])
        for metrics in mix_report["source_metrics"].values()
    )
    validation_records = sum(
        int(metrics["by_split"]["validation"]["records"])
        for metrics in mix_report["source_metrics"].values()
    )
    if (
        generic.get("accepted_dataset_sha256")
        != EXPECTED_INVENTORY["accepted_dataset_sha256"]
        or generic.get("accepted") != EXPECTED_INVENTORY["input_records"]
        or generic.get("rejected") != 0
        or exact.get("duplicate_groups") != 0
        or exact.get("train_validation_overlap_groups") != 0
        or near.get("edge_count") != 0
        or near.get("multi_record_components") != 0
        or pool.get("train_supervised_tokens")
        != EXPECTED_INVENTORY["train_supervised_tokens"]
        or pool.get("validation_supervised_tokens")
        != EXPECTED_INVENTORY["validation_supervised_tokens"]
        or train_records != EXPECTED_INVENTORY["train_records"]
        or validation_records != EXPECTED_INVENTORY["validation_records"]
        or any(closure["task_floor_gaps_behavior_safe"].values())
        or any(closure["language_floor_gaps_behavior_safe"].values())
        or closure.get("source_family_cap_violations")
    ):
        raise ValueError("policy-preflight clean mix projection changed")

    pending_owner: dict[str, dict[str, int]] = {}
    for source_id, metrics in mix_report["source_metrics"].items():
        pending = int(metrics.get("rows_with_pending_owner_review_provenance", 0))
        if pending:
            pending_owner[source_id] = {
                "records": pending,
                "train_supervised_tokens": int(
                    metrics["by_split"]["train"]["supervised_tokens"]
                ),
            }
    if pending_owner != EXPECTED_PENDING_OWNER:
        raise ValueError("policy-preflight pending-owner inventory changed")

    machine_assisted: dict[str, dict[str, int]] = {}
    for source_id in EXPECTED_MACHINE_ASSISTED_NO_HUMAN_REVIEW:
        metrics = mix_report["source_metrics"][source_id]
        machine_assisted[source_id] = {
            "human_review_completed_rows": int(metrics["human_review_completed_rows"]),
            "model_assisted_rows": int(metrics["model_assisted_rows"]),
            "train_supervised_tokens": int(
                metrics["by_split"]["train"]["supervised_tokens"]
            ),
        }
    if machine_assisted != EXPECTED_MACHINE_ASSISTED_NO_HUMAN_REVIEW:
        raise ValueError("policy-preflight machine-assisted inventory changed")

    registry_by_id = {item["source_id"]: item for item in registry_value["sources"]}
    for source_id, blockers in EXPECTED_CRITICAL_BLOCKERS.items():
        observed = tuple(registry_by_id[source_id]["production_blockers"])
        if any(blocker not in observed for blocker in blockers):
            raise ValueError("policy-preflight critical blocker changed")
    notes = contract.get("notes", {})
    if (
        notes.get("ai_generated_records_require_explicit_generator_and_human_review_status")
        is not True
        or notes.get("ai_generated_records_require_owner_gate_before_production_promotion")
        is not True
    ):
        raise ValueError("policy-preflight AI owner-gate contract changed")

    scaled_value = registry_by_id["lumi_ai_assisted_gap_closure_scaled_v0_2"]
    if not _policy_successor_path_is_blocked(contract):
        raise ValueError("policy loader no longer requires a deliberate successor")
    if not _source_probe_is_blocked(source_value=scaled_value, policy=policy):
        raise ValueError("source status loader no longer requires a deliberate successor")

    first_party_ids = tuple(
        source_id for source_id in included if registry.by_id[source_id].access == "first_party"
    )
    first_party_records = sum(
        int(mix_report["source_metrics"][source_id]["records"])
        for source_id in first_party_ids
    )
    first_party_train_tokens = sum(
        int(
            mix_report["source_metrics"][source_id]["by_split"]["train"][
                "supervised_tokens"
            ]
        )
        for source_id in first_party_ids
    )
    if (
        len(first_party_ids) != 7
        or first_party_records != 16_284
        or first_party_train_tokens != 1_359_470
    ):
        raise ValueError("policy-preflight first-party projection changed")

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": REPORT_STATUS,
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "authorization": {
            "bytes": plan["authorization_bytes"],
            "sha256": plan["authorization_sha256"],
        },
        "input_bindings": {
            name: binding["sha256"] for name, binding in sorted(plan["bindings"].items())
        },
        "audited_mix": {
            "accepted_dataset_sha256": generic["accepted_dataset_sha256"],
            "exact_duplicate_groups": 0,
            "included_source_ids": list(included),
            "near_duplicate_edges": 0,
            "records": generic["accepted"],
            "rejected_records": 0,
            "strict_gap": pool["strict_targeted_gap_lower_bound"],
            "train_records": train_records,
            "train_supervised_tokens": pool["train_supervised_tokens"],
            "validation_records": validation_records,
            "validation_supervised_tokens": pool["validation_supervised_tokens"],
        },
        "registry_inventory": {
            "excluded_source_ids": list(excluded),
            "included_status_counts": included_status_counts,
            "production_eligible_included_source_ids": [],
            "production_eligible_registry_source_ids": production_eligible,
            "registry_source_count": len(registry.sources),
            "reviewed_registry_source_ids": reviewed,
            "status_counts": status_counts,
        },
        "unresolved_release_evidence": {
            "bill_sum": {
                "records": mix_report["source_metrics"][
                    "fiscalnote_billsum_writing_candidate"
                ]["records"],
                "source_summary_authorship_established": False,
                "train_writing_supervised_tokens": mix_report["source_metrics"][
                    "fiscalnote_billsum_writing_candidate"
                ]["by_split"]["train"]["supervised_tokens"],
            },
            "first_party_synthetic": {
                "records": first_party_records,
                "source_ids": list(first_party_ids),
                "train_supervised_tokens": first_party_train_tokens,
            },
            "machine_assisted_zero_human_review": {
                "by_source": machine_assisted,
                "human_review_completed_rows": 0,
                "model_assisted_rows": sum(
                    value["model_assisted_rows"] for value in machine_assisted.values()
                ),
                "train_supervised_tokens": sum(
                    value["train_supervised_tokens"] for value in machine_assisted.values()
                ),
            },
            "pending_owner_review_provenance": {
                "by_source": pending_owner,
                "records": sum(value["records"] for value in pending_owner.values()),
                "train_supervised_tokens": sum(
                    value["train_supervised_tokens"] for value in pending_owner.values()
                ),
            },
        },
        "loader_contract": {
            "current_policy_loader_accepts_registry_v0_24_path": False,
            "current_source_loader_can_truthfully_mark_first_party_synthetic_reviewed": False,
            "current_production_seal_rejects_nonproduction_sources": True,
            "minimal_backward_compatible_successor_required": True,
        },
        "decision": {
            "active_data_contract_v0_2_materialization_permitted_by_this_gate": False,
            "private_research_nonproduction_successor_ready_for_explicit_authorization": True,
            "production_contract_activation_permitted_now": False,
            "registry_v0_25_materialization_permitted_by_this_gate": False,
            "recommended_path": "private_research_nonproduction_data_contract_v0_2",
        },
        "production_blockers": [
            "zero_registry_sources_currently_production_eligible",
            "current_contract_requires_ai_owner_gate",
            "machine_assisted_rows_have_zero_completed_human_review",
            "pending_owner_review_provenance_rows_remain",
            "billsum_source_summary_authorship_not_established",
            "policy_loader_requires_versioned_registry_path_successor",
            "source_status_schema_requires_truthful_ai_review_successor",
            "production_seal_and_sft_not_authorized",
        ],
        "successor": {
            "exact_authorization_text": SUCCESSOR_AUTHORIZATION_TEXT,
            "production_path_requirements": [
                "explicit_owner_policy_for_machine_assisted_rows",
                "qualified_resolution_or_replacement_of_billsum_authorship_blocker",
                "resolution_or_explicit_policy_for_pending_owner_review_provenance",
                "truthful_reviewed_ai_source_status_schema",
                "immutable_registry_v0_25_followed_by_fresh_full_mix_reaudit",
                "active_production_data_contract_v0_2_bound_to_that_registry",
            ],
        },
        "resource_accounting": {
            "candidate_row_reads": 0,
            "factual_passes": 1,
            "gpu_use": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "registry_or_data_contract_writes": 0,
            "source_row_reads": 0,
            "test_object_reads": 0,
            "training_actions": 0,
        },
        "raw_training_text_retained": False,
        "production_eligible": False,
        "next_gate": {
            "action": "explicit_private_research_contract_or_production_governance_choice",
            "authorization_required": True,
            "recommended_choice": "private_research_nonproduction_data_contract_v0_2",
        },
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    unresolved = report.get("unresolved_release_evidence", {})
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("status") != REPORT_STATUS
        or report.get("report_sha256") != sha256_value(unsigned)
        or not _SHA256.fullmatch(str(report.get("plan_file_sha256", "")))
        or report.get("production_eligible") is not False
        or report.get("raw_training_text_retained") is not False
        or report.get("decision")
        != {
            "active_data_contract_v0_2_materialization_permitted_by_this_gate": False,
            "private_research_nonproduction_successor_ready_for_explicit_authorization": True,
            "production_contract_activation_permitted_now": False,
            "registry_v0_25_materialization_permitted_by_this_gate": False,
            "recommended_path": "private_research_nonproduction_data_contract_v0_2",
        }
        or report.get("registry_inventory", {}).get("registry_source_count") != 27
        or report.get("registry_inventory", {}).get(
            "production_eligible_registry_source_ids"
        )
        != []
        or report.get("audited_mix", {}).get("records") != 329_200
        or report.get("audited_mix", {}).get("strict_gap") != 0
        or unresolved.get("pending_owner_review_provenance", {}).get("records")
        != 20_471
        or unresolved.get("pending_owner_review_provenance", {}).get(
            "train_supervised_tokens"
        )
        != 2_396_851
        or unresolved.get("machine_assisted_zero_human_review", {}).get(
            "model_assisted_rows"
        )
        != 14_420
        or unresolved.get("machine_assisted_zero_human_review", {}).get(
            "human_review_completed_rows"
        )
        != 0
        or report.get("successor", {}).get("exact_authorization_text")
        != SUCCESSOR_AUTHORIZATION_TEXT
        or report.get("resource_accounting")
        != {
            "candidate_row_reads": 0,
            "factual_passes": 1,
            "gpu_use": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "registry_or_data_contract_writes": 0,
            "source_row_reads": 0,
            "test_object_reads": 0,
            "training_actions": 0,
        }
    ):
        raise ValueError("policy-preflight report boundary changed")
    _assert_raw_text_free(report)


def build_verification(
    *, report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_report(report)
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "checks": {
            "all_candidate_and_source_row_reads_are_zero": True,
            "all_twenty_included_sources_remain_nonproduction": True,
            "billsum_authorship_blocker_preserved": True,
            "candidate_mix_v0_18_clean_aggregate_reproduced": True,
            "current_loader_and_seal_contract_blockers_reproduced": True,
            "machine_assisted_and_pending_owner_counts_reproduced": True,
            "no_registry_or_active_data_contract_mutation": True,
            "no_seal_gpu_inference_or_training": True,
            "private_research_successor_requires_explicit_authorization": True,
            "raw_training_text_absent": True,
        },
        "resource_accounting": {
            "candidate_row_reads": 0,
            "factual_passes": 2,
            "gpu_use": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "registry_or_data_contract_writes": 0,
            "source_row_reads": 0,
            "test_object_reads": 0,
            "training_actions": 0,
        },
        "production_eligible": False,
        "next_gate": report["next_gate"],
    }
    verification["verification_sha256"] = sha256_value(verification)
    validate_verification(verification)
    return verification


def validate_verification(verification: Mapping[str, Any]) -> None:
    unsigned = {
        key: value for key, value in verification.items() if key != "verification_sha256"
    }
    checks = verification.get("checks")
    if (
        verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION
        or verification.get("status") != VERIFICATION_STATUS
        or verification.get("verification_sha256") != sha256_value(unsigned)
        or not _SHA256.fullmatch(str(verification.get("report_file_sha256", "")))
        or not _SHA256.fullmatch(str(verification.get("report_internal_sha256", "")))
        or verification.get("production_eligible") is not False
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
        or verification.get("resource_accounting", {}).get("factual_passes") != 2
        or verification.get("resource_accounting", {}).get("candidate_row_reads") != 0
        or verification.get("resource_accounting", {}).get(
            "registry_or_data_contract_writes"
        )
        != 0
    ):
        raise ValueError("policy-preflight verification boundary changed")
    _assert_raw_text_free(verification)


def preflight_output_paths(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    report = (root / plan["output"]["report_path"]).resolve()
    verification = (root / plan["output"]["verification_path"]).resolve()
    evidence_root = (root / "evidence").resolve()
    if not report.is_relative_to(evidence_root) or not verification.is_relative_to(
        evidence_root
    ):
        raise ValueError("policy-preflight output escaped evidence root")
    if report.exists() or verification.exists():
        raise FileExistsError("policy-preflight pass already started")
    return report, verification


__all__ = [
    "EXPECTED_LIMITS",
    "REPORT_STATUS",
    "SUCCESSOR_AUTHORIZATION_TEXT",
    "VERIFICATION_STATUS",
    "_assert_raw_text_free",
    "_write_exclusive",
    "build_report",
    "build_verification",
    "file_sha256",
    "json_file_bytes",
    "load_plan",
    "preflight_output_paths",
    "sha256_value",
    "validate_report",
    "validate_verification",
]
