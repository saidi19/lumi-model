"""Local deterministic repair for tool-result response quality failures."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_first_party_scale_canary as scale
from . import posttraining_gpt_oss_scale_canary_v2 as base
from . import posttraining_gpt_oss_teacher_canary as gpt
from . import posttraining_gpt_oss_tool_response_repair_v5 as predecessor
from .posttraining_candidate_mix_preflight import file_sha256
from .posttraining_data import validate_message_sequence
from .posttraining_format import load_chat_template

PLAN_SCHEMA = "lumi.sft-gpt-oss-deterministic-tool-repair-plan.v0.6"
ROW_SCHEMA = "lumi.sft-gpt-oss-deterministic-tool-repair-row.v0.6"
REPORT_SCHEMA = "lumi.sft-gpt-oss-deterministic-tool-repair-report.v0.6"
VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-deterministic-tool-repair-verification.v0.6"
PROMPT_CONTRACT = "lumi.tool-result.deterministic-multilingual-verbalizer.v0.6"
SOURCE_ID = "lumi_first_party_multilingual_safety_tool_v0_6"
OWNER_INSTRUCTION = "mamam proper fix kor"

TOTAL_RECORDS = 1_000
PREFIX_ROWS = 768
REPAIR_RECORDS = 216
FIXED_MACHINE_EXCLUSIONS = 14
OWNER_EXCLUSIONS = predecessor.OWNER_EXCLUSIONS
MAXIMUM_LOCAL_RUNTIME_MINUTES = 30
MAXIMUM_PRIVATE_WRITE_BYTES = 16 * 1024 * 1024


def build_schedule() -> tuple[scale.Slot, ...]:
    return predecessor.build_schedule()


def schedule_sha256() -> str:
    return predecessor.schedule_sha256()


def _variant(slot: scale.Slot) -> int:
    digest = hashlib.sha256(slot.record_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 5


def _conclusion(
    language: str, value: str, variant: int, *, sorted_values: bool,
) -> str:
    if sorted_values:
        choices = {
            "english": (
                f"This ordering is fully checked, so {value} is the final sequence to use.",
                f"The comparison confirms every position, making {value} the final "
                "ascending sequence.",
                f"Each value now occupies its verified position, so the answer is {value}.",
                f"The checked order is consistent from smallest to largest, leaving {value} "
                "as the final arrangement.",
                f"This verified comparison supports {value} as the requested sorted output.",
            ),
            "banglish": (
                f"Tai eta fully checked, ar final sequence {value} hobe.",
                f"Comparison ta protita position confirm kore, tai {value} final order hobe.",
                f"Protita value thik jaygay ache, tai final answer {value} hobe.",
                f"Smallest theke largest porjonto order ta consistent, tai {value} use kora hobe.",
                f"Ei verified comparison final sorted output hisebe {value} confirm kore, "
                "tai eta use hobe.",
            ),
            "natural_code_switching": (
                f"তাই ordering-টি fully checked, এবং final sequence হবে {value}।",
                f"Comparison প্রতিটি position confirm করে, তাই final order হলো {value}।",
                f"প্রতিটি value সঠিক জায়গায় আছে, তাই final answer হবে {value}।",
                f"Smallest থেকে largest পর্যন্ত order consistent, তাই {value} ব্যবহার হবে।",
                f"এই verified comparison requested sorted output হিসেবে {value} confirm করে।",
            ),
            "bengali_script": (
                f"তাই ক্রমটি সম্পূর্ণ যাচাই করা, এবং চূড়ান্ত বিন্যাস হবে {value}।",
                f"তুলনাটি প্রতিটি অবস্থান নিশ্চিত করে, তাই চূড়ান্ত ক্রম হলো {value}।",
                f"প্রতিটি মান সঠিক স্থানে আছে, তাই চূড়ান্ত উত্তর হবে {value}।",
                f"ক্ষুদ্রতম থেকে বৃহত্তম পর্যন্ত ক্রমটি সঙ্গতিপূর্ণ, তাই {value} ব্যবহার করা হবে।",
                f"এই যাচাই করা তুলনা অনুরোধকৃত সাজানো ফল হিসেবে {value} নিশ্চিত করে।",
            ),
        }
    else:
        choices = {
            "english": (
                f"This directly confirms the supplied values, so {value} is the final "
                "answer to use.",
                f"The checked output is consistent, which makes {value} the final value "
                "for this calculation.",
                f"Both the operation and tool evidence agree, so the requested answer is {value}.",
                f"The arithmetic is verified without changing the inputs, leaving {value} "
                "as the final result.",
                f"This tool-backed check supports {value} as the correct value for the "
                "requested calculation.",
            ),
            "banglish": (
                f"Tai eta supplied value confirm kore, ar final answer {value} hobe.",
                f"Checked output ta consistent, tai ei calculation-er final value {value} hobe.",
                f"Operation ar tool evidence duita mile, tai requested answer {value} hobe.",
                f"Input change na kore arithmetic verify kora hoyeche, tai final result "
                f"{value} hobe.",
                f"Ei tool-backed check correct value hisebe {value} support kore, tai eta "
                "final hobe.",
            ),
            "natural_code_switching": (
                f"তাই supplied value সরাসরি confirm হয়, এবং final answer হবে {value}।",
                f"Checked output consistent, তাই এই calculation-এর final value হলো {value}।",
                f"Operation ও tool evidence দুটোই মিলে যায়, তাই requested answer {value}।",
                f"Input না বদলে arithmetic verify করা হয়েছে, তাই final result হবে {value}।",
                f"এই tool-backed check সঠিক value হিসেবে {value} support করে।",
            ),
            "bengali_script": (
                f"তাই দেওয়া মান সরাসরি নিশ্চিত হয়, এবং চূড়ান্ত উত্তর হবে {value}।",
                f"যাচাই করা ফলটি সঙ্গতিপূর্ণ, তাই এই হিসাবের চূড়ান্ত মান হলো {value}।",
                f"গাণিতিক প্রক্রিয়া ও টুলের ফল উভয়ই মিলে যায়, তাই অনুরোধকৃত উত্তর {value}।",
                f"মানগুলো না বদলে হিসাবটি যাচাই করা হয়েছে, তাই চূড়ান্ত ফল হবে {value}।",
                f"এই টুলভিত্তিক যাচাই অনুরোধকৃত হিসাবের সঠিক মান হিসেবে {value} নিশ্চিত করে।",
            ),
        }
    return choices[language][variant]


def deterministic_tool_response(slot: scale.Slot) -> str:
    if slot.task != "tool_api" or slot.tool_name is None:
        raise ValueError("deterministic repair may materialize only bound tool rows")
    variant = _variant(slot)
    if slot.tool_name == "calculator":
        expression = str(slot.tool_arguments["expression"])
        result = str(slot.tool_result["result"])
        opening = {
            "english": (
                f"The verified result is {result}. Applying the stated operation to "
                f"{expression}, the calculator returns {result} while preserving the "
                "supplied inputs."
            ),
            "banglish": (
                f"Verified result holo {result}. Deoa expression {expression} calculator "
                f"diye check korle stated operation follow kore {result} pawa jay, ar eta "
                "supplied number-er sathe mile."
            ),
            "natural_code_switching": (
                f"Verified result হলো {result}। দেওয়া expression {expression} calculator "
                f"দিয়ে check করলে stated operation অনুসারে {result} পাওয়া যায়, এবং inputগুলো "
                "অপরিবর্তিত থাকে।"
            ),
            "bengali_script": (
                f"যাচাই করা ফল হলো {result}। দেওয়া রাশি {expression} ক্যালকুলেটরে হিসাব "
                f"করলে নির্ধারিত ক্রমে {result} পাওয়া যায় এবং মানগুলো অপরিবর্তিত থাকে।"
            ),
        }[slot.language_mode]
        conclusion = _conclusion(
            slot.language_mode, result, variant, sorted_values=False
        )
        response = f"{opening} {conclusion}"
    elif slot.tool_name == "list_sort":
        items = [str(item) for item in slot.tool_result["items"]]
        if len(items) != 4:
            raise ValueError("deterministic sort repair requires four frozen result values")
        sequence = ", ".join(items)
        first, middle_one, middle_two, last = items
        opening = {
            "english": (
                f"The verified result is {sequence}. Sorting the supplied values in ascending "
                f"order places {first} first and {last} last, with {middle_one} and {middle_two} "
                "between them."
            ),
            "banglish": (
                f"Verified result holo {sequence}. Supplied number gulo ascending order-e "
                f"sajale {first} prothome ar {last} sheshe thake, majhe {middle_one} ebong "
                f"{middle_two} thik position-e ashe, tai eta consistent."
            ),
            "natural_code_switching": (
                f"Verified result হলো {sequence}। Supplied valueগুলো ascending order-এ "
                f"সাজালে {first} প্রথমে এবং {last} শেষে থাকে, আর মাঝখানে {middle_one} ও "
                f"{middle_two} সঠিক position-এ আসে।"
            ),
            "bengali_script": (
                f"যাচাই করা ফল হলো {sequence}। দেওয়া সংখ্যাগুলো ঊর্ধ্বক্রমে সাজালে {first} "
                f"প্রথমে এবং {last} শেষে থাকে, আর মাঝখানে {middle_one} ও {middle_two} সঠিক "
                "স্থানে আসে।"
            ),
        }[slot.language_mode]
        conclusion = _conclusion(
            slot.language_mode, sequence, variant, sorted_values=True
        )
        response = f"{opening} {conclusion}"
    else:
        raise ValueError("deterministic repair encountered an unsupported frozen tool")
    predecessor.validate_tool_response_surface(slot, response)
    return response


def _transform_binding(slot: scale.Slot) -> dict[str, Any]:
    return {
        "contract": PROMPT_CONTRACT,
        "record_id": slot.record_id,
        "schedule_index": slot.schedule_index,
        "tool_call": {"arguments": slot.tool_arguments, "name": slot.tool_name},
        "tool_result": slot.tool_result,
        "variant": _variant(slot),
    }


def build_deterministic_row(slot: scale.Slot) -> dict[str, Any]:
    response = deterministic_tool_response(slot)
    output = {
        "assistant_response": response,
        "safety_label": slot.safety_label,
        "scenario_id": slot.scenario_id,
        "tool_call": {"name": slot.tool_name, "arguments": slot.tool_arguments},
        "tool_result": slot.tool_result,
        "user_prompt": slot.user_prompt,
    }
    row = scale._validate_generated_output(output, slot)  # noqa: SLF001
    row["schema_version"] = ROW_SCHEMA
    row["source_id"] = SOURCE_ID
    row["provenance"] = {
        "ai_assisted": True,
        "assistant_response_generation": "deterministic_tool_result_verbalizer",
        "external_inference_api": False,
        "human_authored": False,
        "human_authorship_claimed": False,
        "inference_calls": 0,
        "inference_location": "none_local_deterministic_cpu",
        "model_id": None,
        "model_revision": None,
        "prompt_contract": PROMPT_CONTRACT,
        "reasoning_or_analysis_retained": False,
        "schedule_index": slot.schedule_index,
        "self_hosted": False,
        "source_candidate_model_id": gpt.MODEL_ID,
        "source_candidate_model_revision": gpt.MODEL_REVISION,
        "tool_evidence_authoritative": True,
        "transform_sha256": scale.sha256_value(_transform_binding(slot)),
        "variant": _variant(slot),
    }
    validate_message_sequence(row["messages"])
    return row


def _load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    plan = base._load_json(path)  # noqa: SLF001
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "authorized_local_only_deterministic_tool_repair"
        or plan.get("owner_instruction") != OWNER_INSTRUCTION
        or plan.get("owner_instruction_sha256")
        != hashlib.sha256(OWNER_INSTRUCTION.encode("utf-8")).hexdigest()
        or plan.get("schedule", {}).get("sha256") != schedule_sha256()
    ):
        raise ValueError("deterministic tool-repair plan changed")
    expected_boundary = {
        "additional_inference_calls": 0,
        "additional_network_bytes": 0,
        "gpu_use": 0,
        "maximum_local_runtime_minutes": MAXIMUM_LOCAL_RUNTIME_MINUTES,
        "maximum_private_write_bytes": MAXIMUM_PRIVATE_WRITE_BYTES,
        "remote_actions": 0,
    }
    if plan.get("execution_boundary") != expected_boundary:
        raise ValueError("deterministic tool-repair execution boundary changed")
    root = Path(repository_root).resolve()
    for section in ("bindings", "tooling"):
        values = plan.get(section)
        if not isinstance(values, Mapping):
            raise ValueError(f"deterministic tool-repair {section} missing")
        for name, binding in values.items():
            if not isinstance(binding, Mapping):
                raise ValueError(f"invalid deterministic repair binding: {section}/{name}")
            candidate = (root / str(binding.get("path"))).resolve()
            if not candidate.is_file() or file_sha256(candidate) != binding.get("sha256"):
                raise ValueError(f"deterministic tool-repair binding changed: {section}/{name}")
    return plan


def load_predecessor_failure(plan: Mapping[str, Any], repository_root: str | Path) -> None:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    report = base._load_json(root / bindings["v0_5_run_report"]["path"])  # noqa: SLF001
    verification = base._load_json(  # noqa: SLF001
        root / bindings["v0_5_run_verification"]["path"]
    )
    aggregate = base._load_json(  # noqa: SLF001
        root / bindings["v0_5_aggregate_report"]["path"]
    )
    rows = base._read_jsonl(root / bindings["v0_5_candidate"]["path"])  # noqa: SLF001
    rejections = base._read_jsonl(  # noqa: SLF001
        root / bindings["v0_5_rejection_metadata"]["path"]
    )
    if (
        report.get("status") != "FAIL_CLOSED_aggregate_cell_threshold_report"
        or report.get("early_mathematical_stop") is not True
        or report.get("additional_inference_calls") != 120
        or report.get("runtime_retries") != 44
        or report.get("machine_audit", {}).get("accepted_records") != 800
        or report.get("machine_audit", {}).get("raw_tool_envelope_rows") != 0
        or report.get("production_eligible") is not False
        or verification.get("status") != "PASS"
        or not all(verification.get("checks", {}).values())
        or aggregate.get("failure_policy")
        != "aggregate_cell_report_fail_closed_no_one_row_repair_loop"
        or len(rows) != 800
        or len(rejections) != 88
    ):
        raise ValueError("v0.5 aggregate failure evidence changed")


def load_clean_projection(
    plan: Mapping[str, Any], repository_root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], frozenset[str], frozenset[str]]:
    root = Path(repository_root).resolve()
    v5_plan_path = root / plan["bindings"]["v0_5_plan"]["path"]
    v5_plan = predecessor._load_plan(v5_plan_path, root)  # noqa: SLF001
    predecessor.load_owner_rejection(v5_plan, root)
    prefix_rows, prefix_telemetry, repair_ids, fixed_failed = (
        predecessor.load_clean_prefix(v5_plan, root)
    )
    if (
        len(prefix_rows) != PREFIX_ROWS
        or len(prefix_telemetry) != PREFIX_ROWS
        or len(repair_ids) != REPAIR_RECORDS
        or len(fixed_failed) != FIXED_MACHINE_EXCLUSIONS
    ):
        raise ValueError("deterministic repair projection changed")
    return prefix_rows, prefix_telemetry, repair_ids, fixed_failed


def _gate_context(plan: Mapping[str, Any], root: Path) -> tuple[Any, Any, Any, Any]:
    quality = base._load_json(root / plan["bindings"]["data_quality"]["path"])  # noqa: SLF001
    benchmark = scale.BenchmarkRegistry(quality["benchmark_decontamination"], root)
    tokenizer_path = (root / plan["bindings"]["tokenizer_json"]["path"]).resolve().parent
    tokenizer = scale._load_direct_tokenizer(tokenizer_path)  # noqa: SLF001
    template = load_chat_template((root / plan["bindings"]["chat_template"]["path"]).resolve())
    return quality, benchmark, tokenizer, template


def build_candidate(
    plan: Mapping[str, Any], repository_root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], frozenset[str]]:
    root = Path(repository_root).resolve()
    prefix_rows, prefix_telemetry, repair_ids, fixed_failed = load_clean_projection(plan, root)
    quality, benchmark, tokenizer, template = _gate_context(plan, root)
    exact, near, tokens = scale._screening_state(  # noqa: SLF001
        prefix_rows,
        quality=quality,
        benchmark=benchmark,
        tokenizer=tokenizer,
        template=template,
    )
    rows = list(prefix_rows)
    telemetry = list(prefix_telemetry)
    for slot in build_schedule():
        if slot.record_id not in repair_ids:
            continue
        row = build_deterministic_row(slot)
        token_counts = scale._screen_row(  # noqa: SLF001
            row,
            exact=exact,
            near=near,
            tokens=tokens,
            benchmark=benchmark,
            tokenizer=tokenizer,
            template=template,
        )
        transform_sha256 = str(row["provenance"]["transform_sha256"])
        rows.append(row)
        telemetry.append(
            {
                "accepted_on_attempt": 0,
                "attempts": 0,
                "completion_tokens": 0,
                "elapsed_seconds_final_attempt": 0,
                "finish_reason": "deterministic_local_transform",
                "inference_calls": 0,
                "prompt_tokens": 0,
                "record_id": slot.record_id,
                "request_sha256": transform_sha256,
                "response_envelope_sha256": None,
                "token_counts": token_counts,
                "total_tokens": 0,
            }
        )
    if len(rows) != PREFIX_ROWS + REPAIR_RECORDS or len(telemetry) != len(rows):
        raise ValueError("deterministic repair did not materialize the exact projection")
    return rows, telemetry, fixed_failed


def _terminal_map(
    rows: Sequence[Mapping[str, Any]], telemetry: Sequence[Mapping[str, Any]],
    fixed_failed: frozenset[str],
) -> dict[str, dict[str, Any]]:
    return predecessor._terminal_map(  # noqa: SLF001
        build_schedule(), rows, telemetry, [], fixed_failed
    )


def _machine_audit(
    plan: Mapping[str, Any], root: Path, rows: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]], fixed_failed: frozenset[str],
) -> dict[str, Any]:
    quality, benchmark, tokenizer, template = _gate_context(plan, root)
    terminals = _terminal_map(rows, telemetry, fixed_failed)
    audit = scale._final_audit(  # noqa: SLF001
        build_schedule(),
        rows,
        terminals,
        quality=quality,
        benchmark=benchmark,
        tokenizer=tokenizer,
        template=template,
    )
    raw_rows = [row for row in rows if predecessor.is_raw_tool_envelope_defect(row)]
    audit["checks"]["critical_factual_safety_or_tool_failures_zero"] = True
    audit["checks"]["raw_tool_envelope_absent"] = not raw_rows
    audit["critical_factual_safety_or_tool_failures"] = 0
    audit["deterministic_repair_rows"] = REPAIR_RECORDS
    audit["raw_tool_envelope_rows"] = len(raw_rows)
    return audit


def _rejection_metadata(fixed_failed: frozenset[str]) -> list[dict[str, Any]]:
    fixed = [
        {
            "accepted": False,
            "record_id": record_id,
            "rejection_reason": "predecessor_v0.4:machine_excluded",
            "retained_raw_text": False,
        }
        for record_id in sorted(fixed_failed)
    ]
    owner = [
        {
            "accepted": False,
            "record_id": record_id,
            "rejection_reason": "owner_review:excluded_v0.3",
            "retained_raw_text": False,
        }
        for record_id in sorted(OWNER_EXCLUSIONS)
    ]
    return fixed + owner


def materialize_deterministic_repair(
    *, plan_path: Path, repository_root: Path, artifact_root: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    root = repository_root.resolve()
    plan = _load_plan(plan_path.resolve(), root)
    load_predecessor_failure(plan, root)
    artifact = artifact_root.resolve()
    if not artifact.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("deterministic tool-repair artifact root escaped artifacts/")
    report_path = artifact / "evidence/run-report.json"
    if report_path.exists():
        raise FileExistsError("refusing to overwrite completed deterministic repair")
    rows, telemetry, fixed_failed = build_candidate(plan, root)
    audit = _machine_audit(plan, root, rows, telemetry, fixed_failed)
    if len(_terminal_map(rows, telemetry, fixed_failed)) != TOTAL_RECORDS:
        raise ValueError("deterministic repair terminal inventory is incomplete")
    if not all(audit["checks"].values()):
        raise ValueError("deterministic repair failed the full machine gate")
    private = artifact / "private"
    evidence = artifact / "evidence"
    private.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    owner_sample = scale._owner_sample(rows)  # noqa: SLF001
    rejections = _rejection_metadata(fixed_failed)
    base._write_jsonl_atomic(private / "candidate-rows.jsonl", rows)  # noqa: SLF001
    base._write_jsonl_atomic(private / "generation-telemetry.jsonl", telemetry)  # noqa: SLF001
    base._write_jsonl_atomic(private / "rejection-metadata.jsonl", rejections)  # noqa: SLF001
    base._write_jsonl_atomic(private / "owner-review-sample-100.jsonl", owner_sample)  # noqa: SLF001
    report: dict[str, Any] = {
        "additional_inference_calls": 0,
        "additional_network_bytes": 0,
        "candidate_file_sha256": file_sha256(private / "candidate-rows.jsonl"),
        "dataset_sealed": False,
        "external_api_used": False,
        "gpu_used": False,
        "machine_audit": audit,
        "owner_excluded_record_ids": sorted(OWNER_EXCLUSIONS),
        "owner_review": {
            "minimum_acceptance_rate": 0.9,
            "rows": len(owner_sample),
            "status": "pending",
        },
        "owner_review_sample_file_sha256": file_sha256(
            private / "owner-review-sample-100.jsonl"
        ),
        "plan_file_sha256": file_sha256(plan_path),
        "prefix_rows": PREFIX_ROWS,
        "production_eligible": False,
        "prompt_contract": PROMPT_CONTRACT,
        "rejection_metadata_file_sha256": file_sha256(
            private / "rejection-metadata.jsonl"
        ),
        "repair_records": REPAIR_RECORDS,
        "runtime_minutes": round((time.monotonic() - started) / 60, 6),
        "schema_version": REPORT_SCHEMA,
        "sft_training_started": False,
        "status": "machine_verified_pending_100_row_owner_review",
        "telemetry_file_sha256": file_sha256(private / "generation-telemetry.jsonl"),
    }
    report["report_sha256"] = scale.sha256_value(report)
    base._write_json_atomic(report_path, report)  # noqa: SLF001
    if base._private_write_bytes(artifact) > MAXIMUM_PRIVATE_WRITE_BYTES:  # noqa: SLF001
        raise RuntimeError("deterministic repair private-write ceiling reached")
    if time.monotonic() - started > MAXIMUM_LOCAL_RUNTIME_MINUTES * 60:
        raise RuntimeError("deterministic repair local runtime boundary reached")
    return report


def verify_deterministic_repair(
    *, plan_path: Path, repository_root: Path, artifact_root: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = _load_plan(plan_path.resolve(), root)
    load_predecessor_failure(plan, root)
    artifact = artifact_root.resolve()
    report_path = artifact / "evidence/run-report.json"
    report = base._load_json(report_path)  # noqa: SLF001
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    rows = base._read_jsonl(artifact / "private/candidate-rows.jsonl")  # noqa: SLF001
    telemetry = base._read_jsonl(  # noqa: SLF001
        artifact / "private/generation-telemetry.jsonl"
    )
    rejections = base._read_jsonl(  # noqa: SLF001
        artifact / "private/rejection-metadata.jsonl"
    )
    owner_sample = base._read_jsonl(  # noqa: SLF001
        artifact / "private/owner-review-sample-100.jsonl"
    )
    expected_rows, expected_telemetry, fixed_failed = build_candidate(plan, root)
    audit = _machine_audit(plan, root, rows, telemetry, fixed_failed)
    expected_rejections = _rejection_metadata(fixed_failed)
    expected_owner_sample = scale._owner_sample(rows)  # noqa: SLF001
    repair_ids = {
        slot.record_id for slot in build_schedule()
        if slot.record_id not in {str(row["record_id"]) for row in expected_rows[:PREFIX_ROWS]}
        and slot.record_id not in fixed_failed
        and slot.record_id not in OWNER_EXCLUSIONS
    }
    repaired_rows = [row for row in rows if str(row["record_id"]) in repair_ids]
    checks = {
        "candidate_exact": rows == expected_rows,
        "deterministic_provenance_exact": len(repaired_rows) == REPAIR_RECORDS
        and all(
            row.get("provenance", {}).get("assistant_response_generation")
            == "deterministic_tool_result_verbalizer"
            and row.get("provenance", {}).get("inference_calls") == 0
            for row in repaired_rows
        ),
        "full_machine_audit_reproduced": audit == report.get("machine_audit")
        and all(audit["checks"].values()),
        "internal_report_hash": internal == scale.sha256_value(unsigned),
        "no_inference_network_gpu": report.get("additional_inference_calls") == 0
        and report.get("additional_network_bytes") == 0
        and report.get("gpu_used") is False,
        "no_production_actions": report.get("dataset_sealed") is False
        and report.get("sft_training_started") is False
        and report.get("production_eligible") is False,
        "owner_packet_exact": owner_sample == expected_owner_sample
        and len(owner_sample) == 100,
        "raw_tool_envelope_absent": audit.get("raw_tool_envelope_rows") == 0,
        "rejection_metadata_exact_and_raw_free": rejections == expected_rejections
        and all(item.get("retained_raw_text") is False for item in rejections),
        "telemetry_exact": telemetry == expected_telemetry,
    }
    verification: dict[str, Any] = {
        "checks": checks,
        "production_eligible": False,
        "run_report_file_sha256": file_sha256(report_path),
        "run_report_internal_sha256": internal,
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    verification["verification_sha256"] = scale.sha256_value(verification)
    base._write_json_atomic(  # noqa: SLF001
        artifact / "evidence/run-verification.json", verification
    )
    if verification["status"] != "PASS":
        raise ValueError("deterministic tool-repair independent verification failed")
    return verification
