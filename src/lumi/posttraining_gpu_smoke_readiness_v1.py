"""Metadata-only GPU SFT planning. This module cannot authorize or launch training."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

PLAN = "configs/lumi-4b-sft-gpu-smoke-readiness-v0.1.json"
REPORT = "evidence/lumi-4b-sft-gpu-smoke-readiness-v0.1.json"
VERIFICATION = "evidence/lumi-4b-sft-gpu-smoke-readiness-verification-v0.1.json"
CPU_PLAN = "configs/lumi-4b-sft-distributed-cpu-preflight-v0.2.json"
CPU_PLAN_SHA = "3d1c158efc4a7fc7fc7d0e72e69a2c13339848b48757d380bcc2b7115bc0bf9b"
EXTRA_PATHS = {
    "final_base": "configs/lumi-4b-final-base-binding-v0.1.json",
    "distributed_report": "evidence/lumi-4b-sft-distributed-cpu-preflight-v0.2.json",
    "distributed_verification": (
        "evidence/lumi-4b-sft-distributed-cpu-preflight-verification-v0.2.json"
    ),
    "source_registry": "configs/lumi-4b-sft-source-registry-v0.24.json",
    "exporter": "src/lumi/distributed_export.py",
    "implementation": "src/lumi/posttraining_gpu_smoke_readiness_v1.py",
    "runner": "scripts/check_sft_gpu_smoke_readiness_v1.py",
    "tests": "tests/test_sft_gpu_smoke_readiness_v1.py",
}
BOUNDARY = {
    name: False for name in (
        "gpu_authorized", "training_authorized", "network_authorized",
        "remote_or_instance_action_authorized", "inference_authorized",
        "checkpoint_content_read_authorized", "dataset_content_read_authorized",
        "production_eligible", "production_seal_authorized", "promotion_authorized",
    )
}
PENDING = [
    "final_4b_hf_import_receipt_and_exact_checksum_manifest_unbound",
    "sft_specific_authorization_and_bounded_launcher_not_integrated",
    "target_host_runtime_hardware_and_total_resource_caps_unbound",
    "target_host_content_verified_shard_relocation_receipt_unbound",
    "cuda_mask_precision_resume_acceptance_and_fault_tests_not_frozen",
    "separate_exact_gpu_remote_execution_approval_not_granted",
]
DRAFT = {
    "status": "non_executable_research_only_design",
    "scope": "private_research_nonproduction",
    "world_size": 8,
    "strategy": "fsdp_full_shard",
    "device": "cuda",
    "backend": "nccl",
    "precision": "bf16",
    "attention_implementation": "eager",
    "record_limit": 8,
    "per_rank_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "context_window": 4096,
    "target_steps": 2,
    "learning_rate": 0.00001,
    "weight_decay": 0.01,
    "maximum_gradient_norm": 1.0,
    "lr_schedule": "cosine",
    "warmup_steps": 0,
    "minimum_learning_rate_ratio": 0.1,
    "optimizer_profile": "grouped_adamw_beta2_0.95_v0.1",
    "checkpoint_steps": 1,
    "seed": 2027,
    "pad_to_multiple_of": 8,
    "gradient_sync_mode": "last_microbatch",
    "sequence": [
        {"run": "reference", "from_step": 0, "stop_step": 2, "resume": False},
        {"run": "resumed", "from_step": 0, "stop_step": 1, "resume": False},
        {"run": "resumed", "from_step": 1, "stop_step": 2, "resume": True},
    ],
    "repeat_prefix_across_epochs": True,
    "quality_or_representativeness_claim": False,
    "cross_conversation_packing": False,
    "retries_after_started_launch": 0,
    "executable_argv": None,
    "host": None,
    "runtime_digest": None,
    "source_hf_checkpoint": None,
    "source_hf_checksum_manifest_sha256": None,
    "relocated_encoded_metadata_sha256": None,
    "launch_authorization": None,
    "maximum_combined_runtime_minutes": None,
    "maximum_cost_usd": None,
    "maximum_host_ram_bytes": None,
    "maximum_per_gpu_memory_bytes": None,
    "maximum_private_write_bytes": None,
    "maximum_transfer_bytes": None,
    "maximum_checkpoint_read_bytes": None,
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def safe_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    require(not path.is_absolute() and bool(path.parts), "unsafe_relative_path")
    require(".." not in path.parts, "unsafe_relative_path")
    current = root
    for part in path.parts:
        current /= part
        require(not current.is_symlink(), "symlink_path")
    require(current.resolve().is_relative_to(root.resolve()), "escaped_repository")
    return current


def stable_stat(info: os.stat_result) -> tuple:
    # Reading may update atime; it must not change identity, size, mode or mtime/ctime.
    return tuple(getattr(info, field) for field in (
        "st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
        "st_size", "st_mtime_ns", "st_ctime_ns",
    ))


class MetadataReader:
    """Only named, size-capped regular metadata files can be read."""

    def __init__(self, root: Path, allowed: set[str]):
        self.root = root
        self.allowed = allowed
        self.bytes_read = 0
        self.cache: dict[str, bytes] = {}

    def read(self, relative: str, expected: str | None = None) -> bytes:
        require(relative in self.allowed, "metadata_path_not_allowlisted")
        if relative not in self.cache:
            path = safe_path(self.root, relative)
            before = path.lstat()
            require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "unsafe_metadata")
            require(before.st_size <= 1_048_576, "metadata_file_cap")
            require(self.bytes_read + before.st_size <= 4_194_304, "metadata_read_cap")
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb", buffering=0) as handle:
                require(stable_stat(os.fstat(handle.fileno())) == stable_stat(before),
                        "metadata_changed_before_read")
                raw = handle.read(before.st_size + 1)
                require(stable_stat(os.fstat(handle.fileno())) == stable_stat(before),
                        "metadata_changed_during_read")
            require(len(raw) == before.st_size and
                    stable_stat(path.lstat()) == stable_stat(before), "metadata_changed")
            self.bytes_read += len(raw)
            self.cache[relative] = raw
        raw = self.cache[relative]
        if expected is not None:
            require(hashlib.sha256(raw).hexdigest() == expected, "metadata_hash_mismatch")
        return raw


def internal_hash(value: dict, field: str) -> str:
    expected = value.get(field)
    require(isinstance(expected, str), "internal_hash_missing")
    require(digest({k: v for k, v in value.items() if k != field}) == expected,
            "internal_hash_mismatch")
    return expected


def geometry(draft: dict) -> dict:
    for key in ("world_size", "record_limit", "per_rank_batch_size",
                "gradient_accumulation_steps", "target_steps", "context_window"):
        require(type(draft[key]) is int and draft[key] > 0, "geometry_integer")
    world = draft["world_size"]
    batch = world * draft["per_rank_batch_size"]
    require(batch <= draft["record_limit"] <= 12, "bounded_prefix_cannot_fill_batch")
    launches = len(draft["sequence"])
    steps = sum(x["stop_step"] - x["from_step"] for x in draft["sequence"])
    samples = steps * batch * draft["gradient_accumulation_steps"]
    reads = launches * world * draft["record_limit"]
    return {
        "sequential_launches": launches,
        "global_optimizer_steps_across_all_paths": steps,
        "rank_optimizer_steps_across_all_paths": steps * world,
        "logical_sample_events_across_all_paths": samples,
        "maximum_unique_train_records": draft["record_limit"],
        "rank_local_dataset_loads": launches * world,
        "maximum_encoded_index_row_reads": reads,
        "maximum_selected_binary_bytes": reads * draft["context_window"] * 8,
        "maximum_selected_index_bytes_including_oversize_probe": reads * 176,
        "maximum_logical_serialized_tokens": samples * draft["context_window"],
        "not_a_total_io_or_cost_budget": True,
    }


def shard_stat_inventory(root: Path, manifest: dict, expected: str) -> dict:
    """No encoded index/token/label file is opened, even for hashing."""
    require(root.is_dir() and not root.is_symlink(), "unsafe_encoded_directory")
    require(stat.S_IMODE(root.stat().st_mode) == 0o700, "unsafe_encoded_directory_mode")
    names = {"manifest.json": None}
    for shard in manifest["shards"]:
        for entry in shard["files"]:
            name = entry["path"]
            require(Path(name).name == name and name not in names, "invalid_shard_name")
            names[name] = entry["bytes"]
    require(set(os.listdir(root)) == set(names), "encoded_inventory_changed")
    items = []
    for name, size in sorted(names.items()):
        info = (root / name).lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                and info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o600,
                "unsafe_encoded_metadata")
        require(size is None or info.st_size == size, "encoded_size_changed")
        items.append({
            "mode": f"{stat.S_IMODE(info.st_mode):04o}", "relative_path": name,
            **{field: getattr(info, field) for field in (
                "st_ctime_ns", "st_dev", "st_ino", "st_mtime_ns",
                "st_nlink", "st_size", "st_uid",
            )},
        })
    require(digest(items) == expected, "encoded_metadata_inventory_changed")
    return {"files_stat_checked": len(items), "metadata_inventory_sha256": expected,
            "content_rehashed": False, "portable_to_another_host": False}


def code_findings(worker: bytes, runtime: bytes, exporter: bytes) -> dict:
    """Structural checks supplement exact file bindings; these are not CUDA tests."""
    worker_ast, runtime_ast, export_ast = map(ast.parse, (worker, runtime, exporter))
    imports = [n for n in ast.walk(worker_ast) if isinstance(n, ast.ImportFrom)]
    require(any(n.module == "distributed_training" and any(
        a.name == "_validate_execution_scope" for a in n.names) for n in imports),
        "worker_authorization_call_graph_changed")
    calls = [n for n in ast.walk(worker_ast) if isinstance(n, ast.Call)]
    require(any(isinstance(n.func, ast.Name) and n.func.id == "_validate_execution_scope"
                for n in calls), "worker_authorization_call_missing")
    require(any(isinstance(n.func, ast.Attribute) and n.func.attr == "from_pretrained"
                and any(k.arg == "local_files_only" and isinstance(k.value, ast.Constant)
                        and k.value.value is True for k in n.keywords) for n in calls),
            "worker_local_hf_import_changed")
    fn = next(n for n in runtime_ast.body if isinstance(n, ast.FunctionDef)
              and n.name == "_validate_paid_authorization")
    require(not any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and "sft" in n.value.lower() for n in ast.walk(fn)),
            "generic_authorization_contract_changed")
    # The exporter READS the DCP input manifest; inspect its explicit output writes,
    # not every string literal (which would confuse an input read with an output).
    writes = [n for n in ast.walk(export_ast) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name) and n.func.id == "_write_json"]
    require(bool(writes) and all(
        n.args and isinstance(n.args[0], ast.BinOp)
        and isinstance(n.args[0].op, ast.Div)
        and isinstance(n.args[0].right, ast.Name)
        and n.args[0].right.id == "EXPORT_REPORT_FILENAME" for n in writes
    ), "export_manifest_contract_changed")
    return {
        "worker_delegates_authorization_to_generic_pretraining_contract": True,
        "worker_requires_local_hf_checkpoint_plus_separate_checksum_manifest": True,
        "existing_exporter_does_not_emit_worker_checksum_manifest": True,
        "existing_exporter_is_not_a_verified_final_4b_import_receipt": True,
        "static_review_only_no_model_import_or_cuda_test": True,
    }


def audit(root: Path, expected_plan_sha256: str) -> dict:
    reader = MetadataReader(root, {PLAN, CPU_PLAN, *EXTRA_PATHS.values()})
    plan = json.loads(reader.read(PLAN, expected_plan_sha256))
    require(set(plan) == {"schema_version", "authorization_boundary", "cpu_plan_sha256",
                         "bindings", "launch_draft", "pending_prelaunch_gates"},
            "plan_field_inventory_changed")
    require(plan["schema_version"] == "lumi.sft-gpu-smoke-readiness-plan.v0.1",
            "plan_schema_changed")
    # JSON equality alone would accept 0 in place of false.
    require(canonical(plan["authorization_boundary"]) == canonical(BOUNDARY),
            "planning_boundary_changed")
    require(canonical(plan["launch_draft"]) == canonical(DRAFT), "launch_draft_changed")
    require(plan["pending_prelaunch_gates"] == PENDING, "pending_gate_inventory_changed")
    require(plan["cpu_plan_sha256"] == CPU_PLAN_SHA, "cpu_plan_binding_changed")
    cpu = json.loads(reader.read(CPU_PLAN, CPU_PLAN_SHA))
    require(set(plan["bindings"]) == set(EXTRA_PATHS), "binding_inventory_changed")
    blobs = {}
    for name, relative in EXTRA_PATHS.items():
        binding = plan["bindings"][name]
        require(set(binding) == {"path", "sha256"} and binding["path"] == relative,
                "binding_path_changed")
        blobs[name] = reader.read(relative, binding["sha256"])
    for name, binding in cpu["bindings"].items():
        reader.allowed.add(binding["path"])
        blobs["cpu_" + name] = reader.read(binding["path"], binding["sha256"])
    require(plan["bindings"]["source_registry"]["sha256"] ==
            json.loads(blobs["cpu_current_data_contract"])["source_registry"]["sha256"],
            "registry_contract_binding_changed")
    manifest = json.loads(blobs["cpu_encoded_manifest"])
    require(internal_hash(manifest, "manifest_sha256") ==
            cpu["input"]["manifest_internal_sha256"], "encoded_manifest_lineage_changed")
    require(digest(manifest["shards"]) == cpu["input"]["inventory_sha256"],
            "encoded_inventory_hash_changed")
    require(manifest["seal_sha256"] == cpu["input"]["seal_internal_sha256"],
            "encoded_seal_binding_changed")
    require(manifest["production_eligible"] is False, "encoded_production_boundary_changed")
    contract = json.loads(blobs["cpu_current_data_contract"])
    policy = contract["learning_candidate"]
    require(policy["scope"] == "private_research_nonproduction" and
            policy["production_sft_allowed"] is False and
            policy["production_dataset_seal_allowed"] is False and
            policy["external_distribution_allowed"] is False, "private_scope_changed")
    evidence_names = (
        "distributed_report", "distributed_verification", "cpu_cpu_smoke_recovery_report",
        "cpu_cpu_smoke_recovery_verification", "cpu_current_shard_receipt",
        "cpu_current_shard_verification",
    )
    verified = {}
    for name in evidence_names:
        value = json.loads(blobs[name])
        require(value["status"].startswith("PASS"), "predecessor_gate_not_passed")
        require(value["production_eligible"] is False, "predecessor_scope_changed")
        field = "verification_sha256" if "verification" in name else "report_sha256"
        # Shard materialization receipts use receipt_sha256.
        if field not in value and name == "cpu_current_shard_receipt":
            field = "receipt_sha256"
        internal_hash(value, field)
        if "checks" in value:
            require(all(v is True for v in value["checks"].values()), "predecessor_check_false")
        verified[name] = value
    require(verified["distributed_verification"]["report_file_sha256"] ==
            plan["bindings"]["distributed_report"]["sha256"], "cpu_verification_link_changed")
    require(verified["distributed_verification"]["plan_file_sha256"] == CPU_PLAN_SHA,
            "cpu_verification_plan_changed")
    base = json.loads(blobs["final_base"])
    require(base["checkpoint"]["step"] == 76294 and
            base["checkpoint"]["training_tokens"] == 80_000_000_000,
            "final_base_cursor_changed")
    metadata = shard_stat_inventory(
        safe_path(root, cpu["input"]["encoded_directory"]), manifest,
        cpu["input"]["metadata_inventory_sha256"],
    )
    findings = code_findings(blobs["cpu_worker_implementation"],
                             blobs["cpu_distributed_training_runtime"], blobs["exporter"])
    result = {
        "schema_version": "lumi.sft-gpu-smoke-readiness-report.v0.1",
        "status": "PASS_local_metadata_readiness_audit_GPU_launch_BLOCKED",
        "plan_file_sha256": expected_plan_sha256,
        "authorization_boundary": BOUNDARY,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "launch_ready": False,
        "completed_predecessor_gates": {k: v["status"] for k, v in verified.items()},
        "final_base_binding": {"file_sha256": plan["bindings"]["final_base"]["sha256"],
                               "checkpoint": base["checkpoint"],
                               "weight_content_or_remote_backup_reverified": False},
        "dataset": {"records": manifest["records"], "splits": manifest["splits"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "seal_sha256": manifest["seal_sha256"], **metadata},
        "code_contract_findings": findings,
        "launch_draft_geometry": geometry(DRAFT),
        "pending_prelaunch_gates": PENDING,
        "verification_limits": [
            "metadata_and_bound_prior_evidence_not_full_dataset_or_checkpoint_reaudit",
            "stat_inventory_identity_does_not_replace_target_host_content_hash_verification",
            "cpu_tiny_fixture_success_is_not_final_4b_or_cuda_success",
            "planning_does_not_grant_execution_or_production_rights",
        ],
        "resource_accounting": {
            "metadata_files_read": len(reader.cache), "metadata_bytes_read": reader.bytes_read,
            "encoded_index_row_reads": 0, "encoded_binary_bytes_read": 0,
            "source_candidate_or_test_row_reads": 0, "model_weight_bytes_read": 0,
            "network_bytes": 0, "gpu_use": 0, "inference_calls": 0,
            "optimizer_steps": 0, "remote_actions": 0,
        },
    }
    result["report_sha256"] = digest(result)
    return result


def publish(root: Path, relative: str, value: dict) -> None:
    require(relative in {REPORT, VERIFICATION}, "output_not_allowlisted")
    raw = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False).encode() + b"\n"
    require(len(raw) <= 524_288, "output_size_cap")
    target = safe_path(root, relative)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def run(root: Path, plan_sha256: str, *, verify: bool = False) -> dict:
    target = VERIFICATION if verify else REPORT
    require(not safe_path(root, target).exists(), "output_already_exists_no_overwrite")
    result = audit(root, plan_sha256)
    if verify:
        reader = MetadataReader(root, {REPORT})
        raw = reader.read(REPORT)
        prior = json.loads(raw)
        internal_hash(prior, "report_sha256")
        require(prior == result, "replayed_report_differs")
        result = {
            "schema_version": "lumi.sft-gpu-smoke-readiness-verification.v0.1",
            "status": "PASS_fresh_metadata_replay_verified_GPU_launch_BLOCKED",
            "method": "fresh_process_same_implementation_structural_replay_not_independent_algorithm",
            "plan_file_sha256": plan_sha256,
            "report_file_sha256": hashlib.sha256(raw).hexdigest(),
            "report_internal_sha256": prior["report_sha256"],
            "launch_ready": False, "production_eligible": False,
            "raw_training_text_retained": False,
            "checks": {"full_report_recomputed_exact": True,
                       "all_bound_inputs_and_code_rechecked": True,
                       "all_31_encoded_files_stat_checked_without_content_reads": True,
                       "pending_launch_gates_preserved": True},
            "resource_accounting": {
                **prior["resource_accounting"],
                "metadata_bytes_read": prior["resource_accounting"]["metadata_bytes_read"]
                + len(raw),
                "metadata_files_read": prior["resource_accounting"]["metadata_files_read"] + 1,
            },
        }
        result["verification_sha256"] = digest(result)
    publish(root, target, result)
    return result
