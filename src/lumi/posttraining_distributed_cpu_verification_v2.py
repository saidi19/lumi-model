"""Read-only factual replay of the bounded distributed CPU attempt."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp

from . import posttraining_distributed_cpu_preflight_v2 as run
from . import posttraining_distributed_sft_v2 as worker
from .configuration_lumi import LumiConfig
from .distributed_training import _state_dict_digest
from .modeling_lumi import LumiForCausalLM
from .posttraining_sft_runtime_v4 import _require_private_regular, _verify_checksum_manifest

SCHEMA = "lumi.sft-distributed-cpu-preflight-verification.v0.2"
STATUS = "PASS_independently_verified_current_lineage_ddp_fsdp_cpu_resume_v0_2"


def _private_json(path: Path) -> dict[str, Any]:
    _require_private_regular(path)
    return run._read_json(path)


def _tree_fingerprint(root: Path) -> str:
    run._private_tree_bytes(root)
    inventory = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            inventory.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": run.sha256_file(path),
                }
            )
    return run.sha256_value(inventory)


def _model_state(config: Mapping[str, Any], checkpoint: Path) -> dict[str, Any]:
    worker.verify_distributed_sft_checkpoint(checkpoint)
    # Instantiate from bound configuration only; no model download or inference.
    with torch.random.fork_rng(devices=[]):
        model = LumiForCausalLM(LumiConfig(**dict(config)))
    state = model.state_dict()
    dcp.load({"model": state}, checkpoint_id=checkpoint / "model", no_dist=True)
    incompatible = model.load_state_dict(state)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("distributed CPU plain model restore is incompatible")
    return state


def _verify_model_pairs(
    *, config: Mapping[str, Any], reference: Path, resumed: Path
) -> dict[str, str]:
    digests = {}
    for step in (1, 2):
        name = f"checkpoint-{step:08d}"
        left = _model_state(config, reference / name)
        right = _model_state(config, resumed / name)
        if not run._nested_exact(left, right):
            raise ValueError("distributed CPU checkpoint model tensors differ")
        digest = _state_dict_digest(left)
        digests[str(step)] = digest["sha256"]
        if step == 2:
            for output in (reference, resumed):
                published = run._final_model_digest(output)
                if any(published.get(key) != value for key, value in digest.items()):
                    raise ValueError("published model digest differs from loaded tensors")
    return digests


def _verify_run_contract(
    state: Mapping[str, Any], plan: Mapping[str, Any], repository: Path
) -> None:
    specification = state["run_specification"]
    training = plan["training"]
    data = specification["data"]
    verification = {
        **data["read_audit"],
        "all_inventory_file_hashes_verified": False,
        "full_content_hashes_verified": False,
        "profile": "token_consistent_v0.2",
        **{
            key: data[key]
            for key in (
                "inventory_sha256",
                "manifest_file_sha256",
                "manifest_sha256",
                "metadata_inventory_sha256",
                "selected_prefix_sha256",
            )
        },
    }
    source = repository / plan["input"]["source_checkpoint"]
    identity = worker.build_distributed_sft_run_identity(
        source_config_bytes=run._read_regular_file(source / "config.json"),
        source_checkpoint_manifest_sha256=plan["input"]["source_checkpoint_manifest_sha256"],
        shard_manifest={
            "context_window": 4096,
            "manifest_sha256": data["manifest_sha256"],
            "seal_sha256": data["seal_sha256"],
        },
        shard_verification=verification,
        encoded_shards_record_limit=4,
        backend="gloo",
        world_size=2,
        strategy=specification["distributed"]["strategy"],
        per_rank_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        max_grad_norm=training["maximum_gradient_norm"],
        checkpoint_steps=1,
        keep_checkpoints=2,
        target_steps=2,
        seed=training["seed"],
        lr_schedule=training["lr_schedule"],
        warmup_steps=0,
        min_learning_rate_ratio=training["minimum_learning_rate_ratio"],
        attention_implementation="eager",
        optimizer_profile=training["optimizer_profile"],
        precision="fp32",
        gradient_sync_mode=training["gradient_sync_mode"],
        pad_to_multiple_of=8,
        source_commit=None,
    )
    scope = specification.get("authorization", {})
    if (
        scope.get("paid_gpu") is not False
        or scope.get("full_4b_training") is not False
        or scope.get("mode") != "local_cpu_preflight"
    ):
        raise ValueError("distributed CPU authorization scope changed")
    expected = {
        **identity,
        "authorization": scope,
        "locations": {
            "encoded_shards": str((repository / plan["input"]["encoded_directory"]).resolve()),
            "source_checkpoint": str(source.resolve()),
        },
    }
    if specification != expected or state["run_fingerprint"] != run.sha256_value(identity):
        raise ValueError("distributed CPU run contract differs from bound plan")


def verify_distributed_cpu_preflight(
    *, plan_path: str | Path, repository_root: str | Path
) -> dict[str, Any]:
    started = time.monotonic()
    repository = Path(repository_root).resolve()
    plan = run.load_plan(plan_path, repository_root=repository)
    output = plan["outputs"]
    verification_path = repository / output["verification"]
    if os.path.lexists(verification_path):
        raise FileExistsError("distributed CPU verification already exists")
    private = repository / output["private_root"]
    logs_root = repository / output["launch_logs"]
    run._require_final_private_inventory(private, logs_root, plan)
    before = _tree_fingerprint(private)
    marker = _private_json(repository / output["no_retry_marker"])
    expected_marker = {
        "external_network_bytes": 0,
        "gpu_use": 0,
        "inference_calls": 0,
        "launches_authorized": 6,
        "plan_file_sha256": run.sha256_file(plan_path),
        "raw_training_text_retained": False,
        "retries_after_this_marker": 0,
        "schema_version": "lumi.sft-distributed-cpu-preflight-started.v0.2",
        "status": "STARTED_no_retry",
    }
    if marker != expected_marker:
        raise ValueError("distributed CPU parent marker changed")
    report_path = repository / output["report"]
    report = _private_json(report_path)
    unsigned = dict(report)
    if unsigned.pop("report_sha256", None) != run.sha256_value(unsigned):
        raise ValueError("distributed CPU report internal hash changed")
    historical = run.diagnose_historical_v0_1(repository_root=repository, limits=plan["limits"])
    historical_read_total = (
        historical["diagnostic_bytes_read"]
        + report["historical_v0_1_diagnostic"]["diagnostic_bytes_read"]
    )
    if historical_read_total > plan["limits"]["maximum_historical_v0_1_private_bytes_read"]:
        raise ValueError("combined historical diagnostic reads exceed cap")

    launch_logs = []
    audits = []
    for index, spec in enumerate(run.LAUNCH_SPECS, 1):
        log = _private_json(logs_root / f"{index:02d}-{spec.code}.json")
        expected_log = {
            "child_output_bytes_retained": 0,
            "external_network_bytes": 0,
            "gpu_use": 0,
            "inference_calls": 0,
            "processes": 2,
            "raw_training_text_retained": False,
            "return_code": 0,
            "schema_version": "lumi.sft-distributed-cpu-launch-log.v0.2",
            "stage_code": spec.code,
            "status": "completed",
        }
        if log != expected_log:
            raise ValueError("distributed CPU launch receipt changed")
        launch_logs.append(log)
        state = run._state(repository / output[spec.output_key])
        audits.append(run._validate_read_audit(state, plan))
        _verify_run_contract(state, plan, repository)

    source = repository / plan["input"]["source_checkpoint"]
    _verify_checksum_manifest(
        source,
        expected_manifest_sha256=plan["input"]["source_checkpoint_manifest_sha256"],
        require_private=True,
    )
    config = run._read_json(source / "config.json")
    results, model_digests = {}, {}
    for strategy, prefix in (("ddp", "ddp"), ("fsdp_full_shard", "fsdp")):
        reference = repository / output[f"{prefix}_reference"]
        resumed = repository / output[f"{prefix}_resumed"]
        for path, is_resume in ((reference, False), (resumed, True)):
            if (
                _private_json(path / worker.SFT_DISTRIBUTED_STARTED_FILENAME)
                != worker._fresh_marker()
            ):
                raise ValueError("distributed CPU worker marker changed")
            final_spec = run.LaunchSpec(prefix, f"{prefix}_reference", is_resume, 2, strategy)
            run._validate_intermediate_state(output=path, plan=plan, spec=final_spec)
            if is_resume and _private_json(
                path / worker.SFT_DISTRIBUTED_RESUME_STARTED_FILENAME
            ) != worker._resume_marker(stop_after_step=2, target_steps=2):
                raise ValueError("distributed CPU resume marker changed")
        results[strategy] = run._audit_strategy(
            strategy=strategy,
            reference=reference,
            resumed=resumed,
            plan=plan,
        )
        model_digests[strategy] = _verify_model_pairs(
            config=config,
            reference=reference,
            resumed=resumed,
        )
    resources = report.get("resource_accounting", {})
    runtime_seconds = resources.get("runtime_seconds")
    if (
        type(runtime_seconds) not in (int, float)
        or not 0 <= runtime_seconds <= plan["limits"]["maximum_local_runtime_minutes"] * 60
    ):
        raise ValueError("distributed CPU runtime accounting changed")
    accounting = run._aggregate_load_audits(audits, plan["limits"])
    private_bytes = run._private_tree_bytes(private)
    expected_report = run._report(
        plan=plan,
        plan_file_sha256=run.sha256_file(plan_path),
        historical=historical,
        launch_logs=launch_logs,
        load_accounting=accounting,
        strategy_audits=results,
        private_write_bytes=private_bytes,
        runtime_seconds=runtime_seconds,
    )
    if report != expected_report:
        raise ValueError("distributed CPU report differs from factual replay")
    if _tree_fingerprint(private) != before:
        raise ValueError("distributed CPU attempt changed during verification")
    combined_runtime = runtime_seconds + time.monotonic() - started
    if combined_runtime > plan["limits"]["maximum_local_runtime_minutes"] * 60:
        raise TimeoutError("combined distributed CPU runtime cap exceeded")
    result = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "plan_file_sha256": run.sha256_file(plan_path),
        "report_file_sha256": run.sha256_file(report_path),
        "report_internal_sha256": report["report_sha256"],
        "attempt_inventory_sha256": before,
        "independently_loaded_model_sha256": model_digests,
        "checks": {
            "current_input_and_code_bindings_match": True,
            "historical_v0_1_bytes_preserved_and_limitation_disclosed": True,
            "all_eight_checkpoint_models_reloaded_and_compared": True,
            "both_rank_optimizer_rng_and_stable_metrics_exact": True,
            "top_state_keys_and_run_contract_exact": True,
            "markers_launch_receipts_and_private_inventory_verified": True,
            "physical_read_and_runtime_caps_met": True,
            "verification_did_not_open_encoded_rows_or_binary_content": True,
        },
        "resource_accounting": {
            "additional_encoded_binary_bytes_read": 0,
            "additional_encoded_index_bytes_read": 0,
            "additional_optimizer_steps": 0,
            "combined_historical_diagnostic_bytes_read": historical_read_total,
            "combined_runtime_seconds": combined_runtime,
            "private_artifact_bytes": private_bytes,
            "external_network_bytes": 0,
            "gpu_use": 0,
            "inference_calls": 0,
        },
        "production_eligible": False,
        "raw_training_text_retained": False,
    }
    result["verification_sha256"] = run.sha256_value(result)
    reserved = private_bytes + report_path.stat().st_size + len(run.json_file_bytes(result))
    if reserved > plan["limits"]["maximum_private_write_bytes"]:
        raise ValueError("distributed CPU final output byte cap exceeded")
    run._write_exclusive_private_json(
        verification_path,
        result,
        maximum_bytes=plan["limits"]["verification_write_bytes_reserved"],
    )
    return result
