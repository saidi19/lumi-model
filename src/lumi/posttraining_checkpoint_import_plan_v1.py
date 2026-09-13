"""Pure real-sized import planning from externally pinned final-base metadata.

No filesystem, ML, payload, transport or execution APIs. Resource figures are
explicit planning bounds, not observations or OS-enforced limits. A real launch
still needs a private read-only input mount and a separately bounded host runner.
The historical synthetic importer and its limits are not changed or consulted.
"""

from __future__ import annotations

import math

from .posttraining_final_base_metadata_v1 import (
    canonical,
    json_object,
    parse_manifest,
    relative_name,
    sha,
    verify_final_base_metadata,
)

SCHEMA = "lumi.sft-checkpoint-import-plan.v0.1"
MIB = 1024**2
GIB = 1024**3
MAX_HANDOFF_BYTES = 32768
MAX_PLAN_BYTES = 131072
MAX_PARAMETERS = 5_000_000_000
MAX_INVENTORY_BYTES = 64 * GIB
SERIALIZATION_RESERVE_BYTES = MIB
CONTROL_RESERVE_BYTES = 2 * MIB
FALSE_FLAGS = (
    "checkpoint_payload_verified", "tokenizer_payload_verified", "real_4b_import",
    "host_resources_verified", "execution_permitted", "training_authorized",
    "real_sft_execution_permitted", "production_eligible",
)
ERROR_CODES = (
    "handoff_input_cap", "handoff_binding", "handoff_schema", "handoff_lineage",
    "handoff_internal_hash", "handoff_inventory", "handoff_inventory_cap",
    "handoff_closed_manifest", "unsupported_dtype", "unsupported_geometry",
    "parameter_cap", "plan_size_cap", "plan_shape",
)


class ImportPlanError(ValueError):
    def __init__(self, code: str):
        self.code = code if code in ERROR_CODES else "plan_shape"
        super().__init__(self.code)


def require(value: bool, code: str) -> None:
    if not value:
        raise ImportPlanError(code) from None


def tensor_inventory(config: dict) -> list[dict]:
    """Analytical state_dict inventory; checked against real tiny architecture tests.

    Keep the head alias as a separate source value even when parameters are tied.
    Query width is deliberately independent of the residual hidden width.
    """
    fields = (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim",
    )
    require(type(config) is dict, "unsupported_geometry")
    require(all(type(config.get(k)) is int and 0 < config[k] <= 1_000_000 for k in fields),
            "unsupported_geometry")
    require(type(config.get("attention_bias")) is bool
            and type(config.get("tie_word_embeddings")) is bool, "unsupported_geometry")
    require(config["num_hidden_layers"] <= 128
            and config["head_dim"] % 2 == 0
            and config["num_attention_heads"] % config["num_key_value_heads"] == 0,
            "unsupported_geometry")
    v, h = config["vocab_size"], config["hidden_size"]
    i, n = config["intermediate_size"], config["num_hidden_layers"]
    q = config["num_attention_heads"] * config["head_dim"]
    kv = config["num_key_value_heads"] * config["head_dim"]
    items = []

    def add(name, shape, role="parameter", alias_of=None):
        item = {"name": name, "shape": list(shape), "role": role,
                "elements": math.prod(shape)}
        if alias_of is not None:
            item["alias_of"] = alias_of
        if role == "buffer":
            item["dtype"] = "float32"
        items.append(item)

    add("model.embed_tokens.weight", (v, h))
    for layer in range(n):
        prefix = f"model.layers.{layer}."
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            add(prefix + name, (h,))
        for name, shape in (("q_proj", (q, h)), ("k_proj", (kv, h)),
                            ("v_proj", (kv, h)), ("o_proj", (h, q))):
            add(prefix + "self_attn." + name + ".weight", shape)
            if config["attention_bias"]:
                add(prefix + "self_attn." + name + ".bias", (shape[0],))
        add(prefix + "self_attn.rotary.inv_freq", (config["head_dim"] // 2,), "buffer")
        for name, shape in (("gate_proj", (i, h)), ("up_proj", (i, h)),
                            ("down_proj", (h, i))):
            add(prefix + "mlp." + name + ".weight", shape)
    add("model.norm.weight", (h,))
    add("lm_head.weight", (v, h), alias_of=(
        "model.embed_tokens.weight" if config["tie_word_embeddings"] else None
    ))
    unique = sum(item["elements"] for item in items
                 if item["role"] == "parameter" and "alias_of" not in item)
    require(0 < unique <= MAX_PARAMETERS, "parameter_cap")
    return sorted(items, key=lambda item: item["name"])


def _handoff_inventory(raw, base, metadata, manifest_raw, state_raw):
    require(type(raw) is bytes and 0 < len(raw) <= MAX_HANDOFF_BYTES, "handoff_input_cap")
    binding = base.get("handoff")
    require(type(binding) is dict and sha(raw) == binding.get("file_sha256"), "handoff_binding")
    value = json_object(raw)
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1
            and value.get("gate") == "lumi_4b_checkpoint_handoff_v0.1"
            and value.get("status") == "sealed", "handoff_schema")
    digest = sha(canonical({k: v for k, v in value.items() if k != "handoff_sha256"}))
    require(digest == value.get("handoff_sha256") == binding.get("sha256"),
            "handoff_internal_hash")
    checkpoint = value.get("checkpoint")
    require(type(checkpoint) is dict, "handoff_lineage")
    for left, right in (("step", "step"), ("sequence_cursor", "sequence_cursor"),
                        ("tokens_seen", "training_tokens")):
        require(type(checkpoint.get(left)) is int and checkpoint[left] == metadata[right],
                "handoff_lineage")
    require(value.get("source_commit") == metadata["source_commit"]
            and value.get("run_fingerprint") == metadata["run_fingerprint"]
            and type(value.get("world_size")) is int
            and value["world_size"] == metadata["world_size"], "handoff_lineage")
    files = value.get("files")
    require(type(files) is list and 0 < len(files) <= 129, "handoff_inventory_cap")
    observed = {}
    for item in files:
        require(type(item) is dict and set(item) == {"path", "bytes", "sha256"},
                "handoff_inventory")
        name = relative_name(item["path"])
        require(name not in observed and type(item["bytes"]) is int and item["bytes"] > 0,
                "handoff_inventory")
        require(type(item["sha256"]) is str and len(item["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in item["sha256"]), "handoff_inventory")
        observed[name] = dict(item)
    require(sum(item["bytes"] for item in files) <= MAX_INVENTORY_BYTES, "handoff_inventory_cap")
    declared = {**parse_manifest(manifest_raw), "manifest.sha256": sha(manifest_raw)}
    require({name: item["sha256"] for name, item in observed.items()} == declared,
            "handoff_closed_manifest")
    for name, data in (("manifest.sha256", manifest_raw),
                       ("distributed-training-state.json", state_raw)):
        require(observed[name]["bytes"] == len(data), "handoff_closed_manifest")
    allowed = {"manifest.sha256", "distributed-training-state.json"}
    allowed |= {f"trainer-rank-{rank:05d}.pt" for rank in range(metadata["world_size"])}
    require("model/.metadata" in observed
            and all(name in allowed or name.startswith("model/") for name in observed),
            "handoff_inventory")
    return sorted(observed.values(), key=lambda item: item["path"])


def build_import_plan(base_raw: bytes, manifest_raw: bytes, state_raw: bytes,
                      config_raw: bytes, handoff_raw: bytes, *,
                      expected_base_sha256: str, dtype: str) -> dict:
    """Bind complete checkpoint declarations and real-sized resources, never launch.

    Checkpoint and tokenizer bytes are NOT opened. The handoff is transitively
    pinned by the independent base-file hash, including its own internal hash.
    Both FP32 and BF16 branches must preserve persistent FP32 rotary buffers.
    """
    try:
        require(dtype in {"float32", "bfloat16"}, "unsupported_dtype")
        metadata = verify_final_base_metadata(
            base_raw, manifest_raw, state_raw, config_raw,
            expected_base_sha256=expected_base_sha256,
        )
        base, config = json_object(base_raw), json_object(config_raw)
        checkpoint = _handoff_inventory(handoff_raw, base, metadata, manifest_raw, state_raw)
        tensors = tensor_inventory(config)
        parameters = sum(t["elements"] for t in tensors
                         if t["role"] == "parameter" and "alias_of" not in t)
        parameter_state = sum(t["elements"] for t in tensors if t["role"] == "parameter")
        buffers = sum(t["elements"] for t in tensors if t["role"] == "buffer")
        element_bytes = {"float32": 4, "bfloat16": 2}[dtype]
        source_bytes = 4 * (parameter_state + buffers)
        cast_bytes = element_bytes * parameter_state + 4 * buffers
        largest_bytes = 4 * max(t["elements"] for t in tensors)
        # Include independent tied aliases in both source and cast pools. A
        # largest-tensor allowance and 16MiB scratch cover tensor-local staging;
        # allocator, DCP deserialization and library overhead need host headroom.
        projected_tensor_peak = max(source_bytes + cast_bytes, 2 * cast_bytes)
        projected_tensor_peak += largest_bytes + 16 * MIB
        host_ram = ((2 * projected_tensor_peak + 4 * GIB + 16 * GIB - 1)
                    // (16 * GIB)) * 16 * GIB
        backend_output = cast_bytes + SERIALIZATION_RESERVE_BYTES
        tokens = [{"path": item["relative_path"], "bytes": item["bytes"],
                   "sha256": item["sha256"]} for item in base["tokenizer"]["files"]]
        tokens.sort(key=lambda item: item["path"])
        checkpoint_bytes = sum(item["bytes"] for item in checkpoint)
        tokenizer_bytes = sum(item["bytes"] for item in tokens)
        private_writes = backend_output + tokenizer_bytes + CONTROL_RESERVE_BYTES
        result = {
            "schema_version": SCHEMA, "status": "PASS_plan_only_not_execution",
            "metadata": metadata, "handoff_file_sha256": sha(handoff_raw),
            "checkpoint_inventory": checkpoint, "tokenizer_inventory": tokens,
            "tensor_inventory": tensors, "dtype": dtype,
            "buffer_dtype_policy": "configured_buffer_dtype_preserve_values",
            "model_parameters": parameters, "model_tensor_count": len(tensors),
            "buffer_elements": buffers,
            "resource_bounds": {
                "checkpoint_inventory_bytes": checkpoint_bytes,
                "checkpoint_inventory_files": len(checkpoint),
                "tokenizer_inventory_bytes": tokenizer_bytes,
                "source_fp32_tensor_bytes_including_aliases": source_bytes,
                "cast_tensor_bytes_including_aliases": cast_bytes,
                "maximum_individual_tensor_bytes": largest_bytes,
                "maximum_projected_live_tensor_bytes": projected_tensor_peak,
                "minimum_host_ram_planning_bytes": host_ram,
                "maximum_backend_output_bytes": backend_output,
                "maximum_private_output_write_bytes": private_writes,
                "minimum_staged_disk_planning_bytes": checkpoint_bytes + tokenizer_bytes
                    + private_writes,
                "checkpoint_verification_passes": 2,
                "checkpoint_audit_read_bytes": 2 * checkpoint_bytes,
                "tokenizer_audit_read_bytes": 2 * tokenizer_bytes,
                "backend_payload_read_cap_verified": False,
                "os_memory_time_disk_limits_enforced": False,
            },
            "required_runner_controls": [
                "separately_bound_dedicated_host_and_runtime",
                "immutable_read_only_closed_checkpoint_and_tokenizer_mounts",
                "two_streamed_full_inventory_checks_including_trainer_files",
                "no_trainer_deserialization_no_dataset_access",
                "explicit_backend_read_runtime_ram_and_write_limits",
                "exclusive_private_attempt_no_retry_after_start",
                "offline_no_accelerator_no_forward_no_optimizer",
                "tokenizer_payload_copy_with_original_normalized_inventory_semantics",
                "receipt_and_frozen_worker_manifest_consumer_verification",
            ],
            **dict.fromkeys(FALSE_FLAGS, False),
        }
        result["plan_sha256"] = sha(canonical(result))
        require(len(canonical(result)) <= MAX_PLAN_BYTES, "plan_size_cap")
        return result
    except (KeyError, TypeError, IndexError, OverflowError):
        raise ImportPlanError("plan_shape") from None
