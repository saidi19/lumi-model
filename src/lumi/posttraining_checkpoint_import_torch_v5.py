"""CPU import successor with metered checkpoint/HF/control content I/O.

The parent must bind inputs, admit a host, and enforce actual resources before
calling this internal kernel. Explicit caps are not inferred from old synthetic
limits. ``maximum_tensor_bytes`` bounds a conservative projected aggregate live
tensor payload, not allocator/library RSS. The synchronous zero-argument budget
callback must check the parent's resource/time ledger. It cannot interrupt an
in-progress library call; hard process/filesystem limits belong to the parent.

ML imports are lazy. Architecture inspection is meta-only; DCP aliases load into
independent CPU destinations before validation. Checks and digests are chunked.
No forward pass, optimizer, process group, GPU, transport, or retry is provided.

DCP and HF/control bytes use separate explicit ledgers; no native pathname
serializer or checkpoint mmap is used. Libraries still allocate/import outside
those content ledgers. Hard OS enforcement and real host admission remain absent.
HF-compatible shard bytes are reloaded through the real HF state_dict interface.
This is a new v0.5 receipt, not accepted old v3/v4 evidence. No source chmod.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from pathlib import Path

from .posttraining_checkpoint_dcp_reader_v1 import DCPReadError, verified_dcp_reader
from .posttraining_checkpoint_hf_bytes_v1 import (
    HFBytesError,
    prepare_export,
    read_documents,
    reload_state,
    write_export,
)
from .posttraining_checkpoint_hf_io_v1 import ContentIO, ContentIOError

CHUNK_BYTES = 1048576
SHARD_BYTES = 268435456
OUTPUT_OVERHEAD_BYTES = 1048576
MAXIMUM_CONFIG_BYTES = 65536
MAXIMUM_STATE_KEYS = 4096
MAXIMUM_CHUNKS_PER_TENSOR = 4096
MAXIMUM_OUTPUT_FILES = 4096
MODEL_FIELDS = (
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim",
    "max_position_embeddings", "rope_theta", "rms_norm_eps", "attention_bias",
    "tie_word_embeddings", "pad_token_id", "bos_token_id", "eos_token_id", "doc_token_id",
)
SOURCE_FIELDS = frozenset(MODEL_FIELDS) | {
    "model_type", "architectures", "initializer_range", "initialization_strategy",
    "use_cache", "torch_dtype", "dtype",
}
ERROR_CODES = (
    "limits_required", "budget_callback_required", "budget_check_failed", "dtype_rejected",
    "output_exists", "path_rejected", "configuration_rejected", "dependencies_unavailable",
    "distributed_group_rejected", "default_dtype_rejected", "parameter_limit_exceeded",
    "analytical_parameter_count_mismatch", "meta_architecture_rejected", "dcp_key_inventory",
    "dcp_metadata_type", "dcp_shape_inventory", "dcp_dtype_rejected", "dcp_chunk_dimensions",
    "dcp_chunk_bounds", "dcp_chunk_overlap", "dcp_chunk_coverage", "dcp_chunk_limit",
    "tensor_limit_exceeded", "output_limit_exceeded", "tensor_layout_rejected",
    "dcp_restored_inventory", "dcp_nonfinite", "tied_alias_mismatch", "role_inventory_mismatch",
    "buffer_dtype_rejected", "cast_nonfinite", "strict_restore_failed", "cast_restore_mismatch",
    "output_inventory_rejected", "output_config_changed", "reload_key_inventory",
    "reload_tensor_mismatch", "reload_tied_mismatch", "reload_digest_mismatch",
    "conversion_failed", "dcp_read_contract_rejected", "dcp_read_failed",
    "hf_io_contract_rejected", "hf_io_failed", "hf_loading_info_rejected",
)
EVIDENCE_KEYS = (
    "schema_version", "status", "dtype", "parameter_dtype", "buffer_dtype_policy",
    "model_parameters", "model_tensor_count", "source_tensor_sha256", "cast_tensor_sha256",
    "reload_tensor_sha256", "source_state_bytes", "cast_state_bytes", "largest_tensor_bytes",
    "projected_live_tensor_bytes", "projected_output_bytes", "output_bytes", "output_file_count",
    "safetensor_file_count", "chunk_bytes", "budget_checks", "complete_keys_shapes_verified",
    "hf_reload_tensor_equal", "strict_assign_verified", "tied_aliases_verified",
    "source_dicts_released_before_reload", "configuration_fields_preserved", "cpu_only",
    "no_forward_pass", "real_4b_import", "training_authorized", "production_eligible",
    "os_resource_limits_enforced", "complete_source_read_budget_verified",
    "runtime_code_sha256", "dcp_read_verification", "hf_content_verification",
)


class ImportKernelError(ValueError):
    def __init__(self, code):
        self.code = code if type(code) is str and code in ERROR_CODES else "conversion_failed"
        super().__init__(self.code)


def _need(value, code):
    if not value:
        raise ImportKernelError(code) from None


def _positive(value):
    return type(value) is int and 0 < value <= 2**63 - 1


def require_dependencies():
    """Import only the known stack, without fetching, running a model, or authorizing import."""
    try:
        for name in ("torch", "torch.distributed.checkpoint", "transformers", "safetensors",
                     "numpy", "lumi.configuration_lumi", "lumi.modeling_lumi"):
            importlib.import_module(name)
    except Exception:
        raise ImportKernelError("dependencies_unavailable") from None


def _private_path(path, *, absent=False):
    _need(path.is_absolute() and ".." not in path.parts, "path_rejected")
    for ancestor in (path, *path.parents):
        _need(not ancestor.is_symlink(), "path_rejected")
    if absent:
        _need(not path.exists(), "output_exists")
        _need(path.parent.is_dir(), "path_rejected")


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _need(key not in result, "configuration_rejected")
            result[key] = value
        return result

    def nonfinite(_value):
        raise ImportKernelError("configuration_rejected") from None

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    _need(type(value) is dict, "configuration_rejected")
    json.dumps(value, allow_nan=False)
    return value


def _read_config(path, content):
    return _json(content.read_control(path))

def _validate_source_config(payload):
    """Closed scalar source contract, before ML import; no HF loading switches.

    Unknown kwargs are rejected, not stripped. In particular a saved quantizer,
    remote-code mapping or attention loader cannot override the CPU-only reload.
    This admits the original 616-byte config and explicitly small fixtures.
    """
    _need(type(payload) is dict and len(payload) <= len(SOURCE_FIELDS)
          and set(payload) <= SOURCE_FIELDS, "configuration_rejected")
    integers = set(MODEL_FIELDS[:8])
    identifiers = {"pad_token_id", "bos_token_id", "eos_token_id", "doc_token_id"}
    for key, value in payload.items():
        if key in integers:
            _need(type(value) is int and 0 < value <= 1_000_000, "configuration_rejected")
        elif key in identifiers:
            _need(type(value) is int and 0 <= value < payload.get("vocab_size", 65536),
                  "configuration_rejected")
        elif key in {"attention_bias", "tie_word_embeddings", "use_cache"}:
            _need(type(value) is bool, "configuration_rejected")
        elif key in {"rope_theta", "rms_norm_eps", "initializer_range"}:
            _need(type(value) in {int, float} and math.isfinite(value) and value > 0,
                  "configuration_rejected")
        elif key == "model_type":
            _need(value == "lumi", "configuration_rejected")
        elif key == "architectures":
            _need(value == ["LumiForCausalLM"], "configuration_rejected")
        elif key == "initialization_strategy":
            _need(value in {"standard", "residual_scaled_v0.1"}, "configuration_rejected")
        elif key in {"torch_dtype", "dtype"}:
            _need(value in {"float32", "bfloat16"}, "configuration_rejected")
    _need(payload.get("num_hidden_layers", 36) <= 128, "configuration_rejected")
    if "dtype" in payload and "torch_dtype" in payload:
        _need(payload["dtype"] == payload["torch_dtype"], "configuration_rejected")


def _saved_configs_within_reserve(model, dtype):
    # Match the admitted save_pretrained config mutations before its first write.
    # Serialization is explicitly bounded as well as source kwargs being closed.
    model.config.dtype = dtype
    model.config.architectures = [type(model).__name__]
    expected = {}
    for name, config in (("config.json", model.config),
                         ("generation_config.json", model.generation_config)):
        if config is not None:
            raw = config.to_json_string(use_diff=True).encode("utf-8")
            _need(0 < len(raw) <= MAXIMUM_CONFIG_BYTES, "output_limit_exceeded")
            expected[name] = _json(raw)
    return expected


def _source_fields_preserved(saved_config, payload):
    return all(saved_config.get(key) == value for key, value in payload.items()
               if key not in {"torch_dtype", "dtype"})


def _require_complete_chunks(metadata, tick):
    """Original DCP complete, nonoverlapping chunk coverage, now bounded and checked."""
    shape = tuple(metadata.size)
    _need(0 < len(shape) <= 8 and all(_positive(bound) for bound in shape), "dcp_chunk_dimensions")
    _need(0 < len(metadata.chunks) <= MAXIMUM_CHUNKS_PER_TENSOR, "dcp_chunk_limit")
    boxes, volume = [], 0
    for chunk in metadata.chunks:
        tick()
        offsets, sizes = tuple(chunk.offsets), tuple(chunk.sizes)
        _need(len(offsets) == len(shape) and len(sizes) == len(shape), "dcp_chunk_dimensions")
        _need(all(type(start) is int and type(size) is int and start >= 0
                  and size > 0 and start + size <= bound
                  for start, size, bound in zip(offsets, sizes, shape, strict=True)),
              "dcp_chunk_bounds")
        for old_offsets, old_sizes in boxes:
            tick()
            _need(not all(left < right + width and right < left + length
                          for left, length, right, width in
                          zip(offsets, sizes, old_offsets, old_sizes, strict=True)),
                  "dcp_chunk_overlap")
        boxes.append((offsets, sizes))
        volume += math.prod(sizes)
    _need(volume == math.prod(shape), "dcp_chunk_coverage")


def _chunks(tensor, tick):
    _need(tensor.device.type == "cpu" and tensor.is_contiguous()
          and tensor.layout.__str__() == "torch.strided", "tensor_layout_rejected")
    flat = tensor.detach().view(-1)
    elements = max(1, CHUNK_BYTES // tensor.element_size())
    for start in range(0, flat.numel(), elements):
        tick()
        yield flat.narrow(0, start, min(elements, flat.numel() - start))
        tick()


def _finite(tensor, tick, code):
    import torch

    for chunk in _chunks(tensor, tick):
        _need(bool(torch.isfinite(chunk).all()), code)


def _equal(left, right, tick):
    import torch

    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    for a, b in zip(_chunks(left, tick), _chunks(right, tick), strict=True):
        if not torch.equal(a, b):
            return False
    return True


def _tensor_digest(state, tick):
    """Same v2 digest framing; each CPU byte view is at most CHUNK_BYTES, no full copy."""
    import torch

    digest = hashlib.sha256()
    for name in sorted(state):
        tick()
        tensor = state[name]
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                            separators=(",", ":"), ensure_ascii=True).encode("ascii")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update((tensor.numel() * tensor.element_size()).to_bytes(8, "big"))
        for chunk in _chunks(tensor, tick):
            # memoryview exposes existing bytes; unlike tobytes(), it creates no
            # tensor-sized Python allocation, including for BF16 tensors.
            digest.update(memoryview(chunk.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _output_inventory(path, maximum, tick, runtime_code, content):
    tick()
    rows = content.inventory()
    _need(0 < len(rows) <= MAXIMUM_OUTPUT_FILES, "output_inventory_rejected")
    total, weights = sum(row["bytes"] for row in rows), 0
    _need(total <= maximum, "output_limit_exceeded")
    names = {row["path"] for row in rows}
    for row in rows:
        tick()
        weights += bool(re.fullmatch(
            r"model(?:-[0-9]{5}-of-[0-9]{5})?\.safetensors", row["path"]))
        if row["path"] in runtime_code:
            _need(content.read(row["path"], expected_sha256=row["sha256"],
                               expected_bytes=row["bytes"]) == runtime_code[row["path"]],
                  "output_inventory_rejected")
    _need(weights > 0 and "config.json" in names and set(runtime_code) <= names,
          "output_inventory_rejected")
    return total, len(rows), weights


READ_CONTRACT_FIELDS = frozenset({
    "expected_manifest_sha256", "maximum_file_bytes", "maximum_read_bytes",
    "maximum_read_call_bytes",
})


def _read_contract(value):
    _need(type(value) is dict and len(value) == len(READ_CONTRACT_FIELDS)
          and set(value) == READ_CONTRACT_FIELDS, "dcp_read_contract_rejected")
    digest = value["expected_manifest_sha256"]
    _need(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
          "dcp_read_contract_rejected")
    _need(all(_positive(value[key]) for key in READ_CONTRACT_FIELDS - {"expected_manifest_sha256"})
          and value["maximum_read_call_bytes"] <= 1024**3,
          "dcp_read_contract_rejected")
    return dict(value)


HF_IO_FIELDS = frozenset({
    "maximum_read_bytes", "maximum_write_bytes", "maximum_file_bytes",
    "maximum_control_bytes", "target_shard_bytes",
})


def _hf_contract(value):
    _need(type(value) is dict and len(value) == len(HF_IO_FIELDS)
          and set(value) == HF_IO_FIELDS
          and all(_positive(item) for item in value.values()), "hf_io_contract_rejected")
    _need(value["maximum_control_bytes"] <= MAXIMUM_CONFIG_BYTES
          and value["maximum_file_bytes"] <= 2**31
          and value["target_shard_bytes"] <= SHARD_BYTES, "hf_io_contract_rejected")
    return dict(value)


def _convert(config_path, checkpoint_path, output_path, dtype, maximum_parameters,
             maximum_tensor_bytes, maximum_output_bytes, check_budget, checkpoint_read_contract,
             content, hf_io_contract):
    _need(all(_positive(value) for value in
              (maximum_parameters, maximum_tensor_bytes, maximum_output_bytes)), "limits_required")
    _need(callable(check_budget), "budget_callback_required")
    checkpoint_read_contract = _read_contract(checkpoint_read_contract)
    checks = 0

    def tick():
        nonlocal checks
        checks += 1
        try:
            check_budget()
        except Exception:
            raise ImportKernelError("budget_check_failed") from None

    tick()
    _need(dtype in {"float32", "bfloat16"}, "dtype_rejected")
    config_path, checkpoint_path, output_path = map(
        Path, (config_path, checkpoint_path, output_path)
    )
    _private_path(output_path, absent=True)
    _private_path(config_path)
    _private_path(checkpoint_path)
    _need(checkpoint_path.is_dir(), "path_rejected")
    _need(all(not output_path.is_relative_to(path) and not path.is_relative_to(output_path)
              for path in (config_path, checkpoint_path)), "path_rejected")
    payload = _read_config(config_path, content)
    _validate_source_config(payload)
    tick()
    require_dependencies()
    tick()

    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    from .configuration_lumi import LumiConfig
    from .modeling_lumi import LumiForCausalLM
    from .parameter_count import parameter_breakdown

    runtime_code = {}
    for module_name in ("lumi.configuration_lumi", "lumi.modeling_lumi"):
        tick()
        source_path = Path(importlib.import_module(module_name).__file__)
        _private_path(source_path)
        runtime_code[source_path.name] = content.read_control(source_path)
    _need(set(runtime_code) == {"configuration_lumi.py", "modeling_lumi.py"},
          "configuration_rejected")
    _need(not torch.distributed.is_initialized(), "distributed_group_rejected")
    _need(torch.get_default_dtype() == torch.float32, "default_dtype_rejected")
    config = LumiConfig(**payload)
    parameters = parameter_breakdown(config).total
    _need(_positive(parameters) and parameters <= maximum_parameters, "parameter_limit_exceeded")
    config._attn_implementation = "eager"
    original_config = {key: getattr(config, key) for key in MODEL_FIELDS}
    tick()
    with torch.device("meta"):
        model = LumiForCausalLM(config)
    _need(sum(parameter.numel() for parameter in model.parameters()) == parameters,
          "analytical_parameter_count_mismatch")
    model_state = model.state_dict()
    _need(0 < len(model_state) <= MAXIMUM_STATE_KEYS
          and all(tensor.device.type == "meta" for tensor in model_state.values()),
          "meta_architecture_rejected")
    parameter_names = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    buffer_dtypes = {name: tensor.dtype for name, tensor in model.named_buffers()}
    _need(not parameter_names.intersection(buffer_dtypes)
          and parameter_names | set(buffer_dtypes) == set(model_state), "role_inventory_mismatch")
    _need(all(value == torch.float32 for value in buffer_dtypes.values()), "buffer_dtype_rejected")
    shapes = {name: tuple(tensor.shape) for name, tensor in model_state.items()}
    _need(all(0 < len(name) <= 512 for name in shapes), "dcp_key_inventory")
    tick()
    with verified_dcp_reader(checkpoint_path, check_budget=tick,
                             **checkpoint_read_contract) as lease:
        reader = lease.reader
        metadata = reader.read_metadata().state_dict_metadata
        tick()
        _need(set(metadata) == {f"model.{name}" for name in shapes}, "dcp_key_inventory")
        output_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
        output_element_bytes = 4 if dtype == "float32" else 2
        source_bytes = cast_bytes = largest = 0
        for name, shape in shapes.items():
            tick()
            entry = metadata[f"model.{name}"]
            _need(isinstance(entry, TensorStorageMetadata), "dcp_metadata_type")
            _need(tuple(entry.size) == shape, "dcp_shape_inventory")
            _need(entry.properties.dtype in {torch.float32, torch.bfloat16}, "dcp_dtype_rejected")
            _require_complete_chunks(entry, tick)
            size = math.prod(shape)
            source_size = size * (4 if entry.properties.dtype == torch.float32 else 2)
            cast_size = size * (output_element_bytes if name in parameter_names else 4)
            source_bytes += source_size
            cast_bytes += cast_size
            largest = max(largest, source_size, cast_size)
        projected_live = max(source_bytes + cast_bytes, 2 * cast_bytes) + largest + 2 * CHUNK_BYTES
        projected_output = cast_bytes + OUTPUT_OVERHEAD_BYTES
        _need(projected_live <= maximum_tensor_bytes, "tensor_limit_exceeded")
        _need(projected_output <= maximum_output_bytes, "output_limit_exceeded")
        # Generated config/index and tensor-header overhead must fit the reserved
        # allowance before writing, not merely be noticed in the final inventory.
        _need(MAXIMUM_CONFIG_BYTES * 2 + sum(map(len, runtime_code.values()))
              + sum(len(name) * 4 + 2048 for name in shapes)
              <= OUTPUT_OVERHEAD_BYTES, "output_limit_exceeded")
        destinations = {}
        for name, shape in shapes.items():
            tick()
            destinations[name] = torch.empty(
                shape, dtype=metadata[f"model.{name}"].properties.dtype, device="cpu"
            )
        tick()
        dcp.load_state_dict({"model": destinations}, storage_reader=reader,
                            planner=DefaultLoadPlanner(allow_partial_load=False), no_dist=True)
        tick()
        for name, tensor in destinations.items():
            tick()
            entry = metadata[f"model.{name}"]
            _need(tensor.device.type == "cpu" and tensor.dtype == entry.properties.dtype
                  and tuple(tensor.shape) == tuple(entry.size), "dcp_restored_inventory")
            _finite(tensor, tick, "dcp_nonfinite")
        if config.tie_word_embeddings:
            _need(_equal(destinations["model.embed_tokens.weight"],
                         destinations["lm_head.weight"], tick), "tied_alias_mismatch")
        source_digest = _tensor_digest(destinations, tick)
        expected = {}
        for name in shapes:
            tick()
            tensor = destinations.pop(name)
            expected[name] = tensor.to(
                dtype=output_dtype if name in parameter_names else buffer_dtypes[name], device="cpu"
            )
            _finite(expected[name], tick, "cast_nonfinite")
            del tensor
        del destinations, metadata, reader, model_state
        dcp_read_verification = lease.summary()
    tick()
    incompatible = model.load_state_dict(expected, strict=True, assign=True)
    _need(not incompatible.missing_keys and not incompatible.unexpected_keys,
          "strict_restore_failed")
    # Source aliases were independent and equal before casting. Restore the
    # architecture's identity sharing only after assign=True preserved all keys.
    model.tie_weights()
    _need(sum(parameter.numel() for parameter in model.parameters()) == parameters,
          "analytical_parameter_count_mismatch")
    restored = model.state_dict()
    for name, value in expected.items():
        _need(_equal(restored[name], value, tick), "cast_restore_mismatch")
    expected = restored
    cast_digest = _tensor_digest(expected, tick)
    # Suppress HF's copy-with-source-mode behavior on these two instances only.
    # Copy the same exact runtime modules ourselves with exclusive private writes.
    # Preserve auto_map and the self-contained package; no unknown code is admitted.
    model._auto_class = None
    model.config._auto_class = None
    expected_saved_configs = _saved_configs_within_reserve(model, output_dtype)
    tick()
    documents = {"config.json": model.config.to_json_string(use_diff=True).encode("utf-8")}
    if model.generation_config is not None:
        documents["generation_config.json"] = model.generation_config.to_json_string(
            use_diff=True).encode("utf-8")
    _need(all(0 < len(raw) <= hf_io_contract["maximum_control_bytes"]
              and _json(raw) == expected_saved_configs[name] for name, raw in documents.items()),
          "output_config_changed")
    export = prepare_export(
        model, target_shard_bytes=hf_io_contract["target_shard_bytes"],
        maximum_serialized_shard_bytes=hf_io_contract["maximum_file_bytes"],
        maximum_header_bytes=hf_io_contract["maximum_control_bytes"], check_budget=tick)
    projected_live = max(projected_live, 2 * cast_bytes + export["serialization_scratch_bytes"])
    _need(projected_live <= maximum_tensor_bytes, "tensor_limit_exceeded")
    required_write = (export["maximum_weight_and_index_bytes"]
                      + sum(map(len, documents.values())) + sum(map(len, runtime_code.values())))
    _need(required_write <= maximum_output_bytes
          and required_write <= hf_io_contract["maximum_write_bytes"], "output_limit_exceeded")
    initial_reads = content.summary()["hf_content_read_bytes_reserved"]
    _need(initial_reads + required_write + sum(map(len, runtime_code.values()))
          <= hf_io_contract["maximum_read_bytes"], "hf_io_contract_rejected")
    content.create_output(output_path)
    for name, raw in runtime_code.items():
        content.write(name, raw)
    write_export(content, export, documents, check_budget=tick)
    output_bytes, output_files, weight_files = _output_inventory(
        output_path, maximum_output_bytes, tick, runtime_code, content)
    saved_documents = read_documents(content, documents)
    saved_config = saved_documents["config.json"]
    _need(all(saved_documents[name] == value for name, value in expected_saved_configs.items()),
          "output_config_changed")
    _need(all(saved_config.get(key) == value for key, value in original_config.items()),
          "output_config_changed")
    _need(_source_fields_preserved(saved_config, payload), "output_config_changed")
    loaded = reload_state(content, export, check_budget=tick)
    from transformers import GenerationConfig

    generation = (GenerationConfig.from_dict(saved_documents["generation_config.json"])
                  if "generation_config.json" in saved_documents else None)
    tick()
    with torch.device("cpu"):
        reloaded, loading_info = LumiForCausalLM.from_pretrained(
            None, config=LumiConfig(**saved_config), state_dict=loaded,
            generation_config=generation,
            local_files_only=True, dtype=output_dtype, attn_implementation="eager", device_map=None,
            use_safetensors=True, trust_remote_code=False, output_loading_info=True)
    _need(type(loading_info) is dict and loading_info
          and all(not value for value in loading_info.values()), "hf_loading_info_rejected")
    hf_scratch_bytes = export["serialization_scratch_bytes"]
    hf_serialized_shard_bytes = export["maximum_serialized_bytes"]
    del loaded, export
    tick()
    observed = reloaded.state_dict()
    _need(set(observed) == set(expected), "reload_key_inventory")
    for name, tensor in observed.items():
        _need(_equal(tensor, expected[name], tick), "reload_tensor_mismatch")
        _finite(tensor, tick, "reload_tensor_mismatch")
    if config.tie_word_embeddings:
        _need(reloaded.get_input_embeddings().weight.data_ptr()
              == reloaded.get_output_embeddings().weight.data_ptr(), "reload_tied_mismatch")
    _need(sum(parameter.numel() for parameter in reloaded.parameters()) == parameters,
          "analytical_parameter_count_mismatch")
    reload_digest = _tensor_digest(observed, tick)
    _need(reload_digest == cast_digest, "reload_digest_mismatch")
    _need(all(getattr(reloaded.config, key) == value for key, value in original_config.items()),
          "output_config_changed")
    _need(_output_inventory(output_path, maximum_output_bytes, tick, runtime_code, content)
          == (output_bytes, output_files, weight_files), "output_inventory_rejected")
    tick()
    evidence = {
        "schema_version": "lumi.sft-checkpoint-import-kernel.v0.5",
        "status": "PASS_cpu_kernel_roundtrip_not_execution_authority", "dtype": dtype,
        "parameter_dtype": dtype, "buffer_dtype_policy": "configured_buffer_dtype_preserve_values",
        "model_parameters": parameters, "model_tensor_count": len(expected),
        "source_tensor_sha256": source_digest, "cast_tensor_sha256": cast_digest,
        "reload_tensor_sha256": reload_digest, "source_state_bytes": source_bytes,
        "cast_state_bytes": cast_bytes, "largest_tensor_bytes": largest,
        "projected_live_tensor_bytes": projected_live, "projected_output_bytes": projected_output,
        "output_bytes": output_bytes, "output_file_count": output_files,
        "safetensor_file_count": weight_files, "chunk_bytes": CHUNK_BYTES, "budget_checks": checks,
        "complete_keys_shapes_verified": True, "hf_reload_tensor_equal": True,
        "strict_assign_verified": True, "tied_aliases_verified": bool(config.tie_word_embeddings),
        "source_dicts_released_before_reload": True, "configuration_fields_preserved": True,
        "cpu_only": True, "no_forward_pass": True, "real_4b_import": False,
        "training_authorized": False, "production_eligible": False,
        "os_resource_limits_enforced": False, "complete_source_read_budget_verified": False,
        "dcp_read_verification": dcp_read_verification,
        "hf_content_verification": {
            **content.summary(), "serialization_scratch_bytes": hf_scratch_bytes,
            "maximum_serialized_shard_bytes": hf_serialized_shard_bytes,
            "actual_HF_from_pretrained_with_in_memory_state": True,
            "native_path_checkpoint_read_or_write_used": False,
        },
        "runtime_code_sha256": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(runtime_code.items())
        },
    }
    _need(set(evidence) == set(EVIDENCE_KEYS), "conversion_failed")
    return evidence


def convert_and_verify(config_path, checkpoint_path, output_path, dtype, *, maximum_parameters,
                       maximum_tensor_bytes, maximum_output_bytes, check_budget,
                       checkpoint_read_contract, hf_io_contract):
    """Internal CPU kernel with mandatory independently chosen limits and callback.

    Failure preserves any new partial output. The parent owns its fresh attempt
    marker, authorization, source checksums, payload-read accounting and cleanup.
    Nothing here accepts approval or lifts an old adapter's synthetic-only cap.
    """
    content = None
    try:
        hf_io_contract = _hf_contract(hf_io_contract)
        content = ContentIO(**{key: value for key, value in hf_io_contract.items()
                              if key != "target_shard_bytes"}, check_budget=check_budget)
        result = _convert(config_path, checkpoint_path, output_path, dtype, maximum_parameters,
                          maximum_tensor_bytes, maximum_output_bytes, check_budget,
                          checkpoint_read_contract, content, hf_io_contract)
    except ImportKernelError:
        raise
    except DCPReadError as error:
        # The context scrubs caller exceptions too. Preserve only this module's
        # exact typed, allowlisted validation failure, never exception text.
        # Cleanup/read failures take precedence and do not enter this branch.
        if error.code == "operation_failed" and type(error.__context__) is ImportKernelError:
            raise ImportKernelError(error.__context__.code) from None
        raise ImportKernelError("dcp_read_failed") from None
    except (ContentIOError, HFBytesError):
        raise ImportKernelError("hf_io_failed") from None
    except Exception:
        raise ImportKernelError("conversion_failed") from None
    finally:
        if content is not None:
            try:
                content.close()
            except BaseException:
                raise ImportKernelError("hf_io_failed") from None
    return result
