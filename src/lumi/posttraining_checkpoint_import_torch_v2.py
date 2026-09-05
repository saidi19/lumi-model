"""Synthetic-only, CPU DCP-to-HF import backend; no training or inference.

The caller must first authenticate and bound the checkpoint, configuration and
all paths. This backend does not authorize a real model import, publish a
receipt or write a worker manifest. ML dependencies are deliberately lazy.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any

MAXIMUM_SYNTHETIC_PARAMETERS = 1_000_000


def require_dependencies() -> None:
    """Fail explicitly before an attempt if the genuine ML stack is unavailable."""
    try:
        for module in (
            "torch",
            "torch.distributed.checkpoint",
            "transformers",
            "safetensors",
            "numpy",
            "lumi.configuration_lumi",
            "lumi.modeling_lumi",
        ):
            importlib.import_module(module)
    except Exception as error:  # noqa: BLE001 - return only an allowlisted stage code
        raise RuntimeError("ML_dependencies_unavailable") from error


def _tensor_digest(state: dict[str, Any]) -> str:
    """Hash all keys, shapes, dtypes and bytes, retaining no tensor values."""
    import torch

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        header = json.dumps(
            [name, str(tensor.dtype), list(tensor.shape)],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        # Viewing bytes avoids NumPy's lack of a native BF16 dtype.
        payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _require_complete_chunks(metadata: Any) -> None:
    """Reject overlapping or incomplete storage coverage before tensor reads."""
    shape = tuple(metadata.size)
    boxes = []
    volume = 0
    for chunk in metadata.chunks:
        offsets, sizes = tuple(chunk.offsets), tuple(chunk.sizes)
        if len(offsets) != len(shape) or len(sizes) != len(shape):
            raise ValueError("DCP chunk dimensionality mismatch")
        if any(
            start < 0 or size <= 0 or start + size > bound
            for start, size, bound in zip(offsets, sizes, shape, strict=True)
        ):
            raise ValueError("DCP chunk exceeds tensor bounds")
        if any(
            all(
                left < right + width and right < left + length
                for left, length, right, width in zip(
                    offsets, sizes, old_offsets, old_sizes, strict=True
                )
            )
            for old_offsets, old_sizes in boxes
        ):
            raise ValueError("DCP chunks overlap")
        boxes.append((offsets, sizes))
        volume += math.prod(sizes)
    if not boxes or volume != math.prod(shape):
        raise ValueError("DCP chunks do not cover the complete tensor")


def convert_and_verify(
    config_path: Path, checkpoint_path: Path, output_path: Path, dtype: str
) -> dict[str, Any]:
    """Restore at most one million synthetic parameters and verify HF tensors.

    All allocation and reload are CPU-only. No model forward, generation,
    optimizer, process-group initialization or automatic retry is performed.
    A failed output is intentionally not erased or reused by this backend.
    """
    if dtype not in {"float32", "bfloat16"}:
        raise ValueError("synthetic import dtype must be float32 or bfloat16")
    config_path, checkpoint_path, output_path = map(
        Path, (config_path, checkpoint_path, output_path)
    )
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("synthetic import output already exists")
    require_dependencies()

    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    from .configuration_lumi import LumiConfig
    from .modeling_lumi import LumiForCausalLM
    from .parameter_count import parameter_breakdown

    if torch.distributed.is_initialized():
        raise ValueError("synthetic import requires no distributed process group")
    payload = json.loads(config_path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError("synthetic import configuration must be an object")
    config = LumiConfig(**payload)
    parameters = parameter_breakdown(config).total
    if not 0 < parameters <= MAXIMUM_SYNTHETIC_PARAMETERS:
        raise ValueError("synthetic import parameter cap exceeded")
    config._attn_implementation = "eager"
    # Explicit device context prevents caller default-device settings from
    # allocating on CUDA, MPS, or a meta device.
    with torch.device("cpu"):
        model = LumiForCausalLM(config).float()
    if sum(parameter.numel() for parameter in model.parameters()) != parameters:
        raise ValueError("analytical and allocated parameter counts differ")
    model_state = model.state_dict()
    if any(tensor.device.type != "cpu" for tensor in model_state.values()):
        raise ValueError("synthetic import model is not CPU-only")

    reader = dcp.FileSystemReader(checkpoint_path / "model")
    metadata = reader.read_metadata().state_dict_metadata
    expected_keys = {f"model.{name}" for name in model_state}
    if set(metadata) != expected_keys:
        raise ValueError("DCP model key inventory mismatch")
    allowed_dtypes = {torch.float32, torch.bfloat16}
    destinations = {}
    for name, model_tensor in model_state.items():
        entry = metadata[f"model.{name}"]
        if not isinstance(entry, TensorStorageMetadata):
            raise ValueError("DCP model entry is not tensor metadata")
        if tuple(entry.size) != tuple(model_tensor.shape):
            raise ValueError("DCP model shape inventory mismatch")
        if entry.properties.dtype not in allowed_dtypes:
            raise ValueError("DCP model dtype is unsupported")
        _require_complete_chunks(entry)
        # Independent tensors preserve both values of tied aliases so a
        # contradictory source alias cannot be hidden by an in-place load.
        destinations[name] = torch.empty(
            tuple(model_tensor.shape), dtype=entry.properties.dtype, device="cpu"
        )
    dcp.load_state_dict(
        {"model": destinations},
        storage_reader=reader,
        planner=DefaultLoadPlanner(allow_partial_load=False),
        no_dist=True,
    )
    for name, tensor in destinations.items():
        entry = metadata[f"model.{name}"]
        if (
            tensor.device.type != "cpu"
            or tensor.dtype != entry.properties.dtype
            or tuple(tensor.shape) != tuple(entry.size)
        ):
            raise ValueError("DCP restored tensor inventory mismatch")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("DCP restored tensor is nonfinite")
    if config.tie_word_embeddings:
        embedding = destinations["model.embed_tokens.weight"]
        head = destinations["lm_head.weight"]
        if embedding.dtype != head.dtype or not torch.equal(embedding, head):
            raise ValueError("DCP tied aliases disagree")
    source_digest = _tensor_digest(destinations)
    output_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    parameter_names = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    buffer_dtypes = {name: tensor.dtype for name, tensor in model.named_buffers()}
    if parameter_names | set(buffer_dtypes) != set(destinations):
        raise ValueError("parameter and buffer role inventory mismatch")
    # HF initializes floating buffers with the architecture's explicit dtype.
    # Cast parameters only; preserve buffer values at their declared dtype.
    expected = {
        name: tensor.to(output_dtype if name in parameter_names else buffer_dtypes[name])
        for name, tensor in destinations.items()
    }
    if any(not bool(torch.isfinite(tensor).all()) for tensor in expected.values()):
        raise ValueError("DCP output dtype cast is nonfinite")
    model.to(dtype=output_dtype, device="cpu")
    for name, dtype_for_buffer in buffer_dtypes.items():
        parent, _, leaf = name.rpartition(".")
        module = model.get_submodule(parent) if parent else model
        module._buffers[leaf] = module._buffers[leaf].to(dtype=dtype_for_buffer)
    incompatible = model.load_state_dict(expected, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("DCP strict model restore failed")
    if any(not torch.equal(model.state_dict()[name], value) for name, value in expected.items()):
        raise ValueError("DCP restored model differs from cast source")
    cast_digest = _tensor_digest(expected)

    # Exclusive creation prevents reusing even an empty existing output.
    output_path.mkdir(mode=0o700, parents=False, exist_ok=False)
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size="32MB")
    if not list(output_path.glob("model*.safetensors")):
        raise ValueError("synthetic HF export produced no safetensors weights")
    with torch.device("cpu"):
        reloaded = LumiForCausalLM.from_pretrained(
            output_path,
            local_files_only=True,
            dtype=output_dtype,
            attn_implementation="eager",
        )
    observed = reloaded.state_dict()
    if set(observed) != set(expected):
        raise ValueError("HF reload key inventory mismatch")
    for name, tensor in observed.items():
        if (
            tensor.device.type != "cpu"
            or tensor.dtype != expected[name].dtype
            or tensor.shape != expected[name].shape
            or not torch.equal(tensor, expected[name])
        ):
            raise ValueError("HF reload tensor mismatch")
    if config.tie_word_embeddings and (
        reloaded.get_input_embeddings().weight.data_ptr()
        != reloaded.get_output_embeddings().weight.data_ptr()
    ):
        raise ValueError("HF reload did not preserve tied parameters")
    reload_digest = _tensor_digest(observed)
    if reload_digest != cast_digest:
        raise ValueError("HF reload tensor digest mismatch")
    return {
        "complete_keys_shapes_verified": True,
        "dtype": dtype,
        "parameter_dtype": dtype,
        "buffer_dtype_policy": "configured_buffer_dtype_preserve_values",
        "hf_reload_tensor_equal": True,
        "reload_tensor_sha256": reload_digest,
        "model_parameters": parameters,
        "model_tensor_count": len(expected),
        "no_forward_pass": True,
        "cast_tensor_sha256": cast_digest,
        "source_tensor_sha256": source_digest,
        "synthetic_only": True,
    }
