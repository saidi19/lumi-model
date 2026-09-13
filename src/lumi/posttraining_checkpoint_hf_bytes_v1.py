"""Explicit HF-compatible safetensors bytes, never pathname-native content I/O.

Only already validated CPU Lumi states are accepted. The public safetensors
bytes API is used one bounded shard at a time. A caller-owned ContentIO performs
every file read/write; this module never opens a file or patches library globals.
Planning includes serialized-shard scratch, not measured RSS or OS enforcement.
The original model/config/tensor validation and actual HF reload remain outside
this component. These helpers are not a real-import entrypoint or authorization.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import OrderedDict

MAXIMUM_KEYS = 4096
MAXIMUM_HEADER = 1048576
MAXIMUM_SHARD = 2**31
WEIGHT = re.compile(r"model(?:-[0-9]{5}-of-[0-9]{5})?\.safetensors\Z")
ERROR_CODES = frozenset({"invalid_arguments", "state_rejected", "alias_rejected", "shard_limit",
                         "header_rejected", "serialization_failed", "reload_failed",
                         "budget_failed"})


class HFBytesError(ValueError):
    def __init__(self, code):
        self.code = code if type(code) is str and code in ERROR_CODES else "serialization_failed"
        super().__init__(self.code)


def _need(value, code):
    if not value:
        raise HFBytesError(code) from None


def _tick(callback):
    try:
        _need(callback() is None, "budget_failed")
    except BaseException:
        raise HFBytesError("budget_failed") from None


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "header_rejected")
            result[key] = value
        return result

    def constant(_value):
        raise HFBytesError("header_rejected") from None

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def prepare_export(model, *, target_shard_bytes, maximum_serialized_shard_bytes,
                   maximum_header_bytes, check_budget):
    """Bound all tensor/shard metadata and scratch before serialization or output."""
    _need(callable(check_budget) and sys.byteorder == "little", "invalid_arguments")
    _need(type(target_shard_bytes) is int and 0 < target_shard_bytes <= MAXIMUM_SHARD
          and type(maximum_serialized_shard_bytes) is int
          and 0 < maximum_serialized_shard_bytes <= MAXIMUM_SHARD
          and type(maximum_header_bytes) is int and 0 < maximum_header_bytes <= MAXIMUM_HEADER,
          "invalid_arguments")
    _tick(check_budget)
    import torch
    from huggingface_hub import split_torch_state_dict_into_shards

    state = model.state_dict()
    _need(type(state) in {dict, OrderedDict} and 0 < len(state) <= MAXIMUM_KEYS, "state_rejected")
    state = dict(state)
    shapes, pointers = {}, {}
    for name, tensor in state.items():
        _tick(check_budget)
        _need(type(name) is str and 0 < len(name) <= 512 and name != "__metadata__"
              and isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
              and tensor.layout == torch.strided and tensor.is_contiguous()
              and tensor.storage_offset() == 0 and 0 < tensor.ndim <= 8
              and tensor.dtype in {torch.float32, torch.bfloat16}, "state_rejected")
        size = tensor.numel() * tensor.element_size()
        _need(0 < size <= MAXIMUM_SHARD and tensor.untyped_storage().nbytes() == size,
              "state_rejected")
        shapes[name] = {"dtype": "F32" if tensor.dtype == torch.float32 else "BF16",
                        "shape": list(tensor.shape), "bytes": size}
        pointers.setdefault(tensor.untyped_storage().data_ptr(), []).append(name)
    duplicates = [set(names) for names in pointers.values() if len(names) > 1]
    alias, source = "lm_head.weight", "model.embed_tokens.weight"
    tied = model.config.tie_word_embeddings
    _need(type(tied) is bool, "alias_rejected")
    if tied:
        _need(duplicates == [{alias, source}] and shapes[alias] == shapes[source], "alias_rejected")
        state.pop(alias)
        shapes.pop(alias)
    else:
        _need(not duplicates, "alias_rejected")
    split = split_torch_state_dict_into_shards(
        state, filename_pattern="model{suffix}.safetensors", max_shard_size=target_shard_bytes)
    _need(0 < len(split.filename_to_tensors) <= MAXIMUM_KEYS
          and set(split.tensor_to_filename) == set(state), "state_rejected")
    shards, largest, total = [], 0, 0
    seen = set()
    for filename, names in split.filename_to_tensors.items():
        _tick(check_budget)
        _need(type(filename) is str and WEIGHT.fullmatch(filename)
              and type(names) is list and names and not seen.intersection(names), "state_rejected")
        seen.update(names)
        size = sum(shapes[name]["bytes"] for name in names)
        # All offsets are <= size. Using size twice plus per-key slack bounds
        # JSON escaping, Rust serialization punctuation and 8-byte padding.
        header = {name: {"dtype": shapes[name]["dtype"], "shape": shapes[name]["shape"],
                         "data_offsets": [size, size]} for name in names}
        header["__metadata__"] = {"format": "pt"}
        header_bound = len(_bytes(header)) + 64 * len(names) + 64
        header_bound = (header_bound + 7) // 8 * 8
        serialized = size + 8 + header_bound
        _need(header_bound <= maximum_header_bytes
              and serialized <= maximum_serialized_shard_bytes, "shard_limit")
        shards.append({"path": filename, "names": tuple(names), "payload_bytes": size,
                       "header_bound": header_bound, "serialized_bound": serialized})
        largest, total = max(largest, serialized), total + serialized
    _need(seen == set(state), "state_rejected")
    index = None
    if split.is_sharded:
        index = {"metadata": {"total_parameters": model.num_parameters(), **split.metadata},
                 "weight_map": split.tensor_to_filename}
    index_raw = None if index is None else _bytes(index) + b"\n"
    _need(index_raw is None or len(index_raw) <= maximum_header_bytes, "shard_limit")
    _tick(check_budget)
    return {"state": state, "shapes": shapes, "shards": shards, "index_raw": index_raw,
            "maximum_serialized_bytes": largest,
            "maximum_weight_and_index_bytes": total + (len(index_raw) if index_raw else 0),
            # Four shard-sized buffers plus bounded loop/control scratch. This
            # is conservative visible staging, not a native allocator/RSS proof.
            "serialization_scratch_bytes": 4 * largest + 2 * 1048576}


def validate_blob(raw, shard, shapes):
    """Reject malformed/unexpected headers and extents before tensor allocation."""
    _need(type(raw) is bytes and 8 < len(raw) <= shard["serialized_bound"], "header_rejected")
    header_size = int.from_bytes(raw[:8], "little")
    _need(0 < header_size <= shard["header_bound"] and header_size % 8 == 0
          and 8 + header_size + shard["payload_bytes"] == len(raw), "header_rejected")
    try:
        header = _json(raw[8:8 + header_size])
        _need(type(header) is dict and set(header) == set(shard["names"]) | {"__metadata__"}
              and header["__metadata__"] == {"format": "pt"}, "header_rejected")
        ranges = []
        for name in shard["names"]:
            row, expected = header[name], shapes[name]
            _need(type(row) is dict and set(row) == {"dtype", "shape", "data_offsets"}
                  and row["dtype"] == expected["dtype"] and row["shape"] == expected["shape"]
                  and all(type(value) is int for value in row["shape"]), "header_rejected")
            offsets = row["data_offsets"]
            _need(type(offsets) is list and len(offsets) == 2
                  and all(type(value) is int for value in offsets)
                  and 0 <= offsets[0] < offsets[1] <= shard["payload_bytes"]
                  and offsets[1] - offsets[0] == expected["bytes"], "header_rejected")
            ranges.append(offsets)
        cursor = 0
        for start, end in sorted(ranges):
            _need(start == cursor, "header_rejected")
            cursor = end
        _need(cursor == shard["payload_bytes"], "header_rejected")
    except HFBytesError:
        raise
    except BaseException:
        raise HFBytesError("header_rejected") from None


def write_export(content, plan, documents, *, check_budget):
    """Write each planned shard once; old/partial output is never reused."""
    from safetensors.torch import save

    _need(type(documents) is dict and {"config.json"} <= set(documents)
          <= {"config.json", "generation_config.json"}, "invalid_arguments")
    for name, raw in documents.items():
        content.write(name, raw)
    for shard in plan["shards"]:
        _tick(check_budget)
        tensors = {name: plan["state"][name] for name in shard["names"]}
        try:
            raw = save(tensors, metadata={"format": "pt"})
        except BaseException:
            raise HFBytesError("serialization_failed") from None
        _tick(check_budget)
        validate_blob(raw, shard, plan["shapes"])
        content.write(shard["path"], raw)
        del raw, tensors
    if plan["index_raw"] is not None:
        content.write("model.safetensors.index.json", plan["index_raw"])
    _tick(check_budget)


def reload_state(content, plan, *, check_budget):
    """Read hash-bound serialized shards; expose tensors, not paths, to HF."""
    import torch
    from safetensors.torch import load

    rows = {row["path"]: row for row in content.inventory()}
    state = {}
    if plan["index_raw"] is not None:
        row = rows["model.safetensors.index.json"]
        raw = content.read(row["path"], expected_sha256=row["sha256"], expected_bytes=row["bytes"])
        _need(raw == plan["index_raw"], "reload_failed")
        del raw
    for shard in plan["shards"]:
        _tick(check_budget)
        row = rows[shard["path"]]
        raw = content.read(row["path"], expected_sha256=row["sha256"], expected_bytes=row["bytes"])
        validate_blob(raw, shard, plan["shapes"])
        try:
            tensors = load(raw)
        except BaseException:
            raise HFBytesError("reload_failed") from None
        _tick(check_budget)
        _need(set(tensors) == set(shard["names"]) and not set(state).intersection(tensors),
              "reload_failed")
        for name, tensor in tensors.items():
            expected = plan["shapes"][name]
            dtype = torch.float32 if expected["dtype"] == "F32" else torch.bfloat16
            _need(tensor.device.type == "cpu" and tensor.dtype == dtype
                  and list(tensor.shape) == expected["shape"], "reload_failed")
        state.update(tensors)
        del raw, tensors
    _need(set(state) == set(plan["state"]), "reload_failed")
    _tick(check_budget)
    return state


def read_documents(content, expected):
    """Use write receipts to bind every config read; return parsed exact JSON."""
    rows = {row["path"]: row for row in content.inventory()}
    observed = {}
    for name, raw in expected.items():
        row = rows[name]
        _need(row["bytes"] == len(raw) and row["sha256"] == hashlib.sha256(raw).hexdigest(),
              "reload_failed")
        actual = content.read(name, expected_sha256=row["sha256"], expected_bytes=row["bytes"])
        _need(actual == raw, "reload_failed")
        observed[name] = _json(actual)
    return observed
