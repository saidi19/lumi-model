"""Strict SFT import contract and synthetic-only packaging, never a 4B launch permit.

The lineage validator supports the final base metadata contract. The executable
entry point is deliberately restricted to small, newly-created CPU fixtures.
Actual final-4B export needs a separately bounded execution successor.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

RECEIPT = "sft-import-receipt-v0.2.json"
MANIFEST = "manifest.sha256"
STARTED = "import-started-v0.2.json"
SUCCESS = "import-complete-v0.2.json"
STATE = "distributed-training-state.json"
SHA = re.compile(r"[a-f0-9]{64}")
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TREE_BYTES = 64 * 1024 * 1024
MODEL_FIELDS = (
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim",
    "max_position_embeddings", "rope_theta", "rms_norm_eps", "attention_bias",
    "tie_word_embeddings", "pad_token_id", "bos_token_id", "eos_token_id", "doc_token_id",
)
MODEL_EVIDENCE_KEYS = {
    "dtype", "model_parameters", "model_tensor_count", "source_tensor_sha256",
    "cast_tensor_sha256", "reload_tensor_sha256", "complete_keys_shapes_verified",
    "hf_reload_tensor_equal", "no_forward_pass", "synthetic_only",
    "parameter_dtype", "buffer_dtype_policy",
}


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_object(raw: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique)
    require(isinstance(value, dict), "json_object_required")
    canonical(value)  # Reject NaN/Infinity as well as unsupported values.
    return value


def relative_name(name: str) -> str:
    require(isinstance(name, str) and bool(name), "invalid_relative_name")
    require(not name.startswith("/") and "\\" not in name, "invalid_relative_name")
    require(all(part not in {"", ".", ".."} for part in name.split("/")),
            "invalid_relative_name")
    require(not any(ord(c) < 33 or ord(c) == 127 for c in name), "invalid_relative_name")
    return name


def parse_manifest(raw: bytes) -> dict[str, str]:
    require(bool(raw) and raw.endswith(b"\n") and b"\r" not in raw, "manifest_line_endings")
    entries = {}
    for line in raw.decode("utf-8").splitlines():
        fields = line.split("  ")
        require(len(fields) == 2 and SHA.fullmatch(fields[0]) is not None,
                "manifest_line_invalid")
        name = relative_name(fields[1])
        require(name != MANIFEST and name not in entries, "manifest_duplicate_or_self")
        entries[name] = fields[0]
    return entries


def _identity(info: os.stat_result) -> tuple:
    return tuple(getattr(info, key) for key in (
        "st_dev", "st_ino", "st_mode", "st_uid", "st_nlink", "st_size",
        "st_mtime_ns", "st_ctime_ns",
    ))


def _path(root: Path, name: str) -> Path:
    _no_symlink_components(root)
    require(not root.is_symlink() and root.is_dir(), "unsafe_root")
    current = root
    for part in relative_name(name).split("/"):
        current /= part
        require(not current.is_symlink(), "symlink_path")
    require(current.resolve().is_relative_to(root.resolve()), "escaped_root")
    return current


def _no_symlink_components(path: Path) -> None:
    current = Path(os.path.abspath(path)).anchor
    current = Path(current)
    for part in Path(os.path.abspath(path)).parts[1:]:
        current /= part
        require(not current.is_symlink(), "symlink_path_component")


def read_regular(root: Path, name: str) -> bytes:
    path = _path(root, name)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "unsafe_regular_file")
    require(before.st_size <= MAX_FILE_BYTES, "synthetic_file_size_cap")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb", buffering=0) as handle:
        require(_identity(os.fstat(handle.fileno())) == _identity(before), "input_changed")
        raw = handle.read(before.st_size + 1)
        require(_identity(os.fstat(handle.fileno())) == _identity(before), "input_changed")
    require(len(raw) == before.st_size and _identity(path.lstat()) == _identity(before),
            "input_changed")
    return raw


def tree_inventory(root: Path, *, private: bool = False) -> list[dict]:
    _no_symlink_components(root)
    require(root.is_dir() and not root.is_symlink(), "unsafe_tree_root")
    files = []
    total = 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        current = Path(directory)
        info = current.lstat()
        require(not current.is_symlink(), "symlink_directory")
        if private:
            require(stat.S_IMODE(info.st_mode) == 0o700 and info.st_uid == os.geteuid(),
                    "unsafe_private_directory")
        for name in dirs:
            require(not (current / name).is_symlink(), "symlink_directory")
        for name in names:
            path = current / name
            relative = path.relative_to(root).as_posix()
            info = path.lstat()
            if private:
                require(stat.S_IMODE(info.st_mode) == 0o600 and info.st_uid == os.geteuid(),
                        "unsafe_private_file")
            total += info.st_size
            require(total <= MAX_TREE_BYTES and len(files) < 256, "synthetic_tree_cap")
            raw = read_regular(root, relative)
            files.append({"path": relative, "bytes": len(raw), "sha256": sha(raw)})
    return sorted(files, key=lambda item: item["path"])


def verify_closed_manifest(root: Path, expected: str, *, private: bool = False) -> list[dict]:
    raw = read_regular(root, MANIFEST)
    require(sha(raw) == expected, "manifest_binding_mismatch")
    declared = parse_manifest(raw)
    inventory = tree_inventory(root, private=private)
    actual = {x["path"]: x["sha256"] for x in inventory if x["path"] != MANIFEST}
    require(actual == declared, "manifest_closed_inventory_mismatch")
    return inventory


def validate_lineage(base: dict, source_manifest: bytes, state_raw: bytes,
                     config_raw: bytes, tokenizer_manifest: bytes) -> dict:
    """Pure metadata validation: no inferred config, lineage, tokenizer or approval."""
    require(base.get("schema_version") == "lumi.posttraining-base-binding.v0.1",
            "base_schema")
    require(base.get("checkpoint_stage") == "base", "base_stage")
    require(isinstance(base.get("run_fingerprint"), str)
            and SHA.fullmatch(base["run_fingerprint"]) is not None, "run_fingerprint_format")
    require(isinstance(base.get("source_commit"), str)
            and re.fullmatch(r"[a-f0-9]{40}", base["source_commit"]) is not None,
            "source_commit_format")
    expected = base["checkpoint"]
    require(sha(source_manifest) == expected["manifest_sha256"], "source_manifest_binding")
    source_entries = parse_manifest(source_manifest)
    require(source_entries.get(STATE) == sha(state_raw), "checkpoint_state_not_bound")
    require(any(k.startswith("model/") for k in source_entries), "model_state_not_declared")
    state = json_object(state_raw)
    require(state.get("checkpoint_schema_version") == 1
            and type(state.get("checkpoint_schema_version")) is int, "checkpoint_schema")
    mapping = {"step": "step", "sequence_cursor": "sequence_cursor",
               "training_tokens": "tokens_seen"}
    for bound, actual in mapping.items():
        require(type(expected[bound]) is int and type(state.get(actual)) is int
                and expected[bound] == state[actual], "checkpoint_cursor_mismatch")
    spec = state["run_specification"]
    world = spec["distributed"]["world_size"]
    require(type(world) is int and type(expected["world_size"]) is int
            and world == expected["world_size"] and 2 <= world <= 8, "world_size_mismatch")
    require(all(f"trainer-rank-{rank:05d}.pt" in source_entries for rank in range(world)),
            "trainer_rank_not_declared")
    require(state.get("run_fingerprint") == base["run_fingerprint"], "run_fingerprint_mismatch")
    require(spec.get("source_commit") == base["source_commit"], "source_commit_mismatch")
    require(spec.get("config_sha256") == sha(config_raw), "source_config_binding")
    config = json_object(config_raw)
    require(config.get("model_type") == "lumi", "model_type_mismatch")
    require(all(key in config for key in MODEL_FIELDS), "incomplete_explicit_model_config")
    for key in ("vocab_size", "hidden_size", "num_hidden_layers", "max_position_embeddings"):
        require(type(config[key]) is int and config[key] > 0, "model_geometry_type")
    require(sha(tokenizer_manifest) == base["tokenizer"]["manifest_sha256"],
            "tokenizer_manifest_binding")
    token_entries = parse_manifest(tokenizer_manifest)
    files = base["tokenizer"]["files"]
    require(len(files) == 3, "tokenizer_file_count")
    bound = {relative_name(x["relative_path"]): x["sha256"] for x in files}
    require(len(bound) == 3 and all(name.startswith("tokenizer/") for name in bound),
            "tokenizer_path_contract")
    require({Path(x).name for x in bound} == {
        "tokenizer.json", "tokenizer_config.json", "lumi_tokenizer_metadata.json"
    }, "tokenizer_role_contract")
    bare = {Path(k).name: v for k, v in bound.items()}
    require(token_entries in (bound, bare), "tokenizer_manifest_inventory")
    require(all(type(x["bytes"]) is int and x["bytes"] > 0 for x in files),
            "tokenizer_size_contract")
    return {
        "model_id": base["model_id"], "checkpoint": expected,
        "run_fingerprint": base["run_fingerprint"], "source_commit": base["source_commit"],
        "source_config_sha256": sha(config_raw), "tokenizer": base["tokenizer"],
    }


def validate_model_evidence(evidence: dict, dtype: str) -> None:
    require(set(evidence) == MODEL_EVIDENCE_KEYS, "model_evidence_fields")
    require(evidence["dtype"] == dtype and evidence["parameter_dtype"] == dtype,
            "export_dtype_mismatch")
    require(evidence["buffer_dtype_policy"] == "configured_buffer_dtype_preserve_values",
            "buffer_dtype_policy_changed")
    for key in ("model_parameters", "model_tensor_count"):
        require(type(evidence[key]) is int and 0 < evidence[key] <= 1_000_000,
                "synthetic_model_cap")
    for key in ("source_tensor_sha256", "cast_tensor_sha256", "reload_tensor_sha256"):
        require(isinstance(evidence[key], str) and SHA.fullmatch(evidence[key]) is not None,
                "model_digest_invalid")
    require(evidence["cast_tensor_sha256"] == evidence["reload_tensor_sha256"],
            "reload_tensor_mismatch")
    for key in ("complete_keys_shapes_verified", "hf_reload_tensor_equal", "no_forward_pass",
                "synthetic_only"):
        require(evidence[key] is True, "model_evidence_gate_failed")


def write_new(root: Path, name: str, raw: bytes) -> None:
    path = _path(root, name)
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def verify_package(root: Path, expected_manifest: str) -> dict:
    inventory = verify_closed_manifest(root, expected_manifest, private=True)
    receipt_raw = read_regular(root, RECEIPT)
    receipt = json_object(receipt_raw)
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    require(receipt.get("receipt_sha256") == sha(canonical(unsigned)), "receipt_internal_hash")
    require(receipt.get("schema_version") == "lumi.sft-checkpoint-import-receipt.v0.2",
            "receipt_schema")
    require(receipt["status"] == "PASS_synthetic_import_not_final_4b"
            and receipt["production_eligible"] is False and receipt["real_4b_import"] is False,
            "receipt_scope_changed")
    payload = [x for x in inventory if x["path"] not in {RECEIPT, MANIFEST}]
    require(receipt["payload_inventory"] == payload, "receipt_payload_inventory")
    require(receipt.get("training_authorized") is False
            and receipt.get("raw_training_text_retained") is False, "receipt_authority_changed")
    require(sha(read_regular(root, "config.json")) == receipt["export_config_sha256"],
            "receipt_config_binding")
    for entry in receipt["source_lineage"]["tokenizer"]["files"]:
        require({"path": entry["relative_path"], "bytes": entry["bytes"],
                 "sha256": entry["sha256"]} in payload, "receipt_tokenizer_binding")
    validate_model_evidence(receipt["model_evidence"], receipt["dtype"])
    return receipt


def run_synthetic_import(*, base: dict, checkpoint: Path, config: Path,
                         tokenizer: Path, attempt: Path, dtype: str = "bfloat16") -> dict:
    """Only a genuine tiny CPU fixture can run through this current entry point.

    Missing dependencies fail before a started marker or any fixture content read.
    A started attempt is never retried and is preserved on failure.
    """
    require(base.get("synthetic_only") is True
            and base.get("model_id") == "lumi-synthetic-import-fixture",
            "real_import_not_authorized")
    require(dtype in {"float32", "bfloat16"}, "unsupported_dtype")
    for key, cap in (("step", 2), ("sequence_cursor", 128), ("training_tokens", 4096),
                     ("world_size", 2)):
        require(type(base["checkpoint"][key]) is int and 0 < base["checkpoint"][key] <= cap,
                "synthetic_checkpoint_cap")
    require(not attempt.exists() and not attempt.is_symlink(), "attempt_exists_no_retry")
    for path in (checkpoint, config, tokenizer, attempt):
        _no_symlink_components(path)
    paths = [checkpoint.resolve(), config.resolve(), tokenizer.resolve()]
    require(all(not attempt.resolve().is_relative_to(p) and
                not p.is_relative_to(attempt.resolve()) for p in paths), "input_output_overlap")
    from .posttraining_checkpoint_import_torch_v2 import convert_and_verify, require_dependencies
    require_dependencies()
    require(attempt.parent.is_dir() and not attempt.parent.is_symlink(), "unsafe_attempt_parent")
    attempt.mkdir(mode=0o700)
    write_new(attempt, STARTED, canonical({"schema_version": 1, "synthetic_only": True,
                                         "real_4b_import": False, "retry_allowed": False}))
    source_before = verify_closed_manifest(checkpoint, base["checkpoint"]["manifest_sha256"])
    config_raw = read_regular(config.parent, config.name)
    token_before = verify_closed_manifest(tokenizer, base["tokenizer"]["manifest_sha256"])
    lineage = validate_lineage(base, read_regular(checkpoint, MANIFEST),
                               read_regular(checkpoint, STATE), config_raw,
                               read_regular(tokenizer, MANIFEST))
    config_value = json_object(config_raw)
    root = attempt / "checkpoint"
    evidence = convert_and_verify(config, checkpoint, root, dtype)
    validate_model_evidence(evidence, dtype)
    out_config = json_object(read_regular(root, "config.json"))
    require(all(out_config.get(k) == config_value[k] for k in MODEL_FIELDS),
            "export_model_config_changed")
    # Harden only this newly-created private attempt, never any input or predecessor.
    for directory, dirs, names in os.walk(root, followlinks=False):
        require(not Path(directory).is_symlink(), "export_symlink")
        for name in dirs + names:
            require(not (Path(directory) / name).is_symlink(), "export_symlink")
        Path(directory).chmod(0o700)
        for name in names:
            path = Path(directory) / name
            require(path.lstat().st_nlink == 1 and path.is_file(), "unsafe_export_file")
            path.chmod(0o600)
    token_entries = parse_manifest(read_regular(tokenizer, MANIFEST))
    for entry in base["tokenizer"]["files"]:
        target = entry["relative_path"]
        source = target if target in token_entries else Path(target).name
        raw = read_regular(tokenizer, source)
        require(sha(raw) == entry["sha256"] and len(raw) == entry["bytes"],
                "tokenizer_content_changed")
        write_new(root, target, raw)
    payload = tree_inventory(root, private=True)
    require(any(x["path"].endswith(".safetensors") for x in payload), "safe_weights_missing")
    require(all(x["path"] not in {MANIFEST, RECEIPT} for x in payload), "unexpected_export_receipt")
    require(source_before == tree_inventory(checkpoint), "source_mutated")
    require(token_before == tree_inventory(tokenizer)
            and config_raw == read_regular(config.parent, config.name),
            "config_or_tokenizer_mutated")
    receipt = {
        "schema_version": "lumi.sft-checkpoint-import-receipt.v0.2",
        "status": "PASS_synthetic_import_not_final_4b", "real_4b_import": False,
        "production_eligible": False, "raw_training_text_retained": False,
        "base_binding_sha256": sha(canonical(base)), "source_lineage": lineage,
        "source_config_sha256": sha(config_raw),
        "export_config_sha256": sha(read_regular(root, "config.json")),
        "dtype": dtype, "model_evidence": evidence, "payload_inventory": payload,
        "source_and_tokenizer_preserved": True,
        "worker_manifest_covers_payload_and_receipt": True,
        "legacy_export_inventory_reused": False,
        "training_authorized": False,
        "adapter_file_sha256": sha(read_regular(Path(__file__).parent, Path(__file__).name)),
        "backend_file_sha256": sha(read_regular(Path(__file__).parent,
                                               "posttraining_checkpoint_import_torch_v2.py")),
    }
    receipt["receipt_sha256"] = sha(canonical(receipt))
    write_new(root, RECEIPT, canonical(receipt) + b"\n")
    final_payload = tree_inventory(root, private=True)
    manifest_raw = "".join(f"{x['sha256']}  {x['path']}\n" for x in final_payload).encode()
    write_new(root, MANIFEST, manifest_raw)
    verify_package(root, sha(manifest_raw))
    # Exercise the actual frozen consumer, not a rewritten lookalike checksum parser.
    from .posttraining_sft_runtime_v4 import _verify_checksum_manifest
    _verify_checksum_manifest(root, expected_manifest_sha256=sha(manifest_raw),
                              require_private=True)
    result = {"status": receipt["status"], "manifest_file_sha256": sha(manifest_raw),
              "receipt_internal_sha256": receipt["receipt_sha256"],
              "real_worker_checksum_verifier_passed": True, "real_4b_import": False,
              "production_eligible": False}
    write_new(attempt, SUCCESS, canonical(result) + b"\n")
    return result
