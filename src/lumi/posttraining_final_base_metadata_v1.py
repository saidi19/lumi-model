"""Pure bounded final-base metadata lineage, never checkpoint or tokenizer import.

The caller independently supplies the SHA-256 of the already approved base file.
The tokenizer binding is the sorted normalized JSON inventory digest used by
posttraining_readiness.py, not a checksum-line file. Only metadata byte strings
enter this module; there are no filesystem, transport, ML or execution APIs.
Matching metadata does not authenticate its publisher or verify any payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

SCHEMA = "lumi.sft-final-base-metadata.v0.1"
MAXIMUM_INPUT_BYTES = {"base": 16384, "manifest": 16384, "state": 32768, "config": 16384}
MAXIMUM_COMBINED_BYTES = 65536
MAXIMUM_MANIFEST_ENTRIES = 128
MAXIMUM_REPORT_BYTES = 4096
STATE = "distributed-training-state.json"
MODEL_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "rope_theta",
    "rms_norm_eps",
    "attention_bias",
    "tie_word_embeddings",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "doc_token_id",
)
TOKENIZER_ROLES = (
    "tokenizer/lumi_tokenizer_metadata.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)
FALSE_FLAGS = (
    "tokenizer_payload_verified",
    "checkpoint_payload_verified",
    "real_4b_import",
    "execution_permitted",
    "real_sft_execution_permitted",
    "production_eligible",
    "publisher_authenticity_verified",
    "remote_availability_verified",
)
ERROR_CODES = (
    "input_bytes_required",
    "input_size_cap",
    "base_pin_required",
    "base_binding_mismatch",
    "duplicate_json_key",
    "nonfinite_json_number",
    "invalid_json",
    "json_object_required",
    "metadata_shape_rejected",
    "base_schema",
    "base_stage",
    "model_id_mismatch",
    "run_fingerprint_format",
    "source_commit_format",
    "checkpoint_binding_shape",
    "source_manifest_binding",
    "manifest_line_endings",
    "manifest_line_invalid",
    "invalid_relative_name",
    "manifest_duplicate_or_self",
    "manifest_entry_cap",
    "checkpoint_state_not_bound",
    "model_state_not_declared",
    "checkpoint_schema",
    "checkpoint_cursor_mismatch",
    "world_size_mismatch",
    "trainer_rank_not_declared",
    "run_fingerprint_mismatch",
    "source_commit_mismatch",
    "source_config_binding",
    "model_type_mismatch",
    "incomplete_explicit_model_config",
    "model_geometry_type",
    "tokenizer_binding_shape",
    "tokenizer_file_count",
    "tokenizer_path_contract",
    "tokenizer_role_contract",
    "tokenizer_size_contract",
    "tokenizer_hash_contract",
    "tokenizer_manifest_binding",
    "report_size_cap",
)
REPORT_KEYS = (
    "schema_version",
    "status",
    "base_file_sha256",
    "checkpoint_manifest_sha256",
    "checkpoint_state_sha256",
    "source_config_sha256",
    "source_commit",
    "run_fingerprint",
    "model_id",
    "step",
    "sequence_cursor",
    "training_tokens",
    "world_size",
    "checkpoint_manifest_entries",
    "model_entries_declared",
    "trainer_ranks_declared",
    "model_geometry",
    "tokenizer_inventory_sha256",
    "tokenizer_inventory_canonical_bytes",
    "tokenizer_files_declared",
    "tokenizer_bytes_declared",
    "metadata_input_bytes",
    "tokenizer_inventory_semantics",
    "base_pin_verified",
    "metadata_lineage_verified",
    "tokenizer_payload_verified",
    "checkpoint_payload_verified",
    "real_4b_import",
    "execution_permitted",
    "real_sft_execution_permitted",
    "production_eligible",
    "publisher_authenticity_verified",
    "remote_availability_verified",
)


class FinalBaseMetadataError(ValueError):
    def __init__(self, code: str):
        self.code = code if type(code) is str and code in ERROR_CODES else "metadata_shape_rejected"
        super().__init__(self.code)


def require(condition: bool, code: str) -> None:
    if not condition:
        raise FinalBaseMetadataError(code) from None


def canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_value(value) -> bool:
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _positive(value) -> bool:
    return type(value) is int and 0 < value <= 2**63 - 1


def json_object(raw: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    def nonfinite(_value):
        raise FinalBaseMetadataError("nonfinite_json_number") from None

    try:
        result = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
        require(type(result) is dict, "json_object_required")
        canonical(result)
        return result
    except FinalBaseMetadataError:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise FinalBaseMetadataError("invalid_json") from None


def relative_name(name: str) -> str:
    require(type(name) is str and 0 < len(name) <= 512, "invalid_relative_name")
    require(not name.startswith("/") and "\\" not in name, "invalid_relative_name")
    require(all(part not in {"", ".", ".."} for part in name.split("/")), "invalid_relative_name")
    require(not any(ord(c) < 33 or ord(c) == 127 for c in name), "invalid_relative_name")
    return name


def parse_manifest(raw: bytes) -> dict[str, str]:
    require(bool(raw) and raw.endswith(b"\n") and b"\r" not in raw, "manifest_line_endings")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError:
        raise FinalBaseMetadataError("manifest_line_invalid") from None
    require(0 < len(lines) <= MAXIMUM_MANIFEST_ENTRIES, "manifest_entry_cap")
    entries = {}
    for line in lines:
        fields = line.split("  ")
        require(len(fields) == 2 and _sha_value(fields[0]), "manifest_line_invalid")
        name = relative_name(fields[1])
        require(name != "manifest.sha256" and name not in entries, "manifest_duplicate_or_self")
        entries[name] = fields[0]
    return entries


def tokenizer_inventory(tokenizer: dict) -> tuple[str, int, int]:
    require(
        type(tokenizer) is dict and _sha_value(tokenizer.get("manifest_sha256")),
        "tokenizer_binding_shape",
    )
    files = tokenizer.get("files")
    require(type(files) is list and len(files) == 3, "tokenizer_file_count")
    normalized = []
    for item in files:
        require(type(item) is dict, "tokenizer_binding_shape")
        path, size, digest = item.get("relative_path"), item.get("bytes"), item.get("sha256")
        require(type(path) is str and path in TOKENIZER_ROLES, "tokenizer_path_contract")
        require(_positive(size), "tokenizer_size_contract")
        require(_sha_value(digest), "tokenizer_hash_contract")
        normalized.append({"bytes": size, "relative_path": path, "sha256": digest})
    require(
        {item["relative_path"] for item in normalized} == set(TOKENIZER_ROLES),
        "tokenizer_role_contract",
    )
    # Preserve readiness.py normalization: ignore auxiliary inventory fields and
    # sort by path; input list/key order is not part of this inventory digest.
    raw = canonical(sorted(normalized, key=lambda item: item["relative_path"]))
    digest = sha(raw)
    require(digest == tokenizer["manifest_sha256"], "tokenizer_manifest_binding")
    return digest, len(raw), sum(item["bytes"] for item in normalized)


def _verify(base_raw, manifest_raw, state_raw, config_raw, expected_base_sha256):
    inputs = {"base": base_raw, "manifest": manifest_raw, "state": state_raw, "config": config_raw}
    require(all(type(raw) is bytes for raw in inputs.values()), "input_bytes_required")
    require(
        all(0 < len(raw) <= MAXIMUM_INPUT_BYTES[name] for name, raw in inputs.items())
        and sum(map(len, inputs.values())) <= MAXIMUM_COMBINED_BYTES,
        "input_size_cap",
    )
    require(_sha_value(expected_base_sha256), "base_pin_required")
    require(sha(base_raw) == expected_base_sha256, "base_binding_mismatch")
    base = json_object(base_raw)
    require(base.get("schema_version") == "lumi.posttraining-base-binding.v0.1", "base_schema")
    require(base.get("checkpoint_stage") == "base", "base_stage")
    require(base.get("model_id") == "lumi-4b-base", "model_id_mismatch")
    require(_sha_value(base.get("run_fingerprint")), "run_fingerprint_format")
    require(
        type(base.get("source_commit")) is str
        and re.fullmatch(r"[a-f0-9]{40}", base["source_commit"]) is not None,
        "source_commit_format",
    )
    expected = base.get("checkpoint")
    require(
        type(expected) is dict
        and set(expected)
        == {"manifest_sha256", "step", "sequence_cursor", "training_tokens", "world_size"}
        and _sha_value(expected.get("manifest_sha256")),
        "checkpoint_binding_shape",
    )
    require(sha(manifest_raw) == expected["manifest_sha256"], "source_manifest_binding")
    source_entries = parse_manifest(manifest_raw)
    require(source_entries.get(STATE) == sha(state_raw), "checkpoint_state_not_bound")
    require(any(k.startswith("model/") for k in source_entries), "model_state_not_declared")
    state = json_object(state_raw)
    require(
        state.get("checkpoint_schema_version") == 1
        and type(state.get("checkpoint_schema_version")) is int,
        "checkpoint_schema",
    )
    mapping = {
        "step": "step",
        "sequence_cursor": "sequence_cursor",
        "training_tokens": "tokens_seen",
    }
    for bound, actual in mapping.items():
        require(
            _positive(expected[bound])
            and _positive(state.get(actual))
            and expected[bound] == state[actual],
            "checkpoint_cursor_mismatch",
        )
    spec = state.get("run_specification")
    require(type(spec) is dict and type(spec.get("distributed")) is dict, "metadata_shape_rejected")
    world = spec["distributed"].get("world_size")
    require(
        type(world) is int
        and type(expected["world_size"]) is int
        and world == expected["world_size"]
        and 2 <= world <= 8,
        "world_size_mismatch",
    )
    require(
        all(f"trainer-rank-{rank:05d}.pt" in source_entries for rank in range(world)),
        "trainer_rank_not_declared",
    )
    require(state.get("run_fingerprint") == base["run_fingerprint"], "run_fingerprint_mismatch")
    require(spec.get("source_commit") == base["source_commit"], "source_commit_mismatch")
    require(spec.get("config_sha256") == sha(config_raw), "source_config_binding")
    config = json_object(config_raw)
    require(config.get("model_type") == "lumi", "model_type_mismatch")
    require(all(key in config for key in MODEL_FIELDS), "incomplete_explicit_model_config")
    for key in (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
    ):
        require(_positive(config[key]), "model_geometry_type")
    for key in ("rope_theta", "rms_norm_eps"):
        require(
            type(config[key]) in (int, float) and math.isfinite(config[key]) and config[key] > 0,
            "model_geometry_type",
        )
    for key in ("attention_bias", "tie_word_embeddings"):
        require(type(config[key]) is bool, "model_geometry_type")
    for key in ("pad_token_id", "bos_token_id", "eos_token_id", "doc_token_id"):
        require(
            type(config[key]) is int and 0 <= config[key] < config["vocab_size"],
            "model_geometry_type",
        )
    token_sha, token_canonical_bytes, token_bytes = tokenizer_inventory(base.get("tokenizer"))
    report = {
        "schema_version": SCHEMA,
        "status": "PASS_metadata_lineage_only",
        "base_file_sha256": expected_base_sha256,
        "checkpoint_manifest_sha256": sha(manifest_raw),
        "checkpoint_state_sha256": sha(state_raw),
        "source_config_sha256": sha(config_raw),
        "source_commit": base["source_commit"],
        "run_fingerprint": base["run_fingerprint"],
        "model_id": "lumi-4b-base",
        "step": expected["step"],
        "sequence_cursor": expected["sequence_cursor"],
        "training_tokens": expected["training_tokens"],
        "world_size": world,
        "checkpoint_manifest_entries": len(source_entries),
        "model_entries_declared": sum(name.startswith("model/") for name in source_entries),
        "trainer_ranks_declared": world,
        "model_geometry": {key: config[key] for key in MODEL_FIELDS},
        "tokenizer_inventory_sha256": token_sha,
        "tokenizer_inventory_canonical_bytes": token_canonical_bytes,
        "tokenizer_files_declared": 3,
        "tokenizer_bytes_declared": token_bytes,
        "metadata_input_bytes": sum(map(len, inputs.values())),
        "tokenizer_inventory_semantics": "sorted_normalized_json_inventory_no_newline",
        "base_pin_verified": True,
        "metadata_lineage_verified": True,
        **dict.fromkeys(FALSE_FLAGS, False),
    }
    require(
        set(report) == set(REPORT_KEYS) and len(canonical(report)) <= MAXIMUM_REPORT_BYTES,
        "report_size_cap",
    )
    return report


def verify_final_base_metadata(
    base_raw: bytes,
    manifest_raw: bytes,
    state_raw: bytes,
    config_raw: bytes,
    *,
    expected_base_sha256: str,
) -> dict:
    """Verify four bounded metadata objects against the caller's independent base pin.

    The digest parameter has no default and is not inferred from the supplied
    base bytes. The caller, not this pure validator, binds authentic provenance.
    No tokenizer/model/trainer payload is accepted, opened or verified here.
    """
    try:
        return _verify(base_raw, manifest_raw, state_raw, config_raw, expected_base_sha256)
    except FinalBaseMetadataError:
        raise
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise FinalBaseMetadataError("metadata_shape_rejected") from None
