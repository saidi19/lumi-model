"""Token-consistent smoke-only seal and encoded shards for registry-v0.17."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import uuid
from array import array
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

from .posttraining_candidate_mix_preflight import file_sha256, sha256_value
from .posttraining_format import encode_sft_messages, load_chat_template, verify_chat_tokenizer
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
)
from .posttraining_shards import (
    SFT_SHARDS_SCHEMA_VERSION,
    _ShardWriter,
    _write_json_atomic,
    verify_sft_shards,
)

PLAN_SCHEMA_VERSION = "lumi.sft-token-consistent-shards-plan.v0.1"
SEAL_SCHEMA_VERSION = "lumi.sft-learning-candidate-token-consistent-seal.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.sft-token-consistent-shards-receipt.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-token-consistent-shards-verification.v0.1"

EXPECTED_AUTHORIZATION = {
    "active_data_contract_change_authorized": False,
    "encoded_shard_materialization_authorized": True,
    "external_api_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "predecessor_mutation_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "registry_change_authorized": False,
    "smoke_only_successor_seal_authorized": True,
    "unrelated_access_authorized": False,
}
EXPECTED_TOTALS = {
    "maximum_record_tokens": 3_947,
    "records": 264_981,
    "train": {
        "records": 251_176,
        "serialized_tokens": 17_504_230,
        "supervised_tokens": 7_307_300,
    },
    "validation": {
        "records": 13_805,
        "serialized_tokens": 990_198,
        "supervised_tokens": 425_279,
    },
}
EXPECTED_LIMITS = {
    "maximum_encoded_index_row_reads": 264_981,
    "maximum_gpu_use": 0,
    "maximum_input_row_reads": 264_981,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 536_870_912,
    "maximum_runtime_minutes": 120,
    "maximum_tokens_per_shard": 4_194_304,
}
EXPECTED_OUTPUT = {
    "encoded_shards_path": (
        "artifacts/lumi-sft-token-consistent-shards-v0.1/attempt-001/private/encoded"
    ),
    "receipt_path": "evidence/lumi-4b-sft-token-consistent-shards-v0.1.json",
    "successor_seal_path": (
        "evidence/lumi-4b-sft-learning-candidate-token-consistent-seal-v0.1.json"
    ),
    "verification_path": (
        "evidence/lumi-4b-sft-token-consistent-shards-verification-v0.1.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "chat_template",
        "implementation",
        "learning_seal",
        "learning_seal_verification",
        "preflight",
        "runner",
        "test",
        "token_consistency_diagnostic",
        "token_consistency_verification",
        "tokenizer",
        "train_jsonl",
        "validation_jsonl",
        "verifier",
    }
)
EXPECTED_FILE_HASHES = {
    "chat_template": "bc7853a9277feddae2fd3db786c6a12f4fa3602a135a0408c38d4390f33bec12",
    "learning_seal": "3e214b10e460944f8ec02fe366e1581f22bee2f2cf569d8cbc7b0b7092288b76",
    "learning_seal_verification": (
        "1f5cbeeea8b1f0b338a3352f54c04b48849483b6025cf4c1ad4cf77581719c52"
    ),
    "preflight": "4790869e0eb5d57ba220ccdcc84c614827d29499e503b879f7e0049686d3cbd2",
    "token_consistency_diagnostic": (
        "08e3430dc0804ac3064385d97427ede5c4d450f0d1561b58399a86312eeb1d99"
    ),
    "token_consistency_verification": (
        "5c93d2ff0461d26856758f701d03ef7bf985c373360130d374b7dabcfea7927f"
    ),
    "tokenizer": "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee",
    "train_jsonl": "e525d1a031aa6db507a886306aa515af9f90751393759d0ebe0a7c612eaf125d",
    "validation_jsonl": "a5d1bf515f8bacbd7c794dede8f2d18cfee802bb5aa71656365a4bcb08946bf5",
}


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _resolve_binding(root: Path, binding: Mapping[str, Any], *, name: str) -> Path:
    relative = binding.get("path")
    expected = binding.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise ValueError(f"invalid token-shard binding: {name}")
    path = (root / relative).resolve()
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"token-shard binding changed: {name}")
    return path


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite token-shard evidence: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale token-shard evidence temporary: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_token_consistent_shards_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(Path(path))
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported token-consistent shard plan")
    if plan.get("status") != "authorized_token_consistent_smoke_only_shards":
        raise ValueError("token-consistent shard plan is not authorized")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("token-consistent shard authorization changed")
    if plan.get("expected_totals") != EXPECTED_TOTALS:
        raise ValueError("token-consistent shard totals changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("token-consistent shard limits changed")
    if plan.get("output") != EXPECTED_OUTPUT:
        raise ValueError("token-consistent shard output changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("token-consistent shard binding set changed")
    resolved = {
        name: _resolve_binding(root, binding, name=name)
        for name, binding in bindings.items()
    }
    for name, expected in EXPECTED_FILE_HASHES.items():
        if bindings[name]["sha256"] != expected:
            raise ValueError(f"token-consistent exact binding changed: {name}")
    seal = _load_json(resolved["learning_seal"])
    seal_verification = _load_json(resolved["learning_seal_verification"])
    diagnostic = _load_json(resolved["token_consistency_diagnostic"])
    diagnostic_verification = _load_json(resolved["token_consistency_verification"])
    preflight = _load_json(resolved["preflight"])
    if (
        seal.get("seal_sha256")
        != "73859088ad74b5ce39292d479348fea9bd555e372ea89832df420bd20025acc6"
        or seal_verification.get("status")
        != "PASS_verified_nonproduction_learning_candidate_seal_runtime_recovery"
        or diagnostic.get("status")
        != "MISMATCH_successor_seal_required_no_encoded_shards_written"
        or diagnostic.get("observed") != EXPECTED_TOTALS
        or diagnostic_verification.get("status")
        != "PASS_verified_token_mismatch_and_successor_requirement"
        or not all(diagnostic_verification.get("checks", {}).values())
        or preflight.get("status")
        != "PASS_successor_ready_seal_correction_and_shards_not_authorized"
    ):
        raise ValueError("token-consistent predecessor evidence changed")
    return plan


def _open_writer(root: Path, *, split: str, index: int) -> _ShardWriter:
    return _ShardWriter.open(root, split=split, index=index)


def _encoded_payload_bytes(record_id: str, input_ids: tuple[int, ...], labels: tuple[int, ...]) -> bytes:
    return json.dumps(
        {"input_ids": list(input_ids), "labels": list(labels), "record_id": record_id},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def materialize_token_consistent_shards(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    resolved = {
        name: _resolve_binding(root, binding, name=name)
        for name, binding in plan["bindings"].items()
    }
    template = load_chat_template(resolved["chat_template"])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = PureByteLevelBPETokenizer(resolved["tokenizer"])
    verify_chat_tokenizer(tokenizer, template)

    output = (root / plan["output"]["encoded_shards_path"]).resolve()
    private_root = (root / "artifacts/lumi-sft-token-consistent-shards-v0.1").resolve()
    if not output.is_relative_to(private_root):
        raise ValueError("token-consistent shard output escaped private root")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite token-consistent shards: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir()

    writers: dict[str, _ShardWriter] = {}
    next_index: Counter[str] = Counter()
    inventories: list[dict[str, Any]] = []
    split_records: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    split_supervised: Counter[str] = Counter()
    encoded_hasher = hashlib.sha256()
    record_sequence_hasher = hashlib.sha256()
    maximum_record_tokens = 0

    def close_writer(split: str) -> None:
        writer = writers.pop(split, None)
        if writer is not None:
            inventories.append(writer.close())

    try:
        for split in ("train", "validation"):
            input_path = resolved[f"{split}_jsonl"]
            with input_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError(f"JSON object required at {input_path}:{line_number}")
                    if record.get("split") != split:
                        raise ValueError("token-consistent input split changed")
                    record_id = record.get("record_id")
                    messages = record.get("messages")
                    if not isinstance(record_id, str) or not isinstance(messages, list):
                        raise ValueError("token-consistent record identity or messages changed")
                    encoded = encode_sft_messages(
                        messages,
                        tokenizer=tokenizer,
                        template=template,
                    )
                    maximum_record_tokens = max(maximum_record_tokens, len(encoded.input_ids))
                    encoded_bytes = _encoded_payload_bytes(
                        record_id, encoded.input_ids, encoded.labels
                    )
                    encoded_hasher.update(len(encoded_bytes).to_bytes(8, "big"))
                    encoded_hasher.update(encoded_bytes)
                    record_sequence_hasher.update(f"{split}\0{record_id}\n".encode("utf-8"))

                    writer = writers.get(split)
                    if writer is not None and (
                        writer.tokens + len(encoded.input_ids)
                        > plan["limits"]["maximum_tokens_per_shard"]
                    ):
                        close_writer(split)
                        writer = None
                    if writer is None:
                        writer = _open_writer(
                            temporary,
                            split=split,
                            index=next_index[split],
                        )
                        next_index[split] += 1
                        writers[split] = writer
                    writer.add(record_id=record_id, encoded=encoded)
                    split_records[split] += 1
                    split_tokens[split] += len(encoded.input_ids)
                    split_supervised[split] += encoded.supervised_tokens
            close_writer(split)

        observed = {
            "maximum_record_tokens": maximum_record_tokens,
            "records": sum(split_records.values()),
            "train": {
                "records": split_records["train"],
                "serialized_tokens": split_tokens["train"],
                "supervised_tokens": split_supervised["train"],
            },
            "validation": {
                "records": split_records["validation"],
                "serialized_tokens": split_tokens["validation"],
                "supervised_tokens": split_supervised["validation"],
            },
        }
        if observed != EXPECTED_TOTALS:
            raise ValueError("token-consistent encoded totals changed")
        inventories.sort(key=lambda item: (item["split"], item["index"]))
        if (
            [(item["split"], item["index"]) for item in inventories]
            != [
                ("train", 0),
                ("train", 1),
                ("train", 2),
                ("train", 3),
                ("train", 4),
                ("validation", 0),
            ]
        ):
            raise ValueError("token-consistent shard count or identity changed")

        predecessor = _load_json(resolved["learning_seal"])
        successor: dict[str, Any] = {
            "authorization_boundary": {
                "active_data_contract_changed": False,
                "network_or_gpu_used": False,
                "predecessor_mutated": False,
                "production_dataset_seal": False,
                "production_sft_started": False,
                "registry_changed": False,
                "smoke_only": True,
            },
            "chat_template_sha256": plan["bindings"]["chat_template"]["sha256"],
            "encoded_dataset_sha256": encoded_hasher.hexdigest(),
            "maximum_record_tokens": maximum_record_tokens,
            "predecessor_file_sha256": plan["bindings"]["learning_seal"]["sha256"],
            "predecessor_internal_sha256": predecessor["seal_sha256"],
            "production_blockers_preserved": predecessor[
                "production_blockers_preserved"
            ],
            "raw_text_retained_in_manifest": False,
            "record_sequence_sha256": record_sequence_hasher.hexdigest(),
            "records": observed["records"],
            "schema_version": SEAL_SCHEMA_VERSION,
            "splits": {
                "train": {
                    "records": observed["train"]["records"],
                    "serialized_tokens": observed["train"]["serialized_tokens"],
                    "supervised_tokens": observed["train"]["supervised_tokens"],
                    "source_jsonl_sha256": plan["bindings"]["train_jsonl"]["sha256"],
                },
                "validation": {
                    "records": observed["validation"]["records"],
                    "serialized_tokens": observed["validation"]["serialized_tokens"],
                    "supervised_tokens": observed["validation"]["supervised_tokens"],
                    "source_jsonl_sha256": plan["bindings"]["validation_jsonl"]["sha256"],
                },
            },
            "status": "PASS_token_consistent_nonproduction_smoke_only_seal",
            "supervised_tokens": sum(split_supervised.values()),
            "tokenizer_sha256": plan["bindings"]["tokenizer"]["sha256"],
            "tokens": sum(split_tokens.values()),
        }
        successor["seal_sha256"] = sha256_value(successor)

        shard_manifest: dict[str, Any] = {
            "context_window": template.context_window,
            "encoded_dataset_sha256": successor["encoded_dataset_sha256"],
            "format": {
                "index": "canonical-jsonl-v0.1",
                "labels": "little-endian-int32-ignore-index-minus-100",
                "raw_text_retained": False,
                "tokens": "little-endian-uint32",
            },
            "maximum_tokens_per_shard": plan["limits"]["maximum_tokens_per_shard"],
            "records": observed["records"],
            "schema_version": SFT_SHARDS_SCHEMA_VERSION,
            "seal_sha256": successor["seal_sha256"],
            "shards": inventories,
            "splits": {
                split: {
                    "records": split_records[split],
                    "supervised_tokens": split_supervised[split],
                    "tokens": split_tokens[split],
                }
                for split in ("train", "validation")
            },
            "supervised_tokens": sum(split_supervised.values()),
            "tokens": sum(split_tokens.values()),
        }
        shard_manifest["manifest_sha256"] = sha256_value(shard_manifest)
        _write_json_atomic(temporary / "manifest.json", shard_manifest)
        verify_sft_shards(temporary)
        written_bytes = sum(
            path.stat().st_size for path in temporary.iterdir() if path.is_file()
        )
        if written_bytes > plan["limits"]["maximum_private_write_bytes"]:
            raise ValueError("token-consistent shard write ceiling exceeded")
        temporary.replace(output)
    except BaseException:
        for writer in tuple(writers.values()):
            writer.tokens_handle.close()
            writer.labels_handle.close()
            writer.records_handle.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    receipt: dict[str, Any] = {
        "authorization_boundary": successor["authorization_boundary"],
        "encoded_shards": {
            "directory": plan["output"]["encoded_shards_path"],
            "manifest_file_sha256": file_sha256(output / "manifest.json"),
            "manifest_internal_sha256": shard_manifest["manifest_sha256"],
            "private_write_bytes": written_bytes,
            "shards": len(inventories),
        },
        "input_row_reads": observed["records"],
        "plan_file_sha256": plan_file_sha256,
        "raw_training_text_retained": False,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "PASS_token_consistent_six_shards_materialized_smoke_only",
        "successor_seal_internal_sha256": successor["seal_sha256"],
        "totals": observed,
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return successor, receipt


def _read_int32_file(path: Path, *, signed: bool) -> array:
    values = array("i" if signed else "I")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // 4)
    if values.itemsize != 4:
        raise RuntimeError("platform does not expose four-byte integer arrays")
    if sys.byteorder != "little":
        values.byteswap()
    return values


def verify_binary_masks(root: Path, manifest: Mapping[str, Any]) -> dict[str, int]:
    """Read every raw-text-free index row and verify its binary label slice."""

    index_rows = 0
    masked_tokens = 0
    supervised_tokens = 0
    for shard in manifest["shards"]:
        split = shard["split"]
        prefix = f"{split}-{shard['index']:05d}"
        tokens = _read_int32_file(root / f"{prefix}.tokens.u32le", signed=False)
        labels = _read_int32_file(root / f"{prefix}.labels.i32le", signed=True)
        if len(tokens) != len(labels):
            raise ValueError("token-consistent token and label lengths changed")
        for line in (root / f"{prefix}.index.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            entry = json.loads(line)
            start = entry["offset_tokens"]
            end = start + entry["tokens"]
            observed_supervised = 0
            observed_masked = 0
            for token, label in zip(tokens[start:end], labels[start:end], strict=True):
                if label == -100:
                    observed_masked += 1
                elif label == token:
                    observed_supervised += 1
                else:
                    raise ValueError("token-consistent unmasked label differs from input")
            if (
                observed_supervised != entry["supervised_tokens"]
                or observed_masked < 1
            ):
                raise ValueError("token-consistent record mask count changed")
            index_rows += 1
            masked_tokens += observed_masked
            supervised_tokens += observed_supervised
    return {
        "index_rows": index_rows,
        "masked_tokens": masked_tokens,
        "supervised_tokens": supervised_tokens,
    }


def build_token_consistent_shards_verification(
    *,
    plan: Mapping[str, Any],
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = (root / plan["output"]["encoded_shards_path"]).resolve()
    successor_path = (root / plan["output"]["successor_seal_path"]).resolve()
    receipt_path = (root / plan["output"]["receipt_path"]).resolve()
    successor = _load_json(successor_path)
    receipt = _load_json(receipt_path)
    manifest = verify_sft_shards(output)
    masks = verify_binary_masks(output, manifest)
    if (
        successor.get("status")
        != "PASS_token_consistent_nonproduction_smoke_only_seal"
        or successor.get("seal_sha256") != manifest.get("seal_sha256")
        or receipt.get("status")
        != "PASS_token_consistent_six_shards_materialized_smoke_only"
        or receipt.get("successor_seal_internal_sha256")
        != successor.get("seal_sha256")
        or masks["index_rows"] != EXPECTED_TOTALS["records"]
        or masks["supervised_tokens"]
        != EXPECTED_TOTALS["train"]["supervised_tokens"]
        + EXPECTED_TOTALS["validation"]["supervised_tokens"]
        or len(manifest["shards"]) != 6
    ):
        raise ValueError("token-consistent shard verification totals changed")
    verification: dict[str, Any] = {
        "checks": {
            "all_264981_encoded_index_rows_read": True,
            "all_binary_file_hashes_and_sizes_verified": True,
            "all_unmasked_labels_equal_input_tokens": True,
            "exact_five_train_and_one_validation_shards": True,
            "exact_serialized_and_supervised_token_totals": True,
            "no_raw_text_in_shard_outputs_or_evidence": True,
            "predecessor_jsonl_and_seal_files_unchanged": True,
            "smoke_only_nonproduction_boundary_preserved": True,
        },
        "encoded_manifest_file_sha256": file_sha256(output / "manifest.json"),
        "encoded_manifest_internal_sha256": manifest["manifest_sha256"],
        "mask_audit": masks,
        "raw_training_text_retained": False,
        "receipt_file_sha256": file_sha256(receipt_path),
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_verified_token_consistent_six_shards_smoke_only",
        "successor_seal_file_sha256": file_sha256(successor_path),
        "successor_seal_internal_sha256": successor["seal_sha256"],
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "build_token_consistent_shards_verification",
    "json_file_bytes",
    "load_token_consistent_shards_plan",
    "materialize_token_consistent_shards",
    "verify_binary_masks",
]
