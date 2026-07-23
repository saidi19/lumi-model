"""Fail-closed adapters for Lumi's sealed final corpus and checkpoint handoffs.

The final 80B corpus is stored as an ordered list of immutable uint16 objects.
Those objects are intentionally not aligned to the model's 4,096-token context
window: document-boundary packing decides object boundaries, while the trainer
consumes one logical token stream.  This module validates that contract,
provides a memory-mapped logical-sequence dataset, and seals portable
checkpoint directories for cross-instance handoff.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from bisect import bisect_right
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
from torch.utils.data import Dataset

from .configuration_lumi import LumiConfig
from .parameter_count import parameter_breakdown

FINAL_CORPUS_SHARD_MANIFEST_GATE = "lumi_final_corpus_80b_shard_manifest_v0.1"
FINAL_CORPUS_ADAPTER_GATE = "lumi_4b_final_corpus_adapter_v0.1"
CHECKPOINT_HANDOFF_GATE = "lumi_4b_checkpoint_handoff_v0.1"
STAGING_STATE_GATE = "lumi_4b_final_corpus_staging_state_v0.1"
PRODUCTION_PREFLIGHT_GATE = "lumi_4b_production_training_preflight_v0.1"
TOKEN_BYTES = 2
TOKEN_DTYPE = "uint16"
TOKEN_ENDIANNESS = "little"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _fingerprint(value: Mapping[str, Any], omitted: str | None = None) -> str:
    material = dict(value)
    if omitted is not None:
        material.pop(omitted, None)
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _valid_s3_uri(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("final corpus shard S3 URI is invalid")
    parsed = urlparse(value)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError("final corpus shard S3 URI is invalid")
    if parsed.params or parsed.query or parsed.fragment:
        raise ValueError("final corpus shard S3 URI must not contain parameters")
    return value


def staged_shard_name(shard: Mapping[str, Any]) -> str:
    """Return the location-independent local name for one sealed shard."""

    sequence = int(shard["sequence"])
    digest = _valid_sha256(shard.get("object_sha256"), "final shard SHA-256")
    return f"shard-{sequence:05d}-{digest}.uint16.bin"


def validate_final_corpus_training_manifest(
    manifest_path: str | Path,
    *,
    shard_root: str | Path | None = None,
    sequence_length: int = 4096,
    vocab_size: int = 65_536,
    doc_token_id: int = 3,
    expected_manifest_sha256: str | None = None,
    expected_plan_fingerprint: str | None = None,
    expected_selection_result_sha256: str | None = None,
    expected_documents: int | None = None,
    expected_tokens: int | None = None,
    expected_training_token_limit: int | None = None,
    expected_shards: int | None = None,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    """Validate a sealed final-corpus manifest and return a trainer identity.

    The returned mapping deliberately looks like the existing packed-data
    identity where the distributed trainer needs it, but records that sequence
    boundaries are over the logical concatenation of all object streams.
    """

    if sys.byteorder != TOKEN_ENDIANNESS:
        raise RuntimeError("final corpus training currently requires a little-endian host")
    sequence_length = _positive_int(sequence_length, "sequence length")
    vocab_size = _positive_int(vocab_size, "vocabulary size")
    if not isinstance(doc_token_id, int) or isinstance(doc_token_id, bool) or doc_token_id < 0:
        raise ValueError("document token ID must be a nonnegative integer")
    if doc_token_id >= vocab_size:
        raise ValueError("document token ID must be smaller than the vocabulary")

    path = Path(manifest_path)
    raw = path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None and manifest_sha256 != _valid_sha256(
        expected_manifest_sha256, "expected final manifest SHA-256"
    ):
        raise ValueError("final corpus manifest SHA-256 changed")
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported final corpus manifest schema")
    if manifest.get("gate") != FINAL_CORPUS_SHARD_MANIFEST_GATE:
        raise ValueError("unexpected final corpus manifest gate")
    if manifest.get("status") != "sealed":
        raise ValueError("final corpus manifest is not sealed")

    plan_fingerprint = _valid_sha256(
        manifest.get("plan_fingerprint"), "final corpus plan fingerprint"
    )
    selection_sha256 = _valid_sha256(
        manifest.get("selection_result_sha256"), "final selection result SHA-256"
    )
    if expected_plan_fingerprint is not None and plan_fingerprint != _valid_sha256(
        expected_plan_fingerprint, "expected plan fingerprint"
    ):
        raise ValueError("final corpus plan fingerprint changed")
    if expected_selection_result_sha256 is not None and selection_sha256 != _valid_sha256(
        expected_selection_result_sha256, "expected selection result SHA-256"
    ):
        raise ValueError("final corpus selection-result binding changed")

    documents = _positive_int(manifest.get("documents"), "final corpus documents")
    tokens = _positive_int(manifest.get("tokens"), "final corpus tokens")
    training_token_limit = _positive_int(
        manifest.get("training_token_limit"), "final corpus training-token limit"
    )
    if training_token_limit > tokens:
        raise ValueError("final corpus training-token limit exceeds available tokens")
    if training_token_limit % sequence_length:
        raise ValueError("final corpus training-token limit is not sequence aligned")
    expected_values = (
        (expected_documents, documents, "document total"),
        (expected_tokens, tokens, "token total"),
        (expected_training_token_limit, training_token_limit, "training-token limit"),
    )
    for expected, observed, label in expected_values:
        if expected is not None and observed != expected:
            raise ValueError(f"final corpus {label} changed")

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("final corpus contains no shards")
    if expected_shards is not None and len(shards) != expected_shards:
        raise ValueError("final corpus shard count changed")
    root = None if shard_root is None else Path(shard_root)
    object_hashes: set[str] = set()
    object_uris: set[str] = set()
    shard_rows: list[dict[str, Any]] = []
    observed_tokens = 0
    observed_documents = 0
    observed_bytes = 0
    for sequence, shard in enumerate(shards):
        if not isinstance(shard, Mapping) or shard.get("sequence") != sequence:
            raise ValueError("final corpus shard sequence is not contiguous")
        shard_tokens = _positive_int(shard.get("tokens"), "final shard tokens")
        shard_documents = _positive_int(shard.get("documents"), "final shard documents")
        shard_bytes = _positive_int(shard.get("object_bytes"), "final shard bytes")
        if shard_bytes != shard_tokens * TOKEN_BYTES:
            raise ValueError("final shard uint16 byte conservation failed")
        digest = _valid_sha256(shard.get("object_sha256"), "final shard SHA-256")
        uri = _valid_s3_uri(shard.get("s3_uri"))
        if digest in object_hashes or uri in object_uris:
            raise ValueError("final corpus repeats an immutable shard")
        object_hashes.add(digest)
        object_uris.add(uri)
        local_name = staged_shard_name(shard)
        if root is not None:
            local_path = root / local_name
            if not local_path.is_file():
                raise FileNotFoundError(f"staged final shard does not exist: {local_path}")
            if local_path.stat().st_size != shard_bytes:
                raise ValueError(f"staged final shard size changed: {local_path}")
            if verify_hashes and _sha256_file(local_path) != digest:
                raise ValueError(f"staged final shard SHA-256 changed: {local_path}")
        shard_rows.append(
            {
                "bytes": shard_bytes,
                "category": str(shard.get("category")),
                "documents": shard_documents,
                "local_name": local_name,
                "sequence": sequence,
                "sha256": digest,
                "s3_uri": uri,
                "token_end": observed_tokens + shard_tokens,
                "token_start": observed_tokens,
                "tokens": shard_tokens,
            }
        )
        observed_tokens += shard_tokens
        observed_documents += shard_documents
        observed_bytes += shard_bytes
    if observed_tokens != tokens or observed_documents != documents:
        raise ValueError("final corpus shard totals do not match the manifest")

    identity_material = {
        "doc_token_id": doc_token_id,
        "manifest_sha256": manifest_sha256,
        "plan_fingerprint": plan_fingerprint,
        "selection_result_sha256": selection_sha256,
        "sequence_length": sequence_length,
        "token_dtype": TOKEN_DTYPE,
        "training_token_limit": training_token_limit,
        "vocab_size": vocab_size,
    }
    return {
        "build_fingerprint": _fingerprint(identity_material),
        "format": {
            "dtype": TOKEN_DTYPE,
            "endianness": TOKEN_ENDIANNESS,
            "layout": "logical_concatenated_final_corpus",
            "token_bytes": TOKEN_BYTES,
        },
        "gate": FINAL_CORPUS_ADAPTER_GATE,
        "packing": {"sequence_length": sequence_length},
        "schema_version": 1,
        "shards": shard_rows,
        "source": {
            "manifest_sha256": manifest_sha256,
            "plan_fingerprint": plan_fingerprint,
            "selection_result_sha256": selection_sha256,
        },
        "statistics": {
            "available_tokens": tokens,
            "documents": documents,
            "object_bytes": observed_bytes,
            "sequences": training_token_limit // sequence_length,
            "tokens": training_token_limit,
            "trailing_tokens_not_consumed": tokens - training_token_limit,
        },
        "status": "complete",
        "tokenizer": {"doc_token_id": doc_token_id, "vocab_size": vocab_size},
    }


def build_production_training_preflight_plan(
    *,
    contract_path: str | Path,
    manifest_path: str | Path,
    model_config_path: str | Path,
    source_commit: str,
) -> dict[str, Any]:
    """Bind the final corpus and model to planning-only 8-GPU pilot candidates."""

    contract_path = Path(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or contract.get("status") != "planning_only":
        raise ValueError("unsupported 4B production-preflight contract")
    if contract.get("approval") != {
        "full_4b_training": False,
        "paid_multi_gpu_pilot": False,
    }:
        raise ValueError("4B production-preflight contract authorizes paid work")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA-1")

    corpus = contract["corpus"]
    adapter = validate_final_corpus_training_manifest(
        manifest_path,
        sequence_length=int(corpus["sequence_length"]),
        vocab_size=int(corpus["vocab_size"]),
        doc_token_id=int(corpus["doc_token_id"]),
        expected_manifest_sha256=str(corpus["manifest_sha256"]),
        expected_plan_fingerprint=str(corpus["plan_fingerprint"]),
        expected_selection_result_sha256=str(corpus["selection_result_sha256"]),
        expected_documents=int(corpus["documents"]),
        expected_tokens=int(corpus["tokens"]),
        expected_training_token_limit=int(corpus["training_token_limit"]),
        expected_shards=int(corpus["shards"]),
    )
    if int(adapter["statistics"]["object_bytes"]) != int(corpus["object_bytes"]):
        raise ValueError("final corpus packed-byte total changed")

    model_config_path = Path(model_config_path)
    model_bytes = model_config_path.read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != contract["model"]["config_sha256"]:
        raise ValueError("Lumi-4B model config SHA-256 changed")
    model_config = LumiConfig(**json.loads(model_bytes))
    parameters = parameter_breakdown(model_config).total
    if parameters != int(contract["model"]["parameters"]):
        raise ValueError("Lumi-4B analytical parameter count changed")
    if model_config.vocab_size != int(corpus["vocab_size"]):
        raise ValueError("Lumi-4B vocabulary and final corpus differ")
    if model_config.doc_token_id != int(corpus["doc_token_id"]):
        raise ValueError("Lumi-4B document token and final corpus differ")
    if model_config.max_position_embeddings < int(corpus["sequence_length"]):
        raise ValueError("Lumi-4B context is shorter than the final corpus sequence")

    pilot = contract["pilot"]
    world_size = _positive_int(pilot.get("world_size"), "pilot world size")
    per_rank_batch_size = _positive_int(
        pilot.get("per_rank_batch_size"), "pilot per-rank batch size"
    )
    if per_rank_batch_size != 1:
        raise ValueError("the v0.1 exact-tail pilot requires per-rank batch size 1")
    sequence_length = int(corpus["sequence_length"])
    total_sequences = int(adapter["statistics"]["sequences"])
    candidates: list[dict[str, Any]] = []
    accumulation_values = pilot.get("gradient_accumulation_candidates")
    if not isinstance(accumulation_values, list) or not accumulation_values:
        raise ValueError("pilot gradient-accumulation candidates are missing")
    for accumulation in accumulation_values:
        accumulation = _positive_int(accumulation, "gradient accumulation")
        nominal_sequences = world_size * per_rank_batch_size * accumulation
        maximum_steps = (total_sequences + nominal_sequences - 1) // nominal_sequences
        tail_sequences = total_sequences % nominal_sequences
        candidates.append(
            {
                "exact_final_step_sequences": tail_sequences or nominal_sequences,
                "gradient_accumulation_steps": accumulation,
                "maximum_steps_for_exact_80b": maximum_steps,
                "nominal_tokens_per_step": nominal_sequences * sequence_length,
                "requires_masked_final_microbatch": bool(tail_sequences % world_size),
            }
        )

    plan: dict[str, Any] = {
        "approval": {
            "full_4b_training": False,
            "paid_multi_gpu_pilot": False,
        },
        "checkpoint_handoff": dict(contract["checkpoint_handoff"]),
        "corpus": {
            "adapter_build_fingerprint": adapter["build_fingerprint"],
            "documents": adapter["statistics"]["documents"],
            "manifest_sha256": adapter["source"]["manifest_sha256"],
            "object_bytes": adapter["statistics"]["object_bytes"],
            "sequence_length": sequence_length,
            "sequences": total_sequences,
            "shards": len(adapter["shards"]),
            "tokens": adapter["statistics"]["tokens"],
            "trailing_tokens_not_consumed": adapter["statistics"][
                "trailing_tokens_not_consumed"
            ],
        },
        "decision": {
            "next_gate": "owner-authorized bounded 8xH100 learning pilot",
            "status": "preflight_only_no_paid_training",
        },
        "gate": PRODUCTION_PREFLIGHT_GATE,
        "model": {
            "config_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "parameters": parameters,
        },
        "pilot": {
            "candidates": candidates,
            "fixed": {
                key: pilot[key]
                for key in (
                    "attention_implementation",
                    "document_isolation",
                    "gradient_checkpointing",
                    "gradient_sync_mode",
                    "optimizer_profile",
                    "per_rank_batch_size",
                    "precision",
                    "strategy",
                    "torch_compile_candidates",
                    "world_size",
                )
            },
        },
        "schema_version": 1,
        "source_commit": source_commit,
        "status": "planned",
    }
    plan["plan_fingerprint"] = _fingerprint(plan)
    return plan


def validate_production_training_preflight_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Fail if a serialized planning-only preflight report was altered."""

    if plan.get("schema_version") != 1 or plan.get("gate") != PRODUCTION_PREFLIGHT_GATE:
        raise ValueError("unsupported 4B production preflight plan")
    if plan.get("status") != "planned":
        raise ValueError("4B production preflight plan is incomplete")
    if plan.get("approval") != {
        "full_4b_training": False,
        "paid_multi_gpu_pilot": False,
    }:
        raise ValueError("4B production preflight plan authorizes paid work")
    if plan.get("plan_fingerprint") != _fingerprint(plan, "plan_fingerprint"):
        raise ValueError("4B production preflight fingerprint changed")
    return dict(plan)


def build_account_segment_plan(
    *,
    run_fingerprint: str,
    current_step: int,
    current_tokens: int,
    total_tokens: int,
    nominal_global_tokens_per_step: int,
    checkpoint_steps: int,
    measured_tokens_per_second: float,
    throughput_safety_factor: float,
    hourly_rate_usd: float,
    available_credit_usd: float,
    credit_reserve_usd: float,
    provider_runtime_limit_hours: float,
) -> dict[str, Any]:
    """Compute one checkpoint-aligned, credit-bounded training segment.

    This function does not authorize a launch.  It turns a measured pilot rate
    into a conservative stop boundary that can be sealed and resumed on an
    identically configured instance in another account.
    """

    _valid_sha256(run_fingerprint, "run fingerprint")
    for value, label in (
        (current_step, "current step"),
        (current_tokens, "current tokens"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label} must be a nonnegative integer")
    total_tokens = _positive_int(total_tokens, "total tokens")
    nominal_global_tokens_per_step = _positive_int(
        nominal_global_tokens_per_step, "nominal global tokens per step"
    )
    checkpoint_steps = _positive_int(checkpoint_steps, "checkpoint steps")
    if current_tokens > total_tokens:
        raise ValueError("current tokens exceed the full training boundary")
    if current_tokens > current_step * nominal_global_tokens_per_step:
        raise ValueError("current token cursor exceeds its nominal step boundary")
    for value, label in (
        (measured_tokens_per_second, "measured throughput"),
        (hourly_rate_usd, "hourly rate"),
        (available_credit_usd, "available credit"),
        (provider_runtime_limit_hours, "provider runtime limit"),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{label} must be positive")
    if not 0 < throughput_safety_factor <= 1:
        raise ValueError("throughput safety factor must be in (0, 1]")
    if credit_reserve_usd < 0 or credit_reserve_usd >= available_credit_usd:
        raise ValueError("credit reserve must be nonnegative and smaller than available credit")

    usable_credit = available_credit_usd - credit_reserve_usd
    budget_hours = usable_credit / hourly_rate_usd
    maximum_hours = min(provider_runtime_limit_hours, budget_hours)
    conservative_rate = measured_tokens_per_second * throughput_safety_factor
    affordable_tokens = int(conservative_rate * maximum_hours * 3600)
    remaining_tokens = total_tokens - current_tokens
    full_step_limit = math.ceil(total_tokens / nominal_global_tokens_per_step)
    affordable_steps = affordable_tokens // nominal_global_tokens_per_step
    checkpoint_aligned_steps = (affordable_steps // checkpoint_steps) * checkpoint_steps
    stop_step = min(full_step_limit, current_step + checkpoint_aligned_steps)
    if stop_step <= current_step and remaining_tokens:
        raise ValueError("segment budget cannot reach the next checkpoint boundary")
    stop_tokens = min(total_tokens, stop_step * nominal_global_tokens_per_step)
    segment_tokens = stop_tokens - current_tokens
    projected_hours = segment_tokens / conservative_rate / 3600 if segment_tokens else 0.0
    projected_cost = projected_hours * hourly_rate_usd
    plan: dict[str, Any] = {
        "bounds": {
            "available_credit_usd": available_credit_usd,
            "credit_reserve_usd": credit_reserve_usd,
            "hourly_rate_usd": hourly_rate_usd,
            "maximum_runtime_hours": maximum_hours,
            "provider_runtime_limit_hours": provider_runtime_limit_hours,
        },
        "current": {"step": current_step, "tokens": current_tokens},
        "gate": "lumi_4b_account_segment_plan_v0.1",
        "measurement": {
            "conservative_tokens_per_second": conservative_rate,
            "measured_tokens_per_second": measured_tokens_per_second,
            "throughput_safety_factor": throughput_safety_factor,
        },
        "projected": {
            "cost_usd": projected_cost,
            "hours": projected_hours,
            "segment_tokens": segment_tokens,
        },
        "run_fingerprint": run_fingerprint,
        "schema_version": 1,
        "status": "planning_only",
        "stop": {
            "checkpoint_aligned": stop_step == full_step_limit
            or stop_step % checkpoint_steps == 0,
            "full_training_complete": stop_tokens == total_tokens,
            "step": stop_step,
            "tokens": stop_tokens,
        },
        "training": {
            "checkpoint_steps": checkpoint_steps,
            "full_step_limit": full_step_limit,
            "nominal_global_tokens_per_step": nominal_global_tokens_per_step,
            "total_tokens": total_tokens,
        },
    }
    plan["plan_fingerprint"] = _fingerprint(plan)
    return plan


class FinalCorpusTokenDataset(Dataset[torch.Tensor]):
    """Memory-map fixed sequences over the logical concatenation of final shards."""

    def __init__(
        self,
        manifest_path: str | Path,
        shard_root: str | Path,
        *,
        sequence_length: int = 4096,
        vocab_size: int = 65_536,
        doc_token_id: int = 3,
        expected_manifest_sha256: str | None = None,
        verify_hashes: bool = True,
    ) -> None:
        self.shard_root = Path(shard_root)
        self.manifest = validate_final_corpus_training_manifest(
            manifest_path,
            shard_root=self.shard_root,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            doc_token_id=doc_token_id,
            expected_manifest_sha256=expected_manifest_sha256,
            verify_hashes=verify_hashes,
        )
        self.sequence_length = sequence_length
        self._token_ends: list[int] = []
        self._shards: list[torch.Tensor] = []
        for shard in self.manifest["shards"]:
            self._shards.append(
                torch.from_file(
                    str(self.shard_root / shard["local_name"]),
                    shared=False,
                    size=shard["tokens"],
                    dtype=torch.uint16,
                )
            )
            self._token_ends.append(int(shard["token_end"]))
        self._sequences = int(self.manifest["statistics"]["sequences"])

    def __len__(self) -> int:
        return self._sequences

    def __getitem__(self, index: int) -> torch.Tensor:
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)
        token_cursor = index * self.sequence_length
        remaining = self.sequence_length
        pieces: list[torch.Tensor] = []
        while remaining:
            shard_index = bisect_right(self._token_ends, token_cursor)
            shard_start = 0 if shard_index == 0 else self._token_ends[shard_index - 1]
            local_start = token_cursor - shard_start
            available = self._shards[shard_index].numel() - local_start
            take = min(remaining, available)
            pieces.append(self._shards[shard_index][local_start : local_start + take])
            token_cursor += take
            remaining -= take
        if len(pieces) == 1:
            return pieces[0].to(dtype=torch.long)
        return torch.cat(pieces).to(dtype=torch.long)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def stage_final_corpus_shards(
    manifest_path: str | Path,
    output: str | Path,
    *,
    fetch: Callable[[str, Path], None],
    sequence_length: int = 4096,
    vocab_size: int = 65_536,
    doc_token_id: int = 3,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Resumably stage and independently hash every immutable corpus object."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    adapter = validate_final_corpus_training_manifest(
        manifest_path,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        doc_token_id=doc_token_id,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    state_path = output / "staging-state.json"
    completed = 0
    for shard in adapter["shards"]:
        destination = output / shard["local_name"]
        if destination.exists():
            if destination.stat().st_size != shard["bytes"]:
                raise ValueError(f"existing staged shard size changed: {destination}")
            if _sha256_file(destination) != shard["sha256"]:
                raise ValueError(f"existing staged shard SHA-256 changed: {destination}")
        else:
            temporary = destination.with_name(f".{destination.name}.partial")
            if temporary.exists():
                temporary.unlink()
            fetch(shard["s3_uri"], temporary)
            if not temporary.is_file() or temporary.stat().st_size != shard["bytes"]:
                raise ValueError(f"downloaded final shard size changed: {temporary}")
            if _sha256_file(temporary) != shard["sha256"]:
                raise ValueError(f"downloaded final shard SHA-256 changed: {temporary}")
            temporary.replace(destination)
        completed += 1
        _write_json_atomic(
            state_path,
            {
                "completed_shards": completed,
                "final_manifest_sha256": adapter["source"]["manifest_sha256"],
                "gate": STAGING_STATE_GATE,
                "status": "active" if completed < len(adapter["shards"]) else "complete",
                "total_shards": len(adapter["shards"]),
            },
        )
    return validate_final_corpus_training_manifest(
        manifest_path,
        shard_root=output,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        doc_token_id=doc_token_id,
        expected_manifest_sha256=expected_manifest_sha256,
        verify_hashes=True,
    )


def aws_cli_fetch(uri: str, destination: Path) -> None:
    """Fetch one object with the caller's already-scoped AWS credentials."""

    subprocess.run(
        ["aws", "s3", "cp", uri, str(destination), "--only-show-errors"],
        check=True,
    )


def seal_checkpoint_handoff(
    checkpoint: str | Path,
    output: str | Path,
    *,
    source_commit: str,
    run_fingerprint: str,
    world_size: int,
) -> dict[str, Any]:
    """Seal a complete distributed checkpoint for another instance/account."""

    checkpoint = Path(checkpoint)
    output = Path(output)
    _valid_sha256(run_fingerprint, "run fingerprint")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA-1")
    world_size = _positive_int(world_size, "world size")
    state_path = checkpoint / "distributed-training-state.json"
    checksum_path = checkpoint / "manifest.sha256"
    if not state_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError("checkpoint is missing durable state or manifest.sha256")
    from .distributed_training import verify_distributed_checkpoint

    state = verify_distributed_checkpoint(checkpoint)
    if state.get("run_fingerprint") != run_fingerprint:
        raise ValueError("checkpoint handoff targets another run")
    if state.get("run_specification", {}).get("source_commit") != source_commit:
        raise ValueError("checkpoint handoff source commit changed")
    if state.get("run_specification", {}).get("distributed", {}).get("world_size") != world_size:
        raise ValueError("checkpoint handoff world size changed")
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in checkpoint.rglob("*") if item.is_file()):
        relative = path.relative_to(checkpoint).as_posix()
        if relative == output.name and output.parent == checkpoint:
            continue
        files.append(
            {"bytes": path.stat().st_size, "path": relative, "sha256": _sha256_file(path)}
        )
    handoff: dict[str, Any] = {
        "checkpoint": {
            "directory": checkpoint.name,
            "sequence_cursor": int(state["sequence_cursor"]),
            "step": int(state["step"]),
            "tokens_seen": int(state["tokens_seen"]),
        },
        "files": files,
        "gate": CHECKPOINT_HANDOFF_GATE,
        "run_fingerprint": run_fingerprint,
        "schema_version": 1,
        "source_commit": source_commit,
        "status": "sealed",
        "world_size": world_size,
    }
    handoff["handoff_sha256"] = _fingerprint(handoff)
    _write_json_atomic(output, handoff)
    return handoff


def verify_checkpoint_handoff(
    checkpoint: str | Path,
    handoff_path: str | Path,
    *,
    expected_run_fingerprint: str | None = None,
    expected_source_commit: str | None = None,
    expected_world_size: int | None = None,
) -> dict[str, Any]:
    """Verify every handoff byte before a resumed paid segment can start."""

    checkpoint = Path(checkpoint)
    handoff = json.loads(Path(handoff_path).read_text(encoding="utf-8"))
    if handoff.get("schema_version") != 1 or handoff.get("gate") != CHECKPOINT_HANDOFF_GATE:
        raise ValueError("unsupported checkpoint handoff schema")
    if handoff.get("status") != "sealed":
        raise ValueError("checkpoint handoff is not sealed")
    if handoff.get("handoff_sha256") != _fingerprint(handoff, "handoff_sha256"):
        raise ValueError("checkpoint handoff fingerprint changed")
    if expected_run_fingerprint is not None and handoff.get(
        "run_fingerprint"
    ) != expected_run_fingerprint:
        raise ValueError("checkpoint handoff run fingerprint changed")
    if expected_source_commit is not None and handoff.get(
        "source_commit"
    ) != expected_source_commit:
        raise ValueError("checkpoint handoff source commit changed")
    if expected_world_size is not None and handoff.get("world_size") != expected_world_size:
        raise ValueError("checkpoint handoff world size changed")
    observed: set[str] = set()
    for row in handoff.get("files", []):
        relative = Path(str(row.get("path")))
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in observed:
            raise ValueError("checkpoint handoff contains an unsafe file path")
        observed.add(relative.as_posix())
        path = checkpoint / relative
        if not path.is_file() or path.stat().st_size != int(row.get("bytes", -1)):
            raise ValueError(f"checkpoint handoff file size changed: {relative}")
        if _sha256_file(path) != row.get("sha256"):
            raise ValueError(f"checkpoint handoff file SHA-256 changed: {relative}")
    actual = {
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*")
        if path.is_file() and path.resolve() != Path(handoff_path).resolve()
    }
    if actual != observed:
        raise ValueError("checkpoint handoff inventory changed")
    return handoff


def copy_fixture_fetch(uri: str, destination: Path) -> None:
    """Copy a file URI; useful for deterministic tests and offline preflight."""

    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("fixture fetch requires a file:// URI")
    shutil.copyfile(Path(parsed.path), destination)
