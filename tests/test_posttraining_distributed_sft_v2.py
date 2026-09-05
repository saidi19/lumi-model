from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import struct
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch

import lumi.posttraining_distributed_sft_v2 as distributed_sft
import lumi.posttraining_sft_runtime_v4 as runtime_v4
from lumi.configuration_lumi import LumiConfig
from lumi.distributed_training import (
    DistributedContext,
    destroy_distributed,
    write_final_model_digest,
)
from lumi.modeling_lumi import LumiForCausalLM


def _verification() -> dict[str, Any]:
    return {
        "all_inventory_file_hashes_verified": False,
        "binary_bytes_read": 1752,
        "encoded_index_row_reads": 4,
        "file_opens": 4,
        "full_content_hashes_verified": False,
        "fully_hashed_shards": 0,
        "index_bytes_read": 644,
        "inventory_sha256": "d" * 64,
        "manifest_file_sha256": "e" * 64,
        "manifest_sha256": "b" * 64,
        "metadata_inventory_sha256": "f" * 64,
        "metadata_inventory_verified_before_selected_content_read": True,
        "opened_shards": 1,
        "profile": "token_consistent_v0.2",
        "records_loaded": 4,
        "selected_prefix_sha256": "0" * 64,
        "unbuffered_selected_content_reads": True,
    }


def _identity(**overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "attention_implementation": "eager",
        "backend": "gloo",
        "checkpoint_steps": 1,
        "encoded_shards_record_limit": 4,
        "gradient_accumulation_steps": 1,
        "gradient_sync_mode": "last_microbatch",
        "keep_checkpoints": None,
        "learning_rate": 1e-4,
        "lr_schedule": "cosine",
        "max_grad_norm": 1.0,
        "min_learning_rate_ratio": 0.1,
        "optimizer_profile": "grouped_adamw_beta2_0.95_v0.1",
        "pad_to_multiple_of": 8,
        "per_rank_batch_size": 1,
        "precision": "fp32",
        "seed": 2027,
        "shard_manifest": {
            "context_window": 4096,
            "manifest_sha256": "b" * 64,
            "seal_sha256": "c" * 64,
        },
        "shard_verification": _verification(),
        "source_checkpoint_manifest_sha256": "a" * 64,
        "source_commit": None,
        "source_config_bytes": b'{"model_type":"lumi"}',
        "strategy": "ddp",
        "target_steps": 2,
        "warmup_steps": 0,
        "weight_decay": 0.01,
        "world_size": 2,
    }
    arguments.update(overrides)
    return distributed_sft.build_distributed_sft_run_identity(**arguments)


def test_v2_identity_binds_bounded_v4_read_contract() -> None:
    identity = _identity()

    assert identity["schema_version"] == "lumi.sft-distributed-state.v0.2"
    assert identity["objective"] == distributed_sft.SFT_DISTRIBUTED_OBJECTIVE
    assert identity["data"] == {
        "inventory_sha256": "d" * 64,
        "manifest_file_sha256": "e" * 64,
        "manifest_sha256": "b" * 64,
        "metadata_inventory_sha256": "f" * 64,
        "read_audit": {
            "binary_bytes_read": 1752,
            "encoded_index_row_reads": 4,
            "file_opens": 4,
            "fully_hashed_shards": 0,
            "index_bytes_read": 644,
            "metadata_inventory_verified_before_selected_content_read": True,
            "opened_shards": 1,
            "records_loaded": 4,
            "unbuffered_selected_content_reads": True,
        },
        "read_audit_accounting": {
            "all_rank_read_audits_equal": True,
            "rank_local_loads": 2,
        },
        "record_limit": 4,
        "seal_sha256": "c" * 64,
        "selection_scope": "contiguous_prefix_v0.1",
        "sequence_order": "cached_epoch_randperm_v0.1",
        "selected_prefix_sha256": "0" * 64,
        "split": "train",
        "verification_profile": "token_consistent_v0.2",
    }
    assert identity["distributed"]["checkpoint_steps"] == 1
    assert identity["distributed"]["keep_checkpoints"] is None
    assert identity["training"]["pad_to_multiple_of"] == 8
    assert _identity(checkpoint_steps=2) != identity
    assert _identity(pad_to_multiple_of=4) != identity


def test_bounded_contract_rejects_any_complete_shard_hash() -> None:
    verification = _verification()
    verification["fully_hashed_shards"] = 1

    with pytest.raises(ValueError, match="bounded-read contract changed"):
        _identity(shard_verification=verification)


def test_collective_loader_forwards_all_five_bindings_and_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_bytes(b"{}")
    captured: dict[str, Any] = {}

    class Dataset:
        manifest = {
            "context_window": 4096,
            "manifest_sha256": "b" * 64,
            "seal_sha256": "c" * 64,
        }
        verification = _verification()

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

    def gather(target: list[Any], value: Any) -> None:
        target[0] = value
        target[1] = {**value, "rank": 1}

    monkeypatch.setattr(distributed_sft, "EncodedSFTDataset", Dataset)
    monkeypatch.setattr(
        distributed_sft,
        "_verify_checksum_manifest",
        lambda *args, **kwargs: "a" * 64,
    )
    monkeypatch.setattr(distributed_sft.dist, "all_gather_object", gather)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        backend="gloo",
        device=torch.device("cpu"),
    )

    distributed_sft._collective_load_inputs(
        context=context,
        encoded_shards=tmp_path / "encoded",
        source_checkpoint=source,
        source_checkpoint_manifest_sha256="a" * 64,
        encoded_shards_inventory_sha256="d" * 64,
        encoded_shards_manifest_file_sha256="e" * 64,
        encoded_shards_manifest_sha256="b" * 64,
        encoded_shards_metadata_inventory_sha256="f" * 64,
        encoded_shards_record_limit=4,
        encoded_shards_seal_sha256="c" * 64,
    )

    assert captured == {
        "expected_inventory_sha256": "d" * 64,
        "expected_manifest_file_sha256": "e" * 64,
        "expected_manifest_sha256": "b" * 64,
        "expected_metadata_inventory_sha256": "f" * 64,
        "expected_seal_sha256": "c" * 64,
        "record_limit": 4,
        "split": "train",
    }


def test_collective_run_identity_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def gather(target: list[Any], value: Any) -> None:
        target[0] = value
        target[1] = {
            **value,
            "fingerprint": "f" * 64,
            "identity": {**value["identity"], "seed": 999},
            "rank": 1,
        }

    monkeypatch.setattr(distributed_sft.dist, "all_gather_object", gather)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        backend="gloo",
        device=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="one SFT run identity"):
        distributed_sft._collective_require_run_identity(context, _identity())


def _top_state() -> dict[str, Any]:
    return {
        "best_loss": 1.0,
        "created_at": "created",
        "last_checkpoint_seconds": 0.1,
        "last_loss": 1.0,
        "latest_checkpoint": "checkpoint-00000001",
        "records_seen": 2,
        "run_fingerprint": "a" * 64,
        "run_specification": {"schema_version": "lumi.sft-distributed-state.v0.2"},
        "schema_version": "lumi.sft-distributed-state.v0.2",
        "sequence_cursor": 2,
        "status": "active",
        "step": 1,
        "supervised_tokens_seen": 4,
        "tokens_seen": 6,
        "updated_at": "updated",
    }


def test_checkpoint_fields_are_validated_then_removed_from_top_state() -> None:
    reference = distributed_sft._normalize_top_state(_top_state())
    resumed = distributed_sft._normalize_top_state(
        {
            **_top_state(),
            "checkpoint_metric": {"step": 1},
            "checkpoint_schema_version": 1,
        }
    )

    assert resumed == reference
    assert set(resumed) == set(reference)
    assert "checkpoint_schema_version" not in resumed
    assert "checkpoint_metric" not in resumed

    with pytest.raises(ValueError, match="checkpoint fields changed"):
        distributed_sft._normalize_top_state(
            {
                **_top_state(),
                "checkpoint_metric": {"step": 1},
                "checkpoint_schema_version": 2,
            }
        )


def test_markers_are_private_exclusive_and_precede_future_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        distributed_sft,
        "_rank_zero_object",
        lambda context, callback: callback(),
    )
    monkeypatch.setattr(distributed_sft.dist, "barrier", lambda: None)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        backend="gloo",
        device=torch.device("cpu"),
    )
    output = tmp_path / "attempt" / "private" / "run"

    distributed_sft._mark_invocation(
        context=context,
        output=output,
        resume=False,
        stop_after_step=1,
        target_steps=2,
    )
    started = output / distributed_sft.SFT_DISTRIBUTED_STARTED_FILENAME
    assert stat.S_IMODE(started.stat().st_mode) == 0o600
    runtime_v4._write_json_atomic(
        output / "distributed-training-state.json",
        _top_state(),
    )
    metrics = output / "distributed-metrics.jsonl"
    metrics.write_bytes(b"")
    metrics.chmod(0o600)
    distributed_sft._mark_invocation(
        context=context,
        output=output,
        resume=True,
        stop_after_step=2,
        target_steps=2,
    )
    resume = output / distributed_sft.SFT_DISTRIBUTED_RESUME_STARTED_FILENAME
    assert stat.S_IMODE(resume.stat().st_mode) == 0o600

    with pytest.raises(FileExistsError):
        distributed_sft._mark_invocation(
            context=context,
            output=output,
            resume=True,
            stop_after_step=2,
            target_steps=2,
        )


def test_v2_runner_exposes_required_hardened_cli_and_private_umask() -> None:
    root = Path(__file__).resolve().parents[1]
    implementation = (
        root / "src/lumi/posttraining_distributed_sft_v2.py"
    ).read_text(encoding="utf-8")
    runner = (root / "scripts/train_distributed_sft_v2.py").read_text(
        encoding="utf-8"
    )

    assert "from .posttraining_sft_runtime_v4 import" in implementation
    assert "verify_sft_shards" not in implementation
    assert "posttraining_sft_runtime import" not in implementation
    for flag in (
        "--encoded-shards-inventory-sha256",
        "--encoded-shards-manifest-file-sha256",
        "--encoded-shards-manifest-sha256",
        "--encoded-shards-metadata-inventory-sha256",
        "--encoded-shards-record-limit",
        "--encoded-shards-seal-sha256",
    ):
        assert flag in runner
    assert "os.umask(0o077)" in runner
    assert "posttraining_distributed_sft_v2" in runner


def _write_private(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o600)


def _synthetic_encoded_root(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "encoded"
    root.mkdir(mode=0o700)
    specs = (
        (
            "train",
            0,
            tuple((10 + ordinal * 3, f"{ordinal + 1:064x}") for ordinal in range(5)),
        ),
        ("validation", 0, ((40, "f" * 64),)),
    )
    shards: list[dict[str, Any]] = []
    split_records: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    split_supervised: Counter[str] = Counter()
    encoded_hasher = hashlib.sha256()
    sequence_hasher = hashlib.sha256()
    for split, shard_index, records in specs:
        prefix = f"{split}-{shard_index:05d}"
        token_values: list[int] = []
        label_values: list[int] = []
        index_payload = bytearray()
        supervised_tokens = 0
        for ordinal, (base, record_hash) in enumerate(records):
            tokens = (base, base + 1, base + 2)
            labels = (-100, base + 1, base + 2)
            entry = {
                "offset_tokens": len(token_values),
                "record_id_sha256": record_hash,
                "record_ordinal": ordinal,
                "supervised_tokens": 2,
                "tokens": 3,
            }
            index_payload.extend(runtime_v4._canonical_json(entry) + b"\n")
            encoded = runtime_v4._encoded_hash_payload(record_hash, tokens, labels)
            encoded_hasher.update(len(encoded).to_bytes(8, "big"))
            encoded_hasher.update(encoded)
            sequence_hasher.update(f"{split}\0{record_hash}\n".encode())
            token_values.extend(tokens)
            label_values.extend(labels)
            supervised_tokens += 2
        payloads = {
            f"{prefix}.tokens.u32le": struct.pack(
                f"<{len(token_values)}I", *token_values
            ),
            f"{prefix}.labels.i32le": struct.pack(
                f"<{len(label_values)}i", *label_values
            ),
            f"{prefix}.index.jsonl": bytes(index_payload),
        }
        files = []
        for name, payload in payloads.items():
            _write_private(root / name, payload)
            files.append(
                {
                    "bytes": len(payload),
                    "path": name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        shards.append(
            {
                "files": files,
                "index": shard_index,
                "records": len(records),
                "split": split,
                "supervised_tokens": supervised_tokens,
                "tokens": len(token_values),
            }
        )
        split_records[split] += len(records)
        split_tokens[split] += len(token_values)
        split_supervised[split] += supervised_tokens
    manifest: dict[str, Any] = {
        "authorization_boundary": runtime_v4._TOKEN_CONSISTENT_AUTHORIZATION_BOUNDARY,
        "context_window": 4096,
        "encoded_dataset_sha256": encoded_hasher.hexdigest(),
        "format": runtime_v4._TOKEN_CONSISTENT_SHARD_FORMAT,
        "maximum_tokens_per_shard": 4_194_304,
        "production_eligible": False,
        "record_hash_sequence_sha256": sequence_hasher.hexdigest(),
        "records": sum(split_records.values()),
        "schema_version": "lumi.sft-encoded-shards.v0.1",
        "seal_sha256": "c" * 64,
        "shards": shards,
        "splits": {
            split: {
                "records": split_records[split],
                "supervised_tokens": split_supervised[split],
                "tokens": split_tokens[split],
            }
            for split in sorted(split_records)
        },
        "supervised_tokens": sum(split_supervised.values()),
        "tokens": sum(split_tokens.values()),
    }
    manifest["manifest_sha256"] = runtime_v4._sha256(manifest)
    manifest_payload = runtime_v4._canonical_json(manifest) + b"\n"
    _write_private(root / "manifest.json", manifest_payload)
    metadata = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        info = path.lstat()
        metadata.append(
            {
                "mode": f"{stat.S_IMODE(info.st_mode):04o}",
                "relative_path": path.name,
                "st_ctime_ns": info.st_ctime_ns,
                "st_dev": info.st_dev,
                "st_ino": info.st_ino,
                "st_mtime_ns": info.st_mtime_ns,
                "st_nlink": info.st_nlink,
                "st_size": info.st_size,
                "st_uid": info.st_uid,
            }
        )
    bindings = {
        "inventory": runtime_v4._sha256(manifest["shards"]),
        "manifest_file": hashlib.sha256(manifest_payload).hexdigest(),
        "manifest_internal": manifest["manifest_sha256"],
        "metadata": runtime_v4._sha256(metadata),
        "seal": manifest["seal_sha256"],
    }
    return root, bindings


def _synthetic_source_checkpoint(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "source"
    root.mkdir(mode=0o700)
    torch.manual_seed(17)
    model = LumiForCausalLM(
        LumiConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=4096,
        )
    )
    model.save_pretrained(root, safe_serialization=True)
    entries = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_file():
            path.chmod(0o600)
            entries.append((hashlib.sha256(path.read_bytes()).hexdigest(), path.name))
    manifest = root / "manifest.sha256"
    _write_private(
        manifest,
        "".join(f"{digest}  {name}\n" for digest, name in entries).encode(),
    )
    return root, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _two_rank_worker(
    rank: int,
    rendezvous: str,
    arguments: dict[str, Any],
) -> None:
    os.umask(0o077)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if platform.system() == "Darwin" else "lo"
    skip_digest = bool(arguments.pop("skip_digest"))
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    context = DistributedContext(
        rank=rank,
        local_rank=rank,
        world_size=2,
        backend="gloo",
        device=torch.device("cpu"),
    )
    state = None
    try:
        state = distributed_sft.train_distributed_sft(
            context=context,
            **arguments,
        )
    finally:
        destroy_distributed()
    if rank == 0 and not skip_digest:
        assert state is not None
        write_final_model_digest(
            config_path=Path(arguments["source_checkpoint"]) / "config.json",
            checkpoint=Path(arguments["output"]) / state["latest_checkpoint"],
            output=arguments["output"],
        )


def _run_two_rank(
    *,
    root: Path,
    encoded: Path,
    source: Path,
    source_manifest: str,
    bindings: dict[str, str],
    output: Path,
    stop_after_step: int,
    strategy: str = "ddp",
    resume: bool = False,
    skip_digest: bool = False,
) -> None:
    del root
    rendezvous = output.parent / (
        f".rendezvous-{output.name}-{stop_after_step}-{int(resume)}"
    )
    arguments: dict[str, Any] = {
        "attention_implementation": "eager",
        "authorization_path": None,
        "checkpoint_steps": 1,
        "encoded_shards": encoded,
        "encoded_shards_inventory_sha256": bindings["inventory"],
        "encoded_shards_manifest_file_sha256": bindings["manifest_file"],
        "encoded_shards_manifest_sha256": bindings["manifest_internal"],
        "encoded_shards_metadata_inventory_sha256": bindings["metadata"],
        "encoded_shards_record_limit": 4,
        "encoded_shards_seal_sha256": bindings["seal"],
        "gradient_accumulation_steps": 1,
        "gradient_sync_mode": "last_microbatch",
        "keep_checkpoints": 2,
        "learning_rate": 0.0001,
        "lr_schedule": "cosine",
        "max_grad_norm": 1.0,
        "min_learning_rate_ratio": 0.1,
        "optimizer_profile": "grouped_adamw_beta2_0.95_v0.1",
        "output": output,
        "pad_to_multiple_of": 8,
        "per_rank_batch_size": 1,
        "precision": "fp32",
        "resume": resume,
        "seed": 2027,
        "skip_digest": skip_digest,
        "source_checkpoint": source,
        "source_checkpoint_manifest_sha256": source_manifest,
        "source_commit": None,
        "stop_after_step": stop_after_step,
        "strategy": strategy,
        "target_steps": 2,
        "warmup_steps": 0,
        "weight_decay": 0.01,
    }
    worker_arguments = dict(arguments)
    worker_arguments.pop("skip_digest")
    torch.multiprocessing.spawn(
        _two_rank_worker,
        args=(str(rendezvous), {**worker_arguments, "skip_digest": skip_digest}),
        nprocs=2,
        join=True,
    )


def _without_volatile(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in state.items()
        if key not in {"created_at", "last_checkpoint_seconds", "updated_at"}
    }


@pytest.mark.skipif(not torch.distributed.is_available(), reason="distributed unavailable")
@pytest.mark.parametrize("strategy", ["ddp", "fsdp_full_shard"])
def test_real_two_rank_reference_and_resume_are_exact(
    tmp_path: Path, strategy: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    encoded, bindings = _synthetic_encoded_root(tmp_path)
    source, source_manifest = _synthetic_source_checkpoint(tmp_path)
    reference = tmp_path / "reference"
    resumed = tmp_path / "resumed"

    _run_two_rank(
        root=root,
        encoded=encoded,
        source=source,
        source_manifest=source_manifest,
        bindings=bindings,
        output=reference,
        stop_after_step=2,
        strategy=strategy,
    )
    _run_two_rank(
        root=root,
        encoded=encoded,
        source=source,
        source_manifest=source_manifest,
        bindings=bindings,
        output=resumed,
        stop_after_step=1,
        strategy=strategy,
    )
    _run_two_rank(
        root=root,
        encoded=encoded,
        source=source,
        source_manifest=source_manifest,
        bindings=bindings,
        output=resumed,
        stop_after_step=2,
        strategy=strategy,
        resume=True,
    )

    reference_state = json.loads(
        (reference / "distributed-training-state.json").read_text()
    )
    resumed_state = json.loads(
        (resumed / "distributed-training-state.json").read_text()
    )
    assert set(reference_state) == set(resumed_state)
    assert "checkpoint_schema_version" not in resumed_state
    assert "checkpoint_metric" not in resumed_state
    assert _without_volatile(reference_state) == _without_volatile(resumed_state)
    assert reference_state["run_specification"]["data"]["read_audit"][
        "fully_hashed_shards"
    ] == 0
    assert reference_state["run_specification"]["data"][
        "read_audit_accounting"
    ] == {"all_rank_read_audits_equal": True, "rank_local_loads": 2}
    reference_digest = json.loads((reference / "final-model-digest.json").read_text())
    resumed_digest = json.loads((resumed / "final-model-digest.json").read_text())
    assert reference_digest["sha256"] == resumed_digest["sha256"]
    distributed_sft.verify_distributed_sft_checkpoint(
        reference / reference_state["latest_checkpoint"]
    )
    distributed_sft.verify_distributed_sft_checkpoint(
        resumed / resumed_state["latest_checkpoint"]
    )
    from lumi import posttraining_distributed_cpu_preflight_v2 as preflight
    from lumi import posttraining_distributed_cpu_verification_v2 as verification

    audit = preflight._audit_strategy(
        strategy=strategy, reference=reference, resumed=resumed, plan={}
    )
    model_digests = verification._verify_model_pairs(
        config=json.loads((source / "config.json").read_text()),
        reference=reference,
        resumed=resumed,
    )
    assert audit["final_model_semantic_sha256"] == model_digests["2"]
    fixture_plan = {
        "input": {
            "source_checkpoint": str(source),
            "source_checkpoint_manifest_sha256": source_manifest,
            "encoded_directory": str(encoded),
        },
        "training": preflight.EXPECTED_TRAINING,
    }
    verification._verify_run_contract(reference_state, fixture_plan, tmp_path)
    for output in (reference, resumed):
        assert stat.S_IMODE(output.stat().st_mode) == 0o700
        for path in output.rglob("*"):
            mode = stat.S_IMODE(path.lstat().st_mode)
            assert mode == (0o700 if path.is_dir() else 0o600)
