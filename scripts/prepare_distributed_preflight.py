#!/usr/bin/env python3
"""Create a deterministic tiny packed-data fixture for local distributed tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from array import array
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sequences < 8 or args.sequence_length < 8:
        raise ValueError("preflight fixture requires at least eight sequences of eight tokens")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"preflight fixture output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    data = args.output / "data"
    data.mkdir()

    config = {
        "attention_bias": False,
        "bos_token_id": 1,
        "doc_token_id": 3,
        "eos_token_id": 2,
        "head_dim": 16,
        "hidden_size": 32,
        "initialization_strategy": "residual_scaled_v0.1",
        "initializer_range": 0.02,
        "intermediate_size": 64,
        "max_position_embeddings": args.sequence_length,
        "model_type": "lumi",
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10_000.0,
        "tie_word_embeddings": True,
        "use_cache": False,
        "vocab_size": 128,
    }
    config_path = args.output / "lumi-distributed-preflight.json"
    _write_json(config_path, config)

    values = []
    for sequence in range(args.sequences):
        row = [3]
        row.extend(
            4 + ((sequence * 17 + offset * 11) % 124)
            for offset in range(1, args.sequence_length)
        )
        values.extend(row)
    packed = array("I", values)
    if packed.itemsize != 4:
        raise RuntimeError("preflight uint32 width changed")
    if sys.byteorder != "little":
        packed.byteswap()
    shard = data / "shard-00000.bin"
    shard.write_bytes(packed.tobytes())
    shard_sha256 = hashlib.sha256(shard.read_bytes()).hexdigest()
    tokens = args.sequences * args.sequence_length
    build_identity = {
        "sequence_length": args.sequence_length,
        "sequences": args.sequences,
        "shard_sha256": shard_sha256,
        "vocab_size": 128,
    }
    fingerprint = hashlib.sha256(
        json.dumps(build_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "build_fingerprint": fingerprint,
        "format": {
            "dtype": "uint32",
            "endianness": "little",
            "layout": "flat_fixed_sequences",
            "token_bytes": 4,
        },
        "packing": {"sequence_length": args.sequence_length},
        "schema_version": 1,
        "shards": [
            {
                "bytes": shard.stat().st_size,
                "name": shard.name,
                "sequences": args.sequences,
                "sha256": shard_sha256,
                "tokens": tokens,
            }
        ],
        "statistics": {"sequences": args.sequences, "tokens": tokens},
        "status": "complete",
        "tokenizer": {"doc_token_id": 3, "vocab_size": 128},
    }
    manifest_path = data / "manifest.json"
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "config": str(config_path),
                "manifest": str(manifest_path),
                "tokens": tokens,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
