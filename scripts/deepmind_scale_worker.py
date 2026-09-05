#!/usr/bin/env python3
"""Generate one deterministic, bounded DeepMind Mathematics scale shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import deepmind_canary_worker as base

SPLIT_NAMESPACES = {
    "train": "lumi-deepmind-math-train-scale-v0.1",
    "validation": "lumi-deepmind-math-validation-scale-v0.1",
}
SPLIT_MODULE_COUNTS = {
    "train": {
        "algebra": 29_688,
        "arithmetic": 29_688,
        "calculus": 29_688,
        "comparison": 29_688,
        "measurement": 29_687,
        "numbers": 29_687,
        "polynomials": 29_687,
        "probability": 29_687,
    },
    "validation": {
        "algebra": 1_563,
        "arithmetic": 1_563,
        "calculus": 1_563,
        "comparison": 1_563,
        "measurement": 1_562,
        "numbers": 1_562,
        "polynomials": 1_562,
        "probability": 1_562,
    },
}
RNG_SEED_MULTIPLIER = 0x9E3779B1
RNG_SEED_OFFSET = 0x4C554D49
MAX_GENERATED_RECORDS = 250_000
STATS_PREFIX = "LUMI_DEEPMIND_SCALE_STATS="


def _expected_shards() -> dict[tuple[str, str], tuple[int, int]]:
    result: dict[tuple[str, str], tuple[int, int]] = {}
    offset = 0
    for split in ("train", "validation"):
        for module in base.TOP_MODULES:
            count = SPLIT_MODULE_COUNTS[split][module]
            result[(SPLIT_NAMESPACES[split], module)] = (count, offset)
            offset += count
    if offset != MAX_GENERATED_RECORDS:
        raise AssertionError("DeepMind scale worker schedule changed")
    return result


def _identity_seed(namespace: str, module_file: str, index: int) -> int:
    payload = f"{namespace}\0{module_file}\0{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _rng_seed(global_ordinal: int) -> int:
    return (
        RNG_SEED_MULTIPLIER * global_ordinal + RNG_SEED_OFFSET
    ) & 0xFFFFFFFF


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--top-module", choices=base.TOP_MODULES, required=True)
    parser.add_argument("--namespace", choices=sorted(SPLIT_NAMESPACES.values()), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--global-offset", type=int, required=True)
    args = parser.parse_args()

    expected = _expected_shards().get((args.namespace, args.top_module))
    if expected != (args.count, args.global_offset):
        raise ValueError("DeepMind scale worker shard boundary changed")
    source_root = args.source_root.resolve()
    module_file = base.TOP_MODULE_FILES[args.top_module]
    base.ALLOWED_NAMESPACES = set(SPLIT_NAMESPACES.values())
    selected, audit = base._load_train_modules(source_root, args.top_module)  # noqa: SLF001
    names = sorted(selected)

    dropped_length = 0
    exception_counts: Counter[str] = Counter()
    module_sample_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for index in range(args.count):
        global_ordinal = args.global_offset + index
        identity_seed = _identity_seed(args.namespace, module_file, index)
        rng_seed = _rng_seed(global_ordinal)
        base._seed_all_generators(rng_seed)  # noqa: SLF001
        module_name = names[index % len(names)]
        module_fn = selected[module_name]
        accepted = None
        for _attempt in range(base.MAX_ATTEMPTS_PER_RECORD):
            try:
                problem = module_fn()
                question = str(problem.question)
                answer = str(problem.answer)
            except Exception as error:  # noqa: BLE001 - bounded pinned upstream code
                exception_counts[type(error).__name__] += 1
                continue
            if (
                len(question) > base.MAX_QUESTION_CHARACTERS
                or len(answer) > base.MAX_ANSWER_CHARACTERS
            ):
                dropped_length += 1
                continue
            if not question.strip() or not answer.strip():
                exception_counts["EmptyProblem"] += 1
                continue
            accepted = {
                "answer": answer,
                "module": module_name,
                "question": question,
                "rng_seed": rng_seed,
                "seed": identity_seed,
                "split_namespace": args.namespace,
            }
            break
        if accepted is None:
            raise RuntimeError(
                "DeepMind scale worker exhausted bounded attempts without exposing "
                f"source text: top_module={args.top_module} index={index}"
            )
        records.append(accepted)
        module_sample_counts[module_name] += 1

    for record in records:
        sys.stdout.buffer.write(base._canonical_json_bytes(record) + b"\n")  # noqa: SLF001
    sys.stdout.buffer.flush()
    stats = {
        **audit,
        "count": args.count,
        "dropped_length": dropped_length,
        "exception_counts": dict(sorted(exception_counts.items())),
        "global_offset": args.global_offset,
        "identity_seed_bits": 64,
        "module_sample_counts": dict(sorted(module_sample_counts.items())),
        "namespace": args.namespace,
        "records": len(records),
        "rng_seed_algorithm": "injective_uint32_affine_permutation_of_global_ordinal",
        "top_module": args.top_module,
    }
    print(
        STATS_PREFIX + json.dumps(stats, sort_keys=True, separators=(",", ":")),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
