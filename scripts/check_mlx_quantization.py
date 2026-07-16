#!/usr/bin/env python3
"""Run Lumi's canonical 4-bit MLX quality and M5 runtime gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.mlx_quantization_gate import run_quantization_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fp32-model", type=Path, required=True)
    parser.add_argument("--quantized-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences-per-stratum", type=int, default=2)
    parser.add_argument("--benchmark-prefill-tokens", type=int, default=512)
    parser.add_argument("--benchmark-decode-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    evaluation = plan["evaluation"]
    report = run_quantization_gate(
        fp32_model_path=args.fp32_model,
        quantized_model_path=args.quantized_model,
        manifest_path=args.manifest,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        prompts=evaluation["prompts"],
        sequences_per_stratum=args.sequences_per_stratum,
        generation_tokens=evaluation["generation_max_new_tokens"],
        benchmark_prefill_tokens=args.benchmark_prefill_tokens,
        benchmark_decode_tokens=args.benchmark_decode_tokens,
    )
    comparison = report["comparison"]
    print(
        f"PASS: 4-bit loss_delta={comparison['heldout_loss_delta']:+.6f} "
        f"top1={comparison['heldout_top1_agreement']:.3%} "
        f"size_reduction={comparison['artifact_size_reduction_fraction']:.1%}"
    )


if __name__ == "__main__":
    main()
