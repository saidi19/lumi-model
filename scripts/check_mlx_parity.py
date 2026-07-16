#!/usr/bin/env python3
"""Run the canonical PyTorch-to-MLX parity gate for Lumi."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.mlx_parity import check_mlx_parity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mlx-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-absolute-error", type=float, default=2e-2)
    parser.add_argument("--max-mean-absolute-error", type=float, default=3e-3)
    parser.add_argument("--min-cosine-similarity", type=float, default=0.9999999)
    parser.add_argument("--max-normalized-rmse", type=float, default=1e-3)
    parser.add_argument("--max-cached-logprob-error", type=float, default=2e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    evaluation = plan["evaluation"]
    report = check_mlx_parity(
        checkpoint_path=args.checkpoint,
        mlx_path=args.mlx_model,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        prompts=evaluation["prompts"],
        generation_max_new_tokens=evaluation["generation_max_new_tokens"],
        max_absolute_error=args.max_absolute_error,
        max_mean_absolute_error=args.max_mean_absolute_error,
        min_cosine_similarity=args.min_cosine_similarity,
        max_normalized_root_mean_square_error=args.max_normalized_rmse,
        max_cached_logprob_error=args.max_cached_logprob_error,
    )
    maximum_error = max(
        prompt["metrics"]["max_absolute_error"] for prompt in report["prompts"]
    )
    print(
        f"PASS: PyTorch/MLX logits and cached greedy generation agree; "
        f"maximum absolute error={maximum_error:.3e}"
    )


if __name__ == "__main__":
    main()
