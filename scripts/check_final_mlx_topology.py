#!/usr/bin/env python3
"""Run or independently verify the final-width Lumi MLX portability gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.mlx_topology_gate import (
    run_final_mlx_topology_gate,
    verify_final_mlx_topology_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/mlx-final-topology-gate-v0.1.json"),
    )
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only and args.tokenizer is None:
        parser.error("--tokenizer is required unless --verify-only is used")
    return args


def main() -> None:
    args = parse_args()
    if args.verify_only:
        report = verify_final_mlx_topology_gate(args.output)
    else:
        report = run_final_mlx_topology_gate(
            plan_path=args.plan,
            tokenizer_path=args.tokenizer,
            output_path=args.output,
        )
    print(
        "PASS: final-width Lumi MLX topology gate; "
        f"parameters={report['fixture_parameter_count']:,}, "
        f"peak_mlx_bytes={report['parity']['mlx_peak_memory_bytes']:,}"
    )


if __name__ == "__main__":
    main()
