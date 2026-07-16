#!/usr/bin/env python3
"""Convert one self-contained Lumi checkpoint into a native MLX artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.mlx_conversion import SUPPORTED_DTYPES, convert_checkpoint_to_mlx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=SUPPORTED_DTYPES, default="float32")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--quantization-bits", type=int, default=4)
    parser.add_argument("--quantization-group-size", type=int, default=64)
    parser.add_argument("--quantization-mode", choices=("affine",), default="affine")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_checkpoint_to_mlx(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        dtype=args.dtype,
        quantize=args.quantize,
        quantization_bits=args.quantization_bits,
        quantization_group_size=args.quantization_group_size,
        quantization_mode=args.quantization_mode,
    )
    quantization = report["request"]["quantization"]
    precision = "unquantized" if quantization is None else f"{quantization['bits']}-bit"
    print(
        f"PASS: converted {report['output']['model_parameters']:,} parameters "
        f"to {precision} MLX at {args.output}"
    )


if __name__ == "__main__":
    main()
