#!/usr/bin/env python3
"""Verify Hopper FA3 varlen forward/backward parity against pinned FA2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from flash_attn import flash_attn_varlen_func as flash_attn_2_varlen
from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("FA3 parity requires Hopper sm_90")
    torch.manual_seed(20260802)
    device = torch.device("cuda:0")
    lengths = [17, 31, 9, 71]
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    cu = torch.tensor(cumulative, dtype=torch.int32, device=device)
    total = cumulative[-1]
    q = torch.randn(total, 64, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(total, 8, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(total, 8, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    dout = torch.randn_like(q)
    output_2 = flash_attn_2_varlen(
        q,
        k,
        v,
        cu,
        cu,
        max(lengths),
        max(lengths),
        dropout_p=0.0,
        softmax_scale=64**-0.5,
        causal=True,
    )
    gradients_2 = torch.autograd.grad(output_2, (q, k, v), dout, retain_graph=True)
    output_3 = flash_attn_3_varlen(
        q,
        k,
        v,
        cu,
        cu,
        max(lengths),
        max(lengths),
        softmax_scale=64**-0.5,
        causal=True,
    )
    if isinstance(output_3, tuple):
        output_3 = output_3[0]
    gradients_3 = torch.autograd.grad(output_3, (q, k, v), dout)
    output_max_abs = float((output_2 - output_3).abs().max().item())
    gradient_max_abs = [
        float((left - right).abs().max().item())
        for left, right in zip(gradients_2, gradients_3, strict=True)
    ]
    if not torch.allclose(output_2, output_3, atol=3e-2, rtol=3e-2):
        raise RuntimeError(f"FA3 forward parity failed: max_abs={output_max_abs}")
    for index, (left, right) in enumerate(zip(gradients_2, gradients_3, strict=True)):
        if not torch.allclose(left, right, atol=6e-2, rtol=6e-2):
            raise RuntimeError(
                f"FA3 gradient parity failed for tensor {index}: "
                f"max_abs={gradient_max_abs[index]}"
            )
    result = {
        "causal": True,
        "dtype": "bfloat16",
        "gradient_max_abs": gradient_max_abs,
        "head_dim": 64,
        "key_value_heads": 8,
        "output_max_abs": output_max_abs,
        "query_heads": 64,
        "sequence_lengths": lengths,
        "status": "pass",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
