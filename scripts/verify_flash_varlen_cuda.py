#!/usr/bin/env python3
"""Verify Lumi's real FlashAttention varlen path on a CUDA GPU."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import torch

from lumi import LumiConfig, LumiForCausalLM, build_packed_document_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tiny_config(*, attention_implementation: str) -> LumiConfig:
    config = LumiConfig(
        vocab_size=257,
        hidden_size=128,
        intermediate_size=384,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        rope_theta=10_000.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        doc_token_id=3,
        tie_word_embeddings=True,
        use_cache=False,
    )
    config._attn_implementation = attention_implementation
    return config


def packed_batch() -> torch.LongTensor:
    generator = torch.Generator().manual_seed(20260717)
    input_ids = torch.randint(4, 257, (2, 64), generator=generator)
    for row, starts in enumerate(((0, 17, 43), (0, 11, 29, 52))):
        input_ids[row, list(starts)] = 3
    return input_ids


def require_finite_gradients(model: LumiForCausalLM) -> tuple[int, float]:
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        raise RuntimeError("FlashAttention backward produced no gradients")
    if not all(bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
        raise RuntimeError("FlashAttention backward produced non-finite gradients")
    norm = torch.linalg.vector_norm(
        torch.stack([parameter.grad.detach().float().norm() for parameter in parameters])
    )
    if not math.isfinite(float(norm)) or float(norm) <= 0.0:
        raise RuntimeError("FlashAttention backward gradient norm is invalid")
    return len(parameters), float(norm)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")

    try:
        import flash_attn
        import transformers
    except ImportError as error:
        raise RuntimeError("flash-attn and transformers must be installed") from error

    device = torch.device("cuda")
    input_ids_cpu = packed_batch()
    metadata_cpu = build_packed_document_metadata(input_ids_cpu, doc_token_id=3)
    input_ids = input_ids_cpu.to(device)
    metadata = {
        "cu_seq_lens_k": metadata_cpu.cu_seq_lens.to(device),
        "cu_seq_lens_q": metadata_cpu.cu_seq_lens.to(device),
        "document_isolation": True,
        "max_length_k": metadata_cpu.max_length,
        "max_length_q": metadata_cpu.max_length,
        "position_ids": metadata_cpu.position_ids.to(device),
    }

    torch.manual_seed(29)
    initial = LumiForCausalLM(tiny_config(attention_implementation="eager"))
    initial_state = {name: tensor.detach().clone() for name, tensor in initial.state_dict().items()}

    eager = LumiForCausalLM(tiny_config(attention_implementation="eager"))
    eager.load_state_dict(initial_state)
    eager.to(device=device, dtype=torch.bfloat16).eval()

    flash = LumiForCausalLM(tiny_config(attention_implementation="flash_attention_2"))
    flash.load_state_dict(initial_state)
    flash.to(device=device, dtype=torch.bfloat16).eval()

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        expected = eager(input_ids=input_ids, **metadata).logits
        actual = flash(input_ids=input_ids, **metadata).logits
    torch.cuda.synchronize(device)

    difference = (actual - expected).abs()
    max_absolute_error = float(difference.max())
    mean_absolute_error = float(difference.mean())
    cosine_similarity = float(
        torch.nn.functional.cosine_similarity(
            actual.flatten().float(), expected.flatten().float(), dim=0
        )
    )
    if not bool(torch.isfinite(actual).all()):
        raise RuntimeError("FlashAttention forward produced non-finite logits")
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.02)
    if cosine_similarity < 0.999:
        raise RuntimeError(f"FlashAttention cosine similarity too low: {cosine_similarity}")

    changed_cpu = input_ids_cpu.clone()
    changed_cpu[0, 1:17] = torch.arange(101, 117)
    changed = changed_cpu.to(device)
    changed_metadata_cpu = build_packed_document_metadata(changed_cpu, doc_token_id=3)
    changed_metadata = {
        "cu_seq_lens_k": changed_metadata_cpu.cu_seq_lens.to(device),
        "cu_seq_lens_q": changed_metadata_cpu.cu_seq_lens.to(device),
        "document_isolation": True,
        "max_length_k": changed_metadata_cpu.max_length,
        "max_length_q": changed_metadata_cpu.max_length,
        "position_ids": changed_metadata_cpu.position_ids.to(device),
    }
    with torch.no_grad():
        original_tail = flash(input_ids=input_ids, **metadata).logits[0, 43:]
        changed_tail = flash(input_ids=changed, **changed_metadata).logits[0, 43:]
    leakage_max_absolute_error = float((original_tail - changed_tail).abs().max())
    if leakage_max_absolute_error > 5e-4:
        raise RuntimeError(
            "earlier-document tokens changed isolated-document logits: "
            f"max_error={leakage_max_absolute_error}"
        )

    del eager, initial
    flash.train()
    flash.zero_grad(set_to_none=True)
    loss = flash(input_ids=input_ids, labels=input_ids, **metadata).loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError(f"FlashAttention backward loss is invalid: {loss}")
    loss.backward()
    gradient_tensors, gradient_norm = require_finite_gradients(flash)
    torch.cuda.synchronize(device)

    report = {
        "backward": {
            "finite_gradient_tensors": gradient_tensors,
            "gradient_norm": gradient_norm,
            "loss": float(loss.detach()),
        },
        "device": {
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "name": torch.cuda.get_device_name(),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "forward_parity": {
            "atol": 0.02,
            "cosine_similarity": cosine_similarity,
            "max_absolute_error": max_absolute_error,
            "mean_absolute_error": mean_absolute_error,
            "rtol": 0.05,
        },
        "isolation": {
            "changed_prefix": [1, 17],
            "checked_tail": [43, 64],
            "max_absolute_error": leakage_max_absolute_error,
        },
        "metadata": {
            "cu_seq_lens": metadata_cpu.cu_seq_lens.tolist(),
            "max_length": metadata_cpu.max_length,
            "segments": metadata_cpu.cu_seq_lens.numel() - 1,
        },
        "runtime": {
            "cuda": torch.version.cuda,
            "flash_attn": flash_attn.__version__,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "schema_version": 1,
        "status": "pass",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.output)
    print(rendered, end="")
    print(f"PASS: report={args.output}")


if __name__ == "__main__":
    main()
