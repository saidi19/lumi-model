# Lumi Lambda pilot v0.1

Status: **the portable bundle passes local preflight; the paid H100 run is
approved but has not started**

## Purpose

This run is the first paid CUDA gate for Lumi. It proves that the exact sealed
artifact, 150M model, BF16 execution, fused AdamW, metrics, and checkpoints work
together on an H100. It is a plumbing and throughput experiment, not a model
quality claim.

## Pinned target

| Item | Value |
|---|---:|
| Parameters | 150,625,536 |
| Packed tokens | 49,999,872 |
| Sequences | 24,414 × 2,048 |
| Source strata | 24 |
| Shards | 48 |
| Packed manifest SHA-256 | `a8d964164221f81acb5805de767ea0cc1d5baf75eda31462ad01badae4b9522c` |
| Build fingerprint | `2a7187332b1f42f7290310091d11ad0d8cca3a2005b4d6b9df402a98e3347645` |

`configs/cloud-pilot-v0.1.json` pins both smoke and full-run settings. The full
run uses four sequences per microbatch, eight-way gradient accumulation, and
763 optimizer steps. That processes 50,003,968 tokens: one corpus pass plus a
deterministic 4,096-token wrap, which is less than one full optimizer step.

## Selected cloud target

- provider: Lambda Cloud;
- instance: 1× H100 PCIe with 80GB VRAM;
- region: Utah, USA;
- image: Lambda Stack 22.04;
- observed price on 2026-07-16: USD 3.29/hour;
- persistent filesystem: none;
- access: an existing SSH public key, with the private key kept outside Git.

The owner explicitly approved proceeding without a fixed wall-time cost cap.
The instance must still be terminated after results are copied or immediately
after a terminal failure; it must not remain idle.

## Portable preflight

Run this once on the Mac before transfer and again on the server with
`--require-cuda`:

```bash
python scripts/preflight_cloud_pilot.py \
  --plan configs/cloud-pilot-v0.1.json \
  --manifest "$LUMI_DATA/real-pilot-50m-v0.1/packed/manifest.json" \
  --tokenizer "$LUMI_DATA/tokenizer-candidates-v0.2/lumi-tokenizer-65k"
```

The cloud invocation adds `--require-cuda`. It fails closed unless:

- every shard size and SHA-256 matches the sealed manifest;
- tokenizer file sizes, hashes, vocabulary length, and document-token ID match;
- config hash, context length, vocabulary, and analytical parameter count match;
- CUDA is available and reports BF16 support.

## Transfer layout

The Mac sends only the repository, the 198MB packed directory, and the 4.4MB
frozen tokenizer. The original 7.7GB JSONL source is not required for training.
The remote layout is:

```text
~/lumi-model/
~/lumi-data/real-pilot-50m-v0.1/packed/
~/lumi-data/tokenizer-candidates-v0.2/lumi-tokenizer-65k/
~/lumi-runs/
```

Python 3.12 is required. Before installing project dependencies, record
`nvidia-smi`, Python, PyTorch, CUDA, free disk, and GPU memory information. Do
not copy the SSH private key to the server.

## Smoke and resume gate

The first process performs one optimizer step and writes a checkpoint:

```bash
python scripts/train_pilot.py \
  --manifest "$HOME/lumi-data/real-pilot-50m-v0.1/packed/manifest.json" \
  --config configs/lumi-150m.json \
  --output "$HOME/lumi-runs/150m-50m-smoke-v0.1" \
  --steps 1 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --checkpoint-steps 1 \
  --device cuda
```

A fresh process repeats the same settings with `--steps 2 --resume`. The gate
passes only when loss and gradient norm remain finite, checkpoint recovery
advances to step 2, CUDA peak-memory metrics are present, and the measured
throughput is nonzero.

## Full plumbing run

Only after the smoke/resume gate passes:

```bash
python scripts/train_pilot.py \
  --manifest "$HOME/lumi-data/real-pilot-50m-v0.1/packed/manifest.json" \
  --config configs/lumi-150m.json \
  --output "$HOME/lumi-runs/150m-50m-full-v0.1" \
  --steps 763 \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 0.0003 \
  --weight-decay 0.1 \
  --max-grad-norm 1.0 \
  --checkpoint-steps 100 \
  --seed 7 \
  --device cuda
```

The final report must record total runtime and provider cost, loss history,
tokens per second, peak allocated/reserved VRAM, checkpoint duration, restart
behavior, final checkpoint hashes, and the instance termination time.
