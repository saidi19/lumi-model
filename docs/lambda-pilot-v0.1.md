# Lumi Lambda pilot v0.1

Status: **complete; the paid H100 smoke/resume and 50M-token plumbing run pass,
and the final resumable checkpoint is hash-verified on the Mac**

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
- instance: 1× H100 SXM5 with 80GB HBM3 (`gpu_1x_h100_sxm5`);
- region: North Texas, USA (`us-south-2`);
- image: Lambda Stack 22.04;
- observed price on 2026-07-16: USD 4.29/hour;
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

The paid runtime audit recorded an NVIDIA H100 80GB HBM3, compute capability
9.0, driver 580.105.08, and 81,559MiB reported VRAM. The isolated training
environment used Python 3.12.13 and PyTorch 2.7.0+cu126. CUDA and BF16 both
passed. The cloud preflight rehashed all 48 shards and the tokenizer, recovered
the exact 150,625,536 analytical parameter count, and matched the pinned
manifest, model-config, and build fingerprints before training began.

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

### Verified smoke result

| Step | Loss | Gradient norm | Tokens/second | Peak allocated | Peak reserved | Checkpoint |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.229181 | 5.149001 | 35,073 | 12.402GiB | 14.836GiB | 2.476s |
| 2, fresh-process resume | 10.784764 | 6.230686 | 47,627 | 13.525GiB | 16.023GiB | 2.688s |

The second command ran in a new Python process, restored model, optimizer, RNG,
and exact data position from checkpoint 1, then wrote checkpoint 2. Both losses
and gradient norms remained finite.

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

## Verified full-run result

The full process ran from 2026-07-16 10:44:15 UTC through 10:51:44 UTC: 7m29s
wall time for 763 optimizer steps and 50,003,968 tokens. The extra 4,096 tokens
are the deterministic wrap pinned in the plan.

| Metric | Result |
|---|---:|
| First-step loss | 11.234901 |
| Best observed minibatch loss | 4.236981 |
| Final-step loss | 4.669532 |
| Mean step throughput | 117,888 tokens/second |
| Median step throughput | 117,968 tokens/second |
| Peak allocated VRAM | 13.536GiB |
| Peak reserved VRAM | 14.867GiB |
| Checkpoints | steps 100, 200, 300, 400, 500, 600, 700, and 763 |
| Total checkpoint time | 20.654s |
| Final checkpoint time | 2.511s |

All 763 metric rows report finite loss, finite gradient norm, positive
throughput, and CUDA memory measurements. The final state is `complete` at step
763. This validates the artifact, runtime, optimization, metric, checkpoint,
and resume plumbing; it does not establish production model quality.

## Archived checkpoint

The final checkpoint is stored outside Git at
`$LUMI_DATA/lambda-pilot-v0.1/full/checkpoint-00000763`. It contains the model,
optimizer, RNG, and training state required for an exact continuation. The Mac
copy was verified against the server-generated `SHA256SUMS` before cloud
cleanup. Key hashes are:

- `model.safetensors` (602,520,984 bytes):
  `656edd0d2eb74eaeb1cd3e26f7c0265ce3d9a88f47e310e51041ba39e16f1431`;
- `trainer.pt` (1,205,126,597 bytes):
  `80d9f270ef28af0b8af421ffb0e6e543044ee6c5644ea7d7de9938a2a3512bfb`;
- `training-state.json`:
  `c5fb80540bab50a48490bea24fc06e4cf8e021868ea54464dcaa0bef29e6042e`.

The top-level 763-row metrics, completed state, training log, and two-row smoke
resume evidence are archived beside the checkpoint. After the local archive
passed hashes plus model and optimizer load checks, the Lambda console
confirmed **No running instances** at 2026-07-16 11:12 UTC. The finalized usage
entry records a billable window from 10:33 through 11:10 UTC, 0.62 hours at
USD 4.29/hour, and total spend of **USD 2.64**.
