# Lumi-1B Cloud Preflight v0.1

Status: **canonical local artifact preflight passes; provider selection, remote
runtime preflight, and paid GPU launch remain unapproved**

## Decision

The sealed 1.4B-token artifact is technically ready to transfer to a candidate
GPU host. It may not be trained merely because its local preflight passes. The
same command must run again on the selected host with `--require-cuda`, all
shard hashes enabled, and the intended checkpoint workspace supplied.

This schema-v2 gate extends the existing 150M preflight without changing its
schema-v1 contract. A technical PASS never grants spending authority or
approval for the 100-step, 250M-token, 1.4B-token, or 4B stages.

## Pinned bundle

`configs/cloud-pilot-1b-v0.1.json` has SHA-256
`9111b6532e64e20189a71f68259afc5749f3edd2822a205dca9a1f09494d05a9`
and pins:

| Field | Value |
| --- | ---: |
| Model parameters | 1,050,981,888 |
| Packed-manifest bytes | 39,421 |
| Packed-manifest SHA-256 | `9228fa610d4a985d05e15057e8ef12619a14cbedbb5a6a5ab513d3aeca2832c8` |
| Effective tokens | 1,399,996,416 |
| Sequences | 341,796 × 4,096 |
| Selected documents | 1,069,214 |
| Source strata | 24 |
| Shards | 42 |
| Raw shard bytes | 2,799,992,832 |
| Token format | little-endian uint16 |
| Held-out exact/family exclusions | 3,988 / 3,988 |
| ID/content/near-family overlaps | 0 / 0 / 0 |

The plan also pins the selection proof, compact selection-state database,
source-index identity, tokenizer files, Lumi-1B config, scaling-data contract,
and four-stage scaling-pilot contract. The raw 7.7GB JSONL source and 903MB
source-index database are not required on the training host.

## Runtime admission floors

Remote `--require-cuda` mode fails unless one device satisfies all GPU floors
and the host satisfies both resource floors:

| Requirement | Floor |
| --- | ---: |
| CUDA devices | 1 |
| Compute capability | 8.0 |
| Device memory | 40,000,000,000 bytes |
| CUDA BF16 | required |
| Host memory | 34,359,738,368 bytes (32 GiB) |
| Free workspace disk | 85,899,345,920 bytes (80 GiB) |

The checkpoint-aware disk calculation is 51,372,759,040 bytes. It reserves 12
bytes per parameter for FP32 model plus Adam moments, three simultaneous copies
for two retained checkpoints plus one in-flight save, the packed shards, and a
10GiB fixed margin. The 80GiB admission floor is intentionally above that
derived minimum. These are admission floors, not proof that a model will fit;
the one-step smoke run measures actual peak allocated and reserved HBM.

## Canonical local result

The full local command rehashed all 42 shards, both selection artifacts, and
all tokenizer files. It reproduced the exact model parameter count and both
planning-contract hashes. The report is outside Git at
`lumi-data/scaling-pilot-1b-v0.1/cloud-preflight-local-v0.1.json`:

| Field | Result |
| --- | ---: |
| Report bytes | 4,700 |
| Report SHA-256 | `d4974f7b8fee29455a42e3624c5b273178ede1cd67e3fab87d212bc8330ed0c0` |
| Artifact/model/contracts | pass |
| Shard/proof/state/tokenizer hashes | pass |
| Free-disk gate | pass |
| Local CUDA/BF16 gates | fail, expected |
| Local host-memory gate | fail: 16GiB Mac vs 32GiB floor |
| `runtime_ready` | false, expected on the Mac |
| Paid launch approved | false |

The same implementation also re-ran the archived schema-v1 150M/50M
preflight against its real sealed artifact and reproduced its original
150,625,536 parameters, 49,999,872 tokens, manifest hash, and build fingerprint.
Schema-v2 support therefore does not invalidate the completed Lambda pilot.

Run the canonical local check with no hash-skipping option:

```bash
python scripts/preflight_cloud_pilot.py \
  --plan configs/cloud-pilot-1b-v0.1.json \
  --manifest \
    "$LUMI_DATA/scaling-data-1b-v0.1/final-1.4b/manifest.json" \
  --tokenizer \
    "$LUMI_DATA/tokenizer-candidates-v0.2/lumi-tokenizer-65k" \
  --workspace "$LUMI_DATA" \
  --output \
    "$LUMI_DATA/scaling-pilot-1b-v0.1/cloud-preflight-local-v0.1.json"
```

## Remote gate

Provider, region, instance type, image, storage, availability, and current
hourly rate must be recorded immediately before launch; quota or credit alone
does not prove that an instance is launchable. Transfer only:

```text
~/lumi-model/
~/lumi-data/scaling-data-1b-v0.1/final-1.4b/
~/lumi-data/tokenizer-candidates-v0.2/lumi-tokenizer-65k/
~/lumi-runs/
```

After creating `~/lumi-runs`, run:

```bash
python scripts/preflight_cloud_pilot.py \
  --plan configs/cloud-pilot-1b-v0.1.json \
  --manifest \
    "$HOME/lumi-data/scaling-data-1b-v0.1/final-1.4b/manifest.json" \
  --tokenizer \
    "$HOME/lumi-data/tokenizer-candidates-v0.2/lumi-tokenizer-65k" \
  --workspace "$HOME/lumi-runs" \
  --require-cuda \
  --output "$HOME/lumi-runs/cloud-preflight-remote-v0.1.json"
```

Remote mode refuses `--skip-shard-hashes`. It must report `runtime_ready=true`
before any trainer process starts.

## Smoke/resume boundary

Only after provider selection, transfer verification, and remote preflight pass
may the first process run one optimizer step. Use the scaling plan's permanent
optimizer identity from the first step so every later stage can resume exactly:

```bash
python scripts/train_pilot.py \
  --manifest \
    "$HOME/lumi-data/scaling-data-1b-v0.1/final-1.4b/manifest.json" \
  --config configs/lumi-1b.json \
  --output "$HOME/lumi-runs/lumi-1b-scaling-v0.1" \
  --steps 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --gradient-checkpointing \
  --learning-rate 0.0002 \
  --lr-schedule cosine \
  --warmup-steps 500 \
  --schedule-steps 42725 \
  --min-learning-rate-ratio 0.1 \
  --weight-decay 0.1 \
  --max-grad-norm 1.0 \
  --checkpoint-steps 2500 \
  --keep-checkpoints 2 \
  --seed 7 \
  --device cuda
```

The fresh-process resume repeats the same command with `--steps 2 --resume`.
It passes only if loss and gradient norm are finite, step and data position
advance exactly, BF16 is active, CUDA memory metrics are present, and the
checkpoint restores. Stop after step 2. The 100-step calibration remains a
separate decision using measured throughput, peak HBM, checkpoint time/bytes,
startup overhead, and the current hourly rate.
