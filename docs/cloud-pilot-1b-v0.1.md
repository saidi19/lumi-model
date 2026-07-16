# Lumi-1B Cloud Preflight v0.1

Status: **canonical local and Lambda A100 remote preflights, the two-step
smoke/resume gate, cumulative 100-step calibration, local checkpoint
verification, and both instance shutdowns pass; the 250M-token and later
stages remain unapproved**

## Decision

The sealed 1.4B-token artifact was transferred to a selected Lambda A100 host.
The same no-skip command passed there with `--require-cuda` before any trainer
process started. One optimizer step and a separate fresh-process resume then
advanced the run through step 2 with finite metrics and exact checkpoint/data
recovery. Checkpoint 2 and its reports were copied back to the Mac, verified
against the server-generated checksums, and the instance was terminated.

A separately authorized second A100 host repeated the exact remote preflight,
verified the archived checkpoint-2 hashes, and resumed that run to cumulative
step 100. Checkpoint 100 and all calibration evidence were copied back, all 21
server-generated hashes passed on the Mac, and the second instance was
terminated only after that verification.

This completes the 100-step calibration boundary. The schema-v2 gate still
extends the existing 150M preflight without changing its schema-v1 contract,
and the technical PASS does not authorize the 250M-token, 1.4B-token, or 4B
stages.

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

The false approval field is intentional: a technical preflight never grants
spending authority. The later Lambda launch was authorized separately by the
owner immediately after live instance type, region, and rate selection.

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

## Verified smoke-host Lambda A100 admission

The live target was selected immediately before launch:

- provider: Lambda Cloud;
- instance: 1x A100 40GB SXM4 (`gpu_1x_a100_sxm4`);
- region: India (`asia-south-1`);
- image: Lambda Stack 22.04;
- observed rate: USD 1.99/hour;
- persistent filesystem: none;
- SSH key: the existing `Afnan` public key; its private key remained off-server.

The host booted at 2026-07-16 18:14:21 UTC. The runtime audit and remote report
recorded:

| Field | Verified value |
| --- | ---: |
| GPU | NVIDIA A100-SXM4-40GB |
| CUDA device memory | 40,960MiB |
| Compute capability | 8.0 |
| Driver | 580.105.08 |
| Host memory | 232,205,586,432 bytes |
| Initial free disk | 506,388,217,856 bytes |
| Python | 3.12.13 |
| PyTorch / CUDA | 2.7.0+cu126 / 12.6 |
| CUDA / BF16 | true / true |
| Remote report SHA-256 | `0a48500e18936334c756387968d87b4e2d83b18328f000802b2af0c28c7253b2` |
| `runtime_ready` | true |

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

## Verified smoke/resume result

The first process completed one step and wrote checkpoint 1. A separate Python
process restored model, optimizer, CPU/CUDA RNG, and sequence cursor before
advancing to step 2 and checkpoint 2:

| Step | Loss | Gradient norm | Tokens/s | Peak allocated | Peak reserved | Checkpoint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11.451070 | 9.422230 | 7,997.4 | 15.721GiB | 17.039GiB | 18.228s |
| 2, fresh resume | 11.448407 | 10.372132 | 8,734.3 | 20.611GiB | 23.744GiB | 16.487s |

The final state is `complete` at step 2, sequence cursor 16, and 65,536 tokens.
Its run fingerprint is
`66a9c146a0dd085df0de461cc4d30dfa43cebafd030d6933330b3ae3214ec1e9`.
Checkpoint 2 contains a 4,203,973,544-byte model file and an
8,408,144,175-byte optimizer/RNG file. These two cold smoke measurements are
not a steady-state throughput estimate and were not used as the measured cost
input; the separately authorized calibration supplied that measurement.

## Verified smoke archive and shutdown

The reports and latest resumable checkpoint are outside Git at
`lumi-data/scaling-pilot-1b-v0.1/a100-smoke-v0.1/`. The archive occupies about
12GB and contains 14 payload files plus the checksum manifest. All 14 entries
in the 1,540-byte server-generated manifest passed local SHA-256 verification.
The manifest itself has SHA-256
`f69bfa9905288c4178d04a1668fa9244b8ac053056488b7bd8eef156018040e4`.

After verification, Lambda Cloud reported **No running instances** at
2026-07-16 19:08:54 UTC. No calibration or later stage ran on that first host.

## Verified 100-step calibration

The second launch used the same `gpu_1x_a100_sxm4` type, `asia-south-1`
region, Lambda Stack 22.04 image, and observed USD 1.99/hour rate. It booted at
2026-07-16 21:46:05 UTC. Before training, the full no-skip remote preflight
passed again with 1,050,981,888 parameters, 1,399,996,416 packed tokens,
CUDA/BF16 enabled, 232,205,602,816 bytes of host memory, and 506,572,943,360
bytes of initial free disk. The report has SHA-256
`96b313c7c8796ee57397533bbbf88f6cb526cf32b7a69b17f2cc4314d71cd2ac`.
All nine checkpoint-2 checksum entries also passed remotely before resume.

The fresh trainer process restored step 2, sequence cursor 16, 65,536 tokens,
optimizer state, and the unchanged run fingerprint
`66a9c146a0dd085df0de461cc4d30dfa43cebafd030d6933330b3ae3214ec1e9`.
It ran from 2026-07-16 22:00:26 UTC to 22:06:45 UTC and stopped at the authorized
cumulative step 100 boundary:

| Field | Verified value |
| --- | ---: |
| Cumulative tokens / sequence cursor | 3,276,800 / 800 |
| First resumed loss / final loss | 11.335843 / 7.690175 |
| Final gradient norm / learning rate | 2.202486 / 0.00004 |
| Median tokens/s, steps 10--99 | 9,379.6 |
| Steady-state min / max tokens/s | 9,097.4 / 9,449.0 |
| End-to-end resume tokens/s | 8,495.4 |
| Peak allocated / reserved HBM | 20.611GiB / 23.744GiB |
| Final checkpoint time | 17.160s |
| Training-segment wall time / rate cost | 378s / USD 0.20895 |
| Median / maximum GPU utilization | 89.5% / 100% |
| Maximum memory / temperature | 24,841MiB / 65C |

Checkpoint 100 contains a 4,203,973,544-byte model and an
8,408,144,175-byte optimizer/RNG file; its seven files total 12,612,139,802
bytes. The loss reduction is an optimization-health result, not a general
quality claim: only 0.003 tokens per parameter have been processed at this
boundary.

## Calibration archive, cost report, and shutdown

The calibration archive is outside Git at
`lumi-data/scaling-pilot-1b-v0.1/a100-calibration-v0.1/` and occupies about
12GB. Every one of the 21 entries in the 2,407-byte server-generated checksum
manifest passed locally. The manifest itself has SHA-256
`ea986c336a134fb0dde444a331cff95c0debe2a4b7aa54b3f6efa7cc8dec24ab`.

After verification, the console reported **No running instances** at
2026-07-16 22:30:28 UTC. Boot to confirmed shutdown was 44 minutes 23 seconds.
At the observed rate that is a conservative USD 1.472 wall-lifecycle estimate;
it is not an invoice value because Lambda states that booting time is not
billed. Subtracting the steady-state calibration compute estimate leaves a
conservative 0.642680-hour setup, transfer, checkpoint, archive, and teardown
allowance for planning.

The measured cost report is outside Git beside the archive at
`measured-cost-projection-v0.1.json`. It is 2,711 bytes with SHA-256
`0cf7ad8069b03a1269b05b852b9b8881dacfe9562e020089996af4cbc95d7f2d`.
At 9,379.6 tokens/second and USD 1.99/hour, it projects the cumulative
250,019,840-token gate at 7.40 compute-hours and USD 16.01 including that
allowance, and the 1,400,012,800-token gate at 41.46 compute-hours and USD
83.79. These projections do not authorize either stage. No 250M-token or later
training ran.
