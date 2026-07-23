# Lumi-4B / 80B-Token Production Plan v0.1

Status: **planning-only. The exact production schedule, corpus binding,
checkpoint cadence, conservative cost model, and three checkpoint-aligned
operator segments are implemented and locally verified. No paid GPU,
production authorization, or full 4B/80B training was started by this work.**

## Why this is the next step

The bounded 8xH100 pilot passed at step 80 and established that the real
4,002,376,704-parameter model, BF16 full-shard FSDP, marker-aware
document-isolated FlashAttention, checkpoint/resume, and frozen held-out
evaluation work together. The pilot used a cosine schedule whose entire
horizon was only 80 steps, so its checkpoint is evidence, not a valid seed for
the full schedule.

Production therefore starts fresh with the same proven model, optimizer,
topology, corpus, and seed, but with a new 76,294-step learning-rate horizon.
Silently resuming the pilot checkpoint would inherit the exhausted 80-step
schedule and is explicitly rejected.

## Immutable input and model

The planning contract is `configs/4b-80b-production-run-v0.1.json`. It pins:

- marker-aware final-corpus manifest SHA-256
  `6de45464af33d44e1261a023ccb7d50f86346a4126bb2fa060530a91213483e9`;
- corpus plan
  `aab292a514b2a7a046a3e4f0daf2fb75bb6f899ff86798bdfac729a145bc9470`;
- selection result
  `b017a6fec5c6ca1685c5dcd883d8afc48d36fd18cac07d573514ad77307253ee`;
- 45,768,283 documents and document markers;
- 160 immutable uint16 shards and 160,000,016,728 packed bytes;
- 80,000,008,364 available packed tokens, with an exact training boundary of
  80,000,000,000 tokens and 8,364 unused trailing tokens;
- model config SHA-256
  `643e8da652577f802d959740d66a7e6bafc5b065aa45010c2d7e0340ed9509be`;
  and
- verified pilot report SHA-256
  `d7c4ce9db462219bb4feab4dee43f46fe158f99190eb6d3b35d77216519d5130`.

The production planner validates those bindings against the actual sealed
manifest and pilot report before it creates a run fingerprint.

## Exact training geometry

The selected topology remains the measured 8x H100 80GB configuration:

| Setting | Value |
| --- | ---: |
| World size | 8 |
| Per-rank batch | 1 sequence |
| Gradient accumulation | 32 |
| Global batch | 256 sequences |
| Sequence length | 4,096 |
| Nominal tokens per step | 1,048,576 |
| Exact training sequences | 19,531,250 |
| Exact optimizer steps | 76,294 |
| Full steps | 76,293 |
| Final-step real sequences | 242 |
| Final-step tokens | 991,232 |

The final step uses the trainer's already-tested exact-tail path and cannot
wrap into a second corpus epoch.

Fixed optimization settings are:

- fresh residual-scaled initialization;
- AdamW legacy profile, learning rate `2e-4`, weight decay `0.1`;
- BF16, full-shard FSDP, gradient checkpointing;
- FlashAttention 2.8.3 with document isolation;
- last-microbatch gradient synchronization;
- cosine decay across all 76,294 steps;
- 1,000 warmup steps and minimum learning-rate ratio `0.1`;
- seed 7; and
- no `torch.compile`.

The 1,000-step warmup is about 1.31% of the full horizon, close to the
previously tested 1B schedule's warmup fraction, while keeping a simple exact
boundary. It is not copied from the intentionally short pilot schedule.

## Checkpoints, evaluation, and resume

A complete distributed checkpoint is written every 250 steps and at the exact
final step. At the conservative throughput this is roughly every 1.27 training
hours. The newest two local checkpoints are retained; any checkpoint exported
for handoff is separately sealed by hashing every file and binding it to the
source commit, run fingerprint, world size, optimizer state, step, sequence
cursor, and token cursor.

Frozen held-out evaluations are planned at steps 1,000, 5,000, 10,000, 20,000,
40,000, 60,000, and 76,294. Intermediate milestones coincide with durable
checkpoint boundaries. They are learning/health measurements, not public
frontier-model benchmarks.

Each paid segment authorization includes both a start and stop cursor. The
trainer rejects a segment if the imported checkpoint is earlier than its
authorized start, later than its stop, has the wrong token cursor, uses another
run fingerprint, or changes topology/source/schedule. A crash may resume from
any matching checkpoint inside the same authorized segment.

## Measured runtime and conservative budget

The pilot's verified median throughput was 71,935.4315 tokens/second. At the
measured `$31.92` hourly rate:

| Estimate | Training hours | Compute cost |
| --- | ---: | ---: |
| Measured-rate projection | 308.92 h / 12.87 days | $9,860.69 |
| 0.8 throughput safety factor | 386.15 h / 16.09 days | $12,325.87 |

These are training-only estimates. Staging, evaluations, checkpoint transfer,
provider startup, and teardown use the separate runtime and credit reserves.
Actual cost must be read from the provider before each authorization.

The conservative plan leaves eight hours of a 168-hour lifecycle unused and
keeps `$500` of each `$7,500` credit allocation in reserve:

| Segment | Step range | Cumulative token range | Conservative training | Projected compute |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 → 31,500 | 0 → 33,030,144,000 | 159.43 h | $5,089.07 |
| 2 | 31,500 → 63,000 | 33,030,144,000 → 66,060,288,000 | 159.43 h | $5,089.07 |
| 3 | 63,000 → 76,294 | 66,060,288,000 → 80,000,000,000 | 67.29 h | $2,147.74 |

These are operator segments, not permission to rotate accounts or bypass a
provider's terms. Every account, credit grant, and instance must be valid for
the same owner and workload under the provider agreement. The checkpoint
format merely makes an authorized interruption portable.

## Authorization boundary

Repository configuration records:

```text
full_4b_training=false
paid_gpu=false
```

Planning creates one hash-bound template per segment, but every template also
records `paid_gpu=false` and `status=planning_only`. Only a new, explicit owner
decision can create a separate artifact with `paid_gpu=true` and
`full_4b_training=true`.

The distributed trainer accepts that production run kind only when all of the
following match:

- exact source commit and full run fingerprint;
- 8 GPUs and full-shard FSDP;
- exact cumulative start/stop step and token cursors;
- 80B total hard cap and step 76,294 absolute hard cap;
- checkpoint-aligned non-final boundaries;
- per-segment runtime no greater than 168 hours;
- per-segment cost cap no greater than `$7,000`; and
- exact owner acknowledgement, plan hash, and contract hash.

The production plan must be regenerated after this code is merged so its run
fingerprint binds the final merge commit. Until that plan is sealed and the
owner separately approves segment 1, there is no authority to launch an
instance or begin full training.
