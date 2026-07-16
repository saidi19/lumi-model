# Lumi-1B Scaling Pilot v0.1

Status: **design, local input audit, and held-out-safe 1.4B data gate pass; no
paid GPU launch is approved**

## Decision

Lumi now has a machine-validated 1B scaling candidate with exactly
1,050,981,888 parameters. It is a controlled proxy for the 4B candidate: both
use 36 layers, the same 4,096-token context, tokenizer, head dimension, RoPE,
normalization, GQA pattern, tied embeddings, and bias-free projections. The 1B
candidate uses exactly half the 4B residual, MLP, query-head, and KV-head widths.

This freezes an architecture and staged experiment, not a quality claim. The
largest planned stage processes 1,400,012,800 tokens, only 1.332 tokens per
parameter. A simple 20-tokens-per-parameter Chinchilla reference for this model
would be about 21.02B tokens. The current experiment is therefore deliberately
undertrained: it can measure scaling, optimization, reliability, and early
quality movement, but it cannot establish ChatGPT-like quality or justify a
production model.

## Exact architecture

| Field | Lumi-1B | Lumi-4B reference | Relationship |
| --- | ---: | ---: | --- |
| Parameters | 1,050,981,888 | 4,002,376,704 | measured analytically |
| Layers | 36 | 36 | identical |
| Residual width | 1,536 | 3,072 | exact half |
| SwiGLU intermediate | 4,192 | 8,384 | exact half |
| Query heads | 32 | 64 | exact half |
| KV heads | 4 | 8 | exact half |
| Head dimension | 64 | 64 | identical |
| Query projection width | 2,048 | 4,096 | exact half |
| Vocabulary | 65,536 | 65,536 | identical |
| Context | 4,096 | 4,096 | identical |

The per-layer attention and MLP weights are exactly one quarter of the 4B
candidate because both input and output widths halve. The tied vocabulary table
is half-sized because only its hidden width changes. This relationship makes a
1B result materially more informative than an unrelated off-the-shelf model.

- Config: `configs/lumi-1b.json`
- Config SHA-256:
  `7fd53cab43b4cde02f8f09bc33e69f4aeb3594a60d7abf0def1bfc5f496829ce`
- Planning contract: `configs/scaling-pilot-1b-v0.1.json`
- Planning-contract SHA-256:
  `90f81e4d46c0677d62396cfa83d11afb91b40cfe0af9e35bd260e1b25967fa11`

## Pinned data evidence

The local audit verifies every supplied evidence file before producing a design
report:

| Evidence | Verified value |
| --- | ---: |
| Sealed source documents | 1,128,089 |
| Accepted source text | 6,000,636,724 bytes |
| Full-sample 65K-tokenizer count | 1,476,228,500 tokens |
| Source sample SHA-256 | `80bea9f9...1102a` |
| Permanent held-out documents | 3,988 |
| Permanent held-out tokens | 4,999,168 |
| Held-out build fingerprint | `77455ce7...6298` |
| 150M held-out baseline loss | 4.6808116141 |
| 150M held-out baseline perplexity | 107.8575758 |

The selected 1B target is 1,400,000,000 requested tokens, aligned to
1,399,996,416 unique packed tokens at 4,096 tokens per sequence. The required
selection mode is `within_source_hash_ranked_excluding_heldout`, with all 24
source strata preserved and every selected held-out ID, content hash, and
near-duplicate family excluded before token packing.

The dedicated checkpointed builder now implements that selection mode without
silently relabeling the original 6GB development/tokenizer sample. It sealed a
reusable 1,472,219,479-token eligible source index, a 16,777,216-token canary,
and the final 1,399,996,416-token artifact. Independent validation reports zero
held-out ID, content-hash, or near-duplicate-family overlap across all 24
strata. The final uint16 payload is 2,799,992,832 bytes in 42 shards; its
manifest SHA-256 is
`9228fa610d4a985d05e15057e8ef12619a14cbedbb5a6a5ab513d3aeca2832c8`.
The full evidence and reproduction commands are in
`docs/scaling-data-1b-v0.1.md`.

## Staged run contract

The planned geometry is one 4,096-token sequence per microbatch and eight-way
gradient accumulation: 32,768 tokens per optimizer step.

| Cumulative stage | Steps | Target packed tokens | Processed tokens | Purpose |
| --- | ---: | ---: | ---: | --- |
| Smoke/resume | 2 | 65,536 | 65,536 | fresh-process checkpoint recovery |
| Calibration | 100 | 3,276,800 | 3,276,800 | real throughput, memory, disk, cost |
| Learning gate | 7,630 | 249,999,360 | 250,019,840 | beat the frozen 150M held-out loss |
| Unique-corpus gate | 42,725 | 1,399,996,416 | 1,400,012,800 | one undertrained scaling pass |

The 20,480-token and 16,384-token overhangs at the last two stages are both
less than one optimizer step and come from deterministic sequence wrapping.
Each stage is a stop/go decision. Passing smoke does not authorize calibration;
passing calibration does not authorize 250M tokens; and the 1.4B stage starts
only after the 250M checkpoint improves on the frozen 150M held-out baseline.

## Trainer changes required by 1B

The correctness-first trainer now supports these optional, resume-pinned
features while preserving the old constant-LR defaults:

- Transformer gradient checkpointing, with a regression test proving matching
  loss and gradients on the tiny model;
- linear warm-up followed by a deterministic cosine schedule;
- a schedule horizon independent of the current stop point, so a 100-step run
  can later resume to 7,630 or 42,725 without changing earlier learning rates;
- learning-rate recording in every metric row;
- bounded checkpoint retention that deletes only older checkpoints from the
  same run after a newer checkpoint and top-level state are durable.

When all new options remain disabled, the original optimizer identity is kept
byte-compatible. A direct audit reproduces the archived 150M full-run
fingerprint `0ed66fb8...8d56`, so those checkpoints remain resumable.

The pinned 1B settings are BF16 autocast with FP32 parameters, gradient
checkpointing enabled, peak learning rate 2e-4, 500 warm-up steps, cosine decay
to 10%, weight decay 0.1, gradient clipping at 1.0, checkpoints every 2,500
steps plus every requested stop point, and retention of the newest two matching
checkpoints. Actual H100 peak memory remains unmeasured and is a mandatory
calibration output.

## Costing rule and scenarios

`scripts/plan_scaling_pilot.py` never invents 1B throughput or embeds a cloud
price. It costs cumulative milestones only after receiving both a measured
tokens/second value and the selected instance's hourly rate. The report also
records the explicit fixed overhead and optional credit balance.

For scale only, the table below uses the previously observed Lambda H100 rate
of USD 4.29/hour, 0.25 setup/teardown hours, and hypothetical—not measured—1B
throughputs:

| Hypothetical throughput | 1.4B compute time | Scenario cost |
| ---: | ---: | ---: |
| 5,000 tokens/s | 77.78 hours | USD 334.74 |
| 10,000 tokens/s | 38.89 hours | USD 167.91 |
| 20,000 tokens/s | 19.44 hours | USD 84.49 |

These are arithmetic scenarios, not a forecast. Checkpoint duration, provider
availability, storage, data transfer, idle time, and the actual 1B throughput
can change the bill. The canonical design report remains intentionally
unpriced until the 100-step calibration records real throughput.

## Fail-closed approval gates

The first four local gates are complete: the hash-ranked held-out-excluding
builder, byte-identical interruption/resume canary, 24-stratum capacity proof,
and sealed final manifests/shards/hashes all pass. Before any GPU launch:

1. extend cloud preflight to the new plan and exact packed artifact;
2. choose an actually available AWS or Lambda instance and enter its current
   hourly rate—quota alone does not prove capacity;
3. transfer the exact artifact and verify all hashes on the remote disk before
   starting a trainer process.

On the first GPU, run only the one-step smoke and a fresh-process resume. If
finite loss, gradient norm, checkpoint recovery, BF16, memory, and disk gates
pass, continue only to the 100-step calibration. Record median steady-state
tokens/second, peak allocated/reserved HBM, checkpoint seconds/bytes, startup
overhead, and actual rate; then rerun the cost planner before the 250M stage.

The 4B/80B run remains unapproved even if the 1.4B stage passes. That result may
lock a measured final-training plan, but the 4B run still needs an approximately
80B-token provenance-tracked corpus, distributed-training design, and a new
cost/capacity gate.

## Reproduction

Validate the architecture and unpriced plan:

```bash
python scripts/count_parameters.py configs/lumi-1b.json
python scripts/plan_scaling_pilot.py
```

Verify every current local input and create a new immutable report path:

```bash
python scripts/plan_scaling_pilot.py \
  --source-manifest "$LUMI_DATA/tokenizer-sample-v0.3/manifest.json" \
  --heldout-manifest "$LUMI_DATA/evaluation-pilot-v0.1/heldout/manifest.json" \
  --heldout-selection-ledger \
    "$LUMI_DATA/evaluation-pilot-v0.1/heldout/selection-ledger.jsonl" \
  --token-count-report \
    "$LUMI_DATA/tokenizer-candidates-v0.2/comparison-v0.2.json" \
  --baseline-evaluation-report \
    "$LUMI_DATA/evaluation-pilot-v0.1/checkpoint-150m-mps-v0.1/evaluation-report.json" \
  --output "$LUMI_DATA/scaling-pilot-1b-v0.1/design-report-next.json"
```

After calibration, cost with measured values:

```bash
python scripts/plan_scaling_pilot.py \
  --tokens-per-second "$MEASURED_TOKENS_PER_SECOND" \
  --hourly-rate "$SELECTED_GPU_HOURLY_RATE" \
  --fixed-overhead-hours "$MEASURED_OVERHEAD_HOURS" \
  --credit-balance "$CURRENT_SERVICE_CREDIT"
```

The canonical unpriced report is outside Git at
`lumi-data/scaling-pilot-1b-v0.1/design-report.json`, SHA-256
`de9aa950feb8ce17a9323198f1bace8c24689d32d440514d15b7dec243047941`.
