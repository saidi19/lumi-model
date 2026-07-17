# Lumi-1B Scaling Pilot v0.1

Status: **design, local data gates, Lambda A100 remote preflight, the two-step
smoke/resume gate, the cumulative 100-step calibration, and the cumulative
250M-token learning gate complete; the 1.4B-token and 4B stages remain
unapproved**

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

The separately authorized 250M-token gate has now passed its frozen held-out
criterion: loss fell from the 150M baseline of 4.6808116141 to 3.2655044363.
This confirms learning movement at the cumulative checkpoint, not useful chat
behavior. Fixed greedy continuations remain repetitive and factually wrong, as
expected for a 1.05B-parameter base model trained on only 250M tokens.

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

## Verified A100 smoke gate

A Lambda 1x A100 40GB SXM4 in `asia-south-1`, observed at USD 1.99/hour,
passed the exact-artifact remote CUDA/BF16/resource preflight before training.
Step 1 and a separate fresh-process resume to step 2 then produced:

| Step | Loss | Gradient norm | Tokens/s | Peak allocated | Peak reserved |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11.451070 | 9.422230 | 7,997.4 | 15.721GiB | 17.039GiB |
| 2 | 11.448407 | 10.372132 | 8,734.3 | 20.611GiB | 23.744GiB |

The resumed process reached sequence cursor 16 and 65,536 tokens with run
fingerprint
`66a9c146a0dd085df0de461cc4d30dfa43cebafd030d6933330b3ae3214ec1e9`.
This validates single-A100 fit and checkpoint recovery. Two cold steps do not
measure steady-state throughput, so they do not replace the 100-step
calibration or authorize any later stage. Full environment, preflight, and
checkpoint details are in `docs/cloud-pilot-1b-v0.1.md`. The latest checkpoint
was archived on the Mac with all 14 server-generated checksums verified before
the first Lambda instance was terminated.

## Verified A100 calibration gate

A separately authorized Lambda A100 run repeated the no-skip remote preflight,
verified the archived checkpoint-2 hashes, and restored the exact optimizer,
RNG, schedule, and sequence position. The fresh process then advanced from
step 2 to the cumulative step-100 stop:

| Field | Verified value |
| --- | ---: |
| Cumulative processed tokens | 3,276,800 |
| First resumed / final loss | 11.335843 / 7.690175 |
| Final gradient norm / learning rate | 2.202486 / 0.00004 |
| Median throughput, steps 10--99 | 9,379.6 tokens/s |
| End-to-end resume throughput | 8,495.4 tokens/s |
| Peak allocated / reserved HBM | 20.611GiB / 23.744GiB |
| Final checkpoint time / bytes | 17.160s / 12,612,139,802 |
| Training-segment wall time / rate cost | 378s / USD 0.20895 |

The final loss is the best observed loss in this short segment and the
run fingerprint remains
`66a9c146a0dd085df0de461cc4d30dfa43cebafd030d6933330b3ae3214ec1e9`.
Checkpoint 100 and its evidence were archived on the Mac; all 21
server-generated hashes passed before the second console shutdown was
confirmed. This gate validates runtime stability and supplies measured costing
inputs. It was not a held-out quality result and did not itself authorize the
later 250M-token run.

## Verified A100 250M learning gate

A separately authorized third Lambda A100 run verified the exact inputs and
checkpoint-100 archive, resumed the same optimizer/RNG/schedule/data position,
and stopped at the hard cumulative boundary of step 7,630. The frozen held-out
evaluation then ran before the result was sealed:

| Field | Verified value |
| --- | ---: |
| Model parameters | 1,050,981,888 |
| Resume / final cumulative step | 100 / 7,630 |
| Cumulative processed tokens | 250,019,840 |
| First resumed / final / best segment loss | 7.815707 / 2.776807 / 2.249702 |
| Frozen 150M baseline / 1B held-out loss | 4.680812 / 3.265504 |
| Held-out loss delta / perplexity | -1.415307 / 26.193321 |
| Quality-gate result | PASS |
| Median steady-state / end-to-end throughput | 9,382.6 / 9,263.9 tokens/s |
| Training / evaluation duration | 26,635s / 1,230s |
| Peak HBM / median GPU utilization / max temperature | 24,841MiB / 96% / 67C |
| Final checkpoint time / bytes | 16.681s / 12,612,139,807 |
| Training-segment cost at USD 1.99/hour | USD 14.723236 |

The held-out loss is strictly below the frozen baseline, so the specified
learning criterion passes. This proves substantial next-token-learning movement
at 250M tokens. It is not a chatbot-quality pass: all four fixed greedy probes
remain repetitive or factually wrong, including the France-capital,
photosynthesis, Python, and Bengali probes.

Milestone checkpoints 2,500, 5,000, and 7,500 and final
`checkpoint-00007630` are archived outside Git on the Mac. The final checkpoint
contains 12,612,139,807 bytes, including the 4,203,973,544-byte model and
8,408,144,175-byte trainer state. All 35 entries in the server-generated final
manifest passed an independent local SHA-256 verification. The manifest itself
has SHA-256
`4109929835ca18d39fea868032ea851e8f5ec71b5cebf9e4119b979741d007a1`.
The Lambda console was then verified to show no running instances.

This run deliberately preserves the current separator-only fixed packing and
ordinary causal attention, under which adjacent packed documents can
cross-attend. A later separate controlled experiment verified the
document-isolated varlen/`cu_seqlens` implementation on A100 and measured 6.17%
higher steady throughput with identical peak reserved HBM; see
`docs/packed-attention-ab-v0.1.md`. It did not alter this checkpoint. The PASS
makes a 1.4B-token stage eligible for consideration, but an isolated-policy run
must start fresh at step zero. Neither that run nor the 4B/80B run is authorized
by this result.

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
checkpoints. The two-step A100 gate measured 23.744GiB peak reserved memory;
the 100-step gate then reproduced that 23.744GiB peak while measuring steady
throughput and a 17.160-second full checkpoint.

## Costing rule and scenarios

`scripts/plan_scaling_pilot.py` never invents 1B throughput or embeds a cloud
price. It costs cumulative milestones only after receiving both a measured
tokens/second value and the selected instance's hourly rate. The report also
records the explicit fixed overhead and optional credit balance.

The measured report uses the calibration's 9,379.612 tokens/second median and
the selected Lambda A100 rate of USD 1.99/hour. Boot to confirmed shutdown was
44 minutes 23 seconds. Subtracting the steady-state calibration compute
estimate yields a conservative 0.642680-hour fixed allowance for setup,
transfer, checkpointing, archive, and teardown:

| Cumulative stage | Compute hours | Raw compute cost | With fixed allowance |
| --- | ---: | ---: | ---: |
| Calibration, 3,276,800 tokens | 0.10 | USD 0.19 | USD 1.47 |
| Learning gate, 250,019,840 tokens | 7.40 | USD 14.73 | USD 16.01 |
| Unique corpus, 1,400,012,800 tokens | 41.46 | USD 82.51 | USD 83.79 |

The measured compute-only rate is USD 58.93 per billion processed tokens. The
resumed step-2-to-100 training segment itself cost USD 0.20895 at the observed
rate. The fixed allowance deliberately treats the entire boot-to-shutdown wall
interval as billable; it is a safe planning estimate, not an invoice value,
because Lambda states that booting time is not billed. Provider availability,
storage, data transfer, retries, and idle time can still change the bill.

## Fail-closed approval gates

The local gates are complete: the hash-ranked held-out-excluding builder,
byte-identical interruption/resume canary, 24-stratum capacity proof, sealed
final manifests/shards/hashes, and schema-v2 exact-artifact cloud preflight all
pass. The first Lambda launch then followed the required sequence:

1. choose an actually available AWS or Lambda instance and enter its current
   hourly rate—quota alone does not prove capacity;
2. transfer only the exact artifact, tokenizer, and repository;
3. rerun the preflight remotely with mandatory CUDA/BF16, HBM, host-memory,
   free-disk, proof/state, tokenizer, and all-shard hash gates before starting a
   trainer process.

The first GPU stopped after the one-step smoke and fresh-process resume passed.
A separately approved second launch stopped at cumulative step 100 after
recording median steady-state tokens/second, peak allocated/reserved HBM,
checkpoint seconds/bytes, lifecycle overhead, and the selected rate. The
archive and measured cost report were verified before shutdown. Continuing to
the 250M learning gate was a separate owner decision. That run is now complete
and passed the frozen held-out-loss rule, but it does not itself authorize the
1.4B-token stage.

The 4B/80B run remains unapproved even if the 1.4B stage passes. That result may
lock a measured final-training plan, but the 4B run still needs an approximately
80B-token provenance-tracked corpus, distributed-training design, and a new
cost/capacity gate.

The fresh isolated-policy orchestration is now prepared separately in
`docs/isolated-scaling-run-1b-v0.1.md`. It starts at step zero, treats 250M as a
milestone inside the 1.4B run, evaluates with the same document-isolation
semantics, and requires an out-of-repository owner authorization bound to the
exact merged source commit before any paid command can run.

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

Reproduce the calibration cost arithmetic to a new immutable report path:

```bash
python scripts/plan_scaling_pilot.py \
  --tokens-per-second 9379.612129584133 \
  --hourly-rate 1.99 \
  --fixed-overhead-hours 0.642679592991180 \
  --output \
    "$LUMI_DATA/scaling-pilot-1b-v0.1/a100-calibration-v0.1/measured-cost-projection-next.json"
```

The canonical unpriced report is outside Git at
`lumi-data/scaling-pilot-1b-v0.1/design-report.json`, SHA-256
`de9aa950feb8ce17a9323198f1bace8c24689d32d440514d15b7dec243047941`.
The canonical measured report is beside the calibration archive at
`measured-cost-projection-v0.1.json`, SHA-256
`0cf7ad8069b03a1269b05b852b9b8881dacfe9562e020089996af4cbc95d7f2d`.
