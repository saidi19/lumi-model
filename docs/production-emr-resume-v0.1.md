# Lumi 120B production-corpus resume on EMR Serverless v0.1

Status: **local control plane verified; AWS quota and a fresh exact paid authorization
are still required; no production job or model training is running**

## Purpose

This gate resumes from the independently verified prefix of 237,769,640 accepted
tokens / 38,349 documents and builds the seven category floors totaling 120B clean
candidate tokens. The later 4B pretraining target remains 80B tokens. The 150B
number is only a fail-closed raw-candidate ceiling; it is not a required download or
training target.

The production path keeps the already verified network boundary:

1. one internet-facing, CPU-only stager downloads a pinned Common Pile object,
   verifies its byte count and SHA-256, and splits it once into deterministic zstd
   chunks;
2. chunks are conditionally published under immutable private-S3 keys;
3. EMR Serverless receives only same-Region S3 inputs and never contacts Hugging
   Face;
4. the raw chunk versions are deleted only after the matching map result has been
   committed to an immutable checkpoint.

This is required because the bounded cloud canary already proved that default EMR
Serverless workers cannot reach `huggingface.co:443` while the S3-only path has exact
Mac/cloud parity.

## Two-pass execution shape

| Phase | Maximum live shape | Durable output |
|---|---:|---|
| External source staging | 1 × r7i.4xlarge, 16 vCPU, 100GB gp3 | content-addressed raw S3 chunks + manifest |
| Candidate mapping | 7 category lanes × 16 vCPU = 112 vCPU | checkpoint-bound Parquet candidates |
| Global dedup and category prefix | 1 × 128 vCPU | category-partitioned accepted candidates + rejected IDs |
| Final shard writers | 7 categories × 16 vCPU = 112 vCPU | immutable approximately 100M-token shards |
| Local final seal | no paid compute | canonical corpus JSON + `manifest.sha256` |

Only one map reservation can be active per category. Jobs may finish out of order:
each result is bound to its category cursor and reservation token, so a faster lane
cannot invalidate another lane's checkpoint. The controller persists every launch,
AWS job ID, billed utilization, result key, map checkpoint, and phase transition.
There is no automatic retry after an EMR failure.

The first map target is 1.20 times each category's remaining floor, totaling
143,714,676,432 mapped tokens. Global exact/near dedup then selects the canonical
source-order prefix needed for each floor. If a category remains short, the adaptive
gate increases only that category by 1.20 times its measured deficit and resumes at
its exact source line. It refuses the continuation before the 150B raw-token cap.

Accepted output is partitioned by category before the seven shard writers run.
Within each category, deterministic range-partition prefix sums assign 100M-token
shard IDs without an O(n²) mask or a global single-partition Spark window. The
preserved 37,154,932-token science partial is sealed once as its own immutable shard;
all later category shards are written in parallel.

## Hard boundaries

- candidate floors: 120,000,000,000 tokens across seven pinned categories;
- raw candidate maximum: 150,000,000,000 tokens;
- final pretraining sample target: 80,000,000,000 tokens;
- clean shard target: 100,000,000 tokens, with at most one bounded document
  overshoot;
- logical storage maximum: 2,000,000,000,000 bytes;
- EMR ceiling: 128 vCPU;
- equivalent 128-vCPU runtime ceiling: 96 hours;
- no GPU, 1B, 4B, or model-training permission in this control plane.

The final seal rejects missing categories, token/document non-conservation, duplicate
manifests or S3 object keys, changed dedup/job identities, bad token byte counts,
non-contiguous shard sequences, a floor shortfall, or either the 150B/2TB cap.
Every new shard object and publication manifest is independently checked against S3
before the final seal is published.

## Recorded budget

| Allocation | Hard maximum |
|---|---:|
| EMR Serverless compute | $1,200 |
| CPU stager compute/storage | $200 |
| S3 requests/logs | $100 |
| Optional verified archive migration | $250 |
| Contingency | $250 |
| Total | **$2,000** |

The pinned x86 EMR rate is $9.395680/hour at the full 128-vCPU shape. Ninety-six
equivalent hours would be about $901.985280, below the $1,200 EMR allocation. The
stager's 96-hour compute maximum is $101.606400 and its 100GB gp3 maximum is
$1.052055. Actual controller accounting uses AWS's final billed vCPU-, memory-, and
storage-hours, while active jobs reserve their full timeout before another launch is
allowed.

## Verification

The non-authorizing dry run binds all 758 pinned source files totaling
499,817,285,384 compressed bytes, the verified resume checkpoint, tokenizer,
exclusion ledger, quality policy, source manifest, implementation files, pricing,
resource shapes, and budget. Local verification currently passes:

- 278 repository tests, including real localhost Spark and local API tests;
- distributed prefix and shard assignment parity;
- reservation/out-of-order checkpoint tests;
- adaptive deficit and 150B refusal tests;
- final 120B seal conservation tests;
- Ruff lint and Python bytecode compilation.

## Launch boundary

The repository config deliberately records both `paid_cpu_stager` and
`paid_emr_production` as false. Building the local plan does not grant permission.
Before any launch, the live EMR quota must be rechecked and the owner must separately
provide the exact authorization text recorded in the code. No 4B/80B training is
started by that authorization; production acquisition must first finish, verify, and
seal.

After a successful seal, the next roadmap step is to derive and freeze the final 80B
training mixture, then run a short 4B multi-GPU throughput/checkpoint pilot. The full
4B/80B run remains a later, separately priced approval.
