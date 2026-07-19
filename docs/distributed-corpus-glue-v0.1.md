# Lumi distributed corpus acquisition on AWS Glue v0.1

Status: **local implementation and Spark parity gate pass; paid Glue canary and
distributed production acquisition are not authorized**

## Why this migration exists

This is not a new roadmap stage and it does not change the 4B/80B plan. It
replaces the serialized execution backend inside Step 11, final-corpus
acquisition. The first owner-authorized EC2 worker correctly exercised the
production filters and S3 publication path, but global filtering,
near-deduplication, tokenization, and shard writing were effectively serialized.
That execution shape was too slow for a 120B candidate-token floor, so the
worker was stopped before spending more money.

The replacement keeps all existing corpus decisions:

- the final training target remains 80B tokens;
- seven category floors still total 120B accepted candidate tokens;
- 150B remains a fail-closed hard maximum, not a required download;
- the logical object-storage hard cap remains 2TB;
- source revisions, quality filters, PII/secret removal, benchmark exclusion,
  tokenization, provenance, and source-order deduplication remain pinned;
- no model, GPU, 1B, or 4B training is part of this migration.

## Preserved production prefix

The stopped worker was not discarded. Its manual-stop archive, internal
manifest, SQLite database, two sealed S3 shards, and partial current shard were
independently verified.

| Evidence | Verified value |
|---|---:|
| Accepted prefix | 237,769,640 tokens / 38,349 documents |
| Sealed shards | 200,614,708 tokens / 32,393 documents |
| Partial current shard | 37,154,932 tokens / 5,956 documents |
| Resume point | source ordinal 0, next line 38,849 |
| Resume-anchor rows | 38,349 |
| Resume-anchor bytes | 24,785,510 |
| Resume-anchor SHA-256 | `530ac0e1b6773cc2182aba6bd2dd923e4f0362bae3289b312989ab06cebc1ee1` |
| Canonical anchor rows SHA-256 | `41d14457dd9f01973dd17952616451282f570e141b18fdc8b0d68cfebe7b2c51` |

All 38,349 accepted documents become immutable dedup anchors. Distributed work
starts after the recorded source line and cannot re-accept an exact or near
duplicate of that prefix. The old EC2 instance is terminated and both attached
volumes are deleted. The two sealed S3 shards remain content-addressed
production evidence.

## Distributed semantics

Each pinned gzip source object is an independent Spark task. Partition workers
perform the existing record mapping: segmentation, quality and language gates,
PII/secret handling, benchmark exclusion, MinHash construction, frozen-tokenizer
encoding, and provenance capture. They emit candidates; they do not decide
global duplicates independently.

Global exact-content and MinHash/Jaccard conflicts form a graph. Lumi uses a
fixed source-order priority and a parallel greedy maximal-independent-set
algorithm. In each round, candidates with no earlier undecided conflict are
accepted; their conflicting later candidates are rejected. This is equivalent
to the previous sequential source-order greedy decision, including
non-transitive chains. Rejection reasons are derived only from earlier accepted
neighbors, preserving the prior exact-versus-near semantics.

The Spark path fails closed if any of these safety bounds is exceeded:

- 10,000 documents in one LSH bucket;
- 50,000,000 candidate conflict pairs;
- 256 ordered-dedup rounds;
- a production checkpoint location that is not a private `s3://` URI.

## Local verification

The following gates pass without AWS compute:

- deterministic MinHash serialization and resume-anchor export;
- exact and near-duplicate chain parity against the sequential reference;
- 100 randomized conflict-graph parity cases;
- immutable-prefix resume behavior and rejection-reason parity;
- real Spark 3.5.6 execution with repeated-run determinism;
- fail-closed oversized-bucket and non-S3-checkpoint tests;
- package hash, release fingerprint, source-commit, and authorization tests;
- repository lint and the complete 221-test suite.

The local package is intentionally non-authorizing. A package and a paid owner
authorization are separate artifacts, so merely building or uploading code
cannot start compute.

## Bounded paid canary

The next decision is only a parity canary, not production acquisition.

| Bound | Hard maximum |
|---|---:|
| Glue runtime | Glue 5.1 / Spark 3.5.6 / Python 3.11 |
| Worker type | `G.1X` |
| Workers / DPU | 8 / 8 |
| Source objects | 8 |
| Records per source | 2,000 |
| Category coverage | all 7 categories |
| Compressed bytes per source | 700,000,000 |
| Total compressed source input | 1,500,000,000 |
| Runtime | 30 minutes |
| Concurrency | 1 |
| Price | $0.44 per DPU-hour |
| Maximum compute cost | $1.76 |

The canary may write only beneath the distributed canary control prefix. It
cannot publish production shards. It compares Spark accepted IDs and rejection
reasons with an independent sequential evaluation of the same bounded records.
Any mismatch fails the gate.

The canary requires a new, exact, release-bound owner authorization. Current
configuration records `paid_glue_canary=false`, `paid_glue_production=false`,
`paid_gpu_compute=false`, `model_training=false`, and
`full_4b_training=false`.

## Production boundary after the canary

A passing canary provides throughput, shuffle size, dedup-round count, and cost
measurements. Only then can a separate production proposal be calculated. The
current planning envelope permits an initial ceiling of 64 DPU ($28.16/hour)
and fails closed above 128 DPU ($56.32/hour), but neither value authorizes a
job. A new owner decision must set an explicit runtime and dollar cap before
distributed production resumes.

## Reproduction commands

```bash
PYTHONPATH=src python3 scripts/plan_distributed_corpus.py \
  --output "$LUMI_DATA/distributed-corpus-v0.1/plan.json"

PYTHONPATH=src python3 scripts/export_distributed_resume_anchors.py \
  --evidence-directory "$MANUAL_STOP_EVIDENCE" \
  --output-directory "$LUMI_DATA/distributed-corpus-v0.1/resume-anchors"

PYTHONPATH=src python3 scripts/package_glue_distributed_canary.py \
  --anchor-export-directory "$LUMI_DATA/distributed-corpus-v0.1/resume-anchors" \
  --output-directory "$LUMI_DATA/distributed-corpus-v0.1/glue-canary-package"
```

The package builder binds the release to the current full Git commit, verifies
all packaged hashes, and reports `verified_not_authorized`. Launch tooling must
still validate a separate unexpired owner authorization against that exact
release fingerprint.
