# Lumi distributed corpus acquisition on AWS Glue v0.1

Status: **the owner-authorized bounded Glue canary failed safe before data
processing because Glue 5.1 is fixed to Python 3.11; all temporary compute and
IAM resources are deleted and distributed production remains unauthorized**

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

## Bounded paid canary result

The exact bounded canary was owner-authorized and launched once from source
commit `08cee80f2d6a335302d3bb0a748f8cb284e483e3`. Its envelope was:

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

AWS Glue 5.1 started the dependency-bootstrap phase, but its fixed Python 3.11
runtime could not install the pinned `lingua-language-detector==2.2.0` wheel,
which requires Python 3.12 or newer. The job therefore failed before reading a
source record or running Lumi's mapping/deduplication code.

| Result | Verified value |
|---|---:|
| Job state | `failed_safe_before_data_processing` |
| DPU seconds | 306 |
| Execution time | 38 seconds |
| Estimated Glue compute | $0.0374 |
| Canary objects written | 0 |
| Production shards written | false |
| Model training started | false |

The Glue job and its IAM role now return `EntityNotFound`/`NoSuchEntity`, the
canary output prefix is empty, and no EMR Serverless application or CodeBuild
build exists. Existing immutable corpus-prefix objects were retained.

## Decision after the canary

This result does not reject the Spark design; it rejects Glue 5.1 as Lumi's
runtime because silently downgrading Lingua would change already-sealed language
filter decisions. The same bounded parity gate is being moved to EMR Serverless
7.13.0 with a digest-pinned Python 3.12 custom image. See
[`distributed-corpus-emr-serverless-v0.1.md`](distributed-corpus-emr-serverless-v0.1.md).

The old Glue authorization is consumed and cannot authorize EMR, CodeBuild, or
production acquisition. No production resume is permitted until the replacement
canary passes and a separate production proposal receives explicit approval.

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

The archived Glue package remains reproducible evidence of the attempted gate;
it must not be relaunched. The replacement keeps the same source bounds,
ordered-dedup semantics, immutable resume anchors, and no-production-output
policy while changing only the managed runtime.
