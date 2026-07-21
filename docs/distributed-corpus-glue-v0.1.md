# Lumi distributed corpus acquisition on AWS Glue v0.1

Status: **the source-built Python 3.11 Lingua wheel and one corrected 8-DPU
Glue canary pass end to end; Spark output matches the sequential reference,
all temporary compute/IAM/job resources are deleted, and distributed production
remains unauthorized**

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

## Decision after the first canary

At that point the result did not reject the Spark design; it rejected unmodified
Glue 5.1 as Lumi's runtime because silently downgrading Lingua would change
already-sealed language-filter decisions. The same bounded parity gate was
also adapted to EMR Serverless 7.13.0 with a digest-pinned Python 3.12 custom
image. See
[`distributed-corpus-emr-serverless-v0.1.md`](distributed-corpus-emr-serverless-v0.1.md).

The old Glue authorization remains consumed and cannot authorize EMR,
CodeBuild, or production acquisition. The later CPython 3.11 compatibility
work and corrected canary below supersede only that historical runtime
rejection.

## Python 3.11 compatibility follow-up

A later exact owner authorization allowed one Linux CPython 3.11 CodeBuild and
one follow-up 8-DPU, 30-minute Glue canary, with a combined $3 maximum and no
production run. The build used Lumi commit
`624f253fe0af4f042c6d0b76cf1026443007a9bb`, upstream Lingua commit
`77e94d1a2a8e49bed873a7c502bcb51ba803ef0f`, Rust 1.94.0, Maturin 1.14.1,
and Python 3.11.15. The only upstream-source change was the independently
verified package metadata floor from Python 3.12 to Python 3.11.

| CodeBuild result | Verified value |
|---|---:|
| Build ID | `lumi-cb-lingua-cp311-94883c30bde4:ccc87389-27d8-4cf3-aa00-95286e20268c` |
| State | `SUCCEEDED` |
| Total wall time | 445.569 seconds |
| Upstream Python tests | 74 / 74 passed |
| Parity corpus | 225 files / 222,790 records |
| Lumi decision SHA-256 | `ab8a64fc03b886535e3b66db92dddde2d5fdea3184d4fd7057f831a2f938884b` |
| Threshold-0.7 SHA-256 | `3194178d8f08e7af75d8aeb212bb3716de8a02a6bda52e1eea0ffb5d2607be4c` |
| Wheel | `cp311-cp311-manylinux_2_34_x86_64` |
| Wheel bytes | 170,336,000 |
| Wheel SHA-256 | `ae73436bd29491e194c478101b4c4326fe1f13f90d4e313d3720b20f8c88efcf` |
| Conservative build compute | no more than $0.16 |

The wheel, build report, and parity report were downloaded and independently
verified. Their content-addressed S3 keys and VersionIds are pinned by follow-up
release `068ba2d4fcb91f802e2135779aefa3977b5c671d22025ae1a2323ce5178ff024`.
The temporary CodeBuild project, role, and log group were deleted before the
Glue job started.

The follow-up Glue run used source commit
`4b0a7953ae439af53051900fd6984ab0a11aaec5` and run ID
`jr_efe6a144ab76d145088f9a8ff6af60e3586ac1232454356f3b76789e4f307d67`.
Glue downloaded the custom wheel and completed its Python-module setup. The
script then requested the reserved `JOB_RUN_ID` option even though Glue's
`getResolvedOptions` parser already injects it, producing an argparse conflict
before `SparkContext`, source download, or corpus processing.

| Follow-up Glue result | Verified value |
|---|---:|
| Job state | `FAILED` (failed safe before data processing) |
| Execution time | 61 seconds |
| DPU seconds | 493 |
| Estimated Glue compute | $0.0603 |
| Release-specific S3 objects written | 0 |
| Production shards written | false |
| Model training started | false |
| Conservative combined compute | no more than $0.2203 |

The complete run metadata and both CloudWatch streams were sealed locally.
The Glue job, IAM role, and exact log streams were then deleted; live checks
return no matching job, project, or role. The consumed authorization permits no
retry. The local fix removes `JOB_RUN_ID` from the requested option list while
still requiring Glue's injected value; the focused Glue/Spark suite passes
14/14. A paid rerun or production acquisition requires a new explicit owner
authorization.

## Corrected bounded canary result

A fresh exact owner authorization allowed the fix branch to be pushed and one
corrected 8-DPU, 30-minute Glue canary, with a $3 outer cap and no production or
training. No CodeBuild rerun occurred: the job reused the independently
verified, version-pinned CPython 3.11 wheel. Source commit
`acef08b74464a13f2a5afc5f87fc2b4531f13045` produced release
`893cf9dacda660b5cc6ee2d63aac3ca715c32661925764f38b16107f4347c41d`.

The single authorized run
`jr_bc5437059a2845d091dbd75a9375d0b40f2cae2386bb11cd7336521aa82a9908`
passed dependency bootstrap, the corrected Glue argument parser, all eight real
source partitions, immutable-prefix loading, global exact/near deduplication,
and the independent sequential-reference comparison.

| Corrected Glue result | Verified value |
|---|---:|
| Job state | `SUCCEEDED` |
| Worker envelope | 8 x `G.1X` / 8 DPU |
| Execution time | 476 seconds |
| DPU seconds | 3,808 |
| Estimated Glue compute | $0.465422 |
| In-script elapsed time | 399.355 seconds |
| Candidate documents | 13,027 |
| Candidate tokens | 39,196,478 |
| Accepted documents | 12,999 |
| Near-duplicate rejections | 28 |
| Exact-duplicate rejections | 0 |
| Spark/sequential parity | pass |
| Pipeline candidate throughput | 98,149.5 tokens/second |
| End-to-end candidate throughput | 82,345.5 tokens/second |
| Production shards written | false |

The 1,144-byte result report independently matches SHA-256
`3eea9e0fa26e1b326d3e59a5e49596cd1d7e1df66c4bba1bd96c848d6847547e`.
The local 30.85MB evidence archive has a 13-entry independently verified
manifest. S3 retains 38 canary-result objects totaling 162,192,937 bytes under
the non-production control prefix. The temporary checkpoint objects, Glue job,
IAM role, and all nine exact CloudWatch streams were deleted; live absence
checks pass.

The measured run proves Glue 5.1 can execute Lumi's pinned language filtering,
tokenization, resume-anchor, and ordered-dedup semantics. Its end-to-end
10,293.2 candidate tokens per DPU-second gives a purely linear 256-DPU
120B-token projection of 12.65 hours. That figure is a capacity reference, not
a production guarantee or authorization: full-scale duplicate density,
shuffle, adaptive category floors, shard publication, retry policy, and a hard
budget stop still require a production harness and a separate owner-approved
launch proposal.

## Verified production staging result

The owner later authorized the bounded 150B candidate-acquisition workflow.
Live Glue allocation proved that this account's ETL executor ceiling is 100 DPU,
despite the Service Quotas catalog exposing a separate 1,000-DPU task quota.
Production staging therefore used 100 DPU. The owner subsequently fixed the
candidate-mapping target at 512 DPU rather than accepting a slow 100-DPU
production fallback. The $4,000 end-to-end acquisition guard remains unchanged;
the 512-DPU mapping launch stays blocked until Glue's enforced capacity is
raised to match the account's displayed quota.

Three startup attempts failed closed while establishing that live envelope. The
256-DPU attempt was rejected before execution; a 100-DPU attempt exposed an
unnecessary eager `tokenizers` import; and the next exposed a `SparkFiles`
lookup before `SparkContext` initialization. Both runtime defects now have
local regression gates, including an exact archive import smoke without site
packages and the same Spark initialization order used by the successful canary.

Source commit `2a01004ea4fcf021155d975bf35794d24328d4f5` produced
release `ff11ed9ef6457c57722a200e97c7c4400eed6444a87b7a7a8f043947a2e8e0df`.
The one bounded 100-DPU run
`jr_e7c7d9f62197086eee7665fd012fd1b7e2b32e26f5b48a98049db871640fa006`
then staged the complete pinned source inventory.

| Production staging result | Verified value |
|---|---:|
| Job state | `SUCCEEDED` |
| Worker envelope | 100 x `G.1X` / 100 DPU |
| Execution time | 474 seconds |
| DPU seconds | 47,419 |
| Successful-run compute | $5.795656 |
| All production attempts | 58,039 DPU-seconds / $7.093656 |
| Pinned source files | 758 / 758 |
| Source rows | 168,682,770 |
| Immutable zstd chunks | 14,658 |
| Staged compressed bytes | 500,163,871,471 |
| Staged uncompressed bytes | 1,916,352,637,159 |
| Model training started | false |

The immutable driver result has SHA-256
`3a267489759d4c96fd86a5bce18e09543872ca20628a304c75dee8cccbdc6289`.
An independent 64-worker read-only audit downloaded all 758 manifests and
verified every one of the 14,658 chunk objects against its pinned S3 VersionId,
content length, SHA-256 checksum, AES-256 encryption, source digest, and line
interval metadata. The audit report passes with SHA-256
`613bce3b47f2d1ff3cc74bb71c0956ee3637ad5faab10a15f633f5e2c8507ac3`.
This seals source staging only; it does not claim 150B candidate tokens and does
not authorize or start model training. Adaptive candidate mapping is the next
production phase.

## Prepared 512-DPU / 150B mapping gate

The unlaunched mapping harness binds 256 `G.2X` workers to exactly 512 DPU. Its
hard target is 150,000,000,000 total candidate tokens: the verified
237,769,640-token prefix plus at most 149,762,230,360 newly mapped tokens. A
document-boundary prefix selector can finish slightly below a category target,
but it can never cross the 150B aggregate ceiling.

Work is split into immutable waves of at most 2,000 staged chunks. Every
completed wave publishes a hash-bound manifest and checkpoint. A continuation
authorization must bind the exact latest open checkpoint, so a safe stop or
failure does not silently return to step zero. The job soft-stops at 660 minutes
and has a 720-minute hard timeout. At the pinned $0.44/DPU-hour rate, the
12-hour compute ceiling is $2,703.36; no mapping, deduplication, sharding, GPU,
or model-training job has been launched by this preparation.

The local package, authorization, and launcher dry-run all report 512 DPU,
256 `G.2X` workers, and the exact 150B hard stop. A 256-DPU staging probe remains
the live counter-evidence: Glue rejected it before execution with `Exceeded
maximum concurrent compute capacity for your account: 100`, even though Service
Quotas reports 1,000 for `L-08F3B322`. Therefore a 512-DPU paid launch must wait
for AWS to align the executor-side capacity; the harness will not silently
downgrade to 100 DPU.

## Owner-approved 100-DPU execution envelope

Rather than wait indefinitely for AWS to align the hidden executor capacity,
the owner explicitly superseded the unlaunched 512-DPU proposal with one
100-DPU mapping run. The data target is unchanged: the verified 237,769,640-token
prefix plus at most 149,762,230,360 newly mapped tokens, never more than 150B
total candidate tokens.

The approved job uses 100 `G.1X` workers and checkpoints at most 500 staged
chunks per immutable wave. It soft-stops at 3,960 minutes and has a 4,320-minute
hard timeout. At $0.44/DPU-hour, the absolute mapping-compute ceiling is
$3,168. Deduplication, sharding, GPU compute, and model training are not part of
this authorization and require separate later decisions.

The first authorized start reached `FAILED` after zero execution seconds with
no `DPUSeconds`. Glue reported that the new IAM role lacked assume-role
permission, while a read-back immediately confirmed the correct
`glue.amazonaws.com` trust policy. The timestamps and zero-runtime state are
consistent with IAM propagation delay; no mapping output or model training was
created. The launch record and failure report are sealed under release
`3f26da617278f25bfd8a2cd19644d3ef501d4ba77e685a13b203cb2b284cc72a`.

The launcher now waits 60 seconds whenever it creates a fresh role and includes
the exact authorization hash in the Glue job name. This keeps replacement
attempts auditable and avoids reusing a job whose one-run authorization was
already consumed.

The separately authorized replacement proved that the propagation fix works:
Glue assumed the new role and entered dependency bootstrap. It then failed
closed before mapping because the launcher's exact-read policy omitted the
already pinned Lingua CPython 3.11 wheel used by
`--additional-python-modules`. Run
`jr_44baee3465335b556b319ae0046a52258618bf166bd62a5e2bc5cd612d8211d4`
reported 36 execution seconds and 3,676 DPU-seconds, approximately $0.449289 at
$0.44/DPU-hour. No candidate mapping, deduplication, sharding, GPU compute, or
model training started. The launch record and failure report are sealed under
release `5b36827ff7526551f62bf982311164d09d06a0dcbdad764c85efc04c0cf3b15c`.

The launcher now derives two explicit artifact sets: all pinned objects receive
exact `GetObject`/`GetObjectVersion` permission, while only Spark runtime files
enter `--extra-files`. A regression test proves the wheel is readable without
being downloaded twice. This code correction authorizes no additional paid
run; another attempt requires a new exact owner approval.

## Reproduction commands

```bash
PYTHONPATH=src python3 scripts/plan_distributed_corpus.py \
  --output "$LUMI_DATA/distributed-corpus-v0.1/plan.json"

PYTHONPATH=src python3 scripts/export_distributed_resume_anchors.py \
  --evidence-directory "$MANUAL_STOP_EVIDENCE" \
  --output-directory "$LUMI_DATA/distributed-corpus-v0.1/resume-anchors"

PYTHONPATH=src python3 scripts/package_glue_distributed_canary.py \
  --anchor-export-directory "$LUMI_DATA/distributed-corpus-v0.1/resume-anchors" \
  --lingua-wheel-pin "$LUMI_DATA/distributed-corpus-v0.1/lingua-wheel-pin-v0.1.json" \
  --output-directory "$LUMI_DATA/distributed-corpus-v0.1/glue-canary-package"
```

The two failed Glue packages and the successful corrected package are consumed
evidence and must not be relaunched. The corrected result keeps the same source
bounds, ordered-dedup semantics, immutable resume anchors, and
no-production-output policy. It authorizes no production acquisition.
