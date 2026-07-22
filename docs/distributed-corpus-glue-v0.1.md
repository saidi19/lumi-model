# Lumi distributed corpus acquisition on AWS Glue v0.1

Status: **the 150B candidate map is sealed and the checkpoint-bound
global-dedup structural canary passes after two fail-closed corrections;
production dedup, sharding, GPU, and model training remain unauthorized**

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

The next separately authorized attempt confirmed both startup corrections in
the live service: Glue assumed the role, downloaded and installed the exact
pinned Lingua wheel, initialized Spark, and began allocating executors. It then
failed closed before candidate mapping while reading the first sealed staged
manifest. The runner had already hashed the downloaded JSON body and matched it
to the release's `manifest_sha256`, but failed to reattach that verified digest
to the wrapper expected by `validate_published_staged_source`.

Run `jr_5c049a89c20401e315668ed47d21a66717d5a200504ffa334fe5c40b3d92a5fd`
reported 75 execution seconds and 7,521 DPU-seconds, approximately $0.919233 at
$0.44/DPU-hour. Read-only S3 checks found zero checkpoint objects, zero mapped
candidate objects, and zero run-manifest objects for release
`5a07f3a2db3a9264e08fa1e3e8914c2f277a736848b97b3b24f499022fcee5a0`.
No deduplication, sharding, GPU compute, or model training started.

The runner now reattaches only the release-pinned digest after independently
hashing the downloaded body. A behavioral regression test reproduces the exact
stored-manifest shape, including the absent embedded digest, and verifies queue
loading succeeds. The consumed run and package are hash-sealed locally. This
fix does not authorize another paid run.

## Verified 150B candidate-mapping result

The next owner-authorized 100-DPU run mapped and durably checkpointed Waves
1--7, then failed after checkpoint publication when an exhausted-category
closure invariant was evaluated before its deterministic target reallocation.
No completed output was discarded. A fail-closed recovery audit verified every
published object, migrated exact checkpoint sequence 7 to corrected release
`b7a11166a1ecf1b67e55a7962ce24c26e4f97e97f8da0786d7c6be7768998041`,
and bound the continuation to recovered checkpoint
`8f801a0b2d0e61a6301a0068579b69d37dd1e58e3616bebe3df531dcb92c1b82`.
The recovered Wave-7 ledger entry and its 504 objects occur exactly once in the
final twelve-wave chain.

The separately authorized corrected continuation processed Waves 8--12 and
finished with Glue state `SUCCEEDED`. Independent local validation replayed the
sequence-11 to sequence-12 transition, recomputed the canonical checkpoint and
manifest hashes, matched the Wave-12 S3 inventory, and checked the final report
against the authorization and release.

| Final mapping result | Verified value |
|---|---:|
| Final checkpoint status | `ready_for_global_dedup` |
| Final checkpoint sequence | 12 |
| Final checkpoint SHA-256 | `90ccdacf5b8bff23c87061ec68bbce0d0994f811cdfdccaf3abd09d0d01be59b` |
| Pre-existing accepted prefix | 237,769,640 tokens |
| Newly mapped candidates | 149,762,206,414 tokens |
| Total candidate pool | 149,999,976,054 tokens |
| Document-boundary shortfall from 150B | 23,946 tokens |
| Mapped documents | 89,481,434 |
| Mapped Parquet storage | 629,842,275,871 bytes |
| Categories closed | 7 / 7 |
| Model training started | false |

The data-producing failed run and successful continuation used 2,161,503
DPU-seconds in total, approximately $264.1837 at the pinned $0.44/DPU-hour
rate. Including the two earlier nonzero startup failures documented above gives
2,172,700 DPU-seconds, approximately $265.5522, across all mapping attempts.
The final continuation itself used 767,595 DPU-seconds. Its consumed Glue job
definition and dedicated IAM role were deleted after the final evidence archive
passed; the immutable S3 candidates, checkpoints, and manifests remain.

This result seals candidate mapping only. Global exact/near deduplication,
quality selection into the final 80B mixture, held-out extraction, training
shards, and any GPU/model training remain separate gates and are not authorized
by this result.

## Prepared checkpoint-bound global-dedup gate

The next roadmap phase now has a local-only Glue package and runtime contract.
It does not reread raw internet sources or remap the corpus. It binds the exact
final map checkpoint to all twelve immutable wave manifests, flattens only the
verified Parquet outputs, and reuses Lumi's existing source-order exact/MinHash
Spark deduplication core plus the 38,349 immutable accepted-prefix anchors.

The adapter explicitly supports the verified release migration: Waves 1--7
belong to release `c31baae5eab5...`, while Waves 8--12 belong to corrected
release `b7a11166a1ec...`. Every manifest canonical hash, S3 namespace,
category total, object-inventory digest, document count, token count, and byte
count must match the final checkpoint. Missing, reordered, repeated, escaped,
or modified inputs fail before a paid job can be rendered.

| Prepared dedup boundary | Verified value |
|---|---:|
| Final map checkpoint | `90ccdacf5b8bff23c87061ec68bbce0d0994f811cdfdccaf3abd09d0d01be59b` |
| Dedup input-index SHA-256 | `2cd4537a86d6e48e2fa079c1088fef167d856229e208a45284fab216f207d266` |
| Immutable wave manifests | 12 |
| Parquet input locations | 56 |
| Mapped documents | 89,481,434 |
| Mapped candidate tokens | 149,762,206,414 |
| Accepted-prefix tokens | 237,769,640 |
| Total candidate tokens | 149,999,976,054 |
| Mapped Parquet bytes | 629,842,275,871 |
| Dedup-output envelope after 100GB reserve | 769,993,852,658 bytes |
| Original package verification | 312 tests passed; Ruff clean |

Canary and production permission are deliberately separate. The first possible
paid action is one exact **8-DPU, 30-minute structural canary** with a hard Glue
compute maximum of **$1.76**. Only after that result passes and is independently
sealed may the owner separately approve one **100-DPU, 24-hour production
global-dedup run**, whose conservative Glue-compute maximum is **$1,056**.
Neither envelope authorizes mapping, shard writing, GPU work, 4B training, or
the 80B-token training run. Failed jobs have zero automatic retries; a new run
requires a new exact authorization.

## Consumed first global-dedup canary

The first exact canary authorization targeted checkpoint
`90ccdacf5b8bff23c87061ec68bbce0d0994f811cdfdccaf3abd09d0d01be59b`,
input index
`2cd4537a86d6e48e2fa079c1088fef167d856229e208a45284fab216f207d266`,
and release
`fe8104126590fd74c98785eeb2788f6d9c71512f8c81a3ab100cb77a15234fe7`.
The single 8-DPU run
`jr_e93f11027c25a09d97e166df089fb1c85b33ab096d41fa07f619085a48d0a3b3`
failed during the runtime contract before anchor loading, candidate reads,
deduplication, or output publication. AWS Glue 5.1 exposed
`spark.version == "3.5.6-amzn-1"`; the sealed script required exact string
`"3.5.6"` even though the upstream Spark base version matched.

| Consumed canary evidence | Verified value |
|---|---:|
| Glue state | `FAILED` |
| Execution time | 83 seconds |
| DPU-seconds | 671 |
| Estimated compute at $0.44/DPU-hour | $0.082011 |
| Checkpoint-prefix objects | 0 |
| Canary-output objects | 0 |
| Result objects | 0 |
| Production-output objects | 0 |
| Automatic or manual retry | none |

The corrected local runtime contract still pins the release's exact base
version, accepts only an optional numeric AWS `-amzn-N` build suffix, records
both the base and full runtime version, and rejects a different Spark base or
vendor suffix. Regression coverage includes the observed AWS value plus both
rejection paths. The consumed Glue job and least-privilege IAM role are deleted,
and the hash-sealed attempt evidence is archived locally. This correction does
not authorize another paid canary. The corrected tree passes 313/313 tests and
Ruff lint.

## Consumed corrected global-dedup canary

The separately authorized corrected package used release fingerprint
`69649ed40b753a80694c660417b9316546f944c3b578055db094dc1a6afd4482`
and run
`jr_de1baf41cc7054a06354468611fb133e3d99d719d9c80d0672ed0ab5111481d0`.
It accepted Glue's pinned `3.5.6-amzn-1` runtime, loaded the immutable resume
anchors, and read the single bounded candidate input. It then failed while
materializing the deterministic 100,000-document canary prefix. The query
applied `orderBy(...).limit(100000)` to the complete candidate schema, so
Spark's `TakeOrderedAndProject` retained wide text, token, provenance, and
MinHash payloads on one executor. Four attempts reported
`java.lang.OutOfMemoryError: Java heap space`; the later shuffle-fetch failures
were consequences of those executor deaths.

| Corrected canary evidence | Verified value |
|---|---:|
| Glue state | `FAILED` |
| Execution time | 212 seconds |
| DPU-seconds | 1,703 |
| Estimated compute at $0.44/DPU-hour | $0.208144 |
| Candidate rows available before the cap | 249,831 |
| Authorized canary document cap | 100,000 |
| Checkpoint-prefix objects | 0 |
| Canary-output objects | 0 |
| Result objects | 0 |
| Production-output objects | 0 |
| Automatic or manual retry | none |

The failure is a canary sampling implementation defect, not evidence of source
or mapped-Parquet corruption. The local correction projects only
`source_ordinal`, `source_line`, `segment_index`, and `document_id` before the
ordered limit, broadcasts those narrow keys, and left-semi-joins them back to
the original candidate frame. A local Spark 3.5.6 physical-plan smoke confirms
that `TakeOrderedAndProject` outputs only those four keys and the final join is
a `BroadcastHashJoin LeftSemi`; the same deterministic first documents are
selected while wide text stays outside the top-K heap. The consumed Glue job
and least-privilege role are deleted, the four bounded output namespaces remain
empty, and the durable failure manifest verifies every archived evidence file.
The full tree passes 314/314 tests and Ruff lint. This fix authorizes no paid
retry, production dedup, sharding, GPU work, or model training.

## Verified fixed-memory global-dedup canary

A fresh exact owner authorization bound one final canary attempt to map
checkpoint
`90ccdacf5b8bff23c87061ec68bbce0d0994f811cdfdccaf3abd09d0d01be59b`,
input index
`2cd4537a86d6e48e2fa079c1088fef167d856229e208a45284fab216f207d266`,
and fixed release
`9cdb276a119758ef65b181294eb3d369cfaa6da6d18be688ef22712a9ecbf9a6`.
The exact source commit was
`487742309eca4d340aff0e926e834730d53bddb2`; it is contained in merged main.
Replay preflight found no existing deterministic job, role, or bounded output
object before launch.

The single 8-DPU run
`jr_636f897f5c6ee18d599b5c38d89a708e0c0706fe83620b509e360bd1178fca19`
completed successfully. Glue 5.1 ran Spark `3.5.6-amzn-1`, Python 3.11.15,
and Datasketch 2.0.0. The narrow-key top-K plus broadcast left-semi join avoided
the previous wide-row executor heap failure and preserved the deterministic
100,000-document cap.

| Fixed-memory canary evidence | Verified value |
|---|---:|
| Glue state | `SUCCEEDED` |
| Worker envelope | 8 x `G.1X` / 8 DPU |
| Timeout / automatic retries | 30 minutes / 0 |
| Execution time | 189 seconds |
| DPU-seconds | 1,515 |
| Estimated compute at $0.44/DPU-hour | $0.185167 |
| Candidate documents | 100,000 |
| Candidate tokens | 1,111,552,230 |
| Immutable anchor documents | 38,349 |
| Accepted documents | 99,996 |
| Near-duplicate rejections | 4 |
| Exact-duplicate rejections | 0 |
| LSH candidate pairs / near edges | 16 / 4 |
| Production-output objects | 0 |

Independent local verification downloaded every output Parquet object and
validated the release and authorization contracts. The 99,996 accepted and 4
rejected document IDs are individually unique, mutually disjoint, and conserve
all 100,000 canary inputs. The 35 accepted-ID objects total 6,422,420 bytes and
match inventory SHA-256
`5124a27fdd9b18ec5d3b4d67e206541c701f843391b6782076c3dd1c09a400e2`;
the one rejected-ID object is 1,685 bytes and matches
`806f97a178c17840184be211d037dea7476c2001f4fac331bff69eb66f09cdde`.
The immutable 1,569-byte result matches SHA-256
`9527fd70202016db1f19a459120ca83849b83e676f9cc2bf939a843709dfee26`.

The local evidence archive contains 48 hash-verified entries (about 7.2MiB);
its manifest SHA-256 is
`6f43a030fe2298254b99a837eb9add16d66083ee55822203a01556ba682c38da`.
After the archive was sealed, the consumed Glue job and least-privilege IAM role
were deleted and live absence checks passed. This proves only the bounded
structural global-dedup path. It does not authorize or claim completion of the
100-DPU production dedup, final 80B selection, training-shard publication, GPU
work, or model training.

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
