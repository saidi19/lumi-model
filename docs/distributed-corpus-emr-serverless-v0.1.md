# Lumi distributed corpus canary on EMR Serverless v0.1

Status: **the corrected owner-authorized image build passed and one bounded EMR
job started successfully, but the job was cancelled safely after its source
tasks proved that default EMR Serverless networking cannot reach
`huggingface.co`; parity was not reached, no production shard was written, and
all temporary resources are deleted or terminated**

## Scope

This is a runtime replacement inside the existing final-corpus acquisition
roadmap step. It does not add a new corpus target and does not reduce the seven
category floors totaling 120B accepted candidate tokens. The 150B value remains
only a fail-closed candidate-token maximum; the eventual training mix remains
80B tokens.

The small canary exists to prove the distributed implementation before a large
resume. It uses eight source objects, covers all seven categories, and reads at
most 2,000 records per source. Passing it will validate Spark mapping,
source-order exact/near deduplication, immutable-prefix resume behavior, Python
runtime, S3 checkpoints, and sequential-reference parity. It will not produce
training shards.

## Why Glue was replaced

The one authorized Glue 5.1 attempt failed safe in dependency bootstrap. Glue
used Python 3.11, while Lumi's sealed quality policy pins
`lingua-language-detector==2.2.0`, whose available wheel requires Python 3.12.
Changing the detector or version would invalidate earlier language decisions,
so the runtime is changed instead.

EMR Serverless 7.13.0 supplies Spark 3.5.6 and supports a custom image. Lumi's
image starts from the official x86_64 EMR image pinned by digest, installs
Python 3.12, installs a fully hash-pinned wheel set, verifies exact versions and
imports during the build, leaves protected EMR environment variables unchanged,
and ends as `hadoop:hadoop`.

## Fixed canary envelope

| Boundary | Hard maximum |
|---|---:|
| Region | `us-east-1` |
| Runtime | EMR Serverless 7.13.0 / Spark 3.5.6 / Python 3.12 |
| Architecture | x86_64 |
| Driver | 4 vCPU / 14GB RAM / 20GB disk |
| Executors | 3 x (4 vCPU / 14GB RAM / 20GB disk) |
| Application capacity | 16 vCPU / 64GB RAM / 80GB disk |
| Source objects | 8 |
| Records per source | 2,000 |
| Category coverage | all 7 categories |
| Compressed bytes per source | 700,000,000 |
| Total compressed source input | 1,500,000,000 |
| EMR runtime | 30 minutes |
| Job retries | 0 (`maxAttempts=1`) |
| Auto-stop idle timeout | 5 minutes |
| Production shards | forbidden |

The application has no pre-initialized capacity and dynamic allocation is off.
Its deterministic client tokens and immutable run token prevent an accidental
duplicate launch. Candidate/checkpoint/result paths are release- and
run-token-scoped, and output writes use create-only semantics.

## Pinned cost boundary

The 2026-07-17 AWS Price List snapshot records x86 EMR Serverless at
$0.052624/vCPU-hour plus $0.0057785/GB-hour. At the full 16-vCPU/64GB ceiling
for all 30 minutes, EMR compute is capped at $0.605904. A small x86 CodeBuild
image build is capped at 30 minutes and $0.15. Their combined compute ceiling is
$0.755904, excluding only tiny S3, CloudWatch, and ECR storage/request charges.

These are upper bounds, not permission to spend. Current release metadata sets
every paid authorization field to false.

## Fail-closed release and approval flow

The local package contains the exact code archive, job script, Python image
context, benchmark signatures, and 38,349 immutable resume anchors. Every file
is hashed, the image base is digest-pinned, and the release fingerprints the
full Git commit and implementation hashes.

An image-unbound package cannot create an application. A future authorization
must exactly match the package fingerprint, image-context hash, source commit,
run token, expiry, source/runtime/cost caps, and this text:

> I approve one bounded Lumi AWS CodeBuild image build and one bounded AWS EMR Serverless distributed corpus canary only.

The previous Glue approval does not match and is rejected. Even a matching
authorization cannot enable production acquisition, GPU compute, model
training, or the 4B run. A built image must be bound to an immutable private-ECR
digest below 10GB before a job request can be rendered.

## First authorized CodeBuild result

The first exact authorization produced one and only one
`BUILD_GENERAL1_SMALL` build. AWS spent 323 seconds provisioning the environment,
downloaded the exact versioned S3 image context, and then failed immediately in
`PRE_BUILD`: CodeBuild does not define `CODEBUILD_RESOLVED_SOURCE_VERSION` for
this versioned S3 source. The correct built-in variable is
`CODEBUILD_SOURCE_VERSION`.

| Result | Verified value |
|---|---:|
| Build number | 1 |
| Docker login/build/push started | false / false / false |
| ECR images | 0 |
| EMR applications/jobs | 0 / 0 |
| Production shards/model training | false / false |
| Conservative CodeBuild compute upper bound | $0.03 |

The authorization is consumed; no retry occurred. The CodeBuild project and
log group, both temporary IAM roles, empty ECR repository, and live S3
authorization object were deleted and independently confirmed absent. The
machine-readable record is
[`reports/emr-codebuild-canary-v0.1.json`](../reports/emr-codebuild-canary-v0.1.json).

The corrected buildspec now uses `CODEBUILD_SOURCE_VERSION`, is included inside
the digest-pinned image context, and is therefore covered by the release
fingerprint rather than being an untracked launch-time input.

## Corrected build and bounded EMR result

A fresh exact authorization permitted one corrected build and, only after that
build passed, one EMR canary. CodeBuild completed its runtime smoke, Docker build,
and immutable ECR push. EMR then resolved that exact digest, started the driver
and all three executors, and completed the initial anchor and shuffle stages.

The source-fetch stage could not reach `huggingface.co:443`. This is the expected
default EMR Serverless network boundary, not a corpus-filter or Spark-runtime
failure: without VPC connectivity, jobs can reach selected same-Region AWS
services such as S3, but outbound internet access is blocked. Executor logs
record `curl (28)` connection timeouts. The job was cancelled after 273 seconds
instead of allowing ten long curl retries to consume the 30-minute ceiling.
There was no retry or second job.

| Result | Verified value |
|---|---:|
| CodeBuild status / duration | `SUCCEEDED` / 545 seconds |
| Runtime smoke | passed |
| Image digest / size | `sha256:9d7ad4a1e8690f37b8514c250ae81506cd2ee7a4c94d72b249b7155e3a4ce74d` / 2,631,167,478 bytes |
| EMR application / job | `00g7bkjsn1bm5u09` / `00g7bkm8v6n0700b` |
| EMR attempts | 1 |
| Driver / executors started | 1 / 3 |
| EMR billed vCPU-hours / memory-GB-hours | 1.181 / 4.724 |
| Measured EMR compute estimate | $0.089446578 |
| Conservative successful-build estimate | $0.05 |
| Combined compute estimate | $0.139446578 |
| Sequential parity reached | false |
| Production shards/model training | false / false |

The CodeBuild project and log group, both temporary IAM roles, ECR image and
repository, and live S3 authorization object were deleted and independently
confirmed absent. The EMR application is `TERMINATED`; the earlier EC2 worker is
also absent. The complete terminal record is
[`reports/emr-serverless-canary-v0.2.json`](../reports/emr-serverless-canary-v0.2.json),
and the downloaded Spark logs are retained outside Git with the sealed launch
package.

## Canary pass condition

Workers run the existing quality, provenance, PII/secret, benchmark-exclusion,
tokenization, and MinHash mapping code. Spark then executes the ordered global
deduplication against the preserved 38,349-document prefix. The same bounded
candidates are evaluated independently by the sequential implementation.

The gate passes only when accepted document IDs and exact/near rejection reasons
match exactly, runtime and artifact hashes match, all safety caps remain intact,
and no production shard is written. Failure preserves bounded evidence, cleans
all temporary compute/IAM resources, and leaves production stopped.

## Next boundary

Do not rerun the same direct-Hugging-Face job. The replacement is now
implemented locally: exact source files bind to content-addressed S3 objects,
and the EMR mapper uses verified, resumable S3 GETs with no direct-HTTP
fallback. The unchanged eight-source/all-category slice totals 1,456,514,440
compressed bytes. Its local integrity, restart, and real Spark/sequential
parity suite passes 76/76 focused tests, and the complete repository suite
passes 236/236; see
[`s3-staged-corpus-v0.1.md`](s3-staged-corpus-v0.1.md).

No source payload was uploaded and no paid compute was authorized by that local
result. A clean, image-unbound package from source commit `9c67a76` is now
independently verified; its release fingerprint is
`0d2043bceacd06919fd3efe1b323f435ecdf4949740b7482cf8820de8ca4af88`.
The bounded source-staging runner, its exact compute cap, and a new owner
approval are still required before staging the eight objects or running one
replacement EMR canary. Only a passing cloud parity result can produce the measured
production-resume proposal. Resuming acquisition toward the 120B floors remains
a later, separately priced and separately authorized roadmap action.

Official runtime references:

- [EMR Serverless 7.13.0 release](https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/release-version-7130.html)
- [EMR Serverless custom images](https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/application-custom-image.html)
- [Custom Python images](https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/using-custom-images.html)
- [EMR Serverless network access](https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/vpc-access.html)
- [EMR Serverless pricing](https://aws.amazon.com/emr/pricing/)
