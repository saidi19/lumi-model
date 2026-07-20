# Lumi S3-staged corpus transport v0.1

Status: **exact real-source Mac/Spark and bounded S3/EMR cloud parity pass; no
production acquisition, GPU, or model training is authorized**

## Decision

The internet-facing source fetcher and the EMR consumer are now separate. The
fetcher must first verify an exact revision-, byte-, and SHA-256-pinned source
file and conditionally write it to a content-addressed object in the existing
private S3 bucket. EMR receives only that S3 binding. A missing or changed S3
binding fails closed; the job has no direct-HTTP fallback.

This is preferable to adding a NAT gateway to EMR. AWS documents that EMR
Serverless blocks outbound internet by default while allowing selected
same-Region services, including S3, without VPC connectivity. Keeping source
transport outside EMR removes the network assumption that invalidated the
previous canary and makes every worker consume the same immutable bytes.

## Exact local plan

| Boundary | Verified value |
|---|---:|
| Plan fingerprint | `f1a95f43a6df8209345f368c2b6c75435849b566557d91ddb3a1f4f358629333` |
| Canary source objects | 8 |
| Canary compressed bytes | 1,456,514,440 |
| Category coverage | all 7 categories |
| Records per source | at most 2,000 |
| Direct HTTP in EMR | forbidden |
| Cloud mutation / paid compute | false / false |
| Production shards / model training | forbidden / forbidden |

The eight-source selection is unchanged from the previous all-category parity
gate and resumes source ordinal 0 at line 38,849. The largest selected object is
652,928,967 bytes. The largest object in the complete 758-file inventory is
2,704,144,015 bytes, below the pinned 5,000,000,000-byte single-`PutObject`
boundary.

## Integrity and restart contract

Each staged key is derived only from its upstream SHA-256:

`lumi-corpus/v0.1/raw/source-objects/sha256/<first-two-hex>/<sha256>.json.gz`

The producer:

- runs with two streaming workers at the start of the same bounded
  `BUILD_GENERAL1_SMALL` image build;
- keeps the existing 30-minute/$0.15 CodeBuild hard cap and uses its documented
  64GB disk and 4GiB memory envelope;
- verifies the downloaded file's exact byte size and SHA-256 before upload;
- supplies S3's base64 SHA-256 checksum and SSE-S3 (`AES256`);
- uses `If-None-Match: *`, so an existing key cannot be overwritten;
- reuses an existing object only after its stored checksum, size, type,
  encryption, and Lumi identity metadata all match.

The consumer:

- requires a release-fingerprinted `staged_object` binding for every source;
- verifies `HeadObject` with checksum mode before reading;
- preserves interrupted partial downloads and resumes with a byte-range GET;
- rechecks the complete local byte count and SHA-256 before opening gzip;
- never falls back to Hugging Face or another external URL.

Transient raw objects are retained only until their corresponding processed
output and evidence are verified. This does not change the 120B adaptive
candidate floors, 150B fail-closed maximum, or final 80B training mix.

## Local verification

The corpus-focused suite passed **79/79 tests**, and the complete repository
suite passed **239/239 tests**, under Python 3.12.13, Java 17.0.19, and PySpark
3.5.6. It covers:

- new write, verified reuse, conditional collision, and wrong-object refusal;
- interrupted ranged download and exact resume;
- gzip byte/record parity through the S3-only path;
- source identity, plan fingerprint, category, byte, and authorization caps;
- real local Spark ordered deduplication against the independent sequential
  reference;
- preserved resume anchors, token accounting, S3/storage contracts, and EMR
  release/request boundaries.

Ruff passes for the complete repository. The temporary test environment and
generated plan stay outside Git.

## Superseded package evidence

The non-authorizing EMR package was built from clean source commit
`9c67a766b9a1503d2bb99971fee9f92e0ae15e03` and independently verified after
copying to durable local storage. It was never launched. The later same-build
producer integration intentionally supersedes it, so its approval boundary and
fingerprint cannot be reused.

| Package field | Verified value |
|---|---|
| Release fingerprint | `0d2043bceacd06919fd3efe1b323f435ecdf4949740b7482cf8820de8ca4af88` |
| Release SHA-256 | `2d2821e2a61519a3670554e4165c754a27f69f59882fcfbd119b843046593125` |
| Code archive SHA-256 | `e29aa542a414596e363ba1250b1a3e75833ee575add78e03a8803731d5267243` |
| Image context SHA-256 | `3f492b734aab2060af9f4efae039c0c665c8e68bf7356401a824a4f0c94c418d` |
| Manifest SHA-256 | `6af6a170efa442becc0f154e91fe8e9ab9151986cab8ad3e9868ca81eb40214d` |
| Package status | `verified_not_authorized` |

The pinned build-plus-EMR compute ceiling remains $0.755904. At current pinned S3
prices, eight source PUTs cost $0.00004, inbound and same-Region transfer cost
$0, and retaining all 1.4565 GB for a full month would cost about $0.03350.
The internet-facing producer now runs inside that same CodeBuild ceiling, so
there is no separate Batch/Fargate compute line item.

## Final same-build package evidence

The corrected package was rebuilt from clean source commit
`16588aa34f45fdecde1832aabe1c2a64ef0a7300`. Every manifest entry passed an
independent SHA-256 check before and after copying to durable local storage at
`s3-staged-emr-canary-v0.2-f6ac38bb85bd4295`. The package remains explicitly
non-authorizing: it neither uploaded source objects nor started CodeBuild or
EMR Serverless.

| Package field | Verified value |
|---|---|
| Release fingerprint | `f6ac38bb85bd429545f5dea0820e97b4d147d3dbec85ba39631aa3e5daec8437` |
| Release SHA-256 | `a0659844bbc11afd5bd4abdad9e13f37ec1bb47a74b8d41cbe6d3dbfe4c838f9` |
| Code archive SHA-256 | `ada6287a863525fd1c21d42699a873266079007f9f6244fd7b512aec782c50b2` |
| Image context SHA-256 | `04b7909e44bad549b46f986d52cbe3d8c671e2d7b4f18aa4c24d17a4b2d4bcc4` |
| Staging plan SHA-256 | `1774bc048037a9c75c47a390dceabbfd84b93c5a8c9e8ba1e83a8c6f40be65b2` |
| Manifest SHA-256 | `5e6d747359f3e7b6fb0b6c8a3f270ade40a01dae6403e70a2ab8ece8b3b1605a` |
| Package status | `verified_not_authorized` |

## First authorized source-staging build

One fresh exact authorization started CodeBuild build
`lumi-emr-staged-f6ac38bb85bd:c014a713-9146-484e-8b4d-3cc118f2bbfb` from the
version-pinned image context. It spent 303 seconds provisioning and failed in
eight seconds of `PRE_BUILD`, before any external source download, S3 raw write,
Docker build, ECR image push, or EMR launch.

The failure was an IAM contract omission. The producer checks whether each
content-addressed destination already exists with `HeadObject`. AWS returns
`404 Not Found` for a missing object only when the caller also has
`s3:ListBucket`; otherwise the same missing key returns `403 Forbidden`. The
temporary role granted exact-key `s3:GetObject`/`s3:PutObject` but omitted the
bucket-level read-only permission, so the fail-closed client correctly refused
to interpret the `403` as absence.

Post-run verification found zero staged raw objects, zero ECR images, and zero
EMR applications. The CodeBuild project, log group, empty ECR repository, and
temporary IAM role are deleted; EC2 and EMR remain empty. The conservative
CodeBuild compute upper bound is $0.03. Full API/log hashes and cleanup evidence
are recorded in `reports/emr-codebuild-s3-staging-v0.1.json`.

The policy renderer now adds `s3:ListBucket` on only the sealed corpus bucket;
payload reads and writes remain restricted to the exact eight object ARNs. A
regression test requires this permission and continues to reject S3 delete,
EMR, EC2, production, GPU, and model-training access. The consumed approval
cannot authorize a retry.

## Exact real-source Mac canary

The owner-authorized local canary then exercised the production-shaped data
path on the exact eight pinned source objects without mutating AWS. The files
were downloaded once, verified against the sealed byte and SHA-256 pins, and
reused only after another complete hash check. Git contains only the compact
result; the 1.456GB raw cache remains outside the repository.

| Result | Verified value |
|---|---:|
| Source objects / categories | 8 / all 7 |
| Compressed source bytes | 1,456,514,440 |
| Raw records processed | 14,061 |
| Filtered candidates / tokens | 13,027 / 39,196,478 |
| Immutable resume anchors | 38,349 |
| Sequential accepted / rejected | 12,999 / 28 |
| Spark accepted / rejected | 12,999 / 28 |
| Exact / near-duplicate rejections | 0 / 28 |
| Spark/sequential parity | exact IDs and reasons |
| Parity SHA-256 | `13edb46411c3b5eeaaf35237f70e379d72eb22b8a223c0d70586ece649c96837` |
| Report SHA-256 | `cb18d437339eefbf4e4c57e2d91b5f9f00b9b54b745d271e8e4bdb760a391ace` |
| Manifest SHA-256 | `a568a676c47725fefeab5f60192df32e576967b1117a56d7a5e5cab10397df3a` |

The first local Spark transfer exposed a macOS socket-buffer limit when one
large Python partition was sent to the JVM. Partitioning candidates into 16
bounded slices and anchors into 32 bounded slices fixed the transport issue;
the filtering, source order, deduplication rules, and accepted/rejected result
did not change. The corrected cached-resume attempt completed in 316.007
seconds. All raw-source entries in `manifest.sha256` pass an independent hash
check. No S3 object, AWS service, GPU, production shard, or training process was
created.

## Authorized S3-only cloud parity result

A fresh release from merged source commit
`f2251db105bc951d9413cc2d6342e7a0c0648bb1` and one exact owner authorization
permitted one CodeBuild source-staging/image build and one EMR Serverless job.
CodeBuild passed all source checks, staged the eight content-addressed objects,
built and pushed the immutable image, and completed without retry.

| Result | Verified value |
|---|---:|
| CodeBuild ID | `lumi-emr-staged-7bdcf1b7999f:1d0b1a2c-84ba-4b19-8222-914d8181c7e0` |
| CodeBuild status / duration / retries | `SUCCEEDED` / 650 seconds / 0 |
| Image digest | `sha256:8294908a75cbd27301322abfc0c713ed5cd0568c751b9f7c58594136fd8097c9` |
| Image bytes | 2,631,167,482 |
| EMR application / job | `00g7bpg9jgq0b509` / `00g7bph6oo8evg0b` |
| EMR status / attempts / execution | `SUCCESS` / 1 / 361 seconds |
| Billed vCPU-hours / memory-GB-hours | 1.57 / 6.28 |
| Candidate documents / tokens | 13,027 / 39,196,478 |
| Accepted / near rejected | 12,999 / 28 |
| Local/cloud parity SHA-256 | `13edb46411c3b5eeaaf35237f70e379d72eb22b8a223c0d70586ece649c96837` |
| Production shards / model training | false / false |
| Combined compute estimate | $0.173908660 |

The downloaded cloud report has SHA-256
`399b0087b4f1c1fec72c3b0784bbc817768c10c573959d0cb24bd874b78df553`.
An independent local read of the cloud accepted/rejected Parquet files
reproduces the Mac parity digest exactly, not only the aggregate counts. The
72-file evidence archive is sealed by manifest SHA-256
`4c2aefa6c7374cc401f0d713841f5b308b9a20287f3ecd5f17ac36b9590d6d92`.
The machine-readable summary is
`reports/emr-s3-parity-canary-v0.3.json`.

After evidence verification, the EMR application, CodeBuild project/log group,
both temporary IAM roles, ECR image/repository, and every version of this run's
raw source, checkpoint, result, and release-scoped control objects were removed.
Post-cleanup listings contain zero matching versions. The authorization is
consumed and cannot be reused.

## Next bounded boundary

The bounded cloud parity condition is now satisfied. The next possible action
is a separately priced and separately owner-authorized distributed production
resume from the preserved 237,769,640-token prefix toward the seven category
floors totaling 120B candidate tokens. Its capacity, checkpoint cadence,
runtime, spend ceiling, and auto-stop boundary must be measured and approved
before launch. This result does not authorize production acquisition, GPUs, the
4B pilot, or the 4B/80B run.

## Official references

- [EMR Serverless VPC and external-network behavior](https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/vpc-access.html)
- [EMR Serverless same-Region service access](https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/considerations.html)
- [S3 upload checksum validation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity-upload.html)
- [S3 conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html)
- [S3 `HeadObject` missing-key permission semantics](https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html)
- [CodeBuild compute, memory, and disk sizes](https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-compute-types.html)
- [AWS CLI conditional/checksummed `PutObject`](https://docs.aws.amazon.com/cli/latest/reference/s3api/put-object.html)
- [Spark 3.5.6 supported Java runtimes](https://spark.apache.org/docs/3.5.6/)
