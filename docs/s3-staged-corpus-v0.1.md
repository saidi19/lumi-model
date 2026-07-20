# Lumi S3-staged corpus transport v0.1

Status: **local implementation and real Spark parity pass; no source payload was
uploaded, no AWS resource was created, and no paid compute is authorized**

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
| Plan fingerprint | `aac99c3587a611a0b255338d81309cda588d04c7d5ef31827720ac4129dc5468` |
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

The corpus-focused suite passed **76/76 tests**, and the complete repository
suite passed **236/236 tests**, under Python 3.12.13, Java 17.0.19, and PySpark
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

## Next bounded boundary

No cloud retry is implied by this result. Before another paid action:

1. create a clean, commit-exact release package containing this transport;
2. calculate and show the exact staging/build/EMR cost and cleanup envelope;
3. obtain a new exact owner authorization—the previous approval is consumed;
4. stage only the eight pinned objects and independently verify all S3 heads;
5. run one no-retry EMR parity canary and require exact accepted-ID and
   rejection-reason parity;
6. preserve evidence and delete temporary image, roles, application, and raw
   canary objects.

Only a passing parity result can unlock a separately approved distributed
production resume toward the 120B candidate floors. It cannot authorize model
training, GPUs, the 4B pilot, or the 4B/80B run.

## Official references

- [EMR Serverless VPC and external-network behavior](https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/vpc-access.html)
- [EMR Serverless same-Region service access](https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/considerations.html)
- [S3 upload checksum validation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity-upload.html)
- [S3 conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html)
- [AWS Batch array jobs for future production staging](https://docs.aws.amazon.com/batch/latest/userguide/array_jobs.html)
- [Spark 3.5.6 supported Java runtimes](https://spark.apache.org/docs/3.5.6/)
