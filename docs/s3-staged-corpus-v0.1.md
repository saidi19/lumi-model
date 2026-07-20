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

## Next bounded boundary

No cloud retry is implied by this result. Before another paid action:

1. build and independently verify one clean package from the same-build source
   commit;
2. obtain a new exact owner authorization—the previous approval is consumed;
3. stage only the eight pinned objects and independently verify all S3 heads;
4. run one no-retry EMR parity canary and require exact accepted-ID and
   rejection-reason parity;
5. preserve evidence and delete temporary image, roles, application, and raw
   canary objects.

Only a passing parity result can unlock a separately approved distributed
production resume toward the 120B candidate floors. It cannot authorize model
training, GPUs, the 4B pilot, or the 4B/80B run.

## Official references

- [EMR Serverless VPC and external-network behavior](https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/vpc-access.html)
- [EMR Serverless same-Region service access](https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/considerations.html)
- [S3 upload checksum validation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity-upload.html)
- [S3 conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html)
- [CodeBuild compute, memory, and disk sizes](https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-compute-types.html)
- [AWS CLI conditional/checksummed `PutObject`](https://docs.aws.amazon.com/cli/latest/reference/s3api/put-object.html)
- [Spark 3.5.6 supported Java runtimes](https://spark.apache.org/docs/3.5.6/)
