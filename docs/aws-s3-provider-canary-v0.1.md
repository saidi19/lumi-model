# Lumi AWS S3 provider and bounded I/O canary v0.1

Status: **AWS S3 `us-east-1` selected, exact pricing and S3-only cost cap pass,
live bounded I/O canary passes, and every temporary cloud resource is deleted;
bulk corpus materialization, paid compute, and 4B training remain unapproved**

## Decision boundary

This gate selects AWS S3 for the Lumi corpus object store without authorizing
the corpus itself. The contract in
[`aws-s3-provider-v0.1.json`](../configs/aws-s3-provider-v0.1.json) permits local
planning plus at most 32MiB of synthetic canary payload. It explicitly keeps:

- `bulk_corpus_materialization=false`;
- `transfer_bulk_source_data=false`;
- `paid_compute=false`; and
- `full_4b_training=false`.

No source-corpus text or model data was uploaded.

## Pinned pricing and cost projection

The machine-readable snapshot in
[`aws-s3-us-east-1-pricing-v0.1.json`](../configs/aws-s3-us-east-1-pricing-v0.1.json)
was extracted from the official public AWS Price List indexes on 2026-07-19.
Both complete source indexes are retained outside Git with the evidence.

| Rate | Pinned value |
|---|---:|
| S3 Standard, first 50TB | $0.023/GB-month |
| PUT/COPY/POST/LIST | $0.005/1,000 requests |
| GET and other requests | $0.004/10,000 requests |
| Same-region AWS service transfer | $0/GB |
| Internet data transfer out, first 10TB beyond free tier | $0.09/GB |

The S3 index is version `20260714015830`, published 2026-07-14, with SHA-256
`a386068ef92d4273d224c65d172e58920573f1e11d67406302933b038d7be07e`.
The AWS Data Transfer index is version `20260625165719`, published 2026-06-25,
with SHA-256
`471c2892eababac74f75b6781f8053e82c899094b23175213d86a0a0e77d7922`.
The verifier checks those exact public files against the curated snapshot.

Cost projection deliberately assumes no global free-tier benefit:

| Forecast | Cost |
|---|---:|
| Persistent 830GB in S3 Standard | $19.0900/month |
| Peak 1.0375TB in S3 Standard | $23.8625/month |
| 20,000 monthly tier-1 requests | $0.1000 |
| 100,000 monthly tier-2 requests | $0.0400 |
| Conservative 20GB internet egress | $1.8000 |
| Conservative projected total | **$25.8025/month** |
| Owner-approved S3-only cap | **$100/month** |

The unused monthly cap is $74.1975. A hypothetical full 1.0375TB internet
download would cost up to $93.375 at the first egress tier, so corpus compute
must remain in `us-east-1` and only bounded evidence should return to the Mac.
The S3 cap is independent of any later owner-approved GPU budget.

## Security contract

Both ephemeral canary buckets used:

- all four S3 Block Public Access controls;
- `BucketOwnerEnforced` ownership with ACLs disabled;
- explicit default SSE-S3 `AES256` encryption;
- versioning;
- a bucket policy denying non-TLS requests;
- one-day incomplete-multipart abort lifecycle;
- server access logging from the data bucket to the separate audit bucket; and
- owner-bound API calls and content SHA-256 metadata.

The runner derives globally unique bucket names without exposing the AWS
account ID in names or final evidence. It stores no credentials. The AWS CLI
login returned a temporary root console session; an attempted narrower role
could not be assumed by an AWS root account, so that unused role and its policy
were immediately deleted. The live test then used only exact canary bucket
names and mandatory tag checks before cleanup. No IAM role or access key was
retained.

## Live fresh-process resume result

The start process created and audited the two buckets, uploaded a conditional
object, proved that `If-None-Match: *` rejected overwrite, initiated a multipart
upload, uploaded exactly part 1, persisted a private local checkpoint, and
exited normally. A separate process loaded that checkpoint, confirmed the one
remote part, uploaded parts 2 and 3, completed the object, and performed full
and ranged downloads.

| Check | Verified result |
|---|---:|
| Synthetic payload | 17,825,915 bytes |
| Conditional object | 1,048,576 bytes |
| Multipart object | 16,777,339 bytes |
| Multipart parts | 3 |
| Remote parts before resume | 1 |
| Conditional overwrite | Rejected |
| Full-object SHA-256 roundtrip | Pass |
| Range-GET byte parity | Pass |
| Initial and final security audits | Pass |
| Data bucket after cleanup | Deleted |
| Audit bucket after cleanup | Deleted |
| Temporary IAM role after cleanup | Absent |
| Conservative canary estimate | $0.003586 |
| Canary hard cap | $0.10 |

A post-cleanup account query returned no bucket whose name begins with
`lumi-corpus-canary-` or `lumi-corpus-audit-`.

## Evidence

Canonical evidence is outside Git at
`lumi-data/aws-s3-provider-v0.1/`. The provider report reproduces from the
pinned contracts, implementation, and prices; the canary verifier checks its
manifest, plan fingerprint, implementation hash, authorization, security,
fresh-process resume, payload cap, and cleanup invariants.

| Artifact | SHA-256 |
|---|---|
| `provider-report.json` | `7589d19875f9acf28a9782c45259fbad8b45588f03dc5322d2df63dd04cc76a5` |
| `provider-manifest.sha256` | `5855d76a90ea6e38af3b20028993ac9af47c070bc421d1d828d84979944bd174` |
| `canary/report.json` | `85df6a7677fdf6f87523424462828a76255cf96a4a4e5ea2ffc07d00d9c640de` |
| `canary/manifest.sha256` | `0636a9285a38b80d33671e0a42088c2331359687593cd16261326be925cb0c1f` |

The provider plan fingerprint is
`774bd1ff4332207218f7a490791d102507b1ec6125493e972cd6b38f238e37a1`.
The full repository suite passes 188 tests; Ruff and formatting also pass.

```bash
PYTHONPATH=src python scripts/verify_aws_s3_provider.py \
  --report "$LUMI_DATA/aws-s3-provider-v0.1/provider-report.json" \
  --manifest "$LUMI_DATA/aws-s3-provider-v0.1/provider-manifest.sha256"

PYTHONPATH=src python scripts/verify_aws_s3_canary.py \
  --output "$LUMI_DATA/aws-s3-provider-v0.1/canary"
```

## Next gate

Provider pricing, private-bucket controls, and bounded I/O behavior are now
proven. Before production acquisition, Lumi still needs:

1. a complete source-file name/size/revision manifest for all 23 components;
2. an explicit owner decision for the CCCC source-level license basis; and
3. a separate authorization to create the production S3 namespace and begin
   adaptive acquisition.

The production acquisition will stop at the seven category floors totaling
120B clean candidate tokens; 150B remains only a hard failure ceiling.
