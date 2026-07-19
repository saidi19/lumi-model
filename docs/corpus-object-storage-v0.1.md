# Lumi corpus object-storage and adaptive-acquisition design v0.1

Status: **local design and simulation pass; follow-up AWS S3 selection, pricing,
and bounded I/O canary also pass; production cloud resources, bulk corpus
materialization, paid compute, and 4B training remain unapproved**

## Purpose and authorization boundary

This gate converts the 80B corpus capacity and retention results into a
provider-neutral storage and acquisition contract. It prevents a fixed
130B--150B download, Mac disk exhaustion, silent shard replacement, and data
that cannot later be traced or removed.

The contract in
[`corpus-object-storage-v0.1.json`](../configs/corpus-object-storage-v0.1.json)
authorizes local design and metadata-only simulation only. It explicitly sets:

- `create_cloud_resources=false`;
- `bulk_corpus_materialization=false`;
- `transfer_bulk_source_data=false`;
- `paid_compute=false`; and
- `full_4b_training=false`.

No bucket, cloud object, paid server, corpus payload, or training process was
created by this gate.

## Adaptive acquisition rule

The final training target remains 80B tokens. Acquisition stops when every
category has reached its independently sealed 1.5-times candidate floor and
all shards pass integrity, provenance, deduplication, and removal-ledger gates.
The floors total 120B candidate tokens. Repetition is forbidden as a way to
fill a quota; 150B is only a fail-closed hard maximum, not a download target.

| Category | Final target | Candidate floor | Verified planning capacity |
|---|---:|---:|---:|
| Books | 4B | 6B | 15.685B |
| Code | 16B | 24B | 69.599B |
| Government/legal/patents | 4B | 6B | 170.351B |
| Open web | 8B | 12B | 13.210B |
| Questions/discussion | 12B | 18B | 21.863B |
| Reference | 12B | 18B | 25.153B |
| Science/education | 24B | 36B | 86.357B |

The three tightest categories use measured retention-canary projections; the
others use pinned metadata plus frozen-tokenizer fertility. A worker may cross
a category floor by at most one 100M-token clean shard.

## Provider-neutral object contract

All keys are write-once and content-addressed beneath `lumi-corpus/v0.1`.
Clean candidate shards target 100M tokens; final packed uint16 shards target
500M tokens. At the 120B candidate and 80B final targets, the plan estimates
1,200 clean shards and 160 final packed shards.

The selected provider must support all of these capabilities before a canary:

- denied public access, TLS transport, and server-side encryption;
- versioning, conditional create, SHA-256 object metadata, and access logs;
- abortable multipart uploads, lifecycle rules, and range GET.

Workers use 128MiB multipart parts, at most one in-flight shard each, and a
20GB local scratch hard cap. Raw source payloads are streamed and never
persisted in object storage. Manifests must not contain raw or accepted corpus
text, credentials, or secret material.

The content-addressed layout separates clean text, candidate token IDs, final
packed generations, provenance indexes, immutable control manifests, and
removal events. A manifest becomes visible only after its object hash and all
referenced objects pass verification.

## Storage forecast

This is a future cloud logical-capacity forecast, not Mac allocation and not
current disk use.

| Logical artifact | Forecast |
|---|---:|
| Clean 120B candidate text, compressed | 360GB |
| Candidate uint16 token IDs | 240GB |
| Final 80B packed uint16 tokens | 160GB |
| Provenance/dedup indexes | 60GB |
| Control-plane reserve | 10GB |
| Persistent subtotal | 830GB |
| Transient-version allowance | 207.5GB |
| Forecast peak | 1.0375TB |
| Hard logical cap | 2TB |

The uncompressed clean-text reference is 720GB, but it is not an additional
persistent object tier. Forecast peak has 48.125% headroom below the 2TB hard
cap. Raw source-payload storage is zero.

## Resume, deletion, and fail-closed behavior

The metadata-only SQLite simulation binds state to the complete plan
fingerprint, publishes shard commits atomically, makes identical retries
idempotent, detects conflicting retries, and resumes from durable state.

Removal events are immutable and content-addressed. Their lookup supports
dataset ID, pinned revision, source file, upstream document ID, and content
SHA-256 across 4,096 partitions. A pending removal blocks affected shards and
the next training manifest until those shards are rebuilt.

Acquisition halts on a corrupt shard, pending removal, storage-cap breach,
150B hard-maximum breach, or category overshoot beyond one clean shard.

## Verified local result

The deterministic planner and independent verifier pass. The canonical local
evidence contains only a 16,750-byte JSON report and its manifest:

| Item | Verified value |
|---|---|
| Implementation commit | `9c377296c608b43ad06f27b4bfb244844a8bc033` |
| Plan fingerprint | `49daec86e08ef25074cc7c8d746d7ef9081b56f7a48329e3f4e7d955f9d22b62` |
| `report.json` SHA-256 | `2dc159b5695770d7ad89007f084b69b02e6e1e2c3e9b7a55f3393c2295e608eb` |
| `manifest.sha256` SHA-256 | `746cefb5c32665a790fa84de563a1ecbec68ba13d4ed9108d7641c926452e119` |
| Provider readiness | `blocked_unselected_unpriced_unapproved` |

The simulation covers atomic commit, retry, conflict, resume, removal blocking,
manifest validation, and adaptive progress without writing corpus payload.
The full repository test suite and Ruff pass.

```bash
PYTHONPATH=src python scripts/plan_corpus_storage.py \
  --output "$LUMI_DATA/corpus-object-storage-plan-v0.1/report.json" \
  --manifest "$LUMI_DATA/corpus-object-storage-plan-v0.1/manifest.sha256"

PYTHONPATH=src python scripts/verify_corpus_storage_plan.py \
  --report "$LUMI_DATA/corpus-object-storage-plan-v0.1/report.json" \
  --manifest "$LUMI_DATA/corpus-object-storage-plan-v0.1/manifest.sha256"
```

## Completed provider follow-up

AWS S3 `us-east-1` is now selected. Its exact pricing snapshot, $100 S3-only
monthly cap, private/encrypted/versioned bucket policy, conditional writes,
fresh-process multipart resume, range GET, hashes, access logging, and complete
cleanup all pass; see
[`aws-s3-provider-canary-v0.1.md`](aws-s3-provider-canary-v0.1.md).

## Remaining gate

Provider selection, pricing, security controls, the bounded I/O canary, and the
complete 23-component source-file manifest are now closed. See
[`source-file-manifest-v0.1.md`](source-file-manifest-v0.1.md). Before production
acquisition, these decisions remain:

1. record the owner's decision for the CCCC source-level license basis; and
2. separately approve the production S3 namespace and adaptive acquisition.

The successful canary buckets were deleted. Production bucket creation and
bulk acquisition remain separate explicit owner decisions.
