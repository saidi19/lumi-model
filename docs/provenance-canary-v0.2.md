# Lumi provenance canary v0.2

Status: **passed as a metadata-retention gate; not approved for model training**

Historical note: this artifact predates contextual PII v4 and is not eligible
for tokenizer or model input. The audited replacement is documented in
[`development-sample-v0.4.md`](development-sample-v0.4.md).

Run date: 2026-07-15. GPU usage: none.

## Reproducibility anchor

- Config: [`configs/provenance-canary-v0.2.json`](../configs/provenance-canary-v0.2.json)
- Source type: seven individually pinned `common-pile/*_filtered` repositories
- Accepted documents: 1,823
- Accepted text: 10,248,171 UTF-8 bytes
- Materialized JSONL size: 12,388,779 bytes
- Output SHA-256: `07e83fadf29f8cce32e0718018424381108af6236cacd228d24e459669562d4a`

Each accepted segment retains its component repository and revision, source
file/blob/line, original upstream ID and source, Common Pile provenance, URL,
and either document license metadata or the explicit CCCC source-level license
basis. These fields are provenance ledger data; tokenizer/model loaders consume
only the `text` field.

## Domain result

| Domain | Documents | Target bytes | Accepted bytes |
|---|---:|---:|---:|
| Science and education | 409 | 3,000,000 | 3,015,420 |
| Code | 496 | 2,000,000 | 2,005,473 |
| Reference | 243 | 1,500,000 | 1,507,702 |
| Questions and discussion | 481 | 1,500,000 | 1,502,048 |
| Open web | 169 | 1,000,000 | 1,011,550 |
| Books | 5 | 500,000 | 620,469 |
| Government, legal, patents | 20 | 500,000 | 585,509 |

Every domain passed. The book and legal slices overshot more because complete
documents or bounded segments are retained rather than cutting text exactly at
the byte quota.

## Provenance and safety result

- Required retained provenance was present in all 1,823 accepted records.
- One Stack v2 record with a missing required license value was rejected.
- One document matching a private-key pattern and 23 short documents were
  rejected.
- The accepted text had 21 email and 68 phone-like values redacted.
- Independent validation recomputed output hashes, deterministic IDs, source
  identities, category totals, content deduplication, normalization, safety
  patterns, and required provenance fields.

## What this changes

The v0.1 consolidated canary proved streaming and tokenizer plumbing but could
not retain original IDs or licenses because its rows contain only text. This
v0.2 run closes that engineering gap. Lumi development/final samples will use
the individual filtered repositories and preserve a removal/license mapping.

It does not settle legal compatibility for every license value, prove full
corpus metadata completeness, add language identification, perform near
deduplication, or decontaminate benchmarks. Those remain hard gates before the
1GB-2GB development sample can be promoted to a 150M-model training pilot.
