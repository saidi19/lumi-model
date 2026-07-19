# Lumi bounded corpus-retention canary v0.1

Status: **the three tightest corpus categories pass their measured capacity
floors and the follow-up local object-storage design passes; provider pricing,
bulk corpus materialization, paid compute, and 4B training remain unapproved**

## Purpose and boundary

The metadata capacity gate estimated that the 23 pinned Common Pile components
contain enough source text for the candidate 80B-token mix. This follow-up asks
a narrower question: after the actual frozen Lumi tokenizer and quality gates,
do open web, questions/discussion, and reference still clear their per-category
candidate floors?

The machine-readable contract is
[`corpus-retention-canary-v0.1.json`](../configs/corpus-retention-canary-v0.1.json).
It hard-caps configured input at 192,000,000 raw UTF-8 text bytes, with no more
than 64,000,000 bytes in any category or source. A final whole document may
cross a target boundary, which explains the measured 192,528,033 bytes. The
runner also enforces a 2 GiB artifact ceiling and a 20 GB free-space floor.

Raw and accepted text are never written to disk. The durable SQLite state keeps
only cursors, counts, exact hashes, MinHash signatures, source pins, rejection
counters, and PII-redaction counters. The run is atomic per source record,
disconnect-resumable, and bound to hashes of its contract, configuration, and
implementation files. Its authorization explicitly records:

- `bulk_corpus_materialization=false`;
- `paid_compute=false`; and
- `full_4b_training=false`.

## Verified local run

The canonical run used implementation commit `fe20ebe`, the retained 65K Lumi
tokenizer with `tokenizer.json` SHA-256
`1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee`, and
the six pinned first shards already present in the source plan. It ran locally
without a GPU or paid server from 10:00:37 to 10:10:24 UTC on 2026-07-19.

| Measurement | Verified result |
|---|---:|
| Raw source records | 47,192 |
| Raw UTF-8 text scanned | 192,528,033 bytes |
| Accepted documents/segments | 43,061 |
| Accepted UTF-8 text | 190,915,171 bytes |
| Accepted-text retention | 99.1623% |
| Frozen-tokenizer output | 45,981,609 tokens |
| Network retries | 0 |
| Sealed SQLite evidence | 194,260,992 bytes |

PII processing redacted 67 email and 563 phone matches before token counting.
The quality gate also recorded 728 near-duplicate rejections,
3 assigned-credential secret rejections, 3,374 short-text rejections,
123 disallowed-language decisions, and 40 low-confidence language decisions.
These are event counts, not necessarily one-to-one raw-document counts.

## Measured capacity decision

Projection multiplies each pinned source's published UTF-8 size by its measured
accepted-token yield per raw byte. It does not use a generic
four-bytes-per-token estimate.

| Category | Raw | Accepted | Projected | Floor | Headroom | Result |
|---|---:|---:|---:|---:|---:|---|
| Open web | 64.001 MB | 14.577M | 13.210B | 12B | 10.08% | Pass |
| Questions/discussion | 64.017 MB | 15.897M | 21.863B | 18B | 21.46% | Pass |
| Reference | 64.510 MB | 15.508M | 25.153B | 18B | 39.74% | Pass |

The source-level measurements are:

| Component | Raw text | Accepted text | Accepted tokens | Text retention | Projected tokens |
|---|---:|---:|---:|---:|---:|
| CCCC | 64.001 MB | 63.788 MB | 14.577M | 99.667% | 13.210B |
| Stack Exchange | 48.001 MB | 47.988 MB | 10.796M | 99.973% | 20.175B |
| Ubuntu IRC | 16.016 MB | 15.974 MB | 5.101M | 99.740% | 1.688B |
| Wikimedia | 32.000 MB | 31.884 MB | 7.545M | 99.635% | 13.534B |
| WikiTeam | 16.000 MB | 14.772 MB | 4.270M | 92.326% | 3.656B |
| Library of Congress | 16.509 MB | 16.509 MB | 3.693M | 99.997% | 7.963B |

The decision is `retention_canary=pass` and `capacity_projection=pass`. This
does not turn the 130B-150B planning envelope into a mandatory Mac download.
That envelope is a candidate safety buffer before selecting the final 80B mix;
the acquisition runner should stop adaptively once verified category quotas and
deduplication reserves are met.

## Evidence and independent verification

Canonical evidence lives outside Git at
`lumi-data/corpus-retention-canary-v0.1/`. It contains no corpus payload. The
sealed hashes are:

| Artifact | SHA-256 |
|---|---|
| `manifest.sha256` | `d124ae0be5b66308cc5663a1c0bbc9d331d3eb38447d253cc4432d8ae2b0eb2d` |
| `report.json` | `040b19ec8ef018447ade9baab1af03d5c63b4a063d9aa7fd4d4115754239c890` |
| `retention-state.sqlite3` | `3231cc63d01b5ded3efeffe452f1b48e3c02128a8d291629bfdd90ef91781a78` |

Both `shasum -a 256 -c manifest.sha256` and the independent verifier pass. The
verifier reproduces input and implementation hashes, checks SQLite integrity,
reconciles every report counter with durable state, confirms exact-hash and
MinHash index cardinality, recomputes projections, and rejects any authorization
that would permit bulk data, paid compute, or training.

```bash
PYTHONPATH=src python scripts/check_corpus_retention_canary.py \
  --output "$LUMI_DATA/corpus-retention-canary-v0.1"

PYTHONPATH=src python scripts/verify_corpus_retention_canary.py \
  --output "$LUMI_DATA/corpus-retention-canary-v0.1"
```

## What remains uncertain

These deterministic first-shard canaries are not random samples of the complete
repositories. Full-source duplicate rates can be higher, source order affects
cross-source deduplication, and the CCCC source-level licensing exception still
needs an explicit owner decision. The canary therefore does not prove complete
license-value compliance, final 80B packed-corpus readiness, or model quality.

The provider-neutral private object-storage and adaptive-acquisition design is
now complete; see
[`corpus-object-storage-v0.1.md`](corpus-object-storage-v0.1.md). It specifies
immutable source manifests, resumable streaming workers, staged clean shards,
a removal ledger, lifecycle/storage caps, and fail-closed stop conditions. The
next gate is provider selection and pinned pricing. Only a later, separately
owner-approved bounded I/O canary may create cloud storage; bulk source data is
still not authorized.
