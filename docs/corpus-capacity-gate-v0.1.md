# Lumi corpus source and capacity gate v0.1

Status: **official pinned metadata passes the planning estimate and the bounded
retention canary passes; object-storage design is next; bulk corpus
materialization, paid compute, and 4B training remain unapproved**

## Purpose

An 80B-token target is not evidence that 80B clean, unique tokens exist in the
selected mix. This gate inventories the exact upstream revisions, checks their
published sizes, converts those sizes with the frozen Lumi tokenizer's measured
category fertility, and fails closed before any large download.

The upstream basis is the official [Common Pile v0.1 filtered
collection](https://huggingface.co/collections/common-pile/common-pile-v01-filtered-data)
and its [dataset paper](https://arxiv.org/abs/2506.05209). Every repository is
pinned to the revision already recorded in
[`component-sources-v0.1.json`](../configs/component-sources-v0.1.json); no
floating `main` revision is accepted.

## Verified planning inventory

On 2026-07-19, the Hugging Face tree API and the dataset card at each exact
revision reported:

| Measurement | Result |
|---|---:|
| Selected filtered components | 23 |
| Published documents | 188,143,662 |
| Published UTF-8 text | 1,734,967,000,000 bytes (1.735 TB decimal) |
| Compressed repository payload | 499,817,285,384 bytes (499.82 GB decimal) |
| Frozen-tokenizer planning estimate | 405,373,536,322 tokens |

The source-level snapshot is
[`common-pile-capacity-snapshot-v0.1.json`](../configs/common-pile-capacity-snapshot-v0.1.json).
Its file counts exactly match the existing inventory at every pinned revision.

The token estimate uses the exact category-specific byte/token ratios measured
on the sealed 6,000,636,724-byte tokenizer sample with the retained 65K
tokenizer. It is more useful than a generic four-bytes-per-token guess, but it
is still only a planning estimate.

## Category safety floor

The final target remains 80B tokens. Before bulk materialization, each category
must show at least 1.5 times its final target after the frozen tokenizer and
Lumi filters. These category floors total 120B; the 140B planning target leaves
20B flexible capacity. The 130B-150B planning envelope is a safety buffer, not
a mandatory amount to download or store on the Mac.

| Category | Final target | Minimum candidate | Published-token estimate | Estimate headroom |
|---|---:|---:|---:|---:|
| Books | 4B | 6B | 15.685B | 161.41% |
| Code | 16B | 24B | 69.599B | 190.00% |
| Government/legal/patents | 4B | 6B | 170.351B | 2,739.18% |
| Open web | 8B | 12B | 13.394B | **11.62%** |
| Questions/discussion | 12B | 18B | 23.982B | 33.23% |
| Reference | 12B | 18B | 26.005B | 44.47% |
| Science/education | 24B | 36B | 86.357B | 139.88% |

All planning estimates pass, but open web is tight. CCCC therefore goes first
in the bounded acquisition canaries, followed by questions/discussion and
reference. If their measured post-filter retention misses the floor, the mix or
source inventory must change before any bulk download.

The bounded follow-up has now passed. Its first-shard measurements project
13.210B open-web tokens, 21.863B questions/discussion tokens, and 25.153B
reference tokens after the frozen tokenizer and current quality gates. See
[`corpus-retention-canary-v0.1.md`](corpus-retention-canary-v0.1.md). These
figures replace planning fertility for the three tight categories, while the
remaining categories retain their metadata-based planning estimates.

## What this does not prove

Published UTF-8 size does not prove:

- token count after Lumi's language, quality, PII, secret, and benchmark filters;
- unique capacity after exact and cross-component near-deduplication;
- acceptance of every document's license value or provenance;
- readiness of an 80B packed corpus; or
- permission to start paid compute or model training.

The current bounded provenance audit sampled 44,061 rows; it is not a complete
license-value audit. CCCC also retains its documented source-level-license
exception and requires an owner decision before bulk use.

## Decision and next gate

The machine-readable contract is
[`corpus-capacity-gate-v0.1.json`](../configs/corpus-capacity-gate-v0.1.json).
It permits only bounded metadata and source-canary work. It explicitly records
`bulk_corpus_materialization=false`, `paid_compute=false`, and
`full_4b_training=false`.

The bounded streaming and retention measurements are complete. The next gate
will design versioned private object storage, immutable source and shard
manifests, resumable workers, a deletion ledger, lifecycle/cost caps, and
adaptive stop conditions. It will not create a bucket, download bulk data, or
start training without separate owner approval.

Run the deterministic planning report locally with:

```bash
PYTHONPATH=src python scripts/plan_corpus_capacity.py \
  --output /path/to/corpus-capacity-report.json \
  --manifest /path/to/manifest.sha256
```

The canonical local evidence is stored outside Git at
`lumi-data/corpus-capacity-gate-v0.1/`. Its 15,149-byte `report.json` has
SHA-256 `d5cbf20cc46d55b600bbd109235c85c01e5985925b3d278194283eb92f1f04b0`;
an independent `shasum -a 256 -c manifest.sha256` passes. The follow-up
retention evidence and hashes are recorded in its own decision document.

No corpus payload or model checkpoint belongs in Git.
