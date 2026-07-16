# Lumi-1B Scaling Data Gate v0.1

Status: **canonical source index, 16M canary, and aligned 1.4B pack complete;
independent validation passes and no paid GPU launch is approved**

## Decision

The 1B rehearsal may not consume the existing 6GB sample directly. It first
passes a target-independent source-index gate that proves exact tokenizer
capacity and permanently excludes every held-out document ID, content hash,
and near-duplicate family. A separate target-specific pack then selects a
deterministic hash prefix within each of the 24 source strata.

This is local data work. It does not authorize Lambda, AWS, a paid GPU, the 1B
training run, or the 4B run.

## Why two phases

Exact hash-ranked selection cannot know a source quota until every eligible
document has a token count. Repeating that full 1,128,089-document scan for a
canary and the final target would waste hours. The implementation therefore
separates:

1. `source-index`: scan and batch-tokenize the sealed source once; record each
   byte offset, stable identity, source stratum, deterministic rank, exact token
   count, and held-out status in SQLite;
2. `pack`: allocate a requested target in proportion to the exact eligible
   token capacity, select the lowest ranks per stratum, and stream only those
   documents into fixed 4,096-token shards.

The source index is reusable by both the 16,777,216-token canary and the final
1,399,996,416-token artifact.

## Frozen evidence

The canonical CLI fails closed unless these values still match the 1B scaling
plan:

| Evidence | Frozen value |
| --- | ---: |
| Source documents | 1,128,089 |
| Source sample SHA-256 | `80bea9f9634a8daca542b2ffd0aad1c8c994a7cbeef909a5e529b6272671102a` |
| Observed 65K text tokens | 1,476,228,500 |
| Held-out documents | 3,988 |
| Held-out selection-ledger SHA-256 | `5dc1fe84fd347cbaf7cd8f2a0a51125ab1272ec594d649762a50b66661ebdb97` |
| Source strata | 24 |
| Selection seed | 20260717 |
| Sequence length | 4,096 |

The index independently reproduces the global and per-category tokenizer
counts. Exact held-out rows and any additional document in one of their
near-duplicate families are marked ineligible before allocation.

## Selection rule

For each eligible document, the rank is:

```text
SHA256(seed + NUL + document_id + NUL + content_sha256)
```

The target is allocated across strata using exact eligible-token capacity and
integer largest remainders. Within each stratum, documents are selected in
`(rank, document_id)` order until its quota is full. Only the threshold
document may be truncated. Packing follows sealed input-offset order for
sequential disk access; this changes output order, not the selected hash
prefix.

The independent validator proves:

- selected rows contain no held-out document or held-out near-duplicate family;
- every stratum's selected set is exactly its lowest-rank prefix;
- all non-threshold rows use their full indexed token count;
- only a threshold row may be truncated;
- proof, SQLite state, shard sizes, shard hashes, totals, and sequence alignment
  are unchanged.

## Resume boundary

Both phases use SQLite WAL with `synchronous=FULL`.

- Index state commits the input byte offset only with the corresponding rows
  and counts.
- Packing flushes and fsyncs shard bytes before committing its input offset and
  token totals.
- Resume truncates uncommitted shard bytes back to the durable token boundary.
- A completed database is checkpointed out of WAL, switched to the stable
  delete journal, SHA-256 sealed, and opened read-only by validators.

Disconnecting Wi-Fi does not affect either phase. The canonical local process
is run through macOS `launchd` and `caffeinate`; a crash or restart resumes from
the last durable checkpoint.

## Canary geometry

`configs/scaling-data-1b-v0.1.json` pins:

| Field | Value |
| --- | ---: |
| Target tokens | 16,777,216 |
| Sequences | 4,096 |
| Sequence length | 4,096 |
| Sequences per shard | 256 |
| Shards | 16 |
| Raw uint16 shard payload | 33,554,432 bytes (32 MiB) |

The final aligned target's raw uint16 shard payload is exactly 2,799,992,832
bytes, about 2.61 GiB. It uses 8,192 sequences per shard and produces 42 shards
rather than more than a thousand tiny files. The source index and compact
selection state remain local audit artifacts outside Git.

## Verified canonical results

The canonical local run completed on 2026-07-16. The frozen data-contract
config SHA-256 is
`27652c4bd741669af0f8e69adff47fd12294643bb1828489a905c9d08f1f405f`.

### Reusable source index

| Field | Verified value |
| --- | ---: |
| Source documents | 1,128,089 |
| Source text tokens | 1,476,228,500 |
| Eligible documents | 1,124,101 |
| Eligible tokens | 1,472,219,479 |
| Exact held-out documents excluded | 3,988 |
| Held-out-family documents excluded | 3,988 |
| Source strata | 24 |
| SQLite bytes | 903,491,584 |
| Build fingerprint | `23ce71e3c27c7cb94dffcde75dcfa0426214ed37faeaa1c0ef7f3f47209821f6` |
| Manifest SHA-256 | `cefada43aab7b54430197852440c7270cf7ef1a663240ad4d8d9b61b96c7b473` |
| SQLite SHA-256 | `bfce6842fd133f2f10378d4a66e4490a4d907fd988bece53a24ef363b169bf1a` |

The index has 72,219,479 eligible tokens above the requested 1.4B target. Its
independent validator reproduced the frozen global count, all category counts,
the exact source-document count, and the exact 3,988-document held-out ledger.
The sealed directory contains only the manifest and SQLite database; no WAL or
shared-memory checkpoint residue remains.

### Canary and final pack

| Field | 16M canary | Final pack |
| --- | ---: | ---: |
| Requested tokens | 16,777,216 | 1,400,000,000 |
| Effective aligned tokens | 16,777,216 | 1,399,996,416 |
| Selected and packed documents | 13,041 | 1,069,214 |
| Sequences | 4,096 | 341,796 |
| Shards | 16 | 42 |
| Raw shard bytes | 33,554,432 | 2,799,992,832 |
| Source strata | 24 | 24 |
| ID/content/near-family overlaps | 0 / 0 / 0 | 0 / 0 / 0 |
| Build fingerprint | `26e37b3ec6e29ae1ec0a0f3e121c0477d43584cab028d648f4a3bc1302e36325` | `0bce4d9e4a51de1b831b0226e0052ea33130733eae2bb61992c6eacaf203e0c8` |
| Manifest SHA-256 | `90cb5bde047d94dde15579fec474e009e79a5b25a369f9038e8a02e1aa6bfbba` | `9228fa610d4a985d05e15057e8ef12619a14cbedbb5a6a5ab513d3aeca2832c8` |
| Selection-proof SHA-256 | `517d71c8fe04e730bc925feccf229d9f9f44c94a1ed741908ad87e6322b1a6cd` | `e08b9d51371774f8c13afc2692b4f7fff55f3bb6d1cf66ee85497ce08d4766a7` |
| Selection-state SHA-256 | `6c82ad7c933e802cacba558b9c433b2fd217b9002d592cab06caf12df362c96c` | `1791680e51c435472bf64fefd2b6c931ec2c08f0faf721daa97380e87133722c` |

The source-index run was deliberately stopped after a durable checkpoint and
resumed in a fresh process without discarding state. Unit tests also force
mid-index and mid-pack interruptions. After sealing, both canonical packs were
invoked again with `--resume`; each returned the same completed result without
rebuilding data. Independent validation then re-read the proof, selection
state, and every shard hash successfully.

## Canonical commands

Preflight the evidence without building data:

```bash
python scripts/build_scaling_data.py preflight \
  --plan configs/scaling-pilot-1b-v0.1.json \
  --input "$LUMI_DATA/tokenizer-sample-v0.3/sample.jsonl" \
  --tokenizer "$LUMI_DATA/tokenizer-candidates-v0.2/lumi-tokenizer-65k" \
  --source-manifest "$LUMI_DATA/tokenizer-sample-v0.3/manifest.json" \
  --heldout-manifest "$LUMI_DATA/evaluation-pilot-v0.1/heldout/manifest.json" \
  --heldout-selection-ledger \
    "$LUMI_DATA/evaluation-pilot-v0.1/heldout/selection-ledger.jsonl" \
  --token-count-report \
    "$LUMI_DATA/tokenizer-candidates-v0.2/comparison-v0.2.json"
```

Build or resume the reusable index:

```bash
python scripts/build_scaling_data.py index \
  --plan configs/scaling-pilot-1b-v0.1.json \
  --input "$LUMI_DATA/tokenizer-sample-v0.3/sample.jsonl" \
  --tokenizer "$LUMI_DATA/tokenizer-candidates-v0.2/lumi-tokenizer-65k" \
  --source-manifest "$LUMI_DATA/tokenizer-sample-v0.3/manifest.json" \
  --heldout-manifest "$LUMI_DATA/evaluation-pilot-v0.1/heldout/manifest.json" \
  --heldout-selection-ledger \
    "$LUMI_DATA/evaluation-pilot-v0.1/heldout/selection-ledger.jsonl" \
  --token-count-report \
    "$LUMI_DATA/tokenizer-candidates-v0.2/comparison-v0.2.json" \
  --output "$LUMI_DATA/scaling-data-1b-v0.1/source-index" \
  --resume
```

Inspect one concise durable checkpoint:

```bash
python scripts/build_scaling_data.py status \
  --output "$LUMI_DATA/scaling-data-1b-v0.1/source-index"
```

After the index is sealed, build or resume the canary:

```bash
python scripts/build_scaling_data.py pack \
  --index-manifest \
    "$LUMI_DATA/scaling-data-1b-v0.1/source-index/manifest.json" \
  --input "$LUMI_DATA/tokenizer-sample-v0.3/sample.jsonl" \
  --tokenizer "$LUMI_DATA/tokenizer-candidates-v0.2/lumi-tokenizer-65k" \
  --output "$LUMI_DATA/scaling-data-1b-v0.1/canary-16m" \
  --target-tokens 16777216 \
  --sequence-length 4096 \
  --sequences-per-shard 256 \
  --purpose canary \
  --resume
```

Validate both the selected hash prefixes and packed bytes:

```bash
python scripts/build_scaling_data.py validate \
  --index-manifest \
    "$LUMI_DATA/scaling-data-1b-v0.1/source-index/manifest.json" \
  --packed-manifest \
    "$LUMI_DATA/scaling-data-1b-v0.1/canary-16m/manifest.json"
```

Only after that command passes, materialize the final target:

```bash
python scripts/build_scaling_data.py pack \
  --index-manifest \
    "$LUMI_DATA/scaling-data-1b-v0.1/source-index/manifest.json" \
  --input "$LUMI_DATA/tokenizer-sample-v0.3/sample.jsonl" \
  --tokenizer "$LUMI_DATA/tokenizer-candidates-v0.2/lumi-tokenizer-65k" \
  --output "$LUMI_DATA/scaling-data-1b-v0.1/final-1.4b" \
  --target-tokens 1400000000 \
  --sequence-length 4096 \
  --sequences-per-shard 8192 \
  --purpose final \
  --resume
```

Validate the sealed final artifact independently. Re-running the preceding
`pack --resume` command must then return the cached completed result without
writing new shards:

```bash
python scripts/build_scaling_data.py validate \
  --index-manifest \
    "$LUMI_DATA/scaling-data-1b-v0.1/source-index/manifest.json" \
  --packed-manifest \
    "$LUMI_DATA/scaling-data-1b-v0.1/final-1.4b/manifest.json"
```

## Go/no-go rule

The canonical source index, 16M canary, and final aligned pack now satisfy the
local data gate: all 24 strata have nonzero exact quotas, leakage is zero, and
resume and shard-integrity checks pass. This still does not launch an H100. The
schema-v2 local cloud preflight pinned to this exact manifest now also passes.
The next gate is current provider selection, transfer, and remote CUDA preflight,
followed by a one-step 1B smoke run, fresh-process resume, and only then the
100-step measured calibration.
