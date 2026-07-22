# Lumi Final 80B Corpus Gate v0.1

Status: **the deterministic, non-authorizing 120B-to-80B selection and shard
contract passes local validation. No Glue selection, shard publication, GPU, or
model training has been launched or authorized.**

## Why this gate exists

The verified global-dedup result is a quality-filtered candidate pool, not a
training corpus. It contains `120,000,036,715` tokens so each category has a
50% safety buffer around the intended 80B mixture. Training all 120B tokens
would silently change the agreed mixture and compute budget.

This gate deterministically chooses the final mixture, preserves the existing
immutable prefix, and defines the evidence required before any selected object
can become a training shard.

The permanent held-out exclusion is not a new extraction step. Mapping already
applied the sealed 35,889-entry training-exclusion ledger with SHA-256
`709027bb95f3e0f956b22a79d4b29f21490ef0b360ba10f2faca7d5db3d06c62`.
The final plan binds that exact ledger and the benchmark-signature registry, so
selection fails closed if either changes.

## Verified input boundary

The contract in `configs/final-corpus-80b-v0.1.json` pins:

- production dedup result canonical SHA-256
  `dbfbf9fe258ec97d5bd844c165157d41d328d0571cbbada5fd4d67750c46b742`;
- accepted-candidate inventory: `68,671,669` documents,
  `119,762,267,075` tokens, 2,772 Parquet objects, and
  `507,737,434,761` bytes;
- immutable prefix: `38,349` documents and `237,769,640` tokens;
- frozen 65K tokenizer SHA-256
  `1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee`;
- the exact data-mix, filter policy, benchmark registry, and permanent
  training-exclusion ledger files.

The plan accepts only the verified production result with all seven 120B
category floors complete and zero deficit. It also binds the exact release,
authorization, input-index, and map-checkpoint hashes from that run.

## Final mixture

| Category | Final target | Preserved prefix | New selection target |
|---|---:|---:|---:|
| Science and education | 24B | 237,769,640 | 23,762,230,360 |
| Code | 16B | 0 | 16B |
| Reference | 12B | 0 | 12B |
| Questions and discussion | 12B | 0 | 12B |
| Open web | 8B | 0 | 8B |
| Books | 4B | 0 | 4B |
| Government, legal, and patents | 4B | 0 | 4B |
| **Total** | **80B** | **237,769,640** | **79,762,230,360** |

## Deterministic selection

Within each category, every accepted row receives this cross-runtime rank:

```text
sha256(seed_sha256 NUL category NUL component NUL document_id NUL content_sha256)
```

Rows are sorted by that rank and the complete deterministic tie-break key. The
first document-boundary prefix reaching the category target is retained. Hash
ranking prevents source-file order from deciding which already-filtered,
deduplicated documents survive. It does not replace the upstream quality
filters.

Each source segment is capped at 131,072 UTF-8 bytes, so each category may
overshoot by at most 131,072 tokens at its last document boundary. Therefore:

- selected material must contain at least `80,000,000,000` tokens;
- the absolute selected-material maximum is `80,000,917,504` tokens;
- the trainer's consumption boundary remains exactly `80,000,000,000` tokens.

The selection result must record the full cutoff tuple, per-category available
and selected counts, content-addressed S3 inventories, exact input bindings,
and a canonical result fingerprint.

## Packed-shard contract

The output namespace is fixed to
`s3://lumi-corpus-prod-27777c02b29b76efcac3/lumi-corpus/v0.1/final-80b/`.
The plan keeps the three existing prefix artifacts and expects 160 new
approximately 500M-token uint16 shards, for 163 manifest entries in total.

The shard validator requires:

- contiguous shard sequence numbers and unique object keys/hashes;
- exact per-category and aggregate document/token conservation;
- exactly two bytes per token;
- no shard above 500M tokens plus the document-boundary cap;
- exact binding to the selected-result and plan fingerprints.

Selected Parquet, the selection ledger, and packed shards remain separate
immutable products. An object is not training-ready merely because it exists
in S3; the final selection result and shard manifest must both validate.

## Local evidence

The planning CLI is `scripts/plan_final_corpus_80b.py`. It writes canonical
`plan.json` plus `manifest.sha256` only into an empty destination. The focused
tests cover exact 120B-to-80B accounting, authorization/input tampering,
cross-runtime rank stability, category overshoot, S3 boundaries, shard byte
conservation, duplicate objects, and optional local Spark partition parity.

The planning status is
`ready_for_bounded_glue_canary_approval`. This wording is deliberate: local
contract tests have passed, but an AWS Glue canary is a separately approved
paid action.

## Next boundary

The next roadmap action is one bounded 8-DPU Glue selection-and-sharding
canary, at most 30 minutes and `$1.76`, after a fresh owner approval. It must
prove real Parquet schema compatibility, deterministic results across
partition counts, checkpoint/resume behavior, packed-byte parity, and cleanup.

Only after that can a separately approved production selection/shard run be
prepared. A 4B pilot and the full 4B/80B training remain later, independent
approval gates.
