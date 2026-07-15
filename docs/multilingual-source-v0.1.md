# Lumi multilingual tokenizer source v0.1

Status: **source selected, pinned, and source-only plus combined canaries passed**

## Decision

Lumi's representative tokenizer sample reserves 1% of accepted UTF-8 text
(60MB of the 6GB target) for Bengali. The remaining 99% preserves the existing
English-first 23-source proportions. This decision applies only to tokenizer
training; it does not add Bengali to the candidate 80B pretraining mixture.

The selected source is the official `wikimedia/wikipedia` dataset's
`20231101.bn` configuration:

- dataset revision: `3e1f92c331f318af862b87e2319ed5dc26d80f5d`;
- file: `20231101.bn/train-00000-of-00002.parquet`;
- compressed bytes: `192390559`;
- LFS object SHA-256:
  `98a8189cf1d0e4b41d647b3d88bd6201e571e002344b5648496c6bfda402841b`;
- schema: `id`, `url`, `title`, `text`;
- pinned dataset-card license labels: `cc-by-sa-3.0` and `gfdl`.

Primary evidence:

- <https://huggingface.co/datasets/wikimedia/wikipedia/commit/3e1f92c331f318af862b87e2319ed5dc26d80f5d>
- <https://huggingface.co/datasets/wikimedia/wikipedia/tree/3e1f92c331f318af862b87e2319ed5dc26d80f5d/20231101.bn>
- <https://foundation.wikimedia.org/wiki/Legal:Wikimedia_Developer_App_Guidelines>

## Provenance and license handling

Every accepted segment retains its upstream article ID, title, and page URL,
plus the pinned snapshot, dataset revision, file SHA-256, and source-level
license basis. Page URLs provide the attribution/history path required for a
later distribution review. The raw Parquet file and generated sample remain
outside Git.

The dataset card supplies a source-level basis, not a per-row legal opinion.
Before distributing tokenizer artifacts or model weights, review attribution,
ShareAlike, and any imported-text exceptions. The isolated 1% category and
complete provenance make the source removable if that review changes the
decision.

## Quality policy

English remains the default accepted language. `bn` is accepted only when the
record belongs to `multilingual_reference`; Bengali text appearing in other
categories is still rejected. Every segment also passes secret detection,
contextual PII redaction, benchmark decontamination, exact deduplication, and
disk-backed MinHash near-deduplication.

## Source-only canary

The 2026-07-15 bounded network run accepted 1,076,043 UTF-8 text bytes in 20
documents. All 20 were labeled `bn`; no rejection was recorded. Fresh
validation passed from line 0, cached validation resumed at line 20, and a
same-config `--resume` returned without another network read. The sample
SHA-256 is
`a3ce570bfa96c25e033977f2abeb8b765431fcf80a6e183693310c252b11c2c8`.

Parquet footer inspection used 272,391 range-read bytes. The file contains
71,535 rows in 72 row groups; the first row group contains 1,000 rows and has a
8,232,146-byte compressed footprint (22,607,611 bytes uncompressed). That
row-group boundary explains why the 1MB accepted-text canary used roughly
8.5MB of payload rather than downloading the full 192MB shard.

## Combined canary

The 2026-07-15 combined 24-source run accepted 25,637,516 UTF-8 text bytes in
5,014 documents. Every source met its quota. The Bengali category contributed
256,799 bytes in five `bn` documents against a 250,000-byte target. Fresh
validation passed from line 0, cached validation resumed at line 5,014, and a
completed same-config `--resume` required no further source read. The sample
SHA-256 is
`6b28052c9479844f93cf9d7d310a36db5b0931eab02f6ce0c1142878f6d09888`.

## Next gate

Build and fully validate the resumable 6GB representative tokenizer sample,
then use that one finalized artifact for the 32K-versus-65K tokenizer bakeoff.
