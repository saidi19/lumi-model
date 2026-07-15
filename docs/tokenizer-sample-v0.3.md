# Lumi representative tokenizer sample v0.3

Status: **multi-shard implementation, every mandatory capacity gate, and the
25MB all-source canary passed; migration apply and 6GB resume pending**

## Why v0.3 exists

The v0.2 full-size run stopped safely in the `stackv2_edu` component. The logical
component target was 1,180,872,000 accepted UTF-8 text bytes, but the single
pinned `stack-edu-0000.json.gz` anchor exhausted at 1,101,302,254 bytes. The
sampler incorrectly treated one schema anchor as the complete capacity of a
multi-file dataset.

The on-disk sample itself is intact. A fresh scan found 2,883,516,648 accepted
text bytes and 417,740 documents; the last persisted progress checkpoint was 40
documents behind because the failure occurred between periodic checkpoints.

## Structural correction

v0.3 keeps quotas at the logical component level and lets each component stream
an ordered list of physical files:

- each physical file has its own pinned revision, content digest, compressed
  size, provenance counts, and manifest statistics;
- a component continues into the next file only when its current file exhausts
  before the component quota;
- exhausted physical files are persisted in run state and skipped on resume;
- post-write validation reconstructs both component-level and physical-file
  counts;
- required high-risk components must have a verified accepted-text lower bound
  at or above their full 6GB quota before any output directory is created.

The generic behavior is regression-tested with a two-file local component. This
is not a one-off Stack exception.

## Continuation pins

| Component | Primary | Pinned continuation | Selection |
|---|---|---|---|
| Stack v2 educational code | `stack-edu-0000.json.gz` | `stack-edu-0002.json.gz` | `0001` excluded after the upstream automated unsafe-file warning |
| WikiTeam | `wikiteam-0000.json.gz` | `wikiteam-0001.json.gz` | first ordered continuation |
| CCCC | `dolma-cccc-filtered-0000.json.gz` | `dolma-cccc-filtered-0001.json.gz` | first ordered continuation |

The exact revisions, Git/LFS digests, and sizes are recorded in
`configs/component-continuation-files-v0.1.json`. Upstream trees are pinned for
[Stack](https://huggingface.co/datasets/common-pile/stackv2_edu_filtered/tree/c354dbe88469a1153e97c6a63ac50591849654de),
[WikiTeam](https://huggingface.co/datasets/common-pile/wikiteam_filtered/tree/f4ed055b57763a8f12238824140914b9eb098cab),
and [CCCC](https://huggingface.co/datasets/common-pile/cccc_filtered/tree/03a3de5713a0bb23267d26724346508af0f25327).

## Capacity gate status

| Component | Full-plan target | Verified lower bound | Status |
|---|---:|---:|---|
| Stack v2 educational code | 1,180,872,000 | 1,201,305,852 across primary + continuation | Pass |
| WikiTeam | 222,750,000 | 250,000,118 | Pass |
| CCCC | 594,000,000 | 650,000,832 | Pass |

The Stack continuation audit accepted 100,003,598 bytes across 30,362 documents,
with zero network retries, then passed full post-write validation. Its sample
SHA-256 is
`6c3ba9bcc26550edc53cd3475d4ce9dbd5ecad0277a546920b993e182bc2154c`.

The WikiTeam audit accepted 250,000,118 bytes across 186,994 documents; its
sample SHA-256 is
`de839456c8ec44cb859a13ff411ea2b7342ece7e0ad27ffe77dc7bc4b3e25c67`.
The CCCC audit accepted 650,000,832 bytes across 143,973 documents; its sample
SHA-256 is
`afa46b265dac82e99d92ece17c7e8b8c20b3f3d8a93f118939cb2ff253c9f553`.
Both completed with zero network retries and passed full post-write validation.
The no-download 6GB quota preflight now passes. No GPU or Lambda credit was
used by these audits.

## v0.3 all-source canary

The 25MB v0.3 canary completed with 25,391,622 accepted text bytes, 4,859
documents, all 24 logical sources meeting quota, and zero network retries. Fresh
validation passed, cached validation resumed from line 4,859, and a completed
`--resume` returned without another source read.

Its SHA-256 is
`9eba9d35593c77d0579edfb5e740c106cd08a410e0d75c52678c1964fce1a32f`,
identical to the v0.2 canary. The data-selection semantics therefore remained
byte-for-byte stable while v0.3 added continuation and capacity metadata.

## Existing-sample migration

`scripts/migrate_tokenizer_sample_plan.py` performs a read-only dry run by
default. It checks the old config hash and run fingerprint, proves that source
weights and byte quotas are unchanged, scans every existing record under the
new physical-to-logical mapping, verifies the exact exhausted Stack capacity,
and hashes the entire sample.

The first dry run passed with sample SHA-256
`d08a8f717faa5cfb244356b3a2b34f4342c7cffd18df617440d223c2fcdc4dd2`.
It did not modify the sample or run state. The dry run must be repeated after the
final capacity report is committed; only then may `--apply` create a backup and
update `run-state.json`.

## Remaining gates

1. Repeat the deterministic migration dry run, apply it with a state backup,
   and rename the artifact to v0.3.
2. Resume the existing sample under `caffeinate` and persistent network retry.
3. Run fresh and cached full validation before training tokenizer candidates.

The full H100/H200 model-training phase remains unapproved until the tokenizer,
small real-corpus model pilot, checkpoint, and throughput gates pass.
