# Lumi representative tokenizer sample v0.3

Status: **the 6GB contextual-v5 sample is sealed; resumed full validation and
the cached validation check both pass**

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

## Full 6GB result

The deterministic v0.2-to-v0.3 migration passed, was applied with a state
backup, and resumed the intact 2.88GB prefix. The final build accepted
6,000,636,724 UTF-8 text bytes across 1,128,089 documents. All 24 logical
sources met quota, using 25 of 27 pinned physical files. One source-stream
network retry recovered automatically; no GPU or Lambda credit was used.

| Domain | Documents | Target text bytes | Accepted text bytes |
|---|---:|---:|---:|
| Science and education | 79,285 | 1,782,000,000 | 1,782,214,394 |
| Code | 363,153 | 1,188,000,000 | 1,188,003,140 |
| Reference | 275,316 | 891,000,000 | 891,033,923 |
| Questions and discussion | 248,691 | 891,000,000 | 891,003,784 |
| Open web | 132,245 | 594,000,000 | 594,000,301 |
| Books | 15,540 | 297,000,000 | 297,134,268 |
| Government, legal, patents | 10,019 | 297,000,000 | 297,225,059 |
| Bengali reference slice | 3,840 | 60,000,000 | 60,021,855 |

The sealed contextual-v5 artifact is 7,709,921,712 bytes and has sample
SHA-256
`80bea9f9634a8daca542b2ffd0aad1c8c994a7cbeef909a5e529b6272671102a`.
It contains 761,096 English, 363,153 code-bypass (`zxx`), and 3,840 Bengali
language labels. The build removed 23,063 MinHash near-duplicates, 25 hashed
benchmark matches, 2,100 records missing a required license, and 1,536
secret-bearing records. It redacted 11,622 email addresses, 40 contextual IPv4
addresses, and 6,322 contextual phone values.

Resumed full validation completed all 1,128,089 records and all
7,709,921,712 file bytes from the preserved line-166,000 checkpoint. The
canonical representative-sample `--resume` seal then returned the same
document count, accepted-text count, 24-source count, and SHA-256 without
another source read. A final cached validation resumed from line 1,128,089 and
passed the data, safety-scan, hash, and manifest gates without rescanning the
sample.

## Contextual-v5 safety repair

The first full validation stopped safely at line 166,022. One-pass contextual
redaction had replaced an earlier phone value, shortening its context window
and exposing another phone value to the same final-state detector. The validated
prefix through line 166,000 and byte offset 2,341,853,602 remained committed in
SQLite.

`scripts/audit_sample_pii.py` scanned all 962,089 unvalidated records without
retaining matched values. It found exactly two residual contextual phone
records: the first failure and one later record. Contextual v5 now repeats the
same conservative predicates to a fixed point and fails closed if they cannot
converge. `scripts/repair_sample_pii.py` then atomically repaired those two
records, recomputed their content hashes, language/benchmark/MinHash metadata,
updated the durable quality index and manifest, and migrated the checkpoint
fingerprint without changing any byte in the validated prefix.

The pre-repair sample SHA-256 was
`3d619aa6f9a81ba6393d9adbf546051698101aa3fab7797e64e5286983676597`;
the two additional `[PHONE_NUMBER]` replacements added four text/file bytes.
The original sample, manifest, state, and value-free audit remain as local,
uncommitted rollback backups pending explicit cleanup. They are not tokenizer
training inputs.

## Existing-sample migration history

`scripts/migrate_tokenizer_sample_plan.py` performs a read-only dry run by
default. It checks the old config hash and run fingerprint, proves that source
weights and byte quotas are unchanged, scans every existing record under the
new physical-to-logical mapping, verifies the exact exhausted Stack capacity,
and hashes the entire sample.

The first dry run passed with sample SHA-256
`d08a8f717faa5cfb244356b3a2b34f4342c7cffd18df617440d223c2fcdc4dd2`.
After the final capacity report was committed, the repeated dry run passed and
`--apply` created `run-state.pre-v0.3.json` before updating the fingerprint and
physical-source state. Fast quality-index restoration reused 417,700 committed
entries and rebuilt only the 40-record crash tail before the network-backed
resume continued.

## Outcome and next gates

1. Completed: corrected 32K and 65K tokenizer candidates were trained from this
   exact sealed sample with identical settings.
2. Completed: the full-sample comparison, round-trip, special-token,
   vocabulary-length, and multilingual/code gates pass; 65K is frozen in
   [`tokenizer-v0.1.md`](tokenizer-v0.1.md).
3. Completed: deterministic fixed-sequence packing, shard hashing, memory-mapped
   loading, and interrupted-build resume gates pass.
4. Completed: checkpointed training resume is tensor-identical to an
   uninterrupted regression run; a 4.29M-parameter, 8,192-token real-corpus
   Apple-MPS smoke checkpoint resumed successfully in a fresh process. See
   [`real-data-pilot-v0.1.md`](real-data-pilot-v0.1.md).
5. Completed: the exact 24-source mix is materialized as 49,999,872 tokens in
   48 hash-verified shards; cached resume validation passes.
6. Completed: the 150M Lambda H100 plumbing run processes 50,003,968 tokens,
   passes fresh-process resume, and its final checkpoint is hash-verified on the
   Mac. See [`lambda-pilot-v0.1.md`](lambda-pilot-v0.1.md).
7. Completed: a 4,999,168-token, 24-source model-weight-held-out artifact has
   zero ID/content/near-duplicate overlap; the resumed M5/MPS evaluation records
   loss 4.680812 and perplexity 107.858. Fixed generations remain unusable. See
   [`evaluation-pilot-v0.1.md`](evaluation-pilot-v0.1.md).
8. Completed: trained-checkpoint MLX parity, 4-bit local inference, and the
   loopback OpenAI-compatible local serving gate pass.
9. Completed: an exact 1,050,981,888-parameter 1B scaling architecture and
   staged, unpriced plan are frozen. Next: implement and materialize its
   hash-ranked, permanent-held-out-excluding 1.4B-token local data artifact.

The full 4B H100/H200 model-training phase remains unapproved until the 1B
quality pilot and conversion/serving gates pass.
