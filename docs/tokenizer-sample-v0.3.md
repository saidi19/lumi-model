# Lumi representative tokenizer sample v0.3

Status: **multi-shard implementation and Stack capacity gate passed; WikiTeam and
CCCC capacity audits pending; 6GB resume blocked by design**

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
| WikiTeam | 222,750,000 | pending bounded audit | Blocked |
| CCCC | 594,000,000 | pending bounded audit | Blocked |

The Stack continuation audit accepted 100,003,598 bytes across 30,362 documents,
with zero network retries, then passed full post-write validation. Its sample
SHA-256 is
`6c3ba9bcc26550edc53cd3475d4ce9dbd5ecad0277a546920b993e182bc2154c`.

The remaining audits require network permission. The no-download full-plan
preflight currently refuses to launch and reports both missing capacities. No
GPU or Lambda credit is involved in these audits.

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

1. Complete and validate the bounded WikiTeam and CCCC capacity audits.
2. Make the 6GB no-download quota preflight pass.
3. Run the v0.3 25MB all-source fresh, cached-validation, and completed-resume
   canary.
4. Repeat the deterministic migration dry run, apply it with a state backup,
   and rename the artifact to v0.3.
5. Resume the existing sample under `caffeinate` and persistent network retry.
6. Run fresh and cached full validation before training tokenizer candidates.

The full H100/H200 model-training phase remains unapproved until the tokenizer,
small real-corpus model pilot, checkpoint, and throughput gates pass.
