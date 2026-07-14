# Lumi data canary v0.1

Status: **passed as a pipeline canary; not approved for model training**

Run date: 2026-07-15. GPU usage: none.

## Reproducibility anchor

- Dataset: `common-pile/comma_v0.1_training_dataset`
- Pinned revision: `5afc546db324e7f39f297ba757c9a60547151e7c`
- Sampling config: [`configs/tokenizer-canary-v0.1.json`](../configs/tokenizer-canary-v0.1.json)
- Accepted documents: 4,757
- Accepted text: 25,061,506 UTF-8 bytes
- Materialized JSONL size: 28,736,515 bytes
- Output SHA-256: `ff2e619f39686895f9bf6d370bc46763bed3dec0fca32d51fa598b8e130c33ba`

The raw sample and tokenizer artifacts stay outside Git. Every accepted segment
contains the pinned dataset revision, source file and blob, source line,
category, raw-text hash, transformed-content hash, and deterministic document
ID. The validator recomputed all hashes, IDs, category totals, NFC normalization,
exact-duplicate constraints, secret patterns, and PII patterns after writing.

## Domain result

| Domain | Documents | Target bytes | Accepted bytes |
|---|---:|---:|---:|
| Science and education | 718 | 7,500,000 | 7,518,596 |
| Code | 1,425 | 5,000,000 | 5,003,137 |
| Reference | 936 | 3,750,000 | 3,768,466 |
| Questions and discussion | 1,329 | 3,750,000 | 3,760,479 |
| Open web | 296 | 2,500,000 | 2,500,620 |
| Books | 11 | 1,250,000 | 1,257,046 |
| Government, legal, patents | 42 | 1,250,000 | 1,253,162 |

All seven quotas passed. Eight pinned files were touched because the first
LibreTexts shard did not alone fill the science quota and the sampler continued
into the pinned peS2o fallback. Large files were streamed only until their quota
was reached; they were not downloaded in full.

## Safety result

- Dropped 11 documents matching credential patterns: 6 assigned credentials,
  1 AWS access key, 3 Google API keys, and 1 private key.
- Dropped 156 documents below the minimum length and 1 with excessive control
  characters.
- Redacted 637 email addresses, 39 IPv4 addresses, and 369 phone-like strings
  from accepted documents.
- No remaining configured secret or PII pattern was found by the independent
  post-write scan.

These are conservative pattern filters, not proof that the corpus contains no
sensitive data. The development sample still needs stronger secret detection,
language identification, near-duplicate removal, benchmark decontamination, and
manual audit samples before a real pilot.

## Provenance correction before scaling

The consolidated Comma rows used by this canary expose only `text`. A pinned
file and line still reproduce each canary record, but they do not preserve the
original Common Pile document ID, URL, or per-document license. Also, Common
Pile has mixed document licenses rather than one dataset-wide license.

This is acceptable for testing the streaming and filtering plumbing, but not
for Lumi's development or final corpus. A schema audit confirmed that the 23
individual filtered component repositories retain top-level `id` and `source`
plus provenance and usually license/URL metadata. Their exact revisions are in
[`configs/component-sources-v0.1.json`](../configs/component-sources-v0.1.json).
CCCC carries an explicit caveat because its filtered rows do not include a
license field. Its official card documents a whole-domain Creative Commons
review, so the inventory records that source-level basis together with the
schema mismatch rather than pretending a per-document value exists.

A bounded audit of all 23 pinned component schemas then inspected 44,061 rows.
Every inspected row had a stable ID, provenance, and either a non-null license
field or the documented CCCC source-level basis. URL coverage was 39,978 rows:
peS2o missed 83, while UK Hansard and USPTO do not expose URL fields. Stable IDs
remain the required removal key. This was a first-shard presence audit, not a
full legal or random-sample review.

## Tokenizer canary

Both prototypes used this exact sample and retained the frozen special-token
IDs and byte-complete Unicode round trips.

| Prototype | `tokenizer.json` SHA-256 | Artifact size | Bytes/token | Tokens/word |
|---|---|---:|---:|---:|
| 32K | `8afa0549baeca6ccfb0cba213b5fc0b2579924844db8f4decdbfa465381b08b2` | 2.2MB | 3.919508 | 1.762516 |
| 65K | `3cbe482fad18871412705d48fca17c3bca23b89c5a8470b06de27752f7c1d7ba` | 4.5MB | 4.128137 | 1.673441 |

On this small sample, 65K emitted about 5.1% fewer tokens overall, with a 4.0%
to 5.9% reduction across the seven domains. This does **not** freeze 65K: the
sample is too small and lacks an intentional multilingual slice. A Bangla probe
still required 149 tokens with 32K and 109 with 65K, which is byte-complete but
inefficient. Include a small Bangla/multilingual slice before the final
tokenizer is trained even if Lumi pretraining remains English-first.

## Next gate

1. Extend the sampler to retain a removal/license ledger from the pinned
   component repositories.
2. Audit real license-field values and decide whether the documented CCCC
   source-level basis meets Lumi's final inclusion policy.
3. Expand to a 1GB-2GB development sample without GPU use, cover more
   components, and add pinned multilingual sources.
4. Add language labels, near-deduplication, benchmark signatures, and stronger
   secret scanning.
5. Rebuild and compare 32K/65K tokenizers, then scale the winning pipeline to
   the planned 6GB-12GB tokenizer sample.
6. Freeze one tokenizer before any real 150M training pilot.
