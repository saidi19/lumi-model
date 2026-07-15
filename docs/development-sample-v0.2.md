# Lumi development sample v0.2

Status: **integrity validation passed; post-build quality audit failed; superseded by v0.4**

Run date: 2026-07-15. GPU usage: none.

## Why v0.2 was required

The first 1GB development run exposed an over-broad legacy PII regex. It treated
scientific and technical identifiers such as software versions, enzyme EC
numbers, ORCIDs, and catalogue identifiers as IP addresses or phone numbers.
That v0.1 output remains a useful diagnostic artifact, but it is not eligible
for tokenizer or model training.

The contextual v2 detector validates IPv4 syntax and requires nearby network or
URL context. Phone-like values require an international `+` prefix or nearby
contact context. Explicit scholarly/version contexts are preserved. The same
eligibility predicates are used during filtering and independent validation,
and the implementation version is included in the run fingerprint and manifest.

| Run | Documents | Email redactions | IPv4 redactions | Phone redactions |
|---|---:|---:|---:|---:|
| Diagnostic v0.1 | 166,758 | 1,906 | 1,323 | 4,094 |
| Contextual-PII v0.2 | 166,757 | 1,906 | 22 | 1,493 |

The 25MB all-component v0.2 canary passed before this run. Its old-to-new counts
fell from 5 to 0 IPv4 redactions and from 147 to 21 phone redactions, and manual
snippet review found preserved software version `2.6.4.0`. The larger run then
exposed cases absent from that canary: broad nearby network words caused section
numbers and `$Revision:1.2.3.4$` to be redacted as IPv4, while the ambiguous word
`cell` caused a scientific wavenumber series to be redacted as phone data.

The artifact passed structural, hash, provenance, quality-metadata, deduplication,
and contamination validation, but this manual quality finding blocks its use for
tokenizer or model training. Contextual PII v4 and the fresh v0.4 corpus replace it.

## Reproducibility anchor

- Development plan: [`configs/development-sample-v0.1.json`](../configs/development-sample-v0.1.json)
- Pinned component files: [`configs/component-anchor-files-v0.1.json`](../configs/component-anchor-files-v0.1.json)
- Quality policy: [`configs/data-quality-v0.1.json`](../configs/data-quality-v0.1.json)
- Hashed held-out benchmark registry: `data/benchmark-signatures-v0.1.jsonl.gz`
- PII implementation: `contextual-v2`
- Accepted documents: 166,757
- Accepted text: 1,000,895,709 UTF-8 bytes
- Materialized JSONL size: 1,253,923,258 bytes
- Output SHA-256: `d52950e5bce93f0f1bd44c55a792e62865bc2af3b39c679efe59a39a76bd6dd3`
- Manifest SHA-256: `752651e1909109dd7c6836af0057072dc1f1b92410c72fed62f9466d01f8b92c`
- Pinned sources used: 23 of 23
- Network retries: 0

Raw sample files remain outside Git. Every record retains a deterministic ID,
content and raw-text hashes, component repository and revision, source file/blob
and line, category, upstream identity and provenance, and either document-level
license metadata or the tracked CCCC source-level basis.

## Composition result

| Domain | Documents | Target bytes | Accepted bytes |
|---|---:|---:|---:|
| Science and education | 14,612 | 300,000,000 | 300,219,253 |
| Code | 61,177 | 200,000,000 | 200,056,783 |
| Reference | 42,799 | 150,000,000 | 150,015,982 |
| Questions and discussion | 24,203 | 150,000,000 | 150,014,482 |
| Open web | 16,862 | 100,000,000 | 100,024,526 |
| Books | 4,705 | 50,000,000 | 50,242,416 |
| Government, legal, patents | 2,399 | 50,000,000 | 50,322,267 |

Every category and every component-level quota passed. Large upstream shards
were streamed only until their weighted target was reached.

## Quality and safety result

- Language labels: 105,580 English records and 61,177 code-category `zxx`
  bypass records.
- Removed 1,456 MinHash near-duplicates.
- Removed six MMLU-contaminated segments using the raw-text-free held-out
  signature registry.
- Removed 366 records with a missing required document license.
- Removed 6,326 short records and 13 records with excessive repeated lines.
- Redacted 1,906 email addresses, 22 contextual IPv4 addresses, and 1,493
  contextual phone numbers.
- Dropped secret-bearing documents: 140 assigned credentials, 13 AWS access
  keys, one AWS secret key, 54 credentialed database URIs, 10 Google API keys,
  21 JWTs, 13 private keys, one Slack token, and one Stripe secret.
- The build used pinned Lingua 2.2.0 and Datasketch 2.0.0 under Python 3.12.

The benchmark registry covers 46,999 held-out examples from MMLU, ARC Easy/Challenge,
HellaSwag, HumanEval, GSM8K, TruthfulQA, OpenBookQA, and MBPP. Git retains only
exact and anchor hashes, never raw evaluation prompts.

## Limitations and disposition

This v0.2 sample is retained only as a reproducible diagnostic. Do not feed it to
a tokenizer or model. Pattern scanning cannot prove that no private data remains;
manual post-build snippet audits remain a hard gate even after automated
validation passes. Language identification and MinHash-LSH are probabilistic,
the benchmark registry covers only its pinned tasks, and CCCC still relies on a
tracked source-level license basis.
