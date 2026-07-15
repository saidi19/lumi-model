# Lumi development sample v0.4

Status: **passed as a development-data and tokenizer gate; no GPU training is approved**

Run date: 2026-07-15. GPU usage: none.

## Why the PII detector reached v4

Manual snippet review remained a hard gate even when automated integrity checks
passed. That review found and fixed four ambiguity classes before this artifact
was promoted:

- the legacy detector treated software versions, enzyme EC numbers, ORCIDs, and
  catalogue identifiers as IP addresses or phone numbers;
- contextual v2 still treated numeric section headings and source-control
  revisions as IPv4 addresses;
- contextual v3 allowed private-range-looking section numbers and `TCP/IP §...`
  index entries to trigger;
- broad phone proximity around the word `cell` damaged scientific wavenumber
  axes.

Contextual v4 validates IPv4 syntax, checks explicit IP/network/URL/config
relationships, applies section/revision exclusions before network labels, and
does not redact a private-looking numeric quad without configuration context.
Phone-like values require an international `+` prefix or a tight telephone,
fax, hotline, call, contact, or field-label relationship. Scientific/version
contexts are excluded. The same predicates run during filtering and validation,
and `contextual-v4` is included in both the run fingerprint and manifest.

| Artifact | IPv4 redactions | Phone redactions | Disposition |
|---|---:|---:|---|
| Legacy diagnostic v0.1 | 1,323 | 4,094 | rejected: broad regex |
| Contextual v0.2 | 22 | 1,493 | rejected: section/revision and wavenumber false positives |
| Contextual v0.3 | stopped at 117MB | stopped at 117MB | rejected early: private-range headings and `TCP/IP §...` |
| Contextual v0.4 | 9 | 1,318 | all sampled contexts passed manual review |

The nine v0.4 IPv4 matches are explicit IP labels/ranges, IP-host URLs, NFS or
DRBD configuration addresses, a game-server URL, and an OpenVPN command input.
The previously damaged source-control revisions, `TCP/IP §3.3.3.2`, software
version `2.6.4.0`, and scientific `3800 ... 2400` wavenumber sequences remain
intact. Sampled phone matches are telephone/fax/hotline links, contact fields,
or explicitly labelled phone values.

## Reproducibility anchor

- Development plan: [`configs/development-sample-v0.1.json`](../configs/development-sample-v0.1.json)
- Pinned component files: [`configs/component-anchor-files-v0.1.json`](../configs/component-anchor-files-v0.1.json)
- Quality policy: [`configs/data-quality-v0.1.json`](../configs/data-quality-v0.1.json)
- Hashed held-out benchmark registry: `data/benchmark-signatures-v0.1.jsonl.gz`
- PII implementation: `contextual-v4`
- Accepted documents: 166,757
- Accepted text: 1,000,895,350 UTF-8 bytes
- Materialized JSONL size: 1,253,922,152 bytes
- Output SHA-256: `6013680fdbd3ea06cc9b11691f31f7bd46906405e62534f89491a350e1983667`
- Manifest SHA-256: `0c10059c78b8b59a7e7d658494a6f4bd1edf3ebc94692b768b52c962ae16b44d`
- Pinned sources used: 23 of 23
- Successful run network retries: 0

One operator-level run failed during the first source when Wi-Fi disappeared,
before accepting any records. The same fingerprinted output was resumed after
connectivity returned. Raw samples remain outside Git.

## Composition result

| Domain | Documents | Target bytes | Accepted bytes |
|---|---:|---:|---:|
| Science and education | 14,612 | 300,000,000 | 300,219,188 |
| Code | 61,177 | 200,000,000 | 200,056,717 |
| Reference | 42,799 | 150,000,000 | 150,015,966 |
| Questions and discussion | 24,203 | 150,000,000 | 150,014,479 |
| Open web | 16,862 | 100,000,000 | 100,024,387 |
| Books | 4,705 | 50,000,000 | 50,242,416 |
| Government, legal, patents | 2,399 | 50,000,000 | 50,322,197 |

Every category and all 23 component-level quotas passed. Large upstream files
were streamed only until their weighted target was reached.

## Quality and safety result

- Language labels: 105,580 English records and 61,177 code-category `zxx`
  bypass records.
- Removed 1,456 MinHash near-duplicates and six MMLU-contaminated segments.
- Removed 366 records with a missing required document license, 6,326 short
  records, and 13 records with excessive repeated lines.
- Redacted 1,906 email addresses, nine contextual IPv4 addresses, and 1,318
  contextual phone numbers.
- Dropped secret-bearing documents: 140 assigned credentials, 13 AWS access
  keys, one AWS secret key, 54 credentialed database URIs, 10 Google API keys,
  21 JWTs, 13 private keys, one Slack token, and one Stripe secret.
- The build used Python 3.12, Lingua 2.2.0, and Datasketch 2.0.0.

The raw-text-free registry contains 46,999 held-out examples from MMLU, ARC
Easy/Challenge, HellaSwag, HumanEval, GSM8K, TruthfulQA, OpenBookQA, and MBPP.

## Limitations and next gate

This is a bounded development corpus, not the final 80B-token corpus. Pattern
scanning cannot prove that no private data remains; manual audits remain
mandatory. Language identification and MinHash-LSH are probabilistic, the
benchmark registry covers only its pinned tasks, CCCC relies on a tracked
source-level license basis, and this sample is intentionally English-first.

The full post-write validator recomputed every file/content hash, deterministic
ID, source and required provenance field, category/source quota, language label,
benchmark signature, and MinHash decision before sealing the manifest.

Next, add a small explicitly licensed Bangla and multilingual tokenizer slice.
Build a representative tokenizer-only corpus, train the 32K and 65K candidates
from identical bytes, and compare domain and multilingual efficiency before
freezing one tokenizer. Only then run the 50M-token 150M plumbing pilot and a
measured one-H100 throughput calibration.
