# Lumi representative tokenizer sample v0.2

Status: **capacity-corrected 24-source canary passed; fresh 6GB build approved**

## Why v0.2 exists

The first v0.1 6GB attempt stopped safely at 1,653,568,270 accepted UTF-8 text
bytes. Its OER Commons allocation requested 178,200,000 bytes, while the full
pinned source supplied only 50,199,017 bytes after the frozen filters and
cross-source quality gates. The 25MB canary had tested quality and proportional
coverage, but not full-source exhaustion.

Follow-up exhaustion audits measured the other small, single-file sources:

| Component | Verified clean capacity | v0.2 6GB target |
|---|---:|---:|
| OER Commons | 50,199,017 bytes | 35,640,000 bytes |
| Python enhancement proposals | 11,244,664 bytes | 7,128,000 bytes |
| Public Domain Review | 7,021,638 bytes | 4,455,000 bytes |

The unused share stays inside the same top-level category and moves to larger
pinned sources. The seven-domain percentages, English-first 99% base, 1%
Bengali tokenizer slice, filters, and provenance policy do not change. Exact
evidence and pins are in
`reports/tokenizer-source-capacity-audit-v0.1.json`.

## New preflight and failure behavior

- The composed config carries verified accepted-text capacities tied to exact
  dataset revisions, files, blob hashes, filters, and quality-config hash.
- A source target above a verified capacity fails before an output directory or
  network stream is created.
- Interrupted or failed builds persist a structured `failure` object and
  `status: failed` in `run-state.json`.
- A matching `--resume` clears stale failure state while building and still
  preserves byte-level deterministic source quotas.
- The previous v0.1 artifact is retained; v0.2 starts fresh so its sample is
  reproducible directly from the committed config without migration.

## v0.2 combined canary

Completed on 2026-07-15:

| Check | Result |
|---|---:|
| Accepted UTF-8 text | 25,391,622 bytes |
| Documents | 4,859 |
| Pinned sources meeting quota | 24 / 24 |
| Language labels | `en`: 3,504; `zxx`: 1,350; `bn`: 5 |
| Network retries | 0 |
| Fresh validation | Pass, from line 0 |
| Cached validation | Pass, resumed at line 4,859 |
| Completed `--resume` | Pass without another source read |
| Sample SHA-256 | `9eba9d35593c77d0579edfb5e740c106cd08a410e0d75c52678c1964fce1a32f` |

The canary remains outside Git at
`/private/tmp/lumi-tokenizer-24-source-canary-v02`.

## Full build

The default script now composes `configs/tokenizer-sample-v0.2.json`. Launch on
the Mac with a sleep inhibitor, persistent retry, and a new output directory:

```bash
caffeinate -dimsu /usr/bin/env PYTHONPATH=src python \
  scripts/build_representative_tokenizer_sample.py \
  --output /Users/afnan/Documents/lumi-data/tokenizer-sample-v0.2 \
  --resume \
  --retry-forever \
  --timeout-seconds 120
```

This stage uses CPU, SSD, and network I/O. The language detector, gzip/JSON
decode, MinHash, and SQLite indexes do not benefit from forcing the Apple GPU.
The Apple GPU is reserved for the later MLX model pilot and inference gates.

After completion, run one fresh and one cached validation before training the
32K and 65K tokenizer candidates from the exact same sample.
