# Lumi representative tokenizer sample v0.1

Status: **all required 25MB and source-only canaries passed; 6GB build ready**

## Purpose

This stage produces the provenance-preserving, quality-gated text used to train
the 32K and 65K tokenizer candidates. It is not model-pretraining data and it
does not approve an H100 run. The default plan targets 6,000,000,000 accepted
UTF-8 text bytes: 99% from the same 23 pinned components as the audited 1GB
development artifact and 1% from a pinned Bengali Wikipedia source. The
Bengali allocation is therefore 60MB; it changes the tokenizer sample only and
does not modify the candidate 80B pretraining mixture.

The source-level license basis, file digest, dataset revision, article ID,
title, and page URL are retained. This protects future Unicode token efficiency
without changing Lumi v1 into a Bangla-focused model. See
`docs/multilingual-source-v0.1.md` for the source decision and limitations.

## Disconnect-safe implementation

- MinHash signatures and LSH buckets are stored in
  `build-quality-state.sqlite3`; raw text is not stored in that index.
- Full validation stores document IDs, content hashes, MinHash state, byte
  offset, line number, and aggregate counters in `validation-state.sqlite3`.
- Validation commits every 1,000 records by default and resumes from the last
  committed byte offset.
- `--retry-forever` uses bounded exponential backoff during network outages.
  Once connectivity returns, the local process continues without a connected
  Codex session.
- Generated SQLite files are caches outside Git. The JSONL and manifest remain
  the authoritative corpus artifacts.
- The Bengali Parquet source is read through bounded HTTP range requests. The
  full 192MB shard is not downloaded merely to read its footer or first rows.

Progress can be inspected without modifying the job:

```bash
python scripts/data_job_status.py --sample /path/to/output-directory
```

## Required canary

Completed on 2026-07-15:

| Check | Result |
|---|---:|
| Accepted UTF-8 text | 25,570,417 bytes |
| Documents | 5,055 |
| Pinned sources used | 23 / 23 |
| Build MinHash SQLite cache | ~21MB |
| Validation SQLite cache | ~22MB |
| Persisted network retries | 1 |
| Fresh validation | Pass |
| Cached validation | Pass, resumed at line 5,055 |
| Sample SHA-256 | `fad5165ba45a2fd370563cfc5edbf01e800041253075a92f7f29e2f273706a6b` |

The generated canary remains outside Git at
`/private/tmp/lumi-tokenizer-sqlite-canary`.

The Bengali source-only canary completed on 2026-07-15:

| Check | Result |
|---|---:|
| Accepted UTF-8 text | 1,076,043 bytes |
| Documents | 20 |
| Language labels | `bn`: 20 |
| Source revision | `3e1f92c331f318af862b87e2319ed5dc26d80f5d` |
| Parquet SHA-256 | `98a8189cf1d0e4b41d647b3d88bd6201e571e002344b5648496c6bfda402841b` |
| Fresh validation | Pass, from line 0 |
| Cached validation | Pass, resumed at line 20 |
| Sample SHA-256 | `a3ce570bfa96c25e033977f2abeb8b765431fcf80a6e183693310c252b11c2c8` |

It remains outside Git at `/private/tmp/lumi-bn-parquet-canary`. Reproduce the
bounded category-only gate with:

```bash
python scripts/build_representative_tokenizer_sample.py \
  --output /path/to/lumi-bn-parquet-canary \
  --target-bytes 1000000 \
  --canary-category multilingual_reference \
  --retries 0
```

The category-only flag requires a target no larger than 100MB, preventing an
accidental full-size single-category build.

The combined 24-source canary completed on 2026-07-15:

| Check | Result |
|---|---:|
| Accepted UTF-8 text | 25,637,516 bytes |
| Documents | 5,014 |
| Pinned sources meeting quota | 24 / 24 |
| Bengali allocation | 256,799 bytes in 5 `bn` documents |
| Language labels | `en`: 3,659; `zxx`: 1,350; `bn`: 5 |
| Network retries | 0 |
| Build MinHash SQLite cache | ~20MB |
| Validation SQLite cache | ~22MB |
| Fresh validation | Pass, from line 0 |
| Cached validation | Pass, resumed at line 5,014 |
| Completed `--resume` check | Pass, no source read required |
| Sample SHA-256 | `6b28052c9479844f93cf9d7d310a36db5b0931eab02f6ce0c1142878f6d09888` |

The sample remains outside Git at
`/private/tmp/lumi-tokenizer-24-source-canary`. Its JSONL is 33,182,493 bytes;
the build directory is approximately 74MB including both resumable SQLite
indexes. The accepted Bengali share is 1.002% of accepted text, preserving the
planned English-first 99%/1% composition within document-boundary rounding.

Reproduce the bounded all-component build with:

```bash
python scripts/build_representative_tokenizer_sample.py \
  --output /Users/afnan/Documents/lumi-data/tokenizer-sample-v0.1-canary \
  --target-bytes 25000000 \
  --retry-forever
```

Then prove a fresh and cached validation:

```bash
python scripts/validate_data_sample.py \
  --sample /Users/afnan/Documents/lumi-data/tokenizer-sample-v0.1-canary \
  --restart-validation
python scripts/validate_data_sample.py \
  --sample /Users/afnan/Documents/lumi-data/tokenizer-sample-v0.1-canary
```

The plan's 6GB target differs from the canary override, so the override build is
validated against its manifest and run fingerprint. A default 6GB build can
also pass `--config` only when using a fully composed machine-readable config.

## Full-build gate

Do not start the default 6GB build until:

1. the 25MB 23-source SQLite/restart canary remains reproducible (passed once);
2. the Bengali source-only range/quality canary remains reproducible (passed once);
3. one combined 24-source canary passes every source quota (passed once);
4. projected free disk space covers JSONL, both SQLite indexes, tokenizer
   outputs, and temporary files (705GB free before launch);
5. launch the job with `--resume --retry-forever` and local logs;
6. after the sample passes validation, train 32K and 65K tokenizers from the
   exact same finalized sample.
