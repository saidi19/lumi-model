# Lumi representative tokenizer sample v0.1

Status: **25MB scale canary passed; full 6GB build awaits the multilingual slice**

## Purpose

This stage produces the provenance-preserving, quality-gated text used to train
the 32K and 65K tokenizer candidates. It is not model-pretraining data and it
does not approve an H100 run. The default plan targets 6,000,000,000 accepted
UTF-8 text bytes across the same 23 pinned components as the audited 1GB
development artifact.

The English-first component mix is ready. A small explicitly licensed
Bangla/multilingual slice must be added and audited before either tokenizer is
frozen; the slice protects future Unicode token efficiency without changing
Lumi v1 into a Bangla-focused model.

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

Run a bounded all-component build before the full target:

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

1. the 25MB SQLite/restart canary remains reproducible (passed once);
2. the licensed multilingual slice is configured and audited;
3. projected free disk space covers JSONL, both SQLite indexes, tokenizer
   outputs, and temporary files;
4. the job is launched with `--resume --retry-forever` and local logs;
5. 32K and 65K tokenizers are trained from the exact same finalized sample.
