# Lumi complete source-file manifest v0.1

Status: **pass for metadata completeness; corpus acquisition, cloud-resource
creation, paid compute, and model training remain unapproved**.

## Why this gate exists

The earlier capacity gate pinned each Common Pile repository revision, data-file
count, and aggregate compressed bytes. Those aggregates were enough for
planning, but not enough for safe acquisition: a resumable worker needs the
exact path, byte size, and content identity of every expected upstream object.

This gate closes that gap without downloading corpus payload. It uses the
public Hugging Face `list_repo_tree` metadata API at each exact 40-character
commit and records:

- component, category, dataset ID, and pinned revision;
- exact source-file path and byte size;
- Git blob SHA-1;
- Git LFS object SHA-256, used as the future download-integrity digest;
- Xet hash when supplied by the Hub;
- provenance paths and the existing source-license policy status; and
- repository control-file metadata for completeness auditing.

Only `*.json.gz`, `.gitattributes`, and `README.md` are accepted. An unknown
file, missing LFS digest, LFS/tree size disagreement, absent schema anchor,
duplicate path, revision drift, count drift, or aggregate-byte drift fails the
gate closed.

## Authorization boundary

The contract in
[`configs/source-file-manifest-v0.1.json`](../configs/source-file-manifest-v0.1.json)
authorizes only public repository metadata requests. It explicitly records:

- `bulk_corpus_materialization=false`;
- `transfer_source_payload=false`;
- `create_cloud_resources=false`;
- `paid_compute=false`; and
- `full_4b_training=false`.

The API is called with `token=false`, `expand=false`, and pinned revisions. No
AWS bucket or GPU instance was created. The report records zero downloaded
corpus-payload bytes.

## Verified result

| Check | Result |
|---|---:|
| Pinned component repositories | 23 / 23 |
| Exact data-file records | 758 / 758 |
| Aggregate compressed source bytes | 499,817,285,384 |
| Repository control files | 46 |
| Unique Git LFS SHA-256 identities | 758 / 758 |
| Unknown repository files | 0 |
| Revision/count/size/hash mismatches | 0 |
| Corpus payload bytes downloaded | 0 |

The canonical Git snapshot is
[`configs/common-pile-source-files-v0.1.json`](../configs/common-pile-source-files-v0.1.json).
It is 384,827 bytes with SHA-256
`5289fa01c094ddbb456fe7628ea6e134f8e8455d5b8a5fc0bdc5d7d612e87c38`.

CCCC is included so the upstream tree is complete, but its records remain
`blocked_pending_owner_cccc_license_decision_and_production_authorization`.
Completing this manifest is not a license approval.

## Resumability and evidence

The runner commits each completed repository tree atomically to SQLite. With
`--resume`, already-complete sources are revalidated from durable state and are
not fetched again. A changed contract, input hash, revision, expected file
count, or expected byte total cannot reuse the state.

Canonical local evidence is stored outside Git at
`lumi-data/source-file-manifest-v0.1/`:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `state.sqlite3` | 241,664 | `b413d00ed1ab551baf9f25380d05632cf9bf6d8f68417de0bbed11d4e85c6fc6` |
| `source-files.jsonl` | 598,698 | `b46d01b66dbc16ad98642c040e650f30124797fc1db47954fcf6c0ce0a93fe25` |
| `snapshot.json` | 384,827 | `5289fa01c094ddbb456fe7628ea6e134f8e8455d5b8a5fc0bdc5d7d612e87c38` |
| `report.json` | 3,147 | `d3880a5edd1a14115edf0e65a5d944835c474409273a51d2ea7cc006789b25eb` |
| `manifest.sha256` | 323 | `fb49c05e13b8492ab6ce2711161bdb6e5bacb8c165d766842966a2ab9cf44fdd` |

The report fingerprint is
`e477a84fbe292e3d6fb350ee10b24315096d4424fc4eb0921c55a3a748fed1db`.
Independent verification reproduces the report from the pinned contract,
inputs, implementation hashes, SQLite rows, JSONL records, and repository
snapshot, then checks every entry in `manifest.sha256`.

Run or resume the metadata gate with:

```bash
PYTHONPATH=src python scripts/run_source_file_manifest.py \
  --output "$LUMI_DATA/source-file-manifest-v0.1" \
  --resume
```

Verify it independently with:

```bash
PYTHONPATH=src python scripts/verify_source_file_manifest.py \
  --output "$LUMI_DATA/source-file-manifest-v0.1"
```

## Next gate

The complete source-file manifest blocker is closed. Before production corpus
acquisition, two separate owner decisions remain:

1. accept or exclude CCCC after reviewing its source-level license basis; and
2. explicitly authorize creation of the production S3 namespace and the
   adaptive acquisition run.

Until both are resolved, no source payload may be transferred. The future
acquisition still stops when all seven category floors total 120B accepted
candidate tokens; 150B remains a hard failure ceiling, not a download target.
