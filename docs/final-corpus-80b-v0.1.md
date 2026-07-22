# Lumi Final 80B Corpus Gate v0.1

Status: **the deterministic 120B-to-80B selection/shard contract passes local
validation, and its separately authorized bounded Glue canary completed and
passed independent verification. The consumed Glue job and IAM role are
deleted. The exact production package is now commit-bound, locally verified,
and deliberately non-authorizing. Production selection, GPU work, and model
training remain unlaunched and unauthorized.**

## Why this gate exists

The verified global-dedup result is a quality-filtered candidate pool, not a
training corpus. It contains `120,000,036,715` tokens so each category has a
50% safety buffer around the intended 80B mixture. Training all 120B tokens
would silently change the agreed mixture and compute budget.

This gate deterministically chooses the final mixture, preserves the existing
immutable prefix, and defines the evidence required before any selected object
can become a training shard.

The permanent held-out exclusion is not a new extraction step. Mapping already
applied the sealed 35,889-entry training-exclusion ledger with SHA-256
`709027bb95f3e0f956b22a79d4b29f21490ef0b360ba10f2faca7d5db3d06c62`.
The final plan binds that exact ledger and the benchmark-signature registry, so
selection fails closed if either changes.

## Verified input boundary

The contract in `configs/final-corpus-80b-v0.1.json` pins:

- production dedup result canonical SHA-256
  `dbfbf9fe258ec97d5bd844c165157d41d328d0571cbbada5fd4d67750c46b742`;
- accepted-candidate inventory: `68,671,669` documents,
  `119,762,267,075` tokens, 2,772 Parquet objects, and
  `507,737,434,761` bytes;
- immutable prefix: `38,349` documents and `237,769,640` tokens;
- frozen 65K tokenizer SHA-256
  `1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee`;
- the exact data-mix, filter policy, benchmark registry, and permanent
  training-exclusion ledger files.

The plan accepts only the verified production result with all seven 120B
category floors complete and zero deficit. It also binds the exact release,
authorization, input-index, and map-checkpoint hashes from that run.

## Final mixture

| Category | Final target | Preserved prefix | New selection target |
|---|---:|---:|---:|
| Science and education | 24B | 237,769,640 | 23,762,230,360 |
| Code | 16B | 0 | 16B |
| Reference | 12B | 0 | 12B |
| Questions and discussion | 12B | 0 | 12B |
| Open web | 8B | 0 | 8B |
| Books | 4B | 0 | 4B |
| Government, legal, and patents | 4B | 0 | 4B |
| **Total** | **80B** | **237,769,640** | **79,762,230,360** |

## Deterministic selection

Within each category, every accepted row receives this cross-runtime rank:

```text
sha256(seed_sha256 NUL category NUL component NUL document_id NUL content_sha256)
```

Rows are sorted by that rank and the complete deterministic tie-break key. The
first document-boundary prefix reaching the category target is retained. Hash
ranking prevents source-file order from deciding which already-filtered,
deduplicated documents survive. It does not replace the upstream quality
filters.

Each source segment is capped at 131,072 UTF-8 bytes, so each category may
overshoot by at most 131,072 tokens at its last document boundary. Therefore:

- selected material must contain at least `80,000,000,000` tokens;
- the absolute selected-material maximum is `80,000,917,504` tokens;
- the trainer's consumption boundary remains exactly `80,000,000,000` tokens.

The selection result must record the full cutoff tuple, per-category available
and selected counts, content-addressed S3 inventories, exact input bindings,
and a canonical result fingerprint.

## Packed-shard contract

The output namespace is fixed to
`s3://lumi-corpus-prod-27777c02b29b76efcac3/lumi-corpus/v0.1/final-80b/`.
The plan keeps the three existing prefix artifacts and expects 160 new
approximately 500M-token uint16 shards, for 163 manifest entries in total.

The shard validator requires:

- contiguous shard sequence numbers and unique object keys/hashes;
- exact per-category and aggregate document/token conservation;
- exactly two bytes per token;
- no shard above 500M tokens plus the document-boundary cap;
- exact binding to the selected-result and plan fingerprints.

Selected Parquet, the selection ledger, and packed shards remain separate
immutable products. An object is not training-ready merely because it exists
in S3; the final selection result and shard manifest must both validate.

## Local evidence

The planning CLI is `scripts/plan_final_corpus_80b.py`. It writes canonical
`plan.json` plus `manifest.sha256` only into an empty destination. The focused
tests cover exact 120B-to-80B accounting, authorization/input tampering,
cross-runtime rank stability, category overshoot, S3 boundaries, shard byte
conservation, duplicate objects, and optional local Spark partition parity.

The planning artifact originally stopped at
`ready_for_bounded_glue_canary_approval`. The exact bounded canary subsequently
received separate owner approval and passed; that does not authorize the
production materialization run.

## Prepared bounded canary

`configs/glue-final-corpus-canary-v0.1.json` fixes a single structural canary
to 8 G.1X workers, 30 minutes, and `$1.76` maximum Glue compute. The package is
non-authorizing and the launcher is dry-run unless `--execute` is supplied with
a still-valid exact authorization artifact.

The canary is deliberately much smaller than production. It:

- independently matches the complete 2,772-object accepted-candidate S3
  inventory before reading any row;
- chooses exactly one deterministic Parquet object from every category and
  rejects more than 100,000 candidate documents per category;
- selects approximately 1M tokens per category twice, using 16 and 23 Spark
  partitions, and requires identical selected hashes, counts, and cutoffs;
- writes the bounded selected Parquet, assigns deterministic approximately
  250K-token shard sequences, and publishes content-addressed uint16 objects;
- checks exact document/token/byte conservation, runtime pins, immutable
  baseline-object heads, bounded S3 paths, and a 1GB output ceiling;
- grants its one-purpose Glue role access only to the exact bootstrap and
  baseline objects, the accepted-candidate prefix, and its authorization-bound
  canary/result/temp prefixes.

The implementation is in `src/lumi/glue_final_corpus.py`; package,
authorization, launch, and Glue runtime entry points are under `scripts/`. The
launcher atomically consumes each authorization with an immutable S3 launch
claim, refuses an authorization/job that already has a run, uses zero Glue job
retries, and cannot authorize production selection, GPU work, or training.

Local Spark execution remains an optional environment test because this Codex
sandbox cannot bind the loopback socket needed by the local JVM. The pure
contract, package, authorization, IAM-boundary, result-conservation, tamper,
and lint tests pass locally. The bounded AWS canary supplied the required real
Spark execution proof before production materialization.

## Verified AWS canary result

The one owner-authorized run used release
`eb2ac594a0719c9f03806290d0ea109d68756dfbf2a9fb3e5119218a97a2bfb2`
and authorization
`9036630123cf74bda8ea2cc07b3c8c7607b08fa7bdcca05deec69d1f64ea220e`.
Glue run
`jr_5e8e8b1b2e904819fb760dd1d0218eed5ab9604970163a047114b05a5ec0dbb4`
finished `SUCCEEDED` in 235 execution seconds, consumed 1,887 DPU-seconds, and
cost an estimated `$0.230633` at `$0.44` per DPU-hour. It stayed inside the
8-worker, 30-minute, `$1.76` envelope with zero retries.

The run independently matched the complete input inventory of 2,772 objects,
`507,737,434,761` bytes, `68,671,669` documents, and `119,762,267,075`
tokens. Across all seven categories it selected 3,803 documents and
`7,026,092` tokens, produced seven selected Parquet objects plus 28 packed
uint16 objects, and passed 16-versus-23-partition parity in every category.
Selected output occupied `31,041,302` bytes; packed output occupied
`14,052,184` bytes, exactly two bytes per selected token.

The canonical result SHA-256 is
`171974e1fc329b0ede47c2759d7f85a44f2561e68d7322ee6c97966ccaad49ac`.
The release, authorization, input, plan, baseline heads, result conservation,
content-addressed output inventories, and local evidence hashes all pass. The
consumed Glue job and dedicated IAM role were then deleted. The compact sealed
record is `reports/glue-final-corpus-canary-result-v0.1.json`; generated S3
payloads remain outside Git.

## Prepared production package

`configs/glue-final-corpus-production-v0.1.json` and
`src/lumi/glue_final_corpus_production.py` now define the production-only
selection/sharding boundary proven by the canary. The package remains
non-authorizing until a separate exact owner approval creates a short-lived
authorization artifact. Its fixed envelope is:

- one Glue 5.1 job in `us-east-1`, 100 G.1X workers, zero retries, and one
  concurrent run;
- a 12-hour hard timeout and `$528.000000` maximum Glue compute at the pinned
  `$0.44` per DPU-hour rate;
- the exact 2,772-object, `507,737,434,761`-byte accepted-candidate inventory;
- 1,600 deterministic selection partitions, seven immutable category
  checkpoints, 160 new approximately 500M-token shards, and exactly 163 final
  manifest entries after the three preserved baseline shards;
- a `750,000,000,000`-byte hard ceiling on newly materialized selected
  Parquet, copied/materialized baseline shards, and packed uint16 shards;
- no GPU permission and no model-training permission.

The canary calibration processed `301,453,386` input tokens in 1,887
DPU-seconds. Linear projection for the full accepted pool is 749,673
DPU-seconds, approximately 125 minutes on 100 DPU and `$91.626700`. The sealed
planning rule applies a 3x safety factor: 2,249,019 DPU-seconds, approximately
375 minutes and `$274.880100`. These are measured projections, not guarantees;
the 12-hour/`$528` boundary remains the actual fail-closed limit. Projected
maximum new output is `512,395,734,963` bytes, below the 750GB cap.

Before any row is selected, the runtime independently inventories every input
object, validates the exact release, plan, canary report, authorization, Python
3.11 and Spark 3.5.6 bindings, and assigns every input object to exactly one
category. It then hash-ranks and selects each category, writes an immutable
selected inventory, packs content-addressed uint16 shards, and seals a category
checkpoint before advancing. The two existing full baseline shards are
downloaded and rehashed before immutable publication; the partial baseline is
extracted from its hash-pinned archive member and independently rehashed.
Final publication is allowed only after selection-result, shard-manifest,
document/token/byte conservation, baseline, category, runtime, and output-cap
validation all pass.

The launcher is dry-run by default. Its one-purpose IAM policy can read only
the accepted prefix and exact bootstrap/baseline objects, can write only the
authorization-derived run/temp/result prefixes, uses an immutable launch claim,
and refuses any job that already ran. No authorization artifact was created and
no AWS API was called while preparing this package.

The reproducible local package is bound to source commit
`227721d452e2baa9dc1e59311eb2d203b14f3561`. Its release fingerprint is
`b2c5758ec846f3f9f4b75dba20c28c52125cee5753fadf75352c7546a8525a5d`, and
its five-entry `manifest.sha256` file hashes to
`43f7a914a2c7a120fed79f43573c260fa345de8a988670cbe66a547f9091bcf2`.
All five package files and every ZIP member pass independent integrity checks.
The five focused production tests, all 323 collected non-socket repository
tests (three environment-dependent skips), and repository-wide Ruff pass.

## Next boundary

The production package gate is complete. The next possible action is one
separately owner-approved 100-DPU production selection-and-sharding run under
the exact 12-hour/`$528` envelope above. No approval has been inferred from
earlier canary or dedup runs. After a successful run, its result, all seven
checkpoints, 163-shard manifest, S3 inventories, and hashes must be independently
verified before the consumed Glue job/role are removed. The 4B pilot and full
4B/80B training remain later, independent approval gates.
