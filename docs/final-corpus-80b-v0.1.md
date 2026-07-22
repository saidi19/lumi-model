# Lumi Final 80B Corpus Gate v0.1

Status: **the deterministic 120B-to-80B selection/shard contract and bounded
Glue canary pass. The latest exact production authorization was consumed once,
then failed closed before candidate selection on a harmless leading `./` in a
hash-verified baseline tar member. It cost an estimated `$1.024222`, published
no selected corpus or result, and its Glue job and IAM role are deleted. A
safe normalized-member correction passes locally; no replacement run, GPU
work, or model training is authorized.**

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

The corrected owner-authorized run used release
`792a8a604244282655ad5f1f9db69bf39b2ca885c477e2a8a8492eb4f1419de2`
and authorization
`34278ac482691fc9feab90be5c200c0f0c3d60fd2869cb2fd97667e8b61ba8ac`.
Glue run
`jr_d21c36b266cefc068da3bdc575779d8adb69ed447f69bd131ed4570e531ea193`
finished `SUCCEEDED` in 317 execution seconds, consumed 2,536 DPU-seconds, and
cost an estimated `$0.309956` at `$0.44` per DPU-hour. It stayed inside the
8-worker, 30-minute, `$1.76` envelope with zero retries.

The run also verified Spark `3.5.6-amzn-1` and every static setting, including
`spark.speculation=false`, after Glue supplied them at Spark startup. No live
Spark configuration mutation was required.

The run independently matched the complete input inventory of 2,772 objects,
`507,737,434,761` bytes, `68,671,669` documents, and `119,762,267,075`
tokens. Across all seven categories it selected 3,803 documents and
`7,026,092` tokens, produced seven selected Parquet objects plus 28 packed
uint16 objects, and passed 16-versus-23-partition parity in every category.
Selected output occupied `31,041,302` bytes; packed output occupied
`14,052,184` bytes, exactly two bytes per selected token.

The canonical result SHA-256 is
`e5da85af1b58c00f58de395e5041aaf5881d236570d4c7a1df49861babc17eff`.
The release, authorization, input, plan, baseline heads, result conservation,
content-addressed output inventories, and local evidence hashes all pass. The
consumed Glue job and dedicated IAM role were then deleted. The compact sealed
record is `reports/glue-final-corpus-canary-result-v0.1.json`; generated S3
payloads remain outside Git.

## Fail-safe historical production attempt

The earlier production package was authorized and consumed once. Glue stopped
after 51 execution seconds and 5,153 DPU-seconds because its production-only
entry point attempted to change static `spark.speculation` through
`SparkSession.conf` after Spark startup. AWS reported `CANNOT_MODIFY_CONFIG`
before input processing began. The run prefix and result prefix remained empty,
billing stopped, and independent evidence records no corrupt or partial
production output.

The permanent correction centralizes all static Spark settings, passes them via
Glue's startup `--conf`, and verifies the live values read-only. The corrected
canary above exercised that exact path before a replacement production package
was sealed. The failed authorization is not reusable.

## Fail-safe corrected production attempt

The subsequently authorized package was consumed exactly once by Glue run
`jr_ec1f3851e12eaced7cefb83af5eb1485b8485b90bc00fc8e5e8b1d41758c7258`.
It stopped after 83 execution seconds and 8,380 DPU-seconds, an estimated
`$1.024222` at `$0.44` per DPU-hour, before candidate selection began.

The pinned archive was intact, but its partial baseline member was named
`./partial-current-shard/tokens.uint16.bin`; the contract named the equivalent
`partial-current-shard/tokens.uint16.bin`. Python's exact `TarFile.getmember`
lookup therefore raised `KeyError`. Independent extraction verified the member
payload SHA-256 as
`029ab78f2546ac6b45f1f82024129c1b70998fcd5de5571dba5718a1e282b48a`,
exactly matching the contract. This was a path-lookup mismatch, not corpus
corruption.

The result and temporary prefixes are empty, there are zero category
checkpoints and zero selected objects, and the only two run-prefix objects are
hash-verified immutable baseline copies. Billing stopped and live absence
checks confirm the consumed Glue job and IAM role are gone. The sealed compact
record is `reports/glue-final-corpus-production-attempt-v0.1.json`.

The local correction resolves exactly one normalized POSIX tar member. It
accepts a harmless leading `./`, rejects absolute paths and parent traversal,
and rejects missing or ambiguous normalized matches. The consumed authorization
cannot be reused, and no replacement run has been inferred or launched.

## Production package boundary

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

The corrected canary calibration processed `301,453,386` input tokens in 2,536
DPU-seconds. Linear projection for the full accepted pool is 1,007,510
DPU-seconds, approximately 168 minutes on 100 DPU and `$123.140111`. The sealed
planning rule applies a 3x safety factor: 3,022,529 DPU-seconds, approximately
504 minutes and `$369.420211`. These are measured projections, not guarantees;
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
and refuses any job that already ran. The package described below was later
authorized and consumed by the fail-safe attempt above; that authorization is
not reusable.

The consumed reproducible package was bound to source commit
`3a5012995a19f076b8048b9b4a3c96b5fcac2ee5`. Its release fingerprint is
`1397391b580ae7cd5843c393146b387dcaa997d64998e13caf6fd67379210b27`, and
its five-entry `manifest.sha256` file hashes to
`12fc7c470193d228ea36fff78ca60e307ea014dddb5974ea5ffc9d708d0ea242`.

After the failure diagnosis and safe normalized-member correction, a fresh
non-authorizing replacement package was sealed from source commit
`555f80f8396eaa323da3d86430b33aa81b4e01f9`. Its release fingerprint is
`62c0285b84661ae49e80d2972c072df7b16f625c3b1f8750e0406a4b34b4d551`,
and its five-entry `manifest.sha256` file hashes to
`45f6abbb997bcadc231ce6a464772bb033bb26667a28d869547f493e766c776e`.
All five replacement files, every ZIP member, the exact source binding, release
validator, dedup-result binding, plan binding, and generated approval text pass
independent integrity checks. The package itself created no authorization; it
was later bound to the fresh single-use authorization and successful production
run recorded below.

After the normalized-member correction, all 18 focused final-corpus tests
(one environment-dependent skip), all 343 collected repository tests (three
environment-dependent skips), and repository-wide Ruff pass.

## Successful production materialization

The fresh authorization SHA-256
`064fccf72df4f1489c3998fe7c613bc7f75f494028bec2a8b462212d140ad4b1`
was consumed exactly once by Glue run
`jr_fd32da9a73c98b3bff419c2cb7165e2591972b1578512fc6b4764b05caaf0144`.
The 100-DPU Glue 5.1 job completed on attempt zero in 2,527 execution seconds,
using 252,774 DPU-seconds. At the pinned `$0.44` per DPU-hour rate, compute was
approximately `$30.894600`, well below the 12-hour/`$528` boundary.

All seven immutable category checkpoints sealed successfully:

| Category | Final documents | Packed tokens | Manifest shards |
| --- | ---: | ---: | ---: |
| Science and education | 2,850,104 | 24,000,002,963 | 51 |
| Code | 13,155,027 | 16,000,001,896 | 32 |
| Reference | 10,459,423 | 12,000,000,149 | 24 |
| Questions and discussion | 14,430,872 | 12,000,000,381 | 24 |
| Open web | 3,983,935 | 8,000,000,365 | 16 |
| Books | 184,378 | 4,000,008,765 | 8 |
| Government, legal, and patents | 753,898 | 4,000,000,215 | 8 |
| **Total** | **45,817,637** | **80,000,014,734** | **163** |

Every category stays below its 131,072-token document-boundary overshoot cap;
the combined overshoot is 14,734 tokens. The trainer consumption limit remains
exactly `80,000,000,000`. Manifest
sequences are contiguous from 0 through 162. Packed output occupies
`160,000,029,468` bytes, exactly two bytes per uint16 token. Selected Parquet
plus packed output occupies `503,157,241,754` bytes, below the 750GB cap.

Independent post-run verification downloaded the sealed result, standalone
selection result, shard manifest, and all seven checkpoints. Every local file
matches its S3 metadata SHA-256. The production validators pass exact release,
authorization, plan, dedup-result, runtime, category, baseline, output-cap, and
document/token/byte conservation bindings. All seven selected S3 inventories
match their sealed object counts, bytes, and canonical inventory hashes. The
163-object packed S3 inventory exactly matches the manifest at
`160,000,029,468` bytes and canonical inventory SHA-256
`b83a6cd0576126a0b860191cd2bc7c5adb3f85001b737ac99283ff709578b10c`.
The canonical sealed production result SHA-256 is
`c5b1c6aa1fbc7173bac8ba91bd7cdee55d12c2be28725598f7d7a2798a484869`.

After evidence was sealed, the consumed Glue job and bounded IAM role were
deleted and independently confirmed absent. S3 corpus and result objects remain
preserved. The compact repository record is
`reports/glue-final-corpus-production-result-v0.1.json`; the 15-file external
evidence manifest hashes to
`810716bccae1be3e041717ae51f89e314b76372d80dcdcd97411c5a2c976db3c`.
Post-run focused tests pass 13/13, the full repository suite passes 343/343,
and repository-wide Ruff passes.

## Next boundary

Final 80B corpus materialization and sealing are complete. The next roadmap
boundary is a bounded 4B multi-GPU learning pilot using this exact tokenizer,
architecture, optimizer decision, isolated-attention path, and sealed corpus.
The pilot must first prove distributed loading, throughput, memory, checkpoint
resume, held-out improvement, and failure recovery. It requires a new explicit
owner approval and does not authorize full 4B/80B pretraining. Full pretraining,
instruction tuning, alignment, quantization, API hosting, and deployment remain
later independent gates.
