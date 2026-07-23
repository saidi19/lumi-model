# Lumi Final 80B Corpus Gate v0.1

Status: **the corrected marker-aware production corpus is sealed and
independently verified. It contains 45,768,283 documents, exactly 45,768,283
token-ID-3 (`<|doc|>`) markers, 79,954,240,081 content tokens, and
80,000,008,364 packed tokens across 160 content-addressed uint16 shards. The
trainer consumption boundary remains exactly 80B tokens. The older 163-shard
markerless materialization remains quarantined. No GPU work or model training
was authorized.**

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
- historical immutable prefix: `38,349` documents and `237,769,640` content
  tokens, retained only as a deduplication anchor and excluded from the
  marker-aware training stream;
- frozen 65K tokenizer SHA-256
  `1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee`;
- the exact data-mix, filter policy, benchmark registry, and permanent
  training-exclusion ledger files.

The plan accepts only the verified production result with all seven 120B
category floors complete and zero deficit. It also binds the exact release,
authorization, input-index, and map-checkpoint hashes from that run.

## Final mixture

| Category | Final target | Marker-aware selection target |
|---|---:|---:|
| Science and education | 24B | 24B |
| Code | 16B | 16B |
| Reference | 12B | 12B |
| Questions and discussion | 12B | 12B |
| Open web | 8B | 8B |
| Books | 4B | 4B |
| Government, legal, and patents | 4B | 4B |
| **Total** | **80B** | **80B** |

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
The plan treats the three old prefix artifacts as deduplication anchors only
and expects 160 fresh approximately 500M-token uint16 shards. The marker-aware
manifest therefore has exactly 160 entries; no old markerless shard is copied.

The shard validator requires:

- contiguous shard sequence numbers and unique object keys/hashes;
- one token ID 3 (`<|doc|>`) before every document, with
  `tokens = content_tokens + documents`;
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

The marker-aware owner-authorized run used release
`ef29436069e65fb5a0ff9d37fbe82c6298dc76ac5ea557f647dc8dcafb1b985f`
and authorization
`6f7fe1316a6d72dbe9fa4bb255aea764e1be3452d07cbb7164334f79b5e5accb`.
Glue run
`jr_f4b069bdf6436bc796317ca9d3930ac4e311c9d7a72936d5e2aa58b803f77674`
finished `SUCCEEDED` in 287 execution seconds, consumed 2,303 DPU-seconds, and
cost an estimated `$0.281478` at `$0.44` per DPU-hour. It stayed inside the
8-worker, 30-minute, `$1.76` envelope with zero retries.

The run also verified Spark `3.5.6-amzn-1` and every static setting, including
`spark.speculation=false`, after Glue supplied them at Spark startup. No live
Spark configuration mutation was required.

The run independently matched the complete input inventory of 2,772 objects,
`507,737,434,761` bytes, `68,671,669` documents, and `119,762,267,075`
content tokens. Across all seven categories it selected 3,797 documents,
`7,023,704` content tokens, and 3,797 document markers, for `7,027,501`
packed tokens. It produced seven selected Parquet objects plus 28 packed
uint16 objects and passed 16-versus-23-partition parity in every category.
Selected output occupied `31,038,392` bytes; packed output occupied
`14,055,002` bytes, exactly two bytes per packed token.

The canonical result SHA-256 is
`900589aa54c7d8b4b273bb64ed3f7b8870b73e5b502b888967d5faf9885ea5ae`.
The release, authorization, input, plan, baseline heads, result conservation,
content-addressed output inventories, and local evidence hashes all pass.
Independent local inspection downloaded every packed object, recomputed all
28 SHA-256 digests, confirmed that every shard begins with token ID 3, and
counted exactly one ID 3 per selected document. The consumed Glue job and
dedicated IAM role were then deleted and confirmed absent. The compact sealed
record is `reports/glue-final-corpus-canary-result-v0.1.json`; generated S3
audit payloads remain outside Git.

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

## Corrected marker-aware production package boundary

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
  checkpoints, and exactly 160 fresh approximately 500M-token marker-aware
  manifest entries;
- a `750,000,000,000`-byte hard ceiling on newly materialized selected
  Parquet and packed uint16 shards;
- no GPU permission and no model-training permission.

The marker-aware canary calibration processed `301,453,386` input content
tokens in 2,303 DPU-seconds. Linear projection for the full accepted pool is
914,943 DPU-seconds, approximately 153 minutes on 100 DPU and `$111.826367`.
The sealed planning rule applies a 3x safety factor: 2,744,828 DPU-seconds,
approximately 458 minutes and `$335.478978`. These are measured projections,
not guarantees; the 12-hour/`$528` boundary remains the actual fail-closed
limit. Projected maximum new output is `513,342,209,894` bytes, below the
750GB cap.

Before any row is selected, the runtime independently inventories every input
object, validates the exact release, plan, marker-aware canary report,
authorization, Python 3.11 and Spark 3.5.6 bindings, and assigns every input
object to exactly one category. It then hash-ranks and selects each category,
prepends token ID 3 to every document, writes an immutable selected inventory,
packs content-addressed uint16 shards, and seals a category checkpoint before
advancing. No historical baseline shard is copied into the training stream.
Final publication is allowed only after selection-result, shard-manifest,
document/content-token/marker/byte conservation, category, runtime, and
output-cap validation all pass.

The launcher is dry-run by default. Its one-purpose IAM policy can read only
the accepted prefix and exact bootstrap objects, can write only the
authorization-derived run/temp/result prefixes, uses an immutable launch claim,
and refuses any job that already ran. The historical packages described below
were consumed under the older markerless contract; their authorizations are not
reusable.

The marker-aware production package was sealed from source
commit `bdd656e14cee6b995f3e5196933285b4f54897b9`. Its release fingerprint is
`e097ae33ed91a1b998843bebe2452a205b252fcb6e1ef774dce481f9ad02d50d`,
and its five-entry `manifest.sha256` file hashes to
`7fc45e2159a2619323426a8bedf15cd107a7e0a3c43f14d3c56a8fca0f6b13d8`.
The package itself cannot authorize paid work. It was later bound to the fresh
single-use owner authorization and successful production run below; GPU and
model-training authorization remained false.

## Verified corrected marker-aware production materialization

Authorization
`6475dda549a1a8382bb6f777b7ddca3fe315c1a2822be6ed4609918e60cccc08`
bound exactly one 100-DPU Glue run to source commit
`bdd656e14cee6b995f3e5196933285b4f54897b9`, release
`e097ae33ed91a1b998843bebe2452a205b252fcb6e1ef774dce481f9ad02d50d`,
plan `aab292a514b2a7a046a3e4f0daf2fb75bb6f899ff86798bdfac729a145bc9470`,
and dedup result
`dbfbf9fe258ec97d5bd844c165157d41d328d0571cbbada5fd4d67750c46b742`.
Glue run
`jr_73c74effa6f50bf6faec901078e47bb93563d270d72625de0f13a86ee8f731e3`
finished `SUCCEEDED` on attempt zero in 2,745 execution seconds using 274,571
DPU-seconds. At `$0.44` per DPU-hour, compute was approximately `$33.558678`,
well below the 12-hour/`$528` boundary.

All seven immutable category checkpoints sealed:

| Category | Documents / markers | Content tokens | Packed tokens | Shards |
| --- | ---: | ---: | ---: | ---: |
| Science and education | 2,839,568 | 23,997,163,839 | 24,000,003,407 | 48 |
| Code | 13,144,364 | 15,986,856,201 | 16,000,000,565 | 32 |
| Reference | 10,450,031 | 11,989,550,626 | 12,000,000,657 | 24 |
| Questions and discussion | 14,414,153 | 11,985,586,706 | 12,000,000,859 | 24 |
| Open web | 3,982,032 | 7,996,018,342 | 8,000,000,374 | 16 |
| Books | 184,370 | 3,999,816,243 | 4,000,000,613 | 8 |
| Government, legal, and patents | 753,765 | 3,999,248,124 | 4,000,001,889 | 8 |
| **Total** | **45,768,283** | **79,954,240,081** | **80,000,008,364** | **160** |

The 8,364-token document-boundary overshoot is within the fixed cap. Packed
output is `160,000,016,728` bytes, exactly two bytes per token. The 7,478
selected Parquet objects occupy `343,924,248,990` bytes; selected plus packed
output occupies `503,924,265,718` bytes, below the 750GB cap.

Independent local verification:

- ran the production result and all seven category validators against the exact
  release, authorization, plan, and dedup binding;
- matched all seven selected S3 inventories by object count, bytes, and
  canonical inventory SHA-256;
- matched every one of the 160 packed S3 keys, sizes, SHA-256 metadata, and S3
  checksums against the sealed manifest;
- range-read every shard and confirmed its first uint16 token is ID 3
  (`03 00` little-endian);
- proved aggregate and per-shard
  `tokens = content_tokens + document_markers`, with one marker per document;
- matched all seven checkpoint JSON objects and all three final-result JSON
  objects to S3 metadata and checksums; and
- verified the exact 80,000,000,000-token trainer boundary and output cap.

The canonical sealed result SHA-256 is
`0423e29b05e0c31ce73bf6066a00b3ce198eb442cf7143dfc953d4d5b2688e52`;
the selection-result SHA-256 is
`b017a6fec5c6ca1685c5dcd883d8afc48d36fd18cac07d573514ad77307253ee`.
The 33-file external evidence manifest hashes to
`25a5848f22b71241e1e502a0a5414f696056e82d885d795606b5139d0fab7a28`.
After evidence was sealed, the consumed Glue job and bounded IAM role were
deleted and confirmed absent. S3 corpus and result objects remain preserved.
The compact repository record is
`reports/glue-final-corpus-marker-aware-production-result-v0.1.json`.

Post-run focused tests pass with 24 passed, one environment skip, and the known
Mac two-process Gloo test deselected. The near-full suite passes with 356
passed, three environment skips, and that same one deselection; Ruff is clean.
Running the deselected test separately reached no test assertion because
PyTorch repeatedly failed reverse DNS for the localhost IPv6 address and hit
its 120-second subprocess timeout. The corrected-manifest exact-tail case
therefore remains an explicit Linux production-preflight check rather than
being misreported as a local pass.

### Historical markerless package provenance

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

## Historical markerless production materialization

The run below successfully proved deterministic 80B selection, distributed
sharding, checkpointing, and S3 sealing under the older contract. A later
byte-level audit found that all packed shards contain zero `<|doc|>` markers
because the mapper stored content-only `token_bytes` and the old packer copied
those bytes verbatim. These 163 shards are therefore quarantined and are not
training-ready. The production trainer now rejects their manifest.

The historical authorization SHA-256
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

Independent post-run verification under the older markerless contract
downloaded the sealed result, standalone
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
preserved as historical evidence, not as accepted training input. The compact
repository record is
`reports/glue-final-corpus-production-result-v0.1.json`; the 15-file external
evidence manifest hashes to
`810716bccae1be3e041717ae51f89e314b76372d80dcdcd97411c5a2c976db3c`.
Post-run focused tests pass 13/13, the full repository suite passes 343/343,
and repository-wide Ruff passes.

## Next boundary

The corrected 80B corpus data gate is complete. The immediate free/local step
is to rebind the production preflight and bounded learning-pilot fixture from
the quarantined markerless manifest to this exact corrected result and
160-shard manifest, then rerun their contract, exact-tail, checkpoint/resume,
and trainer-readability tests.

After that preflight passes, the roadmap returns to the bounded 4B multi-GPU
learning pilot using this exact tokenizer, architecture, optimizer decision,
isolated-attention path, and corrected corpus. The paid pilot and full 4B/80B
pretraining each require separate owner approval. Instruction tuning,
alignment, quantization, API hosting, and deployment remain later independent
gates.
