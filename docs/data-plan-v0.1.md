# Lumi data plan v0.1

Status: **candidate; capacity, bounded retention, S3 storage, and local
distributed-acquisition gates pass. The first production prefix is preserved;
the Glue canary failed safe, and its EMR replacement, production resume, and
GPU training are not approved**

## Default source direction

Use the Common Pile v0.1 filtered source family as the primary pool. The 521GB
[`common-pile/comma_v0.1_training_dataset`](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset)
is the pinned mix reference and was sufficient for the first streaming canary.
It must not be the source of the development or final corpus because its
consolidated rows contain only `text` and discard the original Dolma IDs,
URLs, and license metadata.

Starting with the 1GB-2GB development sample, materialize from the individual
`common-pile/*_filtered` component repositories. Their rows retain `id`,
`source`, and metadata such as provenance, URL, and usually document license.
All 23 candidate component schemas and revisions are pinned in
[`configs/component-sources-v0.1.json`](../configs/component-sources-v0.1.json).
The overall dataset has no single unified license; license values or an approved
source-level public-domain basis must therefore be retained per document.
A bounded first-shard audit found IDs, provenance, and a non-null license or
documented source-level basis in all 44,061 inspected rows; the machine-readable
counts and limitations are in
[`reports/component-provenance-audit-v0.1.json`](../reports/component-provenance-audit-v0.1.json).

This is a safer and more reproducible default than treating FineWeb-Edu or a
random Common Crawl dump as the entire corpus. FineWeb-Edu remains an optional
ablation source, not part of v0.1: its dataset is ODC-By, but its card also makes
use subject to Common Crawl terms and the underlying pages retain their own
rights.

Personal use does not remove the need for provenance. It matters for deletion
requests, accidental secrets/PII, later model sharing, and any future change in
Lumi's use.

The consolidated canary source is pinned to commit
`5afc546db324e7f39f297ba757c9a60547151e7c`; a floating `main` revision is not
accepted by the sampler.

## Candidate 80B mix

| Domain | Share | Tokens at 80B | Common Pile components |
|---|---:|---:|---|
| Science and education | 30% | 24B | peS2o, arXiv papers, PubMed, DOAB, LibreTexts, OER Commons |
| Code | 20% | 16B | Stack v2 educational filter, Python enhancement proposals |
| Reference | 15% | 12B | Wikimedia, WikiTeam, Library of Congress |
| Questions and discussion | 15% | 12B | Stack Exchange, Ubuntu IRC |
| Openly licensed web | 10% | 8B | CCCC filtered |
| Books | 5% | 4B | pre-1929 books, Project Gutenberg, Pressbooks, Public Domain Review |
| Government, legal, patents | 5% | 4B | case law, regulations, Hansard, USGPO, USPTO |

These are sampling targets, not permission to blindly repeat a small component.
The mix is frozen only after token counts are recomputed with the final Lumi
tokenizer and quality/duplication statistics are available. The machine-readable
candidate is [`configs/data-mix-v0.1.json`](../configs/data-mix-v0.1.json).

The planning-only source-capacity gate now inventories all 23 exact revisions:
their official cards report 1.735TB of UTF-8 text and their repository trees
total 499.82GB compressed. Category-specific frozen-tokenizer fertility gives
a 405.37B-token planning estimate. Every category clears a 1.5-times candidate
floor, but open web has only 11.62% estimated headroom. These are upstream
metadata estimates, not post-filter unique-token measurements; see
[`corpus-capacity-gate-v0.1.md`](corpus-capacity-gate-v0.1.md).

The bounded follow-up measured 192,528,033 raw bytes from the three tightest
categories without persisting text. It accepted 45,981,609 frozen-tokenizer
tokens and projects 13.210B open-web, 21.863B questions/discussion, and 25.153B
reference tokens after the current filters. All three floors pass; see
[`corpus-retention-canary-v0.1.md`](corpus-retention-canary-v0.1.md). This is
still first-shard evidence, not authorization for a full-source download.

The 130B-150B planning envelope is not a required local dataset size. It is a
candidate-token safety buffer around the final 80B mix. Bulk acquisition must
be adaptive: stop when verified per-category quotas, deduplication reserve, and
held-out requirements are met instead of downloading an arbitrary fixed total.
The sealed storage design makes that rule exact: seven 1.5-times category
floors total 120B tokens, while 150B is only a fail-closed hard maximum. Its
future cloud forecast peaks at 1.0375TB under a 2TB hard cap; the local planning
evidence itself is only 16,750 bytes. See
[`corpus-object-storage-v0.1.md`](corpus-object-storage-v0.1.md).

## Staged acquisition

### 1. Tokenizer sample — no GPU

- Completed first gate: a 25MB seven-domain canary, documented in
  [`data-canary-v0.1.md`](data-canary-v0.1.md).
- Completed provenance correction: a 10MB component-repository canary retained
  required IDs, licenses/basis, URLs, and collection provenance in every
  accepted record; see
  [`provenance-canary-v0.2.md`](provenance-canary-v0.2.md).
- Completed all-component development gate: 1,000,895,350 accepted UTF-8 text
  bytes across all 23 pinned components passed source/category quotas, language
  filtering, secret scanning, provenance checks, MinHash near-deduplication,
  hashed benchmark decontamination, full post-write validation, and manual PII
  snippet review; see
  [`development-sample-v0.4.md`](development-sample-v0.4.md).
- Next, add a small, explicitly licensed Bangla/multilingual tokenizer slice.
- Use the metadata-preserving component repositories, audit null/malformed
  license rates, and treat CCCC as a documented source-level-license exception:
  its card describes whole-domain license review, but 0 of 10,000 audited raw
  and filtered rows contained the per-document license field claimed by the
  card.
- Disk-backed SQLite MinHash state, exact-ID/content indexes, and byte-offset
  validation checkpoints are now implemented with network retry-forever mode.
  A 25MB all-component network canary passed with 5,055 documents, all 23
  sources, a recorded network retry, and both fresh and cached validation.
- Then stream a stratified 6GB-12GB uncompressed tokenizer-only sample across
  every domain.
- Keep document IDs, source component, revision, license metadata, and hashes.
- Train both 32K and 65K tokenizers from exactly the same sample.
- Do not store the sample or generated tokenizer in Git.

### 2. Real 150M pilot — 1B to 3B tokens

- Freeze the tokenizer first.
- Apply language/quality filters, secret and PII scanning, exact and near
  deduplication, document-boundary insertion, and benchmark decontamination.
- Train the 150.626M model in resumable shards and record validation loss by
  domain. A short 50M-token plumbing run comes before the 1B-3B pilot.

### 3. Throughput/cost calibration

- Run a brief H100 BF16 benchmark with the real tokenizer, packed 4K sequences,
  optimizer, checkpointing, and data loader.
- Record tokens/second, MFU, peak memory, checkpoint time, and dollars per
  billion tokens.
- Recalculate whether 80B, a lower token count, or a smaller model fits the
  USD 6,000 ceiling. Estimates do not approve the full run.

### 4. Final corpus — 80B first target

- Completed: bounded canaries from open web, questions/discussion, and reference
  pass their post-filter capacity floors without retaining corpus text.
- Completed locally: provider-neutral private object storage, immutable
  manifests, resumable workers, lifecycle/storage caps, a removal ledger, and
  adaptive stop conditions.
- Completed: AWS S3 `us-east-1` pricing and the bounded
  put/get/resume/range/delete canary pass, and both temporary buckets are
  deleted; see
  [`aws-s3-provider-canary-v0.1.md`](aws-s3-provider-canary-v0.1.md).
- Completed: the exact path, size, revision, Git blob ID, and LFS SHA-256 for
  all 758 files across all 23 pinned repositories are metadata-only sealed; see
  [`source-file-manifest-v0.1.md`](source-file-manifest-v0.1.md).
- Completed: the pinned CCCC evidence and fail-closed include/exclude packet;
  the owner selected `include_with_caveat`. See
  [`cccc-owner-decision-v0.1.md`](cccc-owner-decision-v0.1.md).
- Completed and preserved: the owner-authorized single-writer acquisition
  produced 237,769,640 accepted tokens across 38,349 documents before it was
  safely stopped for inadequate parallel throughput. Its two sealed shards,
  partial shard, SQLite state, and 38,349 immutable resume anchors pass
  independent verification; the EC2 instance and volumes are gone.
- Completed locally: a Spark/Glue execution path preserves source-order exact
  and near-deduplication semantics and resumes after the verified prefix. Real
  local Spark parity, repeatability, fail-closed limits, release packaging, and
  the complete repository test suite pass. See
  [`distributed-corpus-glue-v0.1.md`](distributed-corpus-glue-v0.1.md).
- Completed safely: the exact owner-authorized 8-DPU Glue canary stopped in
  dependency bootstrap because Glue 5.1's Python 3.11 cannot install pinned
  Lingua 2.2.0. It read no source records, wrote no corpus objects, started no
  training, cost an estimated $0.0374, and all temporary Glue/IAM resources are
  deleted.
- Completed locally: an EMR Serverless 7.13.0 replacement preserves the same
  eight-source/all-category bounds and Spark semantics while using a
  digest-pinned Python 3.12 image. Package, authorization, cost, capacity, and
  request gates remain non-authorizing; see
  [`distributed-corpus-emr-serverless-v0.1.md`](distributed-corpus-emr-serverless-v0.1.md).
- Completed safely: the first exact CodeBuild approval was consumed by one
  build that failed in `PRE_BUILD` before Docker because the S3 source-version
  variable was wrong. No image or EMR job was created, the conservative compute
  upper bound is $0.03, and every temporary resource is confirmed deleted.
- Completed safely: the corrected exact CodeBuild image passed, and one bounded
  EMR job resolved it, started Spark plus three executors, and consumed 1.181
  vCPU-hours and 4.724 memory-GB-hours. Its source tasks then proved the default
  EMR Serverless external-network block while connecting to Hugging Face. The
  job was cancelled at 273 seconds, no retry/parity/production write occurred,
  combined build-plus-EMR compute is estimated at $0.139446578, and all
  temporary resources are gone or terminated.
- Completed locally: the replacement source transport stages exact pinned
  files under SHA-256-addressed private S3 keys and makes EMR consume only those
  objects. Conditional writes, stored checksum/metadata verification,
  interrupted range resume, wrong-object refusal, and real Spark/sequential
  parity pass 76 corpus-focused tests; the complete repository suite passes
  236/236. The unchanged eight-source plan covers
  all seven categories and 1,456,514,440 compressed bytes; no cloud write or
  paid compute occurred. See [`s3-staged-corpus-v0.1.md`](s3-staged-corpus-v0.1.md).
- Next: package the clean commit, price the exact eight-object staging plus one
  EMR parity canary, and request a new exact approval. The clean EMR package is
  now complete and independently verified at source commit `9c67a76`; its
  release fingerprint is
  `0d2043bceacd06919fd3efe1b323f435ecdf4949740b7482cf8820de8ca4af88`.
  The remaining local prerequisite is the bounded internet-facing staging
  runner plus its exact compute cap. Distributed production remains a separate
  later approval after measured parity, throughput, and cost.
- Materialize only the selected, cleaned shards in private object storage.
- Pin every upstream dataset revision; never train from a floating `main`.
- Maintain a removal ledger mapping source document IDs to processed shards.
- Hold back domain-stratified validation data before global deduplication.
- Increase toward 100B only if measured cost and loss curves justify it.

## Hard exclusions

- secrets, API keys, private keys, credentials, and obvious personal data;
- malware payload collections and generated/vendor code dumps;
- benchmark test sets or near-duplicates of evaluation prompts;
- documents without stable provenance in the tracked source pool;
- repeated low-quality text added only to hit a token target;
- any data whose access terms were not recorded at the pinned revision.

## Storage boundary

A 1TB Mac can hold tokenizer samples, the 192MB text-free retention evidence,
and selected development shards, but raw source pools, multiple dedup stages,
final shards, and checkpoints together can exceed that space. Even an 80B
uint16 token stream is roughly 160GB before indexes, metadata, optimizer state,
or checkpoints. Full corpus artifacts belong in versioned private object
storage. The Mac should keep manifests, small samples, and bounded evidence.

Official references:

- [Common Pile organization and collections](https://huggingface.co/common-pile)
- [Common Pile paper](https://arxiv.org/abs/2506.05209)
- [Common Pile collection code](https://github.com/r-three/common-pile)
- [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Wikimedia dump licensing](https://dumps.wikimedia.org/legal.html)
- [The Stack v2 terms and provenance](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup)
