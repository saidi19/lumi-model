# Lumi data plan v0.1

Status: **candidate; no training corpus has been downloaded or approved**

## Default source direction

Use the filtered
[`common-pile/comma_v0.1_training_dataset`](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset)
as the primary source pool. It is a 521GB consolidated version of Common Pile
v0.1 and exposes the mix used for the Comma models. Common Pile was built from
public-domain and openly licensed text, with source-level collection code and
provenance. Its source pool is much larger than Lumi's 80B-100B-token target, so
Lumi can select a quality-focused subset instead of downloading arbitrary web
text.

This is a safer and more reproducible default than treating FineWeb-Edu or a
random Common Crawl dump as the entire corpus. FineWeb-Edu remains an optional
ablation source, not part of v0.1: its dataset is ODC-By, but its card also makes
use subject to Common Crawl terms and the underlying pages retain their own
rights.

Personal use does not remove the need for provenance. It matters for deletion
requests, accidental secrets/PII, later model sharing, and any future change in
Lumi's use.

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

## Staged acquisition

### 1. Tokenizer sample — no GPU

- Stream a stratified 6GB-12GB uncompressed sample across every domain.
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

A 1TB Mac can hold tokenizer samples and selected processed shards, but keeping
raw source pools, multiple dedup stages, final shards, and checkpoints together
can exceed that space. Full corpus artifacts belong in versioned private object
storage. The Mac should keep manifests, small samples, and development shards.

Official references:

- [Common Pile organization and collections](https://huggingface.co/common-pile)
- [Common Pile paper](https://arxiv.org/abs/2506.05209)
- [Common Pile collection code](https://github.com/r-three/common-pile)
- [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Wikimedia dump licensing](https://dumps.wikimedia.org/legal.html)
- [The Stack v2 terms and provenance](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup)
