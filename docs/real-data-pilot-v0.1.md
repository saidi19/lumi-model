# Lumi real-data pilot plumbing v0.1

Status: **deterministic packing, 24-source stratification, crash-safe resume,
checkpointed training, a real-corpus Apple-Silicon smoke run, and 50M-token
input materialization pass; paid 150M training has not started**

## Purpose

This gate proves that the frozen tokenizer, sealed corpus, packed-data format,
model, optimizer, and checkpoints work together before renting a GPU. It does
not measure model quality and it is not the 150M pilot.

## Packed-data contract

`scripts/pack_pilot_data.py` streams JSONL records through the local frozen
tokenizer and writes fixed-length, little-endian `uint32` shards. Every nonempty
document begins with token ID 3 (`<|doc|>`). Shards contain only complete
sequences and are read through a memory-mapped `PackedTokenDataset` rather than
being loaded into RAM.

The manifest pins:

- the input path, byte count, and SHA-256;
- the tokenizer files, hashes, vocabulary size, and document-token ID;
- sequence length, shard capacity, target, and build fingerprint;
- every shard's byte count, token count, sequence count, and SHA-256.

The sealed sample is intentionally written in category/component blocks, so a
linear 50M-token prefix would be dominated by the first source. For scaled
pilots, `--source-mix-manifest` pins the sealed sample manifest, verifies that
its byte count and SHA-256 match the JSONL input, and allocates exact per-source
token quotas from all 24 planned source byte weights using largest remainders.
The packer scans the clustered file through the final required source while
skipping already-filled strata. Training then shuffles the resulting fixed
sequences deterministically. The small 8,192-token Mac smoke intentionally used
linear mode because it tested plumbing rather than mixture quality.

Packing state commits only after shard bytes are flushed. A resume reconciles
each shard to the durable token count, truncates uncommitted crash-tail bytes,
and continues from the committed JSONL byte offset. Tests simulate an exception
mid-build and prove that resumed shards are byte-identical to a clean build.

## Training checkpoint contract

`scripts/train_pilot.py` provides the correctness-first single-device pilot
runner. It records a fingerprint over the config, packed manifest, optimizer
settings, and seed. Each checkpoint contains:

- self-contained Hugging Face model/config code and safetensors weights;
- AdamW optimizer state plus CPU and active CUDA/MPS RNG state;
- optimizer step, token count, and exact shuffled-sequence cursor;
- the matching run fingerprint.

Metrics beyond the latest durable checkpoint are discarded before resume. If a
crash happens after the checkpoint directory is committed but before the outer
state file is updated, the newest matching checkpoint is discovered and
recovered. A regression test proves that `2 steps + resume to 4` produces the
same losses and model tensors as an uninterrupted four-step CPU run.

On CUDA, the runner uses fused AdamW when PyTorch supports it and records peak
allocated/reserved GPU memory, tokens per second, and checkpoint duration. This
makes the short cloud pilot a measurable cost and capacity gate rather than a
blind training run.

## Real Mac smoke result

The source was the sealed 7,709,921,712-byte sample with SHA-256
`80bea9f9634a8daca542b2ffd0aad1c8c994a7cbeef909a5e529b6272671102a`.
The selected tokenizer was the frozen 65,536-token artifact.

The smoke pack produced 8,192 tokens in 64 sequences of length 128 across two
4,096-token shards. Their SHA-256 values are:

- `b6de338a3ae13a5acfaef2b3299262fe9cabc29ee477950e7563a271eda00516`
- `dfa12506ae4aab2e3e78d3fbd997c2806c64b18ef0f5ac95facc00f929476598`

The cached `--resume` check revalidated the source, tokenizer, manifest, and
shards and returned the same result.

`configs/lumi-real-smoke.json` is deliberately only 4,292,928 parameters while
retaining the real 65K vocabulary. On Apple MPS it completed step 1, saved a
checkpoint, then a fresh process loaded that checkpoint and completed step 2.
It processed 256 real tokens total. Losses were 11.089542 and 11.104599. Their
short-run direction is not a learning claim because the steps use different
batches. The two measured steps ran at 645.1 and 738.3 tokens/s. Resume also
accepted the identical config from a different filesystem path, proving the
run identity is content-based rather than tied to the Mac worktree. These
figures are smoke diagnostics, not H100 throughput estimates.

The local artifacts live outside Git under
`~/Documents/lumi-data/real-pilot-smoke-v0.1/`. Generated data, optimizer state,
and weights remain excluded by repository policy.

## Stratified canary and 50M artifact

A 24,576-token real-data canary exercised the clustered-file scan and exact
largest-remainder quotas across all 24 sources and all eight categories. It
produced 192 sequences in three shards, and cached `--resume` validation passed.
Its build fingerprint is
`6e1f8fb5375c10df518d52af5f1eb7d6f9a988e63e3fc8416c919a8df3e53c7e`.

The cloud-pilot input was then materialized locally at
`~/Documents/lumi-data/real-pilot-50m-v0.1/packed/`. It contains exactly
49,999,872 tokens, or 24,414 complete 2,048-token sequences, in 48 hashed
shards. All 24 source quotas and all eight category totals validate exactly,
and a cached `--resume` check passed. The packed manifest SHA-256 is
`a8d964164221f81acb5805de767ea0cc1d5baf75eda31462ad01badae4b9522c`;
the build fingerprint is
`2a7187332b1f42f7290310091d11ad0d8cca3a2005b4d6b9df402a98e3347645`.

| Category | Packed tokens |
|---|---:|
| Science and education | 14,849,961 |
| Code | 9,899,975 |
| Reference | 7,424,981 |
| Questions and discussion | 7,424,981 |
| Open web | 4,949,987 |
| Books | 2,474,994 |
| Government, legal, and patents | 2,474,994 |
| Bengali reference slice | 499,999 |

This artifact proves mixture and training plumbing, not model quality. Within
each source block it deterministically takes the earliest documents needed to
meet that source's quota. A later 1B-3B-token quality pilot must add a pinned
within-source hash shuffle and held-out, domain-stratified validation before it
can justify a 4B scale-up.

## Verification

- all 77 repository tests pass;
- Ruff and `git diff --check` pass;
- interrupted packing resumes byte-identically;
- interrupted training resumes tensor-identically in the deterministic CPU
  regression;
- a real Apple-MPS checkpoint was loaded in a fresh process and advanced.

## Next paid gate

The next run is the 150,625,536-parameter model with 2,048-token sequences and
an effective 49,999,872-token plumbing target (24,414 complete sequences). It
must run on a cloud GPU, not the 16GB Mac. Before launch, record the selected
GPU, hourly price, expected maximum runtime/cost, storage, image, and shutdown
rule. The run must demonstrate finite loss, stop/restart recovery, measured
tokens/s, peak memory, checkpoint duration, and cost. Only then can the 1B-3B
token 150M pilot or any 4B run be approved.

The completed materialization is reproducible with:

```bash
python scripts/pack_pilot_data.py \
  --input "$LUMI_DATA/tokenizer-sample-v0.3/sample.jsonl" \
  --source-mix-manifest "$LUMI_DATA/tokenizer-sample-v0.3/manifest.json" \
  --tokenizer "$LUMI_DATA/tokenizer-candidates-v0.2/lumi-tokenizer-65k" \
  --model-config configs/lumi-150m.json \
  --output "$LUMI_DATA/real-pilot-50m-v0.1/packed" \
  --target-tokens 50000000 \
  --resume
```
