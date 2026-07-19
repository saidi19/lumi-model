# Lumi production distributed trainer v0.1

Status: **local DDP/FSDP correctness gate passed; CUDA/NCCL gate not yet run**

This stage adds the training infrastructure needed before a 4B pilot. It does
not authorize a paid GPU, a 4B pilot, or the 80B-token pretraining run.

## Implemented contract

`scripts/train_distributed.py` is launched with `torchrun` and supports:

- synchronous DDP for a replicated reference path;
- classic FSDP `FULL_SHARD`, transformer-layer auto-wrapping, original
  parameters, bounded all-gathers, and backward prefetch;
- explicit CPU/Gloo or CUDA/NCCL topology validation;
- FP32 local preflight and BF16-only paid CUDA training;
- the frozen Lumi document-isolated attention inputs, including
  FlashAttention varlen metadata on CUDA;
- optional compilation of the inner Lumi module before DDP/FSDP wrapping;
- exact global-token accounting across rank-local batches and gradient
  accumulation;
- one shard-hash verification per node plus collective manifest-identity and
  local-visibility checks across every rank;
- immutable DCP model shards, per-rank optimizer/RNG state, rank-count pinning,
  SHA-256 manifests, atomic checkpoint publication, pruning, and exact resume;
- independent post-process loading of a distributed checkpoint into one plain
  CPU model and a canonical full-model tensor digest.

Historical single-device checkpoints and `pilot_training.py` keep their old
identity and format.

## Efficient deterministic data order

Every process reconstructs the same global batch and takes only its rank slice.
The checkpoint stores a single global sequence cursor, so no sampler pickle is
needed and every rank advances by exactly the same amount.

The earlier pilot helper rebuilt a full `randperm` for every microbatch. That
would become an avoidable CPU bottleneck at 80B-token scale. The distributed
trainer instead caches one integer permutation per epoch and only materializes
the current batch slice. It preserves the old deterministic order while making
steady-state lookup proportional to batch size. At 80B tokens and 4,096-token
sequences, an int64 cached permutation is approximately 156MB per rank.

## Fail-closed launch boundary

Without a separate authorization artifact, execution is limited to:

- CPU/Gloo;
- two processes;
- at most 20,000,000 parameters;
- at most eight optimizer steps;
- at most 2,000,000 processed tokens.

CUDA requires a run-fingerprint-bound owner authorization. The v0.1 paid schema
only permits `lumi_distributed_smoke_v0.1`, caps the approved boundary at 1,000
steps, and requires `full_4b_training=false`. It cannot authorize the full
4B/80B run.

## Reproducible local gate

The gate is run by:

```bash
PYTHONPATH=src python scripts/run_distributed_preflight.py \
  --workspace /path/outside/git/distributed-trainer-preflight-v0.1 \
  --python /path/to/python
```

It creates one deterministic 26,800-value tiny Lumi state and performs seven
two-process launches:

1. uninterrupted DDP through step 4;
2. DDP through step 2;
3. DDP resume through step 4 after an uncommitted metric and partial checkpoint
   are injected;
4. uninterrupted FSDP `FULL_SHARD` through step 4;
5. FSDP through step 2;
6. FSDP resume through step 4 after the same simulated interruption;
7. a one-step DDP `torch.compile` compatibility canary with a durable
   checkpoint and full-model digest.

The final verified M5/CPU result is:

| Path | Final loss | Global tokens/step | Exact metrics | Exact final model |
|---|---:|---:|---:|---:|
| DDP | 4.8144294024 | 64 | pass | `483f07c3edb4604acbf0a1ce84c896ece428eed31880c51541f7e0fd24c73c71` |
| FSDP `FULL_SHARD` | 4.8507766724 | 32 | pass | `76e5ec8bab0c9f5794cd5dd5b0a092b1ef774ad6d1769bcf65a794ebad3c9bc1` |

Reference and resumed loss, gradient norm, learning rate, token count, and final
model digest match exactly for every compared step. The test deliberately
publishes the step-2 checkpoint, removes the corresponding root state and metric
commit, adds a partial step-3 checkpoint and uncommitted metric, then resumes.
Both partial checkpoints are ignored, both uncommitted metric lines are trimmed,
and the missing step-2 metric is recovered from the sealed checkpoint. The
compile canary finishes at loss 4.8859286308 with 64 global tokens/step and
model SHA-256
`83e25ec2b373ab1a7f61b7f47ea6b48fb0e8f12a5b82f8bf2a227594806212d4`.
It confirms execution and checkpoint compatibility, not a speed improvement:
the local eager document-mask validation causes a graph break, and compiled
CUDA throughput remains unmeasured.

The sealed report pins SHA-256 values for the trainer, preflight scripts,
distributed module, and Lumi model implementation. All 83 top-level evidence
hashes pass again after copying the archive to:

`/Users/afnan/Documents/lumi-data/distributed-trainer-preflight-v0.3`

The sealed manifest SHA-256 is
`a1bfe0a902255e79736604272bdd4e80b4cc1a90fc60765ba7a03175d0acc9c9`.

## One failure found and fixed

The first FSDP attempt completed its optimizer steps and durable checkpoints,
then the macOS CPU process segfaulted while gathering a full state inside the
live process group. The model and checkpoint were not corrupt. Final full-state
verification now happens only after destroying the group: DCP loads the shards
directly into one plain CPU Lumi model. The repaired path resumes the original
checkpoint and matches the uninterrupted FSDP model digest exactly.

PyTorch 2.13 also emits deprecation warnings for classic FSDP's ShardedTensor
checkpoint representation. This does not change the verified tensors, but the
CUDA gate must decide whether to keep this proven path or migrate the production
checkpoint implementation to FSDP2/DTensor before a 4B pilot.

## Still unverified

The Mac gate cannot validate:

- NCCL collectives or multi-GPU deadlocks;
- H100/H200 BF16 and FlashAttention 2 behavior;
- real 4B peak HBM and host-RAM requirements;
- compiled versus eager CUDA throughput;
- FSDP synchronization mode memory/throughput trade-offs;
- checkpoint throughput on production-size optimizer shards.

Those are the only reasons for the later short paid smoke. It is an
infrastructure measurement, not another 250M/1.4B learning run. No CUDA launch
may occur without a new explicit owner approval and a hash-bound authorization.
