# Lumi distributed CUDA smoke v0.1

Status: **local planning and regression gates pass; paid two-GPU run not yet
launched**

This is the final infrastructure gate before a short 4B multi-GPU pilot. It is
not a learning run and it cannot authorize the 80B-token pretraining run. The
smoke uses the real 4,002,376,704-parameter Lumi-4B v0.2 configuration for one
fresh optimizer step and one fresh-process resumed step on two H100/H200 GPUs.

## Exact scope

The repository contract in `configs/distributed-cuda-smoke-v0.1.json` pins:

- two CUDA ranks with NCCL;
- FSDP `FULL_SHARD` and BF16 mixed precision;
- document-isolated FlashAttention 2 varlen attention;
- residual-scaled initialization and the selected legacy AdamW profile;
- one sequence per rank, 4,096 tokens per sequence, and 8,192 global
  tokens/step;
- fresh step 1, immutable distributed checkpoint, then fresh-process resume
  through step 2;
- exactly 16,384 processed tokens and one retained final checkpoint;
- `torch_compile=false` for the correctness-first run;
- `full_4b_training=false`.

The CUDA trainer still has a generic 1,000-step smoke ceiling, but this run's
owner artifact is fingerprint-bound to step 2. Changing the model, data,
topology, source commit, PyTorch version, attention policy, optimizer, batch,
or compile setting changes the fingerprint and invalidates the authorization.

## Real-token fixture

`scripts/prepare_distributed_cuda_smoke.py` verifies the sealed 1.4B-token
manifest and the complete first 64MiB shard before extracting its first 64
fixed-length sequences. The resulting fixture contains 262,144 real tokenizer
tokens, including 45 document markers, in a 512KiB uint16 shard.

The locally verified fixture is:

| Field | Value |
|---|---|
| Parent manifest SHA-256 | `9228fa610d4a985d05e15057e8ef12619a14cbedbb5a6a5ab513d3aeca2832c8` |
| Parent build fingerprint | `0bce4d9e4a51de1b831b0226e0052ea33130733eae2bb61992c6eacaf203e0c8` |
| Fixture manifest SHA-256 | `bf11f32cbb42b8b6fa26c957713c969a945f7f06eda374d00f9d9792050c0a68` |
| Fixture shard SHA-256 | `9ba01f4d19028a8ed8447e4326825f79d456001fab9ea0b5bb9cfa97431b2bbf` |
| Fixture build fingerprint | `585aab7e6ab8f0276adadf8278877173b44e31ede2724cea054b11f0deb1f134` |

This tiny fixture is representative input for plumbing and memory behavior; two
steps are not enough to assess model quality.

## Authorization flow

The source repository never contains a paid approval.

1. Merge the smoke harness and check out the exact 40-character commit on the
   GPU server.
2. Install the pinned CUDA runtime with
   `scripts/setup_distributed_cuda_smoke.sh`.
3. Run `scripts/preflight_distributed_cuda_smoke.sh`. It re-runs tests, Ruff,
   FlashAttention CUDA parity, all fixture hashes, and writes `plan.json` using
   the server's exact PyTorch version and GPUs. No training occurs.
4. Download and verify the plan. Only then create the out-of-repository owner
   artifact with `scripts/authorize_distributed_cuda_smoke.py`.
5. Upload that artifact and start `scripts/run_distributed_cuda_smoke.sh` under
   `tmux`. The trainer independently revalidates the exact fingerprint, commit,
   world size, strategy, and step cap.
6. `scripts/seal_distributed_cuda_smoke.py` rehashes the complete final DCP
   checkpoint on the server and seals a small downloadable evidence bundle.
   The instance must then be terminated.

## Verified local regression

- 159 tests pass and Ruff passes.
- The seven-launch DDP/FSDP interruption preflight passes after the identity
  refactor.
- Historical DDP final model SHA-256 remains
  `483f07c3edb4604acbf0a1ce84c896ece428eed31880c51541f7e0fd24c73c71`.
- Historical FSDP final model SHA-256 remains
  `76e5ec8bab0c9f5794cd5dd5b0a092b1ef774ad6d1769bcf65a794ebad3c9bc1`.
- The local structural 4B plan stops at step 2 and 16,384 tokens. Its local
  fingerprint is not a launch authorization because the Mac has no CUDA and
  uses a different PyTorch version.

No paid GPU, 4B optimizer step, or 80B-token corpus work has been started by
these local checks.

