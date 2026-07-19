# Lumi distributed CUDA smoke v0.1

Status: **the owner-authorized two-H100 run passed, its evidence is
independently verified on the Mac, and the Lambda instance is terminated**

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

## Verified Lambda 2x H100 result

On 2026-07-19, the bounded smoke ran on Lambda Cloud's `2x H100 (80 GB
SXM5)` instance from exact source commit
`208c7ed9dd00313727f08d3b54da1b1dd4999b80`. The server-generated plan
retained `full_4b_training=false`, fixed world size 2, and authorized only
steps 1 and 2. Its run fingerprint was
`ef83bc851bffa69cda3770f6f958c7f11c559f93a0ed50030034efea3f9d869b`.

Remote preflight passed all 159 tests, Ruff, fixture hashes, and the real CUDA
FlashAttention varlen parity/isolation gate. The latter reached 0.99998188
cosine similarity to the dense reference, 0.0078125 maximum absolute error,
finite backward gradients, and exactly zero later-document leakage.

| Field | Verified result |
|---|---|
| Runtime | Python 3.12.13, PyTorch 2.7.0+cu126, FlashAttention 2.8.3 |
| Model | 4,002,376,704 parameters, BF16, FSDP `FULL_SHARD`, NCCL |
| Attention | position-reset varlen document isolation |
| Fresh boundary | step 1, 8,192 tokens, loss 11.655620, 7,824.7 tokens/s |
| Resume boundary | fresh process to step 2, 16,384 cumulative tokens, loss 16.791336, 10,861.3 tokens/s |
| Checkpoint time | 95.654 seconds at step 1; 96.369 seconds at step 2 |
| Peak observed HBM | 47,531 MiB on GPU 0; 47,959 MiB on GPU 1 |
| Final checkpoint | `checkpoint-00000002`, all six entries rehashed |
| Checkpoint-manifest SHA-256 | `68c2857239eefcb75aac7ffff96b55ca32719c963666c7fb290475e9be1fcbaa` |
| Canonical final-model SHA-256 | `707a54af49721e21fa35c6de81d33bdc7bc0e14f359ce443e662e0e462d169f9` |
| Run phase | 555 seconds; USD 1.2919 compute estimate at USD 8.38/hour |

The loss values cover only two different batches and are not a quality or
convergence result. The throughput values measure optimizer execution, while
the checkpoint/resume path dominates this deliberately short lifecycle. The
USD 1.2919 figure covers the sealed 555-second run phase only; provider usage
or invoice data remains authoritative for the complete setup-to-termination
lifecycle.

The small sealed archive is stored outside Git at
`lumi-data/distributed-cuda-smoke-v0.1/lambda-2xh100-run-v0.1`. All 24 entries
listed by its downloaded manifest pass local SHA-256 verification. The archive
manifest itself has SHA-256
`2130512c3408c3d4cc2b1fc1dbe01890f34f58a438ca5fd3bfdc87cd2e5629dc`.
The Lambda dashboard subsequently reported no running instances.

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

Those local checks did not authorize a launch. The later owner artifact
authorized only the verified two-step smoke above; no 4B learning pilot, full
4B/80B run, or 80B-token corpus build was started.
