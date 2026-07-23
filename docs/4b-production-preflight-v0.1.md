# Lumi-4B Production Training Preflight v0.1

Status: **the sealed final-80B corpus is trainer-readable, the exact-tail and
checkpoint-handoff paths pass, and the planning-only 8-GPU configuration is
sealed. No multi-GPU learning pilot or full model training was authorized or
started.**

## Scope

This gate closes the engineering gap between the final corpus seal and a short
Lumi-4B learning pilot. It does not measure model quality or multi-GPU
throughput. The implementation is bound to source commit
`78ce5953e0023c0ee8347fba38c9c0bd25646fcf`.

The contract pins:

- model config `configs/lumi-4b-v0.2.json`, SHA-256
  `643e8da652577f802d959740d66a7e6bafc5b065aa45010c2d7e0340ed9509be`,
  with 4,002,376,704 parameters;
- final-corpus manifest SHA-256
  `c81837652075237de281989513a8301bdd2f78a0c764fdac028ccbe6802b5a24`;
- corpus plan
  `a647579f612b45e23142a1bc8113e6754af67c5432ca3c9513e02d6ee2e302d0`;
- selection result
  `8d9c2eff0f34c63d4e9a32536c56e6503151b8af440f24c4497fec4ca953fa2e`;
- 45,817,637 documents, 163 immutable uint16 objects, and
  160,000,029,468 packed bytes; and
- an exact trainer boundary of 80,000,000,000 tokens. The final 14,734 packed
  tokens remain intentionally outside that boundary.

## Trainer adapter

`FinalCorpusTokenDataset` treats the 163 physical objects as one logical token
stream. A 4,096-token training sequence may cross an object boundary without
dropping or repeating tokens. Every object is size- and hash-verified during
resumable staging, while distributed startup performs one full local hash pass
per node and collectively agrees on the manifest identity.

The exact boundary contains 19,531,250 sequences. Because that count is not
divisible by eight, the last global microbatch uses two real ranks and six
zero-weight dummy ranks. This keeps DDP/FSDP collectives well formed while
averaging gradients only over real sequences. The final optimizer step can
contain both full and partial microbatches, and metrics record actual rather
than nominal tokens. Training cannot wrap into a second corpus epoch.

The planning candidates are:

| Gradient accumulation | Nominal tokens/step | Maximum steps | Real sequences in final step |
|---:|---:|---:|---:|
| 32 | 1,048,576 | 76,294 | 242 |
| 64 | 2,097,152 | 38,147 | 498 |

Both keep per-rank batch size 1, BF16, full-shard FSDP, FlashAttention 2,
document-isolated attention, gradient checkpointing, and last-microbatch
gradient synchronization. `torch.compile` remains a measured pilot A/B rather
than an assumption.

## Portable checkpoint handoff

The handoff sealer verifies the distributed checkpoint first, inventories and
hashes every file, and binds the package to the exact source commit, run
fingerprint, world size, step, sequence cursor, and token cursor. A destination
account or instance must verify the complete inventory before resuming. The
account-segment planner is planning-only: it converts measured pilot throughput,
provider runtime, hourly price, available credit, and a reserve into a
checkpoint-aligned stop boundary. It cannot launch paid work.

## Independent Lambda verification

The exact commit was verified on one Lambda A10 host with Python 3.12.13,
PyTorch 2.7.0+cu126, and Transformers 5.14.1. The host exposed one 23,028 MiB
A10 and CUDA initialized successfully. This inexpensive host was used for
engineering verification only; it is not representative of 8xH100 throughput
or memory behavior.

Results:

- `346 passed, 4 skipped in 30.51s`;
- Ruff: `All checks passed!`;
- real sealed-manifest plan: PASS;
- plan fingerprint:
  `42d742ecec2f63079522098118a2ba3be079ce59c1c0e651d6a3c84b92caacbe`;
- locally downloaded evidence hashes: PASS; and
- authorization flags remain `paid_multi_gpu_pilot=false` and
  `full_4b_training=false`.

The verified logs and plan are kept outside Git under
`lumi-artifacts/4b-production-preflight-v0.1/exact-78ce595/`.

## Next gate

The next paid action is a separately owner-authorized, bounded 8xH100 learning
pilot using a small staged slice of the sealed final corpus. It must measure
peak HBM, host RAM, tokens/second, checkpoint duration, resume equivalence,
loss movement, and `torch.compile` value before pricing or authorizing the full
4B/80B run. Passing this preflight alone is not permission to launch that
pilot.
