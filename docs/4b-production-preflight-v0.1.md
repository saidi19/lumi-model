# Lumi-4B Production Training Preflight v0.1

Status: **the preflight contract is rebound to the corrected 160-shard
marker-aware corpus, and its real-manifest plan, trainer adapter, exact-tail,
checkpoint-handoff, and pilot-fixture paths pass locally. No multi-GPU learning
pilot or full model training was authorized or started by this gate.**

## Scope

This gate closes the engineering gap between the final corpus seal and a short
Lumi-4B learning pilot. It does not measure model quality or multi-GPU
throughput. The original adapter was implemented at
`78ce5953e0023c0ee8347fba38c9c0bd25646fcf`; the accepted marker-aware binding
and byte-range pilot fixture were introduced at
`f745a93b1399afdd07cf4021a07715ff78d8d3c2`.

The contract pins:

- model config `configs/lumi-4b-v0.2.json`, SHA-256
  `643e8da652577f802d959740d66a7e6bafc5b065aa45010c2d7e0340ed9509be`,
  with 4,002,376,704 parameters;
- final-corpus manifest SHA-256
  `6de45464af33d44e1261a023ccb7d50f86346a4126bb2fa060530a91213483e9`;
- corpus plan
  `aab292a514b2a7a046a3e4f0daf2fb75bb6f899ff86798bdfac729a145bc9470`;
- selection result
  `b017a6fec5c6ca1685c5dcd883d8afc48d36fd18cac07d573514ad77307253ee`;
- 45,768,283 documents and document markers, 160 immutable uint16 objects, and
  160,000,016,728 packed bytes; and
- an exact trainer boundary of 80,000,000,000 tokens. The final 8,364 packed
  tokens remain intentionally outside that boundary.

## Trainer adapter

`FinalCorpusTokenDataset` treats the 160 physical objects as one logical token
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

## Verification

The historical implementation was verified on one Lambda A10 host with Python 3.12.13,
PyTorch 2.7.0+cu126, and Transformers 5.14.1. The host exposed one 23,028 MiB
A10 and CUDA initialized successfully. This inexpensive host was used for
engineering verification only; it is not representative of 8xH100 throughput
or memory behavior. That historical run used the now-quarantined markerless
manifest and is retained only as implementation provenance.

The corrected binding was then independently reverified on the Mac against the
sealed production manifest and an S3 byte-range read of the accepted first
shard:

- real-manifest production plan: PASS;
- plan fingerprint:
  `8fe0cdaff97490b79851ef3bf7b13c386aab1812e39dd82cfb42c24fa3bcd328`;
- exact 80B boundary: 19,531,250 sequences and 8,364 trailing tokens;
- real 83,886,080-token pilot fixture: PASS, with 9,885 document markers;
- focused production/pilot tests: 17/17 PASS;
- the two-process exact-tail test now passes locally using an explicit IPv4
  rendezvous instead of macOS reverse-DNS-dependent `--standalone`;
- every non-Spark repository test passes; three unrelated local Spark cases
  were excluded because the installed Java 25 runtime is incompatible with
  this PySpark/Hadoop build, while no JDK 17 is installed; and
- Ruff: `All checks passed!`;
- authorization flags remain `paid_multi_gpu_pilot=false` and
  `full_4b_training=false`.

The current plan, exact fixture, reports, configs, and checksum manifest are
kept outside Git under
`lumi-artifacts/4b-marker-aware-preflight-v0.1/exact-f745a93/`.

## Next gate

The next paid action is a separately owner-authorized, bounded 8xH100 learning
pilot using a small staged slice of the sealed final corpus. It must measure
peak HBM, host RAM, tokens/second, checkpoint duration, resume equivalence,
loss movement, and `torch.compile` value before pricing or authorizing the full
4B/80B run. Passing this preflight alone is not permission to launch that
pilot.
