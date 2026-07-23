# Lumi-4B Multi-GPU Learning Pilot v0.1

Status: **the authorized 8xH100 pilot passed at the exact step-80 /
83,886,080-token hard stop. Its sealed evidence and complete distributed
checkpoint are independently verified, the Lambda instance is terminated, and
no full 4B/80B training was authorized or started.**

## Purpose

This is the last bounded hardware-and-learning gate before a separately
authorized Lumi-4B/80B production run. It answers questions that the earlier
two-GPU CUDA smoke and single-A10 engineering preflight could not:

- can the exact 4,002,376,704-parameter model train under full-shard FSDP on
  the intended H100 topology;
- does document-isolated FlashAttention remain correct under the real model
  shape;
- do checkpoint, interruption, exact resume, held-out evaluation, and portable
  export all work together;
- what are measured HBM, host runtime, tokens/second, checkpoint time, and
  compute cost; and
- does both training loss and the frozen held-out loss improve over the bounded
  run.

Passing this pilot is evidence for engineering readiness and learning. It is
not permission to start full 4B/80B pretraining and it is not a quality claim
against another model.

## Verified 8xH100 outcome

The owner-authorized Lambda run used exact source commit
`7ae753b12f0a84d9598f12e7f9edf0f5d00f138c` and run fingerprint
`c79c8a5e5a681aa61a26439d5a2aa39798cdc3884b626550cd8ff96ac7ac6eb9`.
It trained the 4,002,376,704-parameter model fresh through step 10, resumed the
matching distributed checkpoint, and stopped exactly at step 80 with
83,886,080 tokens consumed and sequence cursor 20,480. There is no step 81.
The authorization records `full_4b_training=false`.

| Verified measurement | Result |
| --- | ---: |
| World size | 8x NVIDIA H100 80GB HBM3 |
| Strategy / precision | FSDP full shard / BF16 |
| Document isolation | FlashAttention 2, `position_reset_varlen_v0.1` |
| Median training throughput | 71,935.432 tokens/s |
| First ten training-loss mean | 12.856966 |
| Last ten training-loss mean | 7.141212 |
| Final step loss / best loss | 7.115641 / 7.078655 |
| Maximum recorded training allocation | 18,948,996,608 bytes |

The frozen held-out set contains 2,441 sequences and 4,999,168 input tokens:

| Frozen held-out checkpoint | Loss | Perplexity | Throughput |
| --- | ---: | ---: | ---: |
| Step 10 | 9.605866 | 14,851.647 | 41,912.0 tokens/s |
| Step 80 | 8.046854 | 3,123.950 | 40,749.3 tokens/s |

Held-out loss improved by `1.559012`, and the final ten-step training-loss mean
is lower than the first ten-step mean. Those two independent conditions establish
bounded learning. At only about 0.021 training tokens per parameter, generation
remains repetitive; this checkpoint is not a usable base or chat model and is not
a quality benchmark.

The canonical runner recorded 3,341 seconds and a `$29.6235` compute estimate
at `$31.92`/hour. Lambda's final usage record includes staging, setup,
evaluation, the 46GB checkpoint download, and teardown: `2.85` billed hours and
`$90.90` total instance spend. The provider dashboard then showed
`No running instances`.

The downloaded archive verifies all 35 sealed evidence entries and all 18
distributed-checkpoint entries. Its evidence manifest SHA-256 is
`78e7f8d3a579a47c833cfce04e301d770d05ddc7d384e954b2c2a32106ef73ec`;
the step-80 checkpoint manifest is
`84ade92bf21f3dc9c2de93d5b2b05ff8470ed7793b88d04b47871e82d6f5eb74`;
and the canonical full-model digest is
`4285581ccf006b96c9c3f9214e3e5a84147eb729562d7df0c73bee735329658f`.

## Immutable workload

The planning-only contract is
`configs/4b-learning-pilot-v0.1.json`. It pins:

- model config `configs/lumi-4b-v0.2.json`, SHA-256
  `643e8da652577f802d959740d66a7e6bafc5b065aa45010c2d7e0340ed9509be`;
- 4,002,376,704 model parameters;
- final-corpus manifest SHA-256
  `6de45464af33d44e1261a023ccb7d50f86346a4126bb2fa060530a91213483e9`;
- final plan
  `aab292a514b2a7a046a3e4f0daf2fb75bb6f899ff86798bdfac729a145bc9470`;
- selection result
  `b017a6fec5c6ca1685c5dcd883d8afc48d36fd18cac07d573514ad77307253ee`;
- the exact content-addressed first final shard and an 83,886,080-token,
  20,480-sequence prefix fixture;
- held-out manifest SHA-256
  `c5022ce7ba9ed41dd9cdf1777639278c5ed111ef679e09d9fbc12e36fd6260dd`,
  containing 2,441 sequences and 4,999,168 tokens; and
- fresh training through step 10, exact resume through step 80, checkpoints
  every 10 steps, and a hard stop at 83,886,080 cumulative tokens.

The fixture is a real immutable slice of the final corpus, not generated text.
It comes from `science_and_education` shard sequence zero. The full shard is
1,000,036,984 bytes, but staging requests only bytes `0-167772159`. That exact
167,772,160-byte prefix hashes to
`f27566d2d8ee7f234a03918ca6a3f6c94f03cd44ec58a7e77cce4c39ef58eed0`
and contains 9,885 token-ID-3 document markers. Range staging avoids downloading
the unused remainder while remaining bound to the full shard identity. This is
a bounded learning/throughput/resume gate rather than a full-mixture quality
evaluation. The broad frozen held-out set remains unchanged and prevents the
gate from passing on training loss alone.

## Equivalent 4x and 8x topologies

Both topologies process exactly 256 sequences and 1,048,576 tokens per optimizer
step:

| H100 topology | Per-rank batch | Gradient accumulation | Steps | Tokens |
|---:|---:|---:|---:|---:|
| 4x 80GB H100 | 1 | 64 | 80 | 83,886,080 |
| 8x 80GB H100 | 1 | 32 | 80 | 83,886,080 |

The data, initialization, optimizer, learning-rate schedule, seed, global
batch, checkpoint boundaries, evaluation, and pass conditions are identical.
Therefore either topology can run this gate. The chosen pilot topology should
match the intended full-run topology; changing topology later requires another
measured throughput/resume gate.

The setup fails closed unless every visible pilot GPU:

- is homogeneous;
- reports compute capability 9.0;
- contains `H100` in the model name; and
- reports between 80,000,000,000 and 90,000,000,000 bytes of memory.

That excludes accidental A100, H200, and mixed-hardware execution.
The host must also expose at least 128GB RAM and 300GB free space on the
workspace volume before planning. This leaves room for two distributed
checkpoints, optimizer state, temporary full-model export, evaluation, and
sealed evidence.

## Fixed training settings

- full-shard FSDP;
- BF16;
- FlashAttention 2.8.3 with `position_reset_varlen_v0.1` document isolation;
- gradient checkpointing;
- last-microbatch gradient synchronization;
- residual-scaled initialization from fresh step zero;
- AdamW legacy profile, learning rate `2e-4`, weight decay `0.1`;
- cosine schedule over 80 steps, 10 warmup steps, minimum LR ratio `0.1`;
- seed 7; and
- no `torch.compile` in this gate.

## Fail-closed execution

Repository configuration cannot authorize paid work. A CUDA machine first
creates a source-commit-, data-, model-, runtime-, and topology-bound plan.
A separate owner artifact must then bind the exact plan, provider, instance
type, hourly rate, and runtime cap.

The maximum accepted runner caps are:

- 8x H100: 240 minutes; and
- 4x H100: 480 minutes.

At the recently observed Lambda rates these ceilings correspond to $127.68 for
8x H100 at $31.92/hour and $130.88 for 4x H100 at $16.36/hour. They are safety
ceilings, not projected spend.

The guarded runner records runtime cumulatively across retries, so restarting a
failed process cannot reset the authorization clock. The step-10 evidence is
sealed before later checkpoints may prune the step-10 model. A resumed process
must honor `fresh.complete` and `resume.complete`; it cannot overwrite the
fresh milestone with a later state. Training runs inside `tmux`, so a client
disconnect does not stop the server process.

The final sealer requires:

- exact step, token, and sequence cursors at steps 10 and 80;
- all 80 ordered finite metrics;
- the exact world-size/accumulation geometry;
- a verified final distributed checkpoint and full-model digest;
- complete step-10 and step-80 frozen held-out evaluations;
- lower final held-out loss than step-10 held-out loss;
- lower mean training loss over the last 10 steps than the first 10;
- FlashAttention varlen CUDA parity;
- telemetry from every GPU;
- cumulative runtime inside the owner cap; and
- a hash inventory for every retained evidence file.

## Verification history

The accepted rebind is implemented at
`f745a93b1399afdd07cf4021a07715ff78d8d3c2`. The real production manifest,
S3-range prefix, generated fixture, and trainer memory map all pass independent
hash and shape checks. The fixture contains exactly 20,480 sequences,
83,886,080 tokens, and 9,885 document markers; its manifest SHA-256 is
`d4cf263f86958929717c6be1695678aa6146378466e9c10b5f0be0bad32d523d`.

Focused production/pilot tests pass 17/17. The former macOS Gloo timeout is
resolved by assigning an explicit IPv4 rendezvous port, so exact-tail execution
now passes locally. Every non-Spark repository test passes and repository-wide
Ruff is clean. Three unrelated local Spark tests are excluded because this Mac
has Java 25 only, while the installed PySpark/Hadoop build requires an older
compatible JDK. The complete Linux suite remains mandatory on the H100 host
before planning or authorization.

Before the paid run, the exact H100 source commit passed the complete Linux
preflight and repository-wide Ruff check. After the run, both downloaded
manifests passed independent SHA-256 verification and the authorization,
source, topology, step, token, cursor, evaluation, and model-digest invariants
were rechecked locally.

## Remaining boundary

This gate is complete. It establishes the real 4B model's H100 execution,
document-isolated attention, FSDP checkpoint/resume, frozen evaluation,
throughput, and failure-safe hard stop. It does not establish useful generation
quality because 83.9M tokens are intentionally tiny relative to four billion
parameters.

The next paid stage is the separately planned full 4B/80B pretraining run. It
requires a new owner-approved topology, budget, provider/account handoff plan,
checkpoint cadence, and exact authorization. No such run is implied or
authorized by this result.
