# Lumi-4B Multi-GPU Learning Pilot v0.1

Status: **the provider-independent 4x/8x H100 harness is implemented and
locally verified, but its historical corpus binding points to the quarantined
markerless manifest. The corrected 160-shard marker-aware corpus is sealed;
the harness and fixture must be rebound and reverified before a paid pilot. No
successful paid pilot, full 4B training, or 80B-token training run has been
completed by this gate.**

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

## Immutable workload

The planning-only contract is
`configs/4b-learning-pilot-v0.1.json`. It pins:

- model config `configs/lumi-4b-v0.2.json`, SHA-256
  `643e8da652577f802d959740d66a7e6bafc5b065aa45010c2d7e0340ed9509be`;
- 4,002,376,704 model parameters;
- final-corpus manifest SHA-256
  `c81837652075237de281989513a8301bdd2f78a0c764fdac028ccbe6802b5a24`;
- final plan
  `a647579f612b45e23142a1bc8113e6754af67c5432ca3c9513e02d6ee2e302d0`;
- selection result
  `8d9c2eff0f34c63d4e9a32536c56e6503151b8af440f24c4497fec4ca953fa2e`;
- the exact content-addressed first final shard and an 83,886,080-token,
  20,480-sequence prefix fixture;
- held-out manifest SHA-256
  `c5022ce7ba9ed41dd9cdf1777639278c5ed111ef679e09d9fbc12e36fd6260dd`,
  containing 2,441 sequences and 4,999,168 tokens; and
- fresh training through step 10, exact resume through step 80, checkpoints
  every 10 steps, and a hard stop at 83,886,080 cumulative tokens.

The fixture is a real immutable slice of the final corpus, not generated text.
It comes from `science_and_education` shard sequence zero. That makes this a
bounded learning/throughput/resume gate rather than a full-mixture quality
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

## Local verification

The new contract, authorization, runtime, export, and topology tests pass:
`18 passed`. Ruff passes for the complete repository. The near-full local suite
passes with `354 passed, 3 skipped, 1 deselected`. The single deliberate
deselection is the pre-existing two-process Gloo test: when run separately on
this Mac it times out while the local resolver repeatedly fails reverse
resolution of the localhost IPv6 address. The exact full suite is intentionally
rerun on the Linux H100 host before planning or authorization, and no paid
training can begin unless that remote preflight passes.

## Remaining boundary

The harness can now be committed and reviewed without GPU capacity. When a
matching 4x or 8x 80GB H100 instance is available, the remaining sequence is:

1. clone the exact source commit and stage the pinned corpus slice, held-out
   artifact, and tokenizer;
2. install the pinned CUDA runtime and run the complete Linux preflight;
3. write and review the immutable hardware-specific plan;
4. obtain a fresh exact owner authorization;
5. launch the bounded `tmux` runner;
6. download and independently verify the sealed evidence; and
7. terminate the instance promptly.

Full 4B/80B pretraining remains a separate decision after this measured result.
