# Lumi Packed-Attention A/B Gate v0.1

Status: **real A100/FlashAttention gate passed; document-isolated varlen is the
engineering policy for the next fresh scaling run**

## Decision

The completed 250M-token Lumi-1B checkpoint remains the immutable quality A baseline.
It uses the existing separator-only fixed packing and ordinary causal attention,
so later documents in one packed row may attend to earlier documents. That is a
valid pretraining policy, not a correctness bug.

Its historical 100-step throughput cannot by itself be the controlled speed
baseline for the new path because that run used the eager attention backend. A
short CUDA comparison therefore trains two fresh 100-step controls with the
same FlashAttention backend: current separator-only causal attention and the
new document-isolated varlen policy. This keeps the kernel, initialization,
data order, optimizer, and schedule fixed while changing packing semantics.

This branch adds an opt-in B policy for future from-scratch experiments. It does
not change the default model or trainer behavior, does not rewrite the sealed
1.4B-token artifact, and cannot silently resume an existing current-policy
checkpoint as an isolated-policy run.

The controlled A100 result removes the expected speed objection: isolation was
6.17% faster at the measured steady-state median, used the same peak reserved
HBM, and reached a 0.113291 lower paired step-100 training loss. Therefore the
next fresh scaling run should use document-isolated varlen attention. This is
an engineering-policy lock, not a claim that 100 steps prove better final
reasoning or generalization. The existing separator-only 250M checkpoint must
remain on its original policy; an isolated 1.4B-token run must start at step
zero and requires a separate paid-run approval.

## B-policy semantics

For every fixed-length batch row:

1. token zero starts a local segment, even when the row begins midway through a
   source document;
2. every `<|doc|>` token starts another segment;
3. RoPE `position_ids` reset to zero at each segment start;
4. `cu_seq_lens_q`, `cu_seq_lens_k`, and the maximum segment length describe
   the same boundaries after flattening the batch in row-major order;
5. a token may attend only to earlier-or-equal tokens in its own segment.

The CPU/MPS/eager correctness path materializes a boolean document-causal mask.
The CUDA path is fail-closed: document isolation on CUDA requires
`flash_attention_2`, passes the cumulative boundaries to its varlen kernel, and
does not construct the dense O(n-squared) mask.

The language-model labels are otherwise unchanged. The final content token of
one document may still learn to predict the following `<|doc|>` separator; no
token in the new document can read hidden states from the previous document.

## Resume and checkpoint boundary

`document_isolation` and the requested attention implementation are included in
the deterministic run fingerprint whenever enabled. Therefore:

- the archived separator-only 250M checkpoint remains resumable by the old
  default invocation;
- adding `--document-isolation` to that invocation fails the fingerprint check;
- a B-policy experiment must start from the same random initialization, data
  order, and schedule rather than changing attention semantics mid-training;
- B-policy interruption/resume must reproduce its own uninterrupted result.

## Verified local gates

The following checks pass on the Mac without launching a GPU server:

- exact reset positions and int32 flattened cumulative boundaries for batches
  containing row continuations and multiple document markers;
- an explicit dense mask blocks both future tokens and every other document;
- changing all tokens in an earlier document leaves later-document logits
  unchanged under isolation, while the current policy's logits do change;
- a fake varlen kernel that independently executes every cumulative segment
  matches dense-reference logits;
- gradient-checkpointed isolated training resumes from a durable checkpoint and
  produces weights and loss history exactly equal to an uninterrupted run;
- default optimizer identity and default training resume remain byte-compatible;
- self-contained Hugging Face save/load preserves document-isolated logits;
- the complete repository suite passes: 125 tests, plus Ruff lint.

## Verified A100 gate

The canonical harness ran on one Lambda A100-SXM4-40GB at USD 1.99/hour using
source commit `f05640aebdb8bb32cac8dac6dc6a77d3d270ce7f`, Python 3.12.13,
PyTorch 2.7.0+cu126, Transformers 5.14.1, and FlashAttention 2.8.3. Before
training, all 125 cloud tests, Ruff, exact data/tokenizer hashes, CUDA/BF16
admission, and the real varlen forward/backward gate passed.

The CUDA parity check measured cosine similarity 0.99998194, maximum absolute
logit error 0.0078125, finite gradients, and exactly zero changed logits in a
later document after mutating an earlier document.

| Fresh 100-step policy | Final loss | Steady median tokens/s | Duration | Peak reserved HBM |
| --- | ---: | ---: | ---: | ---: |
| Separator-only FlashAttention | 7.690148 | 9,239.5 | 401s | 25,276,973,056 bytes |
| Position-reset varlen isolation | 7.576857 | 9,809.5 | 380s | 25,276,973,056 bytes |

The paired isolated-minus-current loss delta was -0.047933 at step 1,
-0.136214 at step 10, -0.020061 at step 50, and -0.113291 at step 100. The
isolated/current throughput ratio was 1.061691. The two measured trainer
windows cost an estimated USD 0.431719 combined; setup, transfer, validation,
and idle lifecycle time are not included in that trainer-only estimate.

The server seal passed and the Mac independently verified all 32 downloaded
manifest entries. The two model weights and two optimizer states, about 25GB
combined, were deliberately not downloaded; their server-sealed hashes remain
in the manifest. The local evidence is stored outside Git at
`lumi-data/packed-attention-ab-v0.1/a100-100-step-v0.1`, and the final manifest
SHA-256 is
`3164c93a2ab543796897f244899c7e3ba3098941958978bec71ac4ddc70840b7`.
The Lambda console was then verified to show no running instances.

## Command contract

The local dense reference path can be exercised with `--document-isolation` on
CPU or MPS. The paid CUDA path must additionally select FlashAttention:

```bash
python scripts/train_pilot.py \
  --manifest "$LUMI_DATA/scaling-data-1b-v0.1/final-1.4b/manifest.json" \
  --config configs/lumi-1b.json \
  --output "$LUMI_DATA/packed-attention-ab-v0.1/isolated-100-step" \
  --steps 100 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 0.0002 \
  --weight-decay 0.1 \
  --max-grad-norm 1.0 \
  --checkpoint-steps 100 \
  --gradient-checkpointing \
  --keep-checkpoints 2 \
  --seed 7 \
  --lr-schedule cosine \
  --warmup-steps 500 \
  --schedule-steps 42725 \
  --min-learning-rate-ratio 0.1 \
  --attention-implementation flash_attention_2 \
  --document-isolation \
  --device cuda
```

The canonical cloud harness is
`scripts/run_packed_attention_cuda_smoke.sh`. It first requires the actual
FlashAttention varlen kernel to match the dense reference, preserve document
isolation, and produce finite backward gradients. It then creates two fresh
hard-capped runs:

1. current packing with `flash_attention_2`, without document isolation;
2. isolated packing with `flash_attention_2` and `--document-isolation`.

Each run stops at step 100 (3,276,800 tokens). The seal compares paired losses,
steady-state throughput, HBM usage, telemetry, runtime, and fingerprints. This
is an implementation and speed smoke, not enough to claim that one packing
policy produces a better final model. It selects the isolated implementation
for the next fresh scaling candidate without authorizing a 1.4B or 4B paid
run.
