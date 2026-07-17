# Lumi Packed-Attention A/B Gate v0.1

Status: **optional document-isolated path implemented and locally verified;
real CUDA/FlashAttention execution not yet verified**

## Decision

The completed 250M-token Lumi-1B checkpoint remains the immutable A baseline.
It uses the existing separator-only fixed packing and ordinary causal attention,
so later documents in one packed row may attend to earlier documents. That is a
valid pretraining policy, not a correctness bug.

This branch adds an opt-in B policy for future from-scratch experiments. It does
not change the default model or trainer behavior, does not rewrite the sealed
1.4B-token artifact, and cannot silently resume an existing current-policy
checkpoint as an isolated-policy run.

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
- the complete repository suite passes: 124 tests, plus Ruff lint.

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

The first real CUDA gate should hard-stop at cumulative step 100 (3,276,800
tokens), seal loss/throughput/HBM evidence, and compare it with the existing
100-step A calibration. It is an implementation and speed smoke, not enough to
claim that one packing policy produces a better model. No 250M rerun, 1.4B
continuation, or 4B training is authorized by this document.
