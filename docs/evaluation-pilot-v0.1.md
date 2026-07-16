# Lumi 150M evaluation and local-inference gate v0.1

Status: **held-out data and PyTorch/MPS evaluation complete; engineering gate
passes, model-usefulness gate does not pass**

## Purpose

This gate evaluates the completed 150,625,536-parameter plumbing checkpoint on
documents that were not used to update its model weights. It verifies
document-level exclusion, resumable evaluation, loss/perplexity, deterministic
generation, and real Apple-MPS memory and throughput before conversion work.

The checkpoint saw only 50,003,968 training tokens, about 0.33 token per
parameter. It was built to validate end-to-end training plumbing, not to produce
a useful assistant.

## Held-out contract

The builder replays the exact per-source prefix selection recorded by the 50M
training manifest. Every whole training document is excluded, including the
unused suffix of a partially consumed final document. The remaining pool is
sampled with a SHA-256 rank over seed `20260716`, stable document ID, and content
hash. A 1/16 candidate fraction is retained, then each of the 24 source strata
is packed in rank order to a whole number of 2,048-token sequences.

Two hash-pinned ledgers independently prove exclusion by:

- stable document ID;
- exact content SHA-256;
- MinHash near-duplicate fingerprint SHA-256.

SQLite state commits every 1,000 scanned documents. A simulated interruption
and fresh-process resume produce byte-identical shards and ledgers in regression
tests.

This is a **model-weight-held-out** set. The frozen 65K tokenizer was previously
trained from the full 6GB sample, so these documents influenced tokenizer
vocabulary construction even though their tokens never updated the 150M model
weights. The 3,988 selected documents must remain excluded from future model
training, and a permanent pre-training/external benchmark split is still
required before a 1B or 4B quality claim.

## Verified held-out artifact

| Item | Result |
|---|---:|
| Source records scanned | 1,128,089 |
| Training documents excluded | 35,889 |
| Hash-ranked candidates | 68,305 |
| Validation documents selected | 3,988 |
| Requested tokens | 5,000,000 |
| Effective sequence-aligned tokens | 4,999,168 |
| Sequences | 2,441 × 2,048 |
| Shards | 10 |
| ID/content/near-duplicate overlaps | 0 / 0 / 0 |
| Build fingerprint | `77455ce7eb2606d8f258ef13f715bbdfeb8be4f5d7910cc26348552f52c96298` |
| Manifest SHA-256 | `c5022ce7ba9ed41dd9cdf1777639278c5ed111ef679e09d9fbc12e36fd6260dd` |

The selection-ledger SHA-256 is
`5dc1fe84fd347cbaf7cd8f2a0a51125ab1272ec594d649762a50b66661ebdb97`;
the training-exclusion-ledger SHA-256 is
`709027bb95f3e0f956b22a79d4b29f21490ef0b360ba10f2faca7d5db3d06c62`.
The canonical `--resume` build revalidated every shard and both ledgers.

## Real M5/MPS execution

The run used a 16GB Apple M5 MacBook Pro (`Mac17,2`), macOS 26.5.2, arm64,
PyTorch 2.13.0, batch size 1, and FP32 checkpoint weights. The first real
2,048-token batch was intentionally interrupted after writing cursor 1; a new
detached process resumed from that durable state and completed all 2,441
sequences.

| Metric | Result |
|---|---:|
| Mean held-out loss | 4.680812 |
| Perplexity | 107.858 |
| Prediction tokens | 4,996,727 |
| Forward time | 1,066.630s |
| Input throughput | 4,686.9 tokens/s |
| MPS allocated, peak observed | 1,139,393,536 bytes |
| MPS driver memory, peak observed | 2,822,291,456 bytes |
| Process peak RSS | 646,365,184 bytes |
| Evaluation fingerprint | `94f9887a8d0957a561515b5453cca35441b410319d9693243d30a3dcf27a6690` |
| Report SHA-256 | `0c171bdf4c1caba468ece1f0c8133b6056ba6fb62194443bc0aeac7597310a81` |

The strict cached evaluation check recomputed artifact identities and verified
overall loss, perplexity, token totals, throughput, all per-source totals, and
the fixed-prompt records.

## Category results

Losses are token-weighted within each category. Small slices have high variance
and must not be ranked as standalone benchmarks.

| Category | Sequences | Loss | Perplexity |
|---|---:|---:|---:|
| Science and education | 726 | 4.5132 | 91.22 |
| Code | 483 | 4.3821 | 80.01 |
| Reference | 363 | 5.1484 | 172.16 |
| Questions and discussion | 363 | 4.6399 | 103.53 |
| Open web | 242 | 4.8683 | 130.10 |
| Books | 120 | 5.1725 | 176.35 |
| Government, legal, and patents | 120 | 4.9271 | 137.98 |
| Bengali reference slice | 24 | 3.7286 | 41.62 |

The low Bengali token loss does not establish Bengali generation quality: its
slice is small and the fixed Bengali continuation degenerates into repeated
subword patterns.

## Fixed-generation result

Greedy generation with 32 new tokens confirms that the checkpoint learned
surface patterns but is not factually or conversationally usable:

| Prompt | Observed behavior |
|---|---|
| `The capital of France is` | repeats “the development of” instead of answering |
| `Photosynthesis is the process by which` | produces generic, incorrect data prose |
| `In Python, a list comprehension can` | begins plausibly, then drifts into generic data text and headings |
| `বাংলাদেশের রাজধানী` | degenerates into repeated Bengali subword fragments |

This is a base checkpoint with no instruction tuning. A ChatGPT-style UI can be
used later as an inference harness, but it must not be presented as a competent
assistant at this stage.

## Decision

- PASS: deterministic model-weight-held-out construction and zero-overlap proof.
- PASS: interruption/resume, PyTorch/MPS loss, speed, and memory plumbing.
- FAIL: factual, coherent fixed-prompt generation; the 50M-token checkpoint is
  not useful as a product model.
- NEXT: MLX conversion, PyTorch/MLX logits parity, and a 4-bit local artifact.
- NOT APPROVED: 1B cloud training until the permanent evaluation exclusion and
  pilot data/training budget are pinned.
- NOT APPROVED: the 4B/80B run until the 1B quality and deployment gates pass.

No AWS or Lambda GPU was used for this evaluation.

## Reproduce the cached seals

```bash
python scripts/build_heldout_evaluation.py \
  --plan configs/evaluation-pilot-v0.1.json \
  --input "$LUMI_DATA/tokenizer-sample-v0.3/sample.jsonl" \
  --tokenizer "$LUMI_DATA/tokenizer-candidates-v0.2/lumi-tokenizer-65k" \
  --training-manifest "$LUMI_DATA/real-pilot-50m-v0.1/packed/manifest.json" \
  --output "$LUMI_DATA/evaluation-pilot-v0.1/heldout" \
  --resume

python scripts/evaluate_checkpoint.py \
  --plan configs/evaluation-pilot-v0.1.json \
  --checkpoint "$LUMI_DATA/lambda-pilot-v0.1/full/checkpoint-00000763" \
  --manifest "$LUMI_DATA/evaluation-pilot-v0.1/heldout/manifest.json" \
  --tokenizer "$LUMI_DATA/tokenizer-candidates-v0.2/lumi-tokenizer-65k" \
  --output "$LUMI_DATA/evaluation-pilot-v0.1/checkpoint-150m-mps-v0.1" \
  --device mps \
  --resume
```

Generated data, ledgers, state, reports, and checkpoints remain outside Git.
