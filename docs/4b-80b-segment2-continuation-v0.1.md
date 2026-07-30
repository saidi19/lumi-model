# Lumi-4B/80B Segment 2 continuation

This continuation is planning-only until the owner repeats the exact approval
sentence generated from the sealed plan.

It binds Segment 2 to:

- the independently verified, read-only Segment-1 checkpoint at step 31,500;
- 33,030,144,000 already-consumed tokens and sequence cursor 8,064,000;
- the exact `batch2_no_gc` checkpoint-gate result;
- the unchanged 160-object, 160,000,016,728-byte production corpus;
- one Lambda Cloud `gpu_8x_h100_sxm5` instance;
- steps 31,500 through 63,000 and tokens 33,030,144,000 through
  66,060,288,000.

The selected profile keeps the same global batch of 256 sequences while using
two sequences per rank, 16 gradient accumulations, and no activation
checkpointing. The exact-checkpoint gate measured a median 100,779 tokens per
second and passed held-out-loss, resume, checkpoint-integrity, and HBM-safety
checks.

The current source does not authorize paid work. Corpus staging and production
training remain blocked until an immutable plan is built from the committed
source and the owner supplies its exact, hash-bound Segment-2 approval.

Segment 3 remains outside this continuation and requires a separate
checkpoint-bound plan and approval after Segment 2 completes.
