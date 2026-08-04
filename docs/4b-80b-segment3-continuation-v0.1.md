# Lumi-4B/80B Segment 3 continuation

This continuation remains planning-only until the owner repeats the exact,
hash-bound approval sentence generated from the sealed plan.

It binds the final production segment to:

- the sealed Segment-2 checkpoint at step 63,000 and 66,060,288,000 tokens;
- the exact checkpoint-gate winner `fsdp_b2_fa3_compile`;
- the retained Lambda `gpu_8x_h100_sxm5` instance at `192.222.52.81`;
- the already-verified local corpus, tokenizer, held-out data, and FA3 runtime;
- steps 63,000 through 76,294 and the exact 80,000,000,000-token cap.

The final step is intentionally partial: 242 sequences / 991,232 tokens. The
training loop already allocates that corpus tail without duplicating sequences.

The exact-checkpoint gate measured 140,272.96 median tokens/s, passed resume,
loss, HBM, and FA3 parity checks, and left the Segment-2 checkpoint unchanged.
The measured training-only estimate is about 27.6 hours; the conservative 0.8x
estimate is about 34.5 hours. The authorization ceiling is 3,600 minutes and
$1,915.20, including operational margin and final evaluation/sealing.

No S3 access, new instance, termination, or paid Segment-3 launch is implied by
the plan. Production remains fail-closed until the exact owner authorization is
written and verified.
