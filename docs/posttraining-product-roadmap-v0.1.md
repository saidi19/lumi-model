# Lumi L1 post-training and product roadmap v0.1

This work runs in a separate branch while the already-authorized Lumi-4B base
pretraining continues. It does not modify, restart, export, or otherwise touch
the live Segment 1 process.

## Why these phases are separate from the 80B run

The 80B boundary is the base model's next-token pretraining contract. It teaches
language, knowledge patterns, code, and broad prediction ability. Supervised
fine-tuning (SFT) comes afterward and teaches the base model how to follow an
instruction and behave as an assistant. Safety evaluation, benchmark
evaluation, and deployment then test the resulting assistant artifact. Folding
those records into the 80B base corpus would remove the clean base/SFT
checkpoint boundary and make regressions harder to diagnose.

## Work that is safe to prepare during base pretraining

| Workstream | Current deliverable | Dependency before real use |
| --- | --- | --- |
| SFT data | Versioned record schema, license/provenance rules, deterministic filtering, benchmark decontamination | Curated source records and owner-approved target size |
| Benchmarks | Hash-bound adapter plan and result aggregator | Frozen benchmark versions and final checkpoint |
| Safety | Lumi-owned prompt-injection, privacy, harmfulness, calibration, Bengali, and over-refusal probes | SFT checkpoint and policy thresholds |
| Product UI | Chat history, streamed responses, readiness dashboard, responsive frontend | Authenticated final inference endpoint |
| API | OpenAI-compatible chat/streaming proxy skeleton | Sealed model server and secret-managed credentials |
| Deployment | Fail-closed artifact and readiness contract | 80B checkpoint, SFT checkpoint, benchmark pass, safety pass |

## Remaining execution gates

1. Finish Segment 1 and verify its frozen evaluation and handoff.
2. Run the already-required exact-checkpoint throughput verification, then
   authorize and execute Segment 2.
3. Verify Segment 2 handoff, authorize and execute Segment 3 through exact step
   76,294 and exactly 80,000,000,000 pretraining tokens.
4. Seal and evaluate the final base checkpoint.
5. Curate, filter, deduplicate, and seal the SFT dataset under
   `lumi.sft.v0.1`; decide its token target from the base evaluation rather than
   guessing today.
6. Run bounded SFT, seal the `lumi-4b-instruct` checkpoint, and compare it with
   the base checkpoint.
7. Run frozen quality benchmarks and the required safety suite. A failed
   required suite blocks serving.
8. Bind every artifact hash into the serving contract, run a deployment canary,
   and only then enable an authenticated public API and website.

No extra pretraining stage is introduced by this roadmap. The work here removes
avoidable waiting after the 80B base run; it does not claim that SFT, safety, or
deployment has already passed.
