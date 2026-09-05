# Lumi L1 post-training and product roadmap v0.1

Current execution status (2026-09-05): see
[SFT execution status v0.6](sft-execution-status-v0.6.md). The current-input
DDP/FSDP CPU preflight and local GPU-readiness metadata audit have passed.
The isolated ML runtime and v2 synthetic import adapter are verified:
31 contract tests plus 17 genuine tiny CPU tests pass, including fresh-process
replay. The local SFT launch-contract checker/compiler also passes 54 synthetic
tests and fresh-process replay; it preserves the S3-backed base and cannot
dispatch the legacy worker. The integrated resource-enforcing launcher and
actual final-4B import are still pending; six pre-launch conditions
remain explicitly open at their documented partial completion levels. Historical
“Current” labels below are retained as a record of earlier stages; production
eligibility and GPU execution remain separate gates.

This work runs in a separate branch from the completed Lumi-4B base pretraining
lineage. The exact 80B-token Segment-3 checkpoint is now sealed, independently
verified, and backed up off-instance. Post-training preparation does not modify
that checkpoint.

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

## Execution gates

1. **Complete:** finish Segment 1 and verify its frozen evaluation and handoff.
2. **Complete:** checkpoint-verify the faster profile and execute Segment 2.
3. **Complete:** execute Segment 3 through exact step 76,294 and exactly
   80,000,000,000 pretraining tokens.
4. **Complete:** seal, evaluate, back up, and independently verify the final
   base checkpoint and tokenizer.
5. **Current:** the revision-pinned metadata/source audit, fail-closed seal
   implementation, first 72-record CPU-only source canary, 320-record
   AI-assisted candidate, and representative source-quality audit are complete.
   The audit produced zero promotions and exposed adapter/provenance defects and
   missing safety/coding/factual/writing coverage. The 32-row owner spot-check
   has passed, and the OASST1 root-direct/rank-0 adapter repair passed its exact
   pinned-slice re-audit with 9 candidates and zero promotions. Aya's full-count
   audit is also complete and failed closed on concentration, formatting,
   deduplication, pagination identity, human review, and attribution. Its exact
   local repair preflight projected 1,478 records / 577,453 supervised tokens
   after closing the measured machine-repair defects. That exact private
   candidate, attribution manifest, and deterministic 32-row owner packet are
   now materialized outside Git. Its pre-review JavaScript escaping defect was
   repaired without changing candidate data, and the owner subsequently
   accepted all 32 exact sample rows. Its reviewed candidate metadata is now
   registered in immutable successor registry `v0.4`. The
   bounded OpenR1 verified-trace repair audit is complete: 18/24 traces were
   selected from `generations`, 12 passed normal filtering, 6 fit 4K, and zero
   were promoted. Its separate raw-text-free PII diagnostic classified four
   line-local math-number collisions at the primary stage. A second
   line/message-scoped diagnostic then resolved all six bounded hits as five
   math-number collisions and one cross-message join artifact, with zero
   unresolved and no global-filter change. The exact 93,733-row/10-shard
   OpenR1 source was then acquired and curated. Its visible-final repair retained
   2,656 records / 1,170,745 supervised tokens, and its 32/32 owner gate passed;
   a later exact rights-map audit cleared zero rows and keeps the full candidate
   quarantined. The eight-lane missing task/language mix is
   now planned at 10M-30M accepted train supervised tokens. A separate 320-row
   Lumi-owned safety/tool candidate has passed machine gates and exact Lumi
   tokenization: 240 train rows contain 18,636 supervised tokens, 80 validation
   rows contain 6,388, and the maximum serialized length is 256/4,096. Its
   human owner accepted all 40 spot-check rows with zero rejections and the
   raw-text-free receipt passed. A separate append-only gate registered the
   exact lane as `candidate_eligible` in source-registry successor `v0.2`; the
   active production contract remains on immutable `v0.1`, and production
   dataset promotions remain zero. The exact OASST1 full-source CPU curation is
   also complete: 88,838 rows produced 1,763 strict candidates and 1,745 final
   4K-fit records containing 477,456 supervised tokens. Its independent receipt,
   32/32 owner review, and candidate-registry `v0.3` promotion passed. Acquire,
   repair, and promote only evidence-complete sources, measure the accepted mix,
   and seal the SFT dataset under `lumi.sft.v0.1`.
6. Run bounded SFT, seal the `lumi-4b-instruct` checkpoint, and compare it with
   the base checkpoint.
7. Run frozen quality benchmarks and the required safety suite. A failed
   required suite blocks serving.
8. Bind every artifact hash into the serving contract, run a deployment canary,
   and only then enable an authenticated public API and website.

No extra pretraining stage is introduced by this roadmap. Base pretraining is
finished; SFT, safety, post-training benchmarks, and deployment remain separate
gates and have not yet passed.

## Current status at a glance

| Phase | Verified state | Next blocking evidence |
| --- | --- | --- |
| Base pretraining | Complete at exactly 80B tokens; final checkpoint, tokenizer, evaluation, handoff, and private backup are sealed | None for the base artifact |
| SFT behavior design | Identity, natural English/Bengali/Banglish behavior, frozen language-routing precedence, honesty, tool boundaries, greeting, typo, arithmetic, punctuation, and safety plans are versioned; the general 320-row AI-assisted pilot and its 32-row owner sample passed; a separate 320-row safety/tool lane passed machine gates in four language modes, its human owner accepted 40/40 spot-check rows, and it is registered as a candidate source | Broaden behavior and source coverage to the measured 10M-30M learning mix |
| SFT source/data quality | Schema, append-only source registries, filtering, source canary, AI provenance, raw-text-free audits, and an exact eight-lane 10M-30M task/language proposal exist; the safety/tool lane contributes 18,636 train supervised tokens and is `candidate_eligible` in successor registry `v0.2`; OASST1's exact 1,745 strict 4K-fit records / 477,456 supervised tokens passed independent curation verification and 32/32 owner review and is registered in successor `v0.3`; Aya's exact 1,478-record / 577,453-supervised-token candidate, attribution, and 32/32 owner gate are registered in immutable candidate successor `v0.4`, and its zero-leakage 1,330-train/148-validation split is privately materialized; OpenR1's pinned full source produced a verified 2,656-record / 1,170,745-token visible-final candidate and 32/32 owner review, but its exact seven-label rights audit cleared zero records and keeps all 2,656 quarantined | Obtain documented downstream rights or replace OpenR1 with rights-clear math/reasoning data; build and review the remaining 10M-30M target mix |
| SFT encoding | Chat serialization, assistant/tool-only labels, padding masks, deterministic seal, and raw-text-free integer shard writer are implemented | A real production dataset seal and train/validation shards |
| SFT runtime | Synthetic single-device two-step loss and fresh-resume canary passes exactly | Distributed FSDP masked-loss path, production-seal-bound CPU canary, and authorized GPU smoke/resume |
| SFT optimization | Candidate LR/epoch/schedule/gradient limits are pinned as hypotheses | Equal-token 8xH100 screen; no winner is preselected |
| Post-SFT evaluation | Base baseline, basic-capability gate, 216-case metadata-only language-routing contract, and a 72-lane punctuation contract requiring at least 288 actual model cases are bound | Instruction, reasoning, math, code, truthfulness, actual language/punctuation quality, Bengali/multilingual adapters and frozen prompts/results |
| Safety | Required suites and zero-tolerance failure classes are defined | Executable frozen suites, red-team results, over-refusal and multilingual safety evidence |
| Preference tuning | Purpose and candidate-pair range are planned | Optional only after SFT; reviewed preference pairs, measurable benefit, and separate authorization |
| Serving/product | API paths, streaming protocol, auth requirement, artifact contract, and a browser-verified local-only safe Markdown/code/KaTeX renderer exist | Sealed SFT model, benchmark/safety reports, inference/quantization measurements, deployment canary |

The immediate project bottleneck is reviewed and complete SFT data, not GPU
availability. Because the project currently has one human owner, the active
pilot now uses explicitly AI-generated drafts plus deterministic validation and
a dataset-level owner spot-check. It never claims human-from-scratch authorship
or a second human reviewer.

The newer immutable source-registry `v0.15` and verified 10M mix-closure update
supersede the earlier 1,050,680-token planning snapshot below. The current
accepted planning mix is 7,256,670 train supervised tokens, leaving 2,743,330
to the 10M floor. A proposed 3,200,000-token first-party multilingual safety/tool
lane would raise the mix to 10,456,670 and satisfy the frozen task, language,
source, and source-family floors, but its provider-bound scale canary, full
generation/curation, owner review, registry promotion, cross-source re-audit,
and dataset seal remain unexecuted gates.

The self-hosted gpt-oss teacher branch now has all 160 individually accepted
rows, a passing full machine gate, and verified owner acceptance of its exact
deterministic 32-row packet. The authorized v0.15 recovery regenerated only
owner-sampled `multilingual_bengali-english-00`; its first and only attempt
matched both exact fields. Assembly preserved the other 159 rows and passed all
frozen checks at 168 calls, eight historical retries, 0.95 overall first-attempt
acceptance, and exact 0.8 for the previously failing cell. Owner review accepted
all 32 sampled rows, exactly two per frozen cell, with zero rejections. This
closes the teacher canary's machine and owner-review gates. The rows remain
candidate-only and non-production. A verified non-executing v0.16 promotion
preflight projects them as a new source without replacing the existing 80-row
teacher source: 128 train rows / 9,473 train supervised tokens and 32 validation
rows / 1,906 validation supervised tokens. Its scenario-level split keeps each
variant wholly on one side of train/validation across all languages, and all 160
canonical projections pass frozen filters. If separately authorized, immutable
registry v0.16 would contain 17 sources and the accepted planning mix would rise
from 7,256,670 to 7,266,143 train supervised tokens, leaving 2,733,857 to the
10M floor. Canonical materialization, registry v0.16, full mix re-audit, data
seal, and SFT remain unexecuted gates.

The active AI-assisted pilot now contains all 320 exact rows: 240 train and 80
validation. All 320 are normalized-unique and passed deterministic schema,
message-order, task-correctness, language-mode, arithmetic/tool-result,
identity/tool-claim, PII/secret, safety, source-binding, and benchmark filters.
The private candidate is content-addressed by source revision
`0456f47ceaa1b462ee1a5f48fb7635e006e7e09295d01ad1f569185a5e04b10e`.
A deterministic 32-row owner sample passed on 2026-08-06. The source remains
candidate-eligible but deliberately not production-eligible because it is a
small behavior pilot and the complete task/source mix is not ready.

The newer safety/tool pilot is deliberately separate from that general
behavior pilot. It contains 320 machine-validated rows covering five safety and
five honest/tool-grounded behavior skills in four language modes. Every row
fits the frozen 4K tokenizer contract. Exact measurement found 39,325 total
serialized tokens and 25,024 supervised tokens; train contributes 18,636
supervised tokens and validation contributes 6,388. The maximum row length is
256 tokens. It has explicit AI provenance, no human-authorship claim, no API or
GPU use, and zero production dataset promotions. Its deterministic 40-row owner
sample was accepted 40/40 with zero rejections, and the fail-closed receipt
passed. An append-only successor-registry gate then registered the exact lane as
`candidate_eligible`; the old registry and active production contract were not
changed.

The older human-only packet workflow still reports `320 blank / 0 in progress /
0 accepted / 0 invalid` and remains available as an optional higher-assurance
track. It is no longer the active blocker. The source contract now distinguishes
a hash-pinned AI-assisted candidate from a reviewed production source; the
candidate can pass local filters but the production seal rejects it until its
owner and full-mix blockers are cleared.

The response-language contract is now explicit: an explicit user request wins,
then the latest unambiguous user-message language, then the established
conversation language, and finally English as the no-context fallback. The
metadata-only policy matrix passes all 216 combinations, including 180 explicit
override combinations. No model inference was run, so this is contract evidence,
not a claim that the future SFT checkpoint already follows the policy. English,
Bengali script, Banglish, and natural code-switching are the core release modes;
other languages remain best-effort until each has suitable data and frozen
evaluation evidence.

The punctuation contract is also explicit but not yet executed. It covers
English, Bengali script, and Banglish across six correction classes and four
plain/protected contexts. Aya's reviewed sample may improve Bengali
naturalness, but it is not treated as punctuation proof. Actual post-SFT
inference, private output review, and at least 288 cases are required; any
meaning change or mutation of code, commands, URLs, paths, decimals, versions,
or math fails the gate.

## Quality-first path to the best defensible Lumi L1

1. **Completed:** review the deterministic 32-row owner sample from the
   320-record AI-assisted pilot while preserving explicit AI provenance. The
   signed receipt passed, but it does not promote the small pilot into the
   complete production mix. The optional human-only workflow remains available.
2. **Completed fail-closed:** audit representative samples from OASST1, Aya
   Bengali, and OpenR1 for structure, provenance, language, answer quality, and
   task coverage. No source was promoted. Repair the measured defects, expand
   the bounded evidence where required, and re-audit before promotion.
3. **Completed as planning:** define exact 10M-30M task and language bands and
   eight proposed source lanes without authorizing acquisition or training.
4. **Completed as machine validation only:** build the first Lumi-owned
   safety/tool lane and measure it with the frozen tokenizer. Its 18,636 train
   supervised tokens are useful seed coverage, not a complete SFT dataset.
5. **Completed:** the human owner accepted all 40 safety/tool sample rows and
   the raw-text-free receipt passed.
6. **Completed as candidate-only:** register the safety/tool lane in immutable
   successor registry `v0.2`, closing candidate/canary task coverage while
   retaining every production blocker.
7. **Completed locally, no fetch:** freeze OASST1's exact two-split object/byte
   envelope, strict English human root-direct/rank-0 quality policy, 35% source
   cap, and raw-text-free attribution contract. The pinned sample preflight
   passed; acquisition and promotion remain unauthorized.
8. **Completed, non-production:** the bounded four-object OASST1 CPU run fetched
   the pinned revision, processed exactly 88,838 rows, selected 1,763 strict
   pairs, and retained 1,745 records / 477,456 supervised tokens after all
   required gates. Independent verification passed; no source was promoted.
9. **Completed human quality sample:** the owner accepted all 32 exact OASST1
   sample rows and the raw-text-free fail-closed receipt passed. This does not
   claim first-party authorship, source promotion, or SFT permission.
10. **Completed candidate-registry promotion:** immutable successor registry
   `v0.3` changes only the existing OASST1 entry to the exact reviewed adapter,
   pinned evidence URLs, English-only declaration, and remaining production
   blockers. Public human sources retain schema status `canary_eligible`; this
   registry promotion is not production eligibility.
11. **Completed bounded materialization:** replay the exact 1,534-row Aya
   Bengali inventory through the frozen schema, PII/secret, benchmark, 4K,
   assistant-only token, normalized-deduplication, terminal-pipe repair, and
   annotator-concentration policies; privately materialize the resulting 1,478
   records / 577,453 supervised tokens, attribution manifest, and deterministic
   32-row owner packet while keeping every production gate closed.
12. **Completed owner quality sample:** accept all 32 exact Aya sample rows and
   seal the raw-text-free receipt with zero production promotions.
13. **Completed candidate-registry promotion:** immutable successor registry
   `v0.4` changes only Aya's reviewed metadata entry, binds its exact candidate,
   attribution, and 32/32 owner gate, and retains every production blocker.
14. **Completed (Aya split and OpenR1 full curation):** Aya's deterministic
   1,330-train / 148-validation split is now
   privately materialized at 520,269 / 57,184 supervised tokens with the exact
   preflight assignment and zero production promotions. OpenR1's bounded
   18-trace slice also resolves all six global phone-regex hits as five
   line-scoped math-number collisions plus one cross-message join artifact;
   unresolved is zero and the global filter remains unchanged. Its exact pinned
   93,733-row/10-shard source was then acquired and curated under the bounded
   CPU contract. The visible-final repair retained 2,656 records / 1,170,745
   supervised tokens after quarantining 32 malformed-style rows, and its owner
   sample passed 32/32.
15. **Completed fail-closed (OpenR1 rights map):** independently reproduce all
   2,656 attribution rows across `olympiads`, `cn_contest`, `aops_forum`,
   `amc_aime`, `inequalities`, `olympiads_ref`, and `number_theory`. Dataset-level
   Apache metadata was not accepted as per-row upstream-rights proof, so zero
   records were cleared and all 2,656 remain quarantined.
16. **Current (DMath promoted to candidate registry; DeepMind canary blocked
   safely before network):** the exact
   three-object / 12,206,887-byte DMath train boundary at revision
   `149a7403619ac1bd6ed10f50694ad82699d5cc03` is privately acquired; the object
   is an ID-keyed 7,943-row JSON root and the official test split remains
   untouched. An isolated fail-closed Python verifier accepts 6,967 rows. The
   global schema, PII/secret, exact-deduplication and benchmark gates reject 18
   more, exact 5-token-shingle Jaccard at 0.90 rejects six, and exact 4K frozen
   tokenization retains 6,943 records / 51,026 supervised tokens split as 6,596
   train records / 48,480 tokens and 347 validation records / 2,546 tokens.
   Abstract expressions and Python programs are hashed verifier evidence only,
   never SFT targets. Independent verification passes with zero cross-split
   exact or prompt-near duplicates. The deterministic owner quality sample is
   human-owner accepted 32/32; its raw-text-free receipt and independent replay
   both pass. A separately authorized append-only promotion then registered the
   exact 6,943-record / 51,026-token candidate as the eleventh
   `canary_eligible` source in immutable registry `v0.5`, while preserving all
   ten predecessor sources and leaving the active production data contract
   unchanged. The planning train total is only 1,050,680 tokens, leaving
   8,949,320 to the 10M floor, so another rights-clear source lane remains
   necessary. DeepMind's Apache-2.0 procedural mathematics adapter preflight is
   locally complete and independently verified. Its separately authorized
   30-object / 234,672-byte acquisition and 2,000-record canary reached the
   required dependency gate and stopped fail-closed before network because the
   isolated runtime has NumPy 2.5.1 and SymPy 1.14.0 but lacks `absl` and
   `six`. Independent replay proves zero source objects, network bytes, decoded
   bytes, generated records, or package installations. **Next gate:** resolve
   and pin the missing runtime dependencies under a new explicit authorization,
   then separately retry the exact source/runtime canary. The no-install
   resolution preflight is now complete: it verified the pinned 2,458-byte
   `setup.py` blob, parsed requirements without executing it, and bound official
   PyPI metadata to `absl-py==2.5.0` and `six==1.17.0` universal wheels with
   exact hashes and a combined 148,460-byte ceiling. Existing NumPy 2.5.1 and
   SymPy 1.14.0 satisfy the source constraints. A separately authorized overlay
   installation then downloaded exactly those two wheels and installed them
   under `/private/tmp/lumi-deepmind-runtime-v0.1` using `--no-index --no-deps
   --require-hashes --no-compile`. Independent replay verifies two
   distributions, 40 files / 520,097 installed bytes, exact inventory hashes,
   and fresh-process `absl`/`six` imports from inside the overlay. Zero canary
   records were generated. **Immediate next gate:** separately retry the exact
   30-object / 2,000-row DeepMind source/runtime canary. Orca-Math stays
   rights-quarantined, NVIDIA OpenMathInstruct stays outside the allowed-license
   set, and GSM8K remains evaluation-only.
17. Deduplicate within SFT and against the base corpus where feasible, freeze
   benchmark contamination signatures, filter secrets/PII, then seal the exact
   raw records and encoded shards.
18. Finish the 8xH100 FSDP assistant-only trainer. Require supervised-token
   weighted gradients, globally consistent padding, exact source-checkpoint
   import, optimizer/RNG/data-cursor resume, immutable checkpoints, and
   train/validation separation.
19. Run the production-seal-bound CPU canary, then a separately authorized
   two-step GPU run plus fresh-process resume. Any loss, label-mask, parity,
   HBM, NaN/Inf, checkpoint, or cursor failure blocks SFT.
20. Compare `5e-6`, `1e-5`, and `2e-5` at equal supervised-token budgets; test
   one versus two epochs only if dataset size supports both. Select on held-out
   assistant-only loss plus human quality, safety, Bengali/Banglish, and base
   capability retention—not throughput alone.
21. Run bounded full-parameter SFT from the sealed 80B-token base. Preserve the
   base checkpoint and seal every resulting artifact independently.
22. Evaluate instruction following, knowledge/reasoning, math, code,
   truthfulness/calibration, Bengali/multilingual behavior, greeting/typo
   recovery, punctuation, multi-turn context, tool honesty, and all required
   safety suites.
   Compare against both the base model and a generic SFT baseline.
23. Add preference optimization only if blinded comparisons reveal a stable
   behavior gap and reviewed preference pairs improve it without benchmark or
   safety regression. It is not an automatic phase.
24. Keep exact arithmetic and current-information guarantees in tools:
   calculator execution, retrieval/search with citations, and explicit tool
   capability honesty. Model-only arithmetic remains a diagnostic.
25. Benchmark BF16/FP16 and safe quantized serving, choose hardware from
    measured latency/concurrency/context memory, run an authenticated private
    deployment canary, and enable the website only after readiness passes.

Context extension beyond the current 4K contract is a separate research gate;
it should not be mixed into first SFT. RAG/search improves current and private
knowledge but does not replace model training, and it belongs after a stable
instruct checkpoint and tool-use evaluation.

## SFT execution order

`configs/lumi-4b-sft-execution-plan-v0.1.json` now binds the final base,
tokenizer/chat contract, source registry, identity/behavior specifications,
first-party authoring pilot, and basic-capability gate by exact SHA-256. It also
records the current blockers rather than pretending the generic pretraining
trainer is SFT-ready.

The encoded-shard writer is now implemented and hash-bound to the production
dataset seal. Production shards still cannot exist until reviewed records and
eligible sources produce a real seal.

The dedicated single-device SFT runtime now passes a synthetic two-step
assistant-only-loss canary and produces the exact same final model when run
continuously or as a fresh-process one-step plus resume. This proves masking,
supervised-token-weighted accumulation, optimizer/RNG/data-cursor persistence,
and immutable checkpoint verification on CPU. It does not replace the later
production-seal-bound CPU canary or the separately authorized 8xH100 FSDP
smoke/resume gate.

The required order is: use the completed eight-lane task/language proposal and
the candidate-registered Lumi-owned safety/tool lane; repair the measured
OASST1, Aya, and OpenR1 defects; acquire the remaining coding, factual-QA,
writing, reasoning, conversation, multilingual, and safety/tool coverage under
separate bounded source plans; measure, re-audit, and promote only
evidence-complete sources; seal and
materialize exact encoded train/valid shards; implement the
distributed assistant/tool-only masked-label path; pass a production-seal-bound
CPU loss/resume canary; then, under separate approval, run a two-step GPU
smoke/resume, an equal-token learning-rate screen, bounded full SFT, checkpoint
sealing, and all frozen quality/safety/basic-capability evaluations.

The provisional screen uses full-parameter FSDP BF16 with learning-rate
candidates `5e-6`, `1e-5`, and `2e-5`, one- and two-epoch candidates, cosine
decay, 3% warmup, weight decay `0.01`, and gradient clipping at `1.0`. These are
screen candidates, not a preselected winner; final steps, tokens, epochs, batch
geometry, attention backend, and cost remain blocked on the sealed dataset and
measured GPU evidence.
