# SFT execution status v0.3

Verified locally on 2026-09-05. The next-step metadata readiness audit and
fresh-process structural replay are complete. **GPU launch remains blocked.**
This updates the roadmap status, not any frozen predecessor evidence.

## Completed chain

The private-research v0.2 data contract, registry v0.24, token-consistent seal
and encoded-shard evidence, recovered single-device CPU smoke and distributed
CPU v0.2 evidence remain hash-bound. The current metadata still reports
329,200 records: 309,643 train and 19,557 validation, with 11,855,217 train and
939,293 validation supervised tokens. Production eligibility remains false.
The prior CPU fixture was tiny; its success is not a final-4B or CUDA result.

The new audit rechecked all 29 bindings inherited from the frozen distributed
CPU plan plus the final-base binding, source registry, distributed evidence,
exporter and new readiness implementation. It checked the existing 31-file
encoded stat inventory without opening any index/token/label content. The
final 4B binding still identifies step 76,294 and 80 billion pretraining tokens;
model weights and the historical remote backup were not reverified here.

Artifacts:

- [Non-executable launch/readiness plan](../configs/lumi-4b-sft-gpu-smoke-readiness-v0.1.json)
- [Metadata audit report](../evidence/lumi-4b-sft-gpu-smoke-readiness-v0.1.json)
- [Fresh-process verification](../evidence/lumi-4b-sft-gpu-smoke-readiness-verification-v0.1.json)

The published audit read 537,613 metadata/code bytes in 39 files. Verification
read 542,240 bytes in 40 files, including the report: 1,079,853 combined bytes.
Report and verification occupy 5,917 bytes, both private mode 0600. These
counts describe the two published passes, not exploratory code inspection or
synthetic test fixtures. No source/candidate/test rows, encoded data content or
model-weight content was read by either pass. Neither used network, remote
actions, GPU, inference or optimizer steps. Verification is a fresh-process
replay using the same implementation, not an independently written algorithm.

Twenty-six synthetic tests pass, including end-to-end report/replay, path and
hash rejection, stat-only access, read caps, exact multi-launch accounting and
refusal to overwrite evidence. Before publication, development checks found
and corrected a draft `1` versus `1.0` representation mismatch and an AST check
that confused reading the input checkpoint manifest with writing an export
manifest. No training launch occurred and no previous evidence was overwritten.

## Six pre-launch conditions, in dependency order

1. **Final-4B import adapter and receipt.** The SFT worker imports a local HF-style
   checkpoint and separately requires an exact `manifest.sha256`. The existing
   DCP exporter writes `distributed-export-report.json`, not that worker checksum
   manifest. First implement and synthetically test a versioned adapter that
   binds the original DCP manifest, final step, model config, tokenizer, dtype,
   output inventory and worker-format checksum manifest. Do not reinterpret the
   tiny CPU source as the 4B checkpoint. Real export/restore is a separately
   bounded operation; no final-4B import receipt/path is currently bound here.
2. **SFT-specific authorization and bounded launcher.** The existing SFT worker
   calls the generic pretraining authorization validator. Its old smoke or
   production approval is not an SFT approval. Create a versioned SFT-specific
   validator/launcher binding objective, private scope, final source, relocated
   data, exact code, all three launch identities, resources and no-retry markers.
   Enforce approval before data/model reads and distributed initialization,
   with per-rank checks, fail-closed process shutdown and no automatic retry.
3. **Target host, runtime and total resource budget.** Bind exact host identity,
   device count/identity, driver/CUDA/NCCL/PyTorch/runtime versions, code-tree
   digest, private storage and CPU RAM/GPU HBM headroom. Freeze combined runtime,
   cost, checkpoint read/write and transfer limits and enforce them. The draft
   retains eight-rank FSDP geometry from the historical plan, but that does not
   assert host availability or memory fit. All unknown caps remain null.
4. **Content-verified data relocation.** Preserve the current manifest, inventory,
   seal and file-content hashes. The local stat binding includes inode/device,
   ownership, ctime and mtime; it cannot simply be copied to another host. Verify
   content and private modes at the target under a bounded transfer/audit plan,
   then create a fresh host-local metadata binding. Never silently drop the
   fifth loader binding or widen the private-research scope.
5. **Frozen CUDA acceptance and fault tests.** Prepare synthetic masking/padding,
   BF16 finite-loss/gradient, per-rank token accounting, import, checkpoint and
   interrupted/fresh-resume tests. Freeze numeric acceptance before execution;
   do not relax it after a failure. State/key inventories and data cursors must
   match exactly. Record peak HBM and compare model, optimizer/RNG and stable
   metrics. Eager is the draft attention path; SDPA/FlashAttention remain
   separate parity gates before selection. No cross-conversation packing.
6. **Exact execution approval.** Only after the preceding bindings are complete,
   obtain the separately required approval for the precise GPU/remote operation
   and caps. No reuse of earlier teacher, pretraining, CPU or curation approvals.

The next local implementation task is condition 1's adapter/receipt contract
and synthetic tests, then condition 2. Neither needs actual GPU execution.
Host preparation, model export and data transfer remain separate real actions.

## Smoke design, not permission to run

The draft uses FSDP full sharding, eight CUDA/NCCL ranks, BF16, eager attention,
one record per rank/microbatch and a fixed eight-record train prefix. Its three
sequential launches are reference step 0→2, comparison step 0→1, and a fresh
process resuming that comparison step 1→2. The two-step schedule stays identical
across paths. The prefix repeats across epochs deliberately: this is an
infrastructure/resume test, not a data-quality or learning-progress claim.

Across all paths that would be four global optimizer steps, 32 logical sample
events, 24 rank-local dataset loads and at most 192 index-row reads. Selected
token/label payload is bounded by 6,291,456 bytes, and index reads including
oversize detection by 33,792 bytes. These are loader-only bounds, **not** total
checkpoint I/O, memory, runtime or cost budgets. No executable command or
authorization is emitted while the six conditions remain open.

## Subsequent roadmap, without skipping gates

After pre-launch closure: separately authorized GPU smoke and fresh-resume
verification → equal-token learning-rate screening and held-out selection →
bounded full SFT and checkpoint verification/seal → frozen instruction,
language, reasoning, coding, safety, tool-honesty and base-retention evaluation
→ conditional preference optimization only if warranted → serving benchmarks,
authenticated private deployment and product readiness.

Before **production** use, close the source/owner-review and production-policy
gates, activate the production data contract, seal/materialize that lineage and
run its bound CPU gate. Private-research CPU or GPU success cannot grant those
rights. Public release remains a further approval/readiness gate.

## Frozen publication hashes

- Plan: `1f93535f30a275f6ee9d8fc2ae482087ebb9f02adc86734be0ad93b8d32ca082`
- Report file: `1a40ca5b6f228e19a6a90018b53768796cbbc46cfd83ce8ea30aba668a556861`
- Report internal: `2772bbe60b519428d2660c421d123e2e73a7581b6bb86e6b24c351fa4b24f039`
- Verification file: `ca0331ab6ddcd01b842a275b47378cd28c7bcb205e39c3b8bd25fe54ab3be9ec`
- Verification internal: `48f6dac1339003453bed70a4faad4fda76cbfb47e53725adc9e834e65eeffc2e`

The audit command is `scripts/check_sft_gpu_smoke_readiness_v1.py` with
`--expected-plan-sha256` and optionally `--verify`. Both outputs now exist;
the tool intentionally refuses to overwrite them. A future changed-input
readiness audit needs a versioned successor, not a rerun over this evidence.
