# SFT execution status v0.2

Verified on 2026-09-05. This status supersedes the historical “current” labels
in the v0.1 roadmap; it does not replace any immutable evidence.

## Completed current input chain

| Gate | Verified result |
| --- | --- |
| Private research data contract v0.2 | Active for the registry-v0.24 candidate mix; production eligibility remains false. |
| Private learning candidate seal v0.3 | 329,200 records: 309,643 train and 19,557 validation. |
| Token-consistent seal and shards v0.2 | 11,855,217 train and 939,293 validation supervised tokens; nine train shards and one validation shard. |
| Single-device CPU smoke | Both training paths completed; v0.3 audit recovery verified exact equality and disclosed the original v0.2 audit failure. No training retry. |
| Current-input distributed CPU preflight v0.2 | DDP and FSDP each passed uninterrupted versus fresh-process resume equality. Independent verification passed. |

The distributed v0.1 run used older shards and an older loader. Its top-state
audit did not compare the field inventory, allowing `checkpoint_schema_version`
to appear only in resumed states. Its logical-record count also did not describe
the legacy loader's full-file reads. The v0.2 successor records those limitations,
preserves all 68 historical files byte-for-byte, and closes the current-input CPU
gate using the hardened v4 loader and corrected distributed state recovery.

## Distributed v0.2 execution evidence

- [Frozen plan](../configs/lumi-4b-sft-distributed-cpu-preflight-v0.2.json)
- [Execution report](../evidence/lumi-4b-sft-distributed-cpu-preflight-v0.2.json)
- [Independent verification](../evidence/lumi-4b-sft-distributed-cpu-preflight-verification-v0.2.json)

Exactly six sequential two-process launches completed once: DDP reference,
pause and resume, followed by the same FSDP sequence. Each strategy reached
step 2, four records, 219 serialized tokens and 153 supervised tokens. Model
tensors, rank-local optimizer/RNG states, stable metrics and top-state fields
match within each strategy. DDP-versus-FSDP bitwise equality is not asserted.

The execution consumed 16 logical training samples and eight global optimizer
steps across all paths. Twelve rank-local dataset loads read 48 index rows,
21,024 encoded binary bytes and 7,728 index bytes; no complete shard hashes
were read. Verification added zero encoded binary/index reads or optimizer
steps. Combined execution and verification runtime was approximately 16.03
seconds. The final private artifact footprint including report and verification
was 168,654,337 bytes. The two historical diagnostic scans read 337,275,704
bytes, below the plan's combined 360,000,000-byte cap.

There were zero external-network bytes, GPU use, inference calls or retries.
Localhost Gloo communication was used. Outputs are private (0700 directories,
0600 files), and `production_eligible=false` remains explicit. Twenty-five
targeted synthetic tests passed, including two-rank DDP/FSDP resume and
checkpoint-verifier integration; Ruff passed.

## Remaining order

1. Prepare the SFT-specific GPU smoke readiness and launch plan locally. Bind
   the final 4B base, current input lineage, execution scope, trainer, device
   geometry, attention/precision, resource limits, resume boundaries and audit.
   The generic pretraining CUDA plan is not a substitute for this SFT plan.
2. Before production use, resolve source/owner-review and production-policy
   gates, activate the production contract under its approval, create the
   production seal/shards and verify a CPU canary bound to them. Current private
   research artifacts do not grant production eligibility. Any research-only
   GPU plan must explicitly preserve its separate research scope.
3. Obtain the separately required bounded GPU/remote execution approval, then
   run the two-step GPU smoke and fresh-process resume; measure memory, numerical
   stability, masking, checkpoint import and resume correctness.
4. Run the equal-token learning-rate screen and select using held-out loss,
   quality, safety and base-capability retention. Then run bounded full SFT and
   seal its checkpoints.
5. Complete the frozen instruction, language, reasoning, coding, safety,
   tool-honesty and retention evaluations; apply preference optimization only
   if its separate comparison gate warrants it.
6. Complete serving benchmarks, authenticated private deployment and product
   readiness before public release.

No 4B GPU SFT, production seal, remote operation or promotion was performed by
the distributed CPU gate.
