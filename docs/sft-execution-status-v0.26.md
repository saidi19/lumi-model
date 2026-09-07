# SFT execution status v0.26

Offline-first engineering checkpoint, 2026-09-07. The user will provide a
dedicated H100 host only when target-only tests are actually needed. No GPU
instance is requested, started, inspected or modified by this work. No SSH key,
PEM, account login or live price is needed for the current local implementation.
The existing busy instance and S3-backed pretrained base remain untouched.

## Scope correction: finish software before renting hardware

The automated Lambda provisioning path from v0.25 is on hold; it is not the next
dependency for a user-managed host. Do not manufacture account, authority or
create/recovery work merely to continue that path. Its existing source and tests
are preserved, not represented as integrated training controls.

H100 is a hardware-family choice, not a host binding. GPU count, memory capacity,
UUIDs, driver/runtime, available CPU RAM/storage and price are still unknown.
The old eight-rank draft and historical A100 cost are not H100 facts or approval.
Keep any future host check read-only until its separate execution scope is fixed.

## New local training integration

The new entrypoint is `scripts/run_sft_offline_integration_v1.py`, with a separate
stdlib-only admission module and a synthetic CPU integration module. It accepts
only `--run-synthetic-cpu`, under Python `-S -B`. Parent, runner and ranks each
check their actual pre-ML runtime before ML discovery/import or fixture access.
There is no real-input, CUDA, provider, credential or user-supplied-path option.
The frozen historical entrypoint remains unsuitable for real SFT admission.

The fixed fixture uses a freshly initialized tiny Lumi model, generated integer
tokens, two CPU/Gloo ranks and three sequential invocations: uninterrupted steps
0 to 2; a separate run from 0 to 1; then fresh-process resume from 1 to 2. It
checks actual training, checkpoint loading and equality of stable training state,
model tensors, optimizer tensors and RNG state. This is synthetic training, not
a final-base import, real-data run, GPU test or model-quality evaluation.

Development exposed a genuine compatibility issue: the installed Transformers
5.2 mask helper requires `cache_position`, which the frozen Lumi call omits.
The successor addresses the exact CPU/eager/no-cache call and independently
checks causal-plus-key-padding mask values. Padding blocks keys, not padded query
rows. Token-count agreement alone is not proof of masked-loss correctness.

The previous Source CI installation log used Torch 2.14.0 and Transformers
5.15.1, while the local runtime is 2.10.0 and 5.2.0. The
[official 5.15.1 helper](https://raw.githubusercontent.com/huggingface/transformers/v5.15.1/src/transformers/masking_utils.py)
has a different signature with optional block, layer and mask-materialization
arguments. Only explicitly supported signatures are accepted; returned masks
must still exactly match the independent oracle. A version range is not a claim
that every runtime in that range has been tested. Read-only GitHub/upstream
inspection was used for this compatibility check, so the entire engineering
turn is not described as zero-network. No dependency installation was performed.

All prior source, tests, launch contracts, budgets, private inventories, evidence
and started markers remain unchanged. New synthetic fixtures are temporary and
must be cleaned up; no private dataset or pretrained weights are accessed.

## Verification boundary

Completed checks: 22 new stdlib admission tests and 30 integration/boundary tests
pass. A fresh combined run passed all 52 in 6.620 seconds, without skips; it
included real two-rank CPU training and fresh-process resume, not mocked optimizer
steps. The 273 frozen control tests also passed with no skips: 325 unique targeted
tests total. Repeated runs and in-test cases are not additional unique tests.
Exact isolated Ruff passed for all five new Python files. Frozen lint retained
exactly 815 findings in 163 unchanged files, with zero new findings; this is not
an all-source-lint-clean claim. No future GitHub result for this successor is
preclaimed by this checkpoint.

The fixed real run used 3,376 parameters, six rank processes and four global
optimizer steps across the reference and resumed paths. The trainable embedding
tensor changed. Four checkpoints and four optimizer/RNG pairs were verified;
final model, optimizer, RNG, stable state and metrics matched exactly. Temporary
fixtures were cleaned. The CPU adapter is scoped/restored and explicitly labelled
in the report; it is not installed into a real GPU training entrypoint.

Local Gloo uses loopback sockets. The managed sandbox initially denied binding;
an explicitly scoped localhost-only CPU test is distinct from cloud networking.
Offline library settings and Python network guards are defense in depth, not an
OS packet meter. A process-group timeout while the controller survives, followed
by cleanup, is not a parent-death guardian, hard aggregate memory/write quota or
billing control. The 180-second timeout triggers teardown, not a claim that all
teardown time fits inside that number. Catchable cancellation is latched without
interrupting cleanup; default SIGCHLD and no external child reaper are required.
Rank exit codes are checked. After an audited coordinator report/EOF, its owned
process group is deliberately retired before the leader is reaped; a natural
coordinator exit is not asserted.
The private-tree size check measures retained tree size, not cumulative writes.
Those limitations must remain explicit in the machine report and final handoff.

## Finite pre-GPU workboard

These are deliverables within the same preparation phase, not additional roadmap
gates or a claim that the 327 private test nodes are 327 user steps.

| Deliverable | Offline work still required |
| --- | --- |
| Actual training connection | CPU training/resume and local compatibility verification are complete. Connect real worker operations to SFT-specific admission and resource accounting; a tiny fixture is not that full connection. |
| Enforced resource controls | A training-capable Linux descendant/cgroup, deadline, private-output and network policy; synthetic fault tests. The old inert IPC policy cannot run Torch and must not be reused as a proven ML sandbox. |
| Real-size CPU import and relocation | A bounded successor preserving chunk coverage, keys, shapes, dtypes, tied aliases, finiteness and reload checks; tokenizer/content/receipt tests. Do not merely lift the tiny-fixture caps or substitute the weaker legacy exporter. |
| Reproducible release and test procedure | Complete source plus dynamic/runtime dependencies, Linux/CUDA version plan, fixed smoke/resume/fault criteria, exact run/stop instructions and integrated offline verification. The existing source-review ZIP is not a deployable release. |

CPU DCP-to-HF conversion does not intrinsically require a GPU. Its implementation
and synthetic tests can finish now. Actual conversion additionally needs an
authorized exact checkpoint/tokenizer source and adequate bounded CPU RAM,
storage and I/O. A portable CPU import receipt is separate from the frozen
target-local receipt; target host/path and relocated content must be checked
later. No S3 download or actual 4B conversion is performed by this checkpoint.

Only after feasible offline work is complete should the user turn on the H100
for genuinely target-dependent verification: exact host/boot/GPU/runtime and
ownership checks, target-local import/relocation receipt, CUDA precision/masking,
memory, distributed resume and fault tests. Final geometry and runtime limits
must be chosen from observed hardware; successful CPU tests do not supply them.
Exact bounded execution approval remains required before real GPU SFT. User
intent to use H100 is not permission to reuse a busy instance or start a rental.

## Roadmap, without conflating research and production

The prior base/data/CPU foundations are recorded in v0.5. This work is still
preparation for real 4B SFT, not SFT completion. The original Gate 5 label includes
production source/owner clearance; it must not obscure the separate private
research route or imply that every local software step is blocked on production
approval. Production eligibility remains false.

After prelaunch conditions close: GPU smoke and fresh resume; equal-token
learning-rate screening; bounded full SFT and checkpoint verification; frozen
capability/safety evaluation; then authenticated private serving and product
readiness. Production additionally needs its source/rights/owner/policy and
release approvals. No finish date or percentage follows from counting these
unequal tasks, and an old four-hour instance cap is not an SFT completion estimate.
