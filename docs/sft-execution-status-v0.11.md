# SFT execution status v0.11

Resume verification on 2026-09-06 completed the previously blocked source-copy
check. This supersedes the pending-copy status in
[v0.10](sft-execution-status-v0.10.md) without changing its historical record.
No Git commit/push, cloud instance, GPU run, real model restore, private dataset
read or operational evidence publication occurred in this batch.

## Verified results

| Check | Result | Scope |
| --- | --- | --- |
| New stdlib guard/validator suite | 120 passed | Replayed in the complete tracked-source clone with reviewed new files overlaid; intentionally private inputs absent |
| Copied file identity | 8 of 8 hashes matched | Six newly copied files plus two previously copied manifest/plugin files matched the workspace |
| Explicit BF16 history contract | 29 passed; 2 required historical errors | All 31 tiny CPU cases ran in the clone; all 17 current v2 cases passed; explicitly not an all-tests-passed result |
| Fixed isolated control suite | 273 passed | Local fresh process; zero errors, failures or skips |
| Current private-research data-contract tests | 34 passed | Source clone, tracked predecessor metadata and synthetic records only |
| New historical filter-binding regression | 8 passed | Local installed runtime only; clone replay is pending |

The new eight tests exercise the real Aya and safety-URI binding resolvers with
temporary synthetic bytes. Both must accept exact hashes, reject same-size
changed content without changing the binding, distinguish missing files from
hash mismatches, and reject existing files outside the workspace boundary.
They do not reconstruct historical candidate outputs or approve historical replay.

The eight-test import chain requires `certifi`; its initial `-S -B` invocation
stopped before test bodies ran. The existing installed runtime's `-B` invocation
passed all eight. No dependency was installed and no import or validator was
mocked to evade that requirement. These eight tests are not part of the stdlib
120-test job; ordinary general pytest will discover them with its dependencies.

## Five historical hash failures remain unresolved

Both tracked historical plans expect filter SHA-256
`df8297e753e97028999c6bcd203edc2a00a02147f7d9ecb2255569c480363970`.
The current tracked `src/lumi/posttraining_data.py` is
`e520c3771869b7221848a456e59c3b15a279fb1ebdf700646600c774e25b74b5`.
The exact old 21,605-byte source exists in local Git commit
`e08bddc8fbd8afe416a0d09aed6a115f06870f36`; the current source is 36,837 bytes.
No old source was restored and no historical plan hash was refreshed.

The one Aya and four safety-URI failures are therefore actual historical binding
rejections, not absent files. The current 34-test suite covers the intended
backwards-compatible v0.1 and scoped v0.2 behavior, but cannot establish the
historical candidate outputs. Keep these five failures blocking until a reviewed
historical replay or explicit historical-test routing strategy is established.

## Remaining blocker and next action

Copying the additional eight-test module into the temporary source clone was
rejected by automatic approval review with a workspace-credit error. That copy
and its proposed clone execution did not run; no indirect route was used.
The earlier successful eight-file copy and 120-test clone result are unaffected.

After the workspace approval/credit service is restored and the remaining copy
is authorized, complete that exact clone replay. Continue the missing-input and
historical binding review before changing general-suite expectations. The local
source-only workflow job is prepared, but no new Linux GitHub result exists.
Full CI remains unresolved. Previous plans, implementations, tests, evidence and
markers are unchanged; the only tracked local edit remains the additive CI job.

Four of eight original roadmap gates remain fully closed. The six documented
pre-GPU conditions are still partially open. These CI checks do not authorize
private historical replay, GPU SFT or production release.
`production_eligible=false` remains unchanged.
