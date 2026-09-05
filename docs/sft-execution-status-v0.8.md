# SFT execution status v0.8

Verified on 2026-09-05. Local runtime-control components now pass **106 tests**,
including real local POSIX synthetic child-process tests. A separate fresh-process
replay also passes with no failures, errors or skips. The prior source-review
package's 167 tests were rechecked successfully. **No real GPU SFT worker has
been dispatched and the integrated launcher is still incomplete.**

## GitHub versus this Mac

The user's concern is correct: Git pushes commits, not loose working-directory
changes. At this turn's initial read-only inventory, branch
`codex/lumi-posttraining-product-v0.1` was still at local commit
`38499dce819bf0afb168448f2289cff0b8b047ad` (2026-08-09). There were 18 modified
tracked files, all unstaged, zero staged files, zero conflicts and 1,770 untracked
files. Additional source/control/evidence documents from this successor increase
that count. Untracked does not mean uploaded, committed or backed up remotely.

The initial untracked inventory was scripts 489, source 309, tests 253,
configs 297, evidence 403, docs 17 and ops 2. This is not an inventory of ignored
private datasets or model weights. The latest local launcher/status/base-binding
files were not tracked in that commit. The preceding live GitHub browser check
also found this branch at `38499dc`; GitHub was not refreshed in this turn.
These facts do not prove absence from every other remote branch or backup.

The pretrained model's S3 backup is separate and was located in the previous
inventory turn. The local source-review ZIP is private and limited to 34 selected
files; it is not an off-device backup of all current project work.

No staging, commit, push, reset, branch mutation or automatic `git add -A` was
performed. Before a future code-sync step, explicitly review the selected source,
configuration, docs and evidence; exclude credentials, raw datasets, weights,
private prompts and owner materials. A private GitHub repo is not a reason to
upload every untracked file indiscriminately. Any deployment must use a reviewed
versioned release, not blindly clone the old branch HEAD.

## Completed runtime work

| Component | Verified work | Still outside its scope |
|---|---|---|
| POSIX process watchdog | 33 tests: monotonic deadline, bounded discarded stdout/stderr, failed start/no retry, live process-group guardian, leader exit with grandchild, ignored TERM, actual parent SIGTERM/SIGHUP/SIGKILL, partial pipe-setup failure, final-deadline/cancellation checks and signal-handler restoration | Linux cgroup containment, escaped `setsid` descendants, memory/network/filesystem quotas, real SFT dispatch, provider termination and billing |
| Standalone bootstrap | 30 tests: no `lumi` or ML import, byte-pinned source/ticket, SFT approval scope/expiry, exact intended runtime and structural receipt bindings | Authentic OS attachment, observed live GPU/rank/runtime checks and worker handoff; `--execute` always rejects |
| Parent-owned ledger protocol | 43 tests: canonical bounded frames, rank tokens, strict sequence/replay refusal, predebit before ACK, confirmation, shared atomic budgets, all-rank optimizer completion, sticky disconnect/failure | Actual IPC transport, PID authentication, durable journal, fresh-process channel epochs/rebinding and worker integration |

The guardian stays alive to preserve process-group ownership until cleanup. Its
lifetime pipe is held only by the supervisor; abrupt parent loss makes it kill
its own group. Source bytes are copied to a private pinned child file. Parent
credentials and general environment are not inherited. Arbitrary trusted-child
code is not an OS sandbox: the utility cannot contain deliberate process-session
escapes or guarantee absence of all descendants on an unvalidated Linux target.

The execution deadline initiates shutdown; TERM grace, cleanup allowance and
synchronous filesystem/OS scheduling add lifecycle time. The result reports the
deadline and cleanup allowance separately. The caller must retain default SIGCHLD
semantics and must not independently reap the owned guardian.

Both published passes observed 21 direct `Popen` events in their parent process.
This count excludes nested guardian forks/children and is not a total process
count. Only synthetic scripts were executed; no model, dataset, GPU, inference,
network transfer or cloud-instance action was performed. The parent audit hook
does not become an inherited OS sandbox. Temporary synthetic fixture files were
created and removed. Reports contain only codes, hashes and aggregate results,
not child stdout/stderr or exception text.

The two published reports and distinct consumed started markers total 5,615
bytes, all mode 0600. This is published evidence size, not total temporary I/O.
Publication was not retried. The replay uses the same implementation/tests in a
fresh process, not an independently written validation suite. Independent source
review identified and corrected process-group ownership, parent-signal cleanup
and late-status boundary problems before publication. Ruff was not available;
no lint-pass claim is made.

## Preservation correction

A pre-publication recheck detected two later development edits in previously
published `posttraining_sft_supervisor_v1.py` and its test: an exception-context
handling change and its assertion. They were restored to the exact bytes in the
verified source-review ZIP before this successor. The prior ZIP, plans, reports
and markers were not rewritten. The restored old package again passes 167 tests
and reproduces its original archive hash. The new passes verify 54 historical
source/control bindings before and after tests. Future improvements to published
versions must be made in a successor, never by modifying the frozen files.

## Budget selection and real next work

The owner delegated necessary project spending. The assistant selected a first
bounded new-instance restore/smoke preparation envelope of **USD 150** and
**four hours total host lifecycle**, recorded in
[the next-run budget](../configs/lumi-4b-sft-next-run-budget-v0.1.json).
This is not a full-SFT budget, exact execution ticket, provider invoice limit or
claim that a billable instance has been created. The historical eight-rank draft
is unchanged. Candidate 8x A100 80GB pricing was checked on 2026-09-05; checkout
price/capacity/tax/transfer and a new target identity must be bound before launch.
The existing busy Lambda instance remains out of scope and untouched.

Do not repeatedly ask the owner to select a budget already delegated. Use this
bounded choice while completing the actual prerequisites, revise it explicitly
if evidence requires, and never equate stopping a child with stopping billing.
An exact new-instance termination path must cover provisioning, installs,
transfers, tests, output verification and teardown.

The same six prelaunch conditions from v0.7 remain partially open, not six new
stages. In dependency order, remaining engineering includes a versioned real
worker with pre-ML SFT admission; parent IPC plus durable predebits and
fresh-process epochs; target Linux memory/process/network controls; full
checkpoint/DCP/safetensors I/O accounting; bounded authentic S3 restore/import;
verified private data relocation; exact CUDA acceptance; and a complete
target-bound execution ticket. The synthetic import adapter's parameter cap was
not raised. Real base training is not being restarted or replaced.

After these close: GPU smoke/resume, learning-rate screening, bounded full SFT,
checkpoint/evaluation gates and private serving readiness. Public/production
rights and release clearance remain separate. `production_eligible=false`.

## Evidence and recheck

- [Runtime-controls plan](../configs/lumi-4b-sft-runtime-controls-v0.1.json)
- [Publication report](../evidence/lumi-4b-sft-runtime-controls-v0.1.json)
- [Fresh-process verification](../evidence/lumi-4b-sft-runtime-controls-verification-v0.1.json)
- [Process watchdog](../src/lumi/posttraining_process_watchdog_v1.py)
- [Standalone bootstrap](../scripts/sft_guarded_bootstrap_v1.py)
- [Parent ledger protocol](../src/lumi/posttraining_sft_ledger_protocol_v1.py)

Plan SHA-256:
`ede0adee1d544217287e3782ff555c99704830fc08cdb0a4dc6e270cb7fa2d70`

Report SHA-256:
`be68974d78e9550314b9b34541ff9a35093f96264e791ac5cbc8574400d82cf3`

Verification SHA-256:
`2ff426cdb98121d8e472a52f79c702f809493aeb9267c022d2cac76e1a5a9f0a`

Non-publishing recheck (still executes bounded synthetic local child processes
and temporary test fixtures; does not execute SFT):

```sh
artifacts/lumi-sft-import-runtime-v0.1/venv/bin/python -S -B \
  scripts/check_sft_runtime_controls_v1.py --mode check \
  --expected-plan-sha256 ede0adee1d544217287e3782ff555c99704830fc08cdb0a4dc6e270cb7fa2d70
```

Do not rerun publication or verification into the consumed evidence paths.
