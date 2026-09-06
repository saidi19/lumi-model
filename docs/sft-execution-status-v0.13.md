# SFT execution status v0.13

Local engineering checkpoint, 2026-09-06. This is a synthetic integration result,
not an execution approval, dataset seal, private-integration pass or GPU run.

## Completed in this step

The parent-owned ledger now has an actual local process integration path. A fixed
inert worker exchanges bounded canonical messages over inherited AF_UNIX
socketpairs. One cumulative Supervisor governs the exact three scheduled
invocations: reference 0→2, resumed 0→1, resumed 1→2. Each invocation uses two fresh
processes and fresh epoch/rank capabilities; no failed invocation is retried.

New versioned files, without changing their historical predecessors:

- `src/lumi/posttraining_sft_attempt_journal_v1.py`: exclusive private journal,
  bounded metadata-only hash chain, conservative write precharge and fsync.
- `src/lumi/posttraining_sft_epoch_protocol_v1.py`: one-invocation sessions,
  epoch-bound requests/ACKs, sequenced confirmations and cumulative accounting.
- `src/lumi/posttraining_sft_local_ipc_v1.py`: durable parent adapter, actual IPC,
  fixed synthetic subprocess lifecycle, deadlines and cleanup.
- `scripts/sft_synthetic_ipc_worker_v1.py`: stdlib-only inert child entry point.
- Three corresponding new test modules; existing tests and bindings are unchanged.

The successful path ran six child processes across three epochs, four global
optimizer steps and 18 confirmed protocol transactions. Six checkpoint bytes were
**synthetically reserved**; no actual checkpoint, model, source-row or candidate
was read. These counters are not measurements of a real training workload.

## Durability and shutdown semantics

The full 65,536-byte journal budget is predebited once before creating the first
record. Every attempted write must also fit within that prepaid budget, including
short-write retries inside a single append. Work reservations are recorded and
fsynced before the parent sends their ACK. Failed sends never refund a reservation.
An existing journal, even empty or truncated, prevents opening another attempt in
the same directory. There is no crash-resume or failed-pass retry API.

At invocation end, the protocol is sealed while workers wait for stop. Expected
EOF is accepted only after this seal. Durable epoch completion is recorded only
after every child has exited successfully and been reaped. A failed cleanup blocks
the next epoch. Failure cleanup precedes best-effort failure journaling.

A journal's `attempt_completed` event records completion of the work/retirement
stage; it is **not** an overall pass, authorization or proof of a successful close.
The final report must additionally pass deadline, cleanup and journal-close checks.
Tests explicitly reject an overall pass after a late deadline or close failure,
even if a complete terminal record is present. Hash chaining is not protection
against wholesale rewriting by the same user; an external expected digest is
required to distinguish removal of complete trailing frames.

## Verification

| Check | Local result |
| --- | --- |
| New journal tests | 31 passed |
| New epoch protocol tests | 61 passed |
| New actual IPC/lifecycle tests | 24 passed |
| New suites together, `python -S -B -m unittest` | 116 passed, no skips |
| Same new suites under ordinary pytest with `PYTHONPATH=src` | 116 passed |
| Frozen seven control suites | 273 passed, no skips |
| Ruff on all seven new Python files | Zero findings |
| Original OPEN inventory | 327 retained; all 123 bound test files unchanged |
| Source/private partition binding verification | Intact; private readiness remains BLOCKED |

The 116 cases are run in two environments, not 232 unique tests. Together with the
273 prior control cases, the unique targeted control coverage is 389 cases. This
is not a full repository test result. An initial ordinary-pytest invocation omitted
`PYTHONPATH=src` and failed before collection; the corrected source-path invocation
passed. No test was skipped, routed out, weakened or marked xfail to obtain this.

Adversarial cases include wrong/stale epochs and tokens, replay, concurrent shared
budget exhaustion, partial requests/ACKs, lost confirmations, stalled workers,
bad exit status, extra bytes after shutdown, journal write/fsync/close failures,
cleanup errors and late deadline expiry. The worker exits on loss of its inherited
channel. This does not claim arbitrary descendants are contained after parent loss.

The existing CI source-suite routing automatically includes these new tests. The
IPC tests execute in fresh `-S -B` children even when pytest has already imported
the ML stack. No workflow filter or historical source binding was changed.

## Still open, without adding or renumbering roadmap gates

The original roadmap remains four of eight gates fully closed. Gate 5 (SFT
preparation) is partial; gates 6–8 (actual SFT, post-SFT evaluation, serving) remain
pending. This step advances the worker/supervisor prerequisite within Gate 5; it
does not close that whole prerequisite or any of the six pre-GPU categories.

In particular, inherited socketpairs provide channel-capability authentication,
**not OS peer-PID authentication**. The fixed synthetic worker is not a guarded real
SFT worker. Memory observations are synthetic, not host/GPU measurements. IPC
deadlines do not bound blocking filesystem syscalls or implement Linux cgroups,
network containment, escaped-descendant control or provider billing termination.
The synthetic wire budget is 131,072 bytes and the requested communication deadline
is at most 20 seconds; these are not total physical I/O or lifecycle cost guarantees.

Next, this integration needs a reviewed guarded-worker boundary with real process
identity and target-Linux containment, followed by the remaining authentic
S3/DCP import, private-data relocation, CUDA acceptance and complete execution
ticket. The frozen bootstrap and admission real-worker entries remain closed.

The 327 private/history integration nodes remain OPEN, never counted as passed.
No existing curation attempt, evidence publisher or source registry was replayed.
No model restore, external data access, GPU/instance action, inference, SFT,
promotion or production eligibility change occurred in this local verification.
The S3-backed Lumi-4B base pretrained on 80B tokens is unchanged.
`production_eligible=false` remains mandatory.

This document records local validation only. It does not claim these new files
have been committed, pushed, or passed a subsequent Linux/GitHub run.
