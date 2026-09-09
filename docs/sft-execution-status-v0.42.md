# SFT execution status v0.42

IPC retarget failure localization, 2026-09-09. Gate 5 remains partial; no GPU,
real data/model, S3, inference, training or production admission is authorized
or performed by this source/remote-generated CPU step.

## Preserved exact-commit result

The five-file IPC successor was published as
`71eda784f59dee2bdf700aa98623664f0f8af4dc`, preserving the accepted filesystem,
netns, cgroup, WorkerRun, filter policy and all preexisting evidence.

[Source run 34312382330](https://github.com/saidi19/lumi-model/actions/runs/34312382330)
finished with three successful guard/validator jobs and one failed test job
`102341524726`: **1 failed, 4614 passed, 58 skipped, 329 deselected, 36 warnings,
8812 subtests passed in 366.62 seconds**. The only failing test was
`IpcLinuxTests.test_ipc_creation_connected_pairs_cleanup_and_pinned_worker_run`.
The fixed report identified case `normal`, component
`ipc_stream_retarget_fixture_failed`, and **cleanup_confirmed=true**.
This is not a successful IPC acceptance or a production/real-host admission.

[Spark run 34312382337](https://github.com/saidi19/lumi-model/actions/runs/34312382337)
and job `102341524382` passed: **3 tests, zero skips, 109.66 seconds**.
No failed run was deleted, cancelled or retried.

## Narrow diagnostic correction

The failed fixture placed disconnect, reconnect, bind/listen, addressed sendto,
addressed sendmsg and same-peer roundtrip under one `stream_retarget` label.
Consequently the observed report does not identify the failed operation or errno.
The correction adds 12 fixed STREAM/SEQPACKET retarget operation labels and
19 fixed failure categories. It retains no source, paths, socket addresses,
PIDs, raw errno values, exception messages or lexical data in public output.
An errno is projected through a finite mapping; even hostile exception text
and overridden errno properties are not consulted.

Every old allowed-error set and success/denial expectation remains unchanged.
Unexpected errors now reach the fixed projection rather than being collapsed
into a generic assertion. Success still requires four-case owned cleanup,
matching exposed STREAM/SEQPACKET controls, no retarget, and pinned WorkerRun.
The original failed code remains available in commit `71eda784`; only its new
IPC fixture receives this localized reporting correction. The seccomp filter,
old accepted fixtures/wrappers, workflow, policy and historical evidence are
unchanged. This is diagnostic improvement, not a claimed root-cause fix.

Primary-source review rejected two unproven shortcuts: CPython does not generally
swallow initial `connect()` EISCONN; and a read-only bind alone does not explain
EROFS for socket inodes. Destination permission checks occur before the kernel's
already-connected state check. No extra error was accepted based on conjecture.
See [CPython connect handling](https://raw.githubusercontent.com/python/cpython/v3.12.11/Modules/socketmodule.c),
[Linux UNIX socket operations](https://raw.githubusercontent.com/torvalds/linux/v6.14/net/unix/af_unix.c),
and [Linux inode permission checks](https://raw.githubusercontent.com/torvalds/linux/v6.14/fs/namei.c).

## Verification and execution boundary

Seven new pure tests cover fixed errno projection, unknown/type-malformed errno,
timeouts, expectation/Python failures, hostile exception subclasses and the
finite code/stage vocabulary. The Linux fixture/wrapper receive AST checks only
on the Mac; no native Linux fixture, Torch or model is imported or run locally.
Only this document, the localized IPC fixture and its new pure diagnostic module
and tests are in scope for the next commit. One exact-commit remote diagnostic
run is pending; the previous run is not rerun. If the new report still fails,
use its fixed operation/category to identify a correction before another run.

### Local checks passed; publication blocked before staging

Exact Ruff passes on the three changed/new Python files. The combined 31 filter
and seven diagnostic pure tests pass (38 tests, zero failures/errors/skips in
0.034 seconds); no ML imports occurred. The Linux fixture was AST-parsed only.
Eight predecessor files were verified unchanged: accepted filesystem/netns/
cgroup/WorkerRun fixtures, IPC filter, filter tests, Linux wrapper and v0.41.

The automatic approval service rejected the exact four-file `git add` request
because workspace credits were unavailable. The command did not start. No
staging, successor commit, GitHub write or new CI run occurred after this
localization correction. HEAD remains `71eda784f59dee2bdf700aa98623664f0f8af4dc`;
the IPC fixture is modified locally and the diagnostic module/test plus this
document are untracked. Unrelated files remain untouched.

Resume after credits/approval become available: recheck the exact branch and
scope, stage these four files only, publish a non-force single-parent successor,
then inspect its native fixed operation/category and cleanup result. Do not
claim the IPC behavior is fixed until native acceptance actually passes.

### User-confirmed credit recovery

After the user confirmed credits were refilled, the bounded GitHub read passed.
Live branch, local HEAD and tracking ref all still matched `71eda784`; the prior
Source failure and Spark success were completed, with no active run among the
latest four branch runs. Proceed with the reviewed four-file diagnostic commit;
the earlier approval rejection remains historical evidence, not a current
native-test outcome.
