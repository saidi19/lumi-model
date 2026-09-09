# SFT execution status v0.43

Connected-pair listener-state probe correction, 2026-09-09. Gate 5 remains
partial, with production eligibility and real-SFT permission false. This step
uses source inspection, tiny local stdlib tests and remote generated CPU CI;
no GPU, real model/data, S3, inference, training, promotion or seal is involved.

## Preserved diagnostic result

Diagnostic commit `bdc0a4e67694d41a6b129ea926912e284d99cb05` was published after
approval-service recovery. Its [Source run 34314108918](https://github.com/saidi19/lumi-model/actions/runs/34314108918)
finished with three guard/validator jobs successful and test job `102346603444`
failed: **1 failed, 4621 passed, 58 skipped, 329 deselected, 36 warnings,
8828 subtests passed in 306.12 seconds**. Only the IPC Linux wrapper failed.
The fixed result was `ipc_stream_retarget_bind_listen_expectation_failed`,
case `normal`, **cleanup_confirmed=true**. Earlier disconnect and reconnect
expectations therefore completed before this failure. The diagnostic alone
does not distinguish bound-path readback failure from a zero-return listen.

[Spark run 34314108967](https://github.com/saidi19/lumi-model/actions/runs/34314108967)
and job `102346603791` passed **3 tests, zero skips, 100.66 seconds**. The
previous failed run `34312382330` and commit `71eda784` remain preserved.
Neither failed run was deleted or retried.

## Source-proven probe assumption defect

In [Linux v6.14 unix_listen](https://github.com/torvalds/linux/blob/v6.14/net/unix/af_unix.c#L734),
an already-connected bound STREAM/SEQPACKET socket returns EINVAL. In
[Linux v6.17 unix_listen](https://github.com/torvalds/linux/blob/v6.17/net/unix/af_unix.c#L776),
`prepare_peercred` can set the local result to zero before the established-state
early exit. Thus listen can report success while never transitioning to
TCP_LISTEN. The exact CI kernel version is not established by the observed
diagnostic; this source difference proves the old errno-only probe is not a
portable test of listener authority, not that the CI failure cause is already
fully confirmed.

The correction tests the actual invariant: exact integer SO_ACCEPTCONN=0
before and after listen, followed by a nonblocking accept that must reject with
EINVAL/EOPNOTSUPP, followed by another exact nonlistener readback. The
[SO_ACCEPTCONN implementation](https://github.com/torvalds/linux/blob/v6.17/net/core/sock.c#L1881)
checks TCP_LISTEN; the kernel UNIX accept path independently rejects any other
state. A successful accept, EAGAIN/timeout, actual state transition, malformed
readback, unexpected errno, or failed timeout restoration remains a failure.
An unexpected returned connection is closed once; its address is never retained
or used. Both the old explicit rejection and the newer no-op success require
all the state and accept checks.

This helper replaces every listen probe, including bound-then-unlinked sockets
after shutdown and the peer-death cases. Unlink does not erase the socket's
bound address, so limiting the correction to the first probe would be incomplete.
Socket-type and same-UID path readbacks, original-peer bidirectional delivery,
matching host sentinel counts, reconnect/retarget restrictions and cleanup
remain unchanged. The seccomp filter and every permitted family/type/protocol,
ABI restriction, denied syscall and old accepted fixture remain byte-for-byte
unchanged. No listener authority is newly admitted. Public diagnostic errors
are from a finite vocabulary, never exception/path/peer text.

## Verification handoff

Fifteen new in-memory socket-double tests cover both kernel outcomes, actual
listener transitions, malformed state, unexpected errno, nonblocking accept
rejection, unexpected accepted-connection cleanup and timeout restoration.
The full source suite can already have Torch imported by unrelated tests, so
the new test checks this module's import additions, not unrelated global state.
Linux fixture/wrapper files are only AST-parsed on the Mac, never imported or
executed. Native validation remains pending on the exact successor commit.

The exact 27-suite local regression passes **992 tests, zero failures/errors/
skips in 0.646 seconds**, with `ML_MODULES_LOADED=False`. Exact Ruff passes on
the three changed/new Python files; both Linux files pass AST parsing only.
All four frozen fixture hashes match the accepted bindings.

Only this document, the IPC fixture correction, the new pure probe module and
its tests are in scope. No workflow, filter, old fixture or previous evidence
is edited. A successful native acceptance must verify the same four lifecycle
cases, four installations, eight matching exposed-socket positive controls,
eight socket removals and the unchanged three WorkerRun invocations/six ranks.
