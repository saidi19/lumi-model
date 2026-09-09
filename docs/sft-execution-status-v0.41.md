# SFT execution status v0.41

Connected-pair-only UNIX IPC successor, 2026-09-09. This is a bounded source and
remote-generated CPU acceptance step, not GPU/real-data execution or admission.
Gate 5 remains partial. Production eligibility remains false.

## Accepted starting point and preserved evidence

Predecessor `cd9498da1d48c8d7a94436a6a9b7d0292e62e6e4` passed all four jobs in
[Source CI 34308560259](https://github.com/saidi19/lumi-model/actions/runs/34308560259):
4583 passed, 58 skipped, 329 deselected, 36 warnings and 8526 subtests in
382.13 seconds. Source selection was 4641; 327 private integrations remained
NOT_EXECUTED and two historical cases stayed separately routed. Full/private-CI
claims remained false. [Spark CI 34308560299](https://github.com/saidi19/lumi-model/actions/runs/34308560299)
passed three tests, zero skips, in 64.39 seconds.

The accepted filesystem fixture remains byte-for-byte pinned to
`31c3f271a15d412419fd0609497f92bc7bb0a4bf08f9a591e7086c956436ad89`.
Its nested netns, cgroup and WorkerRun bindings remain respectively:

- `70400a1428d0e54834721a5163db7724940a271bbeeeaec50714f28108f20bfc`
- `f91ed0c353a20c309ff3e1bfaab8dbe990b28c36626fdeeb0ac5131e0cf75fad`
- `53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0`

No predecessor source, fixture, test, workflow, policy floor, failed run or
evidence is edited. Unrelated local files are excluded. Local v0.40 remains a
separate uncommitted observation; this document records accepted results for
the new source handoff without modifying that historical snapshot.

## Remaining socket authority and exact policy

Landlock ABI 5 and a read-only mount do not prohibit connecting to UNIX sockets
inside an exposed software tree. The new filter addresses creation of those
endpoints while preserving the paired channels used by the accepted WorkerRun.

The deterministic classic-BPF policy requires the native x86-64 syscall ABI;
other architectures, x32-bit-tagged and historical x32 syscall aliases are
terminally rejected. It denies `socket(AF_UNIX)` and permits `socketpair` only
for AF_UNIX STREAM or SEQPACKET with protocol zero and the two known socket
flags. Datagram pairs are denied because they can reconnect/retarget. The
family/type/protocol checks validate full-width arguments, not only truncated
low words.

New sockets otherwise admit only IPv4/IPv6 TCP/UDP and NETLINK_ROUTE raw/datagram
for interface discovery inside the existing isolated network namespace. Other
families/protocols are not admitted. The three io_uring calls, pidfd_getfd and
ptrace are denied; no trace or userspace-notification exception exists.

Installation is one-shot in the trusted single-thread, no-ML root bootstrap,
after private namespaces/mount/Landlock and before permanent privilege drop.
It sets/verifies no-new-privileges and verifies seccomp filter mode. Failure
requires child exit and owned guardian cleanup; there is no fallback/retry.
Fork/exec inheritance is a kernel property, not a Python flag.

The [kernel seccomp contract](https://docs.kernel.org/userspace-api/seccomp_filter.html)
does not allow inspecting pointed-to sockaddr memory. Hence this policy controls
endpoint creation instead of pretending to validate connect destinations.
The [Linux UNIX socket implementation](https://github.com/torvalds/linux/blob/v6.14/net/unix/af_unix.c)
keeps stream/seqpacket pairs connected; such peers cannot become fresh pathname
clients/listeners. Runtime probes must verify the relevant behavior as well.

## Generated acceptance contract

A new remote-only fixture composes the frozen filesystem/cgroup/netns hierarchy
and the same four cases. It uses only owned generated sentinels, sockets,
temporary directories and tiny CPU tensors. Two same-UID exposed-path UNIX
sentinels provide matching STREAM and SEQPACKET positive controls in each case;
new UNIX creation is then rejected, while permitted private pairs and isolated
Gloo loopback must continue to work. Four installations, eight exposed-path
positive controls and eight owned socket removals are required.

Earlier UNIX negative probes tested connect errors after socket creation. The
new policy rejects earlier at creation with EPERM. These are distinct proofs;
the old accepted fixture remains independently tested and unchanged. Existing
IP, file-access, read-only canary, scratch, privilege and cleanup checks remain
requirements in the new composition. Connected-pair retarget/reconnect/listen
attempts must not reach either exposed sentinel. Addressed sends may succeed
only if the bytes reach the original paired peer; shutdown and peer-death
attempts are also checked, including reconnect after a failed send. Counts and fixed stage/error
codes are public; socket paths, PIDs, canary contents and exceptions are not.

The same owned helper handles the additional sentinel so unrelated live
children cannot confuse frozen reaping. Exact socket identity/cleanup must
complete before an overall cleanup receipt is restored. The accepted three
WorkerRun invocations/six ranks and four global/eight rank optimizer steps
remain unchanged. Unsupported native behavior is a failure, never a skip.

## Explicit nonclaims and next handoff

This is not full IPC or a whole-syscall sandbox. SysV IPC, process signaling,
host metadata, runtime origin and same-UID threats outside the tested boundaries
remain separate concerns. Retained channels must have trusted peers that never
send outside descriptors; SCM_RIGHTS and already-open authority are not revoked
by a socket-creation filter. Inherited io_uring mappings/descriptors must already
be excluded by trusted bootstrap. Trusted host software remains a premise.

Real-host/runtime binding, integrated admission and later real-size relocation
and H100 validation are still required. No GPU instance, real model/dataset,
S3, inference, training, registry, promotion or seal action occurs here. The
Mac performs source inspection/edits, tiny stdlib syscall-double tests, exact
Ruff and AST checks only; Linux fixtures/wrappers are never imported or run there.

## Verification

The filter has 59 deterministic instructions, bytecode SHA-256
`375eaaf211d9740b0f57a552363d161288dc00c226e483364e0fd0263fe98f03`.
Its 31 pure tests pass, including an independent cBPF interpreter, full-width
argument/ABI checks and doubled native installation/one-shot failure paths.
A separate pure interpreter review checked 327680 socket/socketpair argument
combinations against the policy without installing any filter.

The exact 25-suite local regression passes 970 tests with zero failures, errors
or skips in 0.606 seconds, and `ML_MODULES_LOADED=False`. Exact Ruff passes on all
four new Python files. Both Linux files pass AST parsing only; neither was
imported or run on the Mac. All four frozen fixture hashes were verified
unchanged. An independent read-only review checked hook composition, per-case
witness propagation, matching positive controls, helper reaping and socket
cleanup. Exact-commit remote acceptance is pending; these local results do not
prove a live kernel filter or multi-rank compatibility.

### Publication blocked before any remote command started

The exact-branch/CI read-only pre-publication check was rejected by the automatic
approval service because workspace credits were unavailable. The shell command
did not start. This is an approval-service blocker, not a native-test failure.
No GitHub write, new CI run, local commit or staging occurred for this successor.
The five new files remain local and untracked; the tracked tree and HEAD remain
unchanged at `cd9498da1d48c8d7a94436a6a9b7d0292e62e6e4`.

Resume only after the approval service is available: verify the exact remote
branch/active runs, review and stage only this document plus the IPC filter,
pure tests, remote fixture and remote wrapper; publish the scoped commit and
observe both Source and Spark checks on that exact SHA. Do not bypass the
approval rejection, run native fixtures locally, or retry a failed native pass
without an evidence-backed correction. No gate completion is claimed while
the remote acceptance remains unexecuted.

### Approval-service recovery and publication handoff

On the user's next resume request, the approval service accepted the bounded
read-only GitHub check. Local HEAD, tracking ref and live remote branch all
matched `cd9498da1d48c8d7a94436a6a9b7d0292e62e6e4`; the latest Source and Spark
runs for that SHA were both completed successfully, with no active run among
the latest four branch runs. The prior rejection remains recorded above.
Proceed with only the five reviewed successor files. Native acceptance for this
successor remains pending until a new exact-commit CI result is observed.
