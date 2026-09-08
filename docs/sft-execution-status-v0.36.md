# SFT execution status v0.36

Single-host direct-IP network namespace successor, 2026-09-08. This step adds
loopback-only network isolation to the accepted cgroup-contained generated
WorkerRun. Filesystem isolation remains a separate next step. Real data/model
access, GPU use, training admission and production eligibility remain closed.

## Accepted predecessor and exact scope

Predecessor `0d728626fbee06c74b296c3bcd6c8058e67567c2` passed all four jobs in
[Source CI](https://github.com/saidi19/lumi-model/actions/runs/34200734317):
4421 passed, 58 skipped, 329 deselected, 36 warnings and 8458 subtests in
328.95 seconds. Source selection was 4479 tests; 327 private integrations were
NOT_EXECUTED and two historical failures were routed separately. Private/full
CI claims remained false. [Spark CI](https://github.com/saidi19/lumi-model/actions/runs/34200734158)
passed three tests with zero skips in 105.84 seconds.

Only five new files belong to this successor:

- `src/lumi/posttraining_sft_netns_v1.py`
- `tests/test_sft_netns_v1.py`
- `scripts/run_sft_netns_fixture_v1.py`
- `tests/test_sft_netns_linux_v1.py`
- This document.

All prior source, evidence, budgets and workflows remain unchanged. Unrelated
local files are not included. The frozen cgroup fixture is bound to SHA-256
`f91ed0c353a20c309ff3e1bfaab8dbe990b28c36626fdeeb0ac5131e0cf75fad`;
its 27839 bytes are not rewritten. The generated WorkerRun remains bound to
`53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0`.

## Narrow network contract

A root, single-threaded, no-site/no-bytecode bootstrap first attaches to its
owned workload cgroup, then creates one fresh network namespace. Namespace
descriptors are checked against genuine procfs/nsfs and namespace type, and the
old namespace remains held while the new identity is compared. A bounded,
authenticated self-procfs check requires the kernel's thread count to be one,
in addition to Python's thread guard. Only loopback
is brought up, with actual flag and interface readback. No host route, device,
firewall, sysctl, controller or named namespace is changed. Setup is one-shot;
a failed or consumed attempt cannot be retried or used to rejoin the host.

The bootstrap then permanently drops UID/GID/groups and current capabilities
using the accepted guardian contract. All owned namespace/control descriptors
are closed. Inherited descriptors, including standard descriptors, are replaced
or closed before unprivileged work; only the exact private UNIX control channel
is retained until its bounded report/exec use. The generated negative test
deliberately supplies host socket authority through standard descriptors before
the bootstrap replaces them with authenticated `/dev/null` descriptors.
The guardian stays outside the
workload cgroup and network namespace, retaining its exact-target cleanup role.

The kernel's [network namespace semantics](https://man7.org/linux/man-pages/man7/network_namespaces.7.html)
separate IP devices/stacks, ports, routing and abstract UNIX socket names.
This does not isolate UNIX pathname sockets in a shared filesystem or prevent
all host-mediated proxy channels. The [interface ioctl contract](https://man7.org/linux/man-pages/man7/netdevice.7.html)
requires privilege for flag changes; loopback readback alone is not proof of
permanent privilege removal. This successor therefore claims direct-IP and
abstract-socket namespace behavior only, not a complete network sandbox.

## Verification boundaries

Local verification is limited to tiny stdlib syscall-double tests, exact Ruff
and AST parsing. The Linux-only wrapper and fixture are never run on the Mac,
including their skip paths. No local ML library/model/data operation is needed.
The 76 new syscall-double tests passed in 0.018 seconds. The exact combined
855-test regression passed with zero failures/errors/skips in 0.549 seconds;
the fresh no-site/no-bytecode process reported `ML_MODULES_LOADED=False`.
Exact isolated Ruff passes for all four new Python files; the two Linux-only
files also pass AST parsing without execution. Remote acceptance must be
observed on the published commit; these local results are not Linux or Torch
acceptance.

The new remote fixture composes the hash-bound cgroup harness in a private
fixture module without changing its on-disk code. Four fresh owned leaves cover
network probes, deadline cleanup, controller-death/escaped-descendant cleanup,
and the pinned generated WorkerRun with its three invocations/six fresh ranks.
Existing resource limits, independent guardian deadline, exact-once kill,
empty/reap/removal checks and timeout nonclaims are retained.

The outer fixture owns actively checked host-only sentinels. Negative probes
run only after fresh namespace setup and privilege drop; external route probes
use numeric documentation addresses with UDP connect only, without any send or
DNS operation. Missing IPv6 or required kernel facilities fail rather than skip.
Positive loopback traffic and generated Gloo resume checks must still work.
The outer namespace must remain unchanged. Reports contain bounded fixed codes,
counts and booleans, not PIDs, paths, source data or third-party exception text.

Remote-only execution uses four new cgroups, the existing 260-second root bound
and 300-second wrapper bound. No named namespace is persisted. Privileged
processes never import ML. The public entrypoint and real-host admission remain
closed even if every generated acceptance case passes.

## Remaining work

Gate 5 is still partial: private filesystem read/write enforcement, inherited
host filesystem/IPC authority, admission/host binding and integrated fault
checks remain. This design supports single-host loopback communication, not
multi-host networking. GPU memory, billing and a complete hostile-process
sandbox are not established. Real-size source import/relocation, runtime
handoff and actual H100 validation remain separate steps. Private research
readiness does not substitute for production rights/owner approval.
