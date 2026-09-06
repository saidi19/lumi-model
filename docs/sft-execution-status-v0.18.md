# SFT execution status v0.18

Local engineering checkpoint, 2026-09-06. The fixed inert Linux kernel guard is
now composed with the authenticated durable epoch ledger. This is source-level
integration for fixed synthetic workers, not permission to execute real SFT.
Gate 5 remains partial; target-host containment and real-worker admission remain
separate prerequisites. Historical code, tests and evidence stay unchanged.

## Previous step verified

Kernel-guard commit `5b8b886ea26d0477880cd241de3d327cff4ab436` was published.
[Source checks](https://github.com/saidi19/lumi-model/actions/runs/34038489570)
passed all four jobs: 2,983 tests passed, 58 existing skips, 329 separately routed
or deselected nodes and 4,066 subtests. Source selection accounted for 3,041 nodes;
327 private nodes remained NOT EXECUTED. The fixed Linux kernel-child platform
cases executed rather than skipping. [Spark checks](https://github.com/saidi19/lumi-model/actions/runs/34038489576)
passed all three exact synthetic cases, with no skips. Those results do not
establish target GPU-host readiness or full private integration.

## New composition

Five new files only: guarded-ledger parent, fixed worker adapter, their two test
modules, and this status document. No workflow, inventory, historical module,
training-data, checkpoint, attempt marker or cloud-instance mutation is needed.

The parent authenticates four public source components once before creating the
exclusive attempt journal. Each fresh child compiles those exact bytes in separate
namespaces. The three frozen components load as libraries, not entrypoints; only
the new adapter executes. No source path is reopened for worker execution, and no
caller-supplied executable, profile or additional syscall permission is accepted.

| Bound component | Bytes | SHA-256 |
| --- | ---: | --- |
| Frozen kernel policy | 10,521 | `33bf272bd1b97f3ce74a9ecec528c88adce972dd8f31873317f3fe912f2c8c85` |
| Frozen guard challenges | 9,092 | `3fe2376fd09ce7f30fdd141f90a8d2e973285375d6e89a762fb50610f6a0855b` |
| Frozen ledger worker | 10,555 | `c5b3e6e58afaa6a385dc14460e2466873b83792f6ec88589222e29376b020e9e` |
| New guarded adapter | 3,932 | `628841b0fa3f54ec4487c0612dccdd83a2241f205513b85da0569f84a35569aa` |

All imports, ctypes preparation, inherited-socket checks and channel construction
finish before the child installs the unchanged kernel policy. The exact typed
installation/readback result and all 17 denial, one permitted anonymous mapping,
and one address-space-rejection challenges must pass before any READY or token
receive. The adapter then calls the frozen ledger conversation. Its only wire
change translates the first exact old READY into `guarded-ledger-ready-v1`.
Later frames, canonical schemas, credentials, message limits and faults are
unchanged. It does not call either frozen worker's main entrypoint.

Before releasing an epoch/rank token, the parent requires that guarded READY,
per-packet credentials, live owned-pidfd evidence, the existing capability and
identity policy, and a filter-mode/count snapshot. Every later authenticated
packet uses the same guarded snapshot reader. Snapshot metadata alone does not
identify this policy; the claim depends jointly on trusted hash-bound fixed
source, successful installation/challenges and the owned-child protocol. This
is not hostile-code attestation.

The ledger still predebits and fsyncs before an ACK, requires its exact digest
confirmation, and never refunds a failed transaction. A rejected guard snapshot
during ACK confirmation keeps its fixed error code rather than being relabeled
as a generic transport error. An epoch commits only after both children have
successfully reaped, strict EOF is checked, and socket/pidfd cleanup succeeds.
Cleanup errors, final deadline overruns and journal-close failures deny PASS,
even if a terminal success record already exists. Failed closes are not retried.

## Unchanged limits and expected success counts

The shared parent caps remain 64 receive-capacity reservations, 64 status-read
reservations and 131,072 send bytes across all children and epochs. Each status
reservation prepays 16,385 bytes; each receive includes the frozen transport's
payload and ancillary capacity. A fresh reader cannot reset the shared budget.
Four source reads are bounded by 81,924 bytes including overflow detection.
The exclusive journal capacity remains 65,536 bytes, predebited once. These are
application capacities, not measurements of all interpreter, kernel or host I/O.

The original three epochs, two ranks, four synthetic optimizer steps and 18 ACK
confirmations are unchanged. Complete success requires six fresh guarded workers,
54 authenticated packets, six terminal reads, 55 status reads (54 guarded child
snapshots plus one controller), 102 denial challenges, six permitted mappings and
six address-space rejections. The journal has 22 committed records; the six
checkpoint-byte reservations and all memory observations are synthetic, not real
checkpoint reads or host/GPU measurements.

The parent and each fixed child retain a 20-second absolute deadline; failure
retirement uses one shared three-second grace. The child retains 32-message
budgets and three-second per-I/O timeouts. Blocking OS/filesystem calls are not
wall-clock bounded. No started attempt is retried. Each child retains the exact
128 MiB address-space, two CPU-second, zero core/file-size, 64-descriptor and
one-process/thread limits described in [v0.17](sft-execution-status-v0.17.md).
These per-process controls are not aggregate cgroup/descendant memory controls.
The policy has not been widened for JSON, hashing, ML, data access or GPU work.

## Verification checkpoint

Local verification results are recorded below before publication. Linux execution
must additionally pass the existing source workflow after this commit is pushed;
this document does not preclaim that result. On Mac, platform cases must reject
before source reads, journal creation or worker spawn. On supported Linux they
must execute the combined inert positive/negative fixtures, not silently skip.
No test installs a policy or changes resource limits in the parent/test runner.

| Check | Result |
| --- | --- |
| New guarded-worker synthetic cases | 27 passed |
| New guarded-ledger cases | 40 passed |
| Both new modules under ordinary pytest | 67 passed; 124 subtests passed |
| New modules plus frozen kernel, ledger, credential, lifecycle, status and IPC suites | 356 passed |
| Frozen safety/control suites | 273 passed; no skips |
| Exact Ruff on all four new Python files | Zero findings |

Targeted coverage totals 629 unique tests including the frozen controls; running
the same cases under both unittest and pytest does not add unique coverage. New
regressions verify token withholding, four-source binding, independent namespaces,
unchanged transaction-body ordering, prepaid aggregate limits, fsync-before-ACK,
guard failure during confirmation, exclusive attempts, cleanup-before-failure
journaling, non-retried closes and late-success rejection. The ten platform cases
cover success and all nine fixed negative faults under the combined policy.

The 273 frozen safety/control cases pass with no skips. Frozen lint retains
exactly 815 historical findings across 163 files, with zero new findings. Private
readiness remains 327 OPEN with zero private tests executed; its expected blocked
result is not silently converted into full/private CI success.

## Remaining gates

Next is target-host containment design and verification: aggregate cgroup memory,
descendant lifecycle/cleanup, delegation and network boundaries. A separately
reviewed real-worker profile is required; this deliberately inert profile cannot
be silently turned into a GPU training profile. Authentic S3/DCP import,
private-data relocation, CUDA acceptance and the complete execution ticket also
remain required before real SFT. The busy Lambda instance and the S3-backed
Lumi-4B pretrained base remain untouched.

Roadmap: gates 1-4 complete, Gate 5 partial, gates 6-8 pending. The 327 private
OPEN nodes are not a count of roadmap steps. No real training, inference, GPU,
model restore, remote instance, promotion, seal or registry action occurred.
Source publication and CI are separate from the inert probe's zero-external-
network contract. `production_eligible=false`, `execution_authorized=false`,
`guarded_real_worker_ready=false`, and `full_private_integration_passed=false`.

- Inventory SHA-256: `441390c9c98cd4b83d224055a5af5ae421e5ef01b2e4b86101e7b17693d87665`.
- Partition SHA-256: `4e595b654880668a4462c7b8db7c40eaf715ebd86b90eb14cdf852720bc831ae`.
