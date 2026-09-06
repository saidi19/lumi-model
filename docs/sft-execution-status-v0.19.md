# SFT execution status v0.19

Local engineering checkpoint, 2026-09-06. A bounded, read-only host-containment
observer is implemented. It gathers cgroup and namespace prerequisites together,
but it does not install containment or create an admission receipt. No GPU is
needed for this engineering step; no new instance has been requested or changed.
The existing busy instance remains out of scope. Gate 5 remains partial.

## Previous step verified

Guarded-ledger commit `65cb640a807cca70813a716b79001050ed526667` was pushed.
[Source CI](https://github.com/saidi19/lumi-model/actions/runs/34039791084) passed
all four jobs: 3,050 tests passed, 58 existing skips, 329 separately routed or
deselected nodes and 4,190 subtests. Source selection accounted for 3,108 nodes;
327 private nodes remained NOT EXECUTED. The combined fixed Linux worker cases
executed successfully. [Spark CI](https://github.com/saidi19/lumi-model/actions/runs/34039791060)
passed all three synthetic tests with zero skips. These are source results, not
full private integration or target-GPU-host acceptance.

## Actual gap and scope

The existing GPU smoke launch contract still has no bound target/resource/
authorization/restore/relocation/CUDA receipt. The next-run budget still has null
instance, GPU UUID, runtime lock, provider termination adapter and execution
ticket fields. An existing host cannot safely be inferred from these documents.
Neither public config is changed by this successor. No owner approval, host
identity, delegated subtree or namespace ownership is fabricated.

Six new files only: pure snapshot parsers, self-observer, explicit CLI, their two
test modules and this checkpoint. All historical guards, ledger modules, tests,
workflow definitions, inventories and evidence remain unchanged. The observer
does not import model/data libraries or connect to a host, cloud service or GPU.

## Read-only collection

The observer has no caller-selectable path, PID, executable or destination.
Supported observation uses nonroot Linux x86-64 LP64 CPython 3.12/3.13 with
`-S -B`, one main Python thread, and the frozen read-only ABI check. It never
calls the frozen policy installer. On this Mac it rejects before kernel reads.

It opens its own proc directory, verifies genuine procfs, and pins mount, cgroup
and network namespace descriptors. Explicit namespace magic-link opens are
checked for nsfs and namespace type. Other path components use directory-relative
NOFOLLOW opens. The cgroup2 mount must map `/` at `/sys/fs/cgroup`, with no stacked
target or child mounts; each cgroup directory/file is checked against the
filesystem magic and mount device. Other mappings fail closed, not heuristically
rewritten. The typed filesystem ABI follows the fixed Linux LP64
[libc layout](https://raw.githubusercontent.com/bminor/glibc/glibc-2.39/sysdeps/unix/sysv/linux/bits/statfs.h),
and namespace type checks use the kernel's
[nsfs interface](https://raw.githubusercontent.com/torvalds/linux/v6.8/include/uapi/linux/nsfs.h).

Collected observations include current controller availability/enabling,
memory/high/current/swap/OOM-group and pids limits/counters, cgroup type/events,
and whether the `cgroup.kill` interface exists. That write-only interface is
only statted, never opened or exercised. Network-device metadata yields only
interface counts and loopback presence, with no interface names or traffic
counters retained. No process list, model, dataset, credentials or host identity
file is read.

The observer rechecks its membership and relevant mount mapping, reopens the
cgroup directory to compare its pinned identity, and rechecks all three namespace
identities before returning observations. These checks establish matching
endpoints, not an atomic or continuously stable system view. Raw paths, namespace
identifiers, mount records, controller/event labels and exception text never
appear in the report. Any failed check discards the partial observations.

Kernel pseudo-files may have zero reported size. Reads therefore continue to EOF
inside a prepaid capacity, rather than trusting st_size or one short read.
An existing empty controller-list file is valid: the
[kernel writer](https://raw.githubusercontent.com/torvalds/linux/v6.8/kernel/cgroup/cgroup.c)
can emit no bytes for an empty mask. That narrow case is distinct from ENOENT;
it does not permit empty scalar/event/network records. Only actual ENOENT marks
an optional interface absent. Permission, symlink and other I/O errors reject.
Conservative ASCII/shape limits reject unusual names rather than skipping rows.

## Why observed does not mean contained

A finite `memory.max`, writable mount option, available controller or namespace
inode does not establish a dedicated delegated subtree. Limits apply
hierarchically, migration and delegation require separate authority, and memory
limits are not an absolute instantaneous RSS guarantee. A visible kill file
does not prove that owned descendants can be retired safely. These distinctions
follow the [cgroup v2 contract](https://docs.kernel.org/admin-guide/cgroup-v2.html).

A network namespace has its own network resources, but may still have external
connectivity. Interface counts alone do not prove namespace ownership, routing,
capability restrictions or safe inherited descriptors; see
[network namespaces](https://man7.org/linux/man-pages/man7/network_namespaces.7.html).
No namespace change, socket creation, connection, write-permission test,
controller enablement, limit write, migration or kill is attempted.

Every report therefore leaves target binding, delegation, dedicated ownership,
aggregate enforcement, descendant cleanup, network isolation, real-worker
profile, provider termination and all training/production permissions false.
The existing launcher/bootstrap does not accept this observation schema as a
receipt. Genuine enforcement and approved real-worker admission remain required.

## Bounds and invocation

One invocation permits at most 128 opens, 32 file-capacity reservations, 128 read
calls and 524,288 prepaid metadata-read bytes. All capacity is cumulative with
no refunds; missing files still consume their attempted reservations. The normal
full path reserves 331,792 read bytes. The five-second deadline includes cleanup,
but cannot bound a blocking kernel call or interpreter startup. Cleanup attempts
every owned descriptor once; failed closes are never retried and deny completion.

The explicit entrypoint is:

```sh
python -S -B scripts/observe_sft_host_containment_v1.py --self-only
```

It returns a raw-text-free JSON observation using at most one 8,192-byte stdout
write. Invalid arguments/startup, module-load errors and output failures cannot
produce a success exit. Zero means only that observation completed; it is not
host readiness or execution approval. No partial-write retry or host operation
is scheduled. Source publication/CI and documentation research are separate from
the observer's zero-external-network contract.

## Verification checkpoint

| Check | Result |
| --- | --- |
| Pure bounded snapshot parsers | 33 tests passed |
| Reader, observation, CLI and platform cases | 46 tests passed |
| Both new modules under ordinary pytest | 79 passed; 1,767 subtests passed |
| New modules plus prior guard/ledger/credential/lifecycle/status/IPC suites | 435 passed |
| Frozen safety/control suites | 273 passed; no skips |
| Exact Ruff on all five new Python files | Zero findings |
| Frozen lint | 815 historical findings/163 files unchanged; zero new |
| Private readiness | 327 OPEN; zero private tests executed |

There are 708 unique targeted tests including frozen controls; repeated test
environments do not add unique cases. The local Mac CLI returns the expected
unsupported result with zero kernel reads/opens/writes. Linux CI must execute an
actual self-only observation, not silently skip it; that CI machine is not the
target SFT host. This is the local pre-publication checkpoint, not a preclaim of
the new commit's Linux CI result. Its exact CI result must be verified separately.

## Next boundary

Keep GPU provisioning off for now. Once the bounded preparation and host lifecycle
workflow are ready, bind a new dedicated host/boot and exact subtree/namespace
ownership before any install, migration or kill operation. Actual aggregate
resource/descendant/network enforcement and a separately reviewed real-worker
profile are still needed. Authentic S3/DCP import, private-data relocation, CUDA
acceptance and the complete execution ticket also remain open. The S3-backed
pretrained Lumi base is unchanged.

Roadmap: gates 1-4 complete, Gate 5 partial, gates 6-8 pending. The 327 private
OPEN nodes are not roadmap steps. No SFT, GPU, model restore, private input,
registry, promotion, seal, remote login or instance action occurred.

- Inventory SHA-256: `441390c9c98cd4b83d224055a5af5ae421e5ef01b2e4b86101e7b17693d87665`.
- Partition SHA-256: `4e595b654880668a4462c7b8db7c40eaf715ebd86b90eb14cdf852720bc831ae`.
