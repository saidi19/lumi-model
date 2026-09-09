# SFT execution status v0.63

Bounded source-preflight correction, 2026-09-09. Parent/publication to preserve:
`01b89f9477dc28a6f2b771f164f347dafb53d6dc`, which published exactly the16-file
v0.62 implementation as one non-force update. Its tree was
`417aa1aa02f82a52436a5b9f0f11f8de9f21437b`; local/GitHub object identities matched.
No failed run, historical source, evidence, marker or artifact is deleted.

## Exact stop, before software acquisition or a native pass

[Guardian workflow34358096760](https://github.com/saidi19/lumi-model/actions/runs/34358096760),
job`102487798348`, stopped in its source/synthetic preflight. Checkout/setup
passed;292 tests ran, with exactly one failure:
`test_global_pax_metadata_is_supported_and_counted` observed header count1
instead of the required2. The artifact acquisition, native composition and input
retirement steps were all **skipped**. No accepted image was downloaded, loaded,
created, exported or executed by that workflow. No workload cgroup or native
case was started. This is not a started-payload retry.

The failed log was22,043bytes with SHA-256
`9265f52397ac50d91d12c89fe303a8f90f9ccaec8a303808f23cea1b3fb63607`.
Only test identifiers/aggregate failure details were retained. A separate
32KiB-capped targeted reread established the exact runtime:CPython3.12.14.
No raw logs were saved. The initial local suite used3.13.7; that cross-version
gap was not caught before publication and is corrected here.

The full [Source run34358096685](https://github.com/saidi19/lumi-model/actions/runs/34358096685)
also finished with that same single failure:5,415 passed,1 failed,58 skipped,
329 deselected,36 warnings and10,476 subtests in431.91s. Its three other jobs
(lint, isolated controls and source validators) passed. Partition accounting
remained valid:5,474 selected,327 private integrations NOT_EXECUTED and2 history
tests separately routed; full/private-integration success remains false.
[Spark34358096640](https://github.com/saidi19/lumi-model/actions/runs/34358096640)
passed all3 tests in99.15s. Old OCI build34358096666 was skipped.

Each completed Source/Spark log was read once under2MiB, retaining aggregates
only: Source64,031bytes, SHA-256
`cc8b276deb9f46a7c561b73b7dbe317a31c5847261a114bdbb874441e5024df0`;
Spark51,413bytes, SHA-256
`f8be5cb57839fe075487867e8deba776a1710ef0bba9b3d5f2d8db9a6b6d9f96`.

## Real cause and narrow code correction

The [CPython3.12.14 tarfile implementation](https://raw.githubusercontent.com/python/cpython/v3.12.14/Lib/tarfile.py)
routes PAX/GNU following headers through a new private `_fromtarfile` entry
point. The extractor only intercepted the public `fromtarfile`, so those
physical headers bypassed its counter/bounds. This was a code defect, not
permission to reduce the expected count or weaken its test.

The same unmodified59-test rootfs suite reproduced the exact1!=2 failure on the
already-installed MacCPython3.14.6. A15-line source diff makes public and private
entry points share the bounded reader. When present, the updated `_frombuf`
receives the original `dircheck` flag, preserving the new stdlib's V7-directory
security semantics. There is no stdlib replacement/copy, global monkeypatch,
test skip or change to expected count2. Ten new rootfs regression tests cover
both dispatch paths, nested metadata, budgets, deadlines, truncated/checksum
failures and directory semantics.

The complete69 rootfs tests now pass on3.11.15,3.13.7 and3.14.6. The full combined
source suite also passes on all three: **387 tests, zero failures, zero skips**,
including70 existing admission/transport/source tests. Timings:0.396s on3.11,
0.376s on3.13 and0.368s on3.14. The exact Ruff policy passes. These are local
stdlib/synthetic tests, not native Linux or ML execution.

## Conditional first-native handoff

The existing new workflow and preparer are rebound only to the direct successor
of`01b89f9477dc28a6f2b771f164f347dafb53d6dc`, still private/first-attempt/exact
checkout with no manual rerun. Before artifact metadata/download, the preparer
makes one additional authenticated read of run`34358096760`, attempt1,
job`102487798348`. It requires that exact SHA/job/source failure and that all
three downstream acquisition/native/retirement stages were skipped. Wrong,
missing, duplicate, started or completed downstream stages fail closed.

Acquisition is therefore now **four HTTPS GET attempts**, not the three in
historical v0.62: prior-stop evidence, artifact metadata, non-followed archive
redirect, then the signed storage request without credentials. Each metadata
body is capped32KiB. The same ZIP/tar pins, streaming byte caps, no-proxy/no-retry
rules,300s logical/6m outer acquisition boundary and all native limits remain.
No other native case, parser policy, image, mount, Landlock, worker adapter,
source identity, filtering or acceptance rule is changed.

Only the rootfs module/tests, narrowly consequent preparer/workflow/tests and
this status record change. The v0.62 record and all three older native workflows
stay unchanged. The current local v0.61 receipt remains unstaged.

This correction does not claim native success or Gate5 completion. The first
actual bounded native composition remains pending; dedicated H100/S3/data
binding and real SFT remain downstream and unauthorized by synthetic success.
