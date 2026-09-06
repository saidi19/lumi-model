# Source checks and the unresolved private integration gate

This change corrects CI routing, not the private integration results. The source
backup is not a self-contained private replay environment. Missing private
prerequisites must neither be presented as ordinary source regressions nor
silently counted as passing tests.

## Exact scope

The original unresolved inventory remains byte-for-byte unchanged: 327 OPEN
nodes across 123 historical test files. Three new source reviews cover the exact
nodes (117 + 71 + 139). Each witness identifies a necessary private plan, report,
artifact, or metadata dependency through its test, fixture, and loader source.
The review is not an authorization to replay those operations. Some tests have
synthetic assertions but still load an exact local plan before reaching them.

`configs/lumi-ci-source-partition-v0.1.json` is a separate reviewed routing
manifest, not a reinterpretation of the inventory's forbidden skip-list use.
It preserves every node as OPEN. The manifest and reviewed source/support files
are hash-bound; the historical tests cannot be refreshed to new hashes. The
five previously demonstrated historical-source failures retain their separate
classification (one Aya repair node and four safety-URI nodes). Their private
dependencies and old policy-loader bindings both still need resolution. Other
private plans can expose additional historical binding failures once supplied.

## Required CI lanes

- Existing isolated controls remain mandatory, with exactly seven routed test
  modules. Their subprocess and no-site-package restrictions are unchanged.
- The two original BF16 nodes still run in the mandatory historical-reproduction
  and current-successor step. No checkpoint code or old hash was changed.
- The source suite opts into exact-node routing. It executes every collected
  node except the reviewed private nodes and those two separately run BF16 nodes.
  New/unclassified nodes run normally. Missing, duplicate, renamed, stale, or
  unexpectedly filtered nodes fail the routing check. No exception becomes a
  pass, skip, or xfail. Existing unrelated skips remain pytest's own results.
- A separate unconditional `private-integration-readiness` job validates the
  manifest and returns `BLOCKED_PRIVATE_INTEGRATION_UNAVAILABLE`, exit code 1.
  It reports zero private tests executed and `full_ci_passed=false`. Neither
  source success nor the earlier 16 bounded local unit passes closes this gate.

The overall workflow is therefore intentionally **not merge-ready** while this
gate is OPEN. Making it green requires actual, separately bounded and verified
private integration work, not disabling the gate or uploading private inputs.

## Read and preservation boundary

The router reads only the two pinned CI manifests and reviewed source/support
bindings. Required private paths are descriptive metadata: it never opens or
stats them. Reads are bounded and reject symlinks, hardlinks, nonregular files,
and concurrent file changes. The final partition checks authenticate source
witness anchors; they do not claim to infer a complete semantic call graph.
Dependency classification came from source inspection, not failure messages.

No historical parser, policy, rights rule, owner-review rule, provenance check,
frozen lint baseline, candidate, seal, or existing evidence is changed. No GPU,
instance, SFT, promotion, private upload, or merge is part of this correction.

Review scope deviation: one reviewer read only the `.bindings` path/hash metadata
of the untracked Aya materialization plan before checking Git tracking. No raw
candidate/data/evidence payload was read. Its contents are not used as a witness;
the test's mandatory read of that absent plan independently establishes the
dependency. Subsequent config reads were tracking-checked.

## Verification

Validation used a clean local clone containing committed source and only the
selected CI changes, without copying local private inputs:

- 256 stdlib validator/guard tests passed, including 38 new partition guards and
  13 workflow guards. The new 51 guards also passed in the clean clone.
- 273 isolated control tests passed. Frozen lint retained exactly 815 historical
  findings across 163 files with zero new findings.
- The mandatory BF16 history/successor contract passed in the clean clone:
  29 passes and the two exact required historical errors; no real model tested.
- Real manifest/readiness validation found 327 reviewed OPEN nodes, zero
  unreviewed nodes, and zero private tests executed. Its expected exit code was 1.
  The manifest binds 423 source/support files totaling 6,361,527 bytes.
- The broad local source run selected the exact 329 exclusions (327 private plus
  two separately executed BF16 nodes): 2,041 passed, 53 skipped, 48 failed,
  seven errors, and 542 subtests passed. This is **not** a full suite pass.
  The local import-only Python 3.13 runtime lacks the declared data extras;
  its Transformers 5.2 API requires `cache_position`, and the desktop sandbox
  denies loopback server binding. Diagnostic spot checks reproduced those
  causes. These additional source failures were not added to the private route.

The existing GitHub Python 3.12 job installs the complete declared dev/data
dependencies and remains the hosted source-acceptance check. Its result must be
read from the actual workflow run, not inferred from these local results.
Production eligibility and full private integration remain false regardless of
source-suite results.
