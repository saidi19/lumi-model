# SFT execution status v0.39

Private-root directory-walk compatibility and fixed failure projection,
2026-09-09. Gate 5 is still partial; this successor does not authorize real
data/model access, GPU, S3, training, host admission or production eligibility.

## Preserved predecessor result

Corrective commit `5ab44f96b677cc23f70c6f34eff79f7a55716eac` fixed the old-namespace
bind-source defect. Its [Source CI 34307249167](https://github.com/saidi19/lumi-model/actions/runs/34307249167)
advanced through the normal, deadline and parent-loss filesystem cases, then
failed the final WorkerRun case. The first three successes follow from the
unchanged sequential case loop, not from an invented aggregate success report.
The final report retained `cleanup_confirmed=true`, `case=worker_run`,
`stage=cases`, and `component_code=filesystem_not_reported_not_reported`.

Source totals: 1 failed, 4567 passed, 58 skipped, 329 deselected, 36 warnings,
8502 subtests, 362.22 seconds. Selection was 4626; 327 private integrations
remained NOT_EXECUTED and two historical cases remained separately routed.
Its three helper jobs passed. [Spark CI 34307249157](https://github.com/saidi19/lumi-model/actions/runs/34307249157)
passed 3 tests with zero skips in 105.30 seconds. All old commits, run evidence,
and v0.37/v0.38 documents remain unchanged; no unchanged CI run is retried.

## Proven contract conflict, not an invented failure trace

The unchanged `_open_secure_root` in `posttraining_sft_runtime_v4.py` starts its
no-symlink walk by opening the absolute root with `O_RDONLY|O_DIRECTORY`.
Landlock's existing selected-subtree rules do not grant READ_DIR on the new
private root. This is a concrete incompatible contract. The previous outer
failure report did not retain the inner WorkerRun stage, so it does not prove
which inner operation failed first.

New source-AST tests extract only the unchanged path-walk definitions and run
them on synthetic syscall doubles, without importing the ML runtime. The old
policy fails its first ancestor open; the corrected policy permits the same
walk. No loader/parser, data policy, token floor or frozen WorkerRun is changed.

## Exact policy adjustment

Landlock gains one explicit opt-in `allow_private_root_listing` role, default
false. The opt-in consumes one of the existing 16 rule slots. Only a root-owned,
0755, 1 MiB tmpfs root with read-only/nosuid/nodev readback can receive this rule.
An ordinary host filesystem, writable root, wrong mode or capacity fails closed.
The rule grants exactly READ_DIR (mask 8), no READ_FILE, WRITE_FILE, EXECUTE,
create/remove/truncate, device ioctl or other content authority.

READ_DIR inherits through the visible private tree: it permits directory names
to be inspected, not merely opening `/` in isolation. This is an intentional
narrow policy adjustment, not a claim that the policy is byte-for-byte unchanged.
The generated root and its scaffold are not a metadata-confidentiality boundary.
Existing content-read, write, execute, private scratch, device, same-UID ptrace,
namespace and inherited-descriptor rules remain unchanged. The integrated
normal case now requires the secure loader's exact initial directory-open flags
to work after privilege drop. The Landlock receipt explicitly reports the opt-in.

## Diagnostic projection

The new pure module projects only the frozen WorkerRun schema/status and four
allowlisted stage/code fields. Unknown fields, IDs, paths, exception objects,
non-string values and unknown text are discarded. A fixed compound component
code carries these fields through the existing final guardian receipt, before
the frozen verifier rejects failure. This changes reporting, not acceptance.
Missing evidence remains `not_reported`; success is never inferred from it.
The frozen WorkerRun, cgroup and netns fixtures remain byte-for-byte unchanged.

## Local verification and remaining acceptance

Landlock passes 42 tests; the failure projector passes 6 tests. The exact
24-suite stdlib regression passes 939 tests with zero failures/errors/skips in
0.566 seconds and `ML_MODULES_LOADED=False`. Exact Ruff passes for the eight
filesystem Python files. Both Linux-only files were AST-parsed, not imported or
executed on the Mac. The three frozen fixture SHA-256 bindings were rechecked.

Scope is seven files: the Landlock implementation/tests, integrated filesystem
fixture/wrapper, new diagnostic implementation/tests, and this document. No
workflow, existing loader, prior mount correction, budget, resource deadline,
dataset, registry, model or instance is changed. Corrected-commit remote
acceptance is still required; all four cases and the pinned three-invocation,
six-rank WorkerRun must pass together before this filesystem step is complete.

Full pathname UNIX IPC containment, runtime origin, real-host binding/admission,
real-size relocation and actual H100 checks remain separate unfinished work.
Production eligibility and real-SFT admission remain false.
