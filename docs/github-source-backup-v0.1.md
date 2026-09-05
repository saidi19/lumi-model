# Reviewed private GitHub source backup v0.1

Prepared on 2026-09-05 for the owner-requested source sync to the existing private
`saidi19/lumi-model` repository, branch `codex/lumi-posttraining-product-v0.1`.
The remote branch and local HEAD both resolved to
`38499dce819bf0afb168448f2289cff0b8b047ad` before this sync. This change does not
merge into `main`, launch an instance, restore a model or execute SFT.

## Included

- Current Python implementations, runners and tests, including the versioned SFT
  admission, launch contract, source-bundle, watchdog and parent-ledger work.
- Local UI renderer and vendored KaTeX assets, including its MIT license.
- Selected non-secret configuration and dependency locks, the final pretrained
  base binding, architecture/serving updates and execution-status documents.
- Ten deliberately selected aggregate engineering reports: runtime controls,
  launch-contract development, import-runtime tests, distributed CPU preflight
  and GPU smoke readiness, each with its verification report.

Existing reviewed file bytes are retained. Historical reports are historical
evidence, not new execution authority. S3 object locations and content hashes in
the included private configuration are metadata, not the model weights or AWS
credentials. Keep this repository private; public release needs a separate
privacy, provenance and rights review.

## Intentionally local-only

Raw datasets, candidate rows, encoded shards, model/checkpoint weights, runtime
environments, credentials, populated owner decisions/reviews, human/rights-review
packets, standalone owner-approval/delegation documents, private attempt artifacts
and consumed started markers are not added by this sync. Some source files
contain contractual approval language or synthetic example prompts; this is not
the private approval packet or permission to run those programs. Machine-specific
launchd jobs and the desktop job-resume shortcut also stay local. No excluded
file is deleted or changed. Some configuration and evidence is conservatively
withheld even when it may ultimately be safe to archive.

The repository already contains a few tracked synthetic data fixtures from its
earlier history; this change adds no dataset, archive or training-weight files.
Existing tracked files are not removed or history-rewritten by this backup.

## What a clone can and cannot reproduce

This is a source backup, not a deployable or self-contained execution package.
Some historical tests, scripts and documentation reference intentionally
untracked private plans, approvals, evidence and artifacts. Their missing-input
fail-closed behavior must remain; do not fabricate receipts or relax gates to
make a fresh clone run. The private source-review ZIP is not included either.

The real S3-backed base restore/import, integrated resource-enforcing GPU worker,
target-bound data relocation and CUDA smoke execution remain unfinished. See
[execution status v0.8](sft-execution-status-v0.8.md) for the precise engineering
state. `production_eligible=false` remains unchanged. The pretrained model's S3
backup is separate from this GitHub source backup.

## Pre-push verification

The selection was explicit, not `git add -A`. Selected text was checked for
common credential/private-key/token/signed-URL patterns; this heuristic scan is
not a complete security audit. Selected Python files passed syntax parsing;
JSON configuration was structurally parsed, and populated private approval/text
fields were excluded. Unexpected file types, links and oversized files were
rejected. Vendor fonts were admitted only from the explicit KaTeX font path.

The local non-publishing runtime-control recheck passed 106 tests and preserved
54 historical bindings. The prior source-review recheck passed 167 tests and
reproduced its original ZIP digest. No reports or started markers were
republished. These are targeted synthetic checks, not a claim that the entire
repository's lint, CI, private-artifact integration or GPU suite passes. The
existing GitHub CI workflow is not disabled or weakened by this backup.
