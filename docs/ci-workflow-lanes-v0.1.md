# Source CI versus full integration readiness

The 2026-09-06 run for `527f97c` passed all four source-check jobs: 2,207 source
tests passed, 58 existing skips remained, and the separate control, validator and
lint-preservation jobs passed. The only failed job was private readiness.

That job checks a frozen OPEN inventory and unconditionally returns exit code 1.
It executes zero private tests. Repeating it on source-backup pushes cannot repair
or verify those 327 private integrations. Mixing these different outcomes into one
push workflow produced a predictable red result after otherwise successful work.

## Corrected routing

- `source-checks.yml`, named **Source checks**, runs on push. It retains all four
  existing source jobs, commands, test selections and BF16 history checks verbatim.
  Its read-only token and per-ref concurrency cancel obsolete source runs.
- `tests.yml`, now named **Full integration readiness**, runs on pull requests or
  explicit manual dispatch. Its five existing jobs remain unchanged, including
  the unconditional private-readiness blocker. A PR therefore still fails full
  readiness while private integration is unavailable.
- Twelve new guards verify triggers, exact matching source jobs, the unchanged
  historical workflow contract, private-blocker retention, permissions and
  cancellation. No old test, source binding, parser, evidence or private object
  was changed. No `continue-on-error`, fabricated pass, new deselection or xfail
  was added.

This supersedes only the automatic push placement described in
`ci-source-partition-v0.1.md`; it does not supersede any private-integration,
rights, owner-review, preservation or production restriction. Existing historical
documents and evidence remain unchanged.

**Green Source checks means source checks passed, not full CI, merge readiness,
SFT permission or production eligibility.** The 327 private/history nodes remain
OPEN and are not represented as executed. No repository branch-protection rules
were changed or newly claimed to exist. No merge, promotion, model restore, GPU
or instance action is part of this correction.

This prevents the known-unavailable full-readiness probe from failing every source
backup. Closing private readiness itself still requires authentic, separately
bounded integration evidence; rerunning its current probe cannot produce that.
