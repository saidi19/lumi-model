# Private local CI validation v0.2

Completed locally on 2026-09-06. This extends [v0.1](ci-private-local-validation-v0.1.md)
without altering any historical implementation, test, plan, receipt or marker.
The hosted general-suite scope is unchanged. No test is deleted or newly skipped.

## Genuine original learning-seal unit cases

The new standalone checker ran 12 unchanged cases from
`tests/test_learning_candidate_seal_v3.py`: one plan path/count assertion, all ten
original parameterized manifest rejection cases, and one verification rejection
case. It calls the original pytest fixtures and parameter values, not substitute
fixtures or mocked validators. Result: **12 passed, zero failed**.

This lane reads the genuine private plan contents, not just stat metadata. The
18,454-byte file is SHA-256-bound to
`c1d478db8b8e334e00c899857a92c289b4c418606266f91dac122c381e0041dc`.
It was opened three times: hash-before, one original fixture read, hash-after.
Plan bytes and metadata, the original test and the 84-module implementation
import closure were preserved. No candidate/tokenizer content, other private
content, network, write, subprocess or ML-import boundary was attempted.

The synthetic manifest/audit rejection cases do not authenticate the plan's
entire operational evidence chain, reproduce the historical private dataset,
or execute a seal. The plan's embedded historical authorization is evidence,
not new execution authority. `private_integration_reproduced=false`,
`operational_seal_executed=false`, and `hosted_ci_fixed=false` are explicit.

```sh
artifacts/lumi-sft-import-runtime-v0.1/venv/bin/python -S -B \
  scripts/check_learning_seal_local_ci_v1.py
```

Use the existing trusted local Python 3.13 venv. The checker temporarily loads
pytest from that runtime, removes the site-packages search path before importing
Lumi, and requires no ML stack import. It uses a 30-second alarm and completion
deadline, separate boundary/timeout exceptions that cannot satisfy ValueError
assertions, and one permitted plan open per phase. The Python audit guard is
defense in depth, not OS isolation; Python output is discarded, not native file
descriptors. The original selected code is pinned and does not emit native output.

## Complete unresolved hosted inventory

The latest checked run
[34011375472](https://github.com/saidi19/lumi-model/actions/runs/34011375472)
at `f7f81dcdf50fa6675172a0fb1e28638916282e86` had 306 failures and 21 setup errors.
The new pinned `configs/lumi-ci-unresolved-inventory-v0.1.json` retains all 327
unique nodes across 123 exact-hash test files. Those source bytes were independently
compared with the run's Git commit. Only test identifiers, exception types and
review categories were retained from the failed log, not exception messages.

Every entry remains **OPEN in hosted CI**, including the four earlier local
vocab passes and these twelve local learning passes. The ledger is not a pytest
skip list or execution permission. Its categories are:

| Category | Original hosted outcomes | Meaning |
| --- | ---: | --- |
| Local vocab metadata unit lane | 4 | Local pass only; withheld files remain absent in hosted CI |
| Local learning-plan unit lane | 12 | Local pass only; no operational chain replay |
| Historical source and private integration | 5 | Old source binding plus additional private dependencies |
| Dependency review pending | 306 | Not executed or waived by these two local lanes |

The inventory checker uses descriptor-relative no-follow file opens and rejects
missing, renamed, duplicate, reordered or closed entries, source drift, and
scope/total changes. It runs no original tests and reads no private files.

```sh
artifacts/lumi-sft-import-runtime-v0.1/venv/bin/python -S -B \
  scripts/check_ci_unresolved_inventory_v1.py
```

Observed: `PASS_OPEN_INVENTORY_INTACT`, 327 open outcomes, 123 unchanged files.

## Verification and remaining work

Local source-only collection: **213 tests passed**: the previous 130, local vocab
26, local learning 35, and inventory 22. This is an overlapping targeted collection,
not the full-repository test total. The isolated lint check retained the exact
815 historical findings with zero new findings. Independent review of the new
learning lane and inventory passed; a reported inventory pathname race was fixed
with no-follow descriptor traversal and synthetic race tests before publication.

Only these new guard modules are added to the existing source-only workflow job.
Neither private checker is invoked in hosted CI; private plan/receipt contents
are not uploaded. The general pytest command, two required BF16 history routes,
and seven separate control-suite routes remain unchanged. This snapshot records
local verification before the new source update's hosted execution.

The five historical binding failures cannot be fixed by replacing one old hash:
the Aya test additionally hashes private audit receipts and tokenizer files; all
four safety tests hash a private predecessor candidate. Full historical replay
would need a specifically bound historical-source overlay and reviewed private
input reads. None was attempted here. The other 306 entries need per-node review
before further local execution. No consumed diagnostic or curation pass was retried.

GPU/SFT, model restore, registry and production eligibility are unchanged.
`production_eligible=false` remains in force.
