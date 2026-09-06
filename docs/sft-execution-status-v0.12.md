# SFT execution status v0.12

Local CI recovery validation completed on 2026-09-06. The eight-test copy/replay
blocker recorded in [v0.11](sft-execution-status-v0.11.md) is now resolved. This
snapshot precedes remote execution of the prepared source update; it is not a
claim that full Linux CI, private integration or GPU readiness has passed.

## Completed checks

| Check | Observed result |
| --- | --- |
| Remaining synthetic filter-binding module | Exact copy hash matched; all 8 tests passed in the source clone |
| New modules plus current data-contract suite | 162 tests passed together under ordinary pytest in the source clone |
| New BF16 workflow-route guards | 10 tests passed locally and as part of the clone's source-only job |
| Complete new stdlib CI job | 130 tests passed locally and in the source clone |
| General pytest importer selection | 29 passed, exactly 2 routed out to mandatory history execution |
| History execution | Earlier clone replay ran all 31 cases: 29 passed and exactly 2 required old-v1 errors; all 17 v2 cases passed |
| Existing CI guards | 43 passed |
| Frozen lint integrity | Zero new findings; 815 historical findings retained unchanged |

The 162-test run predates the ten routing tests; these are targeted, overlapping
collections, not a new total for the whole repository. The source clone contains
tracked repository files plus explicit reviewed CI/test overlays, not intentionally
private plans, model weights or candidate data. It uses the existing local runtime,
so clone success is not a substitute for a Linux GitHub run.

The workflow now runs the old BF16 bug reproduction and current v2 acceptance
as a required step. It removes only those two duplicate success-expectation nodes
from ordinary pytest; the tests still execute in the history step. A history
failure remains blocking, while `always()` lets general pytest report too.
See [CI policy v0.2](ci-preservation-policy-v0.2.md) for the exact boundary.

## Still unresolved

The 322 missing-input-dependent outcomes and five actual historical source-hash
rejections from the last full Linux diagnostic are not waived. The four-node
private-fixture plugin is not enabled in GitHub. Historical filter hashes are
not refreshed, private fixtures are not uploaded, and old evidence is not replayed.
Full CI remains unresolved until a subsequent complete Linux run proves otherwise.

The source update is scoped to CI scripts, synthetic tests, a four-node metadata
manifest and status/policy documentation on the existing private branch
`codex/lumi-posttraining-product-v0.1`. No merge into main, source-registry change,
curation retry, real model restore, GPU instance action or SFT training is included.

The original roadmap still has four of eight gates fully closed. Its six
pre-GPU conditions remain partly open. The S3-backed 80B pretrained model is
unchanged, and `production_eligible=false` remains in force.
