# CI skip accounting v0.1

Read-only audit on 2026-09-06 of source-checks run
[34021787226](https://github.com/saidi19/lumi-model/actions/runs/34021787226),
commit `acb780f73d68547579fbace225dc74d793548ce7`.

The source job reported **2,367 passed, 58 skipped, 329 deselected**, 36 warnings
and 835 passing subtests, with 2,425 source nodes selected. Its explicit summary
retained `private_integration_passed=false` and `full_ci_passed=false`.
The successful source workflow is not full private integration acceptance.

## What the 58 skips mean

The source-backed reconstruction is **55 exact-fixture cases still OPEN plus
3 executable synthetic Spark cases missing their CI dependency**. The 55 are
not waived, passing, or harmless; they still require separately bounded evidence
and, where applicable, authority. The three Spark cases are pending new required
hosted execution. This document records no Spark PASS.

Important evidence limit: that run used quiet pytest output, without per-node
skip-reason reporting. Its log contains aggregate counts and progress glyphs,
not observed skip-reason records. The node attribution below is a reconstruction
from unchanged tracked test source, same-file helper/fixture dependencies,
the exact source partition, tracked-path metadata, installed-package log and
ordered progress glyphs. It must not be presented as a downloaded per-node
result artifact.

Static inspection found 67 selected nodes with possible skip paths, including
fixture consumers and both distributed-strategy parameters. Of these:

| Reconstructed outcome | Nodes |
| --- | ---: |
| Missing exact local/private fixtures, listed below | 55 skipped |
| Missing PySpark dependency, listed below | 3 skipped |
| Available capability branches: zstandard 1, PyArrow 1, Node.js 2, DDP/FSDP 2 | 6 passed |
| Three Glue production-package fixture consumers whose canary mismatch branch was not taken | 3 passed |
| Total skip-capable selected nodes | 67 |

The last three are `test_package_is_non_authorizing_manifested_and_calibrated`,
`test_exact_approval_and_job_definition_are_fail_closed`, and
`test_category_checkpoints_and_final_result_conserve_all_tokens` in
`tests/test_glue_final_corpus_production.py`. Their shared conditional skip is
at line 53. They must not be added to the 58 merely because their source contains
a skip branch. PyArrow and zstandard appear in the actual dependency-install
log; the DDP/FSDP cases also appear by exact node name in the run's warnings.

No original test was collected or executed during this audit. No private
object was opened or statted, and no model, dataset or tokenizer was read.
Only tracked source/control metadata and existing GitHub logs were inspected.

## Three executable source cases: required execution pending

All three use `pytest.importorskip("pyspark")`. The workflow installs
`.[dev,data]` at `.github/workflows/source-checks.yml:75`, but those dependency
groups in `pyproject.toml:22-40` do not include PySpark. It is absent from the
run's install list. The same omission exists in the full-readiness workflow.

| Exact original node | Skip witness | Status |
| --- | --- | --- |
| `tests/test_distributed_corpus.py::test_local_spark_dedup_matches_independent_sequential_reference` | `tests/test_distributed_corpus.py:649` | OPEN: pending required hosted Spark execution |
| `tests/test_final_corpus.py::test_local_spark_hash_selection_matches_python_rank` | `tests/test_final_corpus.py:236` | OPEN: pending required hosted Spark execution |
| `tests/test_glue_production_dedup.py::test_memory_safe_projection_streams_long_text_only_after_selection` | `tests/test_glue_production_dedup.py:481` | OPEN: pending required hosted Spark execution |

These are genuine local CPU/Spark checks, using generated rows and temporary
local files, not missing private-integration inputs. Their sessions use
`local[2]`; the long-payload case generates 8 MiB text and token fields. The
smallest next source-coverage repair is a bounded required Spark/JVM CI lane
that executes these original nodes and rejects skips. Dependency installation
alone, a synthetic substitute, or a locally planned lane is not completion.
Do not change old assertions, refresh source pins, or claim a pass before the
actual hosted result is available.

## Exact inventory of 55 fixture-dependent nodes

Every node in this list has status **OPEN**. Counts by family are: Aya 2;
BigCodeBench 9; code-benchmark preflight 3; DeepMind 6; DMath 3; first-party scale
3; gpt-oss scale 3; LiveCodeBench 5; MiniMax 3; OASST1 1; OpenCodeInstruct 1;
OpenR1 2; production-corpus integration 6; stronger-teacher 8. Total: **55**.

```text
tests/test_aya_bengali_owner_review.py::test_actual_plan_rebuilds_exact_sample
tests/test_aya_bengali_owner_review.py::test_actual_owner_receipt_is_complete_and_nonpromoting
tests/test_bigcodebench_prompt_signatures.py::test_plan_is_prompt_only_and_bounded
tests/test_bigcodebench_prompt_signatures.py::test_materialized_source_replays_exact_signatures_when_present
tests/test_bigcodebench_prompt_signatures.py::test_materialized_report_is_sealed_and_non_production_when_present
tests/test_bigcodebench_signature_promotion_preflight.py::test_preflight_audits_candidate_without_promotion
tests/test_bigcodebench_signature_promotion_preflight.py::test_proposed_successors_are_immutable_and_hash_bound
tests/test_bigcodebench_signature_promotion_preflight.py::test_materialized_outputs_match_reconstruction_when_present
tests/test_bigcodebench_signature_registry_promotion.py::test_promotion_payloads_match_preflight_hashes
tests/test_bigcodebench_signature_registry_promotion.py::test_promotion_does_not_authorize_activation_or_sft
tests/test_bigcodebench_signature_registry_promotion.py::test_materialized_report_is_sealed_when_present
tests/test_code_benchmark_signature_preflight.py::test_preflight_is_metadata_only_and_fail_closed
tests/test_code_benchmark_signature_preflight.py::test_rights_and_acquisition_decisions_are_narrow
tests/test_code_benchmark_signature_preflight.py::test_materialized_outputs_match_reconstruction_when_present
tests/test_deepmind_canary_run.py::test_exact_source_static_audit_excludes_upstream_generation_cli
tests/test_deepmind_dependency_overlay.py::test_receipt_is_sealed_when_materialized
tests/test_deepmind_dependency_resolution.py::test_live_private_inputs_match_contract_when_available
tests/test_deepmind_scale_owner_review.py::test_finalized_owner_receipt_validates_when_present
tests/test_deepmind_source_promotion.py::test_actual_successor_and_receipt_are_nonproduction_when_available
tests/test_deepmind_source_promotion_preflight.py::test_actual_report_is_raw_free_and_nonmutating_when_available
tests/test_dmath_owner_review.py::test_exact_plan_and_sample_are_bound_when_available
tests/test_dmath_owner_review.py::test_actual_owner_receipt_is_complete_and_non_promoting_when_available
tests/test_dmath_source_promotion.py::test_actual_successor_and_receipt_are_nonproduction_when_available
tests/test_first_party_scale_canary_preflight.py::test_preflight_is_no_call_and_nonproduction
tests/test_first_party_scale_canary_preflight.py::test_contract_is_exactly_1000_rows_with_800_200_split
tests/test_first_party_scale_canary_preflight.py::test_report_and_verification_reproduce_when_materialized
tests/test_gpt_oss_scale_canary_preflight_v2.py::test_preflight_is_local_only_and_nonexecuting
tests/test_gpt_oss_scale_canary_preflight_v2.py::test_scale_canary_is_1000_rows_and_tool_inclusive
tests/test_gpt_oss_scale_canary_preflight_v2.py::test_report_and_verification_reproduce
tests/test_livecodebench_research_signatures.py::test_plan_is_private_research_only_and_bounded
tests/test_livecodebench_research_signatures.py::test_materialized_report_is_sealed_when_present
tests/test_livecodebench_signature_registry_promotion.py::test_promotion_payloads_match_preflight_hashes
tests/test_livecodebench_signature_registry_promotion.py::test_promotion_preserves_research_only_boundaries
tests/test_livecodebench_signature_registry_promotion.py::test_materialized_report_is_sealed_when_present
tests/test_minimax_provider_preflight_v1.py::test_plan_authorizes_no_provider_call_or_spend
tests/test_minimax_provider_preflight_v1.py::test_report_reproduces_and_fails_ollama_cloud_closed_when_present
tests/test_minimax_provider_preflight_v1.py::test_verification_is_exact_and_raw_text_free_when_present
tests/test_oasst1_owner_review.py::test_exact_local_plan_and_private_sample_are_bound_when_available
tests/test_opencodeinstruct_owner_review.py::test_exact_plan_and_receipt_validate_when_present
tests/test_openr1_owner_review.py::test_exact_plan_sample_and_receipt_when_materialized
tests/test_openr1_rights.py::test_exact_plan_and_report_when_materialized
tests/test_production_pretraining.py::test_production_plan_binds_exact_tail_cost_and_three_segments
tests/test_production_pretraining.py::test_owner_artifact_is_one_exact_hash_bound_segment
tests/test_production_pretraining.py::test_trainer_accepts_only_exact_production_segment
tests/test_production_pretraining.py::test_production_segment_rejects_the_wrong_resume_checkpoint
tests/test_production_training.py::test_markerless_production_corpus_is_rejected_before_preflight
tests/test_production_training.py::test_corrected_marker_aware_corpus_passes_production_preflight
tests/test_stronger_teacher_ab.py::test_execution_plan_preserves_exact_authorized_bounds
tests/test_stronger_teacher_ab.py::test_prohibited_actions_remain_false
tests/test_stronger_teacher_ab.py::test_continuation_is_exactly_bound_to_the_sealed_boundary
tests/test_stronger_teacher_ab_gpu.py::test_gpu_continuation_preserves_existing_journal_boundary
tests/test_stronger_teacher_ab_gpu_completion.py::test_failure_boundary_is_exact
tests/test_stronger_teacher_ab_preflight.py::test_preflight_performs_no_model_download_or_inference
tests/test_stronger_teacher_ab_preflight.py::test_candidate_object_and_quality_gates_are_exact
tests/test_stronger_teacher_ab_preflight.py::test_report_and_verification_reproduce_when_materialized
```

### Source witnesses for those 55 nodes

Paths and variable names below describe prerequisites found in tracked source;
they are not records of opening or checking the private objects. An absent
prerequisite establishes why the source clone cannot run a case. Supplying it
can expose further exact-binding or runtime requirements; the table is not
permission to replay the whole test or its old publisher.

| Test file and count | Source skip witness | Required metadata/object names |
| --- | --- | --- |
| `test_aya_bengali_owner_review.py` (2) | Lines 91-92 and 107-108: missing `PLAN` / `RECEIPT` | `configs/` and `evidence/lumi-4b-sft-aya-bengali-owner-review-v0.1.json` |
| `test_bigcodebench_prompt_signatures.py` (3) | Lines 22-23; later 40-41 and 55-56 | `lumi-4b-bigcodebench-prompt-signature-acquisition-v0.1.json` plan/report and its acquired source object |
| `test_bigcodebench_signature_promotion_preflight.py` (3) | Lines 22-23; later 58-59 | `lumi-4b-bigcodebench-signature-promotion-preflight-v0.1.json` plan/report and promotion-plan output |
| `test_bigcodebench_signature_registry_promotion.py` (3) | Lines 23-24; later 46-47 | `lumi-4b-bigcodebench-signature-registry-promotion-v0.1.json` authorization/report |
| `test_code_benchmark_signature_preflight.py` (3) | Lines 22-23; later 61-62 | `lumi-4b-code-benchmark-signature-source-preflight-v0.1.json` and acquisition-plan output |
| `test_deepmind_canary_run.py` (1) | Lines 142-143: source directory absent | Exact local DeepMind source directory named by the test |
| `test_deepmind_dependency_overlay.py` (1) | Lines 43-44: receipt absent | `evidence/lumi-4b-sft-deepmind-dependency-overlay-v0.1.json` |
| `test_deepmind_dependency_resolution.py` (1) | Lines 61-65: one of three exact files absent | `/private/tmp/lumi-deepmind-mathematics-setup.py`, `lumi-absl-py-2.5.0.json`, `lumi-six-1.17.0.json` |
| `test_deepmind_scale_owner_review.py` (1) | Lines 94-95: receipt absent | `evidence/lumi-4b-sft-deepmind-scale-owner-review-v0.1.json` |
| `test_deepmind_source_promotion.py` (1) | Lines 70-71: plan, successor or receipt absent | Source-promotion plan/receipt and source-registry `v0.6` |
| `test_deepmind_source_promotion_preflight.py` (1) | Lines 64-65: report absent | `evidence/lumi-4b-sft-deepmind-source-promotion-preflight-v0.1.json` |
| `test_dmath_owner_review.py` (2) | Lines 94-95 and 114-115 | `lumi-4b-sft-dmath-owner-review-v0.1.json` plan/receipt |
| `test_dmath_source_promotion.py` (1) | Lines 68-69: plan, successor or receipt absent | DMath source-promotion plan/receipt and source-registry `v0.5` |
| `test_first_party_scale_canary_preflight.py` (3) | Lines 29-39: missing plan or any bound file; later 63-64 | Tracked scale-preflight plan binds private teacher-model, tokenizer and evidence paths; those paths are not mounted in the source clone |
| `test_gpt_oss_scale_canary_preflight_v2.py` (3) | Lines 28-29; later 54-55 | `lumi-4b-sft-gpt-oss-scale-canary-preflight-v0.2.json` plan/report and verification |
| `test_livecodebench_research_signatures.py` (2) | Lines 22-23; later 54-55 | `lumi-4b-livecodebench-research-signature-acquisition-v0.1.json` authorization/report |
| `test_livecodebench_signature_registry_promotion.py` (3) | Lines 24-25; later 48-49 | `lumi-4b-livecodebench-signature-registry-promotion-v0.1.json` authorization/report |
| `test_minimax_provider_preflight_v1.py` (3) | Lines 29-30; later 45-46 and 64-65 | `lumi-4b-sft-minimax-provider-preflight-v0.1.json` plan/report and verification |
| `test_oasst1_owner_review.py` (1) | Lines 175-176: plan absent | `configs/lumi-4b-sft-oasst1-owner-review-v0.1.json` |
| `test_opencodeinstruct_owner_review.py` (1) | Lines 104-108: plan or receipt absent | `lumi-4b-sft-opencodeinstruct-owner-review-v0.1.json` plan/receipt |
| `test_openr1_owner_review.py` (1) | Lines 102-103: plan or receipt absent | `lumi-4b-sft-openr1-visible-final-owner-review-v0.1.json` plan/receipt |
| `test_openr1_rights.py` (1) | Lines 67-68: plan or report absent | `lumi-4b-sft-openr1-rights-map-v0.1.json` plan/report |
| `test_production_pretraining.py` (4) | Shared `production_plan` fixture, lines 35-38 | Private marker-production-run `evidence/results/shard-manifest.json` named at lines 29-32 |
| `test_production_training.py` (2) | Lines 320-321 and 337-338 | Exact final-corpus and corrected marker-aware manifest paths named in the two tests |
| `test_stronger_teacher_ab.py` (3) | Lines 34-45: missing bound file; later 81-82 | Tracked execution plan's private authorization, baseline model/manifest, quality exemplars, tokenizer bindings; continuation authorization |
| `test_stronger_teacher_ab_gpu.py` (1) | Lines 37-40: either journal absent | Private baseline/candidate call journals under the exact stronger-teacher attempt |
| `test_stronger_teacher_ab_gpu_completion.py` (1) | Lines 63-67: either journal absent | Same exact private baseline/candidate journals used to establish the historical failure boundary |
| `test_stronger_teacher_ab_preflight.py` (3) | Lines 32-42: missing plan or bound file; later 76-77 | Tracked preflight plan's private failed-scale report/verification bindings; preflight report/verification |

These witnesses are in `tests/` at the audited commit. No owner acceptance,
rights clearance, historical source authenticity or data correctness is inferred
from filenames or from an earlier machine gate. A missing private preflight is
not permission to regenerate its report, repeat a consumed attempt, acquire a
model, issue provider calls, or manufacture authorization.

## Routes that are not the 58 skips

The **329 deselections** are a separate, explicitly accounted population:

- **327 frozen unresolved private nodes** from
  `configs/lumi-ci-unresolved-inventory-v0.1.json`, routed by the separate
  source-partition manifest. Every original entry remains OPEN. The source
  router validates exact collection accounting and does not turn them into
  skips or passes (`scripts/ci_source_partition_v1.py:242-315`).
- **2 original BF16 nodes**, explicitly deselected only from the broad source
  command, execute in the required historical-reproduction/successor step.
  That actual step reported 31 executed cases: 29 passes and the exact two
  historical failures reproduced, with zero skips. It explicitly retained
  `v1_bfloat16_contract_passed=false` and did not relabel the failures as passes.

The **273 isolated controls** are also not skips or deselections: their seven
files are ignored by the broad command and executed in a required fresh
`python -S -B` job. The actual run reported all 273 executed, zero skipped,
zero failures/errors. The suite counts are 26 source-bundle, 51 supervisor,
36 admission, 54 launch-contract, 33 watchdog, 30 guarded-bootstrap and
43 ledger-protocol cases. The separate guard regression step passed 43 cases.

The 55 fixture skips listed here are outside the frozen 327-node partition;
this separate accounting does not add, subtract, close or rewrite any historical
baseline entry. The prior bounded safety/Aya local results likewise do not
decrement that immutable hosted inventory. No existing receipt, historical
source/test, plan, assertion, production gate or baseline count was changed by
this audit.
