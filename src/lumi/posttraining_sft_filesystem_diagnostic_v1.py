"""Fixed WorkerRun failure projection only; no I/O, model imports or raw text."""

from __future__ import annotations

from .posttraining_sft_worker_run_v1 import ERROR_CODES, STAGES

SCHEMA = "lumi.sft-worker-run-fixture.v0.1"
OUTER_STAGES = frozenset({"guard", "fixture", "workers", "ledger", "cleanup",
                          "comparison", "accounting", "complete", "not_reported"})
CHILD_STAGES = frozenset({"guard", "dependencies", "gloo", "models", "run", "checks",
                          "result", "cleanup", "done", "not_reported"})
FIELDS = {"stage": OUTER_STAGES, "child_stage": CHILD_STAGES,
          "worker_stage": STAGES, "worker_code": ERROR_CODES}


def project_failure(report):
    if (type(report) is not dict or type(report.get("schema_version")) is not str
            or report["schema_version"] != SCHEMA or type(report.get("status")) is not str
            or report["status"] != "failed"):
        return None
    result = {"schema_version": SCHEMA, "status": "failed"}
    for name, allowed in FIELDS.items():
        value = report.get(name)
        result[name] = value if type(value) is str and value in allowed else "not_reported"
    return result


def component_code(report):
    safe = project_failure(report)
    if safe is None:
        return None
    return "worker_run_" + "_".join(safe[name] for name in FIELDS)
