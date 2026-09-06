"""Explicit self-only metadata observation. Never launches or authorizes SFT."""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path


def emit(report):
    try:
        raw = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        if len(raw) > 8192:
            return False
        return os.write(1, raw) == len(raw)  # Single bounded stdout write; no partial-write retry.
    except BaseException:
        return False


def rejected(code):
    return {"state": "failed", "failure_code": code,
            "execution_authorized": False, "production_eligible": False}


def main():
    if (sys.argv[1:] != ["--self-only"] or not sys.flags.no_site
            or not sys.dont_write_bytecode or sys.flags.optimize):
        emit(rejected("require_stdlib_self_only"))
        return 2
    complete = False
    try:
        package = types.ModuleType("_lumi_self_containment_observation")
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "src/lumi")]
        sys.modules[package.__name__] = package
        observer = importlib.import_module(package.__name__ + ".posttraining_sft_host_observer_v1")
        report = observer.observe_self()
        complete = report["observation_complete"] is True
    except BaseException:
        report = rejected("host_observation_failed")
    # Zero means the observation completed, NOT host readiness or training approval.
    return 0 if emit(report) and complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
