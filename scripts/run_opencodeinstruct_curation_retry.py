#!/usr/bin/env python3
"""Run the deterministic, no-network OpenCodeInstruct curation retry."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from lumi.posttraining_opencodeinstruct_retry import (
    load_retry_plan,
    run_deterministic_unit_test_verifier,
)


def _load_original_runner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("lumi_original_opencode_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load original OpenCodeInstruct runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    original = _load_original_runner(root / "scripts/run_opencodeinstruct_curation.py")
    original_verify_raw_objects = original.verify_raw_objects

    def verify_local_reuse(raw_root: Path) -> dict[str, Any]:
        result = original_verify_raw_objects(raw_root)
        result["local_reuse_bytes"] = result["network_bytes"]
        result["network_bytes"] = 0
        return result

    original.load_opencodeinstruct_plan = load_retry_plan
    original.run_local_unit_test_verifier = run_deterministic_unit_test_verifier
    original.verify_raw_objects = verify_local_reuse
    return int(original.main())


if __name__ == "__main__":
    raise SystemExit(main())
