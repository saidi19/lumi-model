#!/usr/bin/env python3
"""Run only the fixed synthetic CPU SFT integration with Python -S -B.

This is not a GPU launcher or a way to admit existing checkpoint/data paths.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path


def main(argv=None):
    repository = Path(__file__).resolve().parents[1]
    package = types.ModuleType("_lumi_sft_offline")
    package.__path__ = [str(repository / "src" / "lumi")]
    sys.modules[package.__name__] = package
    try:
        admission = importlib.import_module("_lumi_sft_offline.posttraining_sft_cpu_admission_v1")
        admission.validate_cli(sys.argv[1:] if argv is None else argv)
        admission.pre_ml_runtime_check()
        raw = admission.canonical(admission.fixed_request())
        pin = admission.digest(raw)
        admission.validate_request(
            raw,
            pin,
            no_site=sys.flags.no_site,
            no_bytecode=sys.dont_write_bytecode,
        )
        integration = importlib.import_module("_lumi_sft_offline.posttraining_sft_offline_cpu_v1")
        report = integration.run_integration(raw, pin, repository_root=repository)
    except BaseException:
        print('{"status":"FAIL_synthetic_cpu_integration"}', flush=True)
        return 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
