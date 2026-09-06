"""Fixed inert worker entry: no training dispatch, lumi initialization or ML imports."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def main() -> int:
    # Check before any repository module is loaded. Do not print argv/errors.
    if sys.flags.no_site != 1 or not sys.dont_write_bytecode or sys.flags.optimize:
        return 2
    if len(sys.argv) != 3:
        return 2
    try:
        package = types.ModuleType("_lumi_synthetic_ipc")
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "src" / "lumi")]
        sys.modules[package.__name__] = package
        module = importlib.import_module(package.__name__ + ".posttraining_sft_local_ipc_v1")
        return module.synthetic_worker(int(sys.argv[1]), sys.argv[2])
    except BaseException:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
