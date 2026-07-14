from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_self_contained_checkpoint_loads_in_isolated_python() -> None:
    repository_root = Path(__file__).parents[1]
    script = repository_root / "scripts" / "check_isolated_hf.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    assert "PASS: isolated AutoModel and AutoModelForCausalLM" in completed.stdout
