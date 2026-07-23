#!/usr/bin/env bash
set -euo pipefail

workspace="${LUMI_LEARNING_PILOT_WORKSPACE:-$HOME/lumi-runs/4b-learning-pilot-v0.1}"
repo="${LUMI_DISTRIBUTED_REPO:-$HOME/lumi-model}"
authorization="${LUMI_LEARNING_PILOT_AUTHORIZATION:?set LUMI_LEARNING_PILOT_AUTHORIZATION}"
python_bin="$repo/.venv/bin/python"
runtime_used_path="$workspace/runtime-used-seconds"

"$python_bin" "$repo/scripts/check_4b_learning_pilot_authorization.py" \
  --plan "$workspace/plan.json" \
  --authorization "$authorization"
if [[ -f "$workspace/seal.complete" ]]; then
  echo "learning_pilot_already_sealed=true"
  exit 0
fi

readarray -t runtime_values < <("$python_bin" - "$authorization" "$runtime_used_path" <<'PY'
import json
import sys
from pathlib import Path

authorization = json.load(open(sys.argv[1], encoding="utf-8"))
minutes = authorization.get("max_runtime_minutes")
world_size = authorization.get("world_size")
maximum = {4: 480, 8: 240}.get(world_size)
if (
    maximum is None
    or not isinstance(minutes, int)
    or isinstance(minutes, bool)
    or not 30 <= minutes <= maximum
):
    raise SystemExit("invalid learning-pilot runtime cap")
used_path = Path(sys.argv[2])
used = int(used_path.read_text().strip()) if used_path.is_file() else 0
maximum_seconds = minutes * 60
if used < 0 or used >= maximum_seconds:
    raise SystemExit("learning-pilot cumulative runtime cap is exhausted")
print(maximum_seconds)
print(used)
print(maximum_seconds - used)
PY
)
max_runtime_seconds="${runtime_values[0]}"
runtime_used_seconds="${runtime_values[1]}"
remaining_runtime_seconds="${runtime_values[2]}"
attempt_started="$(date +%s)"

set +e
timeout --foreground --signal=TERM --kill-after=300 \
  "$remaining_runtime_seconds" \
  bash "$repo/scripts/run_4b_learning_pilot.sh"
code=$?
set -e
attempt_finished="$(date +%s)"
attempt_seconds="$((attempt_finished - attempt_started))"
"$python_bin" - \
  "$runtime_used_path" \
  "$runtime_used_seconds" \
  "$attempt_seconds" \
  "$max_runtime_seconds" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
previous = int(sys.argv[2])
elapsed = int(sys.argv[3])
maximum = int(sys.argv[4])
observed = min(previous + max(elapsed, 0), maximum)
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(f"{observed}\n", encoding="utf-8")
with temporary.open("r+") as handle:
    os.fsync(handle.fileno())
temporary.replace(path)
PY
if [[ "$code" -eq 124 || "$code" -eq 137 ]]; then
  touch "$workspace/runtime-limit.reached"
fi
if [[ "$code" -eq 0 ]]; then
  "$python_bin" "$repo/scripts/seal_4b_learning_pilot.py" \
    --workspace "$workspace" \
    --authorization "$authorization"
fi
exit "$code"
