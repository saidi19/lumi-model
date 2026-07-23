#!/usr/bin/env bash
set -euo pipefail

workspace="${LUMI_LEARNING_PILOT_WORKSPACE:-$HOME/lumi-runs/4b-learning-pilot-v0.1}"
source_commit="${LUMI_DISTRIBUTED_SOURCE_COMMIT:?set LUMI_DISTRIBUTED_SOURCE_COMMIT}"
world_size="${LUMI_LEARNING_PILOT_WORLD_SIZE:?set LUMI_LEARNING_PILOT_WORLD_SIZE}"
repo="${LUMI_DISTRIBUTED_REPO:-$HOME/lumi-model}"
manifest="${LUMI_LEARNING_PILOT_MANIFEST:?set LUMI_LEARNING_PILOT_MANIFEST}"
heldout_manifest="${LUMI_LEARNING_PILOT_HELDOUT_MANIFEST:?set LUMI_LEARNING_PILOT_HELDOUT_MANIFEST}"
python_bin="$repo/.venv/bin/python"
ruff_bin="$repo/.venv/bin/ruff"

cd "$repo"
mkdir -p "$workspace"
test -f "$workspace/setup.complete"
if [[ -f "$workspace/preflight.complete" ]]; then
  echo "preflight_already_complete=true"
  exit 0
fi

exec 3>&1
{
  echo "preflight_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  test "$(git rev-parse HEAD)" = "$source_commit"
  test -z "$(git status --porcelain)"
  "$python_bin" -m pytest 2>&1 | tee "$workspace/tests.log"
  "$ruff_bin" check . 2>&1 | tee "$workspace/ruff.log"
  "$python_bin" scripts/verify_flash_varlen_cuda.py \
    --output "$workspace/flash-varlen-cuda-report.json"
  "$python_bin" scripts/plan_4b_learning_pilot.py \
    --contract configs/4b-learning-pilot-v0.1.json \
    --manifest "$manifest" \
    --heldout-manifest "$heldout_manifest" \
    --config configs/lumi-4b-v0.2.json \
    --source-commit "$source_commit" \
    --world-size "$world_size" \
    --output "$workspace/plan.json"
  echo "preflight_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$workspace/preflight.log" >&3
exec 3>&-
touch "$workspace/preflight.complete"
