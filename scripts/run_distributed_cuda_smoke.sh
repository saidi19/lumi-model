#!/usr/bin/env bash
set -euo pipefail

workspace="${LUMI_DISTRIBUTED_SMOKE_WORKSPACE:-$HOME/lumi-runs/distributed-cuda-smoke-v0.1}"
source_commit="${LUMI_DISTRIBUTED_SOURCE_COMMIT:?set LUMI_DISTRIBUTED_SOURCE_COMMIT}"
repo="${LUMI_DISTRIBUTED_REPO:-$HOME/lumi-model}"
manifest="${LUMI_DISTRIBUTED_SMOKE_MANIFEST:?set LUMI_DISTRIBUTED_SMOKE_MANIFEST}"
authorization="${LUMI_DISTRIBUTED_SMOKE_AUTHORIZATION:?set LUMI_DISTRIBUTED_SMOKE_AUTHORIZATION}"
python_bin="$repo/.venv/bin/python"
torchrun_bin="$repo/.venv/bin/torchrun"
training="$workspace/training"
plan="$workspace/plan.json"
telemetry="$workspace/gpu-telemetry.csv"
failed="$workspace/run.failed"

cd "$repo"
test -f "$workspace/setup.complete"
test -f "$workspace/preflight.complete"
if [[ -f "$workspace/seal.complete" ]]; then
  echo "distributed_cuda_smoke_already_sealed=true"
  exit 0
fi
test "$(git rev-parse HEAD)" = "$source_commit"
test -z "$(git status --porcelain)"
"$python_bin" scripts/check_distributed_cuda_smoke_authorization.py \
  --plan "$plan" \
  --authorization "$authorization"

rm -f "$failed"
nvidia-smi \
  --query-gpu=index,timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l 1 > "$telemetry" &
telemetry_pid=$!

finish_run() {
  local code=$?
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then
    touch "$failed"
  fi
  exit "$code"
}
trap finish_run EXIT

state_step() {
  "$python_bin" - "$training/distributed-training-state.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(json.loads(path.read_text())["step"] if path.is_file() else 0)
PY
}

run_training() {
  local target_step=$1
  local log=$2
  shift 2
  "$torchrun_bin" --standalone --nnodes=1 --nproc-per-node=2 \
    scripts/train_distributed.py \
    --manifest "$manifest" \
    --config configs/lumi-4b-v0.2.json \
    --output "$training" \
    --steps "$target_step" \
    --strategy fsdp_full_shard \
    --per-rank-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --gradient-checkpointing \
    --learning-rate 0.0002 \
    --weight-decay 0.1 \
    --max-grad-norm 1.0 \
    --checkpoint-steps 1 \
    --keep-checkpoints 1 \
    --seed 7 \
    --attention-implementation flash_attention_2 \
    --document-isolation \
    --optimizer-profile legacy \
    --precision bf16 \
    --gradient-sync-mode last_microbatch \
    --device cuda \
    --backend nccl \
    --timeout-seconds 1800 \
    --source-commit "$source_commit" \
    --authorization "$authorization" \
    "$@" 2>&1 | tee -a "$log"
}

exec 3>&1
{
  echo "run_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  SECONDS=0
  current_step="$(state_step)"
  if (( current_step < 1 )); then
    resume_args=()
    if [[ -f "$training/distributed-training-state.json" ]]; then
      resume_args=(--resume)
    fi
    run_training 1 "$workspace/fresh.log" --skip-final-model-digest "${resume_args[@]}"
  fi
  cp "$training/distributed-training-state.json" "$workspace/fresh-state.json"
  cp "$training/distributed-metrics.jsonl" "$workspace/fresh-metrics.jsonl"
  touch "$workspace/fresh.complete"

  current_step="$(state_step)"
  if (( current_step < 2 )); then
    run_training 2 "$workspace/resume.log" --resume
  fi
  touch "$workspace/resume.complete"
  echo "run_duration_seconds=$SECONDS"
  echo "run_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$workspace/run.log" >&3
exec 3>&-

kill "$telemetry_pid" 2>/dev/null || true
wait "$telemetry_pid" 2>/dev/null || true
trap - EXIT

"$python_bin" scripts/seal_distributed_cuda_smoke.py \
  --workspace "$workspace" \
  --authorization "$authorization"
