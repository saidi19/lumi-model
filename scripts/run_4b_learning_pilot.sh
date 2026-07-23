#!/usr/bin/env bash
set -euo pipefail

workspace="${LUMI_LEARNING_PILOT_WORKSPACE:-$HOME/lumi-runs/4b-learning-pilot-v0.1}"
source_commit="${LUMI_DISTRIBUTED_SOURCE_COMMIT:?set LUMI_DISTRIBUTED_SOURCE_COMMIT}"
world_size="${LUMI_LEARNING_PILOT_WORLD_SIZE:?set LUMI_LEARNING_PILOT_WORLD_SIZE}"
repo="${LUMI_DISTRIBUTED_REPO:-$HOME/lumi-model}"
manifest="${LUMI_LEARNING_PILOT_MANIFEST:?set LUMI_LEARNING_PILOT_MANIFEST}"
heldout_manifest="${LUMI_LEARNING_PILOT_HELDOUT_MANIFEST:?set LUMI_LEARNING_PILOT_HELDOUT_MANIFEST}"
tokenizer="${LUMI_LEARNING_PILOT_TOKENIZER:?set LUMI_LEARNING_PILOT_TOKENIZER}"
authorization="${LUMI_LEARNING_PILOT_AUTHORIZATION:?set LUMI_LEARNING_PILOT_AUTHORIZATION}"
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
  echo "learning_pilot_already_sealed=true"
  exit 0
fi
test "$(git rev-parse HEAD)" = "$source_commit"
test -z "$(git status --porcelain)"
"$python_bin" scripts/check_4b_learning_pilot_authorization.py \
  --plan "$plan" \
  --authorization "$authorization"

readarray -t run_values < <("$python_bin" - "$plan" "$world_size" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
requested_world = int(sys.argv[2])
topology = plan["selected_topology"]
if topology["world_size"] != requested_world:
    raise SystemExit("plan world size differs from the runner")
run = plan["run_identity"]["optimizer"]
print(topology["gradient_accumulation_steps"])
print(plan["target"]["fresh_through_step"])
print(plan["target"]["resume_through_step"])
PY
)
gradient_accumulation_steps="${run_values[0]}"
fresh_step="${run_values[1]}"
final_step="${run_values[2]}"

rm -f "$failed"
nvidia-smi \
  --query-gpu=index,timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l 1 >> "$telemetry" &
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
  "$torchrun_bin" --standalone --nnodes=1 --nproc-per-node="$world_size" \
    scripts/train_distributed.py \
    --manifest "$manifest" \
    --config configs/lumi-4b-v0.2.json \
    --output "$training" \
    --steps "$target_step" \
    --strategy fsdp_full_shard \
    --per-rank-batch-size 1 \
    --gradient-accumulation-steps "$gradient_accumulation_steps" \
    --gradient-checkpointing \
    --learning-rate 0.0002 \
    --weight-decay 0.1 \
    --max-grad-norm 1.0 \
    --checkpoint-steps 10 \
    --keep-checkpoints 2 \
    --seed 7 \
    --lr-schedule cosine \
    --warmup-steps 10 \
    --schedule-steps 80 \
    --min-learning-rate-ratio 0.1 \
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

evaluate_step() {
  local step=$1
  local checkpoint
  local export_dir="$workspace/evaluation-step-${step}-checkpoint"
  local evaluation_dir="$workspace/evaluation-step-${step}"
  checkpoint="$(printf '%s/checkpoint-%08d' "$training" "$step")"
  "$python_bin" scripts/export_distributed_checkpoint_hf.py \
    --config configs/lumi-4b-v0.2.json \
    --checkpoint "$checkpoint" \
    --output "$export_dir" \
    --dtype bfloat16
  cp "$export_dir/distributed-export-report.json" \
    "$workspace/evaluation-step-${step}-export-report.json"
  "$python_bin" scripts/evaluate_checkpoint.py \
    --plan "$plan" \
    --checkpoint "$export_dir" \
    --manifest "$heldout_manifest" \
    --tokenizer "$tokenizer" \
    --output "$evaluation_dir" \
    --attention-implementation flash_attention_2 \
    --document-isolation \
    --device cuda \
    --resume
  rm -rf "$export_dir"
}

exec 3>&1
{
  echo "run_attempt_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "run_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  SECONDS=0
  if [[ ! -f "$workspace/fresh.complete" ]]; then
    current_step="$(state_step)"
    if (( current_step > fresh_step )); then
      echo "training advanced past the unsealed fresh milestone" >&2
      exit 1
    fi
    if (( current_step < fresh_step )); then
      resume_args=()
      if [[ -f "$training/distributed-training-state.json" ]]; then
        resume_args=(--resume)
      fi
      run_training "$fresh_step" "$workspace/fresh.log" \
        --skip-final-model-digest "${resume_args[@]}"
    fi
    cp "$training/distributed-training-state.json" "$workspace/fresh-state.json"
    cp "$training/distributed-metrics.jsonl" "$workspace/fresh-metrics.jsonl"
    cp "$training/checkpoint-00000010/manifest.sha256" \
      "$workspace/fresh-checkpoint-manifest.sha256"
    cp "$training/checkpoint-00000010/distributed-training-state.json" \
      "$workspace/fresh-checkpoint-state.json"
    evaluate_step "$fresh_step"
    touch "$workspace/fresh.complete"
  fi

  if [[ ! -f "$workspace/resume.complete" ]]; then
    current_step="$(state_step)"
    if (( current_step < final_step )); then
      run_training "$final_step" "$workspace/resume.log" --resume
    fi
    evaluate_step "$final_step"
    touch "$workspace/resume.complete"
  fi
  echo "run_duration_seconds=$SECONDS"
  echo "run_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee -a "$workspace/run.log" >&3
exec 3>&-

kill "$telemetry_pid" 2>/dev/null || true
wait "$telemetry_pid" 2>/dev/null || true
trap - EXIT
