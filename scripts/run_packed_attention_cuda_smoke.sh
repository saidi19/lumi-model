#!/usr/bin/env bash
set -euo pipefail

workspace="${LUMI_PACKING_SMOKE_WORKSPACE:-$HOME/lumi-runs/packed-attention-isolated-v0.1}"
python_bin="$HOME/lumi-model/.venv/bin/python"
ruff_bin="$HOME/lumi-model/.venv/bin/ruff"
manifest="$HOME/lumi-data/scaling-data-1b-v0.1/final-1.4b/manifest.json"
tokenizer="$HOME/lumi-data/tokenizer-candidates-v0.2/lumi-tokenizer-65k"

cd "$HOME/lumi-model"
mkdir -p "$workspace/provenance"
test -f "$workspace/setup.complete"
test ! -f "$workspace/seal.complete"

if [[ ! -f "$workspace/preflight.complete" ]]; then
  exec 3>&1
  {
    echo "preflight_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s  %s\n' \
      'ce4dee8b2e72cb1a064ae6434851f0ecba915559f6444f84b3ed15c173c834ca' \
      "$workspace/baseline-calibration-summary.json" | sha256sum -c -
    printf '%s  %s\n' \
      'c5a4667014b3557710bf410fa5e5ac0aa273819ecc6d9ff1f72447c143db7d4f' \
      "$workspace/baseline-calibration-metrics.jsonl" | sha256sum -c -
    "$python_bin" -m pytest 2>&1 | tee "$workspace/tests.log"
    "$ruff_bin" check . 2>&1 | tee "$workspace/ruff.log"
    "$python_bin" scripts/preflight_cloud_pilot.py \
      --plan configs/cloud-pilot-1b-v0.1.json \
      --manifest "$manifest" \
      --tokenizer "$tokenizer" \
      --workspace "$HOME/lumi-runs" \
      --require-cuda \
      --output "$workspace/cloud-preflight.json"
    "$python_bin" scripts/verify_flash_varlen_cuda.py \
      --output "$workspace/flash-varlen-cuda-report.json"
    echo "preflight_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } 2>&1 | tee "$workspace/preflight.log" >&3
  exec 3>&-
  touch "$workspace/preflight.complete"
fi

run_policy() {
  local name=$1
  local isolation=$2
  local training="$workspace/$name-training"
  local marker="$workspace/$name-training.complete"
  local failed="$workspace/$name-training.failed"
  local log="$workspace/$name-training.log"
  local telemetry="$workspace/$name-telemetry.csv"
  local telemetry_pid
  local -a resume=()
  local -a isolation_args=()

  if [[ -f "$marker" ]]; then
    echo "${name}_training_already_complete=true"
    return
  fi

  rm -f "$failed"
  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
    --format=csv,noheader,nounits -l 1 > "$telemetry" &
  telemetry_pid=$!

  finish_policy() {
    local code=$?
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
    trap - EXIT
    if [[ "$code" -ne 0 ]]; then
      touch "$failed"
    fi
    exit "$code"
  }
  trap finish_policy EXIT

  if [[ -f "$training/training-state.json" ]]; then
    resume=(--resume)
  fi
  if [[ "$isolation" == "true" ]]; then
    isolation_args=(--document-isolation)
  fi

  exec 3>&1
  {
    echo "smoke_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    SECONDS=0
    "$python_bin" scripts/train_pilot.py \
      --manifest "$manifest" \
      --config configs/lumi-1b.json \
      --output "$training" \
      --steps 100 \
      --batch-size 1 \
      --gradient-accumulation-steps 8 \
      --gradient-checkpointing \
      --learning-rate 0.0002 \
      --lr-schedule cosine \
      --warmup-steps 500 \
      --schedule-steps 42725 \
      --min-learning-rate-ratio 0.1 \
      --weight-decay 0.1 \
      --max-grad-norm 1.0 \
      --checkpoint-steps 100 \
      --keep-checkpoints 2 \
      --seed 7 \
      --attention-implementation flash_attention_2 \
      --device cuda \
      "${isolation_args[@]}" \
      "${resume[@]}"
    echo "smoke_duration_seconds=$SECONDS"
    echo "smoke_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } 2>&1 | tee "$log" >&3
  exec 3>&-

  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  trap - EXIT
  touch "$marker"
}

run_policy current false
run_policy isolated true

"$python_bin" scripts/seal_packed_attention_cuda_smoke.py \
  --workspace "$workspace" \
  --hourly-rate-usd 1.99
