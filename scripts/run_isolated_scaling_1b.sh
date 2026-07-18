#!/usr/bin/env bash
set -euo pipefail

workspace="${LUMI_1B_ISOLATED_WORKSPACE:-$HOME/lumi-runs/lumi-1b-isolated-1p4b-v0.1}"
authorization="${LUMI_1B_AUTHORIZATION:-}"
source_archive="${LUMI_SOURCE_ARCHIVE:-}"
repo="$HOME/lumi-model"
python_bin="$repo/.venv/bin/python"
ruff_bin="$repo/.venv/bin/ruff"
plan="$repo/configs/isolated-scaling-run-1b-v0.1.json"
training_manifest="$HOME/lumi-data/scaling-data-1b-v0.1/final-1.4b/manifest.json"
heldout_manifest="$HOME/lumi-data/evaluation-pilot-v0.1/heldout/manifest.json"
tokenizer="$HOME/lumi-data/tokenizer-candidates-v0.2/lumi-tokenizer-65k"
training="$workspace/training"

verify_source_archive() {
  local archive=$1
  local extracted
  local source
  local relative
  local status=0
  extracted="$(mktemp -d)"
  tar -xf "$archive" -C "$extracted"
  test -d "$extracted/lumi-model"
  while IFS= read -r -d '' source; do
    relative="${source#"$extracted"/}"
    if [[ ! -f "$HOME/$relative" ]] || ! cmp -s "$source" "$HOME/$relative"; then
      echo "ERROR: extracted source differs from archive: $relative" >&2
      status=1
      break
    fi
  done < <(find "$extracted/lumi-model" -type f -print0)
  rm -rf "$extracted"
  return "$status"
}

if [[ -z "$authorization" || ! -f "$authorization" ]]; then
  echo "ERROR: set LUMI_1B_AUTHORIZATION to the owner-approved JSON artifact" >&2
  exit 2
fi
if [[ -z "$source_archive" || ! -f "$source_archive" ]]; then
  echo "ERROR: set LUMI_SOURCE_ARCHIVE to the transferred source tarball" >&2
  exit 2
fi
test -f "$workspace/setup.complete"
test ! -f "$workspace/seal.complete"

cd "$repo"
source_commit="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "$authorization")"
expected_source_archive_sha256="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_archive_sha256"])' "$authorization")"
actual_source_archive_sha256="$(sha256sum "$source_archive" | cut -d ' ' -f 1)"
if [[ "$actual_source_archive_sha256" != "$expected_source_archive_sha256" ]]; then
  echo "ERROR: source archive SHA-256 changed after setup" >&2
  exit 2
fi
verify_source_archive "$source_archive"
if [[ "$(cat "$workspace/provenance/source-commit.txt")" != "$source_commit" ]]; then
  echo "ERROR: setup provenance does not match the authorized source commit" >&2
  exit 2
fi
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [[ "$(git rev-parse HEAD)" != "$source_commit" || -n "$(git status --porcelain)" ]]; then
    echo "ERROR: source repository changed after authorization" >&2
    exit 2
  fi
fi
"$python_bin" scripts/check_scaling_launch_authorization.py \
  --plan "$plan" \
  --authorization "$authorization" \
  --source-commit "$source_commit" \
  --output "$workspace/authorization-verified.json"
approved_step="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["approved_through_step"])' "$workspace/authorization-verified.json")"
resume_required="$("$python_bin" -c 'import json,sys; print("true" if json.load(open(sys.argv[1])).get("resume") else "false")' "$workspace/authorization-verified.json")"
if [[ "$resume_required" == "true" ]]; then
  test -f "$workspace/resume-import.complete"
  test -f "$workspace/resume-import.json"
fi

if [[ ! -f "$workspace/preflight.complete" ]]; then
  exec 3>&1
  {
    echo "preflight_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "$python_bin" -m pytest 2>&1 | tee "$workspace/tests.log"
    "$ruff_bin" check . 2>&1 | tee "$workspace/ruff.log"
    "$python_bin" scripts/preflight_cloud_pilot.py \
      --plan configs/cloud-pilot-1b-v0.1.json \
      --manifest "$training_manifest" \
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

telemetry="$workspace/gpu-telemetry.csv"
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l 1 >> "$telemetry" &
telemetry_pid=$!

finish_run() {
  local code=$?
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  trap - EXIT
  exit "$code"
}
trap finish_run EXIT

train_to() {
  local name=$1
  local step=$2
  local marker="$workspace/milestone-$name.complete"
  local log="$workspace/milestone-$name.log"
  local -a resume=()

  if [[ -f "$marker" ]]; then
    echo "milestone_${name}_already_complete=true"
    return
  fi
  if (( approved_step < step )); then
    echo "authorization_boundary_reached=$approved_step next_step=$step"
    return 20
  fi
  if [[ -f "$training/training-state.json" ]]; then
    resume=(--resume)
  fi
  exec 3>&1
  {
    echo "milestone_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    SECONDS=0
    "$python_bin" scripts/train_pilot.py \
      --manifest "$training_manifest" \
      --config configs/lumi-1b.json \
      --output "$training" \
      --steps "$step" \
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
      --checkpoint-steps 2500 \
      --keep-checkpoints 2 \
      --seed 7 \
      --attention-implementation flash_attention_2 \
      --document-isolation \
      --device cuda \
      "${resume[@]}"
    "$python_bin" scripts/check_scaling_training_state.py \
      --plan "$plan" \
      --training-output "$training" \
      --expected-step "$step"
    echo "milestone_duration_seconds=$SECONDS"
    echo "milestone_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } 2>&1 | tee "$log" >&3
  exec 3>&-
  touch "$marker"
}

evaluate_isolated() {
  local step=$1
  local checkpoint_name
  local checkpoint
  local snapshot
  local output
  local marker
  local -a resume=()

  printf -v checkpoint_name 'checkpoint-%08d' "$step"
  checkpoint="$training/$checkpoint_name"
  snapshot="$workspace/milestones/$checkpoint_name"
  output="$workspace/evaluations/step-$(printf '%08d' "$step")-isolated"
  marker="$workspace/evaluation-step-$(printf '%08d' "$step")-isolated.complete"
  if [[ -f "$marker" ]]; then
    echo "evaluation_step_${step}_already_complete=true"
    return
  fi
  if [[ ! -d "$snapshot" ]]; then
    "$python_bin" scripts/snapshot_model_checkpoint.py \
      --checkpoint "$checkpoint" \
      --output "$snapshot"
  fi
  if [[ -f "$output/evaluation-state.json" || -f "$output/evaluation-report.json" ]]; then
    resume=(--resume)
  fi
  "$python_bin" scripts/evaluate_checkpoint.py \
    --plan configs/evaluation-pilot-v0.1.json \
    --checkpoint "$snapshot" \
    --manifest "$heldout_manifest" \
    --tokenizer "$tokenizer" \
    --output "$output" \
    --attention-implementation flash_attention_2 \
    --document-isolation \
    --device cuda \
    "${resume[@]}"
  "$python_bin" scripts/evaluate_checkpoint.py \
    --plan configs/evaluation-pilot-v0.1.json \
    --checkpoint "$snapshot" \
    --manifest "$heldout_manifest" \
    --tokenizer "$tokenizer" \
    --output "$output" \
    --attention-implementation flash_attention_2 \
    --document-isolation \
    --device cuda \
    --resume
  touch "$marker"
}

train_to smoke_fresh 1
if (( approved_step < 2 )); then
  echo "authorization_boundary_reached=$approved_step next_step=2"
  exit 0
fi
train_to smoke_resume 2
if (( approved_step < 100 )); then
  echo "authorization_boundary_reached=$approved_step next_step=100"
  exit 0
fi
train_to calibration 100
evaluate_isolated 100
if (( approved_step < 7630 )); then
  echo "authorization_boundary_reached=$approved_step next_step=7630"
  exit 0
fi
train_to learning_gate 7630
evaluate_isolated 7630
if (( approved_step < 42725 )); then
  echo "authorization_boundary_reached=$approved_step next_step=42725"
  exit 0
fi
train_to unique_corpus 42725
evaluate_isolated 42725

legacy_output="$workspace/evaluations/step-00042725-legacy"
legacy_marker="$workspace/evaluation-step-00042725-legacy.complete"
if [[ ! -f "$legacy_marker" ]]; then
  legacy_resume=()
  if [[ -f "$legacy_output/evaluation-state.json" || -f "$legacy_output/evaluation-report.json" ]]; then
    legacy_resume=(--resume)
  fi
  "$python_bin" scripts/evaluate_checkpoint.py \
    --plan configs/evaluation-pilot-v0.1.json \
    --checkpoint "$workspace/milestones/checkpoint-00042725" \
    --manifest "$heldout_manifest" \
    --tokenizer "$tokenizer" \
    --output "$legacy_output" \
    --attention-implementation eager \
    --device cuda \
    "${legacy_resume[@]}"
  "$python_bin" scripts/evaluate_checkpoint.py \
    --plan configs/evaluation-pilot-v0.1.json \
    --checkpoint "$workspace/milestones/checkpoint-00042725" \
    --manifest "$heldout_manifest" \
    --tokenizer "$tokenizer" \
    --output "$legacy_output" \
    --attention-implementation eager \
    --device cuda \
    --resume
  touch "$legacy_marker"
fi

kill "$telemetry_pid" 2>/dev/null || true
wait "$telemetry_pid" 2>/dev/null || true
trap - EXIT

"$python_bin" scripts/seal_isolated_scaling_run.py \
  --plan "$plan" \
  --authorization "$authorization" \
  --source-commit "$source_commit" \
  --workspace "$workspace"
