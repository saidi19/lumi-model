#!/usr/bin/env bash
set -euo pipefail

workspace="${LUMI_LEARNING_PILOT_WORKSPACE:-$HOME/lumi-runs/4b-learning-pilot-v0.1}"
repo="${LUMI_DISTRIBUTED_REPO:-$HOME/lumi-model}"
authorization="${LUMI_LEARNING_PILOT_AUTHORIZATION:?set LUMI_LEARNING_PILOT_AUTHORIZATION}"
source_commit="${LUMI_DISTRIBUTED_SOURCE_COMMIT:?set LUMI_DISTRIBUTED_SOURCE_COMMIT}"
world_size="${LUMI_LEARNING_PILOT_WORLD_SIZE:?set LUMI_LEARNING_PILOT_WORLD_SIZE}"
manifest="${LUMI_LEARNING_PILOT_MANIFEST:?set LUMI_LEARNING_PILOT_MANIFEST}"
heldout_manifest="${LUMI_LEARNING_PILOT_HELDOUT_MANIFEST:?set LUMI_LEARNING_PILOT_HELDOUT_MANIFEST}"
tokenizer="${LUMI_LEARNING_PILOT_TOKENIZER:?set LUMI_LEARNING_PILOT_TOKENIZER}"
session="${LUMI_LEARNING_PILOT_TMUX_SESSION:-lumi-4b-learning-pilot}"

cd "$repo"
test -f "$workspace/setup.complete"
test -f "$workspace/preflight.complete"
"$repo/.venv/bin/python" scripts/check_4b_learning_pilot_authorization.py \
  --plan "$workspace/plan.json" \
  --authorization "$authorization"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "learning_pilot_tmux_already_exists=$session"
  exit 0
fi
tmux new-session -d -s "$session" env \
  "LUMI_LEARNING_PILOT_WORKSPACE=$workspace" \
  "LUMI_DISTRIBUTED_REPO=$repo" \
  "LUMI_LEARNING_PILOT_AUTHORIZATION=$authorization" \
  "LUMI_DISTRIBUTED_SOURCE_COMMIT=$source_commit" \
  "LUMI_LEARNING_PILOT_WORLD_SIZE=$world_size" \
  "LUMI_LEARNING_PILOT_MANIFEST=$manifest" \
  "LUMI_LEARNING_PILOT_HELDOUT_MANIFEST=$heldout_manifest" \
  "LUMI_LEARNING_PILOT_TOKENIZER=$tokenizer" \
  bash "$repo/scripts/run_4b_learning_pilot_guarded.sh"
sleep 1
tmux has-session -t "$session"
echo "learning_pilot_tmux_started=$session"
