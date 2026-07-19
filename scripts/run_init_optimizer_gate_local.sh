#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="${LUMI_PYTHON_BIN:-/private/tmp/lumi-model/.venv/bin/python}"
run_root="${LUMI_INIT_OPTIMIZER_GATE_ROOT:-/Users/afnan/Documents/lumi-data/init-optimizer-ab-v0.1}"
training_manifest="${LUMI_INIT_OPTIMIZER_TRAINING_MANIFEST:-/Users/afnan/Documents/lumi-data/real-pilot-smoke-v0.1/packed/manifest.json}"
heldout_manifest="${LUMI_INIT_OPTIMIZER_HELDOUT_MANIFEST:-/Users/afnan/Documents/lumi-data/evaluation-pilot-v0.1/heldout/manifest.json}"
specification="$repository_root/configs/init-optimizer-ab-v0.1.json"

mkdir -p "$run_root"
exec > >(tee -a "$run_root/run.log") 2>&1
cd "$repository_root"

printf 'run_started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'source_commit=%s\n' "$(git rev-parse HEAD)"
printf 'specification=%s\n' "$specification"
printf 'training_manifest=%s\n' "$training_manifest"
printf 'heldout_manifest=%s\n' "$heldout_manifest"

PYTHONPATH=src "$python_bin" scripts/run_init_optimizer_ab.py \
  --specification "$specification" \
  --training-manifest "$training_manifest" \
  --heldout-manifest "$heldout_manifest" \
  --output "$run_root"

touch "$run_root/run.complete"
(
  cd "$run_root"
  find . -type f \
    ! -name manifest.sha256 \
    ! -name run.log \
    -print0 \
    | sort -z \
    | xargs -0 shasum -a 256
) > "$run_root/manifest.sha256"
printf 'run_completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
