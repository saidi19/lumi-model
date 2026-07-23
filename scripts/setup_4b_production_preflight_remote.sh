#!/usr/bin/env bash
set -euo pipefail

repo="${LUMI_PREFLIGHT_REPO:-$HOME/lumi-4b-preflight}"
workspace="${LUMI_PREFLIGHT_WORKSPACE:-$HOME/lumi-runs/4b-production-preflight-v0.1}"
mkdir -p "$workspace"
exec > >(tee -a "$workspace/setup.log") 2>&1

if [[ -f "$workspace/setup.complete" ]]; then
  echo "setup_already_complete=true"
  exit 0
fi

echo "setup_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
df -h "$HOME"
free -h

if ! command -v python3.12 >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install -y python3.12 python3.12-venv
fi

python3.12 -m venv "$repo/.venv"
python_bin="$repo/.venv/bin/python"
pip_bin="$repo/.venv/bin/pip"
"$python_bin" -m pip install --upgrade pip wheel
"$pip_bin" install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.0
"$pip_bin" install -e "$repo[data,dev]" transformers==5.14.1 psutil
"$pip_bin" check
"$python_bin" - <<'PY'
import platform

import torch
import transformers

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_devices={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("the remote preflight expects exactly one visible CUDA GPU")
PY

"$pip_bin" freeze > "$workspace/pip-freeze.txt"
echo "setup_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
touch "$workspace/setup.complete"
