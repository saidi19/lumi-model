#!/usr/bin/env bash
set -euo pipefail

workspace="${LUMI_PACKING_SMOKE_WORKSPACE:-$HOME/lumi-runs/packed-attention-isolated-v0.1}"
mkdir -p "$workspace/provenance"
exec > >(tee -a "$workspace/setup.log") 2>&1

if [[ -f "$workspace/setup.complete" ]]; then
  echo "setup_already_complete=true"
  exit 0
fi

echo "setup_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
uname -m
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

if ! command -v python3.12 >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install -y python3.12 python3.12-venv
fi

python3.12 -m venv "$HOME/lumi-model/.venv"
python_bin="$HOME/lumi-model/.venv/bin/python"
pip_bin="$HOME/lumi-model/.venv/bin/pip"

"$python_bin" -m pip install --upgrade pip wheel
"$pip_bin" install \
  --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.0
"$pip_bin" install -e "$HOME/lumi-model[data,dev]"
"$pip_bin" install transformers==5.14.1 packaging psutil ninja
MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0 \
  "$pip_bin" install flash-attn==2.8.3 --no-build-isolation
"$pip_bin" check

"$python_bin" - <<'PY'
import platform

import flash_attn
import torch
import transformers

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print(f"flash_attn={flash_attn.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"bf16={torch.cuda.is_bf16_supported()}")
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA/BF16 runtime gate failed")
if transformers.__version__ != "5.14.1":
    raise SystemExit("Transformers version changed")
if flash_attn.__version__ != "2.8.3":
    raise SystemExit("FlashAttention version changed")
PY

"$pip_bin" freeze > "$workspace/provenance/pip-freeze.txt"
echo "setup_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
touch "$workspace/setup.complete"
