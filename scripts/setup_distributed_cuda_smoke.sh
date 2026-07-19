#!/usr/bin/env bash
set -euo pipefail

workspace="${LUMI_DISTRIBUTED_SMOKE_WORKSPACE:-$HOME/lumi-runs/distributed-cuda-smoke-v0.1}"
source_commit="${LUMI_DISTRIBUTED_SOURCE_COMMIT:?set LUMI_DISTRIBUTED_SOURCE_COMMIT}"
repo="${LUMI_DISTRIBUTED_REPO:-$HOME/lumi-model}"
mkdir -p "$workspace/provenance"
exec > >(tee -a "$workspace/setup.log") 2>&1

if [[ -f "$workspace/setup.complete" ]]; then
  echo "setup_already_complete=true"
  exit 0
fi

echo "setup_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd "$repo"
test "$(git rev-parse HEAD)" = "$source_commit"
test -z "$(git status --porcelain)"
uname -a
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader

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
"$pip_bin" install \
  --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.0
"$pip_bin" install -e "$repo[data,dev]"
"$pip_bin" install transformers==5.14.1 packaging psutil ninja

cuda_arch="$($python_bin - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit("at least two visible CUDA GPUs are required")
capabilities = {torch.cuda.get_device_capability(index) for index in range(2)}
if capabilities != {(9, 0)}:
    raise SystemExit(f"the pinned smoke requires two sm_90 H100/H200 GPUs: {capabilities}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("BF16 is unavailable")
print("9.0")
PY
)"
echo "torch_cuda_arch_list=$cuda_arch"
MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="$cuda_arch" \
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
print(f"cuda_devices={torch.cuda.device_count()}")
if torch.cuda.device_count() < 2 or not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA/BF16 runtime gate failed")
if transformers.__version__ != "5.14.1":
    raise SystemExit("Transformers version changed")
if flash_attn.__version__ != "2.8.3":
    raise SystemExit("FlashAttention version changed")
PY

"$pip_bin" freeze > "$workspace/provenance/pip-freeze.txt"
git rev-parse HEAD > "$workspace/provenance/source-commit.txt"
nvidia-smi -q > "$workspace/provenance/nvidia-smi-q.txt"
df -h "$HOME" > "$workspace/provenance/disk.txt"
free -h > "$workspace/provenance/memory.txt"
echo "setup_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
touch "$workspace/setup.complete"
