#!/usr/bin/env bash
set -euo pipefail

workspace="${LUMI_LEARNING_PILOT_WORKSPACE:-$HOME/lumi-runs/4b-learning-pilot-v0.1}"
source_commit="${LUMI_DISTRIBUTED_SOURCE_COMMIT:?set LUMI_DISTRIBUTED_SOURCE_COMMIT}"
world_size="${LUMI_LEARNING_PILOT_WORLD_SIZE:?set LUMI_LEARNING_PILOT_WORLD_SIZE}"
repo="${LUMI_DISTRIBUTED_REPO:-$HOME/lumi-model}"

if [[ "$world_size" != "4" && "$world_size" != "8" ]]; then
  echo "learning pilot world size must be 4 or 8" >&2
  exit 2
fi

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

cuda_arch="$("$python_bin" - "$world_size" <<'PY'
import sys

import torch

world_size = int(sys.argv[1])
if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
    raise SystemExit(f"{world_size} visible CUDA GPUs are required")
capabilities = {
    torch.cuda.get_device_capability(index)
    for index in range(world_size)
}
if capabilities != {(9, 0)}:
    raise SystemExit(f"the pinned pilot requires homogeneous sm_90 H100 GPUs: {capabilities}")
names = {torch.cuda.get_device_name(index) for index in range(world_size)}
if len(names) != 1 or any("H100" not in name.upper() for name in names):
    raise SystemExit(f"the pinned pilot requires homogeneous H100 GPUs: {names}")
memory_bytes = {
    torch.cuda.get_device_properties(index).total_memory
    for index in range(world_size)
}
if any(not 80_000_000_000 <= value <= 90_000_000_000 for value in memory_bytes):
    raise SystemExit(f"the pinned pilot requires 80 GB H100 GPUs: {memory_bytes}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("BF16 is unavailable")
print("9.0")
PY
)"
echo "torch_cuda_arch_list=$cuda_arch"
MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="$cuda_arch" \
  "$pip_bin" install flash-attn==2.8.3 --no-build-isolation
"$pip_bin" check

"$python_bin" - "$world_size" "$workspace" <<'PY'
import platform
import shutil
import sys

import flash_attn
import psutil
import torch
import transformers

world_size = int(sys.argv[1])
print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print(f"flash_attn={flash_attn.__version__}")
print(f"cuda_devices={torch.cuda.device_count()}")
if torch.cuda.device_count() < world_size or not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA/BF16 runtime gate failed")
if transformers.__version__ != "5.14.1":
    raise SystemExit("Transformers version changed")
if flash_attn.__version__ != "2.8.3":
    raise SystemExit("FlashAttention version changed")
host_memory = psutil.virtual_memory().total
free_disk = shutil.disk_usage(sys.argv[2]).free
print(f"host_memory_bytes={host_memory}")
print(f"workspace_free_disk_bytes={free_disk}")
if host_memory < 128_000_000_000:
    raise SystemExit("the learning pilot requires at least 128 GB host RAM")
if free_disk < 300_000_000_000:
    raise SystemExit("the learning pilot requires at least 300 GB free disk")
PY

"$pip_bin" freeze > "$workspace/provenance/pip-freeze.txt"
git rev-parse HEAD > "$workspace/provenance/source-commit.txt"
nvidia-smi -q > "$workspace/provenance/nvidia-smi-q.txt"
df -h "$HOME" > "$workspace/provenance/disk.txt"
free -h > "$workspace/provenance/memory.txt"
echo "setup_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
touch "$workspace/setup.complete"
