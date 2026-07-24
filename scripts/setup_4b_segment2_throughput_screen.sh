#!/usr/bin/env bash
set -euo pipefail

repo="${LUMI_SEGMENT2_SCREEN_REPO:-$HOME/lumi-segment2-screen/repo}"
workspace="${LUMI_SEGMENT2_SCREEN_WORKSPACE:-$HOME/lumi-segment2-screen/workspace}"
source_commit="${LUMI_SEGMENT2_SCREEN_SOURCE_COMMIT:?set exact source commit}"
mkdir -p "$workspace/provenance"
exec > >(tee -a "$workspace/setup.log") 2>&1

if [[ -f "$workspace/setup.complete" ]]; then
  echo "setup_already_complete=true"
  exit 0
fi
test "$(cat "$repo/source-commit.txt")" = "$source_commit"
echo "setup_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
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
"$pip_bin" install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.0
"$pip_bin" install -e "$repo[data]"
"$pip_bin" install transformers==5.14.1 packaging psutil ninja
MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=9.0 \
  "$pip_bin" install flash-attn==2.8.3 --no-build-isolation
"$pip_bin" check

"$python_bin" - <<'PY'
import flash_attn
import torch
import transformers

if torch.cuda.device_count() != 8:
    raise SystemExit("exactly eight visible GPUs are required")
if {torch.cuda.get_device_capability(index) for index in range(8)} != {(9, 0)}:
    raise SystemExit("the screen requires homogeneous sm_90 GPUs")
if any("H100" not in torch.cuda.get_device_name(index).upper() for index in range(8)):
    raise SystemExit("the screen requires H100 GPUs")
if flash_attn.__version__ != "2.8.3":
    raise SystemExit("FlashAttention version changed")
if transformers.__version__ != "5.14.1":
    raise SystemExit("Transformers version changed")
print(f"torch={torch.__version__}")
print(f"flash_attn={flash_attn.__version__}")
print(f"transformers={transformers.__version__}")
PY

"$pip_bin" freeze > "$workspace/provenance/pip-freeze.txt"
nvidia-smi -q > "$workspace/provenance/nvidia-smi-q.txt"
echo "setup_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
touch "$workspace/setup.complete"
