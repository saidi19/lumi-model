#!/usr/bin/env bash
set -euo pipefail

workspace="${LUMI_1B_ISOLATED_WORKSPACE:-$HOME/lumi-runs/lumi-1b-isolated-1p4b-v0.1}"
authorization="${LUMI_1B_AUTHORIZATION:-}"
source_archive="${LUMI_SOURCE_ARCHIVE:-}"
repo="$HOME/lumi-model"
plan="$repo/configs/isolated-scaling-run-1b-v0.1.json"

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

# This intentionally uses only the standard library before installing anything.
python3 - "$authorization" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("status") != "owner_approved":
    raise SystemExit("launch authorization is not owner-approved")
if value.get("full_4b_training") is not False:
    raise SystemExit("1B authorization cannot approve 4B training")
if len(value.get("source_commit", "")) != 40:
    raise SystemExit("launch authorization has no source commit")
if len(value.get("source_archive_sha256", "")) != 64:
    raise SystemExit("launch authorization has no source archive hash")
PY

mkdir -p "$workspace/provenance"
exec > >(tee -a "$workspace/setup.log") 2>&1

cd "$repo"
source_commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "$authorization")"
expected_source_archive_sha256="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_archive_sha256"])' "$authorization")"
actual_source_archive_sha256="$(sha256sum "$source_archive" | cut -d ' ' -f 1)"
if [[ "$actual_source_archive_sha256" != "$expected_source_archive_sha256" ]]; then
  echo "ERROR: source archive SHA-256 does not match authorization" >&2
  exit 2
fi
verify_source_archive "$source_archive"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [[ "$(git rev-parse HEAD)" != "$source_commit" ]]; then
    echo "ERROR: source repository commit does not match authorization" >&2
    exit 2
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: source repository is not clean" >&2
    exit 2
  fi
fi

if [[ -f "$workspace/setup.complete" ]]; then
  python_bin="$repo/.venv/bin/python"
  test -x "$python_bin"
  "$python_bin" scripts/check_scaling_launch_authorization.py \
    --plan "$plan" \
    --authorization "$authorization" \
    --source-commit "$source_commit" \
    --output "$workspace/authorization-verified.json"
  echo "setup_already_complete=true"
  exit 0
fi

echo "setup_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "source_commit=$source_commit"
uname -m
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

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
MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0 \
  "$pip_bin" install flash-attn==2.8.3 --no-build-isolation
"$pip_bin" check

"$python_bin" scripts/check_scaling_launch_authorization.py \
  --plan "$plan" \
  --authorization "$authorization" \
  --source-commit "$source_commit" \
  --output "$workspace/authorization-verified.json"
"$python_bin" scripts/plan_isolated_scaling_run.py \
  --plan "$plan" \
  --hourly-rate-usd "$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["hourly_rate_usd"])' "$workspace/authorization-verified.json")" \
  --output "$workspace/cost-projection.json"

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
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git status --short > "$workspace/provenance/git-status.txt"
fi
printf '%s\n' "$source_commit" > "$workspace/provenance/source-commit.txt"
printf '%s  %s\n' "$actual_source_archive_sha256" "$(basename "$source_archive")" \
  > "$workspace/provenance/source-archive.sha256"
sha256sum "$plan" > "$workspace/provenance/run-plan.sha256"
echo "setup_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
touch "$workspace/setup.complete"
