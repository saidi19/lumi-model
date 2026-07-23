#!/usr/bin/env bash
set -euo pipefail

workspace="${LUMI_LEARNING_PILOT_WORKSPACE:-$HOME/lumi-runs/4b-learning-pilot-v0.1}"
repo="${LUMI_DISTRIBUTED_REPO:-$HOME/lumi-model}"
source_url="${LUMI_LEARNING_PILOT_SOURCE_SHARD_URL:?set presigned source-shard URL}"
final_manifest="${LUMI_LEARNING_PILOT_FINAL_MANIFEST:?set final manifest path}"
heldout_manifest="${LUMI_LEARNING_PILOT_HELDOUT_MANIFEST:?set held-out manifest path}"
tokenizer="${LUMI_LEARNING_PILOT_TOKENIZER:?set tokenizer path}"
python_bin="$repo/.venv/bin/python"
data="$workspace/data"
source_shard="$data/source-shard-00000.uint16.bin"
fixture="$data/training-fixture"

mkdir -p "$data"
exec > >(tee -a "$workspace/staging.log") 2>&1
cd "$repo"
if [[ -f "$workspace/staging.complete" ]]; then
  "$python_bin" - "$heldout_manifest" "$tokenizer" "$fixture/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

from lumi.evaluation_data import validate_heldout_manifest
from lumi.packed_data import validate_packed_manifest

heldout = validate_heldout_manifest(Path(sys.argv[1]))
tokenizer = Path(sys.argv[2])
for item in heldout["tokenizer"]["files"]:
    if not (tokenizer / item["name"]).is_file():
        raise SystemExit(f"missing tokenizer artifact: {item['name']}")
validate_packed_manifest(Path(sys.argv[3]), verify_hashes=True)
PY
  echo "staging_already_complete=true"
  exit 0
fi

echo "staging_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ ! -f "$source_shard" ]]; then
  curl --fail --location --retry 5 --retry-all-errors \
    --continue-at - \
    --output "$source_shard.partial" \
    "$source_url"
  mv "$source_shard.partial" "$source_shard"
fi
"$python_bin" scripts/prepare_4b_learning_pilot_data.py \
  --contract configs/4b-learning-pilot-v0.1.json \
  --final-manifest "$final_manifest" \
  --source-shard "$source_shard" \
  --output "$fixture"
"$python_bin" - "$heldout_manifest" "$tokenizer" <<'PY'
import hashlib
import sys
from pathlib import Path

from lumi.evaluation_data import validate_heldout_manifest

heldout = validate_heldout_manifest(Path(sys.argv[1]))
tokenizer = Path(sys.argv[2])
for item in heldout["tokenizer"]["files"]:
    path = tokenizer / item["name"]
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        raise SystemExit(f"tokenizer artifact changed: {item['name']}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise SystemExit(f"tokenizer artifact hash changed: {item['name']}")
PY
rm -f "$source_shard"
echo "staging_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
touch "$workspace/staging.complete"
