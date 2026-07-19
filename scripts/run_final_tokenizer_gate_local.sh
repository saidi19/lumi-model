#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="${LUMI_PYTHON_BIN:-/private/tmp/lumi-model/.venv/bin/python}"
sample_directory="${LUMI_TOKENIZER_SAMPLE_DIR:-/Users/afnan/Documents/lumi-data/tokenizer-sample-v0.3}"
baseline_tokenizer="${LUMI_BASELINE_TOKENIZER:-/Users/afnan/Documents/lumi-data/tokenizer-candidates-v0.2/lumi-tokenizer-65k}"
run_root="${LUMI_TOKENIZER_GATE_ROOT:-/Users/afnan/Documents/lumi-data/tokenizer-final-ab-v0.1}"
candidate_tokenizer="$run_root/digit-1-3-65k"
comparison="$run_root/comparison.json"
reserved_audit="$run_root/reserved-token-audit.json"
decision="$run_root/decision.json"
config="$repository_root/configs/tokenizer-final-ab-v0.1.json"

mkdir -p "$run_root"
exec > >(tee -a "$run_root/run.log") 2>&1
cd "$repository_root"

printf 'run_started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'source_commit=%s\n' "$(git rev-parse HEAD)"
printf 'sample_directory=%s\n' "$sample_directory"
printf 'baseline_tokenizer=%s\n' "$baseline_tokenizer"
printf 'candidate_tokenizer=%s\n' "$candidate_tokenizer"

if [[ ! -f "$run_root/training.complete" ]]; then
  printf 'stage=training status=started utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ ! -d "$candidate_tokenizer" ]]; then
    PYTHONPATH=src "$python_bin" scripts/train_tokenizer.py \
      --input "$sample_directory/sample.jsonl" \
      --output "$candidate_tokenizer" \
      --digit-chunk-size 3 \
      --no-progress
  fi
  PYTHONPATH=src "$python_bin" scripts/check_tokenizer_contract.py \
    "$candidate_tokenizer" \
    --expected-digit-chunk-size 3
  touch "$run_root/training.complete"
  printf 'stage=training status=complete utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

if [[ ! -f "$run_root/reserved-audit.complete" ]]; then
  printf 'stage=reserved_audit status=started utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ ! -f "$reserved_audit" ]]; then
    PYTHONPATH=src "$python_bin" scripts/audit_tokenizer_reserved_tokens.py \
      --sample "$sample_directory" \
      --output "$reserved_audit"
  fi
  touch "$run_root/reserved-audit.complete"
  printf 'stage=reserved_audit status=complete utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

if [[ ! -f "$run_root/comparison.complete" ]]; then
  printf 'stage=comparison status=started utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ ! -f "$comparison" ]]; then
    PYTHONPATH=src "$python_bin" scripts/compare_tokenizers.py \
      --sample "$sample_directory" \
      --tokenizer "current_65k=$baseline_tokenizer" \
      --tokenizer "digit_1_3_65k=$candidate_tokenizer" \
      --output "$comparison" \
      --batch-size 256
  fi
  touch "$run_root/comparison.complete"
  printf 'stage=comparison status=complete utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

if [[ ! -f "$run_root/decision.complete" ]]; then
  printf 'stage=decision status=started utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ ! -f "$decision" ]]; then
    PYTHONPATH=src "$python_bin" scripts/decide_tokenizer_gate.py \
      --config "$config" \
      --comparison "$comparison" \
      --reserved-audit "$reserved_audit" \
      --output "$decision" \
      --report-only
  fi
  touch "$run_root/decision.complete"
  printf 'stage=decision status=complete utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

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
