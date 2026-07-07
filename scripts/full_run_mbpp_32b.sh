#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=".venv/bin/python"
RUN_ID_FILE="data/raw/.full_run_mbpp_32b_run_id"
DEFAULT_RUN_ID="full_mbpp_32b_20260617T1335Z"
mkdir -p data/raw

if [[ -n "${RUN_ID:-}" ]]; then
  RUN_ID_SOURCE="environment"
elif [[ -f "$RUN_ID_FILE" ]]; then
  RUN_ID="$(< "$RUN_ID_FILE")"
  RUN_ID_SOURCE="$RUN_ID_FILE"
else
  RUN_ID="$DEFAULT_RUN_ID"
  printf '%s\n' "$RUN_ID" > "$RUN_ID_FILE"
  RUN_ID_SOURCE="$RUN_ID_FILE"
fi

LOG="data/raw/full_run_${RUN_ID}.log"
OUT="data/raw/runs_qwen2.5-coder_32b_${RUN_ID}.jsonl"

N_PROBLEMS="$("$PY" - <<'PY'
from evalplus.data import get_mbpp_plus
print(len(get_mbpp_plus()))
PY
)"
EXPECTED_SECONDS_PER_PROBLEM_32B="${EXPECTED_SECONDS_PER_PROBLEM_32B:-360}"
export N_PROBLEMS EXPECTED_SECONDS_PER_PROBLEM_32B
ETA_SECONDS="$("$PY" - <<'PY'
import os
n = int(os.environ["N_PROBLEMS"])
eta = n * float(os.environ.get("EXPECTED_SECONDS_PER_PROBLEM_32B", "360"))
print(int(eta))
PY
)"

echo "MBPP+ 32B-only worker run"
echo "Run id: $RUN_ID"
echo "Run id source: $RUN_ID_SOURCE"
echo "Problems: $N_PROBLEMS"
echo "Model: qwen2.5-coder:32b"
echo "Output: $OUT"
echo "Log: $LOG"
echo "Rough ETA: $((ETA_SECONDS / 3600))h $(((ETA_SECONDS % 3600) / 60))m"
echo "Resumability: rerunning this script reuses $RUN_ID_FILE and skips already logged tasks."

"$PY" -m src.run_eval \
  --config config/experiment_mbpp.yaml \
  --run-id "$RUN_ID" \
  --models qwen2.5-coder:32b \
  2>&1 | tee "$LOG"
