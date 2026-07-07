#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=".venv/bin/python"
RUN_ID_FILE="data/raw/.full_run_mbpp_2models_run_id"
mkdir -p data/raw
if [[ -n "${RUN_ID:-}" ]]; then
  RUN_ID_SOURCE="environment"
elif [[ -f "$RUN_ID_FILE" ]]; then
  RUN_ID="$(< "$RUN_ID_FILE")"
  RUN_ID_SOURCE="$RUN_ID_FILE"
else
  RUN_ID="full_mbpp_2models_$(date -u +%Y%m%dT%H%M%SZ)"
  printf '%s\n' "$RUN_ID" > "$RUN_ID_FILE"
  RUN_ID_SOURCE="$RUN_ID_FILE"
fi
LOG="data/raw/full_run_${RUN_ID}.log"
LOG_15B="data/raw/full_run_${RUN_ID}_1.5b.log"
LOG_7B="data/raw/full_run_${RUN_ID}_7b.log"
N_PROBLEMS="$("$PY" - <<'PY'
from evalplus.data import get_mbpp_plus
print(len(get_mbpp_plus()))
PY
)"

EXPECTED_SECONDS_PER_PROBLEM_15B="${EXPECTED_SECONDS_PER_PROBLEM_15B:-35}"
EXPECTED_SECONDS_PER_PROBLEM_7B="${EXPECTED_SECONDS_PER_PROBLEM_7B:-90}"
export N_PROBLEMS EXPECTED_SECONDS_PER_PROBLEM_15B EXPECTED_SECONDS_PER_PROBLEM_7B
ETA_SECONDS="$("$PY" - <<'PY'
import os
n = int(os.environ["N_PROBLEMS"])
eta = n * (
    float(os.environ["EXPECTED_SECONDS_PER_PROBLEM_15B"])
    + float(os.environ["EXPECTED_SECONDS_PER_PROBLEM_7B"])
)
print(int(eta))
PY
)"

echo "MBPP+ two-model worker run"
echo "Run id: $RUN_ID"
echo "Run id source: $RUN_ID_SOURCE"
echo "Problems: $N_PROBLEMS"
echo "Models: qwen2.5-coder:1.5b, qwen2.5-coder:7b"
echo "Rough ETA: $((ETA_SECONDS / 3600))h $(((ETA_SECONDS % 3600) / 60))m"
echo "ETA assumptions: ${EXPECTED_SECONDS_PER_PROBLEM_15B}s/problem for 1.5B, ${EXPECTED_SECONDS_PER_PROBLEM_7B}s/problem for 7B"
echo "Log: $LOG"
echo "1.5B log: $LOG_15B"
echo "7B log: $LOG_7B"
echo "Resumability: rerunning this script reuses $RUN_ID_FILE and skips already logged tasks."

{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting two-model MBPP+ worker run"
  echo "run_id=$RUN_ID"
} | tee "$LOG"

(
  "$PY" -m src.run_eval \
    --config config/experiment_mbpp_2models.yaml \
    --run-id "$RUN_ID" \
    --models qwen2.5-coder:1.5b
) 2>&1 | tee "$LOG_15B" &
PID_15B=$!

(
  "$PY" -m src.run_eval \
    --config config/experiment_mbpp_2models.yaml \
    --run-id "$RUN_ID" \
    --models qwen2.5-coder:7b
) 2>&1 | tee "$LOG_7B" &
PID_7B=$!

STATUS=0
if ! wait "$PID_15B"; then
  STATUS=1
  echo "1.5B worker failed" | tee -a "$LOG"
fi
if ! wait "$PID_7B"; then
  STATUS=1
  echo "7B worker failed" | tee -a "$LOG"
fi

if [[ "$STATUS" -eq 0 ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] two-model MBPP+ worker run complete" | tee -a "$LOG"
else
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] two-model MBPP+ worker run failed" | tee -a "$LOG"
fi
exit "$STATUS"
