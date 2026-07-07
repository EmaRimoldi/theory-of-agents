#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=".venv/bin/python"
RUN_ID_FILE="data/raw/.full_run_bbh_run_id"

if [[ ! -x "$PY" ]]; then
  echo "Missing .venv. Run: uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt" >&2
  exit 1
fi

mkdir -p data/raw
if [[ -n "${RUN_ID:-}" ]]; then
  RUN_ID_SOURCE="environment"
elif [[ -f "$RUN_ID_FILE" ]]; then
  RUN_ID="$(< "$RUN_ID_FILE")"
  RUN_ID_SOURCE="$RUN_ID_FILE"
else
  RUN_ID="full_bbh_$(date -u +%Y%m%dT%H%M%SZ)"
  printf "%s\n" "$RUN_ID" > "$RUN_ID_FILE"
  RUN_ID_SOURCE="$RUN_ID_FILE"
fi

"$PY" -m src.dataset --config config/experiment_bbh.yaml

if ! "$PY" - <<'PY'
from src.models import resolve_model_info
for model in ["qwen2.5:1.5b", "qwen2.5:7b", "qwen2.5:32b"]:
    info = resolve_model_info(model)
    print(f"{model} digest={info.digest} quantization={info.quantization}")
PY
then
  cat >&2 <<'EOF'
Missing one or more BBH generalist worker models. Pull them, then rerun:
  ollama pull qwen2.5:1.5b
  ollama pull qwen2.5:7b
  ollama pull qwen2.5:32b
EOF
  exit 2
fi

N_EXAMPLES="$("$PY" - <<'PY'
from src.dataset import load_config, load_dataset
bundle = load_dataset(load_config("config/experiment_bbh.yaml"))
print(len(bundle.problems))
PY
)"

EXPECTED_SECONDS_PER_EXAMPLE_1_5B="${EXPECTED_SECONDS_PER_EXAMPLE_1_5B:-8}"
EXPECTED_SECONDS_PER_EXAMPLE_7B="${EXPECTED_SECONDS_PER_EXAMPLE_7B:-22}"
EXPECTED_SECONDS_PER_EXAMPLE_32B="${EXPECTED_SECONDS_PER_EXAMPLE_32B:-90}"
ETA_SECONDS="$("$PY" - <<PY
n = int("$N_EXAMPLES")
eta = n * (float("$EXPECTED_SECONDS_PER_EXAMPLE_1_5B") + float("$EXPECTED_SECONDS_PER_EXAMPLE_7B") + float("$EXPECTED_SECONDS_PER_EXAMPLE_32B"))
print(int(eta))
PY
)"

echo "BBH full worker run"
echo "Run id: $RUN_ID"
echo "Run id source: $RUN_ID_SOURCE"
echo "Examples: $N_EXAMPLES"
echo "Models: qwen2.5:1.5b, qwen2.5:7b, qwen2.5:32b"
echo "Rough ETA: $((ETA_SECONDS / 3600))h $(((ETA_SECONDS % 3600) / 60))m"
echo "32B is the bottleneck. Recommended launch: caffeinate -i scripts/full_run_bbh.sh"
echo "Execution: sequential workers (1.5B, 7B, 32B) to avoid local Ollama/Metal contention"

LOG="data/raw/full_run_${RUN_ID}.log"
LOG_15B="data/raw/full_run_${RUN_ID}_1.5b.log"
LOG_7B="data/raw/full_run_${RUN_ID}_7b.log"
LOG_32B="data/raw/full_run_${RUN_ID}_32b.log"

{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting BBH worker run"
  echo "run_id=$RUN_ID"
  echo "examples=$N_EXAMPLES"
  echo "log_1.5b=$LOG_15B"
  echo "log_7b=$LOG_7B"
  echo "log_32b=$LOG_32B"
} | tee "$LOG"

STATUS=0
run_one() {
  local model="$1"
  local log_file="$2"
  local label="$3"

  for other in qwen2.5:1.5b qwen2.5:7b qwen2.5:32b; do
    if [[ "$other" != "$model" ]]; then
      ollama stop "$other" >/dev/null 2>&1 || true
    fi
  done

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting $label sequential worker" | tee -a "$LOG"
  if ! (
    PYTHONUNBUFFERED=1 "$PY" -m src.run_eval \
      --config config/experiment_bbh.yaml \
      --run-id "$RUN_ID" \
      --models "$model"
  ) 2>&1 | tee "$log_file"; then
    STATUS=1
    echo "$label BBH worker failed" | tee -a "$LOG"
  fi
}

run_one qwen2.5:1.5b "$LOG_15B" "1.5B"
run_one qwen2.5:7b "$LOG_7B" "7B"
run_one qwen2.5:32b "$LOG_32B" "32B"

if [[ "$STATUS" -eq 0 ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] BBH worker run complete" | tee -a "$LOG"
else
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] BBH worker run failed" | tee -a "$LOG"
fi
exit "$STATUS"
