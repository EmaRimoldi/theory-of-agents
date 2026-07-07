#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=".venv/bin/python"
RESULTS="data/derived/bbh_router_results.jsonl"

TOTAL_CALLS="$("$PY" - <<'PY'
from src.load_traces import load_config, load_traces
try:
    traces = load_traces(load_config("config/router_experiment_bbh.yaml"))
    print(len(traces))
except Exception:
    print("unknown")
PY
)"
EXISTING_CALLS=0
if [[ -f "$RESULTS" ]]; then
  EXISTING_CALLS="$(wc -l < "$RESULTS" | tr -d ' ')"
fi

echo "BBH router full run"
echo "Estimated total router calls: $TOTAL_CALLS"
echo "Existing result rows: $EXISTING_CALLS"
if [[ "$TOTAL_CALLS" != "unknown" ]]; then
  remaining=$((TOTAL_CALLS - EXISTING_CALLS))
  if (( remaining < 0 )); then remaining=0; fi
  echo "Estimated remaining router calls: $remaining"
else
  echo "Worker logs not complete or not found; cannot compute remaining calls."
fi

if [[ -z "${ROUTER_MODEL:-}" ]]; then
  echo "Set ROUTER_MODEL before launching, e.g. ROUTER_MODEL=gpt-5.4" >&2
  exit 2
fi

echo "Router model: $ROUTER_MODEL"

"$PY" -m src.run_router_experiment --config config/router_experiment_bbh.yaml
"$PY" -m src.estimate_router --config config/router_experiment_bbh.yaml
"$PY" -m src.plot_router --config config/router_experiment_bbh.yaml
"$PY" -m src.frontier_by_category \
  --config config/router_experiment_bbh.yaml \
  --figure-stem fig_bbh_family_frontier
"$PY" -m src.frontier_by_category \
  --config config/router_experiment_bbh_raw.yaml \
  --figure-stem fig_bbh_subtask_frontier

echo "BBH router run complete."
echo "Summary: data/derived/bbh_router_summary.json"
echo "Family frontier: data/derived/bbh_family_frontier.json"
echo "Raw subtask frontier: data/derived/bbh_subtask_frontier.json"
