#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=".venv/bin/python"
RESULTS="data/derived/router_results.jsonl"
SUMMARY="data/derived/router_summary.json"
FIGDIR="figures/router"

rm -f "$RESULTS" "$SUMMARY"
rm -rf "$FIGDIR"

"$PY" - <<'PY'
from src.load_traces import load_config, load_traces
from src.simulate import execute_allocation

config = load_config("config/router_experiment.yaml")
traces = load_traces(config)
result = execute_allocation(
    "HumanEval/10",
    {"1.5b": 3, "7b": 0, "32b": 0},
    traces,
    order=("1.5b", "7b", "32b"),
)
expected = 1.985619834 + 0.190388417 + 1.349114292
assert result["solved"] is True, result
assert result["first_pass_model"] == "1.5b", result
assert abs(result["time_seconds"] - expected) < 1e-12, result
assert result["tokens"] == 147 + 9 + 80, result
print("execute_allocation unit check passed")
PY

"$PY" -m src.run_router_experiment \
  --config config/router_experiment.yaml \
  --mock-router \
  --folds 0 \
  --limit-test-problems 5 \
  --overwrite

LINES="$(wc -l < "$RESULTS" | tr -d ' ')"
if [[ "$LINES" != "5" ]]; then
  echo "Expected 5 router result lines, got $LINES" >&2
  exit 1
fi

"$PY" -m src.estimate_router --config config/router_experiment.yaml
"$PY" -m src.plot_router --config config/router_experiment.yaml

"$PY" - <<'PY'
import json
from pathlib import Path

summary = json.load(open("data/derived/router_summary.json"))
required = [
    "methods",
    "by_mode",
    "time_saved_vs_best_single",
    "router_gap_to_stat_oracle",
    "allocation_distribution",
]
missing = [key for key in required if key not in summary]
assert not missing, missing
for stem in [
    "fig_time_vs_accuracy",
    "fig_time_saved_by_mode",
    "fig_alloc_distribution",
    "fig_gap_to_oracle",
]:
    assert Path(f"figures/router/{stem}.png").exists(), stem
    assert Path(f"figures/router/{stem}.pdf").exists(), stem
print("router smoke assertions passed")
PY

echo "Router smoke test passed with MOCK router and no API calls."
