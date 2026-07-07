#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=".venv/bin/python"
CATEGORY_CONFIG="config/router_experiment_mbpp_category.yaml"
MATHNOT_CONFIG="config/router_experiment_mbpp_mathnot.yaml"

if [[ -z "${ROUTER_MODEL:-}" ]]; then
  echo "Set ROUTER_MODEL to the Codex subscription model before launching, e.g. ROUTER_MODEL=gpt-5.4" >&2
  exit 2
fi

N_PROBLEMS="$("$PY" - <<'PY'
from src.mbpp_data import load_mbpp_plus
problems, _hash = load_mbpp_plus()
print(len(problems))
PY
)"

TAGGED=0
if [[ -f data/derived/mbpp_modes_category.json ]]; then
  TAGGED="$("$PY" - <<'PY'
import json
try:
    print(len(json.load(open("data/derived/mbpp_modes_category.json"))))
except FileNotFoundError:
    print(0)
PY
)"
elif [[ -f data/derived/mbpp_category_label_records.jsonl ]]; then
  TAGGED="$("$PY" - <<'PY'
import json
seen=set()
try:
    with open("data/derived/mbpp_category_label_records.jsonl") as f:
        for line in f:
            if line.strip():
                seen.add(json.loads(line)["task_id"])
except FileNotFoundError:
    pass
print(len(seen))
PY
)"
fi
REMAINING_TAG_CALLS=$((N_PROBLEMS - TAGGED))
if (( REMAINING_TAG_CALLS < 0 )); then
  REMAINING_TAG_CALLS=0
fi

archive_mock_results() {
  local path="$1"
  local label="$2"
  if [[ -f "$path" ]] && grep -q '"router_backend": "mock"' "$path"; then
    local archive="data/derived/${label}_smoke_$(date -u +%Y%m%dT%H%M%SZ).jsonl"
    echo "Existing mock smoke results detected at $path; archiving to $archive"
    mv "$path" "$archive"
  fi
}

archive_mock_results data/derived/mbpp_router_results_category.jsonl mbpp_router_results_category
archive_mock_results data/derived/mbpp_router_results_mathnot.jsonl mbpp_router_results_mathnot

CATEGORY_EXISTING=0
MATHNOT_EXISTING=0
if [[ -f data/derived/mbpp_router_results_category.jsonl ]]; then
  CATEGORY_EXISTING="$(wc -l < data/derived/mbpp_router_results_category.jsonl | tr -d ' ')"
fi
if [[ -f data/derived/mbpp_router_results_mathnot.jsonl ]]; then
  MATHNOT_EXISTING="$(wc -l < data/derived/mbpp_router_results_mathnot.jsonl | tr -d ' ')"
fi
CATEGORY_REMAINING=$((N_PROBLEMS - CATEGORY_EXISTING))
MATHNOT_REMAINING=$((N_PROBLEMS - MATHNOT_EXISTING))
if (( CATEGORY_REMAINING < 0 )); then CATEGORY_REMAINING=0; fi
if (( MATHNOT_REMAINING < 0 )); then MATHNOT_REMAINING=0; fi
TOTAL_LLM_CALLS=$((REMAINING_TAG_CALLS + CATEGORY_REMAINING + MATHNOT_REMAINING))

echo "MBPP+ category extension full run"
echo "Router/category model: $ROUTER_MODEL"
echo "Temperature: 0"
echo "MBPP+ problems: $N_PROBLEMS"
echo "Remaining tag calls: $REMAINING_TAG_CALLS"
echo "Remaining 4-category router calls: $CATEGORY_REMAINING"
echo "Remaining math/nonmath router calls: $MATHNOT_REMAINING"
echo "Estimated remaining LLM calls: $TOTAL_LLM_CALLS"

"$PY" -m src.tag_categories \
  --model-env ROUTER_MODEL \
  --reasoning-effort medium \
  --records-path data/derived/mbpp_category_label_records.jsonl \
  --modes-path data/derived/mbpp_modes_category.json \
  --meta-path data/derived/mbpp_modes_category_meta.json \
  --sample-path data/derived/category_label_sample.json \
  --mathnot-path data/derived/mbpp_modes_mathnot.json \
  --mathnot-meta-path data/derived/mbpp_modes_mathnot_meta.json

"$PY" -m src.run_router_experiment --config "$CATEGORY_CONFIG"
"$PY" -m src.estimate_router --config "$CATEGORY_CONFIG"
"$PY" -m src.plot_router --config "$CATEGORY_CONFIG"
"$PY" -m src.frontier_by_category --config "$CATEGORY_CONFIG"

"$PY" -m src.run_router_experiment --config "$MATHNOT_CONFIG"
"$PY" -m src.estimate_router --config "$MATHNOT_CONFIG"
"$PY" -m src.plot_router --config "$MATHNOT_CONFIG"
"$PY" -m src.frontier_by_category \
  --config "$MATHNOT_CONFIG" \
  --output-path data/derived/category_frontier_mathnot.json \
  --figures-dir figures/router_mbpp_mathnot \
  --figure-stem fig_category_frontier_mathnot

echo "MBPP+ category extension full run complete."
