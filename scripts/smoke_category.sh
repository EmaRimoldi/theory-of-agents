#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=".venv/bin/python"
CONFIG="config/router_experiment_mbpp_category_smoke.yaml"

rm -f \
  data/derived/mbpp_category_label_records_smoke.jsonl \
  data/derived/mbpp_modes_category_smoke.json \
  data/derived/mbpp_modes_category_meta_smoke.json \
  data/derived/category_label_sample_smoke.json \
  data/derived/mbpp_modes_mathnot_smoke.json \
  data/derived/mbpp_modes_mathnot_meta_smoke.json \
  data/derived/mbpp_folds_category_smoke.json \
  data/derived/mbpp_router_results_category_smoke.jsonl \
  data/derived/mbpp_router_summary_category_smoke.json \
  data/derived/category_frontier_smoke.json
rm -rf figures/router_mbpp_category_smoke

"$PY" -m src.tag_categories \
  --mock \
  --limit 5 \
  --records-path data/derived/mbpp_category_label_records_smoke.jsonl \
  --modes-path data/derived/mbpp_modes_category_smoke.json \
  --meta-path data/derived/mbpp_modes_category_meta_smoke.json \
  --sample-path data/derived/category_label_sample_smoke.json \
  --mathnot-path data/derived/mbpp_modes_mathnot_smoke.json \
  --mathnot-meta-path data/derived/mbpp_modes_mathnot_meta_smoke.json

"$PY" - <<'PY'
import json
from pathlib import Path

modes = json.load(open("data/derived/mbpp_modes_category_smoke.json"))
assert len(modes) == 5, len(modes)
assert set(modes.values()) <= {"string", "math", "list_ds", "logic_control"}
assert Path("data/derived/category_label_sample_smoke.json").exists()
assert Path("data/derived/mbpp_modes_mathnot_smoke.json").exists()
print("category tagger smoke outputs written")
PY

MISSING_LOGS=()
for path in \
  "data/raw/runs_qwen2.5-coder_1.5b_full_mbpp_2models_20260617T114444Z.jsonl" \
  "data/raw/runs_qwen2.5-coder_7b_full_mbpp_2models_20260617T114444Z.jsonl" \
  "data/raw/runs_qwen2.5-coder_32b_full_mbpp_32b_20260617T1335Z.jsonl"; do
  if [[ ! -f "$path" ]]; then
    MISSING_LOGS+=("$path")
  fi
done

if (( ${#MISSING_LOGS[@]} > 0 )); then
  echo "Category tagger smoke passed, but router/frontier smoke cannot run." >&2
  echo "Missing required MBPP+ worker logs:" >&2
  printf '  %s\n' "${MISSING_LOGS[@]}" >&2
  echo "No worker model was re-run and no worker traces were fabricated." >&2
  exit 3
fi

"$PY" -m src.run_router_experiment \
  --config "$CONFIG" \
  --mock-router \
  --folds 0 \
  --limit-test-problems 5 \
  --overwrite

LINES="$(wc -l < data/derived/mbpp_router_results_category_smoke.jsonl | tr -d ' ')"
if [[ "$LINES" != "5" ]]; then
  echo "Expected 5 category router smoke rows, got $LINES" >&2
  exit 1
fi

"$PY" -m src.estimate_router --config "$CONFIG"
"$PY" -m src.plot_router --config "$CONFIG"
"$PY" -m src.frontier_by_category --config "$CONFIG" --bootstrap-B 100

"$PY" - <<'PY'
from pathlib import Path

assert Path("data/derived/category_frontier_smoke.json").exists()
assert Path("figures/router_mbpp_category_smoke/fig_category_frontier.png").exists()
for stem in [
    "fig_time_vs_accuracy",
    "fig_time_saved_by_mode",
    "fig_alloc_distribution",
    "fig_gap_to_oracle",
]:
    assert Path(f"figures/router_mbpp_category_smoke/{stem}.png").exists(), stem
print("category router/frontier smoke assertions passed")
PY

echo "MBPP+ category smoke passed with MOCK router and no worker re-runs."
