#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=".venv/bin/python"
MODEL="qwen2.5:1.5b"
RUN_ID="smoke_bbh"
RAW_FILE="data/raw/runs_qwen2.5_1.5b_${RUN_ID}.jsonl"
RESULTS="data/derived/bbh_router_results_smoke.jsonl"
SUMMARY="data/derived/bbh_router_summary_smoke.json"
FIGDIR="figures/router_bbh_smoke"
TASK_IDS="BBH/boolean_expressions/000,BBH/boolean_expressions/001,BBH/multistep_arithmetic_two/000,BBH/multistep_arithmetic_two/001,BBH/dyck_languages/000"

if [[ ! -x "$PY" ]]; then
  echo "Missing .venv. Run: uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt" >&2
  exit 1
fi

"$PY" -m src.dataset --config config/experiment_bbh.yaml

if ! "$PY" - <<'PY'
from src.models import resolve_model_info
resolve_model_info("qwen2.5:1.5b")
PY
then
  cat >&2 <<'EOF'
Missing BBH generalist worker model. Pull it, then rerun:
  ollama pull qwen2.5:1.5b

For the full BBH worker run, also pull:
  ollama pull qwen2.5:7b
  ollama pull qwen2.5:32b
EOF
  exit 2
fi

rm -f "$RAW_FILE" "$RESULTS" "$SUMMARY" data/derived/bbh_retry_sanity_smoke.json data/derived/bbh_folds_smoke.json
rm -rf "$FIGDIR"

"$PY" -m src.run_eval \
  --config config/experiment_bbh.yaml \
  --run-id "$RUN_ID" \
  --models "$MODEL" \
  --task-ids "$TASK_IDS" \
  --max-attempts 2

lines="$(wc -l < "$RAW_FILE" | tr -d ' ')"
if [[ "$lines" != "5" ]]; then
  echo "Expected 5 BBH smoke rows in $RAW_FILE, found $lines" >&2
  exit 1
fi

"$PY" - <<'PY'
import json
from pathlib import Path

raw = Path("data/raw/runs_qwen2.5_1.5b_smoke_bbh.jsonl")
rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
required = {"attempt_seconds", "attempt_token_counts", "attempt_statuses", "attempt_extracted_answers"}
for row in rows:
    missing = required - set(row)
    assert not missing, (row["task_id"], missing)
    assert len(row["attempt_seconds"]) == len(row["attempt_token_counts"]) == len(row["attempt_statuses"]) == row["n_attempts"]
    assert row["checker_dataset"] == "bbh"
    assert row["answer_extraction_rule"]
assert Path("data/derived/bbh_modes.json").exists()
assert Path("data/derived/bbh_modes_raw.json").exists()
assert Path("data/derived/bbh_mode_groups.json").exists()
print("BBH worker smoke schema assertions passed")
PY

"$PY" -m src.bbh_retry_sanity \
  --raw-file "$RAW_FILE" \
  --max-attempts 2 \
  --require-rise

"$PY" -m src.run_router_experiment \
  --config config/router_experiment_bbh_smoke.yaml \
  --mock-router \
  --folds 0 \
  --limit-test-problems 5 \
  --overwrite

router_lines="$(wc -l < "$RESULTS" | tr -d ' ')"
if [[ "$router_lines" != "5" ]]; then
  echo "Expected 5 BBH router smoke rows in $RESULTS, found $router_lines" >&2
  exit 1
fi

"$PY" -m src.estimate_router --config config/router_experiment_bbh_smoke.yaml
"$PY" -m src.plot_router --config config/router_experiment_bbh_smoke.yaml

"$PY" - <<'PY'
from pathlib import Path
for stem in [
    "fig_time_vs_accuracy",
    "fig_time_saved_by_mode",
    "fig_alloc_distribution",
    "fig_gap_to_oracle",
]:
    assert Path(f"figures/router_bbh_smoke/{stem}.png").exists(), stem
print("BBH router smoke assertions passed")
PY

echo "BBH smoke test passed with 1.5B worker and MOCK router."
