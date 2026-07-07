#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=".venv/bin/python"
RUN_ID="smoke"
MODEL="qwen2.5-coder:1.5b"
RAW_FILE="data/raw/runs_qwen2.5-coder_1.5b_${RUN_ID}.jsonl"

if [[ ! -x "$PY" ]]; then
  echo "Missing .venv. Run: uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt" >&2
  exit 1
fi

rm -f "$RAW_FILE"

"$PY" -m src.run_eval \
  --config config/experiment.yaml \
  --run-id "$RUN_ID" \
  --models "$MODEL" \
  --limit-problems 5 \
  --max-attempts 2

"$PY" -m src.estimate --config config/experiment.yaml --run-id "$RUN_ID"
"$PY" -m src.plot --config config/experiment.yaml

lines="$(wc -l < "$RAW_FILE" | tr -d ' ')"
if [[ "$lines" != "5" ]]; then
  echo "Expected 5 raw rows in $RAW_FILE, found $lines" >&2
  exit 1
fi

"$PY" - <<'PY'
import json
from pathlib import Path

tau = json.loads(Path("data/derived/tau_star.json").read_text())
assert tau, "tau_star.json has no cells"
for path in [
    "figures/fig_frontier.png",
    "figures/fig_frontier.pdf",
    "figures/fig_disagreement.png",
    "figures/fig_disagreement.pdf",
    "figures/fig_success_curves.png",
    "figures/fig_success_curves.pdf",
]:
    assert Path(path).exists(), f"missing {path}"
print("Smoke assertions passed.")
PY

echo "Smoke test passed."
