#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=".venv/bin/python"
RUN_ID="smoke_mbpp"
OUT="data/raw/runs_qwen2.5-coder_1.5b_${RUN_ID}.jsonl"

rm -f "$OUT"

"$PY" -m src.run_eval \
  --config config/experiment_mbpp.yaml \
  --run-id "$RUN_ID" \
  --models qwen2.5-coder:1.5b \
  --limit-problems 5 \
  --max-attempts 2

LINES="$(wc -l < "$OUT" | tr -d ' ')"
if [[ "$LINES" != "5" ]]; then
  echo "Expected 5 MBPP+ smoke rows, got $LINES" >&2
  exit 1
fi

"$PY" - <<'PY'
import json
from pathlib import Path

path = Path("data/raw/runs_qwen2.5-coder_1.5b_smoke_mbpp.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
required = {
    "attempt_seconds",
    "attempt_token_counts",
    "attempt_statuses",
    "tau_tokens",
    "tau_seconds",
    "n_attempts",
    "solved",
    "checker_dataset",
    "dataset",
}
for row in rows:
    missing = required - set(row)
    assert not missing, missing
    n = int(row["n_attempts"])
    assert len(row["attempt_seconds"]) == n, row
    assert len(row["attempt_token_counts"]) == n, row
    assert len(row["attempt_statuses"]) == n, row
    assert row["dataset"] == "mbppplus", row["dataset"]
    assert row["checker_dataset"] == "mbpp", row["checker_dataset"]
print("MBPP+ worker smoke schema check passed")
PY

echo "MBPP+ worker smoke passed: $OUT"
