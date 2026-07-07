#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=".venv/bin/python"
RESULTS="data/derived/router_results.jsonl"
TOTAL_CALLS=164
EXISTING_CALLS=0

if [[ -f "$RESULTS" ]] && grep -q '"router_backend": "mock"' "$RESULTS"; then
  ARCHIVE="data/derived/router_results_smoke_$(date -u +%Y%m%dT%H%M%SZ).jsonl"
  echo "Existing mock smoke results detected; archiving to $ARCHIVE"
  mv "$RESULTS" "$ARCHIVE"
fi

if [[ -f "$RESULTS" ]]; then
  EXISTING_CALLS="$(wc -l < "$RESULTS" | tr -d ' ')"
fi
REMAINING_CALLS=$((TOTAL_CALLS - EXISTING_CALLS))
if (( REMAINING_CALLS < 0 )); then
  REMAINING_CALLS=0
fi

echo "Router full run"
echo "Estimated total API calls: $TOTAL_CALLS"
echo "Existing result rows: $EXISTING_CALLS"
echo "Estimated remaining API calls: $REMAINING_CALLS"

if [[ -z "${ROUTER_MODEL:-}" ]]; then
  echo "Set ROUTER_MODEL before launching, e.g. ROUTER_MODEL=<codex-model-id>" >&2
  exit 2
fi

BACKEND="$("$PY" - <<'PY'
import yaml
print(yaml.safe_load(open("config/router_experiment.yaml"))["router"]["backend"])
PY
)"

if [[ "$BACKEND" == "openai_api" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Set OPENAI_API_KEY before launching with router.backend=openai_api." >&2
  exit 2
fi

echo "Router model: $ROUTER_MODEL"
echo "Router backend: $BACKEND"

"$PY" -m src.run_router_experiment --config config/router_experiment.yaml
"$PY" -m src.estimate_router --config config/router_experiment.yaml
"$PY" -m src.plot_router --config config/router_experiment.yaml

echo "Router full run complete. Summary: data/derived/router_summary.json"
