# Implementation Spec — Retry-Allocation Router (k-fold on HumanEval+)

> **Audience: an autonomous coding agent (Codex / Claude Code).**
> Build this on top of the existing project at
> `/Users/emanuelerimoldi/Documents/theory-of-agents`. The worker logs already
> exist (three `runs_qwen2.5-coder_*_full_20260616T0640Z.jsonl` files). You do
> NOT re-run any worker model. You build the allocation experiment that consumes
> those logs, plus an LLM router that runs via API. Follow the conventions and
> file contracts exactly.

---

## 0. What this experiment does (one paragraph)

We test whether a router, given the past behavior of three Qwen2.5-Coder models
(1.5B / 7B / 32B) on HumanEval+, can **allocate a fixed budget of 10 retry
attempts across the three models so as to minimize wall-clock time to a verified
solution, while matching the accuracy of the best single model.** The router
decides a static allocation `(n_1.5, n_7, n_32)` with `n_1.5 + n_7 + n_32 <= 10`;
the attempts are then executed **sequentially**, fastest model first, summing
their wall-clock seconds, stopping at the first attempt that passes BOTH the base
and the plus tests. Execution is **simulated exactly** from the per-attempt
arrays already in the logs — no model is re-run. We use 5-fold cross-validation
on the 164 HumanEval+ problems, with the router learning in-context from all
training-fold traces and allocating on the held-out test fold. We compare the
LLM router against an always-best-model baseline, an oracle allocation (upper
bound), and a statistical-oracle router (the information-optimal allocation
computed directly from the numbers, no LLM).

---

## 1. Hard requirements and conventions

- **Python env:** use the existing `.venv` (uv). Never the base Python. Add any
  new deps to `requirements.txt`.
- **Separate data generation from analysis from plotting.** Running the router
  and simulating allocations **writes results to disk**; plotting reads from disk
  only. No script both calls the router and draws a figure.
- **No worker re-runs.** Workers are never invoked. All worker behavior comes
  from the existing `runs_*_full_*.jsonl` logs.
- **The router runs once per (fold, test-problem).** It is the only component
  that makes live API calls. Temperature 0; pin and log the exact model
  identifier and version string with every decision.
- **tqdm + console metrics** on every loop over problems.
- **Smoke first.** Validate end-to-end on 1 fold and ~5 test problems with a
  cheap/mock router, confirm the pipeline, then print the full-run command and
  stop. Do not launch the full router run over all folds yourself; the router
  API calls cost money.

---

## 2. Success definition (FIXED)

A single attempt is a SUCCESS iff its `attempt_statuses[i] == "pass"` (passes
BOTH base and plus tests). `"pass/fail"`, `"fail/fail"`, `"fail/pass"` are all
NON-successes. This is the strict, verified-solution threshold. Do not treat
`"pass/fail"` as success in the main analysis. A weak-threshold sensitivity
analysis (success = `"pass"` OR `"pass/fail"`) is optional and, if done, must be
clearly separated.

---

## 3. Worker-trace loader (`src/load_traces.py`)

Reads the three full-run JSONL files and returns, for each `(model, task_id)`,
the per-attempt arrays:

```
traces[task_id][model] = {
    "attempt_seconds": [...],        # parallel arrays, index 0 = attempt 1
    "attempt_token_counts": [...],
    "attempt_statuses": [...],       # "pass" | "pass/fail" | "fail/fail" | "fail/pass"
    "mode": "easy"|"medium"|"hard",
    "solved": bool,                  # recompute under strict threshold; do not trust stored
}
```

Models keyed by short name `"1.5b"`, `"7b"`, `"32b"`. Recompute `solved` from
`attempt_statuses` under the strict threshold (§2). Read mode from
`data/derived/modes.json`.

---

## 4. Allocation execution — the exact simulator (`src/simulate.py`)

```
def execute_allocation(task_id, alloc, traces, order=("1.5b","7b","32b")):
    """
    alloc: dict model -> number of attempts allocated, sum <= 10.
    SEQUENTIAL execution in `order` (default fastest->slowest: 1.5b, 7b, 32b).
    For each model in order, consume its first alloc[model] logged attempts in
    sequence; accumulate attempt_seconds and attempt_token_counts; if any consumed
    attempt has status == "pass", STOP and return success.
    Returns dict:
      solved: bool,
      time_seconds: float,   # sum of consumed attempt_seconds up to & including first pass
      tokens: int,           # same accumulation
      n_attempts_used: int,
      first_pass_model: str or None.
    """
```

Rules:
- Consume in order: all of model A's allocated attempts, then B, then C. Stop at
  first `"pass"`.
- Time and tokens accumulate over EVERY consumed attempt up to and including the
  first pass — failed attempts before the pass are charged. This is the cost of
  the retry policy; do not charge only the winning attempt.
- A model has up to 10 logged attempts; if allocated more than available, consume
  only what exists and note the guard in NOTES.md.
- Order matters for time; default fastest->slowest is fixed for the main run.
  Record the order used.

---

## 5. Baselines and oracles (`src/baselines.py`)

All via simulation over the logs:

- **always-1.5b / always-7b / always-32b:** alloc all 10 to one model.
- **best-single-model:** the always-model with the highest strict solved rate on
  the FULL set. Defines the **accuracy target** the router must match. (Likely
  7b; report identity and solved rate.)
- **oracle allocation (upper bound):** per problem, the hindsight allocation that
  solves it (if solvable by any model) at MINIMUM time. Not visible to router.
- **statistical-oracle router:** allocation rule using ONLY training-fold
  statistics (per mode, per model: solve-rate-by-attempt-count and mean attempt
  time) plus the test problem's mode, choosing the expected-time-optimal
  allocation. No LLM. The information-optimal reference: isolates "information is
  useful" from "the LLM uses it well".

---

## 6. The LLM router (`src/router.py`)

Only live component. Contract:

- **Backend:** a GPT model of the Codex family via API. Read `ROUTER_MODEL` and
  the API key from env. Temperature 0. Log resolved model id/version per call.
- **Receives, per (fold, test problem):**
  - deployment contract: three models `1.5b`/`7b`/`32b`; budget 10 attempts;
    sequential execution fastest->slowest; success = pass both tests; objective =
    minimize expected wall-clock time to verified solution while matching best
    single model accuracy.
  - in-context training material: ALL training-fold traces (first experiment —
    give everything, do not compress yet). Per training problem: its mode, and
    per model the per-attempt status sequence and per-attempt seconds.
  - the test problem's mode, and nothing revealing the held-out outcome.
- **Must NOT receive** the test problem's own attempt outcomes (that is the
  oracle).
- **Returns** strict JSON `{"n_1.5b": int, "n_7b": int, "n_32b": int}`, sum <= 10.
  Parse robustly; on failure log and fall back to always-7b, counting the failure.
- **Role boundary:** allocate, do not solve. Prompt must instruct: output only the
  allocation JSON, never code or a solution. Router never sees/produces problem code.
- **Context size:** if all training traces exceed the window, fall back to a
  per-mode summary (per mode, per model: distribution of attempts-to-first-pass
  and mean attempt time) and note it in NOTES.md. Prefer raw traces if they fit.

---

## 7. k-fold protocol — MAIN (`src/run_router_experiment.py`)

- **5-fold, stratified by mode**, preserving easy/medium/hard proportions. Freeze
  to `data/derived/folds.json` before running. seed 0.
- Per fold: training material = other 4 folds' traces; test = this fold. For each
  test problem: call router -> allocation -> simulate execution (§4) -> record;
  also compute all baselines + both oracles (§5) for the same problem.
- No leakage: each problem in exactly one test fold; router never sees a test
  problem's own outcomes.
- **Output** `data/derived/router_results.jsonl`, one record per (fold, problem):

```json
{
  "fold": 0, "task_id": "HumanEval/10", "mode": "medium",
  "router_alloc": {"n_1.5b": 6, "n_7b": 2, "n_32b": 0},
  "router_model": "<pinned id>",
  "router": {"solved": true, "time_seconds": 7.3, "tokens": 412, "first_pass_model": "1.5b"},
  "always_1.5b": {"solved": true, "time_seconds": 3.5},
  "always_7b": {"solved": true, "time_seconds": 8.9},
  "always_32b": {"solved": false, "time_seconds": 43.4},
  "oracle": {"solved": true, "time_seconds": 3.5, "alloc": {}},
  "stat_oracle": {"solved": true, "time_seconds": 4.1, "alloc": {}}
}
```

tqdm over (fold, problem); log running solved-rate and mean router time.

---

## 8. Aggregation (`src/estimate_router.py`)

Reads `router_results.jsonl`, writes `data/derived/router_summary.json`. No API
calls. Pooled over all 164 decisions and per mode:

- **Solved rate** of router, each always-model, best-single-model, oracle,
  stat-oracle. (Does the router MATCH best-single-model accuracy? The constraint.)
- **Total/mean wall-clock time** for each, over solved and overall. Headline:
  router mean time vs best-single-model mean time at equal-or-better solved rate.
- **Time saved** = best-single-model time − router time (per problem + aggregate).
- **Gap to oracle** and **gap to stat-oracle**. (router vs stat_oracle = does the
  LLM use the info well; stat_oracle vs always-best = is the info useful at all.)
- **Allocation distribution:** split vs commit frequency; mean allocation by mode.
- Bootstrap CIs (B=1000) on mean time-saving and on router-vs-stat_oracle gap,
  resampling test problems.

---

## 9. Plotting (`src/plot_router.py`, derived only)

- **fig_time_vs_accuracy.png:** scatter x = solved rate, y = mean time (inverted);
  points: each always-model, router, oracle, stat-oracle.
- **fig_time_saved_by_mode.png:** bars of mean time saved vs best-single-model per
  mode, bootstrap CIs.
- **fig_alloc_distribution.png:** router mean allocation vector by mode (stacked),
  next to stat-oracle's.
- **fig_gap_to_oracle.png:** router vs stat-oracle vs oracle aggregate, CIs.

PNG 130 dpi + PDF, minimal style.

---

## 10. Config (`config/router_experiment.yaml`)

```yaml
worker_logs:
  - data/raw/runs_qwen2.5-coder_1.5b_full_20260616T0640Z.jsonl
  - data/raw/runs_qwen2.5-coder_7b_full_20260616T0640Z.jsonl
  - data/raw/runs_qwen2.5-coder_32b_full_20260616T0640Z.jsonl
modes_file: data/derived/modes.json
success_threshold: strict
budget_attempts: 10
execution_order: ["1.5b", "7b", "32b"]
kfold:
  n_folds: 5
  stratify_by: mode
  seed: 0
router:
  backend: openai_api
  model_env: ROUTER_MODEL
  temperature: 0
  in_context: full_traces        # fallback: per_mode_summary
  parse_fail_default: {"n_1.5b": 0, "n_7b": 10, "n_32b": 0}
estimation:
  bootstrap_B: 1000
paths:
  results: data/derived/router_results.jsonl
  summary: data/derived/router_summary.json
  figures: figures/router
```

---

## 11. Smoke test (`scripts/smoke_router.sh`)

- Fold 0 only, 5 test problems.
- **MOCK router** (local function returning a fixed allocation, e.g. always-7b) —
  NO API calls.
- Run load_traces -> simulate -> baselines -> estimate -> plot end to end.
- Assert: router_results.jsonl has 5 lines; summary.json has the fields; figures
  exist; and a UNIT CHECK on execute_allocation: allocating 3 attempts to 1.5b on
  HumanEval/10 returns time = 1.985619834 + 0.190388417 + 1.349114292 =
  3.525122543 s, solved true, first_pass_model "1.5b".
- The agent runs this.

## 12. Full run (`scripts/full_router_run.sh`)

- All 5 folds, 164 decisions, real LLM router via API.
- Print estimated API-call count (164) and stop for the human to launch (costs
  money). Do NOT auto-launch.
- Resumable: skip (fold, task_id) already in router_results.jsonl.

---

## 13. Acceptance checklist

- [ ] Uses existing .venv (uv); no base Python; no worker re-runs.
- [ ] Success = strict "pass" on both tests (§2).
- [ ] execute_allocation charges ALL consumed attempts up to and including the
      first pass; unit-checked against HumanEval/10 1.5b (3.525122543 s).
- [ ] Sequential execution fastest->slowest; order logged.
- [ ] 5-fold stratified by mode, frozen to folds.json, no leakage.
- [ ] Router returns only allocation JSON, never code; temperature 0; model id
      pinned/logged; parse-failure fallback counted.
- [ ] Router never sees the test problem's own outcomes.
- [ ] Baselines + oracle + stat-oracle computed per test problem.
- [ ] estimate/plot make no API calls; read only derived data.
- [ ] Smoke passes with a MOCK router (no API calls).
- [ ] Full-run script written, prints API-call estimate, NOT auto-launched.

---

## 14. The question this answers (state in README)

Read `router_summary.json` and the figures:

- **Constraint:** does the router MATCH best-single-model solved rate? If it
  sacrifices accuracy, the time comparison is invalid.
- **Result:** at matched accuracy, is router mean time-to-verified LESS than the
  best single model's, with a CI excluding zero?
- **Mechanism:** does the router's allocation by mode resemble the stat-oracle's
  (easy -> fast model, hard -> strong model)? router-vs-stat_oracle = does the LLM
  use the info well; stat_oracle-vs-always-best = is the info useful at all.
- If the router only ever commits to one model and saves no time, that is the
  documented signal that three same-family size variants do not yield a
  non-degenerate allocation, and the next step is a specialized model (e.g. a math
  model on a code+math suite).
