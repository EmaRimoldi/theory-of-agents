# Implementation Spec — Replicate the Experiment on MBPP+

> **Audience: an autonomous coding agent (Codex / Claude Code).**
> Replicate the HumanEval+ proper-time / retry-allocation experiment on **MBPP+**,
> in the existing project at `/Users/emanuelerimoldi/Documents/theory-of-agents`.
> This is mostly a DATASET SWAP: the worker eval pipeline, the trace schema, the
> allocation simulator, the router, and the estimators already exist and must be
> REUSED, not rewritten. Only the dataset, the mode cutoffs, and the file names
> change. Keep every convention identical so HumanEval+ and MBPP+ results are
> directly comparable.

---

## 0. Goal (one paragraph)

Re-run the entire experiment with MBPP+ in place of HumanEval+, using the SAME
three workers (Qwen2.5-Coder 1.5B / 7B / 32B), the SAME strict success criterion
(pass both base and plus tests), the SAME retry-allocation router (k-fold,
static allocation, sequential execution fastest->slowest), and the SAME
estimators and plots. MBPP+ has ~378 problems (vs 164), giving more statistical
power and tighter bootstrap CIs. The purpose is robustness: does the HumanEval+
finding — the router matches the statistical oracle and converts information into
time savings, but on size-only models cannot keep the strong model's accuracy —
replicate on a second, larger code dataset?

---

## 1. Hard requirements (identical to the existing project)

- Existing `.venv` (uv); never base Python.
- Separate worker-run (writes raw logs) from allocation/estimation (reads logs)
  from plotting (reads derived). No re-running workers during analysis.
- Strict success = `attempt_status == "pass"` (both base and plus). Unchanged.
- Resource clock: cumulative tokens AND wall-clock seconds across ALL attempts
  up to and including the first pass. Per-attempt parallel arrays, exactly the
  HumanEval+ schema.
- Censored (unsolved within max_attempts) kept with tau=inf, solved=false.
- tqdm + console metrics on long loops.
- Workers run LOCALLY via the existing backend (Ollama). The worker run is long
  (32B on ~378 problems); implement it, smoke-test it, print the full-run command
  with an ETA, and let the HUMAN launch it. Do not auto-launch the full worker run.
- The router run is free (Codex subscription) but still gated behind the human
  per the existing convention; print its command, do not auto-launch.

---

## 2. Dataset — MBPP+

- Source: EvalPlus, the MBPP+ dataset (the plus-augmented MBPP). Fetch via the
  same mechanism used for HumanEval+ (HuggingFace `datasets` or the `evalplus`
  package). Verify the exact dataset id at implementation time.
- Each problem provides: a task id, a prompt / text description, a canonical
  solution, an entry point, and the plus test suite. MBPP problems are specified
  by a natural-language description plus a few asserts; the EvalPlus runner
  defines the strict pass criterion (base + plus). REUSE the existing executor /
  EvalPlus test-running path; do not hand-roll the pass check.
- Confirm the problem count and record it (MBPP+ is ~378; use the actual number
  returned).

### Mode assignment (MUST mirror the HumanEval+ method, with MBPP+-specific cutoffs)

- Use the SAME difficulty proxy as HumanEval+: `ref_solution_length` (token
  length of the canonical solution).
- Recompute the tertile cutoffs ON MBPP+ (its length distribution differs from
  HumanEval+). Three modes easy/medium/hard by tertiles, balanced counts.
- Freeze to `data/derived/mbpp_modes.json` and `data/derived/mbpp_modes_meta.json`
  (same format as the HumanEval+ files), recording the proxy and the new cutoffs.
- Document in NOTES.md that the mode SCHEMA is identical (same proxy, same tertile
  method) but the CUTOFFS are MBPP+-specific, so "mode" is a transferable notion.

---

## 3. Worker run (reuse `run_eval.py`)

- Reuse the existing worker eval entry point; point it at MBPP+ via config.
- Output raw logs with the SAME schema as HumanEval+, to new files:
  `data/raw/runs_qwen2.5-coder_<size>_full_mbpp_<timestamp>.jsonl` for sizes
  1.5b, 7b, 32b.
- Same sampling params as HumanEval+ (temperature 0.6, top_p 0.95,
  max_attempts 10, max_new_tokens 1024, seed 0) so the two datasets are
  comparable. Record them.
- Resumable: skip (model, task) pairs already logged.
- Models pulled already (qwen2.5-coder:1.5b/7b/32b). If a model is missing, stop
  and tell the human the `ollama pull` command.

---

## 4. Allocation + router (reuse, just repoint to MBPP+ logs)

REUSE without logic changes:
- `src/load_traces.py` (point at MBPP+ logs and mbpp_modes.json),
- `src/simulate.py` (execute_allocation — unchanged; the unit-check invariant
  still holds: charge every consumed attempt up to and including first pass),
- `src/baselines.py` (always-models, best-single, oracle hindsight, stat-oracle),
- `src/router.py` (same Codex router, temperature 0, full_traces in-context,
  returns allocation JSON only, never code; never sees test-problem outcomes),
- `src/run_router_experiment.py` (k-fold; see below).

k-fold: 5 folds, stratified by mode, frozen to `data/derived/mbpp_folds.json`,
seed 0. With ~378 problems each fold has ~75 test problems — much more power than
HumanEval+. No leakage. Router sees only training-fold traces + the held-out
problem's mode.

Outputs to MBPP+-namespaced files so the HumanEval+ results are not overwritten:
- `data/derived/mbpp_router_results.jsonl`
- `data/derived/mbpp_router_summary.json`
- `figures/router_mbpp/`

Config: add a parallel `config/router_experiment_mbpp.yaml` mirroring the
HumanEval+ one but with MBPP+ logs, mbpp_modes.json, mbpp_folds.json, and the
MBPP+ output paths. Do NOT edit the HumanEval+ config.

---

## 5. Estimation + plots (reuse, repoint)

- `src/estimate_router.py`: same metrics, read mbpp_router_results.jsonl, write
  mbpp_router_summary.json. Pooled over all ~378 decisions and per mode: solved
  rate of router / each always-model / best-single / oracle / stat-oracle; mean
  wall-clock time each; time saved vs best-single (bootstrap CI, B=1000);
  router-vs-stat_oracle gap (bootstrap CI); allocation distribution by mode.
- `src/plot_router.py`: same four figures into `figures/router_mbpp/`:
  time-vs-accuracy, time-saved-by-mode, alloc-distribution (router vs
  stat-oracle), gap-to-oracle.

---

## 6. Comparison to HumanEval+ (new, small)

Add `src/compare_datasets.py` that reads BOTH summaries
(`router_summary.json` and `mbpp_router_summary.json`) and prints a side-by-side
table: for each dataset, the router/best-single/stat-oracle solved rate and mean
time, the time saved (with CI), the router-vs-stat_oracle gap (with CI), and the
matched-accuracy verdict. This is the robustness check: do the two datasets tell
the same story? Write `data/derived/dataset_comparison.json` and a short Markdown
summary. No model or API calls.

---

## 7. Smoke tests

- **Worker smoke:** 5 MBPP+ problems, 2 attempts, 1.5B only, via the existing
  smoke path repointed to MBPP+; confirm 5 raw lines with the correct schema and
  that mbpp_modes.json was frozen.
- **Router smoke:** fold 0, 5 test problems, MOCK router (no API calls); confirm
  mbpp_router_results.jsonl has 5 lines, summary written, figures produced, and
  the execute_allocation unit invariant holds on an MBPP+ solved-on-attempt>=2
  example (sum of consumed attempt_seconds up to and including first pass).
- The agent runs both smoke tests.

## 8. Full-run scripts (written, not auto-launched)

- `scripts/full_run_mbpp.sh`: the three Qwen workers on all MBPP+ problems. Print
  an ETA (the 32B is the bottleneck; ~378 problems). Human launches.
- `scripts/full_router_run_mbpp.sh`: the k-fold router over all MBPP+ test
  decisions. Print the API-call estimate (~378). Resumable. Human launches.

---

## 9. Acceptance checklist

- [ ] MBPP+ fetched; problem count recorded; strict pass = base+plus reused.
- [ ] Mode schema identical to HumanEval+ (ref_solution_length, tertiles);
      cutoffs recomputed on MBPP+; frozen to mbpp_modes.json.
- [ ] Worker logs use the identical per-attempt schema; new MBPP+ file names.
- [ ] simulate/baselines/router/estimators REUSED, not rewritten; only repointed.
- [ ] 5-fold stratified, frozen to mbpp_folds.json, no leakage.
- [ ] All MBPP+ outputs namespaced; HumanEval+ results untouched.
- [ ] Both smoke tests pass (worker smoke; router smoke with mock, no API).
- [ ] compare_datasets.py produces the side-by-side robustness table.
- [ ] Full-run scripts written with ETAs/estimates, NOT auto-launched.

---

## 10. The question this answers (state in README)

Does the HumanEval+ finding replicate on MBPP+?
- Does the router again match the stat-oracle (gap CI includes zero)?
- Does it again save time vs best-single (saving CI excludes zero)?
- Does it again fail to match best-single accuracy, and is the accuracy gap also
  present in the stat-oracle (i.e. structural, not a router defect)?
- With ~378 problems the CIs are tighter; report whether the larger sample
  sharpens or overturns the HumanEval+ verdict. If MBPP+ tells the same story,
  the finding is robust and the next step is the specialized-model (code+math)
  experiment; if it diverges, document how.
