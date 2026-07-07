# Implementation Spec — BIG-Bench Hard (BBH) Domain-Mode Experiment

> **Audience: an autonomous coding agent (Codex / Claude Code).**
> Add a third dataset, **BIG-Bench Hard (BBH)**, to the project at
> `/Users/emanuelerimoldi/Documents/theory-of-agents`. This tests whether the
> proper-time frontier becomes NON-DEGENERATE when modes are heterogeneous
> reasoning SUBTASKS rather than difficulty tiers, using three GENERALIST Qwen
> models. Reuse the existing worker -> allocation -> router -> frontier pipeline;
> the dataset, the models, the verifier, and the mode definition change.

---

## 0. Why this experiment (one paragraph)

On code datasets (HumanEval+, MBPP+) the modes were difficulty tiers, and the
three same-family size-only models stayed ordered (the strong model won every
mode), so allocation bought speed but not accuracy. BBH is a collection of
heterogeneous reasoning subtasks — arithmetic, logical deduction, language,
spatial, algorithmic — that are cognitively different from one another. We use
the GENERALIST Qwen2.5 models (not the Coder variants), because BBH is reasoning,
not code. The question: do different models win different SUBTASKS (a
non-degenerate per-subtask proper-time frontier), where difficulty tiers did not
break the ordering? If yes, allocation has comparative advantage to exploit on a
genuinely different axis.

---

## 1. Hard requirements (unchanged conventions)

- Existing `.venv` (uv); never base Python.
- Separate worker-run (writes raw logs) from allocation/estimation (reads logs)
  from plotting (reads derived). No worker re-runs during analysis.
- Resource clock: cumulative tokens AND wall-clock seconds across ALL attempts up
  to and including the first pass. Per-attempt parallel arrays — identical schema
  to the HumanEval+/MBPP+ logs (attempt_seconds / attempt_token_counts /
  attempt_statuses).
- Censored (unsolved within max_attempts) kept: tau=inf, solved=false.
- tqdm + console metrics on long loops.
- Workers run LOCALLY via Ollama. The full worker run is long (32B is the
  bottleneck); implement, smoke-test, print the full-run command with an ETA, and
  let the HUMAN launch it. Do NOT auto-launch the full worker run.
- Namespace all BBH outputs (`bbh_*`) so HumanEval+/MBPP+ results are untouched.

---

## 2. Models — generalist Qwen2.5 (NOT Coder)

Three generalist instruct models, same family and sizes as the Coder menu, so the
only difference from the code experiments is coder-vs-generalist:

- `qwen2.5:1.5b`   (Qwen2.5-1.5B-Instruct)
- `qwen2.5:7b`     (Qwen2.5-7B-Instruct)
- `qwen2.5:32b`    (Qwen2.5-32B-Instruct)

The human pulls them (put in README, do not run yourself):
```
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:7b
ollama pull qwen2.5:32b
```
Log the exact tag and digest from `ollama show` for each. All run 4-bit on a 64GB
M1 (~1GB / 5GB / 20GB). Same sampling as the code experiments: temperature 0.6,
top_p 0.95, max_attempts 10, max_new_tokens (raise to 2048 — BBH chain-of-thought
can be longer than code completions), seed 0.

---

## 3. Dataset — BBH

- Source: the BIG-Bench Hard dataset (the 23 challenging BIG-Bench subtasks with
  exact-answer targets). Fetch via HuggingFace `datasets` (verify the exact id at
  implementation time; BBH is commonly distributed as `lukaemon/bbh` or via the
  `BIG-Bench-Hard` repo). Record the resolved source and per-subtask counts.
- Each example provides: the subtask name, an input (the question/prompt), and a
  single exact target answer.
- BBH has ~6.5k examples across 23 subtasks (~250 each). For speed, the worker run
  may CAP per-subtask examples (config `max_examples_per_subtask`, default 100);
  record the cap. This keeps the run tractable while leaving enough per-cell data.

### Verifier (exact-answer, generative)
- The model is prompted to reason (chain-of-thought) and then emit a final answer
  in a parseable form (e.g. "the answer is X"). The verifier extracts the final
  answer and compares it to the target by exact match (normalized: strip
  whitespace/case/punctuation as BBH convention dictates; for multiple-choice-style
  subtasks match the option label/text).
- A single attempt is a SUCCESS iff the extracted final answer exactly matches the
  target. Define and freeze the answer-extraction + normalization rule per subtask
  family; document it in NOTES.md. This is the strict success criterion analogous
  to "pass" in the code experiments.
- IMPORTANT (retry must be meaningful): because BBH is generative with
  chain-of-thought, sampling at temperature 0.6 yields different reasoning and
  different answers across attempts, so the success-by-budget curve F(t) is
  non-trivial. Verify this on the smoke (see §7): F(t) should not be flat.

---

## 4. Modes — reasoning subtasks, grouped into 5 cognitive families

The mode of an example is its reasoning TYPE, not its difficulty. Using all 23
subtasks as modes gives thin cells; group them into 5 cognitive families for
better-populated cells. Freeze the grouping to `data/derived/bbh_modes.json`
(`{example_id: family}`) and the mapping subtask->family to
`data/derived/bbh_mode_groups.json`.

Proposed grouping (the agent assigns each BBH subtask to exactly one family; if a
subtask is ambiguous, pick the dominant cognitive demand and record the choice):

1. **arithmetic** — multi-step arithmetic / numeric reasoning
   (e.g. multistep_arithmetic, object_counting).
2. **logical_deduction** — formal/relational deduction
   (e.g. logical_deduction_*, formal_fallacies, boolean_expressions).
3. **language** — linguistic / semantic reasoning
   (e.g. disambiguation_qa, snarks, hyperbaton, salient_translation_errors).
4. **spatial_temporal** — spatial, temporal, ordering
   (e.g. navigate, geometric_shapes, temporal_sequences, tracking_shuffled_objects_*).
5. **algorithmic** — sequence/symbol manipulation and structured procedures
   (e.g. dyck_languages, word_sorting, reasoning_about_colored_objects).

> The exact subtask->family assignment is a judgment call. Assign each of the 23
> subtasks to its dominant family, freeze the mapping, print the per-family
> example counts, and flag any family under ~150 examples (after the per-subtask
> cap). Record the full mapping in NOTES.md. If a cleaner mapping is obvious from
> the actual subtask list, use it and document the deviation.

Also produce, for robustness, the ungrouped view (`data/derived/bbh_modes_raw.json`)
with all 23 subtasks as modes, so the frontier can be inspected at full
granularity even if cells are thin.

---

## 5. Worker run (reuse `run_eval.py`, repointed)

- Reuse the existing worker eval entry point; point it at BBH via a new config.
- Output raw logs with the IDENTICAL per-attempt schema, to new files:
  `data/raw/runs_qwen2.5_<size>_full_bbh_<timestamp>.jsonl` for 1.5b/7b/32b
  (note: generalist tag `qwen2.5`, not `qwen2.5-coder`).
- Resumable: persist the run id; skip already-logged (model, example) pairs.
- tqdm + console metrics; logfile under data/raw/.
- If a model is not pulled, stop and tell the human the ollama pull command.

---

## 6. Allocation + router + frontier (reuse, repoint)

REUSE without logic changes, pointed at the BBH logs and bbh_modes.json:
- `src/load_traces.py`, `src/simulate.py` (execute_allocation — same invariant:
  charge every consumed attempt up to and including first pass), `src/baselines.py`,
  `src/router.py` (Codex subscription router, temp 0, full_traces in-context,
  returns allocation JSON only, never the answer; never sees held-out outcomes),
  `src/run_router_experiment.py`, estimators, plots, `frontier_by_category.py`.
- The router's in-context training material is now training-fold traces grouped by
  reasoning FAMILY, and the held-out example's family. The router allocates the 10
  attempts across the three generalist models; sequential execution
  fastest->slowest (1.5b, 7b, 32b).
- 5-fold stratified by family, frozen to `data/derived/bbh_folds.json`, seed 0, no
  leakage.
- Namespaced outputs: `data/derived/bbh_router_results.jsonl`,
  `data/derived/bbh_router_summary.json`, `figures/router_bbh/`.
- New config `config/router_experiment_bbh.yaml` mirroring the existing ones with
  BBH logs, bbh_modes.json, bbh_folds.json, execution_order ["1.5b","7b","32b"],
  and BBH output paths. Do not edit existing configs.

---

## 7. The decisive diagnostic + retry sanity (`frontier_by_category.py`, reused)

- **Retry sanity (do this first, on the smoke):** for a couple of subtasks, plot/
  report the success-by-budget curve F(t) across attempts. Confirm F(t) RISES with
  attempts (retry changes outcomes). If F(t) is flat for a subtask, flag it —
  proper time is uninformative there and the subtask may need higher temperature
  or exclusion. Record in NOTES.md.
- **Frontier-by-family:** compute per (family, model) the proper time
  tau* = inf_t t/F(t) (KM, censoring floor, strict success) and the strict solved
  rate. Report whether argmin_model tau*(family) and the accuracy winner SWAP
  across families, with bootstrap CIs (B=1000) on per-family log2 tau* ratios.
  THIS is the question: does reasoning-type break the size ordering that
  difficulty did not?
- Write `data/derived/bbh_family_frontier.json` and
  `figures/router_bbh/fig_bbh_family_frontier.png` (grouped bars: x = family,
  groups = model, y = tau*, per-family winner marked, CI whiskers).
- Repeat at raw 23-subtask granularity as a secondary view.

---

## 8. Smoke tests

- **Worker smoke:** 5 BBH examples spanning 2-3 families, 2 attempts, 1.5B only,
  via the existing pipeline repointed to BBH. Confirm 5 raw lines with the
  per-attempt schema, exact-match success extraction working, and that the answer
  parser handles the BBH answer format. Confirm bbh_modes.json was frozen.
- **Retry sanity on the worker smoke:** confirm at least one example was solved on
  an attempt >= 2 (so the cumulative clock path is exercised) and the F(t) idea is
  validated on a tiny scale.
- **Router smoke:** fold 0, 5 examples, MOCK router (no API). Confirm
  bbh_router_results.jsonl has 5 lines, summary written, figures produced, and the
  execute_allocation unit invariant holds (cumulative time across consumed
  attempts up to first pass).
- The agent runs the smoke tests.

## 9. Full-run scripts (written, NOT auto-launched)

- `scripts/full_run_bbh.sh`: the three generalist Qwen on the capped BBH set.
  Print an ETA (32B bottleneck; depends on the per-subtask cap and CoT length).
  Resumable. Human launches.
- `scripts/full_router_run_bbh.sh`: the k-fold router over all BBH test decisions.
  Print the API-call estimate. Resumable. Human launches.

---

## 10. Acceptance checklist

- [ ] Uses existing .venv; no worker re-runs during analysis; pipeline reused and
      repointed, not rewritten.
- [ ] Generalist models qwen2.5:1.5b/7b/32b (NOT coder); tags/digests logged.
- [ ] BBH fetched; per-subtask counts recorded; per-subtask cap recorded.
- [ ] Verifier = exact match on extracted final answer; extraction/normalization
      rule frozen and documented; strict success defined analogously to "pass".
- [ ] Retry is meaningful: F(t) rises with attempts on the smoke; flat subtasks
      flagged.
- [ ] Modes = 5 cognitive families; subtask->family mapping frozen and printed;
      thin families flagged; raw 23-subtask view also produced.
- [ ] Worker logs use the identical per-attempt schema; new bbh filenames.
- [ ] 5-fold stratified by family, frozen, no leakage; router sees family, never
      the held-out answer/outcome.
- [ ] All BBH outputs namespaced; code/math results untouched.
- [ ] frontier_by_category reports whether the per-family tau*/accuracy winner
      SWAPS across families, with bootstrap CIs.
- [ ] Smoke tests pass (worker smoke + retry sanity + router smoke with mock).
- [ ] Full-run scripts written with ETAs/estimates, NOT auto-launched.

---

## 11. The question this answers (state in README)

Difficulty modes (code datasets) left the size ordering intact. Does REASONING
TYPE (BBH cognitive families), with generalist models, break it?
- Does argmin tau* swap across families (a family where 1.5b or 7b is fastest,
  with separated CIs)?
- Does the accuracy winner swap across families (a family where the 32b is NOT
  the most accurate)?
- If YES: the per-family frontier is non-degenerate on a cognitive axis;
  allocation has comparative advantage; re-check whether matched-accuracy becomes
  reachable here.
- If NO even across reasoning families: three same-family size-only models are
  ordered on every axis tested (difficulty AND reasoning type), which is strong
  evidence that a non-degenerate frontier requires cross-FAMILY or specialized
  models, not just scale — pointing to a mixed-family or specialist menu as the
  next step.
- Report the code-vs-reasoning parallel: do same-scale Qwen behave the same way
  on code (difficulty modes) and reasoning (family modes)?
