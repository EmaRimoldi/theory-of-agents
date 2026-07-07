# Implementation Spec — Algorithmic-Category Modes (LLM tagging) on MBPP+

> **Audience: an autonomous coding agent (Codex / Claude Code).**
> Extend the MBPP+ experiment in `/Users/emanuelerimoldi/Documents/theory-of-agents`
> with a SECOND mode schema: algorithmic category, assigned by an LLM. This does
> NOT replace the difficulty-based modes and does NOT re-run any worker model.
> It adds a parallel mode labeling and re-runs the EXISTING allocation/router
> analysis on the new labels, so difficulty-modes and category-modes results sit
> side by side. Reuse everything; only the mode labels change.

---

## 0. Goal (one paragraph)

The difficulty-based modes (easy/medium/hard by reference-solution length) did
not break the model ordering: the strong model wins at every difficulty, so
allocation buys speed but not accuracy. We test a different partition —
**algorithmic category** — to see whether the three Qwen2.5-Coder models SWAP
which one wins across categories (e.g. one model relatively stronger on string
problems, another on math). If the per-category frontier is non-degenerate where
the per-difficulty frontier was not, allocation has real comparative advantage to
exploit. We label every MBPP+ problem into one of 4 categories with an LLM,
freeze the labels, and re-run the existing analysis with category as the mode.

---

## 1. Hard requirements

- Existing `.venv` (uv); never base Python.
- **No worker re-runs.** Reuse the existing MBPP+ worker logs.
- Reuse the existing allocation simulator, baselines, router, estimators, plots.
  Only the mode-label file changes.
- Freeze the category labels BEFORE running any analysis; never adjust them after
  seeing results.
- Namespace all category-mode outputs separately from the difficulty-mode ones so
  both survive.
- tqdm on the tagging loop; log progress.

---

## 2. The four categories (FIXED, mutually exclusive, exhaustive)

Every MBPP+ problem is assigned EXACTLY ONE of:

1. **string** — string/text manipulation: parsing, formatting, substring search,
   character operations, regex-like logic.
2. **math** — numeric/mathematical: arithmetic, number theory, sequences,
   combinatorics, numeric computation, math formulas.
3. **list_ds** — list/data-structure operations: list/dict/set manipulation,
   sorting, searching, aggregation, matrix/array handling.
4. **logic_control** — logic and control flow: complex conditionals, recursion,
   simulation, state machines, case-by-case reasoning not dominated by the above.

**Precedence rule for hybrids** (apply in this order, stop at first match):
- If the core difficulty is non-trivial numeric/mathematical computation → `math`.
- Else if the problem is primarily string/text → `string`.
- Else if it primarily manipulates lists/dicts/sets/arrays → `list_ds`.
- Else → `logic_control`.

State this precedence in the LLM prompt so labeling is deterministic and
reproducible.

---

## 3. LLM tagging (`src/tag_categories.py`)

- For each MBPP+ problem, send the LLM the problem's natural-language
  description, its entry-point signature, and its canonical solution. Ask it to
  return STRICT JSON: `{"category": "string|math|list_ds|logic_control",
  "confidence": 0.0-1.0, "rationale": "<one short sentence>"}`.
- Use the SAME Codex-family subscription model used for the router. Temperature 0.
  Pin and log the model id.
- Embed the four definitions and the precedence rule (§2) in the prompt verbatim.
  Instruct: choose exactly one category; output only JSON; no markdown.
- Robust parse; on failure, retry once, then fall back to `logic_control` and
  flag the record. Count parse failures.
- tqdm over problems.

**Output:** freeze to `data/derived/mbpp_modes_category.json`
(`{task_id: "string"|"math"|"list_ds"|"logic_control"}`) and
`data/derived/mbpp_modes_category_meta.json` (model id, prompt hash, per-category
counts, parse-failure count, mean confidence).

---

## 4. Validation of the labels (cheap, required)

Before trusting the labels:
- Print the per-category counts. Flag if any category has fewer than ~50 problems
  on MBPP+ (too thin for a stable per-cell estimate); note it in NOTES.md.
- Sample 20 random labeled problems and write them to
  `data/derived/category_label_sample.json` (task_id, description snippet,
  assigned category, rationale) so the human can eyeball labeling quality.
- Report the distribution. We EXPECT imbalance (string/list/math common,
  logic_control rarer); document it.

Also produce a **2-category coarse view** `data/derived/mbpp_modes_mathnot.json`
mapping each problem to `math` vs `nonmath` (collapse string+list_ds+logic_control
into nonmath). This high-power view (~189/cell) is the cleanest test of a
math-vs-rest swap.

---

## 5. Re-run the analysis on category modes (reuse, repoint)

Without changing any analysis logic, run the existing pipeline with the mode file
set to the category labels:

- Build category-stratified 5-fold: `data/derived/mbpp_folds_category.json`
  (5 folds, stratified by the 4 categories, seed 0, no leakage).
- Run `run_router_experiment.py` pointed at `mbpp_modes_category.json` and the
  category folds. The router now sees, in-context, training traces grouped by
  CATEGORY, and the held-out problem's category. Outputs:
  `data/derived/mbpp_router_results_category.jsonl`.
- Run estimators -> `data/derived/mbpp_router_summary_category.json`.
- Run plots -> `figures/router_mbpp_category/`.
- Repeat for the 2-category math/nonmath view with its own namespaced outputs
  (`*_mathnot`).

Config: add `config/router_experiment_mbpp_category.yaml` mirroring the MBPP+
config but with the category mode file, category folds, and category output
paths. Do not edit existing configs.

---

## 6. The decisive diagnostic (`src/frontier_by_category.py`)

This is the point of the whole extension. From the MBPP+ worker logs, compute per
(category, model) the proper time `tau*` (inf_t t/F(t), KM with censoring floor,
strict success) AND the strict solved rate. Then:

- For each category, identify the fastest model by tau* and the most accurate
  model by solved rate.
- **Report whether the winner SWAPS across categories** — i.e. whether
  argmin_model tau*(category) differs across categories, and likewise for
  accuracy. This is exactly what difficulty-modes did NOT show.
- Bootstrap CIs on the per-category log2 tau* ratios between models, so a swap is
  only claimed when CIs separate.
- Write `data/derived/category_frontier.json` and a figure
  `figures/router_mbpp_category/fig_category_frontier.png` (grouped bars: x =
  category, groups = model, y = tau*, mark per-category winner, CI whiskers).

Also report the same for the 2-category math/nonmath view, where power is highest.

---

## 7. Smoke test (`scripts/smoke_category.sh`)

- Tag only 5 problems (real LLM, 5 calls — cheap) OR a mock tagger; confirm strict
  JSON parsing and the frozen label file format.
- Run the category analysis on fold 0, 5 test problems, MOCK router (no API).
- Confirm: category label file written; category folds frozen; results jsonl has
  5 lines; category_frontier.json produced; figures exist.
- The agent runs this.

## 8. Full run (`scripts/full_category_run.sh`)

- Full LLM tagging of all MBPP+ problems (~378 LLM calls, cheap, free on the
  subscription). Then the full category router run (~378 router calls) and the
  2-category run.
- Print the total LLM-call estimate and stop for the human to launch. Resumable
  (skip already-tagged problems; skip already-decided (fold, task)).

---

## 9. Acceptance checklist

- [ ] Uses existing .venv; no worker re-runs; existing analysis reused, repointed.
- [ ] 4 categories fixed, mutually exclusive, with the precedence rule in the
      prompt; labels frozen to mbpp_modes_category.json before analysis.
- [ ] LLM tagger: strict JSON, temperature 0, model id pinned, parse failures
      counted, fallback flagged.
- [ ] Per-category counts printed; thin categories (<~50) flagged; 20-sample
      eyeball file written; 2-category math/nonmath view produced.
- [ ] Category-stratified 5-fold frozen; no leakage; router sees category, not
      held-out outcome.
- [ ] All category outputs namespaced; difficulty-mode results untouched.
- [ ] frontier_by_category.py reports whether the per-category tau* / accuracy
      winner SWAPS across categories, with bootstrap CIs.
- [ ] Smoke test passes (mock router, no API beyond optional 5 tag calls).
- [ ] Full run script written with call estimate, NOT auto-launched.

---

## 10. The question this answers (state in README)

Difficulty-modes left the model ordering intact (strong model wins everywhere).
Does ALGORITHMIC CATEGORY break it?
- Does argmin tau* swap across categories (a category where a smaller/different
  model is fastest with separated CIs)?
- Does the accuracy winner swap across categories (a category where the strong
  model is NOT the most accurate)?
- If YES on either, the per-category frontier is non-degenerate: allocation has
  comparative advantage to exploit, and the router should be able to keep accuracy
  while saving time — re-check the matched-accuracy verdict under category modes.
- If NO even by category, the three same-family size variants are ordered on every
  axis we can define, which is strong evidence that a NON-DEGENERATE frontier
  requires model SPECIALIZATION (e.g. a math model on a code+math suite), not just
  scale or task-type partitioning. Either outcome is a clean result.
```
