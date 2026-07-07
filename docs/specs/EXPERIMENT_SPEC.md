# Implementation Spec — Proper-Time Frontier Go/No-Go (Qwen-Coder on HumanEval+)

> **Audience: an autonomous coding agent (Codex / Claude Code).**
> Build the complete project described below. Follow the structure, file
> contracts, and conventions exactly. Where a choice is left open, pick the
> simplest option that satisfies the contract and note it in a `NOTES.md`.

---

## 0. What this project does (one paragraph)

We test a single empirical claim: on a verifiable coding task, three
same-family models of different sizes do **not** share one cost-optimal winner
across difficulty strata ("modes"); instead the **mode-conditional proper-time
frontier is non-degenerate**, and the ranking by proper time disagrees with the
ranking by accuracy. The pipeline runs **Qwen2.5-Coder 1.5B / 7B / 32B** on
**HumanEval+**, with repeated sampling until the unit tests pass, records
**tokens-to-first-pass** (the resource clock), partitions problems into
difficulty modes, estimates the proper time `tau* = inf_t t / F(t)` per
(model, mode) cell via Kaplan–Meier, and plots the frontier. This is a
**go/no-go**: if the frontier is non-degenerate here, the larger paper is worth
building; if not, stop.

---

## 1. Hard requirements and conventions

These are non-negotiable. Apply them throughout.

- **Python environment.** Use a `.venv` in the project root. If none exists,
  create it with `uv` (`uv venv .venv`). **Never use the system/base Python.**
  Install dependencies with `uv pip install`. Pin versions in
  `requirements.txt`.
- **Separate data generation from plotting.** Running models and computing
  estimators **writes data to disk**; plotting **reads from disk only**. A plot
  must be regenerable without re-running any model. No script both runs a model
  and draws a figure.
- **Long jobs are not launched autonomously.** The agent implements and
  validates everything with a **smoke test** (≤5 problems, 2 samples each). It
  then prints the exact command for the full run and **stops**. The human
  launches the full run. Do not start the full benchmark yourself.
- **Progress + console metrics.** Every loop that can take more than a few
  seconds uses `tqdm`. Long jobs log running metrics (problems done, current
  pass rate, tokens so far) to the console.
- **Resource clock = generated tokens, cumulative across retries.** The cost of
  a run is the **sum of generated tokens across all sampling attempts up to and
  including the first passing attempt**. This is the restart cost the theory
  measures. Do **not** record only the winning attempt's tokens. Also log
  wall-clock seconds as a secondary clock.
- **Censoring is kept, not discarded.** A problem the model never solves within
  the attempt budget is **right-censored** (`tau = inf`, `solved = false`), and
  is kept in the data. The estimator handles it via Kaplan–Meier; do not drop
  failed runs.
- **Determinism where possible.** Fix and log all seeds. Log the resolved model
  identifier, quantization, sampling params, and dataset version with every run
  so results are reproducible.

---

## 2. Project structure

Create exactly this layout.

```
proper-time-frontier/
├── README.md                  # how to set up and run (you write this)
├── NOTES.md                   # any choices you made; left-open decisions
├── requirements.txt           # pinned deps
├── .venv/                     # uv-created; not committed
├── config/
│   └── experiment.yaml        # all knobs: models, sampling, modes, budgets
├── src/
│   ├── __init__.py
│   ├── models.py              # model loading + generation (Ollama or MLX)
│   ├── dataset.py             # load HumanEval+, assign difficulty modes
│   ├── execute.py             # run candidate code against unit tests (sandbox)
│   ├── run_eval.py            # MAIN: repeated sampling, writes raw run logs
│   ├── estimate.py            # KM success curves + tau* per cell -> writes data
│   └── plot.py                # READS data only -> writes figures
├── scripts/
│   ├── smoke_test.sh          # tiny end-to-end check (≤5 problems)
│   └── full_run.sh            # the command the human will launch
├── data/
│   ├── raw/                   # per-run JSONL logs (run_eval.py output)
│   └── derived/               # tau* tables, success curves (estimate.py output)
└── figures/                   # plot.py output (PNG/PDF)
```

---

## 3. Dependencies

Pin these in `requirements.txt` (versions are a floor; resolve latest compatible):

```
numpy
scipy
matplotlib
pyyaml
tqdm
datasets            # HuggingFace, to fetch HumanEval+
lifelines           # Kaplan–Meier (or implement KM by hand; see §7)
```

Model backend — choose **one** and document it in NOTES.md:

- **Ollama** (recommended for simplicity): no Python dep; the agent shells out
  to a local Ollama server. Requires the human to have Ollama installed and the
  models pulled (see §4). Use the Ollama HTTP API at `http://localhost:11434`.
- **MLX** (faster on Apple Silicon for the 32B): add `mlx-lm` to requirements.
  Loads models in-process.

Default to **Ollama**. If the 32B is too slow under Ollama, document MLX as the
fallback in NOTES.md but do not switch without the human's say-so.

---

## 4. Models — what to download and from where

**Family: Qwen2.5-Coder. Three sizes: 1.5B, 7B, 32B.** Same family isolates the
size axis. All run 4-bit on a 64GB M1 Max (≈1GB / 5GB / 20GB respectively).

### If using Ollama
The human runs these once before the experiment (put them in README.md, do not
run them yourself):

```bash
ollama pull qwen2.5-coder:1.5b
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:32b
```

Ollama resolves to 4-bit quantized GGUF by default. Log the exact tag and
digest returned by `ollama show <model>` for reproducibility.

### If using MLX
Pull 4-bit MLX builds from the `mlx-community` org on HuggingFace, e.g.:

- `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`
- `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`
- `mlx-community/Qwen2.5-Coder-32B-Instruct-4bit`

(Verify exact repo names at fetch time; prefer the `-Instruct-4bit` variants.)

The three model identifiers go in `config/experiment.yaml` under `models:`.

---

## 5. Dataset — HumanEval+

Source: the **EvalPlus** project. HumanEval+ augments the original 164 HumanEval
problems with many more unit tests, making the verifier strict.

- Fetch via HuggingFace `datasets`: load `evalplus/humanevalplus`
  (verify the exact dataset id at implementation time; if unavailable, fall
  back to the `evalplus` PyPI package which ships the problems and test runner).
- Each problem provides: a `task_id`, a `prompt` (function signature + docstring),
  a `canonical_solution`, an `entry_point`, and the plus test suite.

### Mode assignment (difficulty strata)

Partition the 164 problems into **3 difficulty modes**, frozen before any model
runs. Use a cheap, deterministic difficulty proxy. Implement **both** and pick
one in config (default: `ref_solution_length`):

- `ref_solution_length`: tokenize the canonical solution; tertiles of length →
  {easy, medium, hard}.
- `num_tests`: number of plus unit tests for the problem; tertiles → modes.

Freeze the assignment to a file `data/derived/modes.json`
(`{task_id: mode}`) so it is identical across models. Aim for ~55 problems per
mode. Log the chosen proxy and the tertile cutoffs.

> **Why 3 modes, not more:** HumanEval+ has only 164 problems. Three modes keep
> ~55 problems/cell, enough to estimate tau* with usable (if wide) bootstrap CIs.
> Do not split into more than 3 modes for this dataset.

---

## 6. `run_eval.py` — the measurement loop

This is the core. Contract:

**Input:** `config/experiment.yaml` (models, sampling params, attempt budget,
modes file).

**For each (model, problem):**
1. Repeated sampling: generate a candidate completion at temperature `T`
   (config; default 0.6, top_p 0.95). Build the full program (prompt + completion).
2. Execute against the HumanEval+ test suite in a **sandboxed subprocess** with
   a hard timeout (see §8). Record pass/fail.
3. Accumulate `tokens_generated` for this attempt into a running total for the
   problem. Also accumulate wall-clock seconds.
4. Stop at the first passing attempt → record `solved=true`,
   `tau_tokens = cumulative tokens`, `tau_seconds = cumulative seconds`,
   `n_attempts`.
5. If no attempt passes within `max_attempts` (config; default 10) →
   `solved=false`, `tau_tokens=inf`, `tau_seconds=inf` (censored).

**Output:** append one JSON object per (model, problem) to
`data/raw/runs_<model>_<timestamp>.jsonl`. **One line per problem.** Schema
(this is the contract `estimate.py` depends on — do not change field names):

```json
{
  "model": "qwen2.5-coder:7b",
  "model_digest": "sha256:...",
  "task_id": "HumanEval/12",
  "mode": "hard",
  "solved": true,
  "tau_tokens": 1843,
  "tau_seconds": 14.7,
  "n_attempts": 3,
  "max_attempts": 10,
  "temperature": 0.6,
  "top_p": 0.95,
  "seed": 0,
  "dataset": "humanevalplus",
  "timestamp": "2026-06-16T...Z"
}
```

Use `tqdm` over problems. Every N problems, log to console: problems done,
running solve rate for the current model, mean tokens-to-pass so far.

**Resumability:** before running a (model, problem), check whether it already
exists in the raw logs for this run id; skip if present. This lets a killed
run resume.

---

## 7. `estimate.py` — proper time per cell

Reads `data/raw/*.jsonl`, writes `data/derived/`. **No model calls. No plotting.**

For each (model, mode) cell:

1. Collect `tau_tokens` and `solved` for all problems in the cell.
2. **Kaplan–Meier success curve** `F_hat(t)` over the resource grid (censored
   entries = unsolved problems). If using `lifelines`, fit `KaplanMeierFitter`
   on event = solved, duration = tau_tokens; the success curve is
   `1 - survival`. If implementing by hand, the empirical success-by-budget with
   a censoring floor is acceptable: `F_hat(t) = (#solved with tau<=t) / N`.
3. **Proper time:** `tau_star = min_t  t / max(F_hat(t), alpha)` over the
   observed budget grid, with censoring floor `alpha` from config (default 1e-3).
4. **Bootstrap CI:** resample problems within the cell B times (config B=1000),
   recompute tau_star, report 16/84 and 2.5/97.5 percentiles. Report CIs on the
   **log2 ratio** between models within a mode, not on raw tau_star.

Write two files:

- `data/derived/tau_star.json`:
  `{ "<model>|<mode>": {"tau_star": float, "ci_lo": float, "ci_hi": float,
     "n": int, "n_solved": int} }`
- `data/derived/success_curves.json`:
  `{ "<model>|<mode>": {"grid": [...], "F": [...]} }`  (for the success-by-budget
  plot)

Also compute and write `data/derived/frontier.json`:
- per mode, the `argmin_model tau_star` (the frontier winner) and whether its CI
  is separated from the runner-up (`separated: true/false`).
- per model, average accuracy (solve rate) across modes, for the
  accuracy-vs-proper-time disagreement plot.

---

## 8. `execute.py` — safe test execution

Running model-generated code requires care.

- Execute each candidate in a **subprocess** with `subprocess.run`, a hard
  **timeout** (config; default 10s per attempt), and resource limits if
  available (`resource.setrlimit` for CPU/memory on macOS where supported).
- Capture stdout/stderr; a timeout or exception = fail, not crash.
- Do the execution in a temp directory; clean up after.
- Never `exec`/`eval` model output in the main process.

> The plus test harness from the `evalplus` package can be reused if present;
> prefer it over hand-rolling the test runner, since it defines the exact pass
> criterion. Document in NOTES.md which path you took.

---

## 9. `plot.py` — figures (reads derived data only)

Three figures, matching the paper's target figures. **Reads
`data/derived/*.json` only.** No model calls.

1. **`figures/fig_frontier.png`** — grouped bars: x = modes, groups = models,
   y = tau_star, with bootstrap CI whiskers. Mark the per-mode winner. This is
   the go/no-go figure: are the winners different across modes, with separated
   CIs?
2. **`figures/fig_disagreement.png`** — scatter: x = average accuracy (solve
   rate), y = tau_star (inverted axis, lower=better). One point per model; a
   star for the mode-conditional frontier policy. Shows accuracy vs proper-time
   ranking disagreement.
3. **`figures/fig_success_curves.png`** — the KM success-by-budget curves
   `F(t)` per model, one panel per mode, with the tau_star operating point
   marked. This is the diagnostic behind tau_star.

Style: minimal, no chartjunk. Save both PNG (130 dpi) and PDF.

---

## 10. `config/experiment.yaml` — all knobs in one place

```yaml
backend: ollama            # or: mlx
models:
  - qwen2.5-coder:1.5b
  - qwen2.5-coder:7b
  - qwen2.5-coder:32b
dataset: humanevalplus
mode_proxy: ref_solution_length   # or: num_tests
n_modes: 3
sampling:
  temperature: 0.6
  top_p: 0.95
  max_new_tokens: 1024
  max_attempts: 10        # repeated-sampling budget per problem
  seed: 0
execution:
  timeout_seconds: 10
estimation:
  censoring_floor: 1.0e-3
  bootstrap_B: 1000
paths:
  raw: data/raw
  derived: data/derived
  figures: figures
```

---

## 11. Scripts

**`scripts/smoke_test.sh`** — end-to-end on a tiny subset, must run in a couple
of minutes:
- temporarily restrict to **5 problems** and **2 max_attempts**, only the
  **1.5B** model (fastest);
- run `run_eval.py` → `estimate.py` → `plot.py`;
- assert: raw JSONL has 5 lines, derived tau_star.json has cells, figures exist.
- This is what the agent runs to validate. **The agent runs the smoke test.**

**`scripts/full_run.sh`** — the real run over all 3 models × 164 problems:
- this is the command the **human** launches; the agent writes it and **does not
  execute it**.
- print an ETA estimate at start (rough: tokens/sec × expected tokens).
- log progress to console and to a logfile under `data/raw/`.

---

## 12. Acceptance checklist (the agent verifies all of these)

- [ ] `.venv` created with `uv`; base Python never used; `requirements.txt` pinned.
- [ ] `modes.json` frozen before any run; ~55 problems/mode.
- [ ] `run_eval.py` records **cumulative tokens across retries**, not just the
      winning attempt.
- [ ] Censored (unsolved) problems are written with `tau_tokens=inf`,
      `solved=false`, and kept.
- [ ] `estimate.py` makes **no** model calls; `plot.py` makes **no** model calls
      and reads only `data/derived`.
- [ ] tau_star uses KM with censoring floor; CIs are bootstrap on log2 ratios.
- [ ] Smoke test passes end-to-end (5 problems, 1.5B, 2 attempts).
- [ ] `full_run.sh` is written, prints an ETA, and is **not executed** by the
      agent.
- [ ] README explains: install (uv), pull models (Ollama), run smoke, run full,
      regenerate plots from data without re-running models.

---

## 13. The go/no-go decision (state this in README)

After the full run, read `figures/fig_frontier.png` and
`data/derived/frontier.json`:

- **GO** if the per-mode frontier winner differs across modes **with separated
  bootstrap CIs** — the proper-time frontier is non-degenerate, and (per
  `fig_disagreement.png`) the proper-time ranking differs from the accuracy
  ranking. The larger paper is warranted.
- **NO-GO / inconclusive** if one model wins every mode, or if CIs overlap
  everywhere. Note: with 164 problems, overlapping CIs may be low statistical
  power rather than a true degenerate frontier — the documented next step is to
  re-run on **MBPP+** (~400 problems) before concluding.
