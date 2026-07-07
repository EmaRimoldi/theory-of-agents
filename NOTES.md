# Notes

- Project root: the current repository directory is treated as the root of the spec's `proper-time-frontier/` tree.
- Backend: Ollama, via the local HTTP API at `http://localhost:11434`.
- Dataset and verifier: the implementation uses the `evalplus` Python package for HumanEval+ data and official plus checks. This avoids reimplementing the pass criterion and still uses the EvalPlus HumanEval+ release.
- Difficulty proxy: default `ref_solution_length`, using a deterministic regex token count over `canonical_solution`. Tertiles are rank-based to keep mode sizes near 55 problems each.
- Censoring: raw logs keep unsolved tasks as `tau_tokens=Infinity` and `solved=false`. They also record `tokens_generated_total`, the finite censoring budget used by Kaplan-Meier.
- Estimation: `tau_star.json` includes raw bootstrap intervals for plotting; `frontier.json` includes bootstrap CIs for winner-vs-runner log2 ratios, which determine `separated`.
- Full-run ETA is deliberately rough and controlled by `EXPECTED_TOKENS_PER_ATTEMPT` and `TOKENS_PER_SECOND` environment variables.

## Router experiment notes

- Worker traces: the router experiment consumes only the three existing full-run JSONL files. No Qwen worker is re-run.
- Success threshold: strict `attempt_statuses[i] == "pass"` only. `pass/fail`, `fail/pass`, and `fail/fail` are failures.
- Simulator: `execute_allocation` consumes attempts sequentially in the fixed order `1.5b -> 7b -> 32b` and charges every consumed attempt up to and including the first pass.
- Attempt arrays: logs store per-attempt data as parallel arrays (`attempt_seconds`, `attempt_token_counts`, `attempt_statuses`), not nested objects.
- Allocation guard: if an allocation asks for more attempts than are logged for a model, the simulator consumes only the logged attempts. This matters mainly after a solved trace, because worker logs stop at first pass.
- Oracle allocation: for unsolved-by-all problems, the hindsight oracle returns a zero-attempt unsolved allocation. This is an upper-bound convention because the oracle knows no worker can solve the problem from the logged attempts.
- Statistical oracle: implemented as a training-fold, mode-conditional empirical optimizer over all allocations with total attempts `<= 10`. It uses only training tasks with the same mode, chooses the lowest mean-time allocation that matches the best single model's training solved rate for that mode, and falls back to max solved rate if no allocation meets the target.
- Router backend: `codex_cli` by default, using local Codex ChatGPT subscription auth. `ROUTER_MODEL=gpt-5.4` with `reasoning_effort: medium` is the selected full-run router configuration.
- Router prompt: real router calls prefer full training traces. If the serialized prompt exceeds the internal character guard, the router falls back to a per-mode summary; the smoke path uses only the mock router.
- Codex CLI router isolation: each router call runs from a temporary directory with workspace-write sandboxing and receives the training traces in the prompt. This avoids giving the router filesystem access to held-out worker logs.
- Full router run: `scripts/full_router_run.sh` estimates 164 total router calls before execution.

## MBPP+ category extension notes

- Category labels are a second MBPP+ mode schema and are namespaced under `mbpp_*_category*` and `router_mbpp_category*`; HumanEval+ difficulty-mode outputs are untouched.
- Categories are fixed to `string`, `math`, `list_ds`, and `logic_control`; the LLM prompt embeds the precedence rule from `docs/specs/CATEGORY_SPEC.md`.
- The category tagger uses Codex CLI subscription auth by default through `ROUTER_MODEL`, temperature 0 by contract, and `reasoning_effort=medium`.
- The tagger writes a resumable audit trail to `data/derived/mbpp_category_label_records.jsonl`; the frozen analysis labels are `data/derived/mbpp_modes_category.json`.
- The 2-category high-power view is frozen to `data/derived/mbpp_modes_mathnot.json`.
- `frontier_by_category.py` defaults to wall-clock seconds for tau* because the retry-allocation question is time-to-verified-solution. The same script supports `--clock tokens` if token-budget frontier diagnostics are needed later.
- The public repository keeps the canonical MBPP+ split worker traces for `1.5b`, `7b`, and `32b`; intermediate/pruned variants are local-only because broad globs can duplicate task/model traces.

## MBPP+ worker phase notes

- MBPP+ is loaded through EvalPlus as `mbppplus`, with checker dataset `mbpp` and strict base+plus checking.
- MBPP+ expected-output generation must use EvalPlus `MBPP_OUTPUT_NOT_NONE_TASKS`; using the HumanEval empty list can create unpickleable regex match oracle outputs.
- MBPP+ difficulty modes are frozen to `data/derived/mbpp_modes.json` using the same `ref_solution_length` tertile method as HumanEval+, recomputed on MBPP+ itself. Cutoffs: `[21, 34]`; counts: `easy=126`, `medium=126`, `hard=126`.
- The worker smoke artifact is `data/raw/runs_qwen2.5-coder_1.5b_smoke_mbpp.jsonl` with 5 rows. A separate diagnostic gate artifact, `data/raw/runs_qwen2.5-coder_1.5b_gate_mbpp.jsonl`, was used only to find attempt-2 passes for cumulative-clock verification.
- Reduced MBPP+ worker run: `scripts/full_run_mbpp_2models.sh` runs only `qwen2.5-coder:1.5b` and `qwen2.5-coder:7b`, with run ids prefixed `full_mbpp_2models_`. Two-model downstream configs use `execution_order: ["1.5b", "7b"]` and namespaced outputs ending in `_2models`.
- The old all-in-one `scripts/full_run_mbpp.sh` path is intentionally not part of the public workflow; the current reproducible path is the two-model worker run plus the separate 32B worker run.

## BBH reasoning experiment notes

- BBH source is Hugging Face `lukaemon/bbh`, split `test`. The implementation iterates over the actual configs exposed by the dataset package.
- Deviation from `docs/specs/BBH_SPEC.md`: the resolved `lukaemon/bbh` source exposes 27 BBH subtasks, not 23. The implementation uses all 27 available configs and records `actual_subtask_count` in `data/derived/bbh_modes_meta.json`.
- Default cap is `max_examples_per_subtask: 100`, so the full worker menu covers up to 2700 BBH examples.
- After the initial serial run exposed very slow `dyck_languages` behavior for `qwen2.5:1.5b`, the active BBH worker config was reduced to a diverse 12-subtask panel while preserving already completed valid rows. The selected subtasks are `boolean_expressions`, `causal_judgement`, `date_understanding`, `disambiguation_qa`, `dyck_languages`, `formal_fallacies`, `geometric_shapes`, `hyperbaton`, `multistep_arithmetic_two`, `navigate`, `object_counting`, and `word_sorting` for 1200 examples total.
- Workers are generalist Ollama models, not coder models: `qwen2.5:1.5b`, `qwen2.5:7b`, `qwen2.5:32b`.
- The public BBH router configs use the complete `1.5b`/`7b` saved traces only; the available `32b` BBH trace is partial and remains local-only until completed.
- BBH verifier: exact match on extracted final answer. The prompt asks for a final `Answer: <final answer>` line. Extraction rule `bbh_v1_final_answer_marker_option_label_preserve_symbolic` prefers explicit `Answer:` / `Final answer:` markers and otherwise takes the last non-empty line. Option-label targets such as `(B)` normalize to `b`; alphanumeric answers are case-folded with whitespace collapsed and terminal punctuation stripped; symbol-only answers preserve symbols and remove whitespace.
- Strict success is logged as `attempt_statuses[i] == "pass"`; failures are logged as `fail/fail`. Unsolved examples keep `tau_tokens=Infinity`, `tau_seconds=Infinity`, `solved=false`, with finite `tokens_generated_total` and `seconds_spent_total`.
- BBH cognitive family mapping:
  - `arithmetic`: `multistep_arithmetic_two`, `object_counting`.
  - `logical_deduction`: `boolean_expressions`, `causal_judgement`, `formal_fallacies`, `logical_deduction_three_objects`, `logical_deduction_five_objects`, `logical_deduction_seven_objects`, `web_of_lies`.
  - `language`: `disambiguation_qa`, `hyperbaton`, `movie_recommendation`, `ruin_names`, `salient_translation_error_detection`, `snarks`, `sports_understanding`.
  - `spatial_temporal`: `date_understanding`, `geometric_shapes`, `navigate`, `temporal_sequences`, `tracking_shuffled_objects_three_objects`, `tracking_shuffled_objects_five_objects`, `tracking_shuffled_objects_seven_objects`.
  - `algorithmic`: `dyck_languages`, `penguins_in_a_table`, `reasoning_about_colored_objects`, `word_sorting`.
- `scripts/smoke_bbh.sh` freezes `bbh_modes.json`, runs 5 examples with 2 attempts on `qwen2.5:1.5b`, checks retry sanity, then runs a mock-router smoke using `config/router_experiment_bbh_smoke.yaml`.
- `scripts/full_run_bbh.sh` and `scripts/full_router_run_bbh.sh` are written but not auto-launched by the agent.
