from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset import load_config, load_dataset
from src.execute import execute_candidate
from src.models import (
    ModelUnavailableError,
    build_solution,
    generate_ollama,
    make_completion_prompt,
    resolve_model_info,
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


def read_existing_tasks(path: Path) -> set[str]:
    if not path.exists():
        return set()
    tasks: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                tasks.add(json.loads(line)["task_id"])
            except Exception:
                continue
    return tasks


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True))
        f.write("\n")


def select_task_ids(
    task_ids: list[str],
    limit: int | None,
    *,
    modes: dict[str, str] | None = None,
) -> list[str]:
    if limit is None:
        return task_ids
    if modes:
        grouped: dict[str, list[str]] = {}
        for task_id in task_ids:
            grouped.setdefault(modes[task_id], []).append(task_id)
        selected: list[str] = []
        ordered_modes = sorted(grouped)
        while len(selected) < limit and any(grouped.values()):
            for mode in ordered_modes:
                if grouped[mode] and len(selected) < limit:
                    selected.append(grouped[mode].pop(0))
        return selected
    return task_ids[:limit]


def parse_task_ids(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def run_model(
    *,
    model: str,
    config: dict[str, Any],
    run_id: str,
    limit_problems: int | None,
    max_attempts_override: int | None,
    task_ids_override: list[str] | None,
) -> None:
    if config.get("backend", "ollama") != "ollama":
        raise ValueError("This implementation supports backend: ollama")

    raw_dir = Path(config["paths"]["raw"])
    out_path = raw_dir / f"runs_{safe_model_name(model)}_{run_id}.jsonl"
    existing_tasks = read_existing_tasks(out_path)

    bundle = load_dataset(config)
    model_info = resolve_model_info(model)

    sampling = config["sampling"]
    execution = config["execution"]
    max_attempts = int(max_attempts_override or sampling["max_attempts"])
    temperature = float(sampling["temperature"])
    top_p = float(sampling["top_p"])
    max_new_tokens = int(sampling["max_new_tokens"])
    seed = int(sampling["seed"])
    timeout_seconds = float(execution["timeout_seconds"])

    if task_ids_override is not None:
        missing = [task_id for task_id in task_ids_override if task_id not in bundle.problems]
        if missing:
            raise ValueError(f"Requested task ids not present in {bundle.dataset_name}: {missing}")
        task_ids = task_ids_override
    else:
        task_ids = select_task_ids(
            sorted(bundle.problems),
            limit_problems,
            modes=(bundle.modes if bundle.dataset_name == "bbh" else None),
        )
    task_ids = [task_id for task_id in task_ids if task_id not in existing_tasks]

    solved_count = 0
    completed_count = len(existing_tasks)
    tokens_so_far = 0
    solved_token_totals: list[int] = []

    print(
        f"Running {model} ({model_info.digest}) on {len(task_ids)} remaining problems; "
        f"output={out_path}"
    )
    for task_id in tqdm(task_ids, desc=f"{model} problems", unit="problem"):
        problem = bundle.problems[task_id]
        prompt = make_completion_prompt(problem, bundle.dataset_name)
        cumulative_tokens = 0
        cumulative_seconds = 0.0
        solved = False
        attempt_token_counts: list[int] = []
        attempt_seconds: list[float] = []
        attempt_statuses: list[str] = []
        attempt_extracted_answers: list[str] = []
        attempt_normalized_answers: list[str] = []

        for attempt_idx in range(max_attempts):
            attempt_seed = seed + attempt_idx
            generation = generate_ollama(
                model,
                prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                seed=attempt_seed,
            )
            cumulative_tokens += generation.tokens_generated
            cumulative_seconds += generation.wall_seconds
            tokens_so_far += generation.tokens_generated
            attempt_token_counts.append(generation.tokens_generated)
            attempt_seconds.append(generation.wall_seconds)

            if bundle.checker_dataset == "bbh":
                from src.bbh_data import BBH_EXTRACTION_RULE, evaluate_bbh_completion

                bbh_result = evaluate_bbh_completion(
                    generation.text,
                    str(bundle.expected_outputs[task_id]),
                )
                result = bbh_result.execution
                attempt_extracted_answers.append(bbh_result.extracted_answer)
                attempt_normalized_answers.append(bbh_result.normalized_prediction)
            else:
                solution = build_solution(problem, generation.text)
                result = execute_candidate(
                    solution=solution,
                    problem=problem,
                    expected_output=bundle.expected_outputs[task_id],
                    timeout_seconds=timeout_seconds,
                    checker_dataset=bundle.checker_dataset,
                )
            status = "pass" if result.passed else f"{result.base_status}/{result.plus_status}"
            attempt_statuses.append(status)
            if result.passed:
                solved = True
                break

        completed_count += 1
        if solved:
            solved_count += 1
            tau_tokens: float | int = cumulative_tokens
            tau_seconds: float = cumulative_seconds
            solved_token_totals.append(cumulative_tokens)
        else:
            tau_tokens = math.inf
            tau_seconds = math.inf

        row = {
            "model": model,
            "model_digest": model_info.digest,
            "model_quantization": model_info.quantization,
            "task_id": task_id,
            "mode": bundle.modes[task_id],
            "solved": solved,
            "tau_tokens": tau_tokens,
            "tau_seconds": tau_seconds,
            "n_attempts": len(attempt_token_counts),
            "max_attempts": max_attempts,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "dataset": bundle.dataset_name,
            "checker_dataset": bundle.checker_dataset,
            "dataset_hash": bundle.dataset_hash,
            "timestamp": utc_timestamp(),
            "run_id": run_id,
            "tokens_generated_total": cumulative_tokens,
            "seconds_spent_total": cumulative_seconds,
            "attempt_token_counts": attempt_token_counts,
            "attempt_seconds": attempt_seconds,
            "attempt_statuses": attempt_statuses,
        }
        if bundle.checker_dataset == "bbh":
            from src.bbh_data import BBH_EXTRACTION_RULE, normalize_bbh_answer

            row.update(
                {
                    "answer_extraction_rule": BBH_EXTRACTION_RULE,
                    "target_answer": str(bundle.expected_outputs[task_id]),
                    "normalized_target_answer": normalize_bbh_answer(
                        str(bundle.expected_outputs[task_id]),
                        target=str(bundle.expected_outputs[task_id]),
                    ),
                    "subtask": problem["subtask"],
                    "attempt_extracted_answers": attempt_extracted_answers,
                    "attempt_normalized_answers": attempt_normalized_answers,
                }
            )
        write_jsonl(out_path, row)

        if completed_count % 5 == 0 or completed_count == len(task_ids):
            solve_rate = solved_count / max(1, completed_count - len(existing_tasks))
            mean_tokens = mean(solved_token_totals) if solved_token_totals else float("nan")
            print(
                f"[{model}] done={completed_count} "
                f"current_solve_rate={solve_rate:.3f} "
                f"mean_tokens_to_pass={mean_tokens:.1f} "
                f"tokens_so_far={tokens_so_far}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated-sampling evaluation.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--models", default=None, help="Comma-separated model override.")
    parser.add_argument("--limit-problems", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--task-ids", default=None, help="Comma-separated explicit task ids.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else list(config["models"])
    )

    try:
        for model in models:
            run_model(
                model=model,
                config=config,
                run_id=run_id,
                limit_problems=args.limit_problems,
                max_attempts_override=args.max_attempts,
                task_ids_override=parse_task_ids(args.task_ids),
            )
    except ModelUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
