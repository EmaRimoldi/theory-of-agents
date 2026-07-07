from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.baselines import (
    always_allocation,
    best_single_model,
    oracle_allocation,
    statistical_oracle_allocation,
)
from src.load_traces import MODEL_ORDER, load_config, load_traces
from src.router import CodexCLIRouter, MockRouter, OpenAIRouter, decision_alloc_json
from src.simulate import allocation_to_json_keys, execute_allocation


DEFAULT_MODE_ORDER = ["easy", "medium", "hard"]


def read_existing_results(path: Path) -> set[tuple[int, str]]:
    if not path.exists():
        return set()
    seen: set[tuple[int, str]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            seen.add((int(row["fold"]), str(row["task_id"])))
    return seen


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True))
        f.write("\n")


def ensure_folds(
    traces: dict[str, dict[str, Any]],
    *,
    path: Path,
    n_folds: int,
    seed: int,
    mode_order: list[str],
) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            folds = json.load(f)
        assigned = [task_id for fold in folds["folds"] for task_id in fold["task_ids"]]
        if sorted(assigned) != sorted(traces):
            raise ValueError(f"Existing {path} does not match loaded traces")
        return folds

    rng = np.random.default_rng(seed)
    fold_lists: list[list[str]] = [[] for _ in range(n_folds)]
    trace_anchor_model = next(iter(next(iter(traces.values())).keys()))
    seen_modes = {traces[task_id][trace_anchor_model]["mode"] for task_id in traces}
    ordered_modes = [mode for mode in mode_order if mode in seen_modes]
    ordered_modes.extend(sorted(seen_modes - set(ordered_modes)))
    for mode in ordered_modes:
        task_ids = [
            task_id
            for task_id in sorted(traces)
            if traces[task_id][trace_anchor_model]["mode"] == mode
        ]
        shuffled = list(task_ids)
        rng.shuffle(shuffled)
        for idx, task_id in enumerate(shuffled):
            fold_lists[idx % n_folds].append(task_id)

    folds = {
        "seed": seed,
        "n_folds": n_folds,
        "stratify_by": "mode",
        "mode_order": ordered_modes,
        "folds": [
            {"fold": idx, "task_ids": sorted(task_ids)}
            for idx, task_ids in enumerate(fold_lists)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(folds, f, indent=2, sort_keys=True)
        f.write("\n")
    return folds


def parse_fold_filter(value: str | None, n_folds: int) -> set[int]:
    if value is None:
        return set(range(n_folds))
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def mode_order_from_config(config: dict[str, Any]) -> list[str]:
    return [str(mode) for mode in config.get("mode_order", DEFAULT_MODE_ORDER)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retry-allocation router experiment.")
    parser.add_argument("--config", default="config/router_experiment.yaml")
    parser.add_argument("--folds", default=None, help="Comma-separated fold ids to run.")
    parser.add_argument("--limit-test-problems", type=int, default=None)
    parser.add_argument("--mock-router", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--results-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    traces = load_traces(config)
    budget = int(config["budget_attempts"])
    order = tuple(config.get("execution_order", list(MODEL_ORDER)))
    mode_order = mode_order_from_config(config)

    derived_dir = Path(config["paths"]["results"]).parent
    folds_path = Path(config["paths"].get("folds", derived_dir / "folds.json"))
    kfold_config = config["kfold"]
    folds = ensure_folds(
        traces,
        path=folds_path,
        n_folds=int(kfold_config["n_folds"]),
        seed=int(kfold_config["seed"]),
        mode_order=mode_order,
    )
    selected_folds = parse_fold_filter(args.folds, int(folds["n_folds"]))
    fold_by_id = {int(fold["fold"]): fold for fold in folds["folds"]}

    results_path = Path(args.results_path or config["paths"]["results"])
    if args.overwrite and results_path.exists():
        results_path.unlink()
    existing = read_existing_results(results_path)

    if args.mock_router:
        router = MockRouter(config["router"]["parse_fail_default"], budget=budget, model_order=order)
    elif config["router"].get("backend") == "codex_cli":
        router = CodexCLIRouter(config, budget=budget)
    else:
        router = OpenAIRouter(config, budget=budget)

    best_model, single_rates = best_single_model(traces, budget=budget, order=order)
    all_task_ids = set(traces)
    total_to_run = 0
    fold_task_lists: list[tuple[int, list[str]]] = []
    for fold in folds["folds"]:
        fold_id = int(fold["fold"])
        if fold_id not in selected_folds:
            continue
        task_ids = list(fold["task_ids"])
        if args.limit_test_problems is not None:
            task_ids = task_ids[: args.limit_test_problems]
        fold_task_lists.append((fold_id, task_ids))
        total_to_run += sum((fold_id, task_id) not in existing for task_id in task_ids)

    solved_so_far = 0
    time_so_far = 0.0
    completed = 0
    progress = tqdm(total=total_to_run, desc="router decisions", unit="problem")
    for fold_id, test_task_ids in fold_task_lists:
        test_set = set(test_task_ids)
        train_task_ids = sorted(all_task_ids - set(fold_by_id[fold_id]["task_ids"]))
        if test_set & set(train_task_ids):
            raise ValueError(f"Leakage detected for fold {fold_id}")

        for task_id in test_task_ids:
            if (fold_id, task_id) in existing:
                continue
            mode = traces[task_id][order[0]]["mode"]
            decision = router.decide(
                train_task_ids=train_task_ids,
                traces=traces,
                test_mode=mode,
                budget=budget,
            )
            router_result = execute_allocation(task_id, decision.alloc, traces, order)
            record: dict[str, Any] = {
                "fold": fold_id,
                "task_id": task_id,
                "mode": mode,
                "budget_attempts": budget,
                "execution_order": list(order),
                "router_alloc": decision_alloc_json(decision, order),
                "router_model": decision.router_model,
                "router_backend": decision.router_backend,
                "router_reasoning_effort": decision.reasoning_effort,
                "router_parse_failed": decision.parse_failed,
                "router_prompt_kind": decision.prompt_kind,
                "router": router_result,
                "best_single_model": best_model,
                "best_single_full_solved_rates": single_rates,
            }

            for model in order:
                key = f"always_{model}"
                alloc = always_allocation(model, budget, order=order)
                record[key] = execute_allocation(task_id, alloc, traces, order)
                record[f"{key}_alloc"] = allocation_to_json_keys(alloc, order=order)

            best_alloc = always_allocation(best_model, budget, order=order)
            record["best_single"] = execute_allocation(task_id, best_alloc, traces, order)
            record["best_single_alloc"] = allocation_to_json_keys(best_alloc, order=order)

            oracle_alloc, oracle_result = oracle_allocation(task_id, traces, budget=budget, order=order)
            record["oracle"] = oracle_result
            record["oracle_alloc"] = allocation_to_json_keys(oracle_alloc, order=order)

            stat_alloc, stat_metrics = statistical_oracle_allocation(
                train_task_ids=train_task_ids,
                test_mode=mode,
                traces=traces,
                budget=budget,
                order=order,
            )
            record["stat_oracle_alloc"] = allocation_to_json_keys(stat_alloc, order=order)
            record["stat_oracle_train_metrics"] = stat_metrics
            record["stat_oracle"] = execute_allocation(task_id, stat_alloc, traces, order)

            append_jsonl(results_path, record)
            completed += 1
            solved_so_far += int(record["router"]["solved"])
            time_so_far += float(record["router"]["time_seconds"])
            progress.update(1)
            progress.set_postfix(
                solved_rate=f"{solved_so_far / completed:.3f}",
                mean_time=f"{time_so_far / completed:.3f}",
            )
    progress.close()
    print(f"Wrote router results to {results_path}")


if __name__ == "__main__":
    main()
