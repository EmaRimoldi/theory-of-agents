from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from src.simulate import MODEL_ORDER, allocation_to_json_keys, execute_allocation


def always_allocation(
    model: str,
    budget: int,
    order: tuple[str, ...] = MODEL_ORDER,
) -> dict[str, int]:
    return {candidate: (budget if candidate == model else 0) for candidate in order}


def iter_allocations(
    budget: int,
    order: tuple[str, ...] = MODEL_ORDER,
) -> list[dict[str, int]]:
    allocations: list[dict[str, int]] = []

    def rec(idx: int, remaining: int, current: dict[str, int]) -> None:
        if idx == len(order):
            allocations.append(dict(current))
            return
        model = order[idx]
        for count in range(remaining + 1):
            current[model] = count
            rec(idx + 1, remaining - count, current)

    rec(0, budget, {})
    return allocations


def best_single_model(
    traces: dict[str, dict[str, Any]],
    *,
    budget: int,
    order: tuple[str, ...] = MODEL_ORDER,
) -> tuple[str, dict[str, float]]:
    rates: dict[str, float] = {}
    task_ids = sorted(traces)
    for model in order:
        alloc = always_allocation(model, budget, order=order)
        solved = sum(
            bool(execute_allocation(task_id, alloc, traces, order)["solved"])
            for task_id in task_ids
        )
        rates[model] = solved / len(task_ids)
    winner = max(order, key=lambda model: (rates[model], -order.index(model)))
    return winner, rates


def oracle_allocation(
    task_id: str,
    traces: dict[str, dict[str, Any]],
    *,
    budget: int,
    order: tuple[str, ...] = MODEL_ORDER,
) -> tuple[dict[str, int], dict[str, Any]]:
    best_alloc: dict[str, int] | None = None
    best_result: dict[str, Any] | None = None
    for alloc in iter_allocations(budget, order=order):
        result = execute_allocation(task_id, alloc, traces, order)
        if not result["solved"]:
            continue
        key = (
            float(result["time_seconds"]),
            int(result["tokens"]),
            int(result["n_attempts_used"]),
            sum(alloc.values()),
        )
        if best_result is None:
            best_alloc = alloc
            best_result = result
            best_key = key
            continue
        if key < best_key:
            best_alloc = alloc
            best_result = result
            best_key = key

    if best_alloc is None or best_result is None:
        best_alloc = {model: 0 for model in order}
        best_result = execute_allocation(task_id, best_alloc, traces, order)
    return best_alloc, best_result


def allocation_training_metrics(
    task_ids: list[str],
    traces: dict[str, dict[str, Any]],
    alloc: dict[str, int],
    *,
    order: tuple[str, ...] = MODEL_ORDER,
) -> dict[str, float]:
    results = [execute_allocation(task_id, alloc, traces, order) for task_id in task_ids]
    if not results:
        return {"solved_rate": 0.0, "mean_time": float("inf")}
    return {
        "solved_rate": float(np.mean([result["solved"] for result in results])),
        "mean_time": float(np.mean([float(result["time_seconds"]) for result in results])),
    }


def mode_best_single_rate(
    task_ids: list[str],
    traces: dict[str, dict[str, Any]],
    *,
    budget: int,
    order: tuple[str, ...] = MODEL_ORDER,
) -> float:
    rates = []
    for model in order:
        alloc = always_allocation(model, budget, order=order)
        rates.append(allocation_training_metrics(task_ids, traces, alloc, order=order)["solved_rate"])
    return max(rates) if rates else 0.0


def statistical_oracle_allocation(
    *,
    train_task_ids: list[str],
    test_mode: str,
    traces: dict[str, dict[str, Any]],
    budget: int,
    order: tuple[str, ...] = MODEL_ORDER,
) -> tuple[dict[str, int], dict[str, float]]:
    mode_train_ids = [
        task_id for task_id in train_task_ids if traces[task_id][order[0]]["mode"] == test_mode
    ]
    if not mode_train_ids:
        mode_train_ids = train_task_ids

    target_rate = mode_best_single_rate(mode_train_ids, traces, budget=budget, order=order)
    candidates: list[tuple[tuple[float, float, int], dict[str, int], dict[str, float]]] = []
    for alloc in iter_allocations(budget, order=order):
        if sum(alloc.values()) == 0:
            continue
        metrics = allocation_training_metrics(mode_train_ids, traces, alloc, order=order)
        feasible = metrics["solved_rate"] + 1e-12 >= target_rate
        score = (
            0.0 if feasible else 1.0,
            metrics["mean_time"] if feasible else -metrics["solved_rate"],
            sum(alloc.values()),
        )
        candidates.append((score, alloc, metrics))

    score, alloc, metrics = min(candidates, key=lambda item: item[0])
    metrics = dict(metrics)
    metrics["target_solved_rate"] = target_rate
    metrics["feasible"] = score[0] == 0.0
    return alloc, metrics


def allocation_kind(alloc: dict[str, Any]) -> str:
    positive = sum(1 for value in alloc.values() if int(value) > 0)
    return "commit" if positive == 1 else "split" if positive > 1 else "zero"


def allocation_counter(
    records: list[dict[str, Any]],
    key: str,
    order: tuple[str, ...] = MODEL_ORDER,
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        alloc = record[key]
        label = ",".join(f"{model}:{int(alloc.get(f'n_{model}', alloc.get(model, 0)))}" for model in order)
        counter[label] += 1
    return counter
