from __future__ import annotations

from typing import Any


MODEL_ORDER = ("1.5b", "7b", "32b")


def normalize_allocation(
    alloc: dict[str, Any],
    order: tuple[str, ...] = MODEL_ORDER,
) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for model in order:
        value = alloc.get(model, alloc.get(f"n_{model}", 0))
        normalized[model] = max(0, int(value))
    return normalized


def allocation_to_json_keys(
    alloc: dict[str, Any],
    order: tuple[str, ...] = MODEL_ORDER,
) -> dict[str, int]:
    normalized = normalize_allocation(alloc, order=order)
    return {f"n_{model}": normalized[model] for model in order}


def execute_allocation(
    task_id: str,
    alloc: dict[str, Any],
    traces: dict[str, dict[str, Any]],
    order: tuple[str, ...] = MODEL_ORDER,
) -> dict[str, Any]:
    """Replay a retry allocation from logged per-attempt arrays."""

    normalized = normalize_allocation(alloc, order=order)
    time_seconds = 0.0
    tokens = 0
    n_attempts_used = 0

    for model in order:
        model_trace = traces[task_id][model]
        requested = normalized.get(model, 0)
        available = len(model_trace["attempt_statuses"])
        for idx in range(min(requested, available)):
            time_seconds += float(model_trace["attempt_seconds"][idx])
            tokens += int(model_trace["attempt_token_counts"][idx])
            n_attempts_used += 1
            if model_trace["attempt_statuses"][idx] == "pass":
                return {
                    "solved": True,
                    "time_seconds": time_seconds,
                    "tokens": tokens,
                    "n_attempts_used": n_attempts_used,
                    "first_pass_model": model,
                }

    return {
        "solved": False,
        "time_seconds": time_seconds,
        "tokens": tokens,
        "n_attempts_used": n_attempts_used,
        "first_pass_model": None,
    }
