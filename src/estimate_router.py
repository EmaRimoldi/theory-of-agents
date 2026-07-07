from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.load_traces import MODEL_ORDER, configured_model_order, load_config


BASE_METHODS = ["router", "best_single", "oracle", "stat_oracle"]


DEFAULT_MODE_ORDER = ["easy", "medium", "hard"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def percentile_ci(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"lo": None, "hi": None}
    return {
        "lo": float(np.percentile(values, 2.5)),
        "hi": float(np.percentile(values, 97.5)),
    }


def method_metrics(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    results = [record[method] for record in records]
    solved = [bool(result["solved"]) for result in results]
    times = [float(result["time_seconds"]) for result in results]
    solved_times = [time for time, ok in zip(times, solved) if ok]
    return {
        "n": len(records),
        "solved": int(sum(solved)),
        "solved_rate": float(np.mean(solved)) if records else 0.0,
        "total_time_all": float(np.sum(times)) if records else 0.0,
        "mean_time_all": float(np.mean(times)) if records else None,
        "total_time_solved": float(np.sum(solved_times)) if solved_times else 0.0,
        "mean_time_solved": mean_or_none(solved_times),
    }


def bootstrap_stat(
    records: list[dict[str, Any]],
    stat_fn: Callable[[list[dict[str, Any]]], float],
    *,
    B: int,
    seed: int = 0,
) -> dict[str, Any]:
    if not records:
        return {"mean": None, "ci95": {"lo": None, "hi": None}}
    rng = np.random.default_rng(seed)
    n = len(records)
    values = np.empty(B, dtype=float)
    for idx in range(B):
        sample = [records[int(i)] for i in rng.integers(0, n, size=n)]
        values[idx] = stat_fn(sample)
    return {
        "mean": float(stat_fn(records)),
        "ci95": percentile_ci(values),
    }


def mean_time_delta(records: list[dict[str, Any]], left: str, right: str) -> float:
    return float(
        np.mean(
            [
                float(record[left]["time_seconds"]) - float(record[right]["time_seconds"])
                for record in records
            ]
        )
    )


def time_saved_vs_best(records: list[dict[str, Any]]) -> float:
    return mean_time_delta(records, "best_single", "router")


def router_gap_to_stat(records: list[dict[str, Any]]) -> float:
    return mean_time_delta(records, "router", "stat_oracle")


def router_gap_to_oracle(records: list[dict[str, Any]]) -> float:
    return mean_time_delta(records, "router", "oracle")


def stat_gap_to_oracle(records: list[dict[str, Any]]) -> float:
    return mean_time_delta(records, "stat_oracle", "oracle")


def methods_for_order(model_order: tuple[str, ...]) -> list[str]:
    return ["router", *[f"always_{model}" for model in model_order], "best_single", "oracle", "stat_oracle"]


def alloc_to_values(alloc: dict[str, Any], model_order: tuple[str, ...]) -> list[int]:
    return [int(alloc.get(f"n_{model}", alloc.get(model, 0))) for model in model_order]


def allocation_summary(
    records: list[dict[str, Any]],
    key: str,
    model_order: tuple[str, ...],
) -> dict[str, Any]:
    vectors = np.array([alloc_to_values(record[key], model_order) for record in records], dtype=float)
    kinds: Counter[str] = Counter()
    vector_counts: Counter[str] = Counter()
    for record in records:
        values = alloc_to_values(record[key], model_order)
        positive = sum(value > 0 for value in values)
        kinds["commit" if positive == 1 else "split" if positive > 1 else "zero"] += 1
        vector_counts[json.dumps({f"n_{model}": values[i] for i, model in enumerate(model_order)}, sort_keys=True)] += 1
    if vectors.size == 0:
        mean_alloc = {model: 0.0 for model in model_order}
    else:
        mean_alloc = {
            model: float(vectors[:, idx].mean())
            for idx, model in enumerate(model_order)
        }
    return {
        "mean_allocation": mean_alloc,
        "kind_counts": dict(kinds),
        "vector_counts": dict(vector_counts),
    }


def ordered_modes(records: list[dict[str, Any]], mode_order: list[str] | None = None) -> list[str]:
    available = {str(record["mode"]) for record in records}
    order = mode_order or DEFAULT_MODE_ORDER
    modes = [mode for mode in order if mode in available]
    modes.extend(sorted(available - set(modes)))
    return modes


def build_summary(
    records: list[dict[str, Any]],
    *,
    B: int,
    mode_order: list[str] | None = None,
    model_order: tuple[str, ...] = MODEL_ORDER,
) -> dict[str, Any]:
    best_single_model = records[0].get("best_single_model") if records else None
    parse_failures = sum(bool(record.get("router_parse_failed", False)) for record in records)
    modes = ordered_modes(records, mode_order)
    methods = methods_for_order(model_order)

    summary: dict[str, Any] = {
        "n_records": len(records),
        "best_single_model": best_single_model,
        "parse_failures": parse_failures,
        "mode_order": modes,
        "model_order": list(model_order),
        "methods": {method: method_metrics(records, method) for method in methods},
        "time_saved_vs_best_single": bootstrap_stat(records, time_saved_vs_best, B=B, seed=0),
        "router_gap_to_stat_oracle": bootstrap_stat(records, router_gap_to_stat, B=B, seed=1),
        "router_gap_to_oracle": bootstrap_stat(records, router_gap_to_oracle, B=B, seed=2),
        "stat_oracle_gap_to_oracle": bootstrap_stat(records, stat_gap_to_oracle, B=B, seed=3),
        "allocation_distribution": {
            "router": allocation_summary(records, "router_alloc", model_order),
            "stat_oracle": allocation_summary(records, "stat_oracle_alloc", model_order),
        },
        "by_mode": {},
    }

    for mode in modes:
        mode_records = [record for record in records if record["mode"] == mode]
        summary["by_mode"][mode] = {
            "n_records": len(mode_records),
            "methods": {method: method_metrics(mode_records, method) for method in methods},
            "time_saved_vs_best_single": bootstrap_stat(
                mode_records, time_saved_vs_best, B=B, seed=10 + len(mode)
            ),
            "router_gap_to_stat_oracle": bootstrap_stat(
                mode_records, router_gap_to_stat, B=B, seed=20 + len(mode)
            ),
            "allocation_distribution": {
                "router": allocation_summary(mode_records, "router_alloc", model_order),
                "stat_oracle": allocation_summary(mode_records, "stat_oracle_alloc", model_order),
            },
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate router experiment results.")
    parser.add_argument("--config", default="config/router_experiment.yaml")
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--bootstrap-B", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    results_path = Path(args.results_path or config["paths"]["results"])
    summary_path = Path(args.summary_path or config["paths"]["summary"])
    B = int(args.bootstrap_B or config["estimation"]["bootstrap_B"])
    records = read_jsonl(results_path)
    if not records:
        raise SystemExit(f"No router records found in {results_path}")
    mode_order = [str(mode) for mode in config.get("mode_order", DEFAULT_MODE_ORDER)]
    model_order = configured_model_order(config)
    summary = build_summary(records, B=B, mode_order=mode_order, model_order=model_order)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote router summary to {summary_path}")


if __name__ == "__main__":
    main()
