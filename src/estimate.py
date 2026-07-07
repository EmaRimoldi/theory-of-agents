from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from lifelines import KaplanMeierFitter

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset import load_config


def parse_float(value: Any) -> float:
    if isinstance(value, str) and value.lower() in {"inf", "infinity"}:
        return math.inf
    return float(value)


def iter_raw_rows(raw_dir: Path, run_id: str | None = None) -> Iterable[dict[str, Any]]:
    for path in sorted(raw_dir.glob("*.jsonl")):
        if run_id and run_id not in path.name:
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if run_id and row.get("run_id") != run_id:
                    continue
                yield row


def finite_duration(row: dict[str, Any]) -> float:
    solved = bool(row["solved"])
    tau = parse_float(row["tau_tokens"])
    if solved:
        return tau
    censor = row.get("tokens_generated_total")
    if censor is not None:
        return max(1.0, float(censor))
    return tau


def km_curve(durations: np.ndarray, events: np.ndarray) -> tuple[list[float], list[float]]:
    kmf = KaplanMeierFitter()
    kmf.fit(durations=durations, event_observed=events)
    survival = kmf.survival_function_.iloc[:, 0]
    grid = [float(x) for x in survival.index.to_numpy()]
    success = [float(1.0 - x) for x in survival.to_numpy()]
    return grid, success


def success_at_events(
    durations: np.ndarray,
    events: np.ndarray,
    alpha: float,
) -> tuple[float, float, list[float], list[float]]:
    grid, success = km_curve(durations, events)
    candidate_pairs = [
        (t, f)
        for t, f in zip(grid, success)
        if t > 0 and (f > 0 or not events.any())
    ]
    if not candidate_pairs:
        finite = [float(d) for d in durations if math.isfinite(float(d)) and d > 0]
        fallback = min(finite) if finite else 1.0
        return fallback / alpha, fallback, grid, success

    tau_star, tau_arg = min((t / max(f, alpha), t) for t, f in candidate_pairs)
    return float(tau_star), float(tau_arg), grid, success


def cell_tau_star(rows: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    durations = np.array([finite_duration(row) for row in rows], dtype=float)
    events = np.array([bool(row["solved"]) for row in rows], dtype=bool)

    if not np.isfinite(durations).all():
        finite = durations[np.isfinite(durations)]
        replacement = float(finite.max()) if finite.size else 1.0
        durations = np.where(np.isfinite(durations), durations, replacement)

    tau_star, tau_arg, grid, success = success_at_events(durations, events, alpha)
    return {
        "tau_star": tau_star,
        "tau_argmin": tau_arg,
        "grid": grid,
        "F": success,
        "n": len(rows),
        "n_solved": int(events.sum()),
        "accuracy": float(events.mean()) if len(events) else 0.0,
    }


def bootstrap_cell_tau(rows: list[dict[str, Any]], alpha: float, B: int, rng: np.random.Generator) -> np.ndarray:
    if not rows:
        return np.array([], dtype=float)
    values = np.empty(B, dtype=float)
    n = len(rows)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        sample = [rows[int(i)] for i in idx]
        values[b] = cell_tau_star(sample, alpha)["tau_star"]
    return values


def percentile_or_nan(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def build_frontier(
    *,
    cell_stats: dict[str, dict[str, Any]],
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]],
    boot_values: dict[tuple[str, str], np.ndarray],
    modes: list[str],
    models: list[str],
) -> dict[str, Any]:
    per_mode: dict[str, Any] = {}
    for mode in modes:
        candidates = [
            (model, cell_stats[f"{model}|{mode}"]["tau_star"])
            for model in models
            if f"{model}|{mode}" in cell_stats
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[1])
        winner, winner_tau = candidates[0]
        runner_up = candidates[1][0] if len(candidates) > 1 else None
        separated = False
        ratio_ci80 = [float("nan"), float("nan")]
        ratio_ci95 = [float("nan"), float("nan")]
        ratio_median = float("nan")
        if runner_up is not None:
            w_boot = boot_values[(winner, mode)]
            r_boot = boot_values[(runner_up, mode)]
            m = min(len(w_boot), len(r_boot))
            ratio = np.log2(w_boot[:m] / r_boot[:m])
            ratio_median = float(np.median(ratio))
            ratio_ci80 = [float(np.percentile(ratio, 16)), float(np.percentile(ratio, 84))]
            ratio_ci95 = [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))]
            separated = ratio_ci95[1] < 0.0

        per_mode[mode] = {
            "argmin_model": winner,
            "tau_star": winner_tau,
            "runner_up_model": runner_up,
            "separated": separated,
            "winner_vs_runner_log2_ratio_median": ratio_median,
            "winner_vs_runner_log2_ratio_ci80": ratio_ci80,
            "winner_vs_runner_log2_ratio_ci95": ratio_ci95,
        }

    model_accuracy: dict[str, float] = {}
    model_tau_star_weighted: dict[str, float] = {}
    for model in models:
        rows = [
            row
            for (cell_model, _mode), cell_rows in grouped_rows.items()
            if cell_model == model
            for row in cell_rows
        ]
        model_accuracy[model] = (
            float(sum(bool(row["solved"]) for row in rows) / len(rows)) if rows else 0.0
        )
        tau_terms = []
        weights = []
        for mode in modes:
            key = f"{model}|{mode}"
            if key in cell_stats:
                tau_terms.append(cell_stats[key]["tau_star"])
                weights.append(cell_stats[key]["n"])
        model_tau_star_weighted[model] = (
            float(np.average(tau_terms, weights=weights)) if tau_terms else float("nan")
        )

    frontier_rows = []
    frontier_tau_terms = []
    frontier_weights = []
    for mode, info in per_mode.items():
        model = info["argmin_model"]
        rows = grouped_rows[(model, mode)]
        frontier_rows.extend(rows)
        frontier_tau_terms.append(info["tau_star"])
        frontier_weights.append(len(rows))
    frontier_policy = {
        "accuracy": (
            float(sum(bool(row["solved"]) for row in frontier_rows) / len(frontier_rows))
            if frontier_rows
            else 0.0
        ),
        "tau_star_weighted": (
            float(np.average(frontier_tau_terms, weights=frontier_weights))
            if frontier_tau_terms
            else float("nan")
        ),
    }

    return {
        "per_mode": per_mode,
        "model_accuracy": model_accuracy,
        "model_tau_star_weighted": model_tau_star_weighted,
        "frontier_policy": frontier_policy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate proper-time frontier from raw logs.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--bootstrap-B", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    raw_dir = Path(config["paths"]["raw"])
    derived_dir = Path(config["paths"]["derived"])
    derived_dir.mkdir(parents=True, exist_ok=True)

    alpha = float(config["estimation"]["censoring_floor"])
    B = int(args.bootstrap_B or config["estimation"]["bootstrap_B"])
    seed = int(config["sampling"]["seed"])
    rng = np.random.default_rng(seed)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in iter_raw_rows(raw_dir, run_id=args.run_id):
        grouped[(row["model"], row["mode"])].append(row)

    if not grouped:
        raise SystemExit(f"No raw rows found in {raw_dir} for run_id={args.run_id!r}")

    modes = sorted({mode for _, mode in grouped}, key=["easy", "medium", "hard"].index)
    models = sorted({model for model, _ in grouped})

    tau_star: dict[str, dict[str, Any]] = {}
    success_curves: dict[str, dict[str, Any]] = {}
    boot_values: dict[tuple[str, str], np.ndarray] = {}

    for (model, mode), rows in sorted(grouped.items()):
        key = f"{model}|{mode}"
        stats = cell_tau_star(rows, alpha)
        boots = bootstrap_cell_tau(rows, alpha, B, rng)
        boot_values[(model, mode)] = boots
        tau_star[key] = {
            "tau_star": stats["tau_star"],
            "tau_argmin": stats["tau_argmin"],
            "ci_lo": percentile_or_nan(boots, 16),
            "ci_hi": percentile_or_nan(boots, 84),
            "ci95_lo": percentile_or_nan(boots, 2.5),
            "ci95_hi": percentile_or_nan(boots, 97.5),
            "n": stats["n"],
            "n_solved": stats["n_solved"],
            "accuracy": stats["accuracy"],
        }
        success_curves[key] = {"grid": stats["grid"], "F": stats["F"]}

    frontier = build_frontier(
        cell_stats=tau_star,
        grouped_rows=grouped,
        boot_values=boot_values,
        modes=modes,
        models=models,
    )

    outputs = {
        "tau_star.json": tau_star,
        "success_curves.json": success_curves,
        "frontier.json": frontier,
    }
    for name, data in outputs.items():
        with (derived_dir / name).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
    print(f"Wrote derived estimates to {derived_dir}")


if __name__ == "__main__":
    main()
