from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.load_traces import MODEL_ORDER, configured_model_order, load_config, load_traces


DEFAULT_MODE_ORDER = ["string", "math", "list_ds", "logic_control"]


def trace_duration(trace: dict[str, Any], *, clock: str) -> tuple[float, bool]:
    if clock == "seconds":
        values = [float(x) for x in trace["attempt_seconds"]]
    elif clock == "tokens":
        values = [float(x) for x in trace["attempt_token_counts"]]
    else:
        raise ValueError(f"Unsupported clock {clock!r}; expected seconds or tokens")

    total = 0.0
    for value, status in zip(values, trace["attempt_statuses"]):
        total += value
        if status == "pass":
            return max(total, 1.0e-9), True
    return max(total, 1.0e-9), False


def km_curve(durations: np.ndarray, events: np.ndarray) -> tuple[list[float], list[float]]:
    kmf = KaplanMeierFitter()
    kmf.fit(durations=durations, event_observed=events)
    survival = kmf.survival_function_.iloc[:, 0]
    return (
        [float(x) for x in survival.index.to_numpy()],
        [float(1.0 - x) for x in survival.to_numpy()],
    )


def tau_star_from_rows(rows: list[dict[str, Any]], *, alpha: float) -> dict[str, Any]:
    durations = np.array([float(row["duration"]) for row in rows], dtype=float)
    events = np.array([bool(row["solved"]) for row in rows], dtype=bool)
    grid, success = km_curve(durations, events)
    candidates = [(t, f) for t, f in zip(grid, success) if t > 0 and (f > 0 or not events.any())]
    if not candidates:
        fallback = float(np.min(durations)) if durations.size else 1.0
        tau_star = fallback / alpha
        tau_argmin = fallback
    else:
        tau_star, tau_argmin = min((t / max(f, alpha), t) for t, f in candidates)
    return {
        "tau_star": float(tau_star),
        "tau_argmin": float(tau_argmin),
        "grid": grid,
        "F": success,
        "n": len(rows),
        "n_solved": int(events.sum()),
        "solved_rate": float(events.mean()) if events.size else 0.0,
    }


def bootstrap_tau(
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not rows:
        return np.array([], dtype=float)
    n = len(rows)
    values = np.empty(B, dtype=float)
    for idx in range(B):
        sample = [rows[int(i)] for i in rng.integers(0, n, size=n)]
        values[idx] = tau_star_from_rows(sample, alpha=alpha)["tau_star"]
    return values


def percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def ordered_modes(modes: set[str], mode_order: list[str]) -> list[str]:
    ordered = [mode for mode in mode_order if mode in modes]
    ordered.extend(sorted(modes - set(ordered)))
    return ordered


def pairwise_ratio_summary(
    boot_values: dict[tuple[str, str], np.ndarray],
    *,
    mode: str,
    models: list[str],
) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    for left_idx, left in enumerate(models):
        for right in models[left_idx + 1 :]:
            a = boot_values[(left, mode)]
            b = boot_values[(right, mode)]
            m = min(len(a), len(b))
            if m == 0:
                continue
            ratio = np.log2(a[:m] / b[:m])
            pairs[f"{left}_vs_{right}"] = {
                "median": float(np.median(ratio)),
                "ci95": [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))],
            }
    return pairs


def build_frontier(
    traces: dict[str, dict[str, Any]],
    *,
    clock: str,
    alpha: float,
    B: int,
    seed: int,
    mode_order: list[str],
    model_order: tuple[str, ...],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task_id, per_model in traces.items():
        mode = str(per_model[model_order[0]]["mode"])
        for model in model_order:
            duration, solved = trace_duration(per_model[model], clock=clock)
            grouped[(model, mode)].append(
                {
                    "task_id": task_id,
                    "model": model,
                    "mode": mode,
                    "duration": duration,
                    "solved": solved,
                }
            )

    modes = ordered_modes({mode for _, mode in grouped}, mode_order)
    models = list(model_order)
    cells: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    boot_values: dict[tuple[str, str], np.ndarray] = {}
    for mode in modes:
        for model in models:
            rows = grouped[(model, mode)]
            stats = tau_star_from_rows(rows, alpha=alpha)
            boots = bootstrap_tau(rows, alpha=alpha, B=B, rng=rng)
            boot_values[(model, mode)] = boots
            key = f"{model}|{mode}"
            cells[key] = {
                "tau_star": stats["tau_star"],
                "tau_argmin": stats["tau_argmin"],
                "tau_star_ci95": [percentile(boots, 2.5), percentile(boots, 97.5)],
                "tau_star_ci80": [percentile(boots, 16), percentile(boots, 84)],
                "n": stats["n"],
                "n_solved": stats["n_solved"],
                "solved_rate": stats["solved_rate"],
            }
            curves[key] = {"grid": stats["grid"], "F": stats["F"]}

    per_mode: dict[str, Any] = {}
    for mode in modes:
        by_tau = sorted(
            [(model, cells[f"{model}|{mode}"]["tau_star"]) for model in models],
            key=lambda item: (item[1], models.index(item[0])),
        )
        by_accuracy = sorted(
            [(model, cells[f"{model}|{mode}"]["solved_rate"]) for model in models],
            key=lambda item: (-item[1], models.index(item[0])),
        )
        winner, winner_tau = by_tau[0]
        runner, _runner_tau = by_tau[1]
        w_boot = boot_values[(winner, mode)]
        r_boot = boot_values[(runner, mode)]
        m = min(len(w_boot), len(r_boot))
        ratio = np.log2(w_boot[:m] / r_boot[:m]) if m else np.array([], dtype=float)
        ratio_ci95 = [
            percentile(ratio, 2.5),
            percentile(ratio, 97.5),
        ]
        per_mode[mode] = {
            "argmin_tau_model": winner,
            "argmin_tau_star": winner_tau,
            "tau_runner_up_model": runner,
            "winner_vs_runner_log2_ratio_ci95": ratio_ci95,
            "winner_vs_runner_log2_ratio_median": percentile(ratio, 50),
            "winner_separated_from_runner": (
                bool(ratio_ci95[1] is not None and ratio_ci95[1] < 0.0)
            ),
            "accuracy_winner_model": by_accuracy[0][0],
            "accuracy_winner_solved_rate": by_accuracy[0][1],
            "pairwise_log2_tau_ratio_ci95": pairwise_ratio_summary(
                boot_values,
                mode=mode,
                models=models,
            ),
        }

    tau_winners = {info["argmin_tau_model"] for info in per_mode.values()}
    accuracy_winners = {info["accuracy_winner_model"] for info in per_mode.values()}
    separated_modes = {
        mode: info
        for mode, info in per_mode.items()
        if bool(info["winner_separated_from_runner"])
    }
    separated_tau_winners = {info["argmin_tau_model"] for info in separated_modes.values()}
    return {
        "clock": clock,
        "alpha": alpha,
        "bootstrap_B": B,
        "modes": modes,
        "models": models,
        "cells": cells,
        "curves": curves,
        "per_mode": per_mode,
        "tau_star_winner_swaps": len(tau_winners) > 1,
        "separated_tau_star_winner_swaps": len(separated_tau_winners) > 1,
        "accuracy_winner_swaps": len(accuracy_winners) > 1,
    }


def save_figure(frontier: dict[str, Any], *, figures_dir: Path, stem: str) -> None:
    modes = list(frontier["modes"])
    models = list(frontier["models"])
    cells = frontier["cells"]

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    x = np.arange(len(modes))
    width = 0.8 / max(1, len(models))
    colors = ["#4E79A7", "#F28E2B", "#59A14F"]
    for idx, model in enumerate(models):
        offsets = x - 0.4 + width / 2 + idx * width
        heights = []
        yerr_lo = []
        yerr_hi = []
        for mode in modes:
            entry = cells[f"{model}|{mode}"]
            value = float(entry["tau_star"])
            lo, hi = entry["tau_star_ci80"]
            heights.append(value)
            yerr_lo.append(max(0.0, value - float(lo)) if lo is not None else 0.0)
            yerr_hi.append(max(0.0, float(hi) - value) if hi is not None else 0.0)
        ax.bar(offsets, heights, width=width, label=model, color=colors[idx % len(colors)])
        ax.errorbar(offsets, heights, yerr=[yerr_lo, yerr_hi], fmt="none", color="black", lw=0.8, capsize=2)

    for mode_idx, mode in enumerate(modes):
        winner = frontier["per_mode"][mode]["argmin_tau_model"]
        model_idx = models.index(winner)
        xpos = x[mode_idx] - 0.4 + width / 2 + model_idx * width
        ypos = cells[f"{winner}|{mode}"]["tau_star"]
        ax.scatter([xpos], [ypos], marker="*", s=95, color="black", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_yscale("log")
    ylabel = "tau* wall-clock seconds" if frontier["clock"] == "seconds" else "tau* generated tokens"
    ax.set_ylabel(ylabel)
    ax.set_xlabel("algorithmic category")
    ax.set_title("Category-conditional proper-time frontier")
    ax.legend(frameon=False, ncols=3)
    ax.grid(axis="y", alpha=0.25)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.png", dpi=130, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate category-conditional frontier from worker logs.")
    parser.add_argument("--config", default="config/router_experiment_mbpp_category.yaml")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--figure-stem", default="fig_category_frontier")
    parser.add_argument("--clock", choices=["seconds", "tokens"], default=None)
    parser.add_argument("--bootstrap-B", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    traces = load_traces(config)
    frontier_config = config.get("frontier", {})
    clock = str(args.clock or frontier_config.get("clock", "seconds"))
    alpha = float(frontier_config.get("censoring_floor", 1.0e-3))
    B = int(args.bootstrap_B or frontier_config.get("bootstrap_B", config.get("estimation", {}).get("bootstrap_B", 1000)))
    seed = int(config.get("kfold", {}).get("seed", 0))
    mode_order = [str(mode) for mode in config.get("mode_order", DEFAULT_MODE_ORDER)]
    model_order = configured_model_order(config)

    frontier = build_frontier(
        traces,
        clock=clock,
        alpha=alpha,
        B=B,
        seed=seed,
        mode_order=mode_order,
        model_order=model_order,
    )
    output_path = Path(
        args.output_path
        or config["paths"].get("category_frontier", "data/derived/category_frontier.json")
    )
    figures_dir = Path(
        args.figures_dir
        or config["paths"].get("category_frontier_figures", config["paths"].get("figures", "figures/router_mbpp_category"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(frontier, f, indent=2, sort_keys=True)
        f.write("\n")
    save_figure(frontier, figures_dir=figures_dir, stem=args.figure_stem)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "clock": clock,
                "tau_star_winner_swaps": frontier["tau_star_winner_swaps"],
                "separated_tau_star_winner_swaps": frontier["separated_tau_star_winner_swaps"],
                "accuracy_winner_swaps": frontier["accuracy_winner_swaps"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
