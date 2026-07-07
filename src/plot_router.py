from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.load_traces import MODEL_ORDER, load_config


METHOD_LABELS = {
    "router": "router",
    "always_1.5b": "always 1.5B",
    "always_7b": "always 7B",
    "always_32b": "always 32B",
    "oracle": "oracle",
    "stat_oracle": "stat oracle",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.png", dpi=130, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def ci_yerr(entry: dict[str, Any]) -> list[list[float]]:
    mean = float(entry["mean"])
    lo = entry["ci95"]["lo"]
    hi = entry["ci95"]["hi"]
    if lo is None or hi is None:
        return [[0.0], [0.0]]
    return [[max(0.0, mean - float(lo))], [max(0.0, float(hi) - mean)]]


def ordered_modes(summary: dict[str, Any]) -> list[str]:
    preferred = [str(mode) for mode in summary.get("mode_order", [])]
    available = set(summary.get("by_mode", {}))
    modes = [mode for mode in preferred if mode in available]
    modes.extend(sorted(available - set(modes)))
    return modes


def plot_time_vs_accuracy(summary: dict[str, Any], figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    model_order = tuple(str(model) for model in summary.get("model_order", MODEL_ORDER))
    methods = [f"always_{model}" for model in model_order] + ["router", "stat_oracle", "oracle"]
    for method in methods:
        metrics = summary["methods"].get(method)
        if not metrics or metrics["mean_time_all"] is None:
            continue
        ax.scatter(metrics["solved_rate"], metrics["mean_time_all"], s=70)
        ax.annotate(
            METHOD_LABELS.get(method, method),
            (metrics["solved_rate"], metrics["mean_time_all"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("solved rate")
    ax.set_ylabel("mean wall-clock seconds (lower is better)")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.set_title("Router time vs verified accuracy")
    save_figure(fig, figures_dir, "fig_time_vs_accuracy")


def plot_time_saved_by_mode(summary: dict[str, Any], figures_dir: Path) -> None:
    modes = ordered_modes(summary)
    means = [summary["by_mode"][mode]["time_saved_vs_best_single"]["mean"] for mode in modes]
    lows = []
    highs = []
    for mode, mean in zip(modes, means):
        ci = summary["by_mode"][mode]["time_saved_vs_best_single"]["ci95"]
        lows.append(max(0.0, float(mean) - float(ci["lo"])) if ci["lo"] is not None else 0.0)
        highs.append(max(0.0, float(ci["hi"]) - float(mean)) if ci["hi"] is not None else 0.0)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    x = np.arange(len(modes))
    ax.bar(x, means, color="#5B8FB9")
    ax.errorbar(x, means, yerr=[lows, highs], fmt="none", color="black", lw=0.8, capsize=3)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("mean seconds saved vs best single")
    ax.set_title("Router time saved by mode")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir, "fig_time_saved_by_mode")


def plot_alloc_distribution(summary: dict[str, Any], figures_dir: Path) -> None:
    modes = ordered_modes(summary)
    model_order = tuple(str(model) for model in summary.get("model_order", MODEL_ORDER))
    labels = []
    values = []
    for mode in modes:
        for policy in ["router", "stat_oracle"]:
            labels.append(f"{mode}\n{policy}")
            alloc = summary["by_mode"][mode]["allocation_distribution"][policy]["mean_allocation"]
            values.append([alloc[model] for model in model_order])
    data = np.array(values, dtype=float)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    colors = ["#5B8FB9", "#D9A441", "#7A9E65", "#AF7AA1"]
    for idx, model in enumerate(model_order):
        ax.bar(x, data[:, idx], bottom=bottom, label=model, color=colors[idx % len(colors)])
        bottom += data[:, idx]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean allocated attempts")
    ax.set_ylim(0, 10.5)
    ax.legend(frameon=False, ncols=max(1, len(model_order)))
    ax.set_title("Allocation distribution by mode")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir, "fig_alloc_distribution")


def plot_gap_to_oracle(summary: dict[str, Any], figures_dir: Path) -> None:
    methods = ["router", "stat_oracle", "oracle"]
    means = [summary["methods"][method]["mean_time_all"] for method in methods]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    x = np.arange(len(methods))
    ax.bar(x, means, color=["#5B8FB9", "#D9A441", "#7A9E65"])
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(method, method) for method in methods])
    ax.set_ylabel("mean wall-clock seconds")
    ax.set_title("Gap to oracle")
    ax.grid(axis="y", alpha=0.25)

    router_gap = summary.get("router_gap_to_oracle", {})
    stat_gap = summary.get("stat_oracle_gap_to_oracle", {})
    if router_gap.get("mean") is not None and stat_gap.get("mean") is not None:
        text = (
            f"router-oracle gap: {router_gap['mean']:.2f}s\n"
            f"stat-oracle gap: {stat_gap['mean']:.2f}s"
        )
        ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=8)
    save_figure(fig, figures_dir, "fig_gap_to_oracle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot router figures from derived data only.")
    parser.add_argument("--config", default="config/router_experiment.yaml")
    parser.add_argument("--summary-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary_path = Path(args.summary_path or config["paths"]["summary"])
    figures_dir = Path(config["paths"]["figures"])
    summary = load_json(summary_path)
    plot_time_vs_accuracy(summary, figures_dir)
    plot_time_saved_by_mode(summary, figures_dir)
    plot_alloc_distribution(summary, figures_dir)
    plot_gap_to_oracle(summary, figures_dir)
    print(f"Wrote router figures to {figures_dir}")


if __name__ == "__main__":
    main()
