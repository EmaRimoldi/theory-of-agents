from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


MODE_ORDER = ["easy", "medium", "hard"]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_key(key: str) -> tuple[str, str]:
    model, mode = key.rsplit("|", 1)
    return model, mode


def save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.png", dpi=130, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def finite_positive(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v) and v > 0]


def plot_frontier(tau: dict[str, Any], frontier: dict[str, Any], figures_dir: Path) -> None:
    cells = [(key, *split_key(key)) for key in tau]
    modes = sorted({mode for _, _, mode in cells}, key=lambda m: MODE_ORDER.index(m))
    models = sorted({model for _, model, _ in cells})

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(modes))
    width = 0.8 / max(1, len(models))

    for i, model in enumerate(models):
        offsets = x - 0.4 + width / 2 + i * width
        heights = []
        yerr_lo = []
        yerr_hi = []
        for mode in modes:
            entry = tau.get(f"{model}|{mode}")
            if not entry:
                heights.append(np.nan)
                yerr_lo.append(0)
                yerr_hi.append(0)
                continue
            value = float(entry["tau_star"])
            heights.append(value)
            yerr_lo.append(max(0.0, value - float(entry.get("ci_lo", value))))
            yerr_hi.append(max(0.0, float(entry.get("ci_hi", value)) - value))
        ax.bar(offsets, heights, width=width, label=model)
        ax.errorbar(offsets, heights, yerr=[yerr_lo, yerr_hi], fmt="none", color="black", lw=0.8, capsize=2)

    for mode_idx, mode in enumerate(modes):
        winner = frontier.get("per_mode", {}).get(mode, {}).get("argmin_model")
        if winner in models:
            model_idx = models.index(winner)
            xpos = x[mode_idx] - 0.4 + width / 2 + model_idx * width
            value = tau[f"{winner}|{mode}"]["tau_star"]
            ax.scatter([xpos], [value], marker="*", s=90, color="black", zorder=5)

    positives = finite_positive([float(v["tau_star"]) for v in tau.values()])
    if positives:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("proper time tau* (generated tokens)")
    ax.set_xlabel("difficulty mode")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Mode-conditional proper-time frontier")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir, "fig_frontier")


def plot_disagreement(frontier: dict[str, Any], figures_dir: Path) -> None:
    model_accuracy = frontier.get("model_accuracy", {})
    model_tau = frontier.get("model_tau_star_weighted", {})
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    for model, accuracy in sorted(model_accuracy.items()):
        tau = model_tau.get(model, float("nan"))
        if not math.isfinite(float(tau)):
            continue
        ax.scatter(float(accuracy), float(tau), s=60)
        ax.annotate(model, (float(accuracy), float(tau)), xytext=(5, 4), textcoords="offset points", fontsize=8)

    policy = frontier.get("frontier_policy", {})
    if math.isfinite(float(policy.get("tau_star_weighted", float("nan")))):
        ax.scatter(
            [float(policy["accuracy"])],
            [float(policy["tau_star_weighted"])],
            marker="*",
            s=150,
            color="black",
            label="frontier policy",
        )
        ax.legend(frameon=False, fontsize=8)

    ax.set_xlabel("average accuracy")
    ax.set_ylabel("weighted tau* (lower is better)")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.set_title("Accuracy and proper-time ranking")
    save_figure(fig, figures_dir, "fig_disagreement")


def plot_success_curves(
    tau: dict[str, Any],
    curves: dict[str, Any],
    figures_dir: Path,
) -> None:
    cells = [(key, *split_key(key)) for key in curves]
    modes = sorted({mode for _, _, mode in cells}, key=lambda m: MODE_ORDER.index(m))
    models = sorted({model for _, model, _ in cells})

    fig, axes = plt.subplots(1, len(modes), figsize=(5.0 * len(modes), 4.4), sharey=True)
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        for model in models:
            key = f"{model}|{mode}"
            if key not in curves:
                continue
            grid = [float(x) for x in curves[key]["grid"]]
            F = [float(x) for x in curves[key]["F"]]
            ax.step(grid, F, where="post", label=model)
            tau_arg = float(tau[key].get("tau_argmin", tau[key]["tau_star"]))
            if grid:
                idx = min(range(len(grid)), key=lambda i: abs(grid[i] - tau_arg))
                ax.scatter([grid[idx]], [F[idx]], s=25)

        positives = finite_positive([x for key, curve in curves.items() for x in curve["grid"]])
        if positives:
            ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(mode)
        ax.set_xlabel("generated-token budget")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("KM success F(t)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Success-by-budget curves")
    save_figure(fig, figures_dir, "fig_success_curves")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot figures from derived estimates only.")
    parser.add_argument("--config", default="config/experiment.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    derived_dir = Path(config["paths"]["derived"])
    figures_dir = Path(config["paths"]["figures"])

    tau = load_json(derived_dir / "tau_star.json")
    curves = load_json(derived_dir / "success_curves.json")
    frontier = load_json(derived_dir / "frontier.json")

    plot_frontier(tau, frontier, figures_dir)
    plot_disagreement(frontier, figures_dir)
    plot_success_curves(tau, curves, figures_dir)
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
