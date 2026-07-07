from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CORE_METHODS = ["router", "best_single", "stat_oracle", "oracle"]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ci_excludes_zero(ci: dict[str, float | None]) -> bool:
    lo = ci.get("lo")
    hi = ci.get("hi")
    if lo is None or hi is None:
        return False
    return bool(lo > 0.0 or hi < 0.0)


def ci_contains_zero(ci: dict[str, float | None]) -> bool:
    lo = ci.get("lo")
    hi = ci.get("hi")
    if lo is None or hi is None:
        return False
    return bool(lo <= 0.0 <= hi)


def stat(summary: dict[str, Any], key: str) -> dict[str, Any]:
    entry = summary.get(key, {})
    return {
        "mean": entry.get("mean"),
        "ci95": entry.get("ci95", {"lo": None, "hi": None}),
    }


def compact_methods(summary: dict[str, Any]) -> dict[str, Any]:
    methods = summary.get("methods", {})
    compact: dict[str, Any] = {}
    for method in CORE_METHODS:
        if method not in methods:
            continue
        entry = methods[method]
        compact[method] = {
            "solved": entry.get("solved"),
            "solved_rate": entry.get("solved_rate"),
            "mean_time_all": entry.get("mean_time_all"),
            "mean_time_solved": entry.get("mean_time_solved"),
        }
    return compact


def dataset_summary(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    methods = compact_methods(summary)
    saved = stat(summary, "time_saved_vs_best_single")
    router_gap = stat(summary, "router_gap_to_stat_oracle")
    stat_gap = stat(summary, "stat_oracle_gap_to_oracle")

    router_rate = methods.get("router", {}).get("solved_rate")
    best_rate = methods.get("best_single", {}).get("solved_rate")
    stat_rate = methods.get("stat_oracle", {}).get("solved_rate")
    oracle_rate = methods.get("oracle", {}).get("solved_rate")

    return {
        "dataset": name,
        "n_records": summary.get("n_records"),
        "model_order": summary.get("model_order"),
        "best_single_model": summary.get("best_single_model"),
        "parse_failures": summary.get("parse_failures"),
        "methods": methods,
        "time_saved_vs_best_single": saved,
        "router_gap_to_stat_oracle": router_gap,
        "stat_oracle_gap_to_oracle": stat_gap,
        "diagnostics": {
            "router_matches_best_single_accuracy": (
                None if router_rate is None or best_rate is None else bool(router_rate >= best_rate)
            ),
            "router_accuracy_gap_vs_best_single": (
                None if router_rate is None or best_rate is None else float(router_rate - best_rate)
            ),
            "stat_oracle_accuracy_gap_vs_best_single": (
                None if stat_rate is None or best_rate is None else float(stat_rate - best_rate)
            ),
            "oracle_accuracy_gap_vs_best_single": (
                None if oracle_rate is None or best_rate is None else float(oracle_rate - best_rate)
            ),
            "router_saves_time_vs_best_single_ci_excludes_zero": ci_excludes_zero(saved["ci95"]),
            "router_close_to_stat_oracle_time_ci_contains_zero": ci_contains_zero(router_gap["ci95"]),
            "stat_oracle_close_to_oracle_time_ci_contains_zero": ci_contains_zero(stat_gap["ci95"]),
        },
    }


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}s"


def fmt_ci(stat_entry: dict[str, Any], unit: str = "s") -> str:
    mean = stat_entry.get("mean")
    ci = stat_entry.get("ci95", {})
    lo = ci.get("lo")
    hi = ci.get("hi")
    if mean is None or lo is None or hi is None:
        return "n/a"
    return f"{float(mean):.3f}{unit} [{float(lo):.3f}, {float(hi):.3f}]"


def render_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Dataset Comparison",
        "",
        "| Dataset | n | menu | best single | router solved | best solved | router mean time | best mean time | saved vs best, CI | router-stat gap, CI |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparison["datasets"]:
        methods = item["methods"]
        router = methods.get("router", {})
        best = methods.get("best_single", {})
        menu = ", ".join(item.get("model_order") or [])
        lines.append(
            "| {dataset} | {n} | {menu} | {best_model} | {router_solved} | {best_solved} | "
            "{router_time} | {best_time} | {saved} | {gap} |".format(
                dataset=item["dataset"],
                n=item.get("n_records"),
                menu=menu or "n/a",
                best_model=item.get("best_single_model") or "n/a",
                router_solved=fmt_pct(router.get("solved_rate")),
                best_solved=fmt_pct(best.get("solved_rate")),
                router_time=fmt_seconds(router.get("mean_time_all")),
                best_time=fmt_seconds(best.get("mean_time_all")),
                saved=fmt_ci(item["time_saved_vs_best_single"]),
                gap=fmt_ci(item["router_gap_to_stat_oracle"]),
            )
        )

    lines.extend(["", "## Diagnostics", ""])
    for item in comparison["datasets"]:
        d = item["diagnostics"]
        lines.extend(
            [
                f"- {item['dataset']}: router saves time vs best-single with CI excluding zero: "
                f"{d['router_saves_time_vs_best_single_ci_excludes_zero']}.",
                f"- {item['dataset']}: router-stat_oracle time gap CI contains zero: "
                f"{d['router_close_to_stat_oracle_time_ci_contains_zero']}.",
                f"- {item['dataset']}: router accuracy gap vs best-single: "
                f"{fmt_pct(d['router_accuracy_gap_vs_best_single'])}.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare HumanEval+ and MBPP+ router summaries.")
    parser.add_argument("--humaneval-summary", default="data/derived/router_summary.json")
    parser.add_argument("--mbpp-summary", default="data/derived/mbpp_router_summary_2models.json")
    parser.add_argument("--output-json", default="data/derived/dataset_comparison.json")
    parser.add_argument("--output-md", default="data/derived/dataset_comparison.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    he_path = Path(args.humaneval_summary)
    mbpp_path = Path(args.mbpp_summary)
    missing = [str(path) for path in [he_path, mbpp_path] if not path.exists()]
    if missing:
        raise SystemExit(f"Missing summary file(s): {', '.join(missing)}")

    comparison = {
        "datasets": [
            dataset_summary("HumanEval+", read_json(he_path)),
            dataset_summary("MBPP+ reduced", read_json(mbpp_path)),
        ],
        "notes": [
            "HumanEval+ may use the original model menu from its summary; MBPP+ reduced uses the two-model 1.5b/7b menu.",
            "All comparisons are based on saved router summary files; this script makes no model calls.",
        ],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, sort_keys=True)
        f.write("\n")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(comparison), encoding="utf-8")
    print(f"Wrote {output_json} and {output_md}")


if __name__ == "__main__":
    main()
