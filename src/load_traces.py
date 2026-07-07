from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any

import yaml


MODEL_NAME_MAP = {
    "qwen2.5-coder:1.5b": "1.5b",
    "qwen2.5-coder:7b": "7b",
    "qwen2.5-coder:32b": "32b",
    "qwen2.5:1.5b": "1.5b",
    "qwen2.5:7b": "7b",
    "qwen2.5:32b": "32b",
}
MODEL_ORDER = ("1.5b", "7b", "32b")
STRICT_SUCCESS = "pass"


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def short_model_name(model: str) -> str:
    if model in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[model]
    match = re.search(r"(?P<size>1\.5b|7b|32b)", model, flags=re.IGNORECASE)
    if match:
        return match.group("size").lower()
    raise ValueError(f"Cannot map model name {model!r} to a short router key")


def strict_solved(statuses: list[str]) -> bool:
    return any(status == STRICT_SUCCESS for status in statuses)


def configured_model_order(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(model) for model in config.get("execution_order", MODEL_ORDER))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_log_paths(path_raw: str) -> list[Path]:
    if any(ch in path_raw for ch in "*?["):
        matches = [Path(path) for path in sorted(glob.glob(path_raw))]
        if not matches:
            raise FileNotFoundError(
                f"No worker logs matched {path_raw!r}. "
                "For MBPP+ category analysis, run or provide the saved MBPP+ "
                "worker logs before running router/frontier analysis."
            )
        return matches
    path = Path(path_raw)
    if not path.exists():
        raise FileNotFoundError(
            f"Worker log not found: {path}. "
            "For MBPP+ category analysis, run or provide the saved MBPP+ "
            "worker logs before running router/frontier analysis."
        )
    return [path]


def load_traces(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load worker traces keyed as traces[task_id][short_model].

    The stored solved flag is intentionally not trusted; it is recomputed from
    attempt_statuses under the strict success threshold.
    """

    if config.get("success_threshold") != "strict":
        raise ValueError("Router experiment requires success_threshold: strict")

    modes_path = Path(config["modes_file"])
    with modes_path.open("r", encoding="utf-8") as f:
        modes: dict[str, str] = json.load(f)

    model_order = configured_model_order(config)
    allow_partial_modes = bool(config.get("allow_partial_modes", False))
    traces: dict[str, dict[str, Any]] = {}
    for log_path_raw in config["worker_logs"]:
        for log_path in resolve_log_paths(str(log_path_raw)):
            for row in read_jsonl(log_path):
                task_id = row["task_id"]
                if task_id not in modes:
                    if allow_partial_modes:
                        continue
                    raise KeyError(
                        f"Task {task_id!r} from {log_path} has no mode in {modes_path}"
                    )
                model = short_model_name(row["model"])
                attempt_seconds = [float(x) for x in row["attempt_seconds"]]
                attempt_tokens = [int(x) for x in row["attempt_token_counts"]]
                attempt_statuses = [str(x) for x in row["attempt_statuses"]]
                if not (
                    len(attempt_seconds)
                    == len(attempt_tokens)
                    == len(attempt_statuses)
                    == int(row["n_attempts"])
                ):
                    raise ValueError(
                        f"Bad attempt arrays for {task_id} {model} in {log_path}"
                    )
                if model in traces.get(task_id, {}):
                    raise ValueError(f"Duplicate trace for {task_id} {model}")

                traces.setdefault(task_id, {})[model] = {
                    "attempt_seconds": attempt_seconds,
                    "attempt_token_counts": attempt_tokens,
                    "attempt_statuses": attempt_statuses,
                    "mode": modes[task_id],
                    "solved": strict_solved(attempt_statuses),
                    "source_model": row["model"],
                    "source_log": str(log_path),
                }

    missing_modes = set(modes) - set(traces)
    if missing_modes and not allow_partial_modes:
        raise ValueError(f"Worker logs are missing {len(missing_modes)} tasks")

    for task_id, per_model in traces.items():
        missing = [model for model in model_order if model not in per_model]
        if missing:
            raise ValueError(f"Task {task_id} is missing model traces: {missing}")

    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect router worker traces.")
    parser.add_argument("--config", default="config/router_experiment.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    model_order = configured_model_order(config)
    traces = load_traces(config)
    counts = {model: 0 for model in model_order}
    solved = {model: 0 for model in model_order}
    for per_model in traces.values():
        for model in model_order:
            counts[model] += 1
            solved[model] += int(per_model[model]["solved"])
    print(json.dumps({"n_tasks": len(traces), "counts": counts, "solved": solved}, indent=2))


if __name__ == "__main__":
    main()
