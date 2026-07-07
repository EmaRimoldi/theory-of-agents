from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first_pass_attempt(statuses: list[str]) -> int | None:
    for idx, status in enumerate(statuses, start=1):
        if status == "pass":
            return idx
    return None


def success_curve(rows: list[dict[str, Any]], max_attempts: int) -> list[float]:
    first_passes = [first_pass_attempt([str(x) for x in row["attempt_statuses"]]) for row in rows]
    return [
        sum(fp is not None and fp <= k for fp in first_passes) / max(1, len(first_passes))
        for k in range(1, max_attempts + 1)
    ]


def verify_cumulative_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        task_id = row["task_id"]
        statuses = [str(x) for x in row["attempt_statuses"]]
        seconds = [float(x) for x in row["attempt_seconds"]]
        tokens = [int(x) for x in row["attempt_token_counts"]]
        n_attempts = int(row["n_attempts"])
        if not (len(statuses) == len(seconds) == len(tokens) == n_attempts):
            errors.append(f"{task_id}: per-attempt array length mismatch")
            continue
        first_pass = first_pass_attempt(statuses)
        if first_pass is None:
            if bool(row["solved"]):
                errors.append(f"{task_id}: solved=true but no pass status")
            if not math.isinf(float(row["tau_seconds"])):
                errors.append(f"{task_id}: unsolved tau_seconds is not inf")
            if not math.isinf(float(row["tau_tokens"])):
                errors.append(f"{task_id}: unsolved tau_tokens is not inf")
            continue
        expected_seconds = sum(seconds[:first_pass])
        expected_tokens = sum(tokens[:first_pass])
        if not bool(row["solved"]):
            errors.append(f"{task_id}: pass status but solved=false")
        if not math.isclose(float(row["tau_seconds"]), expected_seconds, rel_tol=0, abs_tol=1e-6):
            errors.append(f"{task_id}: tau_seconds is not cumulative")
        if int(row["tau_tokens"]) != expected_tokens:
            errors.append(f"{task_id}: tau_tokens is not cumulative")
    return errors


def build_report(rows: list[dict[str, Any]], max_attempts: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("subtask", row["task_id"].split("/")[1]))].append(row)
    curves = {
        subtask: success_curve(subtask_rows, max_attempts)
        for subtask, subtask_rows in sorted(grouped.items())
    }
    overall = success_curve(rows, max_attempts)
    flat_subtasks = {
        subtask: curve
        for subtask, curve in curves.items()
        if len(curve) > 1 and max(curve) <= min(curve)
    }
    solved_late = [
        row["task_id"]
        for row in rows
        if (first_pass_attempt([str(x) for x in row["attempt_statuses"]]) or 0) >= 2
    ]
    return {
        "n_rows": len(rows),
        "max_attempts": max_attempts,
        "overall_F_by_attempt": overall,
        "subtask_F_by_attempt": curves,
        "flat_subtasks": flat_subtasks,
        "solved_on_attempt_2_or_later": solved_late,
        "retry_meaningful_on_smoke": bool(len(overall) > 1 and overall[-1] > overall[0]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check BBH retry sanity from raw worker logs.")
    parser.add_argument("--raw-file", required=True)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--output", default="data/derived/bbh_retry_sanity_smoke.json")
    parser.add_argument("--require-rise", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.raw_file))
    if not rows:
        raise SystemExit(f"No rows in {args.raw_file}")
    errors = verify_cumulative_rows(rows)
    report = build_report(rows, args.max_attempts)
    report["schema_errors"] = errors
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)
    if args.require_rise and not report["retry_meaningful_on_smoke"]:
        print("BBH smoke retry curve is flat; retry sanity failed.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
