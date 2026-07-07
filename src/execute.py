from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExecutionResult:
    passed: bool
    base_status: str
    plus_status: str
    timed_out: bool
    stdout: str
    stderr: str


RUNNER = r"""
import json
import multiprocessing
import os
import pickle
import sys

from evalplus.eval import PASS
from evalplus.evaluate import check_correctness

if sys.platform == "darwin":
    multiprocessing.set_start_method("fork", force=True)

payload_path, result_path = sys.argv[1:3]
with open(payload_path, "rb") as f:
    payload = pickle.load(f)

try:
    result = check_correctness(
        payload["dataset"],
        0,
        payload["problem"],
        payload["solution"],
        payload["expected_output"],
        base_only=False,
        fast_check=True,
    )
    base_status = result["base"][0]
    plus_status = result["plus"][0]
    passed = base_status == PASS and plus_status == PASS
    output = {
        "passed": passed,
        "base_status": base_status,
        "plus_status": plus_status,
    }
except BaseException as exc:
    output = {
        "passed": False,
        "base_status": "exception",
        "plus_status": "exception",
        "error": repr(exc),
    }

with open(result_path, "w", encoding="utf-8") as f:
    json.dump(output, f)
"""


def _limit_child_resources(timeout_seconds: int) -> None:
    try:
        import resource

        cpu_limit = max(1, int(timeout_seconds) + 2)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    except Exception:
        return


def execute_candidate(
    *,
    solution: str,
    problem: dict[str, Any],
    expected_output: dict[str, Any],
    timeout_seconds: float,
    checker_dataset: str = "humaneval",
) -> ExecutionResult:
    with tempfile.TemporaryDirectory(prefix="ptf_exec_") as tmp:
        tmp_path = Path(tmp)
        payload_path = tmp_path / "payload.pkl"
        result_path = tmp_path / "result.json"
        runner_path = tmp_path / "runner.py"

        with payload_path.open("wb") as f:
            pickle.dump(
                {
                    "solution": solution,
                    "problem": problem,
                    "expected_output": expected_output,
                    "dataset": checker_dataset,
                },
                f,
            )
        runner_path.write_text(RUNNER, encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("EVALPLUS_MAX_MEMORY_BYTES", "-1")

        try:
            completed = subprocess.run(
                [sys.executable, str(runner_path), str(payload_path), str(result_path)],
                cwd=tmp,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                preexec_fn=(
                    (lambda: _limit_child_resources(int(timeout_seconds)))
                    if hasattr(os, "fork")
                    else None
                ),
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                passed=False,
                base_status="timeout",
                plus_status="timeout",
                timed_out=True,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
            )

        if not result_path.exists():
            return ExecutionResult(
                passed=False,
                base_status="crash",
                plus_status="crash",
                timed_out=False,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

        with result_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        return ExecutionResult(
            passed=bool(result.get("passed", False)),
            base_status=str(result.get("base_status", "unknown")),
            plus_status=str(result.get("plus_status", "unknown")),
            timed_out=False,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
