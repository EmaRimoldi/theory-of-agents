from __future__ import annotations

import re
from typing import Any


def load_mbpp_plus() -> tuple[dict[str, dict[str, Any]], str]:
    from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash

    return get_mbpp_plus(), get_mbpp_plus_hash()


def description_from_prompt(prompt: str) -> str:
    text = prompt.strip()
    if text.startswith('"""') and text.endswith('"""'):
        text = text[3:-3].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    description_lines = [line for line in lines if not line.startswith("assert ")]
    return " ".join(description_lines) if description_lines else " ".join(lines)


def signature_for_problem(problem: dict[str, Any]) -> str:
    entry_point = str(problem.get("entry_point", "")).strip()
    canonical = str(problem.get("canonical_solution", ""))
    if entry_point:
        pattern = rf"def\s+{re.escape(entry_point)}\s*\([^)]*\)"
        match = re.search(pattern, canonical)
        if match:
            return match.group(0)
        return f"def {entry_point}(...)"
    match = re.search(r"def\s+\w+\s*\([^)]*\)", canonical)
    return match.group(0) if match else ""


def problem_for_tagging(problem: dict[str, Any]) -> dict[str, str]:
    return {
        "task_id": str(problem["task_id"]),
        "description": description_from_prompt(str(problem.get("prompt", ""))),
        "signature": signature_for_problem(problem),
        "canonical_solution": str(problem.get("canonical_solution", "")),
    }
