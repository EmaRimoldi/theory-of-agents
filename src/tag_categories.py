from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.mbpp_data import load_mbpp_plus, problem_for_tagging
from src.router import extract_json_object


CATEGORIES = ("string", "math", "list_ds", "logic_control")
MATHNOT = {"math": "math", "string": "nonmath", "list_ds": "nonmath", "logic_control": "nonmath"}

CATEGORY_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string"},
    },
    "required": ["category", "confidence", "rationale"],
    "additionalProperties": False,
}

CATEGORY_PROMPT_RULES = """Every MBPP+ problem is assigned EXACTLY ONE of:

1. **string** — string/text manipulation: parsing, formatting, substring search,
   character operations, regex-like logic.
2. **math** — numeric/mathematical: arithmetic, number theory, sequences,
   combinatorics, numeric computation, math formulas.
3. **list_ds** — list/data-structure operations: list/dict/set manipulation,
   sorting, searching, aggregation, matrix/array handling.
4. **logic_control** — logic and control flow: complex conditionals, recursion,
   simulation, state machines, case-by-case reasoning not dominated by the above.

**Precedence rule for hybrids** (apply in this order, stop at first match):
- If the core difficulty is non-trivial numeric/mathematical computation → `math`.
- Else if the problem is primarily string/text → `string`.
- Else if it primarily manipulates lists/dicts/sets/arrays → `list_ds`.
- Else → `logic_control`."""

SYSTEM_PROMPT = (
    "You label MBPP+ programming tasks by algorithmic category. "
    "Choose exactly one category. Output only strict JSON and no markdown."
)


def prompt_hash() -> str:
    return hashlib.sha256((SYSTEM_PROMPT + "\n" + CATEGORY_PROMPT_RULES).encode("utf-8")).hexdigest()


def build_prompt(problem: dict[str, str]) -> str:
    payload = {
        "task_id": problem["task_id"],
        "description": problem["description"],
        "entry_point_signature": problem["signature"],
        "canonical_solution": problem["canonical_solution"],
    }
    return (
        CATEGORY_PROMPT_RULES
        + "\n\nReturn STRICT JSON with exactly these keys: "
        + '{"category":"string|math|list_ds|logic_control","confidence":0.0-1.0,'
        + '"rationale":"one short sentence"}.\n'
        + "Problem:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def validate_category_json(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("category response is not an object")
    category = value.get("category")
    if category not in CATEGORIES:
        raise ValueError(f"invalid category {category!r}")
    confidence = float(value.get("confidence"))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"invalid confidence {confidence!r}")
    rationale = str(value.get("rationale", "")).strip()
    if not rationale:
        raise ValueError("empty rationale")
    return {
        "category": category,
        "confidence": confidence,
        "rationale": rationale,
    }


def parse_category_response(raw: str) -> dict[str, Any]:
    return validate_category_json(extract_json_object(raw))


class MockCategoryTagger:
    model_id = "mock-category-tagger"
    backend = "mock"
    reasoning_effort = None

    def classify(self, problem: dict[str, str]) -> str:
        text = " ".join(
            [
                problem["description"].lower(),
                problem["signature"].lower(),
                problem["canonical_solution"].lower(),
            ]
        )
        math_terms = [
            "prime",
            "factor",
            "gcd",
            "lcm",
            "sqrt",
            "math.",
            "combin",
            "number",
            "integer",
            "arithmetic",
            "formula",
            "sum of",
            "product",
        ]
        string_terms = ["string", "substring", "character", "vowel", "word", "text", "split", "join"]
        list_terms = [
            "list",
            "tuple",
            "dictionary",
            "dict",
            "set",
            "array",
            "matrix",
            "sort",
            "heap",
            "queue",
        ]
        if any(term in text for term in math_terms):
            return "math"
        if any(term in text for term in string_terms):
            return "string"
        if any(term in text for term in list_terms):
            return "list_ds"
        return "logic_control"

    def tag(self, problem: dict[str, str]) -> dict[str, Any]:
        category = self.classify(problem)
        return {
            "category": category,
            "confidence": 0.75,
            "rationale": f"Mock heuristic selected {category}.",
            "parse_failed": False,
            "raw_response": None,
        }


class CodexCLICategoryTagger:
    backend = "codex_cli"

    def __init__(self, *, model_env: str, reasoning_effort: str, timeout_seconds: int) -> None:
        model = os.environ.get(model_env)
        if not model:
            raise RuntimeError(f"Set {model_env} before running the category tagger")
        self.model_id = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds

    def call_once(self, problem: dict[str, str]) -> str:
        full_prompt = SYSTEM_PROMPT + "\n\n" + build_prompt(problem)
        with tempfile.TemporaryDirectory(prefix="category_codex_") as tmp:
            tmp_path = Path(tmp)
            schema_path = tmp_path / "category_schema.json"
            output_path = tmp_path / "category_output.txt"
            schema_path.write_text(json.dumps(CATEGORY_SCHEMA), encoding="utf-8")
            command = [
                "codex",
                "-a",
                "never",
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-rules",
                "-C",
                str(tmp_path),
                "-s",
                "workspace-write",
                "-m",
                self.model_id,
                "-c",
                f'model_reasoning_effort="{self.reasoning_effort}"',
                "--output-schema",
                str(schema_path),
                "-o",
                str(output_path),
                "-",
            ]
            completed = subprocess.run(
                command,
                input=full_prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
            )
            raw = output_path.read_text(encoding="utf-8") if output_path.exists() else completed.stdout
            if completed.returncode != 0:
                raw = raw + "\n" + completed.stderr
            return raw

    def tag(self, problem: dict[str, str]) -> dict[str, Any]:
        errors: list[str] = []
        raw = ""
        for _attempt in range(2):
            try:
                raw = self.call_once(problem)
                parsed = parse_category_response(raw)
                return {
                    **parsed,
                    "parse_failed": False,
                    "raw_response": raw,
                }
            except Exception as exc:
                errors.append(str(exc))
        return {
            "category": "logic_control",
            "confidence": 0.0,
            "rationale": "Fallback after parse failure.",
            "parse_failed": True,
            "raw_response": raw,
            "errors": errors,
        }


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                records[str(row["task_id"])] = row
    return records


def append_record(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def write_outputs(
    *,
    records: dict[str, dict[str, Any]],
    problems: dict[str, dict[str, str]],
    dataset_hash: str,
    model_id: str,
    backend: str,
    reasoning_effort: str | None,
    modes_path: Path,
    meta_path: Path,
    sample_path: Path,
    mathnot_path: Path,
    mathnot_meta_path: Path,
    thin_threshold: int,
    seed: int,
) -> None:
    modes = {task_id: str(row["category"]) for task_id, row in records.items()}
    counts = dict(Counter(modes.values()))
    mean_confidence = (
        float(np.mean([float(row["confidence"]) for row in records.values()])) if records else None
    )
    parse_failures = sum(bool(row.get("parse_failed", False)) for row in records.values())
    thin_categories = {
        category: counts.get(category, 0)
        for category in CATEGORIES
        if counts.get(category, 0) < thin_threshold
    }
    meta = {
        "dataset": "mbppplus",
        "dataset_hash": dataset_hash,
        "n_total_available": len(problems),
        "n_tagged": len(modes),
        "categories": list(CATEGORIES),
        "counts": {category: counts.get(category, 0) for category in CATEGORIES},
        "thin_threshold": thin_threshold,
        "thin_categories": thin_categories,
        "model_id": model_id,
        "backend": backend,
        "temperature": 0,
        "reasoning_effort": reasoning_effort,
        "prompt_hash": prompt_hash(),
        "parse_failures": parse_failures,
        "mean_confidence": mean_confidence,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    modes_path.parent.mkdir(parents=True, exist_ok=True)
    with modes_path.open("w", encoding="utf-8") as f:
        json.dump(modes, f, indent=2, sort_keys=True)
        f.write("\n")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")

    rng = np.random.default_rng(seed)
    task_ids = sorted(records)
    sample_ids = list(task_ids)
    if len(sample_ids) > 20:
        sample_ids = sorted(rng.choice(sample_ids, size=20, replace=False).tolist())
    sample = []
    for task_id in sample_ids:
        problem = problems[task_id]
        row = records[task_id]
        sample.append(
            {
                "task_id": task_id,
                "description_snippet": problem["description"][:240],
                "assigned_category": row["category"],
                "confidence": row["confidence"],
                "rationale": row["rationale"],
                "parse_failed": bool(row.get("parse_failed", False)),
            }
        )
    with sample_path.open("w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")

    mathnot = {task_id: MATHNOT[category] for task_id, category in modes.items()}
    mathnot_counts = dict(Counter(mathnot.values()))
    with mathnot_path.open("w", encoding="utf-8") as f:
        json.dump(mathnot, f, indent=2, sort_keys=True)
        f.write("\n")
    mathnot_meta = {
        **meta,
        "source_modes_file": str(modes_path),
        "categories": ["math", "nonmath"],
        "counts": {"math": mathnot_counts.get("math", 0), "nonmath": mathnot_counts.get("nonmath", 0)},
    }
    with mathnot_meta_path.open("w", encoding="utf-8") as f:
        json.dump(mathnot_meta, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps({"counts": meta["counts"], "thin_categories": thin_categories}, indent=2))
    print(f"Wrote category modes to {modes_path}")
    print(f"Wrote math/nonmath modes to {mathnot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-tag MBPP+ problems into algorithmic categories.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model-env", default="ROUTER_MODEL")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thin-threshold", type=int, default=50)
    parser.add_argument("--records-path", default="data/derived/mbpp_category_label_records.jsonl")
    parser.add_argument("--modes-path", default="data/derived/mbpp_modes_category.json")
    parser.add_argument("--meta-path", default="data/derived/mbpp_modes_category_meta.json")
    parser.add_argument("--sample-path", default="data/derived/category_label_sample.json")
    parser.add_argument("--mathnot-path", default="data/derived/mbpp_modes_mathnot.json")
    parser.add_argument("--mathnot-meta-path", default="data/derived/mbpp_modes_mathnot_meta.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_problems, dataset_hash = load_mbpp_plus()
    problems = {
        task_id: problem_for_tagging(problem)
        for task_id, problem in sorted(raw_problems.items())
    }
    target_ids = sorted(problems)
    if args.limit is not None:
        target_ids = target_ids[: args.limit]

    records_path = Path(args.records_path)
    records = load_records(records_path)
    records = {task_id: row for task_id, row in records.items() if task_id in target_ids}

    if args.mock:
        tagger: Any = MockCategoryTagger()
    else:
        tagger = CodexCLICategoryTagger(
            model_env=args.model_env,
            reasoning_effort=args.reasoning_effort,
            timeout_seconds=args.timeout_seconds,
        )

    missing = [task_id for task_id in target_ids if task_id not in records]
    for task_id in tqdm(missing, desc="category tagging", unit="problem"):
        problem = problems[task_id]
        tagged = tagger.tag(problem)
        row = {
            "task_id": task_id,
            "description": problem["description"],
            "signature": problem["signature"],
            "category": tagged["category"],
            "confidence": tagged["confidence"],
            "rationale": tagged["rationale"],
            "parse_failed": bool(tagged.get("parse_failed", False)),
            "raw_response": tagged.get("raw_response"),
            "errors": tagged.get("errors", []),
            "model_id": tagger.model_id,
            "backend": tagger.backend,
            "temperature": 0,
            "reasoning_effort": tagger.reasoning_effort,
            "prompt_hash": prompt_hash(),
        }
        records[task_id] = row
        append_record(records_path, row)

    write_outputs(
        records=records,
        problems=problems,
        dataset_hash=dataset_hash,
        model_id=tagger.model_id,
        backend=tagger.backend,
        reasoning_effort=tagger.reasoning_effort,
        modes_path=Path(args.modes_path),
        meta_path=Path(args.meta_path),
        sample_path=Path(args.sample_path),
        mathnot_path=Path(args.mathnot_path),
        mathnot_meta_path=Path(args.mathnot_meta_path),
        thin_threshold=int(args.thin_threshold),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
