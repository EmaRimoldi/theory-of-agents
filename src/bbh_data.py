from __future__ import annotations

import hashlib
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.execute import ExecutionResult


BBH_SOURCE_ID = "lukaemon/bbh"
BBH_SPLIT = "test"

BBH_FAMILY_BY_SUBTASK = {
    "boolean_expressions": "logical_deduction",
    "causal_judgement": "logical_deduction",
    "date_understanding": "spatial_temporal",
    "disambiguation_qa": "language",
    "dyck_languages": "algorithmic",
    "formal_fallacies": "logical_deduction",
    "geometric_shapes": "spatial_temporal",
    "hyperbaton": "language",
    "logical_deduction_five_objects": "logical_deduction",
    "logical_deduction_seven_objects": "logical_deduction",
    "logical_deduction_three_objects": "logical_deduction",
    "movie_recommendation": "language",
    "multistep_arithmetic_two": "arithmetic",
    "navigate": "spatial_temporal",
    "object_counting": "arithmetic",
    "penguins_in_a_table": "algorithmic",
    "reasoning_about_colored_objects": "algorithmic",
    "ruin_names": "language",
    "salient_translation_error_detection": "language",
    "snarks": "language",
    "sports_understanding": "language",
    "temporal_sequences": "spatial_temporal",
    "tracking_shuffled_objects_five_objects": "spatial_temporal",
    "tracking_shuffled_objects_seven_objects": "spatial_temporal",
    "tracking_shuffled_objects_three_objects": "spatial_temporal",
    "web_of_lies": "logical_deduction",
    "word_sorting": "algorithmic",
}

BBH_FAMILY_ORDER = [
    "arithmetic",
    "logical_deduction",
    "language",
    "spatial_temporal",
    "algorithmic",
]

BBH_EXTRACTION_RULE = "bbh_v1_final_answer_marker_option_label_preserve_symbolic"


@dataclass(frozen=True)
class BBHEvalResult:
    execution: ExecutionResult
    extracted_answer: str
    normalized_prediction: str
    normalized_target: str


def load_bbh(
    *,
    max_examples_per_subtask: int | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], str, dict[str, Any]]:
    from datasets import get_dataset_config_names, load_dataset

    configs = sorted(get_dataset_config_names(BBH_SOURCE_ID))
    missing_mapping = sorted(set(configs) - set(BBH_FAMILY_BY_SUBTASK))
    if missing_mapping:
        raise ValueError(f"BBH subtasks missing family mapping: {missing_mapping}")

    problems: dict[str, dict[str, Any]] = {}
    expected_outputs: dict[str, str] = {}
    subtask_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    hash_payload: list[tuple[str, str, str]] = []

    for subtask in configs:
        dataset = load_dataset(BBH_SOURCE_ID, subtask, split=BBH_SPLIT)
        source_counts[subtask] = len(dataset)
        count = len(dataset)
        if max_examples_per_subtask is not None:
            count = min(count, int(max_examples_per_subtask))
        subtask_counts[subtask] = count
        for idx in range(count):
            item = dataset[idx]
            task_id = f"BBH/{subtask}/{idx:03d}"
            target = str(item["target"])
            problems[task_id] = {
                "task_id": task_id,
                "prompt": str(item["input"]),
                "target": target,
                "subtask": subtask,
                "family": BBH_FAMILY_BY_SUBTASK[subtask],
                "source": BBH_SOURCE_ID,
                "source_split": BBH_SPLIT,
                "source_index": idx,
            }
            expected_outputs[task_id] = target
            hash_payload.append((task_id, str(item["input"]), target))

    digest = hashlib.sha256(
        json.dumps(
            {
                "source": BBH_SOURCE_ID,
                "split": BBH_SPLIT,
                "max_examples_per_subtask": max_examples_per_subtask,
                "items": hash_payload,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "source": BBH_SOURCE_ID,
        "split": BBH_SPLIT,
        "actual_subtask_count": len(configs),
        "subtasks": configs,
        "source_counts": source_counts,
        "used_counts": subtask_counts,
        "max_examples_per_subtask": max_examples_per_subtask,
        "dataset_hash": digest,
    }
    return problems, expected_outputs, digest, metadata


def ensure_bbh_modes(
    *,
    problems: dict[str, dict[str, Any]],
    derived_dir: Path,
    modes_filename: str,
    raw_modes_filename: str,
    groups_filename: str,
    meta_filename: str,
    metadata: dict[str, Any],
) -> dict[str, str]:
    derived_dir.mkdir(parents=True, exist_ok=True)
    modes_path = derived_dir / modes_filename
    raw_modes_path = derived_dir / raw_modes_filename
    groups_path = derived_dir / groups_filename
    meta_path = derived_dir / meta_filename

    if modes_path.exists():
        with modes_path.open("r", encoding="utf-8") as f:
            modes = json.load(f)
        missing = set(problems) - set(modes)
        if missing:
            raise ValueError(f"Existing {modes_path} is missing {len(missing)} BBH examples")
        return {task_id: str(modes[task_id]) for task_id in problems}

    modes = {task_id: str(problem["family"]) for task_id, problem in problems.items()}
    raw_modes = {task_id: str(problem["subtask"]) for task_id, problem in problems.items()}
    family_counts = {family: 0 for family in BBH_FAMILY_ORDER}
    for family in modes.values():
        family_counts[family] = family_counts.get(family, 0) + 1
    thin_families = {family: count for family, count in family_counts.items() if count < 150}

    with modes_path.open("w", encoding="utf-8") as f:
        json.dump(modes, f, indent=2, sort_keys=True)
        f.write("\n")
    with raw_modes_path.open("w", encoding="utf-8") as f:
        json.dump(raw_modes, f, indent=2, sort_keys=True)
        f.write("\n")
    with groups_path.open("w", encoding="utf-8") as f:
        json.dump(BBH_FAMILY_BY_SUBTASK, f, indent=2, sort_keys=True)
        f.write("\n")
    meta = dict(metadata)
    meta.update(
        {
            "mode_schema": "bbh_cognitive_family",
            "family_order": BBH_FAMILY_ORDER,
            "family_counts": family_counts,
            "thin_family_threshold": 150,
            "thin_families": thin_families,
            "raw_modes_file": str(raw_modes_path),
            "groups_file": str(groups_path),
        }
    )
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        "Frozen BBH modes "
        f"source={metadata['source']} subtasks={metadata['actual_subtask_count']} "
        f"family_counts={family_counts} thin={thin_families}"
    )
    return modes


def extract_final_answer(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    marker_patterns = [
        r"^(?:final\s+answer|answer)\s*[:\-]\s*(.+)$",
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\-]?\s*(.+)$",
    ]
    for line in reversed(lines):
        clean = line.strip().strip("`")
        for pattern in marker_patterns:
            match = re.search(pattern, clean, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
    return lines[-1].strip().strip("`")


def _is_option_label(value: str) -> bool:
    return re.fullmatch(r"\(?[A-Za-z]\)?", value.strip()) is not None


def _normalize_option(value: str) -> str:
    explicit = re.search(r"\(([A-Za-z])\)", value)
    if explicit:
        return explicit.group(1).lower()
    words = re.findall(r"[A-Za-z]", value)
    return words[0].lower() if len(words) == 1 else value.strip().lower()


def normalize_bbh_answer(value: str, *, target: str | None = None) -> str:
    value = value.strip()
    value = re.sub(r"^\s*(?:answer|final answer)\s*[:\-]\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^\s*(?:the\s+)?answer\s+is\s*", "", value, flags=re.IGNORECASE)
    value = value.strip().strip("\"'`")
    value = value.strip()

    if target is not None and _is_option_label(target):
        return _normalize_option(value)

    if re.search(r"[A-Za-z0-9]", value):
        normalized = value.lower()
        normalized = normalized.strip()
        normalized = normalized.strip(string.whitespace + ".;, :")
        normalized = normalized.replace(",", "")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    return re.sub(r"\s+", "", value)


def evaluate_bbh_completion(completion: str, target: str) -> BBHEvalResult:
    extracted = extract_final_answer(completion)
    normalized_prediction = normalize_bbh_answer(extracted, target=target)
    normalized_target = normalize_bbh_answer(target, target=target)
    passed = normalized_prediction == normalized_target
    status = "pass" if passed else "fail"
    execution = ExecutionResult(
        passed=passed,
        base_status=status,
        plus_status=status,
        timed_out=False,
        stdout="",
        stderr="",
    )
    return BBHEvalResult(
        execution=execution,
        extracted_answer=extracted,
        normalized_prediction=normalized_prediction,
        normalized_target=normalized_target,
    )
