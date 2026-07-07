from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.load_traces import MODEL_ORDER, configured_model_order
from src.simulate import allocation_to_json_keys, normalize_allocation


def system_prompt(model_order: tuple[str, ...]) -> str:
    keys = ", ".join(f"n_{model}" for model in model_order)
    models = ", ".join(model_order)
    execution = " -> ".join(model_order)
    return f"""You are a retry-allocation router.

Your only job is to allocate up to 10 retry attempts across these worker models:
{models}. Execution is sequential in this fixed order: {execution}. Success means
an attempt status is exactly "pass"; "pass/fail" is not success. Optimize for
minimum expected wall-clock time to a verified solution while matching the best
single model's accuracy.

Output only JSON with integer keys {keys}. Do not output code or a problem
solution."""


def allocation_schema(model_order: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            f"n_{model}": {"type": "integer", "minimum": 0, "maximum": 10}
            for model in model_order
        },
        "required": [f"n_{model}" for model in model_order],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class RouterDecision:
    alloc: dict[str, int]
    router_model: str
    router_backend: str
    parse_failed: bool
    prompt_kind: str
    reasoning_effort: str | None = None
    raw_response: str | None = None


def allocation_from_router_json(
    value: Any,
    *,
    budget: int,
    model_order: tuple[str, ...] = MODEL_ORDER,
) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("Router response is not a JSON object")
    alloc = normalize_allocation(value, order=model_order)
    if sum(alloc.values()) > budget:
        raise ValueError(f"Router allocation exceeds budget {budget}: {alloc}")
    return alloc


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in router output")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Extracted router JSON is not an object")
    return parsed


def attempt_summary(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "seconds": [round(float(x), 6) for x in trace["attempt_seconds"]],
        "statuses": list(trace["attempt_statuses"]),
    }


def build_full_trace_material(
    train_task_ids: list[str],
    traces: dict[str, dict[str, Any]],
    model_order: tuple[str, ...],
) -> list[dict[str, Any]]:
    material = []
    for task_id in train_task_ids:
        mode = traces[task_id][model_order[0]]["mode"]
        material.append(
            {
                "task_id": task_id,
                "mode": mode,
                "models": {
                    model: attempt_summary(traces[task_id][model])
                    for model in model_order
                },
            }
        )
    return material


def first_pass_attempt(statuses: list[str]) -> int | None:
    for idx, status in enumerate(statuses, start=1):
        if status == "pass":
            return idx
    return None


def build_mode_summary(
    train_task_ids: list[str],
    traces: dict[str, dict[str, Any]],
    *,
    budget: int,
    model_order: tuple[str, ...],
) -> dict[str, Any]:
    modes = sorted({traces[task_id][model_order[0]]["mode"] for task_id in train_task_ids})
    summary: dict[str, Any] = {}
    for mode in modes:
        mode_ids = [task_id for task_id in train_task_ids if traces[task_id][model_order[0]]["mode"] == mode]
        summary[mode] = {"n": len(mode_ids), "models": {}}
        for model in model_order:
            first_passes = [
                first_pass_attempt(traces[task_id][model]["attempt_statuses"])
                for task_id in mode_ids
            ]
            solve_by_k = []
            for k in range(1, budget + 1):
                solve_by_k.append(
                    sum(fp is not None and fp <= k for fp in first_passes) / max(1, len(first_passes))
                )
            max_attempts = max(len(traces[task_id][model]["attempt_seconds"]) for task_id in mode_ids)
            mean_attempt_seconds = []
            for idx in range(max_attempts):
                values = [
                    traces[task_id][model]["attempt_seconds"][idx]
                    for task_id in mode_ids
                    if idx < len(traces[task_id][model]["attempt_seconds"])
                ]
                mean_attempt_seconds.append(float(np.mean(values)) if values else float("nan"))
            summary[mode]["models"][model] = {
                "solve_rate_by_allocated_attempts": solve_by_k,
                "first_pass_attempts": first_passes,
                "mean_attempt_seconds": mean_attempt_seconds,
            }
    return summary


def build_router_prompt(
    *,
    train_task_ids: list[str],
    traces: dict[str, dict[str, Any]],
    test_mode: str,
    budget: int,
    in_context: str,
    model_order: tuple[str, ...],
    max_full_trace_chars: int = 180_000,
) -> tuple[str, str]:
    base = {
        "deployment_contract": {
            "models": list(model_order),
            "budget_attempts": budget,
            "execution_order": list(model_order),
            "success": "attempt_status == 'pass'",
            "objective": "minimize expected wall-clock time to verified solution while matching best single model accuracy",
        },
        "test_problem": {
            "mode": test_mode,
            "held_out_outcomes": "not provided",
        },
    }
    if in_context == "full_traces":
        payload = dict(base)
        payload["training_traces"] = build_full_trace_material(train_task_ids, traces, model_order)
        text = json.dumps(payload, separators=(",", ":"))
        if len(text) <= max_full_trace_chars:
            return text, "full_traces"

    payload = dict(base)
    payload["training_summary"] = build_mode_summary(
        train_task_ids,
        traces,
        budget=budget,
        model_order=model_order,
    )
    return json.dumps(payload, separators=(",", ":")), "per_mode_summary"


class MockRouter:
    def __init__(
        self,
        alloc: dict[str, Any],
        *,
        budget: int,
        model_order: tuple[str, ...] = MODEL_ORDER,
    ) -> None:
        self.model_order = model_order
        self.alloc = allocation_from_router_json(alloc, budget=budget, model_order=model_order)

    def decide(
        self,
        *,
        train_task_ids: list[str],
        traces: dict[str, dict[str, Any]],
        test_mode: str,
        budget: int,
    ) -> RouterDecision:
        return RouterDecision(
            alloc=self.alloc,
            router_model="mock-router",
            router_backend="mock",
            parse_failed=False,
            prompt_kind="mock",
        )


class OpenAIRouter:
    def __init__(self, config: dict[str, Any], *, budget: int) -> None:
        from openai import OpenAI

        router_config = config["router"]
        model_env = router_config.get("model_env", "ROUTER_MODEL")
        model = os.environ.get(model_env)
        if not model:
            raise RuntimeError(f"Set {model_env} before running the real router")
        self.client = OpenAI()
        self.model = model
        self.temperature = float(router_config.get("temperature", 0))
        self.reasoning_effort = str(router_config.get("reasoning_effort", "medium"))
        self.in_context = str(router_config.get("in_context", "full_traces"))
        self.model_order = configured_model_order(config)
        self.fallback_alloc = allocation_from_router_json(
            router_config.get("parse_fail_default", {"n_7b": 10}),
            budget=budget,
            model_order=self.model_order,
        )

    def decide(
        self,
        *,
        train_task_ids: list[str],
        traces: dict[str, dict[str, Any]],
        test_mode: str,
        budget: int,
    ) -> RouterDecision:
        prompt, prompt_kind = build_router_prompt(
            train_task_ids=train_task_ids,
            traces=traces,
            test_mode=test_mode,
            budget=budget,
            in_context=self.in_context,
            model_order=self.model_order,
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt(self.model_order)},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            reasoning={"effort": self.reasoning_effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "retry_allocation",
                    "schema": allocation_schema(self.model_order),
                    "strict": True,
                }
            },
            max_output_tokens=200,
            store=False,
        )
        raw = getattr(response, "output_text", "") or ""
        router_model = str(getattr(response, "model", self.model))
        try:
            parsed = extract_json_object(raw)
            alloc = allocation_from_router_json(parsed, budget=budget, model_order=self.model_order)
            parse_failed = False
        except Exception:
            alloc = self.fallback_alloc
            parse_failed = True
        return RouterDecision(
            alloc=alloc,
            router_model=router_model,
            router_backend="openai_api",
            parse_failed=parse_failed,
            prompt_kind=prompt_kind,
            reasoning_effort=self.reasoning_effort,
            raw_response=raw,
        )


class CodexCLIRouter:
    def __init__(self, config: dict[str, Any], *, budget: int) -> None:
        router_config = config["router"]
        model_env = router_config.get("model_env", "ROUTER_MODEL")
        model = os.environ.get(model_env)
        if not model:
            raise RuntimeError(f"Set {model_env} before running the Codex CLI router")
        self.model = model
        self.reasoning_effort = str(router_config.get("reasoning_effort", "medium"))
        self.in_context = str(router_config.get("in_context", "full_traces"))
        self.model_order = configured_model_order(config)
        self.fallback_alloc = allocation_from_router_json(
            router_config.get("parse_fail_default", {"n_7b": 10}),
            budget=budget,
            model_order=self.model_order,
        )

    def decide(
        self,
        *,
        train_task_ids: list[str],
        traces: dict[str, dict[str, Any]],
        test_mode: str,
        budget: int,
    ) -> RouterDecision:
        prompt, prompt_kind = build_router_prompt(
            train_task_ids=train_task_ids,
            traces=traces,
            test_mode=test_mode,
            budget=budget,
            in_context=self.in_context,
            model_order=self.model_order,
        )
        full_prompt = (
            system_prompt(self.model_order)
            + "\n\nReturn the allocation as your final answer and nothing else.\n\n"
            + prompt
        )
        with tempfile.TemporaryDirectory(prefix="router_codex_") as tmp:
            tmp_path = Path(tmp)
            schema_path = tmp_path / "allocation_schema.json"
            output_path = tmp_path / "allocation_output.txt"
            schema_path.write_text(json.dumps(allocation_schema(self.model_order)), encoding="utf-8")
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
                self.model,
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
                timeout=300,
            )
            raw = output_path.read_text(encoding="utf-8") if output_path.exists() else completed.stdout
            if completed.returncode != 0:
                raw = raw + "\n" + completed.stderr

        try:
            parsed = extract_json_object(raw)
            alloc = allocation_from_router_json(parsed, budget=budget, model_order=self.model_order)
            parse_failed = False
        except Exception:
            alloc = self.fallback_alloc
            parse_failed = True
        return RouterDecision(
            alloc=alloc,
            router_model=self.model,
            router_backend="codex_cli",
            parse_failed=parse_failed,
            prompt_kind=prompt_kind,
            reasoning_effort=self.reasoning_effort,
            raw_response=raw,
        )


def decision_alloc_json(
    decision: RouterDecision,
    order: tuple[str, ...] = MODEL_ORDER,
) -> dict[str, int]:
    return allocation_to_json_keys(decision.alloc, order=order)
