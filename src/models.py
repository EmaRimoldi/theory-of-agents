from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_OLLAMA_URL = "http://localhost:11434"


class ModelUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelInfo:
    model: str
    digest: str
    quantization: str
    details: dict[str, Any]


@dataclass(frozen=True)
class Generation:
    text: str
    tokens_generated: int
    wall_seconds: float


def _post_json(url: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code} for {url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise ModelUnavailableError(
            "Could not reach Ollama at http://localhost:11434. "
            "Start Ollama, then rerun the command."
        ) from exc


def _get_json(url: str, timeout: float = 15.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise ModelUnavailableError(
            "Could not reach Ollama at http://localhost:11434. "
            "Start Ollama, then rerun the command."
        ) from exc


def resolve_model_info(model: str, base_url: str = DEFAULT_OLLAMA_URL) -> ModelInfo:
    tags = _get_json(f"{base_url.rstrip('/')}/api/tags")
    candidates = tags.get("models", [])
    for item in candidates:
        names = {item.get("name"), item.get("model")}
        if model in names:
            details = item.get("details") or {}
            quantization = details.get("quantization_level") or details.get("quantization") or "unknown"
            return ModelInfo(
                model=model,
                digest=item.get("digest", "unknown"),
                quantization=quantization,
                details=details,
            )

    raise ModelUnavailableError(
        f"Ollama model {model!r} is not pulled locally. Run:\n"
        f"  ollama pull {model}"
    )


def generate_ollama(
    model: str,
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> Generation:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_new_tokens,
            "seed": seed,
        },
    }
    response = _post_json(f"{base_url.rstrip('/')}/api/generate", payload, timeout=900.0)
    text = response.get("response", "")
    tokens_generated = response.get("eval_count")
    if tokens_generated is None:
        tokens_generated = max(1, len(text.split()))
    wall_seconds = float(response.get("total_duration", 0.0)) / 1_000_000_000.0
    return Generation(
        text=text,
        tokens_generated=int(tokens_generated),
        wall_seconds=wall_seconds,
    )


def make_completion_prompt(problem: dict[str, Any], dataset_name: str = "humanevalplus") -> str:
    if dataset_name == "mbppplus":
        instruction = (
            "Write Python code that solves the following MBPP+ task. "
            "Return only Python code, with no Markdown fences or explanation."
        )
    elif dataset_name == "bbh":
        return (
            "Solve the following BIG-Bench Hard problem. Reason step by step, "
            "then end with exactly one final line in this format:\n"
            "Answer: <final answer>\n\n"
            f"{problem['prompt']}"
        )
    else:
        instruction = (
            "Complete the following Python function for HumanEval. "
            "Return only Python code, with no Markdown fences or explanation."
        )
    return f"{instruction}\n\n{problem['prompt']}"


def strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if "```" not in stripped:
        return stripped

    parts = stripped.split("```")
    code_blocks = []
    for idx in range(1, len(parts), 2):
        block = parts[idx]
        if block.lstrip().startswith("python"):
            block = block.lstrip()[len("python") :]
        code_blocks.append(block.strip("\n"))
    return "\n\n".join(code_blocks).strip() if code_blocks else stripped.replace("```", "")


def build_solution(problem: dict[str, Any], completion: str) -> str:
    code = strip_markdown_fences(completion)
    entry_point = problem["entry_point"]
    if f"def {entry_point}" in code:
        return code
    return problem["prompt"] + code
