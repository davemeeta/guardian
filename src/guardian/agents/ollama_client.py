"""Thin wrapper around a local Ollama server. No network calls leave the
machine — this is the only place in the agent layer that talks to the LLM,
so it's also the one place data-sovereignty depends on staying local."""
import json
from dataclasses import dataclass

import requests

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_HOST = "http://localhost:11434"
TIMEOUT_S = 120


@dataclass
class LLMResponse:
    text: str
    model: str
    eval_count: int
    eval_duration_s: float


def generate(
    prompt: str,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    json_mode: bool = False,
    temperature: float = 0.2,
    host: str = DEFAULT_HOST,
) -> LLMResponse:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system
    if json_mode:
        payload["format"] = "json"

    resp = requests.post(f"{host}/api/generate", json=payload, timeout=TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    return LLMResponse(
        text=data["response"],
        model=model,
        eval_count=data.get("eval_count", 0),
        eval_duration_s=data.get("eval_duration", 0) / 1e9,
    )


def generate_json(
    prompt: str,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    host: str = DEFAULT_HOST,
) -> tuple[dict | None, LLMResponse]:
    """Best-effort structured output. Returns (parsed_dict_or_None, raw_response) —
    callers must handle None (a small local model occasionally emits
    malformed JSON even in JSON mode) rather than assume parsing succeeds."""
    response = generate(prompt, system=system, model=model, json_mode=True, temperature=temperature, host=host)
    try:
        return json.loads(response.text), response
    except json.JSONDecodeError:
        return None, response
