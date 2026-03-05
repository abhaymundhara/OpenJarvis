"""Ollama inference engine backend."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, Dict, List

import httpx

from openjarvis.core.registry import EngineRegistry
from openjarvis.core.types import Message
from openjarvis.engine._base import (
    EngineConnectionError,
    InferenceEngine,
    messages_to_dicts,
)


@EngineRegistry.register("ollama")
class OllamaEngine(InferenceEngine):
    """Ollama backend via its native HTTP API."""

    engine_id = "ollama"

    def __init__(
        self,
        host: str = "http://localhost:11434",
        *,
        timeout: float = 120.0,
    ) -> None:
        self._host = host.rstrip("/")
        self._client = httpx.Client(base_url=self._host, timeout=timeout)

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages_to_dicts(messages),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": kwargs.get("num_ctx", 8192),
            },
        }
        # Pass tools if provided
        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = tools
        try:
            resp = self._client.post("/api/chat", json=payload)
            if resp.status_code == 400 and tools:
                # Model may not support function calling -- retry without tools
                payload.pop("tools", None)
                resp = self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise EngineConnectionError(
                f"Ollama not reachable at {self._host}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500] if exc.response else ""
            raise RuntimeError(
                f"Ollama returned {exc.response.status_code}: {body}"
            ) from exc
        data = resp.json()
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        result: Dict[str, Any] = {
            "content": data.get("message", {}).get("content", ""),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "model": data.get("model", model),
            "finish_reason": "stop",
        }
        # Extract timing from Ollama response (nanoseconds → seconds)
        result["ttft"] = data.get("prompt_eval_duration", 0) / 1e9
        result["engine_timing"] = {k: data[k] for k in
            ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")
            if k in data}
        # Extract tool calls if present
        raw_tool_calls = data.get("message", {}).get("tool_calls", [])
        if raw_tool_calls:
            tool_calls = []
            for i, tc in enumerate(raw_tool_calls):
                raw_args = tc.get("function", {}).get(
                    "arguments", "{}",
                )
                tool_calls.append({
                    "id": tc.get("id", f"call_{i}"),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": (
                        json.dumps(raw_args)
                        if isinstance(raw_args, dict)
                        else raw_args
                    ),
                })
            result["tool_calls"] = tool_calls
        return result

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages_to_dicts(messages),
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": kwargs.get("num_ctx", 8192),
            },
        }
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done", False):
                        break
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise EngineConnectionError(
                f"Ollama not reachable at {self._host}"
            ) from exc

    def list_models(self) -> List[str]:
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
            return []
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]

    def health(self) -> bool:
        try:
            resp = self._client.get("/api/tags", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()


__all__ = ["OllamaEngine"]
