"""Engine discovery — probe running engines and aggregate available models."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from openjarvis.core.config import JarvisConfig
from openjarvis.core.registry import EngineRegistry
from openjarvis.engine._base import InferenceEngine

logger = logging.getLogger(__name__)

# Map registry keys to config host attribute (None = no host arg)
_HOST_MAP: Dict[str, str | None] = {
    "ollama": "ollama_host",
    "vllm": "vllm_host",
    "llamacpp": "llamacpp_host",
    "sglang": "sglang_host",
    "mlx": "mlx_host",
    "lmstudio": "lmstudio_host",
    "exo": "exo_host",
    "nexa": "nexa_host",
    "uzu": "uzu_host",
    "apple_fm": "apple_fm_host",
    "cloud": None,
    "litellm": None,
}


def _make_engine(key: str, config: JarvisConfig) -> InferenceEngine:
    """Instantiate a registered engine with the appropriate config host."""
    cls = EngineRegistry.get(key)
    host_attr = _HOST_MAP.get(key)
    if host_attr is not None:
        host = getattr(config.engine, host_attr, None)
        if host:
            return cls(host=host)
    return cls()


def discover_engines(config: JarvisConfig) -> List[Tuple[str, InferenceEngine]]:
    """Probe registered engines and return ``[(key, instance)]`` for healthy ones.

    Results are sorted with the config default engine first.
    """
    healthy: List[Tuple[str, InferenceEngine]] = []
    for key in EngineRegistry.keys():
        try:
            engine = _make_engine(key, config)
            if engine.health():
                healthy.append((key, engine))
        except Exception as exc:
            logger.debug("Engine %r failed during discovery: %s", key, exc)
            continue

    default_key = config.engine.default

    def sort_key(item: Tuple[str, Any]) -> Tuple[int, str]:
        return (0 if item[0] == default_key else 1, item[0])

    healthy.sort(key=sort_key)
    return healthy


def discover_models(
    engines: List[Tuple[str, InferenceEngine]],
) -> Dict[str, List[str]]:
    """Call ``list_models()`` on each engine and return a dict."""
    result: Dict[str, List[str]] = {}
    for key, engine in engines:
        try:
            result[key] = engine.list_models()
        except Exception as exc:
            logger.debug("Failed to list models for engine %r: %s", key, exc)
            result[key] = []
    return result


def get_engine(
    config: JarvisConfig, engine_key: str | None = None
) -> Tuple[str, InferenceEngine] | None:
    """Get a specific engine by key, or the default with fallback.

    Returns ``(key, engine_instance)`` or ``None`` if no engine is available.
    """
    if engine_key:
        if EngineRegistry.contains(engine_key):
            try:
                engine = _make_engine(engine_key, config)
                if engine.health():
                    return (engine_key, engine)
            except Exception as exc:
                logger.debug("Engine %r health check failed: %s", engine_key, exc)
        return None

    # Try default first
    default_key = config.engine.default
    if EngineRegistry.contains(default_key):
        try:
            engine = _make_engine(default_key, config)
            if engine.health():
                return (default_key, engine)
        except Exception as exc:
            logger.debug("Default engine %r health check failed: %s", default_key, exc)

    # Fallback to any healthy engine
    healthy = discover_engines(config)
    return healthy[0] if healthy else None


__all__ = ["discover_engines", "discover_models", "get_engine"]
