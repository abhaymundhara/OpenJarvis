"""Built-in model catalog with well-known ModelSpec entries."""

from __future__ import annotations

from typing import List

from openjarvis.core.registry import ModelRegistry
from openjarvis.core.types import ModelSpec, Quantization

BUILTIN_MODELS: List[ModelSpec] = [
    # -----------------------------------------------------------------------
    # Local models — Dense
    # -----------------------------------------------------------------------
    ModelSpec(
        model_id="qwen3:8b",
        name="Qwen3 8B",
        parameter_count_b=8.2,
        context_length=32768,
        supported_engines=("vllm", "ollama", "llamacpp", "sglang"),
        provider="alibaba",
        metadata={"architecture": "dense"},
    ),
    ModelSpec(
        model_id="qwen3:32b",
        name="Qwen3 32B",
        parameter_count_b=32.0,
        context_length=32768,
        min_vram_gb=20.0,
        supported_engines=("ollama", "vllm"),
        provider="alibaba",
        metadata={"architecture": "dense"},
    ),
    ModelSpec(
        model_id="llama3.3:70b",
        name="Llama 3.3 70B",
        parameter_count_b=70.0,
        context_length=131072,
        min_vram_gb=40.0,
        supported_engines=("ollama", "vllm"),
        provider="meta",
        metadata={"architecture": "dense"},
    ),
    ModelSpec(
        model_id="llama3.2:3b",
        name="Llama 3.2 3B",
        parameter_count_b=3.0,
        context_length=131072,
        supported_engines=("ollama", "vllm", "llamacpp"),
        provider="meta",
        metadata={"architecture": "dense"},
    ),
    ModelSpec(
        model_id="deepseek-coder-v2:16b",
        name="DeepSeek Coder V2 16B",
        parameter_count_b=16.0,
        context_length=131072,
        supported_engines=("ollama", "vllm"),
        provider="deepseek",
        metadata={"architecture": "dense"},
    ),
    ModelSpec(
        model_id="mistral:7b",
        name="Mistral 7B",
        parameter_count_b=7.0,
        context_length=32768,
        supported_engines=("ollama", "vllm", "llamacpp"),
        provider="mistral",
        metadata={"architecture": "dense"},
    ),
    # -----------------------------------------------------------------------
    # Local models — Mixture of Experts (MoE)
    # -----------------------------------------------------------------------
    ModelSpec(
        model_id="gpt-oss:120b",
        name="GPT-OSS 120B",
        parameter_count_b=117.0,
        active_parameter_count_b=5.1,
        context_length=131072,
        min_vram_gb=12.0,
        supported_engines=("vllm", "ollama"),
        provider="open-source",
        metadata={"architecture": "moe"},
    ),
    ModelSpec(
        model_id="glm-4.7-flash",
        name="GLM 4.7 Flash",
        parameter_count_b=30.0,
        active_parameter_count_b=3.0,
        context_length=131072,
        min_vram_gb=8.0,
        supported_engines=("vllm", "sglang", "llamacpp"),
        provider="zhipu",
        metadata={"architecture": "moe"},
    ),
    ModelSpec(
        model_id="trinity-mini",
        name="Trinity Mini",
        parameter_count_b=26.0,
        active_parameter_count_b=3.0,
        context_length=128000,
        min_vram_gb=8.0,
        supported_engines=("vllm", "llamacpp"),
        provider="trinity",
        metadata={"architecture": "moe"},
    ),
    # -----------------------------------------------------------------------
    # Local models — TeichAI Distilled
    # -----------------------------------------------------------------------
    ModelSpec(
        model_id="teichai/glm-4.7-flash-opus-distill",
        name="GLM 4.7 Flash Claude Opus 4.5 Distill",
        parameter_count_b=30.0,
        active_parameter_count_b=3.0,
        context_length=131072,
        min_vram_gb=8.0,
        supported_engines=("vllm", "llamacpp"),
        provider="teichai",
        metadata={
            "architecture": "moe",
            "hf_repo": (
                "TeichAI/GLM-4.7-Flash-Claude-"
                "Opus-4.5-High-Reasoning-Distill-GGUF"
            ),
            "teacher": "Claude Opus 4.5",
            "quantization": "GGUF Q4_K_M / Q8_0",
            "license": "apache-2.0",
        },
    ),
    ModelSpec(
        model_id="teichai/qwen3-14b-gpt5.2-distill",
        name="Qwen3 14B GPT-5.2 Distill",
        parameter_count_b=14.8,
        context_length=32768,
        min_vram_gb=10.0,
        supported_engines=("vllm", "llamacpp"),
        provider="teichai",
        metadata={
            "architecture": "dense",
            "hf_repo": "TeichAI/Qwen3-14B-GPT-5.2-Distill-GGUF",
            "teacher": "GPT-5.2",
            "quantization": "GGUF Q4_K_M / Q8_0",
            "license": "apache-2.0",
        },
    ),
    ModelSpec(
        model_id="teichai/nemotron-14b-opus-distill",
        name="Nemotron Cascade 14B Claude Opus Distill",
        parameter_count_b=14.8,
        context_length=32768,
        min_vram_gb=10.0,
        supported_engines=("vllm", "llamacpp"),
        provider="teichai",
        metadata={
            "architecture": "dense",
            "hf_repo": "TeichAI/Nemotron-Cascade-14B-Claude-Opus-Distill-GGUF",
            "teacher": "Claude 4.5 Opus",
            "quantization": "GGUF Q4_K_M / Q8_0",
            "license": "apache-2.0",
        },
    ),
    # -----------------------------------------------------------------------
    # Local models — Grid Search Eval Models
    # -----------------------------------------------------------------------
    ModelSpec(
        model_id="openai/gpt-oss-120b",
        name="GPT-OSS 120B (HuggingFace)",
        parameter_count_b=120.0,
        active_parameter_count_b=5.1,
        context_length=131072,
        min_vram_gb=80.0,
        supported_engines=("vllm", "sglang"),
        provider="openai",
        metadata={
            "architecture": "moe",
            "hf_repo": "openai/gpt-oss-120b",
            "license": "apache-2.0",
        },
    ),
    ModelSpec(
        model_id="Qwen/Qwen3.5-122B-A10B-FP8",
        name="Qwen3.5 122B A10B FP8",
        parameter_count_b=122.0,
        active_parameter_count_b=10.0,
        context_length=262144,
        quantization=Quantization.FP8,
        min_vram_gb=60.0,
        supported_engines=("vllm", "sglang"),
        provider="alibaba",
        metadata={
            "architecture": "moe",
            "hf_repo": "Qwen/Qwen3.5-122B-A10B-FP8",
            "license": "apache-2.0",
        },
    ),
    ModelSpec(
        model_id="unsloth/Qwen3.5-397B-A17B-GGUF",
        name="Qwen3.5 397B A17B GGUF",
        parameter_count_b=397.0,
        active_parameter_count_b=17.0,
        context_length=262144,
        quantization=Quantization.GGUF_Q4,
        min_vram_gb=214.0,
        supported_engines=("llamacpp", "ollama"),
        provider="alibaba",
        metadata={
            "architecture": "moe",
            "hf_repo": "unsloth/Qwen3.5-397B-A17B-GGUF",
            "quantization": "GGUF Dynamic 2.0 (multiple variants)",
        },
    ),
    ModelSpec(
        model_id="unsloth/Kimi-K2.5-GGUF",
        name="Kimi K2.5 GGUF",
        parameter_count_b=1000.0,
        active_parameter_count_b=32.0,
        context_length=262144,
        quantization=Quantization.GGUF_Q4,
        min_vram_gb=240.0,
        supported_engines=("llamacpp", "ollama"),
        provider="moonshot",
        metadata={
            "architecture": "moe",
            "hf_repo": "unsloth/Kimi-K2.5-GGUF",
            "num_experts": 384,
            "experts_per_token": 8,
            "quantization": "GGUF Dynamic 2.0 (multiple variants)",
        },
    ),
    ModelSpec(
        model_id="unsloth/GLM-5-GGUF",
        name="GLM-5 GGUF",
        parameter_count_b=744.0,
        active_parameter_count_b=40.0,
        context_length=202752,
        quantization=Quantization.GGUF_Q4,
        min_vram_gb=241.0,
        supported_engines=("llamacpp", "ollama"),
        provider="zhipu",
        metadata={
            "architecture": "moe",
            "hf_repo": "unsloth/GLM-5-GGUF",
            "quantization": "GGUF Dynamic 2.0 (multiple variants)",
        },
    ),
    # -----------------------------------------------------------------------
    # Cloud models — OpenAI
    # -----------------------------------------------------------------------
    ModelSpec(
        model_id="gpt-4o",
        name="GPT-4o",
        parameter_count_b=0.0,
        context_length=128000,
        supported_engines=("cloud",),
        provider="openai",
        requires_api_key=True,
        metadata={"pricing_input": 2.50, "pricing_output": 10.00},
    ),
    ModelSpec(
        model_id="gpt-4o-mini",
        name="GPT-4o Mini",
        parameter_count_b=0.0,
        context_length=128000,
        supported_engines=("cloud",),
        provider="openai",
        requires_api_key=True,
        metadata={"pricing_input": 0.15, "pricing_output": 0.60},
    ),
    ModelSpec(
        model_id="gpt-5-mini",
        name="GPT-5 Mini",
        parameter_count_b=0.0,
        context_length=400000,
        supported_engines=("cloud",),
        provider="openai",
        requires_api_key=True,
        metadata={"pricing_input": 0.25, "pricing_output": 2.00},
    ),
    ModelSpec(
        model_id="gpt-5-mini-2025-08-07",
        name="GPT-5 Mini (2025-08-07)",
        parameter_count_b=0.0,
        context_length=400000,
        supported_engines=("cloud",),
        provider="openai",
        requires_api_key=True,
        metadata={"pricing_input": 0.25, "pricing_output": 2.00},
    ),
    # -----------------------------------------------------------------------
    # Cloud models — Anthropic
    # -----------------------------------------------------------------------
    ModelSpec(
        model_id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        parameter_count_b=0.0,
        context_length=200000,
        supported_engines=("cloud",),
        provider="anthropic",
        requires_api_key=True,
        metadata={"pricing_input": 3.00, "pricing_output": 15.00},
    ),
    ModelSpec(
        model_id="claude-opus-4-20250514",
        name="Claude Opus 4",
        parameter_count_b=0.0,
        context_length=200000,
        supported_engines=("cloud",),
        provider="anthropic",
        requires_api_key=True,
        metadata={"pricing_input": 15.00, "pricing_output": 75.00},
    ),
    ModelSpec(
        model_id="claude-opus-4-6",
        name="Claude Opus 4.6",
        parameter_count_b=0.0,
        context_length=200000,
        supported_engines=("cloud",),
        provider="anthropic",
        requires_api_key=True,
        metadata={"pricing_input": 5.00, "pricing_output": 25.00},
    ),
    ModelSpec(
        model_id="claude-sonnet-4-6",
        name="Claude Sonnet 4.6",
        parameter_count_b=0.0,
        context_length=200000,
        supported_engines=("cloud",),
        provider="anthropic",
        requires_api_key=True,
        metadata={"pricing_input": 3.00, "pricing_output": 15.00},
    ),
    ModelSpec(
        model_id="claude-haiku-4-5",
        name="Claude Haiku 4.5",
        parameter_count_b=0.0,
        context_length=200000,
        supported_engines=("cloud",),
        provider="anthropic",
        requires_api_key=True,
        metadata={"pricing_input": 1.00, "pricing_output": 5.00},
    ),
    # -----------------------------------------------------------------------
    # Cloud models — Google
    # -----------------------------------------------------------------------
    ModelSpec(
        model_id="gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        parameter_count_b=0.0,
        context_length=1000000,
        supported_engines=("cloud",),
        provider="google",
        requires_api_key=True,
        metadata={"pricing_input": 1.25, "pricing_output": 10.00},
    ),
    ModelSpec(
        model_id="gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        parameter_count_b=0.0,
        context_length=1000000,
        supported_engines=("cloud",),
        provider="google",
        requires_api_key=True,
        metadata={"pricing_input": 0.30, "pricing_output": 2.50},
    ),
    ModelSpec(
        model_id="gemini-3-pro",
        name="Gemini 3 Pro",
        parameter_count_b=0.0,
        context_length=1000000,
        supported_engines=("cloud",),
        provider="google",
        requires_api_key=True,
        metadata={"pricing_input": 2.00, "pricing_output": 12.00},
    ),
    ModelSpec(
        model_id="gemini-3-flash",
        name="Gemini 3 Flash",
        parameter_count_b=0.0,
        context_length=1000000,
        supported_engines=("cloud",),
        provider="google",
        requires_api_key=True,
        metadata={"pricing_input": 0.50, "pricing_output": 3.00},
    ),
]


def register_builtin_models() -> None:
    """Populate ``ModelRegistry`` with well-known models."""
    for spec in BUILTIN_MODELS:
        if not ModelRegistry.contains(spec.model_id):
            ModelRegistry.register_value(spec.model_id, spec)


def merge_discovered_models(engine_key: str, model_ids: List[str]) -> None:
    """Create minimal ``ModelSpec`` entries for models not already in the registry."""
    for model_id in model_ids:
        if not ModelRegistry.contains(model_id):
            spec = ModelSpec(
                model_id=model_id,
                name=model_id,
                parameter_count_b=0.0,
                context_length=0,
                supported_engines=(engine_key,),
            )
            ModelRegistry.register_value(model_id, spec)


__all__ = ["BUILTIN_MODELS", "merge_discovered_models", "register_builtin_models"]
