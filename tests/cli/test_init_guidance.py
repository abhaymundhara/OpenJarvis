"""Tests for ``jarvis init`` next-steps guidance."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from openjarvis.cli import cli
from openjarvis.cli.init_cmd import _next_steps_text


class TestInitShowsNextSteps:
    def test_init_shows_next_steps(self, tmp_path: Path) -> None:
        """Init command prints next-steps panel after writing config."""
        config_dir = tmp_path / ".openjarvis"
        config_path = config_dir / "config.toml"
        with (
            mock.patch(
                "openjarvis.cli.init_cmd.DEFAULT_CONFIG_DIR", config_dir
            ),
            mock.patch(
                "openjarvis.cli.init_cmd.DEFAULT_CONFIG_PATH", config_path
            ),
        ):
            result = CliRunner().invoke(cli, ["init"])
        assert result.exit_code == 0
        assert "Getting Started" in result.output
        assert "jarvis ask" in result.output
        assert "jarvis doctor" in result.output


class TestNextStepsOllama:
    def test_next_steps_ollama(self) -> None:
        text = _next_steps_text("ollama")
        assert "ollama.com/install.sh" in text
        assert "ollama serve" in text
        assert "ollama pull" in text
        assert "jarvis ask" in text
        assert "jarvis doctor" in text

    def test_next_steps_ollama_with_model(self) -> None:
        text = _next_steps_text("ollama", "qwen3.5:14b")
        assert "ollama pull qwen3.5:14b" in text

    def test_next_steps_ollama_default_model(self) -> None:
        text = _next_steps_text("ollama")
        assert "ollama pull qwen3.5:3b" in text


class TestNextStepsVllm:
    def test_next_steps_vllm(self) -> None:
        text = _next_steps_text("vllm")
        assert "pip install vllm" in text
        assert "vllm serve" in text
        assert "jarvis ask" in text
        assert "jarvis doctor" in text


class TestNextStepsLlamacpp:
    def test_next_steps_llamacpp(self) -> None:
        text = _next_steps_text("llamacpp")
        assert "llama.cpp" in text
        assert "llama-server" in text
        assert "jarvis ask" in text
        assert "jarvis doctor" in text


class TestNextStepsMlx:
    def test_next_steps_mlx(self) -> None:
        text = _next_steps_text("mlx")
        assert "mlx-lm" in text
        assert "mlx_lm.server" in text
        assert "jarvis ask" in text
        assert "jarvis doctor" in text
