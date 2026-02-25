"""Command-line interface for OpenJarvis (Click-based)."""

from __future__ import annotations

import click

import openjarvis
from openjarvis.cli.ask import ask
from openjarvis.cli.bench_cmd import bench
from openjarvis.cli.channel_cmd import channel
from openjarvis.cli.init_cmd import init
from openjarvis.cli.memory_cmd import memory
from openjarvis.cli.model import model
from openjarvis.cli.scheduler_cmd import scheduler
from openjarvis.cli.serve import serve
from openjarvis.cli.telemetry_cmd import telemetry


@click.group(help="OpenJarvis — modular AI assistant backend")
@click.version_option(version=openjarvis.__version__, prog_name="jarvis")
def cli() -> None:
    """Top-level CLI group."""


cli.add_command(init, "init")
cli.add_command(ask, "ask")
cli.add_command(serve, "serve")
cli.add_command(model, "model")
cli.add_command(memory, "memory")
cli.add_command(telemetry, "telemetry")
cli.add_command(bench, "bench")
cli.add_command(channel, "channel")
cli.add_command(scheduler, "scheduler")


def main() -> None:
    """Entry point registered as ``jarvis`` console script."""
    cli()


__all__ = ["cli", "main"]
