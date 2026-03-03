"""``jarvis serve`` — OpenAI-compatible API server."""

from __future__ import annotations

import sys

import click
from rich.console import Console

from openjarvis.core.config import load_config
from openjarvis.core.events import EventBus
from openjarvis.engine import (
    discover_engines,
    discover_models,
    get_engine,
)
from openjarvis.intelligence import (
    merge_discovered_models,
    register_builtin_models,
)


@click.command()
@click.option("--host", default=None, help="Bind address (default: config).")
@click.option(
    "--port", default=None, type=int,
    help="Port number (default: config).",
)
@click.option("-e", "--engine", "engine_key", default=None, help="Engine backend.")
@click.option("-m", "--model", "model_name", default=None, help="Default model.")
@click.option(
    "-a", "--agent", "agent_name", default=None,
    help="Agent for non-streaming requests (simple, orchestrator, react, openhands).",
)
def serve(
    host: str | None,
    port: int | None,
    engine_key: str | None,
    model_name: str | None,
    agent_name: str | None,
) -> None:
    """Start the OpenAI-compatible API server."""
    console = Console(stderr=True)

    # Check for server dependencies
    try:
        import uvicorn  # noqa: F401
        from fastapi import FastAPI  # noqa: F401
    except ImportError:
        console.print(
            "[red bold]Server dependencies not installed.[/red bold]\n\n"
            "Install the server extra:\n"
            "  [cyan]uv pip install 'openjarvis[server]'[/cyan]\n"
            "  or\n"
            "  [cyan]pip install 'openjarvis[server]'[/cyan]"
        )
        sys.exit(1)

    config = load_config()

    # Resolve host/port from CLI args or config
    bind_host = host or config.server.host
    bind_port = port or config.server.port

    # Set up engine
    register_builtin_models()
    bus = EventBus(record_history=False)

    # Set up telemetry
    telem_store = None
    if config.telemetry.enabled:
        try:
            from openjarvis.telemetry.store import TelemetryStore

            telem_store = TelemetryStore(config.telemetry.db_path)
            telem_store.subscribe_to_bus(bus)
        except Exception:
            pass  # telemetry is best-effort

    resolved = get_engine(config, engine_key)
    if resolved is None:
        console.print(
            "[red bold]No inference engine available.[/red bold]\n\n"
            "Make sure an engine is running."
        )
        sys.exit(1)

    engine_name, engine = resolved

    # Discover models
    all_engines = discover_engines(config)
    all_models = discover_models(all_engines)
    for ek, model_ids in all_models.items():
        merge_discovered_models(ek, model_ids)

    # Resolve model
    if model_name is None:
        model_name = config.server.model or config.intelligence.default_model
    if not model_name:
        engine_models = all_models.get(engine_name, [])
        if engine_models:
            model_name = engine_models[0]
        else:
            console.print("[red]No model available on engine.[/red]")
            sys.exit(1)

    # Resolve agent
    agent = None
    agent_key = agent_name or config.server.agent
    if agent_key:
        try:
            import openjarvis.agents  # noqa: F401
            from openjarvis.core.registry import AgentRegistry

            if AgentRegistry.contains(agent_key):
                agent_cls = AgentRegistry.get(agent_key)
                agent_kwargs = {"bus": bus}

                # Load tools for agents that support them
                if getattr(agent_cls, "accepts_tools", False):
                    import openjarvis.tools  # noqa: F401  # trigger registration
                    from openjarvis.core.registry import ToolRegistry
                    from openjarvis.tools._stubs import BaseTool

                    tools = []
                    for name in ToolRegistry.keys():
                        tool_cls = ToolRegistry.get(name)
                        if isinstance(tool_cls, type) and issubclass(tool_cls, BaseTool):
                            tools.append(tool_cls())
                        elif isinstance(tool_cls, BaseTool):
                            tools.append(tool_cls)
                    if tools:
                        agent_kwargs["tools"] = tools

                if getattr(agent_cls, "accepts_tools", False):
                    agent_kwargs["max_turns"] = config.agent.max_turns

                agent = agent_cls(engine, model_name, **agent_kwargs)
        except Exception as exc:
            import traceback
            console.print(f"[yellow]Agent '{agent_key}' failed to load: {exc}[/yellow]")
            traceback.print_exc()

    # Set up channel backend if enabled
    channel_bridge = None
    if config.channel.enabled and config.channel.default_channel:
        try:
            from openjarvis.system import SystemBuilder

            # Reuse _resolve_channel logic from SystemBuilder
            sb = SystemBuilder(config)
            sb._bus = bus
            channel_bridge = sb._resolve_channel(config, bus)
            if channel_bridge is not None:
                channel_bridge.connect()
                console.print(
                    f"  Channel: [cyan]{config.channel.default_channel}[/cyan]"
                )
        except Exception as exc:
            console.print(f"[yellow]Channel failed to start: {exc}[/yellow]")
            channel_bridge = None

    # Set up speech backend
    speech_backend = None
    try:
        from openjarvis.speech._discovery import get_speech_backend
        speech_backend = get_speech_backend(config)
        if speech_backend:
            console.print(f"  Speech: [cyan]{speech_backend.backend_id}[/cyan]")
    except Exception:
        pass

    # Create app
    from openjarvis.server.app import create_app

    app = create_app(
        engine, model_name, agent=agent, bus=bus,
        engine_name=engine_name, agent_name=agent_key or "",
        channel_bridge=channel_bridge, config=config,
        speech_backend=speech_backend,
    )

    console.print(
        f"[green]Starting OpenJarvis API server[/green]\n"
        f"  Engine: [cyan]{engine_name}[/cyan]\n"
        f"  Model:  [cyan]{model_name}[/cyan]\n"
        f"  Agent:  [cyan]{agent_key or 'none'}[/cyan]\n"
        f"  URL:    [cyan]http://{bind_host}:{bind_port}[/cyan]"
    )

    import uvicorn
    uvicorn.run(app, host=bind_host, port=bind_port, log_level="info")
