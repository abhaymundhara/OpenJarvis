"""Recipe loader — load and resolve TOML recipe files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


# Built-in recipes directory (package data)
_PROJECT_RECIPES_DIR = Path(__file__).resolve().parent / "data"
# User-level recipes directory
_USER_RECIPES_DIR = Path.home() / ".openjarvis" / "recipes"


@dataclass(slots=True)
class Recipe:
    """A composable pillar configuration loaded from TOML."""

    name: str
    description: str = ""
    version: str = "1.0.0"

    # Intelligence
    model: Optional[str] = None
    quantization: Optional[str] = None

    # Engine
    engine_key: Optional[str] = None

    # Agent
    agent_type: Optional[str] = None
    max_turns: Optional[int] = None
    temperature: Optional[float] = None
    tools: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None

    # Learning
    routing_policy: Optional[str] = None
    agent_policy: Optional[str] = None

    # Eval
    eval_suites: List[str] = field(default_factory=list)

    # Raw TOML data for forward-compat
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_builder_kwargs(self) -> Dict[str, Any]:
        """Convert recipe fields to kwargs for SystemBuilder/Jarvis.

        Returns a dict with only the non-None fields, keyed to match
        the SystemBuilder fluent API or Jarvis constructor parameters.
        """
        kwargs: Dict[str, Any] = {}
        if self.model is not None:
            kwargs["model"] = self.model
        if self.engine_key is not None:
            kwargs["engine_key"] = self.engine_key
        if self.agent_type is not None:
            kwargs["agent"] = self.agent_type
        if self.tools:
            kwargs["tools"] = self.tools
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_turns is not None:
            kwargs["max_turns"] = self.max_turns
        if self.system_prompt is not None:
            kwargs["system_prompt"] = self.system_prompt
        if self.routing_policy is not None:
            kwargs["routing_policy"] = self.routing_policy
        if self.agent_policy is not None:
            kwargs["agent_policy"] = self.agent_policy
        if self.quantization is not None:
            kwargs["quantization"] = self.quantization
        if self.eval_suites:
            kwargs["eval_suites"] = self.eval_suites
        return kwargs


def load_recipe(path: str | Path) -> Recipe:
    """Load a recipe from a TOML file.

    Expected TOML format::

        [recipe]
        name = "coding_assistant"
        description = "..."
        version = "1.0.0"

        [intelligence]
        model = "qwen3:8b"
        quantization = "q4_K_M"

        [engine]
        key = "ollama"

        [agent]
        type = "native_react"
        max_turns = 10
        temperature = 0.3
        tools = ["file_read", "file_write", "code_interpreter", "think"]
        system_prompt = "You are a coding assistant..."

        [learning]
        routing = "grpo"
        agent = "icl_updater"

        [eval]
        suites = ["coding", "reasoning"]

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Recipe file not found: {path}")

    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    recipe_sec = data.get("recipe", {})
    intel_sec = data.get("intelligence", {})
    engine_sec = data.get("engine", {})
    agent_sec = data.get("agent", {})
    learning_sec = data.get("learning", {})
    eval_sec = data.get("eval", {})

    return Recipe(
        name=recipe_sec.get("name", path.stem),
        description=recipe_sec.get("description", ""),
        version=recipe_sec.get("version", "1.0.0"),
        model=intel_sec.get("model"),
        quantization=intel_sec.get("quantization"),
        engine_key=engine_sec.get("key"),
        agent_type=agent_sec.get("type"),
        max_turns=agent_sec.get("max_turns"),
        temperature=agent_sec.get("temperature"),
        tools=agent_sec.get("tools", []),
        system_prompt=agent_sec.get("system_prompt"),
        routing_policy=learning_sec.get("routing"),
        agent_policy=learning_sec.get("agent"),
        eval_suites=eval_sec.get("suites", []),
        raw=data,
    )


def discover_recipes(
    extra_dirs: Optional[List[str | Path]] = None,
) -> List[Recipe]:
    """Discover all TOML recipes from known directories.

    Search order (later entries override earlier ones by name):
    1. Project ``recipes/`` directory
    2. User ``~/.openjarvis/recipes/`` directory
    3. Any additional directories in *extra_dirs*
    """
    dirs: List[Path] = [_PROJECT_RECIPES_DIR, _USER_RECIPES_DIR]
    if extra_dirs:
        dirs.extend(Path(d) for d in extra_dirs)

    recipes: Dict[str, Recipe] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for toml_path in sorted(d.glob("*.toml")):
            try:
                recipe = load_recipe(toml_path)
                recipes[recipe.name] = recipe
            except Exception:
                # Skip malformed recipe files
                continue

    return list(recipes.values())


def resolve_recipe(name: str) -> Optional[Recipe]:
    """Find a recipe by name from all known directories.

    Returns ``None`` if no recipe with the given name is found.
    """
    for recipe in discover_recipes():
        if recipe.name == name:
            return recipe
    return None


__all__ = ["Recipe", "discover_recipes", "load_recipe", "resolve_recipe"]
