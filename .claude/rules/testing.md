# Testing Conventions

## Structure
- Tests mirror `src/` structure under `tests/`.
- Run tests: `uv run pytest tests/ -v --tb=short`
- Run a single file: `uv run pytest tests/agents/test_native_react.py -v`
- Run a single test: `uv run pytest tests/agents/test_native_react.py::test_function_name -v`

## Markers
- `live` — needs a running inference engine
- `cloud` — needs API keys (OpenAI, Anthropic, etc.)
- `nvidia` / `amd` / `apple` — GPU-specific tests
- `slow` — long-running tests

Skip markers for local testing: `uv run pytest -m "not live and not cloud" tests/`

## Fixtures
- `conftest.py` provides hardware detection fixtures (`hardware_nvidia`, `hardware_apple`, etc.) and a `mock_engine` factory.
- All registries are auto-cleared between tests via a `conftest.py` fixture — no manual cleanup needed.

## Lint Exceptions
- E501 (line length) is relaxed for `evals/datasets/*.py` and `evals/scorers/*.py`.
