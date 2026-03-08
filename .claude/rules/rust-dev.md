# Rust Development

## Workspace Layout

The Rust implementation lives in `rust/` and mirrors the Python pillars:

```
rust/
  Cargo.toml          # workspace root
  crates/
    core/             # core types, registry
    engine/           # inference engine abstraction
    agents/           # agent implementations
    tools/            # tool implementations
    learning/         # learning orchestrator
    telemetry/        # GPU monitoring, energy measurement
    traces/           # execution trace recording
    security/         # PII scanning, capability policies
    mcp/              # Model Context Protocol
    python/           # PyO3 bindings exposing Rust to Python
```

## Commands

```bash
cd rust

# Lint — treat warnings as errors
cargo clippy --workspace --all-targets -- -D warnings

# Run tests
cargo test --workspace

# Build
cargo build --workspace
```

## PyO3 Bindings

The `python` crate (`rust/crates/python/`) exposes Rust functionality to the Python package via PyO3. When modifying Rust APIs that are bound to Python, update both the Rust implementation and the PyO3 binding layer.
