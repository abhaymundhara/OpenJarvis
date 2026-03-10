# Workflow Rules

## Task Completion Contract
- A task is not done until tests pass and lint is clean (`uv run ruff check src/ tests/`).
- Never leave stubs, TODOs, or placeholder implementations unless explicitly told to.
- Run the relevant tests before declaring a task complete.

## Context Recovery After Compaction
- After context compaction, re-read your task plan and the relevant source files before continuing.
- Do not rely on memory of file contents from before compaction — re-read them.

## Research vs. Implementation
- If unsure about the right approach, research first: read the relevant code, check tests, understand the pattern.
- Once you've decided on an approach, implement it without second-guessing. Don't explore alternatives mid-implementation.

## Precision
- Make only the changes requested. Don't refactor surrounding code, add comments to unchanged code, or "improve" things beyond scope.
- When fixing a bug, understand the root cause before writing a fix.

## Neutral Investigation
- When asked to find bugs or review code, report findings neutrally.
- Don't bias toward finding problems — if the code is correct, say so.
- Don't inflate minor style issues into bugs.
