# Patch notes: Python module naming
Task TASK-003 · branch orch/task-003 · 2026-10-28

## For users
Preserves existing reading and training behavior.

## For developers
Renames `cardReader/` to `card_reader/` and updates imports, tests, and current documentation.
Renames the card art module to `card_art.py`.
Removes temporary N999 naming exceptions from `pyproject.toml` without changing dependencies.
Preserves data paths and serialized filenames.
Git recognizes all 16 library files as renames.

## Checks
Independent Ruff validation passes with caching disabled.
Both reading and training help commands pass.
Supplied validation reports pytest passing.

## Known gaps
Sandbox restrictions prevent independent pytest execution and checks through uv.
