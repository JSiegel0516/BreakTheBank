# Patch notes: Reader accuracy and replay evaluation
Task TASK-002 · branch orch/task-002 · 2026-10-28

## For users
Confirms cards when reads agree on rank, even when suits differ.
Warns when a later round repeats a previously counted card name.
Adds replay scoring for placement order, card names, seats, and running count.

## For developers
Updates rank agreement and duplicate detection in `cardReader/tracker.py`.
Adds the duplicate event kind in `cardReader/cards.py`.
Uses existing reader event handling to print and log warnings with seats.
Adds parsing, scoring, reporting, and command-line options in `evaluate.py`.
Documents expected comma-separated values (CSV) files and replay usage in `docs/reading.md`.
Adds regression tests for agreement, duplicates, logging, and replay scoring.

## Checks
Independent Ruff validation passes.
Independent tracker and blackjack tests pass: 56 tests.
Supplied validation reports the full test suite passes.

## Known gaps
The read-only sandbox prevents independently repeating the full test suite.
