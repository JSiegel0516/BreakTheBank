# Patch notes: Rescan the current round
Task TASK-013 · branch orch/task-013 · 2026-10-29

## For users
Adds X and a page button to read the current round again after a misread.
Keeps earlier rounds counted while removing this round’s count changes.
Confirms cards again from scratch and restarts their numbering at 1.
Documents that rescan cannot undo completed rounds.

## For developers
Adds round-start count snapshots and rescan() to cardReader/tracker.py.
Adds recounting, feedback, and rescan logging to cardReader/reader.py.
Previously, pressing X immediately after R could erase a completed round from evaluation.
Skips the rescan log row when the open round has no placements, preserving that completed round.
Updates evaluate.py to replace rescanned placements under the same round number.
Adds controls, documentation, and regression coverage.
Combines rotation controls into one legend row to preserve space.

## Checks
Independent tracker and heads-up display tests pass: 132 tests.
Ruff passes.
An independent in-memory Reader-to-evaluator regression preserves both rounds after R followed by X.
Reported full-suite validation passes.

## Known gaps
Read-only temporary-directory restrictions prevent an independent rerun of the full suite.
