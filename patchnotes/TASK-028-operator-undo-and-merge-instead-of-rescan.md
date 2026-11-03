# Patch notes: Undo and merge card placements
Task TASK-028 · branch orch/task-028 · 2026-11-03

## For users
Adds U to undo the latest card while preserving other cards and their numbers.
Adds M to merge two placements through window input or browser fields.
Preserves previously counted face-down cards when undoing and detecting them again.
Shows correction messages and records corrections for scoring.

## For developers
Adds correction methods and per-card count bookkeeping in cardReader/tracker.py.
Undo previously loses carried face-down identity; remembered positions now preserve it after tracks are deleted.
Adds correction logging and window number entry in cardReader/reader.py.
Routes browser merge fields directly to the reader in read.py.
Adds browser number fields in cardReader/sidebar/page.html.
Updates evaluate.py to omit corrected placements and close numbering gaps.
Documents correction controls and log behavior in docs/reading.md.

## Checks
Independent tracker and heads-up display tests pass: 234 tests.
Ruff reports no issues.
Supplied validation reports the full test suite passes.

## Known gaps
Independent full-suite execution remains incomplete because the workspace prohibits temporary-file creation and socket binding.
