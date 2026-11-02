# Patch notes: Deck toggle and per-copy tracking
Task TASK-020 · branch orch/task-020 · 2026-11-01

## For users
Adds a page button that cycles between one, two, and three decks.
Recounts cards already seen when the deck count changes.
Tracks legitimate copies separately and reports excess copies as duplicates.
Shows remaining copies through partial tile shading, rank totals, composition facts, and hand odds.

## For developers
Adds the validated DECKS setting in settings.py.
Adds per-copy bookkeeping and distance-based joining in cardReader/tracker.py.
Adds deck cycling and capped recounting in cardReader/reader.py.
Corrects the stale duplicate-event description to describe copy limits and reference detailed tracker rules.
Adds copy-aware grid data and shoe totals in cardReader/sidebar/snapshot.py.
Adds the toggle and fractional tile shading in cardReader/sidebar/page.html.
Adds partial card fills and shoe denominators in cardReader/hud.py.
Documents deck selection and distance-based limitations in docs/reading.md.

## Checks
Independent tracker and settings validation passes all 133 tests.
Independent Ruff validation passes.
Supplied validation reports the full test suite passing.

## Known gaps
Read-only sandbox restrictions prevent independently repeating the full test suite.
