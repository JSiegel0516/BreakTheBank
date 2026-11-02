# Patch notes: Multiple deck tracking and readable odds
Task TASK-021 · branch orch/task-021 · 2026-11-01

## For users
Keeps nearby copies of the same card separate with two or three decks.
Preserves single-deck matching behavior.
Displays hand odds as tiles with bold labels above large percentages.
Wraps odds tiles when the page is narrow.

## For developers
Limits same-name joins to cards with fewer than two bound face tracks in cardReader/tracker.py.
Counts bound face tracks until they expire.
Adds tile markup and styling in cardReader/sidebar/page.html.
Adds fanned-copy and nearby hole-card reveal coverage in tests/test_tracker.py.
Updates markup assertions in tests/test_sidebar.py.
Documents the track limit, phantom-copy caveat, and odds tiles in docs/reading.md.

## Checks
Supplied validation reports the full test suite passing.
Independent validation passes 95 tests and Ruff checks.
Sixteen sidebar tests encounter sandbox restrictions on temporary files or network sockets.

## Known gaps
Full-suite execution and browser appearance remain independently unverified in this read-only environment.
