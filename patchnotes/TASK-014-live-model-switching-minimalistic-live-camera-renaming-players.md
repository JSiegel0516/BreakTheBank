# Patch notes: Live reader controls
Task TASK-014 · branch orch/task-014 · 2026-11-01

## For users
Adds persistent player names editable from the browser dashboard.
Hides placement numbers on video tags by default; N toggles them.
Adds a model dropdown that switches detection models while keeping tracked cards.
Limits tables to five players.
Bounds the Cards Left grid by its column width.

## For developers
Adds display-name persistence alongside saved layouts in cardReader/seats.py.
Adds structured action bodies and validation in cardReader/sidebar/server.py.
Adds model discovery in cardReader/provenance.py.
Adds synchronous detector switching and refusal messages in read.py.
Fixes retained startup model memory by releasing the local detector reference after ownership transfers.
Adds a weak-reference regression test in tests/test_read.py.
Updates usage documentation and feature tests.

## Checks
Supplied validation reports the test suite passes.
Independent Ruff validation passes.

## Known gaps
Sandbox restrictions prevent an independent test-suite rerun.
Browser layout and interaction checks remain independently unverified.
