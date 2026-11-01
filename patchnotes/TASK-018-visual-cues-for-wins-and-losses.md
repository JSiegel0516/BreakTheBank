# Patch notes: Dashboard outcome cues
Task TASK-018 · branch orch/task-018 · 2026-11-01

## For users
Adds gold highlights for player blackjacks, 21s, and wins.
Adds red bust cues, grey losses, and win, push, or lost tags.
Celebrates player blackjacks with confetti and highlights strong dealer hands with an amber pulse.
Respects reduced-motion preferences.
Waits for the dealer to finish before showing final results, including standing on soft 17.

## For developers
Adds pure outcome functions and a dealer standing threshold in `cardReader/blackjack.py`.
Adds an outcome field to every hand in `cardReader/sidebar/snapshot.py`.
Adds outcome styles, tags, and confetti tracking in `cardReader/sidebar/page.html`.
Documents the cues and dealer-finished rule in `docs/reading.md`.
Adds rule coverage in `tests/test_blackjack.py`.
Adds snapshot and page markup coverage in `tests/test_sidebar.py`.

## Checks
Ruff passes independently.
All 23 focused tests pass independently.
JavaScript syntax and confetti transition checks pass independently.
Supplied validation reports the full test suite passes.

## Known gaps
Sandbox restrictions prevent independent full-suite validation and browser rendering checks.
