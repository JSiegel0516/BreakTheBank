# Patch notes: Reader standards and count corrections
Task TASK-001 · branch orch/task-001 · 2026-10-28

## For users
Invalid settings stop startup with messages naming each problem and its allowed range.
Revealing a hole card preserves its placement number without adding a phantom card.
Renaming cards preserves labels still held by other cards.
Cards left excludes unrevealed face-down cards.
The true count uses a minimum divisor of one quarter deck.

## For developers
Adds Ruff configuration and a development dependency in `pyproject.toml`.
Adds lint checks before tests in `.github/workflows/tests.yml`.
Adds aggregated numeric validation in `settings.py`.
Updates `cardReader/tracker.py` to associate reveals and preserve count ownership.
Tracks already-dealt face-down cards across manual round resets.
Extends `cardReader/blackjack.py` with an optional unrevealed-card count and a named divisor floor.
Updates `cardReader/reader.py` to include unrevealed cards when recounting.
Removes decorative dividers, clarifies local names, and updates affected docstrings.
Corrects the reveal-association limitation in `docs/reading.md` and restores the separate suit-confusion bullet.

## Checks
Independent Ruff validation passes.
Independent testing reports 83 passing tests and 14 temporary-directory setup errors caused by sandbox restrictions.
Supplied validation reports all 97 tests passing.

## Known gaps
The read-only environment prevents independent completion of tests requiring temporary files.
Locked dependency synchronization remains independently unverified.
