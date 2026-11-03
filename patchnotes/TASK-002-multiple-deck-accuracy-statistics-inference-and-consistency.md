# Patch notes: Consistent counts, play tips, and accuracy tooling
Task TASK-002 · branch orch/task-002 · 2026-11-02

## For users
Shows the same unseen-card count across the dashboard, window, and terminal.
Names unrevealed face-down cards beside that count.
Prevents returning face-down cards from increasing the unrevealed count twice.
Adds play tips whose wording reflects their expected advantage.
Adds count-based bet tips for empty player seats.
Displays count fractions with stacked values.
Adds focus-card confusion tables and documents model 6 training.

## For developers
Adds cached play values and toned tips in cardReader/tips.py.
Extends cardReader/sidebar/snapshot.py with tip fields while preserving hint identifiers.
Tracks returning face-down positions in cardReader/tracker.py.
Adds validated focus settings in settings.py.
Extends cardReader/dataset.py with extra training copies of focus photos and their rotations.
Adds focus-table reporting and filtering in benchmark.py.
Documents the real face-down training-data gap in docs/training.md.

## Checks
Ruff passes independently.
396 selected tests pass independently.
Supplied full-suite validation reports PASS.
A five-player, three-deck snapshot takes 9.09 milliseconds cold and 1.52 milliseconds warm.

## Known gaps
Read-only restrictions prevent three temporary-file fixtures and an independent full-suite run.
Browser layout receives code inspection without independent visual verification.
Model training, new weights, and promotion remain human steps.
