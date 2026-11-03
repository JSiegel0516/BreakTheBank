# Patch notes: Fewer double counts with two or three decks
Task TASK-026 · branch orch/task-026 · 2026-11-03

## For users
Adds matching rules that keep respawned corners, corner misreads, and moved cards attached to their original card.
Holds nearby copies outside the count until both cards appear together for the configured duration.
Shows a magenta question mark on held cards and reports held and counted events.
Preserves the count when a held copy replaces its removed counterpart.
Documents recording and scoring two-deck clips.

## For developers
Adds current-frame track matching and suspect-copy handling in `cardReader/tracker.py`.
Fixes unrelated count handovers that previously counted held cards prematurely.
Uses `_copy_taker` to prioritize counterpart handovers and exclude other held cards.
Adds hold state to `cardReader/cards.py`.
Adds validated `COPY_HOLD_SECONDS` configuration in `settings.py`.
Adds the held marker and legend entry in `cardReader/hud.py`.
Prints running counts for counted events in `cardReader/reader.py`.
Confirms `evaluate.py` accepts both new event kinds without changing placement scoring.

## Checks
Independent tracker, display, and settings checks pass: 271 tests.
Ruff reports no problems.
Supplied full-suite validation reports a pass.

## Known gaps
The read-only sandbox prevents independently repeating tests requiring temporary files.
No recorded two-deck clip validation is supplied.
A second copy confirmed while its counterpart is covered can merge into that counterpart.
