# Patch notes: Hidden-card expiry and sidebar views
Task TASK-010 · branch orch/task-010 · 2026-10-29

## For users
Covered cards leave their hands after five seconds hidden while remaining counted in the shoe.
Returning cards keep their numbers before expiry; later placements receive fresh numbers.
Adds highlighted CARDS LEFT and GAME tabs with a V key hint.
Renames the views to Cards Left and Game.
Sets timing thresholds in seconds using one frames per second (FPS) setting.
Reports measured speed on quit when it differs from the setting by more than twenty percent.

## For developers
Adds hidden timestamps and removed events in `cardReader/cards.py`.
Adds expiry, count preservation, and placement numbering in `cardReader/tracker.py`.
Converts seconds to frame thresholds through `TrackerConfig.from_settings`.
Validates timing settings in `settings.py`.
Adds sidebar tabs and a pure tab-label helper in `cardReader/hud.py`.
Updates reader controls, documentation, and tests for the renamed views.
Preserves evaluation behavior by ignoring removed events.

## Checks
Ruff passes independently.
All 146 independently selected tracker, display, settings, and count tests pass.
Supplied validation reports the full test suite passing.

## Known gaps
Read-only temporary-file restrictions prevent independently repeating the full suite.
Expired face-down cards returning later can reduce cards left twice, as documented.
