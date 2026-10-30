# Patch notes: Browser dashboard and page controls
Task TASK-012 · branch orch/task-012 · 2026-10-29

## For users
Adds an optional browser dashboard with live counts, remaining cards, hands, strategy hints, and status.
Adds page buttons for player counts, rounds, shoes, display modes, rotation, reconnecting, and quitting.
Supports running without a video window using `--no-display --web`.
Allows phone access through the documented network binding option.
Preserves the existing window when `--web` is absent.

## For developers
Adds basic strategy tables and a pure hint function in `cardReader/strategy.py`.
Adds snapshot building, Server-Sent Events streaming, action queues, and a static page under `cardReader/sidebar/`.
Adds optional sidebar rendering and one-shot flash retrieval in `cardReader/hud.py`.
Processes queued controls once per frame in `read.py`, preserving rotation dimensions and individual flash messages.
Adds validated dashboard port configuration in `settings.py`.

## Checks
Independent Ruff and 147 focused tests pass.
Recorded validation reports all 481 tests passing.
Recorded Chromium checks cover live updates, card markings, controls, responsive layouts, and headless quit.

## Known gaps
The read-only sandbox prevents an independent full-suite rerun.
Browser checks use a fake camera and scripted detector.
Real camera and physical phone checks remain unverified.
