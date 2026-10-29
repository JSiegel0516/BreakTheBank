# Patch notes: Zone editing responsiveness
Task TASK-006 · branch orch/task-006 · 2026-10-29

## For users
Keeps detection running at a reduced rate while editing zones.
Widens handle grab areas and selects nearby handles more accurately.
Moves template zones inward so every player count has visible corners.
Allows edge handles to be grabbed from just outside the video.

## For developers
Adds edit-only detection throttling and time-budgeted mouse event processing in read.py.
Updates template positioning and handle hit testing in cardReader/seats.py.
Adjusts drawn handle sizes in cardReader/hud.py.
Adds detection scheduling checks in tests/test_read.py.
Adds geometry, dragging, edge handling, and persistence checks in tests/test_seats.py.
Updates editing guidance in docs/reading.md.

## Checks
Independent seat and overlay checks pass 171 tests.
A further test encounters a temporary-directory setup error under read-only restrictions.
Reported full-suite validation passes 345 tests.

## Known gaps
Hands-on macOS validation remains unrun, with approval to proceed.
The read-only environment prevents independent validation requiring temporary files.
