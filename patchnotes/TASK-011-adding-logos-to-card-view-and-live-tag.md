# Patch notes: Suit shapes and camera rotation
Task TASK-011 · branch orch/task-011 · 2026-10-29

## For users
Card names show ranks and coloured suit shapes across video tags, Game hands, and legend samples.
Face-down cards still read BACK.
Cards Left cells show miniature cards with distinct pending, locked, and duplicate marks.
Small cells retain suit shapes and omit unreadable ranks.
The `<` and `>` keys turn the camera view anticlockwise and clockwise.
CAMERA_ROTATION sets the starting angle to 0, 90, 180, or 270 degrees.
Tracked cards keep their placement numbers through turns.
Saved frames use the turned picture.

## For developers
`cardReader/hud.py` shares card-name drawing and sizing across tags, chips, and legends.
`cardReader/camera.py` adds quarter-turn frame rotation.
`cardReader/reader.py` reloads seat layouts when rotation changes.
`cardReader/tracker.py` rotates stored positions and rescales pixel thresholds.
`settings.py` validates CAMERA_ROTATION.
During zone editing, cached detections previously retained coordinates from before a turn.
`read.py` now forces detection on the first frame after rotation changes.

## Checks
Ruff passes independently.
All 169 independently selected tests pass; 21 temporary-file-dependent tests are deselected.
Supplied validation reports all 411 tests pass.

## Known gaps
Read-only temporary-directory restrictions prevent independently rerunning the full suite.
Saved zones may need adjustment after a quarter turn.
