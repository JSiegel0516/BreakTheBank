# Patch notes: Tracker-bound detection boxes
Task TASK-007 · branch orch/task-007 · 2026-10-29

## For users
Confirmed cards keep their box colour when one frame misreads their name.
Thin grey boxes identify detections without a confirmed card binding.
Adds G to hide or show grey boxes.
Suppresses overlapping detections across card names.
Shows the detector’s actual device in the startup model line.

## For developers
Adds detection-aligned confirmed-card bindings in `cardReader/tracker.py`.
Updates `cardReader/hud.py` to colour boxes from bindings and optionally hide unmatched detections.
Adds the toggle and flash messages in `cardReader/reader.py`.
Updates `read.py` to pass bindings and print the detector device.
Adds class-independent non-maximum suppression and startup warm-up in `cardReader/detector.py`.
Updates `docs/reading.md` with box meanings, controls, and device guidance.
Adds tests for bindings, colouring, toggling, prediction arguments, and device reporting.

## Checks
Independent Ruff validation passes.
Independent detector, tracker, and overlay tests pass: 84 tests.
Reported full-suite validation passes.

## Known gaps
The read-only sandbox prevents an independent full-suite rerun because temporary files are unavailable.
