# Patch notes: Frame preprocessing, timings, and camera recovery
Task TASK-005 · branch orch/task-005 · 2026-10-29

## For users
Adds live detection and preprocessing timings, with average stage timings on quit.
Adds optional contrast enhancement, with automatic brightness adjustment enabled by default.
Adds P to cycle enhancement modes and Shift+V to compare enhanced and raw frames.
Keeps saved frames raw.
Shows a no-signal message when camera frames stop and adds C to reconnect.
Adds an offline benchmark comparing detection accuracy and processing time across brightness levels.
Documents benchmark results and the limits of correcting glare.

## For developers
Adds cached gamma lookup tables and lightness-channel contrast enhancement in cardReader/preprocess.py.
Adds staleness tracking and same-source reopening in cardReader/camera.py.
Adds stage timing and camera recovery handling in read.py.
Adds preprocessing controls in cardReader/reader.py.
Adds timing, mode, and signal indicators in cardReader/hud.py.
Adds preprocessing configuration and validation in settings.py.
Adds brightness comparisons and corner-matching scores in benchmark.py.
Records benchmark results and the recommended default in docs/results.md.

## Checks
Independent validation passes 83 targeted tests and Ruff.
Supplied validation reports the full test suite passes.

## Known gaps
Read-only sandbox restrictions prevent independently repeating the full suite and live-model benchmark.
Physical camera interruption and recovery remain unverified during this review.
