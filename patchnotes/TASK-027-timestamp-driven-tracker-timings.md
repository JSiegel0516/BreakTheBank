# Patch notes: Timestamp-driven tracker timings
Task TASK-027 · branch orch/task-027 · 2026-11-03

## For users
Tracker settings measure seconds regardless of processing speed.
Recorded videos use their playback timestamps for consistent timing.
Removes manual frame-rate tuning.
Pauses tracker timing while the camera reports no signal.

## For developers
Adds optional capture timestamps to CardTracker.update in cardReader/tracker.py.
Stores timing thresholds in seconds while preserving nominal-frame behavior for callers without timestamps.
The first timestamped frame previously credited unmeasured time; it now contributes zero elapsed time.
Retains sightings for the full confirmation and label windows.
Passes timestamps through cardReader/reader.py.
Adds live and recorded-video timestamp handling in read.py.
Removes the FPS setting and its validation from settings.py.
Updates docs/reading.md to describe timing in seconds.

## Checks
Independent tracker and settings validation passes all 183 tests.
Independent Ruff static checks pass.
Supplied validation reports the full test suite passes.

## Known gaps
The full suite is not independently rerun because this read-only environment prevents temporary-file creation.
