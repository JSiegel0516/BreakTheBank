# Patch notes: Confidence hysteresis for track survival
Task TASK-029 · branch orch/task-029 · 2026-11-03

## For users
Weak detections keep existing card tracks alive during confidence dips.
They cannot start tracks or contribute to card names and confirmation.
The display hides unmatched weak boxes.
Setting SUSTAIN_CONFIDENCE equal to CONFIDENCE disables weak detections with the default command-line threshold.

## For developers
settings.py adds SUSTAIN_CONFIDENCE with range and threshold validation.
read.py passes the lower detection floor and the selected track-start threshold.
cardReader/tracker.py matches strong detections first and sustains remaining tracks with weak detections.
Track.last_seen controls expiry independently of confident sighting history.
cardReader/hud.py filters unmatched boxes using the start threshold.
docs/reading.md explains tuning and clip comparisons.
docs/results.md adds a pending before-and-after measurement table.

## Checks
Independent focused validation passes all 304 tracker, settings, detector, and display tests.
Ruff passes.
Supplied validation reports full pytest success.
The read-only sandbox prevents independently repeating the full suite because temporary files are unavailable.

## Known gaps
Single-deck and two-deck clip measurements remain pending because no recorded clips are available.
The default sustain threshold's effect on extra, missed, and wrong-name counts remains unmeasured.
