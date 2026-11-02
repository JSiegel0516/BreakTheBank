# Patch notes: Counting systems and hand odds
Task TASK-019 · branch orch/task-019 · 2026-11-01

## For users
Explains whether the count indicates high or low cards left and whom it favours.
Shows exact unseen tens-and-aces and ace counts.
Adds a dropdown for Hi-Lo, Hi-Opt II, and Uston Advanced Point Count (APC).
Updates count displays and grid shading when the system changes.
Places Hands above Cards Left on narrow screens.
Adds whole-percentage blackjack chances and next-card odds beneath eligible hands.

## For developers
Adds counting-system definitions and pure probability functions in cardReader/blackjack.py.
Adds live recounting through Reader.set_count_system in cardReader/reader.py.
Extends cardReader/sidebar/snapshot.py with composition facts, system choices, rank values, and hand odds.
Updates cardReader/sidebar/page.html with dropdown controls, scaled shading, responsive ordering, and odds rendering.
Updates cardReader/hud.py with system-aware values, shading, titles, and count cues.
Validates the starting system in settings.py.
Handles switch_count actions in read.py.
Documents the controls and probability behavior in docs/reading.md.

## Checks
Independent validation passes 175 relevant tests and Ruff.
Supplied validation reports the full test suite passing.

## Known gaps
Read-only sandbox restrictions prevent independently repeating tests requiring temporary files or local server access.
