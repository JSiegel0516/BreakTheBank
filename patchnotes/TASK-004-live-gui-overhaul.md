# Patch notes: Live counter view
Task TASK-004 · branch orch/task-004 · 2026-10-28

## For users
Adds a counter view showing dealt cards by rank and suit.
Distinguishes pending cards, locked cards, and duplicate warnings.
Shows remaining cards per rank and unrevealed face-down cards.
Press V to switch between counter and seats views.
Makes the true count prominent and adds a word cue.
Adds view-specific legends and shortens the bottom key bar.

## For developers
Adds DealtCards and counter rendering to cardReader/hud.py.
Adds shoe_pending, shoe_locked, and shoe_duplicates to cardReader/tracker.py.
Wires view switching through cardReader/reader.py.
Passes shoe state into the overlay from read.py.
Updates documentation and tests for both views.

## Checks
Independent checks pass: 76 overlay and tracker tests, plus Ruff static analysis.
Supplied validation reports the full test suite passing.

## Known gaps
The read-only environment prevents independent full-suite validation because some tests require temporary files.
