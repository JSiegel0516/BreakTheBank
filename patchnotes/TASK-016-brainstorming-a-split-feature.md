# Patch notes: Split hands and felt-green dashboard
Task TASK-016 · branch orch/task-016 · 2026-11-01

## For users
Adds split and join controls for player pairs in the browser and window.
Shows two numbered hands under the same seat, with separate totals.
Provides separate browser strategy hints without resplitting.
Assigns later cards to the nearest original card.
Clears splits when rounds reset, player counts change, or either original card leaves.
Recolours the browser dashboard with readable felt greens.

## For developers
Adds split records and nearest-card assignment in cardReader/table.py.
Adds guarded split operations and reset handling in cardReader/reader.py.
Adds a strategy flag that treats unsplittable pairs as hard or soft totals in cardReader/strategy.py.
Adds hand parts and split eligibility in cardReader/sidebar/snapshot.py.
Dispatches browser split and join actions in read.py.
Adds numbered split rows in cardReader/hud.py.
Fixes marked-tile text contrast with opaque text backings in cardReader/sidebar/page.html.
Preserves seat identifiers and comma-separated values (CSV) log fields.

## Checks
Ruff passes.
All 177 independently run focused tests pass.
Checked section text contrast ranges from 4.59:1 to 6.01:1.
Marked-tile text contrast measures 18.88:1, 5.34:1, and 6.69:1.
Supplied validation reports the full test suite passing.

## Known gaps
The read-only environment prevents a full-suite rerun because temporary files cannot be created.
