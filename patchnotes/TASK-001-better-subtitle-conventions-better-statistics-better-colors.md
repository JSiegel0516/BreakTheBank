# Patch notes: Seat labels, finished-hand odds, and dashboard styling
Task TASK-001 · branch orch/task-001 · 2026-11-02

## For users
Live video labels show custom seat names inside their zones.
Finished hands show an Odds were row with their result probabilities.
The dashboard uses sentence-case labels, bundled Inter text, and darker greens.
The font works offline.

## For developers
Adds stateless post-round probability functions in cardReader/blackjack.py.
Adds documented finished-hand figures to cardReader/sidebar/snapshot.py.
Updates cardReader/hud.py to centre labels beneath each zone's highest edge and fit them inside its outline.
Updates cardReader/sidebar/page.html to render finished-hand statistics and revised styling.
Adds a bundled font route in cardReader/sidebar/server.py.
Includes the font's licence alongside its asset.
Updates docs/reading.md with the new behavior and accurate model-status examples.

## Checks
Independent checks pass 204 blackjack and overlay tests, plus 11 page tests.
Ruff passes.
Supplied validation reports full-suite success.

## Known gaps
Read-only environment restrictions prevent independently repeating tests requiring temporary files or server sockets.
