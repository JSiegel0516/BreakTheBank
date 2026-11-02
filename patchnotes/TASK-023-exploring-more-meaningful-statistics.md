# Patch notes: Dealer finals, standing chances, and insurance
Task TASK-023 · branch orch/task-023 · 2026-11-01

## For users
Adds dealer final-total chances and separates blackjack from other totals of 21.
Adds win, push, and lose chances when a live player hand stands.
Shows TAKE or SKIP for insurance when the dealer shows an ace over a hidden card.
Hides dealer and standing chances when the unseen cards cannot complete every possible dealer draw sequence.

## For developers
Adds exact dealer draw calculations, standing comparisons, and insurance decisions in cardReader/blackjack.py.
Fixes discarded exhausted-shoe probabilities by returning unavailable results for incomplete distributions.
Extends hand fields in cardReader/sidebar/snapshot.py and shares cached dealer calculations across hands.
Adds ordered dealer tiles and a STAND row in cardReader/sidebar/page.html.
Documents the new tiles and rounding behavior in docs/reading.md.

## Checks
Independent checks pass: 89 blackjack tests, two page-render tests, and Ruff.
Supplied validation reports the full test suite passes.

## Known gaps
Independent full-suite execution stops because the read-only sandbox prevents temporary-file creation.
