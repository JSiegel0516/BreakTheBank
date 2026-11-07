# Patch notes: A box on a card's body is the card
Task TASK-032 · branch main (uncommitted) · 2026-11-06

## For users
Fixes a 2 (and other cards with large pips) read as two cards with two or three decks: a box the model draws on the card's pips, reading the card's rank, now joins the card instead of becoming a second copy that is held and then counted.
The rule is kept to the card's own rank, so a card of another rank fanned over it is untouched.
If the pip box was confirmed before the card's second corner and placed as a copy, it is folded back into the card as soon as both corners show, printed as `merged`, whether it was still held or already counted.
A card showing two boxes also takes a third when one of the two turns out to lie between the other and the new one.

## For developers
`cardReader/tracker.py`: `_on_body` (between `MID_CARD_ALONG` of the way from one corner track to the other, within `MID_CARD_ACROSS` of that line), `_card_under` in `_card_for` after the same-corner rule, the third-track branch of `_takes_corner`, `_fold_if_on_card` run every frame before the holds advance, and `_corners_of` (the two in-view face tracks farthest apart). Corner span samples are keyed by card; `_span_excluding` gives the span measured from other cards, which the pip rules use so a card's own polluted samples cannot decide whether its track is a pip, and `_forget_spans` drops a card's samples once a pip is identified.
`docs/reading.md`: the On the card rule, the decks section, the merged event, and the residual layout.
`tests/test_tracker.py`: pip boxes at the centre and on a 2's pips, pip confirmed before the corner with and without the span known, a counted pip card folded, another rank on the body, a same-rank box beside the body line, straight fans, the diagonal half-card residual, one deck, and no span measured from a card carrying a pip. The copy-hold timing test's corners are now diagonal, as corners are.

## Checks
`pytest`: 905 passed, with the pre-existing port-binding test the only failure.
`ruff check .` passes.
Band geometry checked numerically: a 2's pips lie at 0.31 and 0.69 along the diagonal; fans straight to the side (50, 80 px) and straight down (60, 80, 100 px) fall outside the band; diagonal fans fall inside it.

## Known gaps
Not yet run on the clip that showed the 2 of clubs double: the `placed`, `held` and `counted` rows for that round would confirm the cause was a pip box.
Two cards of one rank fanned diagonally by about half a card: the top card's near corner joins the bottom card and the top card is placed from its far corner alone. Before the span is known, a same-rank pair fanned along the diagonal by about two thirds of a card can merge.
