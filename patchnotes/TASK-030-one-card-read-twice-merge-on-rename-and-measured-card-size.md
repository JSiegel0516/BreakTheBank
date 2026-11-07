# Patch notes: One card read twice is one card
Task TASK-030 · branch main (uncommitted) · 2026-11-06

## For users
Fixes a card counted twice with two or three decks when one of its corners was misread for a moment (a 2S read as 2C, a QH read as JH): once the name corrects itself the second card is merged back into the first, printed as `merged`, and the count drops back.
Fixes the same card showing as a phantom second card in a hand with one deck.
Keeps a card whose two corners read two suits for good as one card with one name, once the reader has measured how far apart a card's corners are.
Stops a card's name flipping between suits when one corner is missed.
Counts a second copy fanned over a card whose far corner is covered, instead of folding it into that card.
Joins the two corners of cards larger than `SAME_CARD_DISTANCE` once a face-down card has shown how big the cards are.
Adds `SAME_CARD_DISTANCE` (was four `MATCH_DISTANCE`s, same default) for a camera close to the table, and `NMS_IOU` for stacked boxes on one corner.
A second box on one corner index is never a card of its own with one deck either.

## For developers
`cardReader/tracker.py`: `_read_twice_of` and `_fold` merge a renamed face card into the card its tracks would have joined under the new name, keeping the lower number, via `merge()`; a revealed hole card is never folded. `_rename` holds a card beside a card of its new name whether or not it was held. `_voted_label` needs `SUIT_RENAME_SHARE` of the reads for a suit-only rename. `corner_span` is measured from cards showing both corners and gates same-name joins to a lone corner (`_takes_corner`) and lets a same-rank, other-suit track join a lone corner (`_card_of_rank`). `card_diagonal`, from BACK boxes, widens `same_card_dist()`. `_card_at_corner` applies to every deck count. `TrackerConfig.same_card_dist` replaces the hard-coded four match distances and turns with the frame.
`cardReader/cards.py`: `Event.into` and the `merged` kind.
`cardReader/reader.py`: a merged event is printed with the count, flashed, and logged as a `merge` row, so `evaluate.py` needs no change.
`cardReader/detector.py`, `read.py`, `benchmark.py`: `iou` from `NMS_IOU`.
`settings.py`: `SAME_CARD_DISTANCE` (at least `MATCH_DISTANCE`) and `NMS_IOU`, validated.
`docs/reading.md`: the merged event, the measured corner span and card size, the new settings and tuning rows.

## Checks
`pytest`: 879 passed, with the pre-existing port-binding test deselected (it needs a free socket).
`ruff check .` passes.
The failure paths and the fixes were driven through the real tracker with the shipped thresholds at 1920 px: a corner misread 2C then 2S, a top corner misread, QH read JH, a persistent 2C corner, an 8S/8C pair fanned, a second 2S fanned over a lone corner, a 180 x 260 px card with and without a face-down box.

## Known gaps
No recorded two-deck clip: the before-and-after `extra` count in `docs/reading.md`, Measuring double counting with two decks, is still to be taken.
A corner persistently misread stays a second card until the corner span is measured, which needs a card to show both corners for about a second first.
Cards bigger than `SAME_CARD_DISTANCE` split in two until a face-down card is seen or the setting is raised; measure a card in a saved frame.
`SPAN_TOLERANCE` (35%) is a module constant: a camera tilted enough that near cards look over a third bigger than far ones would need it raised.
Model-side work (focus training on 2S and QH) is unchanged and still worth doing: the tracker can only merge what the model eventually reads right.
