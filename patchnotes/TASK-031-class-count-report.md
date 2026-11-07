# Patch notes: Which cards have the fewest examples
Task TASK-031 · branch main (uncommitted) · 2026-11-06

## For users
Adds `python train.py --class-counts`: one row per card with how many real training photos and boxes it has, how many boxes its rotated and focus copies add, how many synthetic boxes, and how many validation boxes, the fewest first.
Marks the cards below the median and ends with a ready `benchmark.py --confusions --focus` line for the cards with the fewest examples, so their counts can be set against their mistakes before choosing `FOCUS_CLASSES`.
`--csv file` also writes the counts to a file.

## For developers
`cardReader/dataset.py`: `ClassCount`, `class_counts` (reads a built dataset's label files back and sorts them by name into real photos, copies, synthetic images, and real val), `_source`, and `class_count_table`.
`train.py`: `--class-counts` and `--csv`, before any build; without a built dataset it says to run `--rebuild`.
`docs/training.md`: a section on running and reading the table. `docs/results.md`: a pending table to fill on the model 5 data. `README.md`: the command.

## Checks
`pytest` passes (the pre-existing port-binding test aside); `ruff check .` passes.
The report is not run on the real data here: the labelled exports are not on this machine.

## Known gaps
The real-data table in `docs/results.md` is pending a `--rebuild` on the exports in `data/manifest.json`.
A real photo whose own file name starts with `r<digits>_` or `f<digits>_` would be counted as a copy.
