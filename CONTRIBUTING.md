# Contributing

How we work on BreakTheBank. The README has the quick start and the map of
the repository; this page is the workflow.

## Set up once

```bash
git clone https://github.com/JSiegel0516/BreakTheBank.git
cd BreakTheBank
uv sync                      # .venv with the exact versions in uv.lock
source .venv/bin/activate
pytest                       # should pass before you change anything
```

Run `uv sync` again whenever `uv.lock` changes after a pull. To train you
also need the labelled photos: see [data/README.md](data/README.md).

## Making a change

1. **Branch from main**, named for the change: `git checkout -b hud-split-hands`.
2. **Write or update tests** in `tests/`. The tracker, table, seats, reader,
   HUD, blackjack rules, dataset builder, synthetic generator, autolabel and
   provenance all have test files; add cases next to the ones they resemble.
   Tests run in under a second and need no camera or model.
3. **Run `pytest`** until it is green.
4. **Try it for real** if you touched the reader: `python read.py` with a
   camera, or `python read.py --source clip.mp4 --no-display --log events.csv`
   with a short video recorded on a phone. Check the terminal events and the CSV.
5. **Commit** with a subject line that says what changed, in the imperative,
   and a body that says why. Look at `git log` for the house style.
6. **Push and open a pull request** against `main`. Describe what changed,
   why, and what you checked. CI runs `pytest` on every push; a PR is merged
   only when it is green, by someone other than its author when possible.
7. After the merge, delete the branch.

One change per PR. A refactor and a feature go in separate PRs, so each can
be reviewed and reverted on its own.

## Where things go

| Change | Where |
|---|---|
| a tunable number | `settings.py`, with a comment saying what it does and which way to turn it |
| what happens per frame or per key | `card_reader/reader.py` |
| how cards are grouped into hands | `card_reader/table.py` |
| how detections become cards | `card_reader/tracker.py` (read its docstring first) |
| how the overlay looks | `card_reader/hud.py`; it only draws, it does not decide |
| blackjack rules | `card_reader/blackjack.py` |
| a new data export | `data/raw/labeled/<name>/`, then `python train.py --manifest` and commit the manifest |
| a new trained model | `python train.py --name model_N_<date>_<tag>`; write its `notes.txt`; `--promote` the winner |
| how something works, for users | `docs/reading.md` or `docs/training.md` |
| scores and comparisons | `docs/results.md` and the run's `notes.txt` |

## Conventions

- Files and folders are camelCase; Python identifiers are snake_case. Source
  files are plain ASCII.
- Every module starts with a docstring that explains what it does and why it
  works the way it does. Comments say why, not what.
- Boxes go around the corner index of a face-up card and around the whole
  card for `BACK`.
- Distances in `settings.py` are fractions of the frame width, times in
  frames. `TrackerConfig.from_settings` converts them.
- No new dependency without a line in the PR saying why. Add it with
  `uv add <package>` so `pyproject.toml` and `uv.lock` change together.

## Never commit

- Model weights (`*.pt`) other than the live `models/cards/weights/best.pt`,
  which `--promote` writes.
- Photos, exports, or `data.zip`. They live in the shared folder.
- `data/seats.json`: it is one person's camera setup.
- Absolute paths in docs or code. `settings.PROJECT_ROOT` is the anchor.

## Training runs

`python train.py` writes to `models/<name>/`. Every run's `notes.txt`,
`args.yaml` and `results.csv` are tracked in git; weights and charts are not.
Compare runs on the same real-photo validation set before promoting one
(see [docs/training.md](docs/training.md#comparing-models)), and record the
comparison in `docs/results.md`.
