# BreakTheBank

A camera looks down at a blackjack table. The card reader, the component in this
repository, recognises every card and reports them in the order they are placed,
with each seat's hand and the Hi-Lo count.

```
python train.py     # teach the detector what cards look like, using everything in data/raw
python read.py      # read cards off the table, in placement order
```

## Quick start

A trained model is included (`models/cards/weights/best.pt`), so the reader
works straight after setup. Training is only needed to improve it, and needs
the labelled photos from the team's shared folder (see [data/README.md](data/README.md)).

1. Set up the environment (see [Setup](#setup)).
2. Point a camera straight down at the table (see [Setting up the camera](docs/reading.md#setting-up-the-camera)).
3. Run `python read.py` and place cards one at a time. Each gets a number in the order placed.
4. Press `1` to `7` for the number of players. Rounds end on their own when the table is cleared;
   press `Shift+R` after a reshuffle to reset the count, `Q` to quit.

## How it works

```
camera frame
   |
   v
detector      finds every corner index in the frame and names it ("KH", "10S", "BACK")
   |          a YOLO model trained by train.py            cardReader/detector.py
   v
tracker       waits until a card is seen several frames in a row, ignores
   |          misses while a hand covers it, votes on the label, and gives
   |          each new card the next placement number     cardReader/tracker.py
   v
table         which seat each card is in, each seat's hand   cardReader/table.py
   |          and total, the Hi-Lo count                    cardReader/blackjack.py
   v
1:KH  2:7D  3:BACK  4:AS
```

The detector is the only learned part. Everything else is plain rules and is
covered by `pytest`, including the pipeline that builds the training set.

## Repository structure

```
BreakTheBank/                 (the folder git clone creates)
├── train.py                  build the training set from data/raw, then train the detector
├── read.py                   the live reader: camera -> detector -> tracker -> ordered cards
├── settings.py               every tunable number and path, with comments
├── pyproject.toml            dependencies and pytest config; uv.lock pins the exact versions
├── CONTRIBUTING.md           branch, test, PR: how changes get in
├── .github/workflows/        CI: pytest on every push and pull request
├── docs/                     reading.md, training.md, results.md (see Documentation below)
├── cardReader/               the library
│   ├── camera.py             frames from a camera, capture card, or video file
│   ├── detector.py           YOLO wrapper: frame -> list of detections
│   ├── cards.py              the Detection, Card and Event types
│   ├── hud.py                the overlay read.py draws on the live feed
│   ├── seats.py              seat layout by number of players, and the mouse zone editor
│   ├── blackjack.py          hand totals and the Hi-Lo count
│   ├── tracker.py            detections -> cards in placement order (thresholds: TrackerConfig)
│   ├── table.py              cards -> the hand at every seat
│   ├── reader.py             per frame and per key: what read.py does between camera and window
│   ├── deck.py               the 53 class labels (52 faces + BACK)
│   ├── dataset.py            combines data/raw into data/dataset          (used by train.py)
│   ├── synthetic.py          renders card art onto tables as extra data    (used by train.py)
│   ├── cardArt.py            crops photos of your deck into card art       (train.py --extract-art)
│   ├── autolabel.py          fills in corners an export left unlabelled    (train.py --complete-corners)
│   └── provenance.py         data manifest and live-model promotion        (train.py --manifest, --check-data, --promote)
├── tests/
│   ├── test_tracker.py       placement order, hidden cards, rounds, shoes, renaming
│   ├── test_table.py         hands per seat
│   ├── test_reader.py        events, logging, keys
│   ├── test_hud.py           the overlay draws
│   ├── test_dataset.py       combining exports: class remap, splits, skips
│   ├── test_synthetic.py     card art, geometry, composed images
│   ├── test_autolabel.py     corner completion with a faked model
│   ├── test_blackjack.py     hand totals, Hi-Lo, seat layout
│   ├── test_seats.py         zone editing, saving and loading
│   └── test_provenance.py    manifest checks, promotion
├── data/
│   ├── README.md             where the training data comes from and how to check it
│   ├── manifest.json         what data/raw must hold, with fingerprints (train.py --check-data)
│   ├── seats.json            zones edited with the mouse (key E), if any (not in git)
│   ├── raw/                  everything you provide
│   │   ├── labeled/          YOLO-format exports, one subfolder each
│   │   │   ├── playingCards/       353 table photos   (Roboflow)
│   │   │   ├── blackjackCorners/   650 table photos   (Roboflow, relabelled to corners)
│   │   │   └── cardsComplete/      852 composited photos (Roboflow, missing corners filled by the model)
│   │   ├── unused/           exports kept but never trained on
│   │   ├── cardArt/          AS.png ... KC.png and indexRect.json, used for synthetic images
│   │   ├── backgrounds/      photos of the empty table, used for synthetic images
│   │   └── frames/           frames saved from read.py, to label and add to labeled/
│   └── dataset/              built by train.py, safe to delete
│       ├── data.yaml         train + val lists used by training
│       ├── realVal.yaml      the same val set without synthetic images
│       ├── images/{train,val}/
│       └── labels/{train,val}/
└── models/
    ├── cards/                the live model: what read.py uses (put here by train.py --promote)
    │   ├── LIVE.txt          which run this is
    │   ├── weights/best.pt   the epoch with the best validation score (in git)
    │   ├── weights/last.pt   the latest epoch, for resuming
    │   ├── results.csv       one row of scores per epoch
    │   └── *.png, *.jpg      charts and example predictions
    ├── model_1_2026-09-17/   one folder per trained model, numbered and dated. notes.txt,
    ├── model_2_2026-10-13/   args.yaml and results.csv are in git; weights and charts are not
    ├── model_3_2026-10-14_n150/  (live, copied to cards/)
    ├── model_4_2026-10-15_s150/  YOLO11s, tied with model 3, not live
    ├── yolo11n.pt            pretrained starting weights
    └── yolo11s.pt            pretrained starting weights for the larger model
```

## Setup

```bash
uv sync                      # creates .venv with the exact versions in uv.lock
source .venv/bin/activate
pytest
```

[uv](https://docs.astral.sh/uv/) installs Python 3.12 itself if needed. The
dependencies are listed in `pyproject.toml` and pinned in `uv.lock`; `uv sync`
after pulling keeps everyone on the same versions.

Every command in this README and in `docs/` assumes this project's virtual
environment is active (`source .venv/bin/activate`, run from the project folder).

If you see `ModuleNotFoundError: No module named 'cv2'`, a different
environment is active, for example one in a parent folder that is also named
`.venv` (the prompt looks the same). Run `deactivate`, activate this one again,
and check that `which python` ends in `BreakTheBank/.venv/bin/python`.

## Documentation

- [docs/reading.md](docs/reading.md): running the reader, keys, camera setup, seats and the count, tuning, troubleshooting, known limitations
- [docs/training.md](docs/training.md): how training works, the step-by-step guide, improving the model, Colab, glossary
- [docs/results.md](docs/results.md): every trained model's scores and the comparisons between them

Every tunable number is in `settings.py`, with a comment on each.
To change the code, read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## Conventions

- Files and folders are camelCase. Python identifiers are snake_case.
- Source files are plain ASCII.
- Boxes go around the corner index of a face-up card and around the whole card for BACK.
