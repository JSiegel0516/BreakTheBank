# Card Reader

A camera looks down at a table. The card reader recognises every card and
reports them in the order they are placed.

```
python train.py     # train the detector on everything in data/raw
python read.py      # read cards off the table, in placement order
```

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
1:KH  2:7D  3:BACK  4:AS
```

The detector is the only learned part. The tracker is plain rules and is
covered by `pytest`.

## Folder map

```
train.py            build the training set from data/raw, then train
read.py             the live reader
settings.py         every tunable number and path
cardReader/         the library
  camera.py         frames from a camera, capture card, or video file
  detector.py       YOLO wrapper: frame -> detections
  tracker.py        detections -> ordered list of cards
  deck.py           the 53 class labels (52 faces + BACK)
  dataset.py        combines data/raw into data/dataset       (used by train.py)
  synthetic.py      renders card art onto tables as extra data (used by train.py)
  cardArt.py        crops photos of your deck into card art    (train.py --extract-art)
tests/
data/
  raw/              everything you provide
    labeled/        YOLO-format exports of table photos, one subfolder each (Roboflow: YOLOv11)
    cardArt/        AS.png ... KC.png, BACK_1.png, indexRect.json
    backgrounds/    photos of the empty table
    frames/         frames saved from read.py, to label and add to labeled/
  dataset/          built by train.py, safe to delete
models/cards/weights/best.pt   written by train.py, read by read.py
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
pytest
```

macOS asks for camera permission the first time OpenCV opens a camera from a
terminal or VS Code. Grant it under System Settings > Privacy & Security > Camera.

## Training

1. Put labeled exports in `data/raw/labeled/<name>/` (each with its `data.yaml`).
   Class ids are matched by name, so exports with a different class order are fine.
2. Optional, for better synthetic data: photograph each card of your deck flat,
   one per photo, label the whole card (`cards`), the rank, and the suit in
   Roboflow, export, and run `python train.py --extract-art <export folder>`.
   Photos of card backs need only the card box and must be named `BACK_1`, `BACK_2` ...
3. `python train.py`. The first run builds `data/dataset` (your exports plus
   `SYNTHETIC_IMAGES` rendered images). After adding data, use `--rebuild`.

For long runs, a Google Colab GPU is several times faster than a Mac: upload
the project and run the same command.

## Reading

`python read.py` shows the feed with each card numbered in placement order,
and prints every placement and removal:

```
[   412] placed  #1   KH
[   530] placed  #2   7D
```

- `--source 1` for a capture card, or a video file path.
- `--log round.csv` appends every event to a CSV.
- Keys: `q` quit, `s` save the frame to `data/raw/frames`, `r` new round (numbering restarts).

Real frames from your own table improve the model more than anything else:
save them with `s`, label them, export into `data/raw/labeled/`, and retrain
with `--rebuild`.

## Camera notes

- Mount the camera directly overhead so every card is the same size.
- Lock exposure, white balance, and focus so the image does not drift.
- Sony a6700: HDMI capture card, clean HDMI output, manual everything, auto power-off disabled.
- At 4K over a whole table, raise `INPUT_SIZE` in `settings.py` to 1280 or 1600
  so the corner indices stay readable.

## Rules the tracker assumes

- One deck: two sightings of the same face are the same card. This merges a
  card's two corner indices and stops a stray misread from adding a duplicate.
- Every face-down card (BACK) is a separate card.
- A card turned face up in place (a dealer's hole card) keeps its BACK label
  for a while, because the label is a vote over every frame it has been seen.

## Conventions

- Files and folders are camelCase. Python identifiers are snake_case.
- Source files are plain ASCII.
- Boxes go around the corner index of a face-up card and around the whole card for BACK.
