# Blackjack table card recognition

A camera looks down at a blackjack table with a dealer and four players. This
project reads every card on the table, works out which seat each card belongs
to, and keeps a running total for each hand.

If you are new to computer vision, read "How it works" and the glossary below,
then run `scripts/simulateRound.py`. It needs no camera and no trained model.

## How it works

One card's journey through the system:

```
camera frame
   |
   v
YOLO detector        finds every corner index in the frame and names it ("KH", "10S", "BACK")
   |                 cardVision/deck.py lists the 53 names; runLive.py runs the model
   v
zone lookup          the box centre falls inside one seat polygon, e.g. "player3"
   |                 cardVision/zones.py, polygons made by scripts/1_calibrateZones.py
   v
tracker              waits until the same card is seen several frames in a row,
   |                 ignores misses while a hand covers it, votes on the label,
   |                 and enforces "one King of Hearts on the table at a time"
   |                 cardVision/tracker.py
   v
blackjack totals     player3: 21 (soft)      dealer: 10 +1 hidden
                     cardVision/blackjack.py
```

The detector is the only learned part. Everything after it is plain rules.

## Folder map

```
runLive.py              the one script you run at the table
settings.py             every tunable number and path, with comments
cardVision/             the library, one file per stage above
scripts/                tools, numbered in the order you use them
  1_calibrateZones.py   click out the seat polygons
  2_extractCardArt.py   crop and straighten photos of your deck
  3_makeSyntheticData.py  generate training images
  4_importDatasets.py   merge datasets into one training set
  5_train.py            train the detector
  simulateRound.py      demo the tracker with no camera
tests/                  pytest for the tracker and blackjack rules
data/
  raw/                  things YOU put here (see below)
  synthetic/            made by script 3
  merged/               made by script 4
  zones.json            made by script 1
models/                 trained weights, made by script 5
```

What goes in `data/raw/`:

| Folder | Contents |
|---|---|
| `raw/roboflow/` | datasets exported from Roboflow Universe in YOLOv11 format |
| `raw/frames/` | frames saved from `runLive.py` for labelling |
| `raw/cardArt/` | clean card images and `indexRect.json`, made by script 2 |
| `raw/backgrounds/` | photos of the empty table |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

macOS asks for camera permission the first time OpenCV opens a camera from a
terminal or VS Code. Grant it under System Settings > Privacy & Security > Camera.

## Try it with no camera

```bash
pytest                              # tracker and blackjack rules
python scripts/simulateRound.py     # a scripted round through the tracker
```

## Workflow

1. **Zones.** `python scripts/1_calibrateZones.py` and click out the dealer
   and player polygons. Re-run whenever the camera moves.
2. **Card art.** Photograph each card of your deck flat and straight down, one
   card per photo, and label it in Roboflow with a box around the whole card
   (class `cards`), one around the rank, and one around the suit. Export as
   YOLOv11 into `data/raw/roboflow/` and run
   `python scripts/2_extractCardArt.py data/raw/roboflow/<export folder>`.
   It crops and straightens every card into `data/raw/cardArt/` and measures
   where the corner index sits. Photos of card backs need only the card box
   and must be named `BACK_1`, `BACK_2` and so on. Jokers are skipped.
3. **Synthetic data.** `python scripts/3_makeSyntheticData.py --n 3000`.
   Uses your card art if step 2 has run, otherwise placeholder cards. Put a
   few photos of the empty table in `data/raw/backgrounds/` first.
4. **Merge.** `python scripts/4_importDatasets.py "data/raw/roboflow/Playing Cards" data/synthetic`.
   Class ids are remapped by name so datasets with different orderings combine.
   Do not pass the card-art export here; its classes are ranks and suits, not cards.
5. **Train.** `python scripts/5_train.py`. Weights land in `models/cards/weights/best.pt`.
   For long runs, upload `data/merged` to Google Colab and run the same script on a GPU.
6. **Run.** `python runLive.py`. Press `s` to save frames, label them
   (Roboflow, CVAT, or LabelImg, same 53 classes), merge, and retrain. Real
   frames from your own table improve the model more than anything else.

## Camera notes

- Mount the camera directly overhead. Every card is then the same size and
  nearly axis-aligned, and each seat is a fixed region of the frame.
- Lock exposure, white balance, and focus so the image does not drift when
  hands move through the frame.
- Sony a6700: use an HDMI capture card, set clean HDMI output, manual
  everything, and disable auto power-off. Pass the capture card's device
  index as `--source`.
- At 4K over a whole table, raise `INPUT_SIZE` in `settings.py` to 1280 or
  1600 so the corner indices stay readable. Frame tiling is not implemented yet.

## Glossary

- **Bounding box**: a rectangle around one object in an image. Here, around a card's corner index.
- **Class**: the name attached to a box. This project has 53: 52 cards plus BACK.
- **Label file**: one text line per box, `class_id cx cy w h`, all as fractions of the image size. This is the YOLO format.
- **Detector**: a model that outputs boxes and classes for an image. YOLO is the family used here.
- **Confidence**: the model's score for a box, 0 to 1. Boxes under `CONFIDENCE` in `settings.py` are ignored.
- **Inference**: running the trained model on a new image.
- **Epoch**: one pass over the whole training set. Training runs for many.
- **mAP**: mean average precision, the standard accuracy score for detectors. Higher is better, 1.0 is perfect.
- **Augmentation**: random changes to training images (rotation, brightness, blur) so the model generalises.
- **Input size**: the width an image is resized to before the model sees it. Small objects need a larger input size.
- **Track**: one physical card followed across frames by the tracker.

## Conventions

- Files and folders are camelCase. Python identifiers follow the language's snake_case.
- Source files are plain ASCII.
- Boxes go around the corner index of a face-up card and around the whole card for BACK. Every dataset must follow this.
