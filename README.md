# Card Reader

A camera looks down at a table. The card reader recognises every card and
reports them in the order they are placed.

```
python train.py     # teach the detector what cards look like, using everything in data/raw
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

## Repository structure

```
Blackjack_Project/
├── train.py                  build the training set from data/raw, then train the detector
├── read.py                   the live reader: camera -> detector -> tracker -> ordered cards
├── settings.py               every tunable number and path, with comments
├── requirements.txt
├── cardReader/               the library
│   ├── camera.py             frames from a camera, capture card, or video file
│   ├── detector.py           YOLO wrapper: frame -> list of detections
│   ├── tracker.py            detections -> cards in placement order
│   ├── deck.py               the 53 class labels (52 faces + BACK)
│   ├── dataset.py            combines data/raw into data/dataset          (used by train.py)
│   ├── synthetic.py          renders card art onto tables as extra data    (used by train.py)
│   └── cardArt.py            crops photos of your deck into card art       (train.py --extract-art)
├── tests/
│   └── test_tracker.py
├── data/
│   ├── raw/                  everything you provide
│   │   ├── labeled/          YOLO-format exports, one subfolder each
│   │   │   ├── playingCards/       353 table photos   (Roboflow)
│   │   │   └── blackjackCorners/   650 table photos   (Roboflow, relabelled to corners)
│   │   ├── cardArt/          AS.png ... KC.png and indexRect.json, used for synthetic images
│   │   ├── backgrounds/      photos of the empty table, used for synthetic images
│   │   └── frames/           frames saved from read.py, to label and add to labeled/
│   └── dataset/              built by train.py, safe to delete
│       ├── data.yaml
│       ├── images/{train,val}/
│       └── labels/{train,val}/
└── models/
    └── cards/                written by train.py
        ├── weights/best.pt   the model read.py uses
        ├── weights/last.pt   the latest epoch, for resuming
        ├── results.csv       one row of scores per epoch
        └── *.png, *.jpg      charts and example predictions
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
pytest
```

Every command below assumes this project's virtual environment is active
(`source .venv/bin/activate`, run from inside `Blackjack_Project`).

If you see `ModuleNotFoundError: No module named 'cv2'`, a different
environment is active, for example one in a parent folder that is also named
`.venv` (the prompt looks the same). Run `deactivate`, activate this one again,
and check that `which python` ends in `Blackjack_Project/.venv/bin/python`.

## Training and validation, step by step

### The ideas in two minutes

- **Training** shows the model thousands of labelled images. After each image
  it nudges its internal numbers so its boxes and names get closer to the labels.
- **An epoch** is one pass over every training image. Training runs for many
  epochs (100 by default), getting a little better each time.
- **Validation** is the exam. About 15% of the images are held back and never
  trained on. After every epoch the model is scored on them. Because it has never
  seen them, the score tells you how it will do on new photos, not how well it
  memorised the old ones.
- **Overfitting** is memorising instead of learning: training scores keep
  improving while validation scores stall or drop. More varied data is the cure.
- **Train and val folders.** `data/dataset/images/train` is studied,
  `data/dataset/images/val` is the exam. `train.py` keeps each export's own split,
  so a photo is never in both.

### 1. Check your data

Every export in `data/raw/labeled/` needs a `data.yaml` and `train/` (and usually
`valid/`) folders, each with `images/` and `labels/`. One `.txt` label file per
image, one line per box: `class_id centre_x centre_y width height`, all as
fractions of the image size.

Boxes go around the **corner index** (the rank and suit printed in the corner)
of each face-up card, named like `10H`, `KS`, `AD`. Face-down cards get one box
around the whole card, named `BACK`. An export whose boxes are whole cards is
skipped automatically.

### 2. Build the dataset and run a one-epoch test

```bash
python train.py --rebuild --epochs 1
```

`--rebuild` combines every export in `data/raw/labeled/` and adds 2000 synthetic
images (`SYNTHETIC_IMAGES` in `settings.py`). Read the first lines it prints:

```
  blackjackCorners: train 498, val 152, skipped (no boxes) 0
  playingCards: train 303, val 49, skipped (no boxes) 1
dataset: train 2502, val 502 -> data/dataset
```

Check that every export you expect is listed with sensible counts. If one says
`skip`, read why. The one-epoch run then proves training works on your machine
before you commit hours to it.

Only use `--rebuild` when `data/raw` has changed. Without it, `train.py` reuses
the existing `data/dataset`.

### 3. Train for real

```bash
python train.py
```

This runs 100 epochs (`EPOCHS` in `settings.py`, or `--epochs 150`). On an
Apple M4 Pro expect a few minutes per epoch, so several hours in total. Leave
the laptop plugged in and stop it from sleeping (`caffeinate -i python train.py`
on macOS).

Each epoch prints a progress bar for training and then a line of validation scores:

```
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100      6.3G      1.088      2.417      0.930        114       1024
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        502       9811      0.612      0.540      0.583      0.402
```

| Column | Meaning | Want it to |
|---|---|---|
| `box_loss` | how far the boxes are from the labels | go down |
| `cls_loss` | how often the card name is wrong | go down |
| `dfl_loss` | fine detail of box edges | go down |
| `P` (precision) | of the boxes it drew, the share that were right | go up |
| `R` (recall) | of the real corners, the share it found | go up |
| `mAP50` | overall accuracy score, 0 to 1, counting a box as right if it overlaps the label by half | go up |
| `mAP50-95` | the same, but stricter about exact box placement | go up |

`mAP50` is the headline number. Roughly: below 0.5 the reader will be unreliable,
0.8 is usable, above 0.9 is good.

If training stops part way (laptop closed, Ctrl-C), resume from the last epoch:

```bash
yolo train resume model=models/cards/weights/last.pt
```

### 4. Read the results

Everything lands in `models/cards/`:

| File | What to look for |
|---|---|
| `weights/best.pt` | the epoch with the best validation score. This is the model `read.py` uses. |
| `results.png` | loss and score curves per epoch. Losses should fall and flatten; scores should rise and flatten. Validation loss rising while training loss falls means overfitting. |
| `results.csv` | the same numbers as a table |
| `confusion_matrix_normalized.png` | rows are the true card, columns the prediction. A bright diagonal is good. Bright squares off the diagonal show which cards it confuses (often 6/9, or suits of the same colour). |
| `val_batch0_labels.jpg` and `val_batch0_pred.jpg` | the same validation images with the true labels and with the model's predictions. The fastest way to see what it gets wrong. |
| `BoxPR_curve.png` | precision against recall per class. Curves hugging the top right are good. |

### 5. Validate a model on its own

To score any trained model on the validation set again, for example after
adding data or to compare two runs:

```bash
yolo val model=models/cards/weights/best.pt data=data/dataset/data.yaml imgsz=1024 project=models name=val
```

It prints the same P, R, and mAP table, with a row for every card, and writes
the charts to `models/val/`. Look for the cards with the lowest mAP50; those
need more examples.

### 6. Try it

```bash
python read.py                          # the default camera
python read.py --source round1.mp4      # a recorded video
```

## Improving the model

Real frames from your own table improve the model more than anything else:

1. Run `python read.py` and press `s` to save frames to `data/raw/frames/`.
2. Upload them to Roboflow and label the corner indices.
3. Export as YOLOv11 (no resize, no augmentation) into a new folder in `data/raw/labeled/`.
4. `python train.py --rebuild`.

Other levers, in `settings.py` or as flags:

- `--epochs 150` if the validation scores are still rising at the end.
- `--base yolo11s.pt` for a bigger model: more accurate, slower.
- `BATCH_SIZE = 4` if training runs out of memory.

### Training on Google Colab

A Colab GPU is several times faster than a laptop. Zip the project without
`.venv`, upload it, then in a notebook:

```
!unzip Blackjack_Project.zip && cd Blackjack_Project && pip install -r requirements.txt
!cd Blackjack_Project && python train.py --rebuild
```

Download `models/cards/weights/best.pt` back into the same place locally.

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

macOS asks for camera permission the first time OpenCV opens a camera from a
terminal or VS Code. Grant it under System Settings > Privacy & Security > Camera.

### Camera notes

- Mount the camera directly overhead so every card is the same size.
- Lock exposure, white balance, and focus so the image does not drift.
- Sony a6700: HDMI capture card, clean HDMI output, manual everything, auto power-off disabled.
- At 4K over a whole table, raise `INPUT_SIZE` in `settings.py` to 1280 or 1600
  so the corner indices stay readable.

### Rules the tracker assumes

- One deck: two sightings of the same face are the same card. This merges a
  card's corner indices and stops a stray misread from adding a duplicate.
- Every face-down card (BACK) is a separate card.
- A card turned face up in place (a dealer's hole card) keeps its BACK label
  for a while, because the label is a vote over every frame it has been seen.

## Glossary

- **Bounding box**: a rectangle around one object in an image. Here, around a card's corner index.
- **Class**: the name attached to a box. This project has 53: 52 cards plus BACK.
- **Label file**: one text line per box, `class_id cx cy w h`, as fractions of the image size. This is the YOLO format.
- **Detector**: a model that outputs boxes and classes for an image. YOLO is the family used here, via the Ultralytics library.
- **Pretrained weights**: a model already trained on everyday photos (`yolo11n.pt`). Training starts from it rather than from scratch, so it needs far less data.
- **Confidence**: the model's score for a box, 0 to 1. Boxes under `CONFIDENCE` in `settings.py` are ignored.
- **Inference**: running the trained model on a new image.
- **Synthetic data**: images rendered by `cardReader/synthetic.py` by pasting card art onto backgrounds, with labels known exactly.
- **Augmentation**: random changes to training images (rotation, brightness, blur) so the model generalises. Ultralytics does this during training; do not also do it in Roboflow.
- **Input size**: the width an image is resized to before the model sees it. Small objects need a larger input size.

## Conventions

- Files and folders are camelCase. Python identifiers are snake_case.
- Source files are plain ASCII.
- Boxes go around the corner index of a face-up card and around the whole card for BACK.
