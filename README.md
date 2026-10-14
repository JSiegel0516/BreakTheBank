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
│   ├── hud.py                the overlay read.py draws on the live feed
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
│       ├── data.yaml         train + val lists used by training
│       ├── realVal.yaml      the same val set without synthetic images
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

## How the training works

This section explains what happens inside `python train.py`, from the data to
the final model. The numbers are the ones this project actually uses, so they
can go straight into a report.

### The ideas in two minutes

- **Training** shows the model thousands of labelled images. After each small
  group of images it nudges its internal numbers (its *weights*) so its boxes and
  names get closer to the labels.
- **An epoch** is one pass over every training image. Training runs for many
  epochs, getting a little better each time.
- **Validation** is the exam. About 15% of the images are held back and never
  trained on. After every epoch the model is scored on them. Because it has never
  seen them, the score shows how it will do on new photos, not how well it
  memorised the old ones.
- **Overfitting** is memorising instead of learning: the training loss keeps
  falling while the validation score stalls or drops. More varied data is the cure.
- **Train and val folders.** `data/dataset/images/train` is studied,
  `data/dataset/images/val` is the exam. `train.py` keeps each export's own split,
  so a photo is never in both.

### What the model is

The detector is **YOLO11n** ("You Only Look Once", version 11, nano size) from
the Ultralytics library. It is an *object detector*: given an image, it outputs
a list of boxes, each with a class name and a confidence score. It looks at the
whole image in a single pass, which is what makes it fast enough for live video.
The nano size has about 2.6 million weights, the smallest in the YOLO11 family.

Training does not start from random weights. `yolo11n.pt` has already been
trained by Ultralytics on COCO, a public set of about 118,000 everyday photos
with 80 kinds of object (people, cars, cups). It already knows how to find
edges, shapes, and objects in general. Training here only has to teach it the
53 card classes. This is called **transfer learning**, and it is why about 1,000
real photos can be enough instead of hundreds of thousands. When training starts
it prints `Transferred 448/499 items from pretrained weights`: everything is
reused except the final layer, which is rebuilt for 53 classes instead of 80.

### The data it learns from

| Source | Train images | Val images | What it adds |
|---|---|---|---|
| `playingCards` (Roboflow) | 303 | 49 | real photos, one background, cards spread out |
| `blackjackCorners` (Roboflow, relabelled) | 498 | 152 | real photos, many backgrounds and lighting, overlapping hands |
| synthetic (`cardReader/synthetic.py`) | 1,687 | 313 | unlimited extra layouts, exact labels, the only BACK examples |
| **Total** | **2,488** | **514** | |

There are 1,002 real photos in total and about 10,400 labelled corner boxes in
`blackjackCorners` alone.

Note for reporting results: the validation set mixes real and synthetic images.
Synthetic images are easier, so the overall score is somewhat optimistic. The
per-class table at the end of training, and a `yolo val` run on real images
only (step 5 below), give the honest picture. `BACK` in particular appears only in
synthetic images, so its score says little about real card backs yet.

### One training step

Training happens in small groups of images called **batches**:

1. **Load 8 images** (`BATCH_SIZE`) and resize them to 1024 pixels wide
   (`TRAIN_INPUT_SIZE`). Large inputs keep the small corner indices readable.
2. **Augment them**: apply random changes so the model never sees exactly the
   same image twice (table below).
3. **Predict**: the model outputs its boxes and class scores.
4. **Measure the loss**: a number for how wrong the predictions are. It has three parts:
   - `box_loss`: how far the predicted boxes are from the labelled boxes (CIoU loss).
   - `cls_loss`: how wrong the class scores are, e.g. calling a 6H a 9H (binary cross-entropy).
   - `dfl_loss`: how sharp the predicted box edges are (distribution focal loss).
5. **Work out which way to nudge each weight** to make the loss smaller (backpropagation).
6. **Update the weights.** Ultralytics adds up 8 batches (64 images, its "nominal
   batch size") before each update, so the weights change about 39 times per epoch.

Augmentations used:

| Augmentation | Setting | Why |
|---|---|---|
| Mosaic | 4 images stitched into one, turned off for the last 10 epochs | more cards and contexts per image |
| Scale | up to +/-50% | cards at different camera heights |
| Translate | up to +/-10% | cards anywhere in the frame |
| Rotation | up to +/-15 degrees (set in `train.py`) | cards are rarely perfectly straight |
| Colour | hue 0.015, saturation 0.7, brightness 0.4 | different lighting and tables |
| Flips | off (set in `train.py`) | a mirrored corner index is not a real card |

### How fast it learns: the optimizer and learning rate

The **optimizer** is the rule that turns the loss into weight updates. With
`optimizer=auto`, Ultralytics picks **AdamW** for short trainings like this one
(fewer than 10,000 updates) and sets the **learning rate**, the size of each
nudge, to 0.000175 for 53 classes. It prints this when training starts:

```
optimizer: AdamW(lr=0.000175, momentum=0.9) ...
```

The learning rate changes over the run:

- **Warm-up** (first 3 epochs): it rises gradually from near zero, so the
  pretrained weights are not knocked off course by big early updates.
- **Linear decay**: it then shrinks steadily to 1% of the starting value by
  epoch 100. Big steps early to learn quickly, small steps late to fine-tune.

### One epoch

An epoch is 311 batches (2,488 images / 8). At the end of every epoch:

1. **Validate.** The model predicts on all 514 validation images, with no
   augmentation, and is scored (next section).
2. **Save.** `weights/last.pt` is always overwritten with the current weights.
   If this epoch's score is the best so far, it is also saved as `weights/best.pt`.
3. **Log.** One row is added to `results.csv`, and `results.png` is redrawn.

### How a prediction is scored

- **IoU (intersection over union)** measures how well two boxes overlap: the area
  they share divided by the area they cover together. 1.0 is a perfect match, 0 is no overlap.
- A predicted box **counts as correct** if it has the right class and an IoU of at
  least 0.5 with a labelled box. Each labelled box can be matched only once.
- **Precision (P)**: of all the boxes the model drew, the share that were correct.
  Low precision means false alarms.
- **Recall (R)**: of all the labelled boxes, the share the model found. Low recall
  means missed cards.
- **AP (average precision)**, per class: precision and recall trade off as the
  confidence threshold changes. AP summarises the whole trade-off curve as one
  number from 0 to 1.
- **mAP50**: AP averaged over all 53 classes, counting a box as correct at IoU 0.5.
  The headline number.
- **mAP50-95**: the same averaged over stricter IoU thresholds from 0.5 to 0.95,
  so it also rewards placing boxes exactly. Ultralytics uses this one to decide
  which epoch is "best".

### When training stops: epochs and patience

Two settings decide how long training runs:

- **`EPOCHS = 100` is the maximum.** Training never runs more than 100 epochs.
- **`PATIENCE = 20` can end it sooner.** After every epoch, Ultralytics checks
  whether the validation score (`mAP50-95`) beat the best score so far. If 20
  epochs in a row fail to beat it, training stops there, even before epoch 100.

The count restarts every time a new best appears. It is not "run 20 epochs";
it is "give up after 20 epochs in a row with no new best". For example:

| Epochs | What happens |
|---|---|
| 1 to 45 | scores keep improving, a new best is set often |
| 45 | best so far; the count of epochs without improvement resets to 0 |
| 46 to 65 | none of these beat epoch 45; the count climbs to 20 |
| 65 | the count reaches 20, so **training stops**; `best.pt` is epoch 45 |

Without patience, epochs 66 to 100 would also run: about 2.5 more hours that
would very likely not beat epoch 45. If the model keeps improving (a new best at
epoch 90, say), the count never reaches 20 and all 100 epochs run. Patience never
cuts off a run that is still making progress.

Patience does not decide which model you get. `best.pt` is always the epoch with
the best validation score, whether the run stopped early or not, so later epochs
that overfit are never kept. Patience only saves the time those epochs would take.
When it triggers, Ultralytics prints a line like
`Stopping training early as no improvement observed in last 20 epochs. Best results observed at epoch 45`.

### Training settings used (for reports)

| Setting | Value | Where it is set |
|---|---|---|
| Model | YOLO11n, pretrained on COCO, about 2.6 million weights | `BASE_WEIGHTS` in `settings.py` |
| Classes | 53 (52 card faces + BACK) | `cardReader/deck.py` |
| Training input size | 1024 pixels | `TRAIN_INPUT_SIZE` |
| Batch size | 8 images, weights updated every 64 images | `BATCH_SIZE` |
| Maximum epochs | 100 | `EPOCHS` |
| Early stopping | patience 20 epochs on validation mAP50-95 | `PATIENCE` |
| Optimizer | AdamW, learning rate 0.000175, momentum 0.9, weight decay 0.0005 | chosen by Ultralytics (`optimizer=auto`) |
| Learning rate schedule | 3 warm-up epochs, then linear decay to 1% | Ultralytics defaults |
| Augmentation | mosaic, scale, translate, rotation +/-15 degrees, colour; no flips | `train.py` and Ultralytics defaults |
| Data | 2,488 train / 514 val images (1,002 real, 2,000 synthetic) | `data/raw`, `SYNTHETIC_IMAGES` |
| Hardware | Apple M4 Pro GPU (MPS), about 3.6 minutes per epoch | detected by `train.py` |
| Software | Python 3.12, PyTorch 2.14, Ultralytics 8.4 | `requirements.txt` |

Every value Ultralytics actually used is also recorded in `models/cards/args.yaml`.

## Training and validation, step by step

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
dataset: train 2488, val 514 -> data/dataset
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

This runs up to 100 epochs (`EPOCHS` in `settings.py`, or `--epochs 150`). On an
Apple M4 Pro expect a few minutes per epoch, so several hours in total. Leave
the laptop plugged in and stop it from sleeping (`caffeinate -i python train.py`
on macOS).

Each epoch prints a progress bar for training and then a line of validation scores:

```
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100      6.3G      1.088      2.417      0.930        114       1024
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        514       9811      0.612      0.540      0.583      0.402
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

**Early stopping** is on: if the validation score has not improved for 20 epochs
in a row, training stops by itself (see "When training stops" above). Change it
with `PATIENCE` in `settings.py` or `--patience 30`.

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
yolo val model=models/cards/weights/best.pt data=data/dataset/data.yaml imgsz=1024 project="$PWD/models" name=val
```

It prints the same P, R, and mAP table, with a row for every card, and writes
the charts to `models/val/`. Look for the cards with the lowest mAP50; those
need more examples.

To score it on **real photos only** (the 201 real validation images, no
synthetic ones), which is the number to report:

```bash
yolo val model=models/cards/weights/best.pt data=data/dataset/realVal.yaml imgsz=1024 project="$PWD/models" name=valReal
```

`project="$PWD/models"` must be a full path: with a relative one, Ultralytics
writes into `runs/detect/models/` instead. `realVal.yaml` is written by
`train.py --rebuild` next to `data.yaml`. Add
`device=cpu` to run it while a training run is using the GPU.

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

`python read.py` opens a window with the live feed and an overlay on top
(drawn by `cardReader/hud.py`):

| On screen | Meaning |
|---|---|
| Panel at the top left | every card on the table, in the order placed: `1 KH  2 7D  3 BACK` |
| Yellow badge and name above a card | that card's placement number and name |
| Coloured boxes | the corner indices of confirmed cards: red for hearts and diamonds, white for spades and clubs, amber for face-down |
| Thin grey boxes | detections the tracker has not confirmed yet (a card still moving, or a one-frame misread) |
| Bottom bar | the keys, and frames per second |

The number is the placement order: 1 is the first card confirmed (seen for 8
frames in a row), 2 the next, and so on. Numbers are never reused: a removed
card leaves a gap, and a card put back gets a new number.

| Key | Action |
|---|---|
| `R` | reset the count: forget the table, numbering restarts at 1 |
| `S` | save the current frame to `data/raw/frames/` for labelling |
| `Q` | quit (or Ctrl+C in the terminal) |

Keys work when the reader window is selected. The window can be resized.
Every placement and removal is also printed in the terminal:

```
[   412] placed  #1   KH
[   530] placed  #2   7D
```

and on quitting the full sequence is printed: `placed in order: KH 7D`.

- `--source 0` picks the camera. With an iPhone as a Continuity Camera, OpenCV
  usually numbers it 0 and the built-in camera 1; try the other number if the
  wrong one opens.
- `--source round1.mp4` reads a recorded video instead.
- `--log round.csv` appends every event to a CSV.
- `--no-display` runs without the window; stop it with Ctrl+C.

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
