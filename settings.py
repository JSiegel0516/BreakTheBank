"""Every tunable number and path for the project, in one place.

read.py and train.py take their defaults from here; command-line flags override
them. Paths are relative to the project folder. Both call validate() at start-up,
so an out-of-range value stops them straight away with a message naming every
bad setting and its allowed range.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Data: you fill data/raw, the rest is generated.
LABELED_DIR = "data/raw/labeled"        # Roboflow exports (YOLO format) of table photos, one subfolder each
CARD_ART_DIR = "data/raw/cardArt"       # AS.png ... KC.png, BACK_1.png, indexRect.json (train.py --extract-art)
BACKGROUNDS_DIR = "data/raw/backgrounds"  # photos of the empty table
FRAMES_DIR = "data/raw/frames"          # frames saved from read.py for labelling
DATASET_DIR = "data/dataset"            # the training set train.py builds from everything above
MANIFEST_FILE = "data/manifest.json"    # what data/raw should hold (train.py --manifest / --check-data)

# Training.
SYNTHETIC_IMAGES = 2000    # synthetic images added to the dataset. 0 = real data only.
VAL_FRACTION = 0.15        # share of images held out to measure accuracy
BASE_WEIGHTS = "yolo11n.pt"   # pretrained starting point, downloaded automatically. yolo11s.pt is bigger and slower.
EPOCHS = 100
PATIENCE = 20          # stop early after this many epochs with no validation improvement. best.pt is kept either way.
TRAIN_INPUT_SIZE = 1024
BATCH_SIZE = 8
MODELS_DIR = "models"

# Reading.
MODEL_PATH = "models/cards/weights/best.pt"   # the live model, put there by train.py --promote
LIVE_MODEL_DIR = "models/cards"               # holds LIVE.txt naming the promoted run
CAMERA_SOURCE = "0"        # "0" = first camera. A capture card is often "1". A video path also works.
CAMERA_WIDTH = None        # e.g. 3840. None = whatever the camera gives.
CAMERA_HEIGHT = None       # e.g. 2160
INPUT_SIZE = 1280          # frame is resized to this width before detection. Bigger = reads smaller indices, slower.
CONFIDENCE = 0.6           # detections below this score are ignored
FPS = 12.4                 # frames per second the reader runs at; turns every *_SECONDS below into frames.
                           # read.py prints the measured fps on quit when it is far from this.
CONFIRM_SECONDS = 0.5      # a card must be seen for this long in total ...
CONFIRM_WINDOW_SECONDS = 0.7  # ... within the last this long, so flicker never adds up
CONFIRM_AGREEMENT = 0.8    # share of reads that must agree on the card's rank (0.8 = 80%); suits may differ
CONFIRM_CONFIDENCE = 0.7   # the model's average confidence over those frames must be at least this
STILL_DISTANCE = 0.006     # fraction of frame width the corner may move over those frames (about 12 px at 1080p)
DROP_SECONDS = 1.0         # a card missing this long without a break becomes hidden (kept, shown greyed out)
HIDDEN_SECONDS = 5.0       # a card hidden this long was picked up: removed from the round, still counted.
                           # 0 removes a card as soon as it is hidden.
MATCH_DISTANCE = 0.03      # fraction of frame width; detections closer than this are the same corner
LABEL_WINDOW_SECONDS = 1.5  # a card's name is the vote over its reads in the last this long, so a card
                            # turned face up (a hole card) changes from BACK to its name in about half that
ROUND_END_SECONDS = 3.0    # no card visible for this long ends the round

# Preprocessing: contrast help for dim or washed-out tables (key P in read.py cycles the mode).
PREPROCESS = "auto"        # "off", "gamma" (brighten), "clahe" (local contrast), or "auto" (gamma by brightness).
                           # auto is the benchmarked choice (docs/results.md); clahe lowered accuracy at every level.
PREPROCESS_BRIGHTEN_GAMMA = 0.6  # used by "gamma", and by "auto" in a dim room. Below 1 lifts the shadows.
PREPROCESS_DARKEN_GAMMA = 1.4    # used by "auto" on a washed-out table. Above 1 pulls the highlights down.
PREPROCESS_DARK_BELOW = 110      # "auto": a mean grey level (0 to 255) below this counts as dim ...
PREPROCESS_BRIGHT_ABOVE = 190    # ... and above this as washed out; in between the frame is left alone
PREPROCESS_CLAHE_CLIP = 2.0      # "clahe": how far local contrast may be stretched. Higher = stronger, noisier.
PREPROCESS_CLAHE_TILES = 8       # "clahe": the frame is split into this many tiles across and down

# Table.
PLAYERS = 5                # player seats when nothing is saved yet; change live with keys 1-7 in read.py
SEATS_FILE = "data/seats.json"  # zones edited with the mouse (key E in read.py), one layout per player count


def _is_whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def validate(config) -> None:
    """Check the values in a settings module: this one, or any object with the same names.

    Raises ValueError listing every invalid value with its allowed range."""
    from cardReader.preprocess import MODES  # imported here so settings.py stays cheap to import
    from cardReader.seats import MAX_PLAYERS

    rules = [
        (("INPUT_SIZE", "TRAIN_INPUT_SIZE", "BATCH_SIZE", "EPOCHS", "PATIENCE", "PREPROCESS_CLAHE_TILES"),
         lambda v: _is_whole(v) and v >= 1, "a whole number of 1 or more"),
        (("FPS", "CONFIRM_SECONDS", "CONFIRM_WINDOW_SECONDS", "DROP_SECONDS", "LABEL_WINDOW_SECONDS",
          "ROUND_END_SECONDS"),
         lambda v: _is_number(v) and v > 0, "a number above 0"),
        (("HIDDEN_SECONDS",), lambda v: _is_number(v) and v >= 0, "a number of 0 or more"),
        (("SYNTHETIC_IMAGES",), lambda v: _is_whole(v) and v >= 0, "a whole number of 0 or more"),
        (("CAMERA_WIDTH", "CAMERA_HEIGHT"),
         lambda v: v is None or (_is_whole(v) and v >= 1), "None or a whole number of 1 or more"),
        (("STILL_DISTANCE", "MATCH_DISTANCE"),
         lambda v: _is_number(v) and 0 < v <= 1, "a fraction above 0 and at most 1"),
        (("VAL_FRACTION",), lambda v: _is_number(v) and 0 < v < 1, "a fraction above 0 and below 1"),
        (("CONFIDENCE", "CONFIRM_CONFIDENCE", "CONFIRM_AGREEMENT"),
         lambda v: _is_number(v) and 0 <= v <= 1, "a number from 0 to 1"),
        (("PLAYERS",), lambda v: _is_whole(v) and 1 <= v <= MAX_PLAYERS, f"a whole number from 1 to {MAX_PLAYERS}"),
        (("PREPROCESS",), lambda v: v in MODES, "one of " + ", ".join(repr(mode) for mode in MODES)),
        (("PREPROCESS_BRIGHTEN_GAMMA", "PREPROCESS_DARKEN_GAMMA", "PREPROCESS_CLAHE_CLIP"),
         lambda v: _is_number(v) and v > 0, "a number above 0"),
        (("PREPROCESS_DARK_BELOW", "PREPROCESS_BRIGHT_ABOVE"),
         lambda v: _is_number(v) and 0 <= v <= 255, "a number from 0 to 255"),
    ]
    problems = []
    for names, is_valid, allowed in rules:
        for name in names:
            value = getattr(config, name)
            if not is_valid(value):
                problems.append(f"{name} = {value!r}: must be {allowed}")

    confirm, window = config.CONFIRM_SECONDS, config.CONFIRM_WINDOW_SECONDS
    if _is_number(confirm) and _is_number(window) and confirm > window:
        problems.append(f"CONFIRM_SECONDS = {confirm} must be at most CONFIRM_WINDOW_SECONDS = {window}, "
                        "the time it is counted in")
    dark, bright = config.PREPROCESS_DARK_BELOW, config.PREPROCESS_BRIGHT_ABOVE
    if _is_number(dark) and _is_number(bright) and dark >= bright:
        problems.append(f"PREPROCESS_DARK_BELOW = {dark} must be below PREPROCESS_BRIGHT_ABOVE = {bright}")
    if problems:
        raise ValueError("invalid values in settings.py:\n" + "\n".join(f"  {p}" for p in problems))
