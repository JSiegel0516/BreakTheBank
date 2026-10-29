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
CONFIRM_FRAMES = 15        # a card must be seen in this many frames ...
CONFIRM_WINDOW = 20        # ... out of the last this many, so flicker never adds up (about 0.7 s at 30 fps)
CONFIRM_AGREEMENT = 0.8    # share of reads that must agree on the card's name (0.8 = 80%)
CONFIRM_CONFIDENCE = 0.7   # the model's average confidence over those frames must be at least this
STILL_DISTANCE = 0.006     # fraction of frame width the corner may move over those frames (about 12 px at 1080p)
DROP_FRAMES = 30           # a card missing this many frames in a row becomes hidden (kept, shown greyed out)
MATCH_DISTANCE = 0.03      # fraction of frame width; detections closer than this are the same corner
LABEL_WINDOW = 45          # a card's name is the vote over its last this many frames of reads, so a card
                           # turned face up (a hole card) changes from BACK to its name in about half that
ROUND_END_FRAMES = 90      # no card visible for this many frames (about 3 s at 30 fps) ends the round

# Table.
PLAYERS = 5                # player seats when nothing is saved yet; change live with keys 1-7 in read.py
SEATS_FILE = "data/seats.json"  # zones edited with the mouse (key E in read.py), one layout per player count


def _is_whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def validate(config) -> None:
    """Check the numbers in a settings module: this one, or any object with the same names.

    Raises ValueError listing every invalid value with its allowed range."""
    from cardReader.seats import MAX_PLAYERS  # imported here so settings.py stays cheap to import

    rules = [
        (("CONFIRM_FRAMES", "CONFIRM_WINDOW", "DROP_FRAMES", "LABEL_WINDOW", "ROUND_END_FRAMES",
          "INPUT_SIZE", "TRAIN_INPUT_SIZE", "BATCH_SIZE", "EPOCHS", "PATIENCE"),
         lambda v: _is_whole(v) and v >= 1, "a whole number of 1 or more"),
        (("SYNTHETIC_IMAGES",), lambda v: _is_whole(v) and v >= 0, "a whole number of 0 or more"),
        (("CAMERA_WIDTH", "CAMERA_HEIGHT"),
         lambda v: v is None or (_is_whole(v) and v >= 1), "None or a whole number of 1 or more"),
        (("STILL_DISTANCE", "MATCH_DISTANCE"),
         lambda v: _is_number(v) and 0 < v <= 1, "a fraction above 0 and at most 1"),
        (("VAL_FRACTION",), lambda v: _is_number(v) and 0 < v < 1, "a fraction above 0 and below 1"),
        (("CONFIDENCE", "CONFIRM_CONFIDENCE", "CONFIRM_AGREEMENT"),
         lambda v: _is_number(v) and 0 <= v <= 1, "a number from 0 to 1"),
        (("PLAYERS",), lambda v: _is_whole(v) and 1 <= v <= MAX_PLAYERS, f"a whole number from 1 to {MAX_PLAYERS}"),
    ]
    problems = []
    for names, is_valid, allowed in rules:
        for name in names:
            value = getattr(config, name)
            if not is_valid(value):
                problems.append(f"{name} = {value!r}: must be {allowed}")

    frames, window = config.CONFIRM_FRAMES, config.CONFIRM_WINDOW
    if _is_whole(frames) and _is_whole(window) and frames > window:
        problems.append(f"CONFIRM_FRAMES = {frames} must be at most CONFIRM_WINDOW = {window}, "
                        "the frames it is counted in")
    if problems:
        raise ValueError("invalid values in settings.py:\n" + "\n".join(f"  {p}" for p in problems))
