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
TRAIN_ROTATED_COPIES = (90, 270)  # each real train photo is also added turned clockwise by these degrees (90, 180
                                  # or 270), so corners are seen sideways. 180 adds nothing: cards show both ways up.
FOCUS_CLASSES = ("AD", "2D", "6D", "BACK", "9S", "4C", "4H", "2H")  # the cards most often misread (docs/training.md):
                                  # real train photos holding any of them are added FOCUS_EXTRA_COPIES more times
FOCUS_EXTRA_COPIES = 2     # extra copies of each such photo and of its rotated copies. 0 = none.
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
CAMERA_ROTATION = 0        # degrees the frame is turned clockwise before anything else: 0, 90, 180, or 270.
                           # For a camera mounted sideways or upside down; keys < and > in read.py turn it live.
INPUT_SIZE = 1280          # frame is resized to this width before detection. Bigger = reads smaller indices, slower.
CONFIDENCE = 0.6           # a detection must score at least this to start a track (a new corner)
SUSTAIN_CONFIDENCE = 0.4   # weaker detections down to this only keep an existing track seen: they move it and stop
                           # it missing, but add nothing to its name vote or its confirmation. At most CONFIDENCE;
                           # equal to it turns this off. benchmark.py scores the model at CONFIDENCE.
# Every *_SECONDS below is real time: the tracker times each frame by when it was captured, at any frame rate.
CONFIRM_SECONDS = 0.5      # a card must be seen for this long in total ...
CONFIRM_WINDOW_SECONDS = 0.7  # ... within the last this long, so flicker never adds up
CONFIRM_AGREEMENT = 0.8    # share of reads that must agree on the card's rank (0.8 = 80%); suits may differ
CONFIRM_CONFIDENCE = 0.7   # the model's average confidence over those frames must be at least this
STILL_DISTANCE = 0.006     # fraction of frame width the corner may move over those frames (about 12 px at 1080p)
DROP_SECONDS = 1.0         # a card missing this long without a break becomes hidden (kept, shown greyed out)
HIDDEN_SECONDS = 5.0       # a card hidden this long was picked up: removed from the round, still counted.
                           # 0 removes a card as soon as it is hidden.
MATCH_DISTANCE = 0.03      # fraction of frame width; detections closer than this are the same corner
SAME_CARD_DISTANCE = 0.12  # fraction of frame width; a face corner this close to a card of its name showing one
                           # corner is that card's other corner, farther away a second copy (two or three decks).
                           # Must exceed a card's diagonal in your frames: measure one corner index to the other in a
                           # saved frame (key S). Widened on its own once face-down cards show the cards are bigger.
NMS_IOU = 0.7              # boxes overlapping more than this share of their area are merged into the most confident
                           # (non-maximum suppression). Lower (0.5) merges more stacked boxes on one corner index.
LABEL_WINDOW_SECONDS = 1.5  # a card's name is the vote over its reads in the last this long, so a card
                            # turned face up (a hole card) changes from BACK to its name in about half that
ROUND_END_SECONDS = 3.0    # no card visible for this long ends the round
COPY_HOLD_SECONDS = 1.5    # with DECKS above 1: a second copy placed beside a card of its name is counted only
                           # once both were seen together this long, so one card read twice is not counted twice.
                           # 0 counts it at once.
WEB_PORT = 8765            # port of the browser dashboard (read.py --web); change it if another program uses it

# Preprocessing: contrast help for dim or washed-out tables (key P in read.py cycles the mode).
PREPROCESS = "auto"        # "off", "gamma" (brighten), "clahe" (local contrast), or "auto" (gamma by brightness).
                           # auto is the benchmarked choice (docs/results.md): best or near best in dim light, and
                           # best on average over overlit and hazy pictures. clahe lowered accuracy at every level
                           # except a very hazy picture. A contrast stretch and a highlight curve were worse overall.
PREPROCESS_BRIGHTEN_GAMMA = 0.6  # used by "gamma", and by "auto" in a dim room. Below 1 lifts the shadows.
PREPROCESS_DARKEN_GAMMA = 1.4    # used by "auto" on a washed-out table. Above 1 pulls the highlights down.
PREPROCESS_DARK_BELOW = 110      # "auto": a mean grey level (0 to 255) below this counts as dim ...
PREPROCESS_BRIGHT_ABOVE = 190    # ... and above this as washed out; in between the frame is left alone
PREPROCESS_CLAHE_CLIP = 2.0      # "clahe": how far local contrast may be stretched. Higher = stronger, noisier.
PREPROCESS_CLAHE_TILES = 8       # "clahe": the frame is split into this many tiles across and down

# Table.
PLAYERS = 5                # player seats when nothing is saved yet; change live with keys 1-5 in read.py
SEATS_FILE = "data/seats.json"  # zones edited with the mouse (key E in read.py), one layout per player count
COUNT_SYSTEM = "hi_lo"     # card count read.py starts with: "hi_lo", "hi_opt_ii", or "uston_apc" (Uston APC).
                           # The browser page's dropdown switches it live.
DECKS = 1                  # decks in the shoe, 1 to 3. Each card may be seen once per deck before a duplicate.


def _is_whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def validate(config) -> None:
    """Check the values in a settings module: this one, or any object with the same names.

    Raises ValueError listing every invalid value with its allowed range."""
    from cardReader.blackjack import COUNT_SYSTEMS, MAX_DECKS  # imported here so settings.py stays cheap to import
    from cardReader.dataset import ROTATIONS
    from cardReader.deck import CLASSES
    from cardReader.preprocess import MODES
    from cardReader.seats import MAX_PLAYERS

    rules = [
        (("INPUT_SIZE", "TRAIN_INPUT_SIZE", "BATCH_SIZE", "EPOCHS", "PATIENCE", "PREPROCESS_CLAHE_TILES"),
         lambda v: _is_whole(v) and v >= 1, "a whole number of 1 or more"),
        (("CONFIRM_SECONDS", "CONFIRM_WINDOW_SECONDS", "DROP_SECONDS", "LABEL_WINDOW_SECONDS",
          "ROUND_END_SECONDS"),
         lambda v: _is_number(v) and v > 0, "a number above 0"),
        (("HIDDEN_SECONDS", "COPY_HOLD_SECONDS"), lambda v: _is_number(v) and v >= 0, "a number of 0 or more"),
        (("SYNTHETIC_IMAGES", "FOCUS_EXTRA_COPIES"), lambda v: _is_whole(v) and v >= 0, "a whole number of 0 or more"),
        (("FOCUS_CLASSES",),
         lambda v: isinstance(v, tuple) and all(name in CLASSES for name in v) and len(set(v)) == len(v),
         "a tuple of different card names from deck.CLASSES, such as ('AD', 'BACK'), or () for none"),
        (("CAMERA_WIDTH", "CAMERA_HEIGHT"),
         lambda v: v is None or (_is_whole(v) and v >= 1), "None or a whole number of 1 or more"),
        (("STILL_DISTANCE", "MATCH_DISTANCE", "SAME_CARD_DISTANCE"),
         lambda v: _is_number(v) and 0 < v <= 1, "a fraction above 0 and at most 1"),
        (("VAL_FRACTION",), lambda v: _is_number(v) and 0 < v < 1, "a fraction above 0 and below 1"),
        (("TRAIN_ROTATED_COPIES",),
         lambda v: isinstance(v, tuple) and all(_is_whole(d) and d in ROTATIONS for d in v) and len(set(v)) == len(v),
         "a tuple of different angles from 90, 180, 270, or () for none"),
        (("CONFIDENCE", "SUSTAIN_CONFIDENCE", "CONFIRM_CONFIDENCE", "CONFIRM_AGREEMENT", "NMS_IOU"),
         lambda v: _is_number(v) and 0 <= v <= 1, "a number from 0 to 1"),
        (("PLAYERS",), lambda v: _is_whole(v) and 1 <= v <= MAX_PLAYERS, f"a whole number from 1 to {MAX_PLAYERS}"),
        (("DECKS",), lambda v: _is_whole(v) and 1 <= v <= MAX_DECKS, f"a whole number from 1 to {MAX_DECKS}"),
        (("WEB_PORT",), lambda v: _is_whole(v) and 1 <= v <= 65535, "a whole number from 1 to 65535"),
        (("CAMERA_ROTATION",),lambda v: _is_whole(v) and v in (0, 90, 180, 270), "one of 0, 90, 180, 270"),
        (("PREPROCESS",), lambda v: v in MODES, "one of " + ", ".join(repr(mode) for mode in MODES)),
        (("COUNT_SYSTEM",), lambda v: isinstance(v, str) and v in COUNT_SYSTEMS,
         "one of " + ", ".join(repr(system) for system in COUNT_SYSTEMS)),
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
    sustain, start = config.SUSTAIN_CONFIDENCE, config.CONFIDENCE
    if _is_number(sustain) and _is_number(start) and sustain > start:
        problems.append(f"SUSTAIN_CONFIDENCE = {sustain} must be at most CONFIDENCE = {start}, "
                        "the score that starts a track")
    same_card, match = config.SAME_CARD_DISTANCE, config.MATCH_DISTANCE
    if _is_number(same_card) and _is_number(match) and same_card < match:
        problems.append(f"SAME_CARD_DISTANCE = {same_card} must be at least MATCH_DISTANCE = {match}, "
                        "the distance within which detections are one corner")
    dark, bright = config.PREPROCESS_DARK_BELOW, config.PREPROCESS_BRIGHT_ABOVE
    if _is_number(dark) and _is_number(bright) and dark >= bright:
        problems.append(f"PREPROCESS_DARK_BELOW = {dark} must be below PREPROCESS_BRIGHT_ABOVE = {bright}")
    if problems:
        raise ValueError("invalid values in settings.py:\n" + "\n".join(f"  {p}" for p in problems))
