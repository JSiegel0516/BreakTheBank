"""Every tunable number and path for the project, in one place.

read.py and train.py take their defaults from here; command-line flags override
them. Paths are relative to the project folder.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- data (you fill data/raw, the rest is generated)
LABELED_DIR = "data/raw/labeled"        # Roboflow exports (YOLO format) of table photos, one subfolder each
CARD_ART_DIR = "data/raw/cardArt"       # AS.png ... KC.png, BACK_1.png, indexRect.json (train.py --extract-art)
BACKGROUNDS_DIR = "data/raw/backgrounds"  # photos of the empty table
FRAMES_DIR = "data/raw/frames"          # frames saved from read.py for labelling
DATASET_DIR = "data/dataset"            # the training set train.py builds from everything above

# ---------------------------------------------------------------- training
SYNTHETIC_IMAGES = 2000    # synthetic images added to the dataset. 0 = real data only.
VAL_FRACTION = 0.15        # share of images held out to measure accuracy
BASE_WEIGHTS = "yolo11n.pt"   # pretrained starting point, downloaded automatically. yolo11s.pt is bigger and slower.
EPOCHS = 100
PATIENCE = 20          # stop early after this many epochs with no validation improvement. best.pt is kept either way.
TRAIN_INPUT_SIZE = 1024
BATCH_SIZE = 8
MODELS_DIR = "models"

# ---------------------------------------------------------------- reading
MODEL_PATH = "models/cards/weights/best.pt"   # written by train.py
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
ROUND_END_FRAMES = 90      # no card visible for this many frames (about 3 s at 30 fps) ends the round

# ---------------------------------------------------------------- table
PLAYERS = 5                # player seats when nothing is saved yet; change live with keys 1-7 in read.py
SEATS_FILE = "data/seats.json"  # zones edited with the mouse (key E in read.py), one layout per player count
