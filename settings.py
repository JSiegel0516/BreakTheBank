"""Every tunable number and path for the project, in one place.

Open this file first when something needs adjusting. Each script and the live
runner read their defaults from here; command-line flags override them.
Paths are relative to the project folder.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- camera
CAMERA_SOURCE = "0"        # "0" = first camera. A capture card is often "1". A video path also works.
CAMERA_WIDTH = None        # e.g. 3840. None = whatever the camera gives.
CAMERA_HEIGHT = None       # e.g. 2160

# ---------------------------------------------------------------- detector
MODEL_PATH = "models/cards/weights/best.pt"   # trained weights from scripts/5_train.py
INPUT_SIZE = 1280          # frame is resized to this width before detection. Bigger = reads smaller indices, slower.
CONFIDENCE = 0.4           # detections below this score are ignored

# ---------------------------------------------------------------- tracker
CONFIRM_FRAMES = 8         # a card must be seen this many frames in a row before it counts
DROP_FRAMES = 30           # a card must be missing this many frames in a row before it is forgotten
MATCH_DISTANCE = 0.03      # fraction of frame width; detections closer than this are the same card

# ---------------------------------------------------------------- folders
ZONES_FILE = "data/zones.json"          # seat polygons, made by scripts/1_calibrateZones.py
FRAMES_DIR = "data/raw/frames"          # frames saved from runLive.py for labelling
CARD_ART_DIR = "data/raw/cardArt"       # made by scripts/2_extractCardArt.py: AS.png ... KC.png, BACK_1.png, indexRect.json
BACKGROUNDS_DIR = "data/raw/backgrounds"  # photos of the empty table
SYNTHETIC_DIR = "data/synthetic"        # made by scripts/3_makeSyntheticData.py
MERGED_DIR = "data/merged"              # made by scripts/4_importDatasets.py
MODELS_DIR = "models"                   # training output

# ---------------------------------------------------------------- training
BASE_WEIGHTS = "yolo11n.pt"   # pretrained starting point, downloaded automatically. yolo11s.pt is bigger and slower.
EPOCHS = 40
TRAIN_INPUT_SIZE = 1024
BATCH_SIZE = 8
