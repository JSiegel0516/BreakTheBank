"""Step 5: fine-tune a YOLO detector on the merged dataset.

    python scripts/5_train.py                      # uses data/merged and settings.py defaults
    python scripts/5_train.py --epochs 100 --base yolo11s.pt

Weights land in models/<name>/weights/best.pt, which is what runLive.py loads.
On a Mac the Apple GPU is used automatically. A Google Colab GPU is a few
times faster for long runs: upload data/merged and run the same command there.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import settings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(settings.MERGED_DIR) / "data.yaml"))
    ap.add_argument("--base", default=settings.BASE_WEIGHTS, help="pretrained weights to start from")
    ap.add_argument("--epochs", type=int, default=settings.EPOCHS)
    ap.add_argument("--imgsz", type=int, default=settings.TRAIN_INPUT_SIZE)
    ap.add_argument("--batch", type=int, default=settings.BATCH_SIZE)
    ap.add_argument("--name", default="cards")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = settings.PROJECT_ROOT / settings.MODELS_DIR
    model = YOLO(str(models_dir / args.base) if (models_dir / args.base).exists() else args.base)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=device, name=args.name, project=str(models_dir), exist_ok=True,
                # Cards are never mirrored, so disable flips; keep rotation and colour jitter.
                fliplr=0.0, flipud=0.0, degrees=15, hsv_v=0.4)
    print("best weights:", model.trainer.best)


if __name__ == "__main__":
    main()
