"""Train the card detector on everything in data/raw.

    python train.py                        # build data/dataset if missing, then train
    python train.py --rebuild              # rebuild data/dataset first (after adding data)
    python train.py --epochs 150 --base yolo11s.pt --name s150   # an experiment, kept in models/s150/
    python train.py --extract-art data/raw/myDeckExport   # one-off: photos of your deck -> data/raw/cardArt
    python train.py --complete-corners data/raw/unused/cards  # add unlabelled corners using the current model
    python train.py --promote models/n150  # make a finished run the live model read.py loads
    python train.py --check-data           # does data/raw match data/manifest.json (the shared folder)?
    python train.py --manifest             # rewrite data/manifest.json after adding or relabelling data

Every run goes to models/<name>/ (--name, default cards). read.py loads
models/cards/weights/best.pt, so train with --name and --promote the winner.
On a Mac the Apple GPU is used automatically. A Google Colab GPU is a few
times faster for long runs: upload the project and run the same command there.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import settings
from card_reader import autolabel, card_art, dataset, provenance

ROOT = settings.PROJECT_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild data/dataset from data/raw before training")
    ap.add_argument("--synthetic", type=int, default=settings.SYNTHETIC_IMAGES, help="synthetic images to generate")
    ap.add_argument("--base", default=settings.BASE_WEIGHTS, help="pretrained weights to start from")
    ap.add_argument("--epochs", type=int, default=settings.EPOCHS)
    ap.add_argument("--patience", type=int, default=settings.PATIENCE,
                    help="epochs without improvement before stopping")
    ap.add_argument("--imgsz", type=int, default=settings.TRAIN_INPUT_SIZE)
    ap.add_argument("--batch", type=int, default=settings.BATCH_SIZE)
    ap.add_argument("--name", default="cards",
                    help="run name: output goes to models/<name>/. read.py uses models/cards")
    ap.add_argument("--extract-art", metavar="EXPORT", help="make card art from a one-card-per-photo export, then stop")
    ap.add_argument("--complete-corners", metavar="EXPORT",
                    help="label the corners an export left out, using the current model, then stop")
    ap.add_argument("--promote", metavar="RUN", help="copy RUN's best.pt and record to models/cards, then stop")
    ap.add_argument("--manifest", action="store_true", help="write data/manifest.json from data/raw, then stop")
    ap.add_argument("--check-data", action="store_true", help="compare data/raw with data/manifest.json, then stop")
    args = ap.parse_args()
    try:
        settings.validate(settings)
    except ValueError as error:
        raise SystemExit(error) from None

    manifest = ROOT / settings.MANIFEST_FILE
    if args.promote:
        provenance.promote(args.promote, ROOT / settings.LIVE_MODEL_DIR)
        return
    if args.manifest:
        provenance.write_manifest(ROOT / "data" / "raw", manifest)
        return
    if args.check_data:
        raise SystemExit(1 if provenance.check_manifest(ROOT / "data" / "raw", manifest) else 0)
    if args.complete_corners:
        autolabel.complete_corners(args.complete_corners, ROOT / settings.LABELED_DIR, str(ROOT / settings.MODEL_PATH))
        return
    if args.extract_art:
        card_art.extract_card_art(args.extract_art, ROOT / settings.CARD_ART_DIR)
        return

    data_dir = ROOT / settings.DATASET_DIR
    if args.rebuild or not (data_dir / "data.yaml").exists():
        if provenance.check_manifest(ROOT / "data" / "raw", manifest):
            print(f"warning: data/raw differs from {settings.MANIFEST_FILE} (above). Training anyway;"
                  " run `python train.py --manifest` afterwards if this data is what the model should learn from.")
        dataset.build(data_dir, ROOT / settings.LABELED_DIR, ROOT / settings.CARD_ART_DIR,
                      ROOT / settings.BACKGROUNDS_DIR, args.synthetic, settings.VAL_FRACTION)

    import torch
    from ultralytics import YOLO

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = ROOT / settings.MODELS_DIR
    # A bare name such as yolo11s.pt is downloaded into models/ the first time, not into the current folder.
    base = Path(args.base) if Path(args.base).parent != Path(".") else models_dir / args.base
    model = YOLO(str(base))
    model.train(data=str(data_dir / "data.yaml"), epochs=args.epochs, patience=args.patience,
                imgsz=args.imgsz, batch=args.batch, device=device,
                name=args.name, project=str(models_dir), exist_ok=True,
                # Cards are never mirrored, so disable flips; keep rotation and colour jitter.
                fliplr=0.0, flipud=0.0, degrees=15, hsv_v=0.4)
    print("best weights:", model.trainer.best)


if __name__ == "__main__":
    main()
