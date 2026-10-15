"""Fill in corner indices an export left unlabelled, using the trained model.

    python train.py --complete-corners data/raw/unused/cards

Some exports label only two diagonal corners of each card, although the deck
prints an index in all four. Training treats an unlabelled corner as "nothing
here", which teaches the model to ignore real corners. This runs the current
model on every photo and adds its boxes, but only when all of these hold:

  * the card is already labelled somewhere in the same photo, so the model can
    only add corners of cards a person confirmed, never invent a card
  * the model's confidence is at least MIN_CONF
  * the box does not overlap a labelled box (that corner is already done)

Exports draw their boxes tighter or looser than the data the model learned
from, so the added boxes are then scaled around their centres until their
median size matches the export's own labels. Every box in a photo then
follows the same convention.

The result is a new export in data/raw/labeled/<name>Complete with images
hard-linked, the original labels kept, and the added boxes appended. preview/
holds sheets to check by eye: green = original labels, magenta = added.
"""
from __future__ import annotations

import random
import shutil
import statistics
from pathlib import Path

import cv2
import yaml

from .deck import CLASS_TO_ID

MIN_CONF = 0.6
MAX_OVERLAP = 0.2  # IoU with a labelled box above this means the corner is already labelled


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def complete_corners(export: str | Path, out_dir: Path, weights: str, imgsz: int = 1024, preview: int = 36):
    from ultralytics import YOLO

    export = Path(export)
    out = out_dir / (export.name + "Complete")
    names = yaml.safe_load((export / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    export_id = {str(n).upper(): i for i, n in enumerate(names)}  # deck name -> this export's class id
    model = YOLO(weights)

    if out.exists():
        shutil.rmtree(out)
    (out / "preview").mkdir(parents=True)
    shutil.copy2(export / "data.yaml", out / "data.yaml")
    totals = {"images": 0, "original": 0, "added": 0, "images_changed": 0}
    results = []  # (split, image path, labelled, added, W, H, original lines)
    for split_dir in sorted(d for d in export.iterdir() if (d / "images").is_dir()):
        for ip in sorted((split_dir / "images").iterdir()):
            lp = split_dir / "labels" / (ip.stem + ".txt")
            lines = [l for l in lp.read_text().splitlines() if len(l.split()) == 5] if lp.exists() else []
            img = cv2.imread(str(ip))
            if img is None:
                continue
            H, W = img.shape[:2]
            labelled = []  # (deck name, box px)
            for l in lines:
                c, cx, cy, w, h = l.split()
                cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                labelled.append((str(names[int(c)]).upper(), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
            present = {n for n, _ in labelled}

            added = []
            res = model.predict(img, imgsz=imgsz, conf=MIN_CONF, verbose=False)[0]
            for b in res.boxes:
                name = model.names[int(b.cls)]
                box = tuple(b.xyxy[0].tolist())
                if name not in present or name not in export_id:
                    continue  # only corners of cards a person labelled in this photo
                if any(iou(box, lb) > MAX_OVERLAP for _, lb in labelled + added):
                    continue  # this corner already has a box
                added.append((name, box))
            results.append((split_dir.name, ip, labelled, added, W, H, lines))

    # Match the export's own box size: scale added boxes around their centres.
    def sizes(key):
        return [(b[2] - b[0], b[3] - b[1]) for r in results for _, b in r[key]]
    orig, new = sizes(2), sizes(3)
    sx = statistics.median(w for w, _ in orig) / statistics.median(w for w, _ in new) if new else 1.0
    sy = statistics.median(h for _, h in orig) / statistics.median(h for _, h in new) if new else 1.0

    def scaled(box):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        hw, hh = (box[2] - box[0]) * sx / 2, (box[3] - box[1]) * sy / 2
        return cx - hw, cy - hh, cx + hw, cy + hh

    samples = []
    for split, ip, labelled, added, W, H, lines in results:
        added = [(n, scaled(b)) for n, b in added]
        new_lines = lines + [
            f"{export_id[n]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
            for n, (x1, y1, x2, y2) in added]
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        (out / split / "images" / ip.name).hardlink_to(ip)
        (out / split / "labels" / (ip.stem + ".txt")).write_text("\n".join(new_lines) + "\n")
        totals["images"] += 1
        totals["original"] += len(lines)
        totals["added"] += len(added)
        totals["images_changed"] += bool(added)
        samples.append((ip, labelled, added))

    random.Random(0).shuffle(samples)
    _preview_sheets(samples[:preview], out / "preview")
    summary = (f"{export.name} -> {out}\n"
               f"images {totals['images']}, original boxes {totals['original']}, added {totals['added']} "
               f"(in {totals['images_changed']} images), confidence >= {MIN_CONF}\n"
               f"added boxes scaled by {sx:.2f} x {sy:.2f} to match the export's own box size\n")
    (out / "preview" / "summary.txt").write_text(summary)
    print(summary + f"check the sheets in {out / 'preview'} (green = original, magenta = added)")
    return out


def _preview_sheets(samples, folder: Path, per_sheet=6, tile=540):
    for s in range(0, len(samples), per_sheet):
        sheet = []
        for ip, labelled, added in samples[s:s + per_sheet]:
            view = cv2.imread(str(ip))
            for colour, boxes in (((0, 255, 0), labelled), ((255, 0, 255), added)):
                for name, (x1, y1, x2, y2) in boxes:
                    cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
                    cv2.putText(view, name, (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
            sheet.append(cv2.resize(view, (tile, tile)))
        while len(sheet) < per_sheet:
            sheet.append(sheet[0] * 0)
        rows = [cv2.hconcat(sheet[i:i + 3]) for i in range(0, per_sheet, 3)]
        cv2.imwrite(str(folder / f"sheet_{s // per_sheet + 1:02d}.jpg"), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
