"""Turn photos of your deck into clean card art for the synthetic generator.

    python train.py --extract-art data/raw/myDeckExport

Input is a Roboflow export where each photo shows ONE card and has a box around
the whole card (class "cards") plus, for face cards, a box around the rank and
a box around the suit. The card's name comes from the filename: 10C_jpg.rf...
is the ten of clubs. Card backs must be named BACK_1, BACK_2 ... (any prefix
starting with BACK) and need only the card box. Anything else without rank and
suit boxes (a Joker, say) is skipped.

For each photo this:
  * crops the card and straightens it (finds the card's outline and warps it to
    a flat 500x700 rectangle),
  * rounds the corners so no background leaks in,
  * saves <out>/<LABEL>.png (backs become BACK_1.png, BACK_2.png ...),
  * measures where the rank+suit index sits as fractions of the card.

The averaged index position is written to <out>/indexRect.json, which
synthetic.py uses to place the corner-index boxes.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from .deck import CLASSES

OUT_W, OUT_H = 500, 700  # standard card aspect 2.5 x 3.5
CARD_CLASS = "cards"
RANK_CLASSES = {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"}
SUIT_CLASSES = {"c": "C", "d": "D", "h": "H", "s": "S"}


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return 4 points as top-left, top-right, bottom-right, bottom-left, portrait orientation."""
    sums = pts.sum(axis=1)                 # x + y: smallest top-left, largest bottom-right
    diffs = np.diff(pts, axis=1).ravel()   # y - x: smallest top-right, largest bottom-left
    tl, br = pts[np.argmin(sums)], pts[np.argmax(sums)]
    tr, bl = pts[np.argmin(diffs)], pts[np.argmax(diffs)]
    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    # If the card is wider than tall in the photo, rotate the corner order so output is portrait.
    if np.linalg.norm(tr - tl) > np.linalg.norm(bl - tl):
        quad = np.array([tr, br, bl, tl], dtype=np.float32)
    return quad


def find_card_quad(crop: np.ndarray, expected_area: float) -> np.ndarray | None:
    """Find the card outline inside a loose crop. Returns 4 corners or None.

    A card's border is white, so it has low colour saturation and high
    brightness, while the table underneath is usually coloured. Threshold on
    that, take the largest blob, and fit a rotated rectangle to it."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    mask = ((sat < 70) & (val > 120)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    rect = cv2.minAreaRect(hull)
    w, h = rect[1]
    if min(w, h) == 0:
        return None
    aspect = min(w, h) / max(w, h)
    area_ratio = (w * h) / expected_area
    if not (0.62 < aspect < 0.80 and 0.80 < area_ratio < 1.08):
        return None
    quad = order_corners(cv2.boxPoints(rect))
    # pull the corners 1% toward the centre so no table fringe survives the warp
    centre = quad.mean(axis=0)
    return (centre + (quad - centre) * 0.99).astype(np.float32)


def rounded_alpha(w: int, h: int, radius: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)
    for cx, cy in ((radius, radius), (w - radius, radius), (radius, h - radius), (w - radius, h - radius)):
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


def extract_card_art(export: str | Path, out: str | Path, pad: float = 0.04):
    """pad: loose-crop margin around the card box, as a fraction of its size."""
    export, out = Path(export), Path(out)
    names = yaml.safe_load((export / "data.yaml").read_text())["names"]
    out.mkdir(parents=True, exist_ok=True)

    index_rects = []
    backs = 0
    made = set()
    for split in ("train", "valid", "test"):
        for lp in sorted((export / split / "labels").glob("*.txt")) if (export / split).exists() else []:
            ip = next((export / split / "images").glob(lp.stem + ".*"), None)
            if ip is None:
                continue
            img = cv2.imread(str(ip))
            H, W = img.shape[:2]
            boxes = {}
            for line in lp.read_text().splitlines():
                fields = line.split()
                if len(fields) < 5:
                    continue
                cls = names[int(fields[0])]
                cx, cy, bw, bh = (float(v) for v in fields[1:5])
                boxes[cls] = (cx * W - bw * W / 2, cy * H - bh * H / 2, cx * W + bw * W / 2, cy * H + bh * H / 2)
            if CARD_CLASS not in boxes:
                print("skip (no card box):", ip.name)
                continue

            prefix = ip.name.split("_")[0].upper()
            rank = next((c for c in boxes if c in RANK_CLASSES), None)
            suit = next((c for c in boxes if c in SUIT_CLASSES), None)
            if rank and suit:
                label = rank + SUIT_CLASSES[suit]
                if prefix in CLASSES and prefix != label:
                    print(f"warning: filename says {prefix} but boxes say {label}; using boxes")
            elif prefix in CLASSES:
                label = prefix
            elif prefix.startswith("BACK"):
                backs += 1
                label = f"BACK_{backs}"
            else:
                print(f"skip {ip.name[:24]}: no rank/suit boxes and not named BACK (a Joker?)")
                continue

            # loose crop around the card box, then find the true outline inside it
            x1, y1, x2, y2 = boxes[CARD_CLASS]
            px, py = (x2 - x1) * pad, (y2 - y1) * pad
            cx1, cy1 = int(max(0, x1 - px)), int(max(0, y1 - py))
            cx2, cy2 = int(min(W, x2 + px)), int(min(H, y2 + py))
            crop = img[cy1:cy2, cx1:cx2]
            quad = find_card_quad(crop, (x2 - x1) * (y2 - y1))
            if quad is None:
                print(f"note: outline not found for {label}, using the box as-is")
                box_corners = [[x1 - cx1, y1 - cy1], [x2 - cx1, y1 - cy1], [x2 - cx1, y2 - cy1], [x1 - cx1, y2 - cy1]]
                quad = np.array(box_corners, np.float32)
            dst = np.array([[0, 0], [OUT_W, 0], [OUT_W, OUT_H], [0, OUT_H]], np.float32)
            warp = cv2.getPerspectiveTransform(quad, dst)
            card = cv2.warpPerspective(crop, warp, (OUT_W, OUT_H), flags=cv2.INTER_AREA)
            alpha = rounded_alpha(OUT_W, OUT_H, int(OUT_W * 0.045))
            cv2.imwrite(str(out / f"{label}.png"), np.dstack([card, alpha]))
            made.add(label)

            if rank and suit:
                pts = []
                for glyph in (rank, suit):
                    bx1, by1, bx2, by2 = boxes[glyph]
                    pts += [[bx1 - cx1, by1 - cy1], [bx2 - cx1, by1 - cy1],
                            [bx2 - cx1, by2 - cy1], [bx1 - cx1, by2 - cy1]]
                warped = cv2.perspectiveTransform(np.array([pts], np.float32), warp)[0]
                ix1, iy1 = warped.min(axis=0)
                ix2, iy2 = warped.max(axis=0)
                index_rects.append((ix1 / OUT_W, iy1 / OUT_H, (ix2 - ix1) / OUT_W, (iy2 - iy1) / OUT_H))

    missing = [c for c in CLASSES if c != "BACK" and c not in made]
    print(f"wrote {len(made)} card images to {out} ({backs} backs)")
    if missing:
        print("missing faces:", missing)
    if backs == 0:
        print("no card backs yet: photograph one, name the file BACK_1.jpg, label only the card box, re-run."
              " Until then the synthetic generator draws a placeholder back.")
    if index_rects:
        rects = np.array(index_rects)
        x, y, w, h = rects.mean(axis=0)
        pad = 0.01  # a hair of margin so the box does not clip the glyphs
        rect = {"x": round(float(x - pad), 4), "y": round(float(y - pad), 4),
                "w": round(float(w + 2 * pad), 4), "h": round(float(h + 2 * pad), 4),
                "note": f"corner index position as fractions of the card, averaged over {len(rects)} cards"}
        (out / "indexRect.json").write_text(json.dumps(rect, indent=2))
        print("index rect (x, y, w, h fractions):", rect["x"], rect["y"], rect["w"], rect["h"],
              " spread:", np.round(rects.std(axis=0), 3).tolist())
