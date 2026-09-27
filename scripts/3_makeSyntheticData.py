"""Step 3: generate a synthetic training set of cards scattered on a table.

    python scripts/3_makeSyntheticData.py --n 2000

Card art: if data/raw/cardArt/ holds a png for every face (AS ... KC) plus at
least one BACK*.png, those are used (made by scripts/2_extractCardArt.py, which
also writes indexRect.json saying where the corner index sits). Otherwise simple
placeholder cards are drawn so the pipeline can be exercised before the real
deck is photographed. --index-rect x,y,w,h overrides the json.

Backgrounds: any images in data/raw/backgrounds are used at random; otherwise a
noisy felt colour with a lighting gradient is synthesised.

Labels: one box per visible corner index (two per face-up card), and one box
around the whole card for BACK. Indices covered by another card are skipped.
Output is YOLO format: images/ and labels/ per split plus data.yaml.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import settings  # noqa: E402
from cardVision.deck import BACK, CLASSES, CLASS_TO_ID, RED_SUITS, SUIT_SYMBOL, rank_of, suit_of  # noqa: E402

CARD_W, CARD_H = 250, 350
PLACEHOLDER_INDEX_RECT = (0.04, 0.03, 0.20, 0.26)  # x, y, w, h fractions of the card
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "DejaVuSans-Bold.ttf",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ------------------------------------------------------------------ card art
def draw_placeholder(label: str) -> Image.Image:
    img = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, CARD_W - 1, CARD_H - 1), radius=18, fill=(250, 250, 248, 255), outline=(60, 60, 60, 255), width=2)
    if label == BACK:
        d.rounded_rectangle((14, 14, CARD_W - 15, CARD_H - 15), radius=10, fill=(40, 70, 160, 255))
        for k in range(-CARD_H, CARD_W + CARD_H, 22):
            d.line((k, 14, k + CARD_H, CARD_H - 15), fill=(90, 120, 210, 255), width=3)
            d.line((k, CARD_H - 15, k + CARD_H, 14), fill=(90, 120, 210, 255), width=3)
        return img
    rank, suit = rank_of(label), suit_of(label)
    color = (200, 20, 30, 255) if suit in RED_SUITS else (20, 20, 20, 255)
    f_rank = load_font(44 if rank != "10" else 38)
    f_suit = load_font(40)
    f_big = load_font(120)
    sym = SUIT_SYMBOL[suit]
    x, y, w, h = [v * s for v, s in zip(PLACEHOLDER_INDEX_RECT, (CARD_W, CARD_H, CARD_W, CARD_H))]
    corner = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    cd = ImageDraw.Draw(corner)
    cd.text((w / 2, 2), rank, font=f_rank, fill=color, anchor="mt")
    cd.text((w / 2, h - 2), sym, font=f_suit, fill=color, anchor="mb")
    img.alpha_composite(corner, (int(x), int(y)))
    img.alpha_composite(corner.rotate(180), (int(CARD_W - x - w), int(CARD_H - y - h)))
    d.text((CARD_W / 2, CARD_H / 2), sym, font=f_big, fill=color, anchor="mm")
    return img


def load_card_art(assets: Path):
    """Returns (art, index_rect). art maps each label to a list of RGBA images (BACK may have several)."""
    faces = {c: assets / f"{c}.png" for c in CLASSES if c != BACK}
    backs = sorted(assets.glob("BACK*.png"))
    rect_file = assets / "indexRect.json"
    if all(p.exists() for p in faces.values()):
        art = {c: [Image.open(p).convert("RGBA")] for c, p in faces.items()}
        if backs:
            print(f"using real card art from {assets} ({len(backs)} back design(s))")
            art[BACK] = [Image.open(p).convert("RGBA") for p in backs]
        else:
            print(f"using real card art from {assets}; no BACK*.png yet so drawing a placeholder back")
            ph = draw_placeholder(BACK)
            ref = next(iter(art.values()))[0]
            art[BACK] = [ph.resize(ref.size, Image.LANCZOS)]
        rect = None
        if rect_file.exists():
            j = json.loads(rect_file.read_text())
            rect = (j["x"], j["y"], j["w"], j["h"])
        return art, rect
    print("no complete card set in data/raw/cardArt, drawing placeholder cards")
    return {c: [draw_placeholder(c)] for c in CLASSES}, PLACEHOLDER_INDEX_RECT


# ------------------------------------------------------------------ backgrounds
def make_background(w: int, h: int, rng: random.Random, pool: list[Path]) -> Image.Image:
    if pool:
        bg = Image.open(rng.choice(pool)).convert("RGB")
        scale = max(w / bg.width, h / bg.height) * rng.uniform(1.0, 1.4)
        bg = bg.resize((int(bg.width * scale), int(bg.height * scale)))
        x0 = rng.randint(0, bg.width - w)
        y0 = rng.randint(0, bg.height - h)
        return bg.crop((x0, y0, x0 + w, y0 + h))
    base = rng.choice([(20, 90, 50), (25, 60, 120), (110, 30, 30), (40, 40, 45), (120, 100, 70)])
    arr = np.zeros((h, w, 3), dtype=np.float32) + np.array(base, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
    arr *= (1 + 0.35 * (gx * (xx / w - 0.5) + gy * (yy / h - 0.5)))[..., None]
    arr += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 6, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ------------------------------------------------------------------ geometry
def rotate_points(pts, angle_deg, cw, ch, out_w, out_h):
    """Map card-space points through PIL's rotate(angle, expand=True)."""
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    out = []
    for x, y in pts:
        dx, dy = x - cw / 2, y - ch / 2
        rx = dx * c + dy * s
        ry = -dx * s + dy * c
        out.append((rx + out_w / 2, ry + out_h / 2))
    return out


def rect_corners(x, y, w, h):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def visible_fraction(id_canvas: np.ndarray, poly, card_id: int) -> float:
    mask = Image.new("L", id_canvas.shape[::-1], 0)
    ImageDraw.Draw(mask).polygon([(float(x), float(y)) for x, y in poly], fill=255)
    m = np.array(mask) > 0
    if not m.any():
        return 0.0
    return float((id_canvas[m] == card_id).mean())


# ------------------------------------------------------------------ one image
def compose(rng, art, index_rect, bg_pool, W, H, max_cards, glare_p):
    bg = make_background(W, H, rng, bg_pool).convert("RGBA")
    id_canvas = np.zeros((H, W), dtype=np.int32)
    labels = []  # (class_id, x1, y1, x2, y2)
    pending = []  # (card_id, class_id, polygons)
    n = rng.randint(1, max_cards)
    for card_id in range(1, n + 1):
        label = rng.choice(CLASSES) if rng.random() > 0.15 else BACK
        card = rng.choice(art[label])
        cw0, ch0 = card.size
        target_w = W * rng.uniform(0.07, 0.15)
        scale = target_w / cw0
        cw, ch = int(cw0 * scale), int(ch0 * scale)
        cardr = card.resize((cw, ch), Image.LANCZOS)
        angle = rng.uniform(-180, 180)
        rot = cardr.rotate(angle, expand=True, resample=Image.BICUBIC)
        rw, rh = rot.size
        px = rng.randint(-rw // 4, W - rw + rw // 4)
        py = rng.randint(-rh // 4, H - rh + rh // 4)
        bg.alpha_composite(rot, (px, py))
        alpha = np.array(rot.split()[-1]) > 128
        ys, xs = np.nonzero(alpha)
        xs, ys = xs + px, ys + py
        keep = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        id_canvas[ys[keep], xs[keep]] = card_id

        if label == BACK:
            polys = [rect_corners(0, 0, cw, ch)]
        else:
            ix, iy, iw, ih = index_rect
            tl = rect_corners(ix * cw, iy * ch, iw * cw, ih * ch)
            br = rect_corners(cw - (ix + iw) * cw, ch - (iy + ih) * ch, iw * cw, ih * ch)
            polys = [tl, br]
        polys = [[(x + px, y + py) for x, y in rotate_points(p, angle, cw, ch, rw, rh)] for p in polys]
        pending.append((card_id, CLASS_TO_ID[label], polys))

    for card_id, cls, polys in pending:
        for poly in polys:
            if visible_fraction(id_canvas, poly, card_id) < 0.6:
                continue
            xs, ys = zip(*poly)
            x1, y1, x2, y2 = max(0, min(xs)), max(0, min(ys)), min(W, max(xs)), min(H, max(ys))
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue
            labels.append((cls, x1, y1, x2, y2))

    img = bg.convert("RGB")
    if rng.random() < glare_p:  # soft glare blob
        glare = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gx, gy, gr = rng.randint(0, W), rng.randint(0, H), rng.randint(W // 8, W // 3)
        ImageDraw.Draw(glare).ellipse((gx - gr, gy - gr, gx + gr, gy + gr), fill=(255, 255, 255, rng.randint(40, 110)))
        glare = glare.filter(ImageFilter.GaussianBlur(gr / 2))
        img = Image.alpha_composite(img.convert("RGBA"), glare).convert("RGB")
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    arr = np.array(img).astype(np.float32)
    arr = (arr - 128) * rng.uniform(0.8, 1.2) + 128 + rng.uniform(-25, 25)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default=settings.SYNTHETIC_DIR)
    ap.add_argument("--size", default="1280x960")
    ap.add_argument("--max-cards", type=int, default=12)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--glare", type=float, default=0.4)
    ap.add_argument("--index-rect", default=None, help="x,y,w,h fractions for real card art")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="also write images with boxes drawn to <out>/debug")
    args = ap.parse_args()

    W, H = map(int, args.size.lower().split("x"))
    root = Path(__file__).resolve().parents[1]
    art, index_rect = load_card_art(root / settings.CARD_ART_DIR)
    if args.index_rect:
        index_rect = tuple(map(float, args.index_rect.split(",")))
    if index_rect is None:
        sys.exit("real card art needs indexRect.json (from scripts/2_extractCardArt.py) or --index-rect x,y,w,h")
    bg_pool = [p for p in (root / settings.BACKGROUNDS_DIR).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    if args.debug:
        (out / "debug").mkdir(exist_ok=True)

    rng = random.Random(args.seed)
    for i in range(args.n):
        img, labels = compose(rng, art, index_rect, bg_pool, W, H, args.max_cards, args.glare)
        split = "val" if rng.random() < args.val_frac else "train"
        stem = f"synth_{i:06d}"
        img.save(out / "images" / split / f"{stem}.jpg", quality=90)
        with open(out / "labels" / split / f"{stem}.txt", "w") as f:
            for cls, x1, y1, x2, y2 in labels:
                f.write(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
        if args.debug:
            dbg = img.copy()
            d = ImageDraw.Draw(dbg)
            for cls, x1, y1, x2, y2 in labels:
                d.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
                d.text((x1, y1 - 12), CLASSES[cls], fill=(0, 255, 0))
            dbg.save(out / "debug" / f"{stem}.jpg", quality=85)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{args.n}")

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print("wrote", out / "data.yaml")


if __name__ == "__main__":
    main()
