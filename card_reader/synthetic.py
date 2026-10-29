"""Generate synthetic training images of cards scattered on a table.

Card art: if the card art folder holds a png for every face (AS ... KC) those
are used, plus any BACK*.png (made by card_art.py, which also writes
indexRect.json saying where the corner index sits). Otherwise simple
placeholder cards are drawn so the pipeline can be exercised before the real
deck is photographed.

Backgrounds: any images in the backgrounds folder are used at random;
otherwise a noisy felt colour with a lighting gradient is synthesised.

Labels: one box per visible corner index (two per face-up card), and one box
around the whole card for BACK. Indices covered by another card are skipped.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .deck import BACK, CLASS_TO_ID, CLASSES, RED_SUITS, SUIT_SYMBOL, rank_of, suit_of

CARD_W, CARD_H = 250, 350
PLACEHOLDER_INDEX_RECT = (0.04, 0.03, 0.20, 0.26)  # x, y, w, h fractions of the card
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "DejaVuSans-Bold.ttf",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for font_path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_placeholder(label: str) -> Image.Image:
    img = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, CARD_W - 1, CARD_H - 1), radius=18,
                           fill=(250, 250, 248, 255), outline=(60, 60, 60, 255), width=2)
    if label == BACK:
        draw.rounded_rectangle((14, 14, CARD_W - 15, CARD_H - 15), radius=10, fill=(40, 70, 160, 255))
        for k in range(-CARD_H, CARD_W + CARD_H, 22):
            draw.line((k, 14, k + CARD_H, CARD_H - 15), fill=(90, 120, 210, 255), width=3)
            draw.line((k, CARD_H - 15, k + CARD_H, 14), fill=(90, 120, 210, 255), width=3)
        return img
    rank, suit = rank_of(label), suit_of(label)
    color = (200, 20, 30, 255) if suit in RED_SUITS else (20, 20, 20, 255)
    f_rank = load_font(44 if rank != "10" else 38)
    f_suit = load_font(40)
    f_big = load_font(120)
    sym = SUIT_SYMBOL[suit]
    card_size = (CARD_W, CARD_H, CARD_W, CARD_H)
    x, y, w, h = [frac * size for frac, size in zip(PLACEHOLDER_INDEX_RECT, card_size, strict=True)]
    corner = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    corner_draw = ImageDraw.Draw(corner)
    corner_draw.text((w / 2, 2), rank, font=f_rank, fill=color, anchor="mt")
    corner_draw.text((w / 2, h - 2), sym, font=f_suit, fill=color, anchor="mb")
    img.alpha_composite(corner, (int(x), int(y)))
    img.alpha_composite(corner.rotate(180), (int(CARD_W - x - w), int(CARD_H - y - h)))
    draw.text((CARD_W / 2, CARD_H / 2), sym, font=f_big, fill=color, anchor="mm")
    return img


def load_card_art(assets: Path):
    """Returns (art, index_rect). art maps each label to a list of RGBA images (BACK may have several)."""
    faces = {c: assets / f"{c}.png" for c in CLASSES if c != BACK}
    backs = sorted(assets.glob("BACK*.png"))
    rect_file = assets / "indexRect.json"
    if all(path.exists() for path in faces.values()):
        art = {c: [Image.open(path).convert("RGBA")] for c, path in faces.items()}
        if backs:
            print(f"using real card art from {assets} ({len(backs)} back design(s))")
            art[BACK] = [Image.open(path).convert("RGBA") for path in backs]
        else:
            print(f"using real card art from {assets}; no BACK*.png yet so drawing a placeholder back")
            placeholder = draw_placeholder(BACK)
            ref = next(iter(art.values()))[0]
            art[BACK] = [placeholder.resize(ref.size, Image.LANCZOS)]
        rect = None
        if rect_file.exists():
            saved = json.loads(rect_file.read_text())
            rect = (saved["x"], saved["y"], saved["w"], saved["h"])
        return art, rect
    print(f"no complete card set in {assets}, drawing placeholder cards")
    return {c: [draw_placeholder(c)] for c in CLASSES}, PLACEHOLDER_INDEX_RECT


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


def rotate_points(pts, angle_deg, cw, ch, out_w, out_h):
    """Map card-space points through PIL's rotate(angle, expand=True)."""
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    out = []
    for x, y in pts:
        dx, dy = x - cw / 2, y - ch / 2
        rx = dx * cos_a + dy * sin_a
        ry = -dx * sin_a + dy * cos_a
        out.append((rx + out_w / 2, ry + out_h / 2))
    return out


def rect_corners(x, y, w, h):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def visible_fraction(id_canvas: np.ndarray, poly, card_id: int) -> float:
    mask = Image.new("L", id_canvas.shape[::-1], 0)
    ImageDraw.Draw(mask).polygon([(float(x), float(y)) for x, y in poly], fill=255)
    inside = np.array(mask) > 0
    if not inside.any():
        return 0.0
    return float((id_canvas[inside] == card_id).mean())


def compose(rng, art, index_rect, bg_pool, W, H, max_cards, glare_p):
    """One W x H image of 1 to max_cards random cards on a background.

    Returns (image, labels), labels as (class id, x1, y1, x2, y2) in pixels. A
    corner index less than 60% visible gets no label. glare_p is the chance of
    a glare spot."""
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
            xs, ys = zip(*poly, strict=True)
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


def generate(out: Path, n: int, card_art_dir: Path, backgrounds_dir: Path, val_frac: float,
             size=(1280, 960), max_cards=12, glare=0.4, seed=0):
    """Write n images and labels into out/images/{train,val} and out/labels/{train,val}."""
    W, H = size
    art, index_rect = load_card_art(card_art_dir)
    if index_rect is None:
        raise SystemExit(f"real card art needs {card_art_dir}/indexRect.json (made by train.py --extract-art)")
    bg_pool = [path for path in backgrounds_dir.glob("*") if path.suffix.lower() in (".jpg", ".jpeg", ".png")]

    rng = random.Random(seed)
    for i in range(n):
        img, labels = compose(rng, art, index_rect, bg_pool, W, H, max_cards, glare)
        split = "val" if rng.random() < val_frac else "train"
        stem = f"synth_{i:06d}"
        img.save(out / "images" / split / f"{stem}.jpg", quality=90)
        lines = [f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n"
                 for cls, x1, y1, x2, y2 in labels]
        (out / "labels" / split / f"{stem}.txt").write_text("".join(lines))
        if (i + 1) % 250 == 0:
            print(f"  synthetic {i + 1}/{n}")
