"""Build the one training set, data/dataset, from everything in data/raw.

  * Every subfolder of the labeled folder is a YOLO-format export (Roboflow or
    similar) with a data.yaml. Class ids are remapped by NAME to the order in
    deck.py, so exports with different orderings combine cleanly. An export
    whose class names are mostly not cards (e.g. a rank/suit card-art export)
    is skipped with a warning, and so is an export whose boxes are whole cards
    rather than corner indices.
  * Images are hard-linked, not copied, so the dataset takes almost no extra
    disk space. Label files are rewritten with the remapped ids.
  * An export's own train/valid/test split is kept (test counts as val). An
    export with only a train split gets a stable hash-based val split.
  * Synthetic images from synthetic.py are added on top.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import yaml

from . import synthetic
from .deck import CLASSES, CLASS_TO_ID

WHOLE_CARD_WIDTH = 0.08  # median box width (fraction of image) above this means whole-card boxes
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build(out: Path, labeled_dir: Path, card_art_dir: Path, backgrounds_dir: Path,
          n_synthetic: int, val_frac: float):
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    exports = sorted(d for d in labeled_dir.iterdir() if (d / "data.yaml").exists()) if labeled_dir.is_dir() else []
    for i, export in enumerate(exports):
        import_export(export, out, f"e{i}_", val_frac)
    if n_synthetic:
        synthetic.generate(out, n_synthetic, card_art_dir, backgrounds_dir, val_frac)

    names = "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n{names}")
    # The same exam without synthetic images, for an honest score: yolo val data=data/dataset/realVal.yaml
    real_val = sorted(str(p.resolve()) for p in (out / "images" / "val").iterdir() if not p.name.startswith("synth_"))
    (out / "realVal.txt").write_text("\n".join(real_val) + "\n")
    (out / "realVal.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: realVal.txt\n{names}")
    counts = {s: sum(1 for _ in (out / "images" / s).iterdir()) for s in ("train", "val")}
    if not counts["train"]:
        raise SystemExit(f"dataset is empty: put exports in {labeled_dir} or set SYNTHETIC_IMAGES")
    print(f"dataset: train {counts['train']}, val {counts['val']} -> {out}")


def import_export(export: Path, out: Path, prefix: str, val_frac: float):
    names = yaml.safe_load((export / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    remap = {i: CLASS_TO_ID[n] for i, n in enumerate(names) if n in CLASS_TO_ID}
    if len(remap) < len(names) / 2:
        print(f"skip {export.name}: its classes are not card names ({names[:5]} ...)")
        return
    widths = [float(line.split()[3]) for lp in export.glob("*/labels/*.txt")
              for line in lp.read_text().splitlines() if len(line.split()) == 5]
    if widths and sorted(widths)[len(widths) // 2] > WHOLE_CARD_WIDTH:
        print(f"skip {export.name}: boxes are whole cards, not corner indices")
        return
    dropped = [n for n in names if n not in CLASS_TO_ID]
    if dropped:
        print(f"  {export.name}: dropping classes not in the deck: {dropped}")

    splits = {s: export / s / "images" for s in ("train", "valid", "val", "test") if (export / s / "images").is_dir()}
    has_val = any(s != "train" for s in splits)
    counts = {"train": 0, "val": 0, "empty": 0}
    for split, img_dir in splits.items():
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            lp = export / split / "labels" / (img.stem + ".txt")
            lines = []
            if lp.exists():
                for line in lp.read_text().splitlines():
                    f = line.split()
                    if len(f) >= 5 and int(f[0]) in remap:
                        lines.append(" ".join([str(remap[int(f[0])])] + f[1:5]))
            if not lines:
                counts["empty"] += 1
                continue
            if has_val:
                target = "train" if split == "train" else "val"
            else:
                h = int(hashlib.md5(img.name.encode()).hexdigest(), 16) % 1000
                target = "val" if h < val_frac * 1000 else "train"
            stem = prefix + img.stem
            dst = out / "images" / target / (stem + img.suffix.lower())
            try:
                dst.hardlink_to(img)
            except OSError:
                shutil.copy2(img, dst)
            (out / "labels" / target / (stem + ".txt")).write_text("\n".join(lines) + "\n")
            counts[target] += 1
    print(f"  {export.name}: train {counts['train']}, val {counts['val']}, skipped (no boxes) {counts['empty']}")
