"""Step 4: merge YOLO-format datasets into one training set with this project's class order.

    python scripts/4_importDatasets.py "data/raw/roboflow/Playing Cards" data/synthetic

Each source needs a data.yaml with a `names` list (Roboflow exports and
scripts/3_makeSyntheticData.py both write one). Class ids are remapped by NAME
to the order in cardVision/deck.py, so datasets with different orderings merge
cleanly. Names not in the deck (e.g. "Joker") are dropped with a warning.

Splits: a source's own train/valid/test folders are honoured (test is folded
into val). A source with only a train split gets a deterministic hash-based
val split of --val-frac. Images with no boxes are skipped unless --keep-empty.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import settings  # noqa: E402
from cardVision.deck import CLASSES, CLASS_TO_ID  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_split_dirs(root: Path) -> dict[str, Path]:
    """Return {split: images_dir} for whichever of train/valid/val/test exist."""
    out = {}
    for name, split in (("train", "train"), ("valid", "val"), ("val", "val"), ("test", "val")):
        d = root / name / "images"
        if d.is_dir():
            out.setdefault(split, []).append(d)
    # tools/synth_data.py layout: images/{train,val}
    for split in ("train", "val"):
        d = root / "images" / split
        if d.is_dir():
            out.setdefault(split, []).append(d)
    return out


def label_path_for(img: Path) -> Path:
    # .../<split>/images/x.jpg -> .../<split>/labels/x.txt   or  .../images/<split>/x.jpg -> .../labels/<split>/x.txt
    parts = list(img.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--out", default=settings.MERGED_DIR)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--keep-empty", action="store_true")
    ap.add_argument("--link", action="store_true", help="hard-link images instead of copying")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    totals = {"train": 0, "val": 0}
    for si, src in enumerate(args.sources):
        src = Path(src)
        names = yaml.safe_load((src / "data.yaml").read_text())["names"]
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        remap = {}
        for i, n in enumerate(names):
            if n in CLASS_TO_ID:
                remap[i] = CLASS_TO_ID[n]
            else:
                print(f"  warning: {src.name}: class {n!r} not in deck, dropping")
        splits = find_split_dirs(src)
        if not splits:
            sys.exit(f"no image folders found under {src}")
        has_val = "val" in splits
        counts = {"train": 0, "val": 0, "skipped": 0}
        for split, dirs in splits.items():
            for d in dirs:
                for img in sorted(d.iterdir()):
                    if img.suffix.lower() not in IMG_EXT:
                        continue
                    lp = label_path_for(img)
                    lines = []
                    if lp.exists():
                        for line in lp.read_text().splitlines():
                            f = line.split()
                            if len(f) < 5 or int(f[0]) not in remap:
                                continue
                            lines.append(" ".join([str(remap[int(f[0])])] + f[1:5]))
                    if not lines and not args.keep_empty:
                        counts["skipped"] += 1
                        continue
                    target = split
                    if not has_val:
                        h = int(hashlib.md5(img.name.encode()).hexdigest(), 16) % 1000
                        target = "val" if h < args.val_frac * 1000 else "train"
                    stem = f"s{si}_{img.stem}"
                    dst = out / "images" / target / (stem + img.suffix.lower())
                    if args.link:
                        try:
                            dst.hardlink_to(img)
                        except OSError:
                            shutil.copy2(img, dst)
                    else:
                        shutil.copy2(img, dst)
                    (out / "labels" / target / (stem + ".txt")).write_text("\n".join(lines) + "\n")
                    counts[target] += 1
        print(f"{src}: train {counts['train']}, val {counts['val']}, skipped (no boxes) {counts['skipped']}")
        totals["train"] += counts["train"]
        totals["val"] += counts["val"]

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print(f"merged: train {totals['train']}, val {totals['val']} -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
