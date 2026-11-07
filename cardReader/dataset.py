"""Build the one training set, data/dataset, from everything in data/raw.

  * Every subfolder of the labeled folder is a YOLO-format export (Roboflow or
    similar) with a data.yaml. Class ids are remapped by NAME to the order in
    deck.py, ignoring upper/lower case, so exports with different orderings combine cleanly. An export
    whose class names are mostly not cards (e.g. a rank/suit card-art export)
    is skipped with a warning, and so is an export whose boxes are whole cards
    rather than corner indices.
  * Images are hard-linked, not copied, so the dataset takes almost no extra
    disk space. Label files are rewritten with the remapped ids.
  * An export's own train/valid/test split is kept (test counts as val). An
    export with only a train split gets a stable hash-based val split.
  * Synthetic images from synthetic.py are added on top.
  * Each real train photo can also be added turned by 90, 180 or 270 degrees,
    with its boxes turned to match (TRAIN_ROTATED_COPIES in settings.py). These
    copies are written as new files and never go to val.
  * A real train photo whose labels include a focus class (FOCUS_CLASSES, the
    cards most often misread) is added FOCUS_EXTRA_COPIES more times, and so is
    each of its rotated copies, as prefix + f<k>_ + its name (k from 1), so the
    model sees those cards more often. The copies are identical files; training's
    random augmentation makes each pass over them differ. They never go to val.
  * class_counts reads a built dataset back and counts every card's boxes by
    source (real photos, copies, synthetic images, real val), so the cards with
    the fewest real examples can be found before choosing FOCUS_CLASSES
    (train.py --class-counts).
"""
from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

from . import synthetic
from .deck import CLASS_TO_ID, CLASSES

WHOLE_CARD_WIDTH = 0.08  # median box width (fraction of image) above this means whole-card boxes
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}  # clockwise


def build(out: Path, labeled_dir: Path, card_art_dir: Path, backgrounds_dir: Path,
          n_synthetic: int, val_frac: float, rotated_copies: tuple[int, ...] = (),
          focus_classes: tuple[str, ...] = (), focus_copies: int = 0):
    """Rebuild `out` from every export in labeled_dir plus n_synthetic synthetic images.

    Deletes `out` first. Writes images/ and labels/ for train and val, data.yaml,
    and realVal.yaml (the val images without synthetic ones). val_frac is the
    share of images held out when an export has no split of its own.
    rotated_copies: angles from ROTATIONS; every real train photo is added again
    turned by each (see add_rotated_copies).
    focus_classes, focus_copies: a real train photo holding any of these class names is
    added focus_copies more times, with its rotated copies (see add_focus_copies).
    Prints a summary line with the rotated and focus copies. Exits with a message if
    the train split ends up empty."""
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    exports = sorted(d for d in labeled_dir.iterdir() if (d / "data.yaml").exists()) if labeled_dir.is_dir() else []
    rotated = focused = 0
    for i, export in enumerate(exports):
        export_rotated, export_focused = import_export(export, out, f"e{i}_", val_frac, rotated_copies,
                                                       focus_classes, focus_copies)
        rotated += export_rotated
        focused += export_focused
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
    print(f"dataset: train {counts['train']} ({rotated} rotated copies, {focused} focus copies), "
          f"val {counts['val']} -> {out}")


def import_export(export: Path, out: Path, prefix: str, val_frac: float, rotated_copies: tuple[int, ...] = (),
                  focus_classes: tuple[str, ...] = (), focus_copies: int = 0) -> tuple[int, int]:
    """Add one labelled export to the dataset at `out`, with class ids remapped to deck.py.

    File names get `prefix` so exports cannot collide. Skips the export with a
    message if most of its classes are not cards or its boxes are whole cards.
    Images with no card boxes left after remapping are skipped and counted.
    Each image that lands in train also gets a copy per angle in rotated_copies,
    and, when its labels hold a class in focus_classes, focus_copies more copies of
    itself and of each rotated copy. Returns (rotated copies, focus copies) written."""
    focus_ids = {str(CLASS_TO_ID[name]) for name in focus_classes}
    names = yaml.safe_load((export / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    names = [str(n).upper() for n in names]  # some exports write suits in lower case: 10h, Ks
    remap = {i: CLASS_TO_ID[n] for i, n in enumerate(names) if n in CLASS_TO_ID}
    if len(remap) < len(names) / 2:
        print(f"skip {export.name}: its classes are not card names ({names[:5]} ...)")
        return 0, 0
    widths = []
    for label_path in export.glob("*/labels/*.txt"):
        boxes = [line.split() for line in label_path.read_text().splitlines()]
        widths += [float(fields[3]) for fields in boxes if len(fields) == 5]
    median_width = sorted(widths)[len(widths) // 2] if widths else 0.0
    if median_width > WHOLE_CARD_WIDTH:
        print(f"skip {export.name}: boxes are whole cards, not corner indices")
        return 0, 0
    dropped = [n for n in names if n not in CLASS_TO_ID]
    if dropped:
        print(f"  {export.name}: dropping classes not in the deck: {dropped}")

    split_names = ("train", "valid", "val", "test")
    splits = {split: export / split / "images" for split in split_names if (export / split / "images").is_dir()}
    has_val = any(split != "train" for split in splits)
    counts = {"train": 0, "val": 0, "rotated": 0, "focus": 0, "empty": 0}
    for split, img_dir in splits.items():
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            label_path = export / split / "labels" / (img.stem + ".txt")
            lines = []
            if label_path.exists():
                for line in label_path.read_text().splitlines():
                    fields = line.split()
                    if len(fields) >= 5 and int(fields[0]) in remap:
                        lines.append(" ".join([str(remap[int(fields[0])])] + fields[1:5]))
            if not lines:
                counts["empty"] += 1
                continue
            if has_val:
                target = "train" if split == "train" else "val"
            else:
                bucket = int(hashlib.md5(img.name.encode()).hexdigest(), 16) % 1000
                target = "val" if bucket < val_frac * 1000 else "train"
            stem = prefix + img.stem
            _link(img, out / "images" / target / (stem + img.suffix.lower()))
            (out / "labels" / target / (stem + ".txt")).write_text("\n".join(lines) + "\n")
            counts[target] += 1
            if target == "train":
                counts["rotated"] += add_rotated_copies(img, lines, out, prefix, rotated_copies)
                if any(line.split()[0] in focus_ids for line in lines):
                    names_in_train = [img.stem] + [f"r{degrees}_{img.stem}" for degrees in rotated_copies]
                    counts["focus"] += add_focus_copies(out, prefix, names_in_train, img.suffix.lower(),
                                                        focus_copies)
    print(f"  {export.name}: train {counts['train']}, val {counts['val']}, rotated copies {counts['rotated']}, "
          f"focus copies {counts['focus']}, skipped (no boxes) {counts['empty']}")
    return counts["rotated"], counts["focus"]


def _link(src: Path, dst: Path):
    """dst as a hard link to src, or a copy where the file system cannot link."""
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def add_focus_copies(out: Path, prefix: str, names: list[str], suffix: str, copies: int) -> int:
    """Add `copies` more of each train image prefix + name + suffix already in out, with its label,
    as prefix + f<k>_ + name for k from 1. A name not in train (a rotated copy that could not be
    read) is passed over. Returns the number of copies written."""
    images, labels = out / "images" / "train", out / "labels" / "train"
    written = 0
    for name in names:
        image = images / f"{prefix}{name}{suffix}"
        if not image.exists():
            continue
        label = (labels / f"{prefix}{name}.txt").read_text()
        for k in range(1, copies + 1):
            _link(image, images / f"{prefix}f{k}_{name}{suffix}")
            (labels / f"{prefix}f{k}_{name}.txt").write_text(label)
            written += 1
    return written


def rotate_box(cx: float, cy: float, w: float, h: float, degrees: int) -> tuple[float, float, float, float]:
    """The YOLO box (centre and size as fractions of the image) on the same image turned `degrees` clockwise."""
    if degrees == 90:
        return 1 - cy, cx, h, w
    if degrees == 180:
        return 1 - cx, 1 - cy, w, h
    if degrees == 270:
        return cy, 1 - cx, h, w
    raise ValueError(f"rotation must be one of {sorted(ROTATIONS)}, not {degrees}")


def add_rotated_copies(img: Path, lines: list[str], out: Path, prefix: str, rotations: tuple[int, ...]) -> int:
    """Write `img` turned clockwise by each angle in `rotations` to out's train split, as a new file named
    prefix + r<angle>_ + the image's name, with its label `lines` turned to match.
    Returns the number of copies written: 0 when the image cannot be read."""
    if not rotations:
        return 0
    image = cv2.imread(str(img))
    if image is None:
        print(f"  cannot read {img.name}: no rotated copies")
        return 0
    for degrees in rotations:
        stem = f"{prefix}r{degrees}_{img.stem}"
        cv2.imwrite(str(out / "images" / "train" / (stem + img.suffix.lower())), cv2.rotate(image, ROTATIONS[degrees]))
        turned = []
        for line in lines:
            cls, *box = line.split()
            new_box = rotate_box(*(float(v) for v in box), degrees)
            turned.append(" ".join([cls, *(f"{v:.6f}" for v in new_box)]))
        (out / "labels" / "train" / (stem + ".txt")).write_text("\n".join(turned) + "\n")
    return len(rotations)


@dataclass(frozen=True)
class ClassCount:
    """How often one card appears in a built dataset, in boxes: corner indices, or whole cards for BACK."""
    name: str
    photos: int     # real train photos holding it: what a focus copy multiplies, with every other card in them
    boxes: int      # boxes in those photos
    copies: int     # boxes in their rotated and focus copies
    synthetic: int  # boxes in synthetic train images
    val: int        # boxes in the real val photos, the exam


def class_counts(dataset_dir: Path) -> list[ClassCount]:
    """Count every class in a built dataset (dataset_dir/labels/train and val), in CLASSES order.

    A train label file is a synthetic image (synth_), a rotated or focus copy (r<angle>_ or f<k>_
    after the export prefix), or else a real photo. Val counts real photos only, the exam the models
    are compared on. Raises FileNotFoundError when there is no built dataset."""
    labels = dataset_dir / "labels"
    if not (labels / "train").is_dir():
        raise FileNotFoundError(f"no built dataset at {dataset_dir}")
    fields = ("photos", "boxes", "copies", "synthetic", "val")
    tallies = {name: dict.fromkeys(fields, 0) for name in CLASSES}
    for split in ("train", "val"):
        for label_path in sorted((labels / split).glob("*.txt")):
            source = _source(label_path.stem)
            if split == "val":
                if source != "real":
                    continue
                field = "val"
            else:
                field = {"real": "boxes", "copy": "copies", "synthetic": "synthetic"}[source]
            ids = [int(line.split()[0]) for line in label_path.read_text().splitlines() if line.strip()]
            names = [CLASSES[i] for i in ids if 0 <= i < len(CLASSES)]
            for name in names:
                tallies[name][field] += 1
            if field == "boxes":
                for name in set(names):
                    tallies[name]["photos"] += 1
    return [ClassCount(name, **tallies[name]) for name in CLASSES]


def _source(stem: str) -> str:
    """Where a dataset file came from, by its name: synth_ images are synthetic; after the export
    prefix (e<i>_), r<angle>_ and f<k>_ mark rotated and focus copies; anything else is a real photo."""
    if stem.startswith("synth_"):
        return "synthetic"
    after_export = stem.split("_", 1)[1] if re.match(r"e\d+_", stem) else stem
    return "copy" if re.match(r"[rf]\d+_", after_export) else "real"


def class_count_table(counts: list[ClassCount], fewest: int = 8) -> str:
    """The counts as text, the fewest real train boxes first. Classes below the median are marked *.
    Ends with the totals, the median, and the `fewest` cards with the fewest real train boxes as a
    ready `benchmark.py --confusions --focus` list, to set their count against their mistakes."""
    ordered = sorted(counts, key=lambda c: (c.boxes, c.name))
    boxes = [c.boxes for c in ordered]
    middle = len(boxes) // 2
    median = boxes[middle] if len(boxes) % 2 else (boxes[middle - 1] + boxes[middle]) / 2
    rows = [["class", "photos", "boxes", "copies", "synthetic", "val", ""]]
    for c in ordered:
        rows.append([c.name, str(c.photos), str(c.boxes), str(c.copies), str(c.synthetic), str(c.val),
                     "*" if c.boxes < median else ""])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for row in rows:
        cells = [cell.rjust(w) if i else cell.ljust(w) for i, (cell, w) in enumerate(zip(row, widths, strict=True))]
        lines.append(" ".join(cells).rstrip())
    totals = (f"real train boxes {sum(boxes)}, in copies {sum(c.copies for c in counts)}, synthetic "
              f"{sum(c.synthetic for c in counts)}, real val {sum(c.val for c in counts)}")
    few = ordered[:fewest]
    return (
        "\n".join(lines) + "\n\n" + totals
        + f"\nmedian {median:g} real train boxes per class; * marks the classes below it\n"
        + "fewest: " + ", ".join(f"{c.name} {c.boxes}" for c in few) + "\n"
        + "python benchmark.py --confusions --focus " + ",".join(c.name for c in few) + "\n")
