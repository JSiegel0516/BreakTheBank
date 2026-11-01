"""Does preprocessing help the detector in dim or bright light, and what does it cost?

    python benchmark.py                          # every real validation image, all levels and modes
    python benchmark.py --limit 20               # a quick run on the first 20 images
    python benchmark.py --levels 2.5 1.0 --modes off auto --csv bench.csv
    python benchmark.py --data /path/to/data/dataset/realVal.yaml --imgsz 1024
    python benchmark.py --confusions             # which names does the live model get wrong, and on what boxes?
    python benchmark.py --confusions --model models/<run>/weights/best.pt --csv model5.csv

Runs the live model (MODEL_PATH, at --imgsz and CONFIDENCE from settings.py) over
the real validation photos listed in data/dataset/realVal.yaml (written by
train.py --rebuild). Each photo is first relit to every brightness level, then
preprocessed by every mode (cardReader/preprocess.py) and passed to the detector.

A level is a gamma applied to the photo: each pixel level v (0 to 255) becomes
255 * (v / 255) ** level, so 3.5 is very dark, 2.5 much darker, 1.8 darker,
1.0 unchanged and 0.6 brighter. This is the same convention preprocess.py uses. Brightening this
way cannot clip new highlights, so it imitates a washed-out picture rather than
glare.

Scoring follows docs/results.md: a labelled corner index is found when a box's
centre lies within half the labelled box's longer side of its centre (each box
matches one corner at most, nearest first), and right when the box also has the
corner's name. A box matching no labelled corner is extra. Times are the mean
milliseconds per image for preprocessing and for the detector.

Prints one row per level and mode; --csv also writes them to a file.

--confusions instead runs the model once on each photo, unchanged (level 1,
mode off), and writes one CSV row per wrong-name box: image, export prefix,
true name, predicted name, confidence, and the labelled box's width, height
and aspect (width / height). The CSV goes to --csv, or by default to
confusions.csv in the model's run folder. It prints the totals (labelled,
found, wrong names, missed), the rank swaps with counts, and the wrong-name
rate (wrong / found) by aspect: upright below 0.8, diagonal 0.8 to 1.25,
sideways above 1.25; and by box width tercile.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

import settings
from cardReader.cards import Detection
from cardReader.deck import BACK, RANKS, rank_of
from cardReader.preprocess import MODES, PreprocessConfig, enhance

LEVELS = (3.5, 2.5, 1.8, 1.0, 0.6)
MATCH_FRACTION = 0.5  # a box matches a corner when its centre is this share of the corner's longer side away, or less
COLUMNS = ["level", "mode", "labelled", "found", "found %", "right", "right %", "extra", "prep ms", "detect ms"]
CONFUSION_COLUMNS = ["image", "export", "true", "predicted", "confidence", "width", "height", "aspect"]
UPRIGHT_ASPECT = 0.8     # a labelled box narrower than this share of its height is an upright corner
SIDEWAYS_ASPECT = 1.25   # wider than this, a sideways one; between the two, diagonal


@dataclass(frozen=True)
class Label:
    """One labelled corner index, in pixels: its name, centre, width and height."""
    name: str
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class Tally:
    """Totals for one level and mode over every image."""
    images: int = 0
    labelled: int = 0
    found: int = 0
    right: int = 0
    extra: int = 0
    preprocess_total: float = 0.0  # milliseconds
    detect_total: float = 0.0

    @property
    def preprocess_ms(self) -> float:
        return self.preprocess_total / max(self.images, 1)

    @property
    def detect_ms(self) -> float:
        return self.detect_total / max(self.images, 1)


def relight(image: np.ndarray, level: float) -> np.ndarray:
    """The image at another brightness: gamma `level` above 1 darkens, below 1 brightens, 1 returns it as is."""
    if level == 1.0:
        return image
    table = np.round(255 * (np.arange(256) / 255) ** level).astype(np.uint8)
    return cv2.LUT(image, table)


def match(labels: list[Label], dets: list[Detection]) -> list[tuple[int, int]]:
    """(label index, box index) pairs: each labelled corner with the box whose centre is nearest it and
    within reach. Closest pairs are matched first, one to one; unmatched corners and boxes are left out."""
    pairs = []
    for i, label in enumerate(labels):
        reach = MATCH_FRACTION * max(label.w, label.h)
        for j, d in enumerate(dets):
            distance = float(np.hypot(d.cx - label.cx, d.cy - label.cy))
            if distance <= reach:
                pairs.append((distance, i, j))
    matched_labels, matched_dets, matches = set(), set(), []
    for _, i, j in sorted(pairs):
        if i in matched_labels or j in matched_dets:
            continue
        matched_labels.add(i)
        matched_dets.add(j)
        matches.append((i, j))
    return matches


def score(labels: list[Label], dets: list[Detection]) -> tuple[int, int, int]:
    """(found, right, extra) for one image: corners with a box near their centre, those whose box
    has their name, and boxes near no corner."""
    matches = match(labels, dets)
    right = sum(dets[j].label == labels[i].name for i, j in matches)
    return len(matches), right, len(dets) - len(matches)


def run(samples: Iterable[tuple[np.ndarray, list[Label]]], detector: Callable[[np.ndarray], list[Detection]],
        levels=LEVELS, modes=MODES, config: PreprocessConfig | None = None) -> dict[tuple[float, str], Tally]:
    """Relight each image to every level, preprocess it with every mode, detect, and score.
    Returns a Tally per (level, mode), in the order levels then modes."""
    config = config or PreprocessConfig.from_settings(settings)
    tallies = {(level, mode): Tally() for level in levels for mode in modes}
    for image, labels in samples:
        for level in levels:
            lit = relight(image, level)
            for mode in modes:
                start = time.perf_counter()
                enhanced = enhance(lit, mode, config)
                prepared = time.perf_counter()
                dets = detector(enhanced)
                done = time.perf_counter()
                found, right, extra = score(labels, dets)
                tally = tallies[level, mode]
                tally.images += 1
                tally.labelled += len(labels)
                tally.found, tally.right, tally.extra = tally.found + found, tally.right + right, tally.extra + extra
                tally.preprocess_total += 1000 * (prepared - start)
                tally.detect_total += 1000 * (done - prepared)
    return tallies


def rows(tallies: dict[tuple[float, str], Tally]) -> list[list[str]]:
    out = []
    for (level, mode), t in tallies.items():
        labelled = max(t.labelled, 1)
        out.append([f"{level:g}", mode, str(t.labelled), str(t.found), f"{100 * t.found / labelled:.1f}%",
                    str(t.right), f"{100 * t.right / labelled:.1f}%", str(t.extra),
                    f"{t.preprocess_ms:.1f}", f"{t.detect_ms:.1f}"])
    return out


def table(tallies: dict[tuple[float, str], Tally]) -> str:
    """A plain text table: a header, then one row per level and mode."""
    return _aligned([COLUMNS, *rows(tallies)])


def _aligned(lines: list[list[str]]) -> str:
    widths = [max(len(line[i]) for line in lines) for i in range(len(lines[0]))]
    return "\n".join("  ".join(cell.rjust(width) for cell, width in zip(line, widths, strict=True))
                     for line in lines)


def _label_path(image: Path) -> Path:
    """YOLO's layout: .../images/<split>/x.jpg has its labels in .../labels/<split>/x.txt."""
    parts = list(image.parts)
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def val_images(data_yaml: Path, limit: int | None = None) -> Iterator[tuple[np.ndarray, list[Label]]]:
    """(image, labelled corners) for each validation image of a YOLO dataset yaml, read one at a time.
    See val_samples."""
    for _, image, labels in val_samples(data_yaml, limit):
        yield image, labels


def val_samples(data_yaml: Path, limit: int | None = None) -> Iterator[tuple[Path, np.ndarray, list[Label]]]:
    """(path, image, labelled corners) for each validation image of a YOLO dataset yaml, read one at a time.

    The yaml's `val` is a text file of image paths (as realVal.yaml has) or a folder.
    limit: only the first this many images. Raises SystemExit if the yaml is missing."""
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise SystemExit(f"no {data_yaml}: build the dataset with python train.py --rebuild, or pass --data")
    spec = yaml.safe_load(data_yaml.read_text())
    names = spec["names"]
    val = Path(spec["path"]) / spec["val"]
    if val.is_dir():
        paths = sorted(p for p in val.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    else:
        paths = [Path(line) for line in val.read_text().split("\n") if line.strip()]
    for path in paths[:limit]:
        image = cv2.imread(str(path))
        H, W = image.shape[:2]
        labels = []
        label_file = _label_path(path)
        if label_file.exists():
            for line in label_file.read_text().splitlines():
                cls, cx, cy, w, h = line.split()[:5]
                labels.append(Label(names[int(cls)], float(cx) * W, float(cy) * H, float(w) * W, float(h) * H))
        yield path, image, labels


@dataclass(frozen=True)
class Corner:
    """One labelled corner on one image and what the model called it: predicted and conf are None when missed."""
    image: str
    label: Label
    predicted: str | None
    conf: float | None


def read_corners(samples: Iterable[tuple[Path | str, np.ndarray, list[Label]]],
                 detector: Callable[[np.ndarray], list[Detection]]) -> list[Corner]:
    """Run the detector once on each unchanged image and pair every labelled corner with its box (see match).
    Boxes near no corner are dropped."""
    corners = []
    for path, image, labels in samples:
        dets = detector(image)
        box_of = dict(match(labels, dets))
        for i, label in enumerate(labels):
            if i in box_of:
                box = dets[box_of[i]]
                corners.append(Corner(str(path), label, box.label, box.conf))
            else:
                corners.append(Corner(str(path), label, None, None))
    return corners


def aspect_bucket(label: Label) -> str:
    """"upright" when the box is tall (width / height below 0.8), "sideways" when wide (above 1.25),
    otherwise "diagonal": a corner index tilted near 45 degrees has a squarish box."""
    aspect = label.w / label.h
    if aspect < UPRIGHT_ASPECT:
        return "upright"
    if aspect > SIDEWAYS_ASPECT:
        return "sideways"
    return "diagonal"


def wrong_name_rows(corners: list[Corner]) -> list[list[str]]:
    """One CONFUSION_COLUMNS row per corner whose box has another name. The export prefix is the
    file name up to its first underscore (e0, e1, ... from dataset.py)."""
    out = []
    for c in corners:
        if c.predicted is None or c.predicted == c.label.name:
            continue
        name = Path(c.image).name
        out.append([name, name.split("_")[0], c.label.name, c.predicted, f"{c.conf:.3f}",
                    f"{c.label.w:.1f}", f"{c.label.h:.1f}", f"{c.label.w / c.label.h:.2f}"])
    return out


def rank_swaps(corners: list[Corner]) -> list[tuple[str, int]]:
    """("6/9", count) for each pair of ranks read as one another, either way, most common first.
    Mistakes of suit only, and those involving a face-down card, are left out."""
    swaps = Counter()
    for c in corners:
        if c.predicted is None or BACK in (c.predicted, c.label.name):
            continue
        ranks = (rank_of(c.label.name), rank_of(c.predicted))
        if ranks[0] != ranks[1]:
            swaps["/".join(sorted(ranks, key=RANKS.index))] += 1
    return sorted(swaps.items(), key=lambda item: (-item[1], item[0]))


def width_terciles(corners: list[Corner]) -> dict[str, list[Corner]]:
    """The corners split into thirds by labelled box width: "narrow", "middle", "wide"."""
    if not corners:
        return {}
    widths = sorted(c.label.w for c in corners)
    narrow_edge = widths[(len(widths) - 1) // 3]
    middle_edge = widths[(2 * len(widths) - 1) // 3]
    groups = {"narrow": [], "middle": [], "wide": []}
    for c in corners:
        if c.label.w <= narrow_edge:
            groups["narrow"].append(c)
        elif c.label.w <= middle_edge:
            groups["middle"].append(c)
        else:
            groups["wide"].append(c)
    return groups


def _group_row(name: str, corners: list[Corner]) -> list[str]:
    found = [c for c in corners if c.predicted is not None]
    wrong = sum(c.predicted != c.label.name for c in found)
    return [name, str(len(corners)), str(len(found)), str(wrong), f"{100 * wrong / max(len(found), 1):.1f}%",
            str(len(corners) - len(found))]


def confusion_summary(corners: list[Corner]) -> str:
    """Totals, rank swaps with counts, and the wrong-name rate (wrong / found) by box aspect and by width tercile."""
    found = [c for c in corners if c.predicted is not None]
    wrong = sum(c.predicted != c.label.name for c in found)
    parts = [f"{len(corners)} labelled, {len(found)} found, {wrong} wrong names, {len(corners) - len(found)} missed"]
    swaps = rank_swaps(corners)
    parts.append("rank swaps (either way):\n" + ("\n".join(f"  {pair}  {count}" for pair, count in swaps) or "  none"))

    by_aspect = {bucket: [c for c in corners if aspect_bucket(c.label) == bucket]
                 for bucket in ("upright", "diagonal", "sideways")}
    aspect_lines = [["aspect", "labelled", "found", "wrong", "wrong %", "missed"]]
    aspect_lines += [_group_row(bucket, group) for bucket, group in by_aspect.items()]
    parts.append(_aligned(aspect_lines))

    width_lines = [["width", "px", "labelled", "found", "wrong", "wrong %", "missed"]]
    for tercile, group in width_terciles(corners).items():
        span = f"{min(c.label.w for c in group):.0f}-{max(c.label.w for c in group):.0f}" if group else "-"
        row = _group_row(tercile, group)
        width_lines.append([row[0], span, *row[1:]])
    parts.append(_aligned(width_lines))
    return "\n\n".join(parts)


def default_confusions_csv(model: str) -> Path:
    """confusions.csv in the model's run folder (the folder holding weights/), or beside the model."""
    folder = Path(model).parent
    return (folder.parent if folder.name == "weights" else folder) / "confusions.csv"


def list_confusions(args: argparse.Namespace, detector: Callable[[np.ndarray], list[Detection]]):
    """Read every validation photo once, print confusion_summary, and write the wrong-name rows to a CSV."""
    print(f"model {args.model} at imgsz {args.imgsz}, unchanged photos")
    corners = read_corners(val_samples(Path(args.data), args.limit), detector)
    if not corners:
        raise SystemExit(f"no labelled corners in the validation images of {args.data}")
    print(f"{len({c.image for c in corners})} images")
    print(confusion_summary(corners))
    path = Path(args.csv) if args.csv else default_confusions_csv(args.model)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CONFUSION_COLUMNS)
        writer.writerows(wrong_name_rows(corners))
    print(f"wrong-name boxes written to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(settings.PROJECT_ROOT / settings.DATASET_DIR / "realVal.yaml"),
                    help="dataset yaml whose val images are scored (default: the real validation photos)")
    ap.add_argument("--model", default=settings.MODEL_PATH)
    ap.add_argument("--imgsz", type=int, default=settings.INPUT_SIZE)
    ap.add_argument("--device", default=None, help="mps, cuda, or cpu (auto if omitted)")
    ap.add_argument("--levels", type=float, nargs="+", default=list(LEVELS),
                    help="brightness gammas: above 1 darker, below 1 brighter")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--limit", type=int, default=None, help="only the first this many images, for a quick run")
    ap.add_argument("--csv", default=None, help="also write the rows to this CSV file")
    ap.add_argument("--confusions", action="store_true",
                    help="list every wrong-name box on the unchanged photos instead (CSV default: the run folder)")
    args = ap.parse_args()
    try:
        settings.validate(settings)
    except ValueError as error:
        raise SystemExit(error) from None
    if any(level <= 0 for level in args.levels):
        raise SystemExit("--levels must all be above 0")

    from cardReader.detector import Detector
    detector = Detector(args.model, args.imgsz, settings.CONFIDENCE, args.device)
    if args.confusions:
        list_confusions(args, detector)
        return
    samples = val_images(Path(args.data), args.limit)
    first = next(samples, None)
    if first is None:
        raise SystemExit(f"no validation images listed in {args.data}")
    detector(first[0])  # the first call loads the model onto the device; keep it out of the times
    print(f"model {args.model} at imgsz {args.imgsz}, "
          f"levels {' '.join(f'{level:g}' for level in args.levels)}, modes {' '.join(args.modes)}")
    tallies = run(itertools.chain([first], samples), detector, args.levels, args.modes)
    print(f"{next(iter(tallies.values())).images} images")
    print(table(tallies))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(rows(tallies))
        print(f"rows written to {args.csv}")


if __name__ == "__main__":
    main()
