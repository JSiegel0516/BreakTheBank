"""Which data a model was trained on, and which model is live.

The labelled photos are too big for git, so they travel as a shared folder.
This module keeps a small record in git that says exactly what that folder
should hold, and marks which training run read.py is using.

    python train.py --manifest        write data/manifest.json from data/raw
    python train.py --check-data      compare data/raw with data/manifest.json
    python train.py --promote models/model_3_2026-10-14_n150

The manifest lists every export under data/raw/labeled and data/raw/unused:
where it came from (the Roboflow block in its data.yaml), when it was exported,
how many images, label files and boxes it holds, and one SHA-256 fingerprint
over every file in it. --check-data recomputes the fingerprints, so a teammate
unpacking the shared folder can tell at once whether they have the same data a
model was trained on, down to a single relabelled box.

--promote copies a run's best.pt and its small record (notes.txt, args.yaml,
results.csv) into the live folder and writes LIVE.txt there naming the run.
read.py prints that name at start-up.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import date
from pathlib import Path

import yaml

RECORD_FILES = ("notes.txt", "args.yaml", "results.csv")  # a run's small record, kept in git
SKIP_DIRS = {"preview"}                                    # autolabel's check sheets, not data
SKIP_FILES = {".DS_Store"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def describe_export(folder: Path) -> dict:
    """Source, export date, counts, and a fingerprint over every file of one export."""
    info: dict = {}
    data_yaml = folder / "data.yaml"
    if data_yaml.exists():
        config = yaml.safe_load(data_yaml.read_text()) or {}
        roboflow = config.get("roboflow") or {}
        if roboflow:
            info["source"] = {k: roboflow[k] for k in ("workspace", "project", "url", "license") if k in roboflow}
        info["classes"] = config.get("nc") or len(config.get("names") or [])
    readme = folder / "README.roboflow.txt"
    if readme.exists():
        match = re.search(r"exported via roboflow\.com on (.+)$", readme.read_text(), re.MULTILINE)
        if match:
            info["exported"] = match.group(1).strip()

    splits, label_files, boxes = {}, 0, 0
    digest, files, size = hashlib.sha256(), 0, 0
    # Sort by name parts, not Path order: Windows compares paths case-insensitively,
    # which would put README files elsewhere and change the digest.
    for path in sorted(folder.rglob("*"), key=lambda p: p.relative_to(folder).parts):
        rel = path.relative_to(folder)
        if not path.is_file() or path.name in SKIP_FILES or SKIP_DIRS & set(rel.parts):
            continue
        files += 1
        size += path.stat().st_size
        digest.update(f"{rel.as_posix()}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n".encode())
        if len(rel.parts) == 3 and rel.parts[1] == "images" and path.suffix.lower() in IMG_EXT:
            splits[rel.parts[0]] = splits.get(rel.parts[0], 0) + 1
        elif len(rel.parts) == 3 and rel.parts[1] == "labels" and path.suffix == ".txt":
            label_files += 1
            boxes += sum(1 for line in path.read_text().splitlines() if len(line.split()) >= 5)
    info.update(images=dict(sorted(splits.items())), label_files=label_files, boxes=boxes,
                files=files, bytes=size, sha256=digest.hexdigest())
    return info


def exports_in(raw_dir: Path) -> dict[str, Path]:
    """'labeled/<name>' and 'unused/<name>' for every export folder (one with a data.yaml)."""
    found = {}
    for group in ("labeled", "unused"):
        group_dir = raw_dir / group
        if group_dir.is_dir():
            for export in sorted(p for p in group_dir.iterdir() if (p / "data.yaml").exists()):
                found[f"{group}/{export.name}"] = export
    return found


def write_manifest(raw_dir: Path, path: Path) -> dict:
    """Write `path` as JSON: every export under raw_dir (see describe_export) and the background count.

    Prints one line per export. Returns the manifest that was written."""
    exports = exports_in(raw_dir)
    backgrounds_dir = raw_dir / "backgrounds"
    backgrounds = 0
    if backgrounds_dir.is_dir():
        backgrounds = sum(1 for p in backgrounds_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    manifest = {
        "written": date.today().isoformat(),
        "note": "What data/raw should hold. Written by `python train.py --manifest`, "
                "checked by `python train.py --check-data`. See data/README.md.",
        "exports": {name: describe_export(folder) for name, folder in exports.items()},
        "backgrounds": backgrounds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    for name, export in manifest["exports"].items():
        print(f"  {name}: {sum(export['images'].values())} images, {export['boxes']} boxes, {export['sha256'][:12]}")
    print(f"manifest: {len(exports)} exports -> {path}")
    return manifest


def check_manifest(raw_dir: Path, path: Path) -> list[str]:
    """Problems found comparing data/raw with the manifest. Empty means they match."""
    if not path.exists():
        return [f"no manifest at {path}; write one with: python train.py --manifest"]
    manifest = json.loads(path.read_text())
    local = exports_in(raw_dir)
    problems = []
    for name, want in manifest["exports"].items():
        folder = local.get(name)
        if folder is None:
            print(f"  {name}: MISSING")
            problems.append(f"{name} is missing from {raw_dir}")
            continue
        have = describe_export(folder)
        if have["sha256"] == want["sha256"]:
            print(f"  {name}: ok")
            continue
        if have["images"] != want["images"] or have["boxes"] != want["boxes"]:
            why = (f"manifest {want['images']} images, {want['boxes']} boxes; "
                   f"here {have['images']} images, {have['boxes']} boxes")
        else:
            why = "same counts but different contents (another export version, or edited labels)"
        print(f"  {name}: DIFFERENT: {why}")
        problems.append(f"{name} differs: {why}")
    for name in local:
        if name not in manifest["exports"]:
            print(f"  {name}: not in the manifest (new export? run --manifest after training on it)")
    if not problems:
        print(f"data matches the manifest written {manifest.get('written', '?')}")
    return problems


def promote(run_dir: Path, live_dir: Path) -> Path:
    """Make a training run the live model: copy its weights and record, write LIVE.txt."""
    run_dir, live_dir = Path(run_dir).resolve(), Path(live_dir).resolve()
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"{run_dir} has no weights/best.pt; is it a finished training run?")
    if run_dir == live_dir:
        raise SystemExit(f"{run_dir} is already the live folder")
    (live_dir / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, live_dir / "weights" / "best.pt")
    copied = ["weights/best.pt"]
    for record in RECORD_FILES:
        if (run_dir / record).exists():
            shutil.copy2(run_dir / record, live_dir / record)
            copied.append(record)
    args = {}
    if (run_dir / "args.yaml").exists():
        args = yaml.safe_load((run_dir / "args.yaml").read_text()) or {}
    lines = [
        run_dir.name,
        f"promoted {date.today().isoformat()}",
        f"best.pt sha256 {hashlib.sha256(best.read_bytes()).hexdigest()}",
    ]
    if args:
        lines.append(f"base {Path(str(args.get('model', '?'))).name}, epochs {args.get('epochs', '?')}, "
                     f"imgsz {args.get('imgsz', '?')}, batch {args.get('batch', '?')}")
    lines.append("To change the live model: python train.py --promote models/<run>")
    (live_dir / "LIVE.txt").write_text("\n".join(lines) + "\n")
    print(f"live model is now {run_dir.name}: copied {', '.join(copied)} to {live_dir}")
    return live_dir / "LIVE.txt"


def live_model(live_dir: Path) -> str | None:
    """Name of the run that is live, from LIVE.txt, or None if nothing was promoted yet."""
    live_file = Path(live_dir) / "LIVE.txt"
    return live_file.read_text().splitlines()[0].strip() if live_file.exists() else None
