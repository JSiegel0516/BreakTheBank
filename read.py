"""The card reader. Point it at the table and it lists cards in the order they are placed.

    python read.py                                   # camera and model from settings.py
    python read.py --source 1                        # a capture card
    python read.py --source round1.mp4 --log round1.csv --no-display

Per frame:  camera -> detector -> tracker -> ordered list of cards

Every placement and removal is printed as it happens, and appended to --log as
CSV if given. Without a trained model it runs in capture mode: it shows the live
feed and lets you press 's' to save frames for labelling.

Keys: q quit | s save frame to data/raw/frames | r reset count (numbering restarts at 1)
The overlay is drawn by cardReader/hud.py.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

import settings
from cardReader.camera import Camera, parse_source
from cardReader.hud import Hud
from cardReader.tracker import CardTracker

WINDOW = "card reader"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=settings.CAMERA_SOURCE, help="camera index, video file, or stream URL")
    ap.add_argument("--model", default=settings.MODEL_PATH, help="trained .pt weights; capture mode if the file is missing")
    ap.add_argument("--imgsz", type=int, default=settings.INPUT_SIZE)
    ap.add_argument("--conf", type=float, default=settings.CONFIDENCE)
    ap.add_argument("--device", default=None, help="mps, cuda, or cpu (auto if omitted)")
    ap.add_argument("--width", type=int, default=settings.CAMERA_WIDTH)
    ap.add_argument("--height", type=int, default=settings.CAMERA_HEIGHT)
    ap.add_argument("--log", default=None, help="append placements and removals to this CSV file")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    detector = None
    if Path(args.model).exists():
        from cardReader.detector import Detector
        detector = Detector(args.model, args.imgsz, args.conf, args.device)
    else:
        print(f"no model at {args.model}: running in capture mode (press s to save frames)")

    cam = Camera(parse_source(args.source), args.width, args.height)
    W, H = cam.size
    print(f"source {args.source} at {W}x{H}")
    tracker = CardTracker(settings.CONFIRM_FRAMES, settings.DROP_FRAMES, settings.MATCH_DISTANCE * W)
    frames_dir = settings.PROJECT_ROOT / settings.FRAMES_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)

    log = None
    if args.log:
        new_file = not Path(args.log).exists()
        log_file = open(args.log, "a", newline="")
        log = csv.writer(log_file)
        if new_file:
            log.writerow(["time", "frame", "event", "order", "card", "x", "y"])

    hud = Hud()
    if not args.no_display:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  # resizable, so a 4K feed fits on a laptop screen
    t_last, fps = time.time(), 0.0
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                if cam.is_file:
                    break
                time.sleep(0.01)
                continue

            dets = detector(frame) if detector else []
            for e in tracker.update(dets):
                print(e)
                if log:
                    log.writerow([f"{time.time():.3f}", e.frame, e.kind, e.card.order, e.card.label,
                                  round(e.card.x), round(e.card.y)])

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6)
            t_last = now

            if args.no_display:
                continue
            cv2.imshow(WINDOW, hud.draw(frame, dets, tracker.table(), fps, capture_mode=detector is None))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("s"):
                p = frames_dir / f"frame_{int(time.time() * 1000)}.jpg"
                cv2.imwrite(str(p), frame)
                print("saved", p)
                hud.flash(f"frame saved: {p.name}")
            elif k == ord("r"):
                tracker.reset()
                print("--- count reset ---")
                hud.flash("count reset: numbering restarts at 1")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        if log:
            log_file.close()
        print("placed in order:", " ".join(c.label for c in tracker.history) or "-")


if __name__ == "__main__":
    main()
