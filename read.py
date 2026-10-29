"""The card reader. Point it at the table and it lists cards in the order they are placed.

    python read.py                                   # camera and model from settings.py
    python read.py --source 1                        # another camera or a capture card
    python read.py --source round1.mp4 --log round1.csv --no-display
    python read.py --list-cameras                    # which number is which camera?

Per frame:  camera -> detector -> tracker -> table (seats, hands, Hi-Lo count) -> overlay

Every event (placed, hidden, returned, renamed, duplicate, round over) is printed
as it happens, and appended to --log as CSV if given. Without a trained model it runs
in capture mode: it shows the live feed and lets you press 's' to save frames
for labelling.

Keys: 1-7 number of players | e edit zones with the mouse (e again saves them) | l legend
      r new round | Shift+R new shoe (count back to 0) | s save frame to data/raw/frames | q quit

This file only owns the command line, the camera, and the window. What happens
per frame and per key is card_reader/reader.py; the overlay is card_reader/hud.py.
"""
from __future__ import annotations

import argparse
import csv
import time
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

import settings
from card_reader.camera import Camera, parse_source, snapshot_cameras
from card_reader.hud import Hud
from card_reader.provenance import live_model
from card_reader.reader import LOG_HEADER, Reader
from card_reader.seats import MAX_PLAYERS, SeatLayout
from card_reader.tracker import CardTracker, TrackerConfig

WINDOW = "card reader"


def list_cameras():
    """Print every camera number and show one window with a labelled snapshot from each."""
    found = snapshot_cameras()
    if not found:
        print("no cameras found (check System Settings > Privacy & Security > Camera)")
        return
    tiles = []
    for i, frame in found:
        if frame is None:
            print(f"--source {i}: opened but sent no picture (permission, or the device is asleep)")
            frame = np.zeros((360, 640, 3), np.uint8)
            note = "no picture"
        else:
            h, w = frame.shape[:2]
            # almost no edges: lens blocked, out of focus, or pointing at a plain wall or ceiling
            blank = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 20
            note = f"{w}x{h}" + ("  nearly blank: lens blocked or no detail in view" if blank else "")
            print(f"--source {i}: {note}")
        tile = cv2.resize(frame, (640, round(640 * frame.shape[0] / frame.shape[1])))
        cv2.rectangle(tile, (0, 0), (640, 44), (25, 25, 25), -1)
        cv2.putText(tile, f"--source {i}   {note}", (12, 31), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 220, 255), 2,
                    cv2.LINE_AA)
        tiles.append(tile)
    height = max(tile.shape[0] for tile in tiles)
    padded = [cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT) for tile in tiles]
    sheet = np.hstack(padded)
    cv2.imshow("cameras (press any key to close)", sheet)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("then run: python read.py --source <number>   (or set CAMERA_SOURCE in settings.py)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=settings.CAMERA_SOURCE, help="camera index, video file, or stream URL")
    ap.add_argument("--model", default=settings.MODEL_PATH,
                    help="trained .pt weights; capture mode if the file is missing")
    ap.add_argument("--imgsz", type=int, default=settings.INPUT_SIZE)
    ap.add_argument("--conf", type=float, default=settings.CONFIDENCE)
    ap.add_argument("--device", default=None, help="mps, cuda, or cpu (auto if omitted)")
    ap.add_argument("--width", type=int, default=settings.CAMERA_WIDTH)
    ap.add_argument("--height", type=int, default=settings.CAMERA_HEIGHT)
    ap.add_argument("--players", type=int, default=None,
                    help=f"player seats, 1 to {MAX_PLAYERS} "
                         "(default: the count used last, else PLAYERS in settings.py)")
    ap.add_argument("--log", default=None, help="append every event to this CSV file")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--list-cameras", action="store_true", help="show a snapshot from every camera, then exit")
    args = ap.parse_args()
    try:
        settings.validate(settings)
    except ValueError as error:
        raise SystemExit(error) from None
    if args.list_cameras:
        list_cameras()
        return

    detector = None
    if Path(args.model).exists():
        from card_reader.detector import Detector
        detector = Detector(args.model, args.imgsz, args.conf, args.device)
        live = live_model(settings.PROJECT_ROOT / settings.LIVE_MODEL_DIR)
        is_live = Path(args.model).resolve() == (settings.PROJECT_ROOT / settings.MODEL_PATH).resolve()
        print(f"model {args.model}" + (f" (live model: {live})" if live and is_live else ""))
    else:
        print(f"no model at {args.model}: running in capture mode (press s to save frames)")

    cam = Camera(parse_source(args.source), args.width, args.height)
    W, H = cam.size
    print(f"source {args.source} at {W}x{H}")

    seats_file = settings.PROJECT_ROOT / settings.SEATS_FILE
    layout = SeatLayout.load(seats_file, W, H, args.players)
    if layout is not None:
        print(f"loaded your saved {layout.players}-player zones from {settings.SEATS_FILE}")
    else:
        layout = SeatLayout(args.players or settings.PLAYERS, W, H)

    with ExitStack() as cleanup:  # closes the log file, if any, on the way out
        log = None
        if args.log:
            new_file = not Path(args.log).exists()
            log = csv.writer(cleanup.enter_context(open(args.log, "a", newline="")))
            if new_file:
                log.writerow(LOG_HEADER)

        tracker = CardTracker(TrackerConfig.from_settings(settings, W))
        hud = Hud()
        reader = Reader(tracker, layout, hud, seats_file, settings.PROJECT_ROOT / settings.FRAMES_DIR, log=log)
        if not args.no_display:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  # resizable, so a 4K feed fits on a laptop screen
            cv2.setMouseCallback(WINDOW, reader.editor.on_mouse)
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
                reader.process(frame, dets)

                now = time.time()
                fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6)
                t_last = now
                if args.no_display:
                    continue
                cv2.imshow(WINDOW, hud.draw(frame, dets, tracker.round, fps, reader.layout, reader.count,
                                            capture_mode=detector is None, editor=reader.editor))
                if not reader.handle_key(cv2.waitKey(1) & 0xFF, frame):
                    break
        finally:
            cam.release()
            cv2.destroyAllWindows()
            count = reader.count
            print("counted this shoe:", " ".join(tracker.shoe_seen) or "-")
            print(f"running count {count.running:+d}, true count {count.true:+.1f}, {count.cards_left} cards left")


if __name__ == "__main__":
    main()
