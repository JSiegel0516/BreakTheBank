"""The card reader. Point it at the table and it lists cards in the order they are placed.

    python read.py                                   # camera and model from settings.py
    python read.py --source 1                        # another camera or a capture card
    python read.py --source round1.mp4 --log round1.csv --no-display
    python read.py --list-cameras                    # which number is which camera?

Per frame:  camera -> detector -> tracker -> seats, hand totals, Hi-Lo count -> overlay

Every event (placed, hidden, returned, round over) is printed as it happens, and
appended to --log as CSV if given. Without a trained model it runs in capture
mode: it shows the live feed and lets you press 's' to save frames for labelling.

Keys: 1-7 number of players | e edit zones with the mouse (e again saves them) | l legend
      r new round | Shift+R new shoe (count back to 0) | s save frame to data/raw/frames | q quit
The overlay is drawn by cardReader/hud.py, the seat layout by cardReader/seats.py.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

import settings
from cardReader.blackjack import shoe_count
from cardReader.camera import Camera, parse_source, snapshot_cameras
from cardReader.hud import Hud
from cardReader.seats import MAX_PLAYERS, SeatLayout, ZoneEditor
from cardReader.tracker import CardTracker

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
        cv2.putText(tile, f"--source {i}   {note}", (12, 31), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    height = max(t.shape[0] for t in tiles)
    sheet = np.hstack([cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles])
    cv2.imshow("cameras (press any key to close)", sheet)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("then run: python read.py --source <number>   (or set CAMERA_SOURCE in settings.py)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=settings.CAMERA_SOURCE, help="camera index, video file, or stream URL")
    ap.add_argument("--model", default=settings.MODEL_PATH, help="trained .pt weights; capture mode if the file is missing")
    ap.add_argument("--imgsz", type=int, default=settings.INPUT_SIZE)
    ap.add_argument("--conf", type=float, default=settings.CONFIDENCE)
    ap.add_argument("--device", default=None, help="mps, cuda, or cpu (auto if omitted)")
    ap.add_argument("--width", type=int, default=settings.CAMERA_WIDTH)
    ap.add_argument("--height", type=int, default=settings.CAMERA_HEIGHT)
    ap.add_argument("--players", type=int, default=None,
                    help=f"player seats, 1 to {MAX_PLAYERS} (default: the count used last, else PLAYERS in settings.py)")
    ap.add_argument("--log", default=None, help="append every event to this CSV file")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--list-cameras", action="store_true", help="show a snapshot from every camera, then exit")
    args = ap.parse_args()
    if args.list_cameras:
        list_cameras()
        return

    detector = None
    if Path(args.model).exists():
        from cardReader.detector import Detector
        detector = Detector(args.model, args.imgsz, args.conf, args.device)
    else:
        print(f"no model at {args.model}: running in capture mode (press s to save frames)")

    cam = Camera(parse_source(args.source), args.width, args.height)
    W, H = cam.size
    print(f"source {args.source} at {W}x{H}")
    tracker = CardTracker(settings.CONFIRM_FRAMES, settings.DROP_FRAMES, settings.MATCH_DISTANCE * W,
                          settings.ROUND_END_FRAMES, confirm_window=settings.CONFIRM_WINDOW,
                          confirm_agreement=settings.CONFIRM_AGREEMENT,
                          confirm_confidence=settings.CONFIRM_CONFIDENCE,
                          still_distance=settings.STILL_DISTANCE * W)
    seats_file = settings.PROJECT_ROOT / settings.SEATS_FILE

    def seats_for(players: int | None, width: int, height: int) -> SeatLayout:
        """The layout saved for this player count (None: the count used last), else the template."""
        saved = SeatLayout.load(seats_file, width, height, players)
        if saved is not None:
            print(f"loaded your saved {saved.players}-player zones from {settings.SEATS_FILE}")
            return saved
        return SeatLayout(players or settings.PLAYERS, width, height)

    layout = seats_for(args.players, W, H)
    editor = ZoneEditor(layout)
    frames_dir = settings.PROJECT_ROOT / settings.FRAMES_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)

    log = None
    if args.log:
        new_file = not Path(args.log).exists()
        log_file = open(args.log, "a", newline="")
        log = csv.writer(log_file)
        if new_file:
            log.writerow(["time", "frame", "event", "order", "card", "seat", "running_count", "x", "y"])

    hud = Hud()
    if not args.no_display:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  # resizable, so a 4K feed fits on a laptop screen
        cv2.setMouseCallback(WINDOW, editor.on_mouse)
    t_last, fps = time.time(), 0.0
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                if cam.is_file:
                    break
                time.sleep(0.01)
                continue
            if frame.shape[1] != layout.width or frame.shape[0] != layout.height:
                layout = editor.layout = layout.resized(frame.shape[1], frame.shape[0])

            dets = detector(frame) if detector else []
            events = tracker.update(dets)
            count = shoe_count(tracker.shoe_seen)
            for e in events:
                seat = layout.assign(e.card.x, e.card.y) if e.card else None
                print(e, f"({seat})" if seat else "", f"count {count.running:+d}" if e.kind == "placed" else "")
                if e.kind == "round_over":
                    hud.flash("round over: table cleared")
                if log:
                    c = e.card
                    log.writerow([f"{time.time():.3f}", e.frame, e.kind, c.order if c else "", c.label if c else "",
                                  seat or "", count.running, round(c.x) if c else "", round(c.y) if c else ""])

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6)
            t_last = now

            if args.no_display:
                continue
            cv2.imshow(WINDOW, hud.draw(frame, dets, tracker.round, fps, layout, count,
                                        capture_mode=detector is None, editor=editor))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif ord("1") <= k <= ord(str(MAX_PLAYERS)):
                n = k - ord("0")
                saved = SeatLayout.load(seats_file, frame.shape[1], frame.shape[0], n)
                layout = editor.layout = saved or SeatLayout(n, frame.shape[1], frame.shape[0])
                editor.selected = None
                hud.flash(f"{n} player{'s' if n > 1 else ''}: {'your saved layout' if saved else 'template layout'}")
            elif k == ord("e"):
                editor.active = not editor.active
                if editor.active:
                    hud.flash("edit zones: drag corners or zones", 2.5)
                else:
                    layout.save(seats_file)
                    editor.selected = None
                    print(f"{layout.players}-player zones saved to {settings.SEATS_FILE}")
                    hud.flash(f"{layout.players}-player zones saved")
            elif k == ord("l"):
                hud.show_legend = not hud.show_legend
            elif k == ord("r"):
                tracker.new_round()
                print("--- new round ---")
                hud.flash("new round: numbering restarts at 1")
            elif k == ord("R"):
                tracker.new_shoe()
                print("--- new shoe: count reset ---")
                hud.flash("new shoe: count reset to 0")
            elif k == ord("s"):
                p = frames_dir / f"frame_{int(time.time() * 1000)}.jpg"
                cv2.imwrite(str(p), frame)
                print("saved", p)
                hud.flash(f"frame saved: {p.name}")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        if log:
            log_file.close()
        count = shoe_count(tracker.shoe_seen)
        print("counted this shoe:", " ".join(tracker.shoe_seen) or "-")
        print(f"running count {count.running:+d}, true count {count.true:+.1f}, {count.cards_left} cards left")


if __name__ == "__main__":
    main()
