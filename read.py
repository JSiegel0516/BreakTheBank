"""The card reader. Point it at the table and it lists cards in the order they are placed.

    python read.py                                   # camera and model from settings.py
    python read.py --source 1                        # another camera or a capture card
    python read.py --source round1.mp4 --log round1.csv --no-display
    python read.py --list-cameras                    # which number is which camera?
    python read.py --preprocess auto                 # contrast help for a dim or washed-out table

Per frame:  camera -> rotate -> preprocess -> detector -> tracker -> table (seats, hands, Hi-Lo count) -> overlay

Every event (placed, hidden, returned, removed, renamed, duplicate, round over) is printed
as it happens, and appended to --log as CSV if given. Without a trained model it runs
in capture mode: it shows the live feed and lets you press 's' to save frames
for labelling.

Tabs at the top of the sidebar name the two views, CARDS LEFT and GAME, and
highlight the current one. It starts in the Cards Left view, a rank by suit grid
of the deck: a card counted this round is slashed (pending), one counted in an
earlier round is crossed off (locked), and a duplicate is ringed. Press v for
the Game view, each seat's hand.

Keys: 1-7 number of players | e edit zones with the mouse (e again saves them)
      v Cards Left or Game view | l legend and key list | g show or hide the grey boxes
      r new round | Shift+R new shoe (count back to 0) | s save frame to data/raw/frames
      c reconnect the camera | p preprocess mode (off, gamma, clahe, auto)
      Shift+V show the raw camera frame or the preprocessed one
      < > turn the view a quarter turn anticlockwise or clockwise | q quit

The detector always gets the preprocessed frame (cardReader/preprocess.py); the
window shows it too unless Shift+V switches to the raw frame. s always saves the
raw frame.

Each camera frame is first turned by CAMERA_ROTATION in settings.py (0, 90, 180, or
270 degrees clockwise), for a camera mounted sideways or upside down. < and > turn
it live; the message names the CAMERA_ROTATION that keeps the new angle. Everything
after uses the turned frame: detection, zones, tags, and the frame s saves.

If a live camera sends no new frame for two seconds (a cable pulled, the phone
locked), the window says "no signal" over the last frame and the tracker waits.
Press c to reopen the same source.

The bottom bar shows the fps, the smoothed detect milliseconds, and the preprocess
mode with its milliseconds. On quit the average milliseconds per stage are printed: read (the newest
camera frame), preprocess, detect, and draw (the overlay and the window). When the fps
averaged over the run is more than a fifth away from FPS in settings.py, both are printed:
the tracker's *_SECONDS settings are turned into frames at FPS, so FPS should match.

While editing zones (key E) the detector runs on one frame in EDIT_DETECT_EVERY and the
cards found are reused in between, and each frame spends EDIT_MOUSE_MS handling the mouse
(counted in draw), so a drag starts where it was clicked even when detection is slow.
The frame after < or > is always detected, since reused cards would be in the old turn's
coordinates.

This file only owns the command line, the camera, and the window. What happens
per frame and per key is cardReader/reader.py; the overlay is cardReader/hud.py.
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
from cardReader.camera import Camera, parse_source, rotate, snapshot_cameras
from cardReader.hud import DealtCards, Hud
from cardReader.preprocess import MODES, PreprocessConfig, enhance
from cardReader.provenance import live_model
from cardReader.reader import LOG_HEADER, Reader
from cardReader.seats import MAX_PLAYERS, SeatLayout
from cardReader.tracker import CardTracker, TrackerConfig

WINDOW = "card reader"
STAGES = ("read", "preprocess", "detect", "draw")  # the timed parts of one pass of the loop
EDIT_DETECT_EVERY = 3  # while editing zones, detect on one frame in this many and reuse its detections between
EDIT_MOUSE_MS = 40     # while editing zones, time per frame spent handling mouse events


def wait_key_draining_mouse(budget_ms: float) -> int:
    """Call cv2.waitKey(1) until budget_ms have passed or a key is pressed; returns that key, else -1.

    The macOS Cocoa backend handles one queued mouse event per waitKey call (each takes about
    12 ms) and reports the pointer where it is when the event is handled. With one call per slow
    frame, a click is handled after the pointer has left the handle, so the drag misses."""
    deadline = time.perf_counter() + budget_ms / 1000
    while True:
        key = cv2.waitKey(1)
        if key != -1 or time.perf_counter() >= deadline:
            return key


class StageTimer:
    """Milliseconds spent in each stage of the loop: smoothed like the fps for the bottom
    bar, and averaged over the whole run for the summary printed on quit."""

    def __init__(self):
        self.smoothed = dict.fromkeys(STAGES, 0.0)
        self._total = dict.fromkeys(STAGES, 0.0)
        self._count = dict.fromkeys(STAGES, 0)

    def add(self, stage: str, ms: float):
        self.smoothed[stage] = ms if self._count[stage] == 0 else 0.9 * self.smoothed[stage] + 0.1 * ms
        self._total[stage] += ms
        self._count[stage] += 1

    def lap(self, stage: str, since: float) -> float:
        """Record the time from `since` (a time.perf_counter() value) to now. Returns now."""
        now = time.perf_counter()
        self.add(stage, 1000 * (now - since))
        return now

    def averages(self) -> dict[str, float]:
        """Mean milliseconds per stage; 0 for a stage never timed."""
        return {stage: self._total[stage] / max(self._count[stage], 1) for stage in STAGES}

    def summary(self) -> str:
        return ", ".join(f"{stage} {ms:.1f} ms" for stage, ms in self.averages().items())


def fps_note(measured: float, expected: float) -> str:
    """The quit summary's warning when the measured fps is more than a fifth away from FPS in
    settings.py, which turns every *_SECONDS setting into frames; "" when they are close enough."""
    if abs(measured - expected) <= 0.2 * expected:
        return ""
    return (f"measured {measured:.1f} fps, but FPS in settings.py is {expected}: "
            f"set FPS to about {measured:.1f} so the *_SECONDS settings take the time they say")


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


def reconnect(cam: Camera, hud: Hud, source):
    """Reopen the camera, e.g. after a cable or power interruption. Prints and flashes how it went."""
    print(f"reconnecting to source {source} ...")
    if cam.reopen():
        W, H = cam.size
        print(f"reconnected to source {source} at {W}x{H}")
        hud.flash("camera reconnected")
    else:
        print(f"could not reopen source {source}: check the cable or wake the phone, then press C again")
        hud.flash("reconnect failed: press C to try again", 2.5)


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
    ap.add_argument("--preprocess", choices=MODES, default=settings.PREPROCESS,
                    help="contrast help before detection (default: PREPROCESS in settings.py; key P cycles it)")
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
        from cardReader.detector import Detector
        detector = Detector(args.model, args.imgsz, args.conf, args.device)
        live = live_model(settings.PROJECT_ROOT / settings.LIVE_MODEL_DIR)
        is_live = Path(args.model).resolve() == (settings.PROJECT_ROOT / settings.MODEL_PATH).resolve()
        print(f"model {args.model} on {detector.device_used}" + (f" (live model: {live})" if live and is_live else ""))
    else:
        print(f"no model at {args.model}: running in capture mode (press s to save frames)")

    cam = Camera(parse_source(args.source), args.width, args.height)
    W, H = cam.size
    print(f"source {args.source} at {W}x{H}")
    rotation = settings.CAMERA_ROTATION // 90
    view_w, view_h = (H, W) if rotation % 2 else (W, H)  # the frame size after the starting turn

    seats_file = settings.PROJECT_ROOT / settings.SEATS_FILE
    layout = SeatLayout.load(seats_file, view_w, view_h, args.players)
    if layout is not None:
        print(f"loaded your saved {layout.players}-player zones from {settings.SEATS_FILE}")
    else:
        layout = SeatLayout(args.players or settings.PLAYERS, view_w, view_h)

    with ExitStack() as cleanup:  # closes the log file, if any, on the way out
        log = None
        if args.log:
            new_file = not Path(args.log).exists()
            log = csv.writer(cleanup.enter_context(open(args.log, "a", newline="")))
            if new_file:
                log.writerow(LOG_HEADER)

        tracker = CardTracker(TrackerConfig.from_settings(settings, view_w))
        hud = Hud()
        reader = Reader(tracker, layout, hud, seats_file, settings.PROJECT_ROOT / settings.FRAMES_DIR, log=log,
                        preprocess=args.preprocess, rotation=rotation)
        preprocess_config = PreprocessConfig.from_settings(settings)
        if not args.no_display:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  # resizable, so a 4K feed fits on a laptop screen
            cv2.setMouseCallback(WINDOW, reader.editor.on_mouse)
        timer = StageTimer()
        t_last, fps = time.time(), 0.0
        signal_lost = False
        dets, frames_processed = [], 0
        dets_rotation = reader.rotation  # the turn the cached dets were detected at
        run_start = time.perf_counter()
        try:
            while True:
                start = time.perf_counter()
                ok, frame = cam.read()
                timer.lap("read", start)
                if not ok and cam.is_file:
                    break
                if not ok and not cam.stale:
                    time.sleep(0.01)  # a live source still starting up
                    continue
                if not ok:
                    frame = np.zeros((H, W, 3), np.uint8)  # it never sent a frame: say so over black
                frame = rotate(frame, reader.rotation)

                # A stale camera repeats its last frame: nothing new to detect, so the tracker is left alone.
                no_signal = cam.stale
                if no_signal and not signal_lost:
                    print(f"no signal from source {args.source}: press C in the window to reconnect")
                elif signal_lost and not no_signal:
                    print(f"signal back from source {args.source}")
                signal_lost = no_signal

                start = time.perf_counter()
                enhanced = enhance(frame, reader.preprocess, preprocess_config)
                timer.lap("preprocess", start)
                if no_signal:
                    dets = []
                    time.sleep(0.01)
                else:
                    turned = dets_rotation != reader.rotation  # cached dets are in the old frame's coordinates
                    editing_skip = reader.editor.active and frames_processed % EDIT_DETECT_EVERY != 0
                    if detector and (turned or not editing_skip):
                        start = time.perf_counter()
                        dets = detector(enhanced)
                        dets_rotation = reader.rotation
                        timer.lap("detect", start)
                    reader.process(frame, dets)
                    frames_processed += 1

                now = time.time()
                fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6)
                t_last = now
                if args.no_display:
                    continue
                start = time.perf_counter()
                dealt = DealtCards(tracker.shoe_locked, tracker.shoe_pending, tracker.shoe_duplicates,
                                   tracker.shoe_unrevealed)
                shown = enhanced if reader.show_enhanced else frame
                showing_raw = shown is not enhanced  # the window shows something other than what was detected
                cv2.imshow(WINDOW, hud.draw(shown, dets, tracker.round, fps, reader.layout, reader.count,
                                            bound=tracker.bound,
                                            capture_mode=detector is None, editor=reader.editor, dealt=dealt,
                                            stage_ms=timer.smoothed, no_signal=no_signal,
                                            preprocess=reader.preprocess, showing_raw=showing_raw))
                key = wait_key_draining_mouse(EDIT_MOUSE_MS) if reader.editor.active else cv2.waitKey(1)
                key &= 0xFF
                timer.lap("draw", start)
                if key == ord("c") and not cam.is_file:  # read.py owns the camera, so C is handled here
                    reconnect(cam, hud, args.source)
                elif not reader.handle_key(key, frame):  # turned but not preprocessed, so S saves what the camera sent
                    break
        finally:
            cam.release()
            cv2.destroyAllWindows()
            count = reader.count
            print("counted this shoe:", " ".join(tracker.shoe_seen) or "-")
            print(f"running count {count.running:+d}, true count {count.true:+.1f}, {count.cards_left} cards left")
            print(f"average time per frame: {timer.summary()}")
            run_seconds = time.perf_counter() - run_start
            note = fps_note(frames_processed / run_seconds, settings.FPS) if frames_processed else ""
            if note:
                print(note)


if __name__ == "__main__":
    main()
