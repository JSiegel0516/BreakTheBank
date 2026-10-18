"""What the reader does with each frame and each key press.

read.py owns the camera, the window, and the command line; this class owns
everything in between, so it can be driven in a test without either:

    reader = Reader(tracker, layout, hud, seats_file, frames_dir, log=csv_writer)
    events = reader.process(frame, detections)     # tracker -> seats -> print, log, flash
    keep_going = reader.handle_key(key, frame)     # the keys listed in KEYS

Events are printed as they happen and appended to the CSV log as
    time, frame, event, order, card, seat, running_count, x, y
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .blackjack import ShoeCount, shoe_count
from .cards import Detection, Event
from .hud import Hud
from .seats import MAX_PLAYERS, SeatLayout, ZoneEditor
from .table import Table
from .tracker import CardTracker

LOG_HEADER = ["time", "frame", "event", "order", "card", "seat", "running_count", "x", "y"]


class Reader:
    def __init__(self, tracker: CardTracker, layout: SeatLayout, hud: Hud,
                 seats_file: Path, frames_dir: Path, log=None):
        self.tracker, self.hud, self.log = tracker, hud, log
        self.seats_file, self.frames_dir = Path(seats_file), Path(frames_dir)
        self.editor = ZoneEditor(layout)
        self.table = Table(layout)
        self.count: ShoeCount = shoe_count(tracker.shoe_seen)

    @property
    def layout(self) -> SeatLayout:
        return self.table.layout

    def set_layout(self, layout: SeatLayout):
        """Swap the seat layout everywhere it is held."""
        self.table.layout = self.editor.layout = layout
        self.editor.selected = None

    # ------------------------------------------------------------ per frame
    def process(self, frame: np.ndarray, dets: list[Detection]) -> list[Event]:
        h, w = frame.shape[:2]
        if (w, h) != (self.layout.width, self.layout.height):
            self.set_layout(self.layout.resized(w, h))
        events = self.tracker.update(dets)
        self.count = shoe_count(self.tracker.shoe_seen)
        for e in events:
            seat = self.table.seat_of(e.card) if e.card else None
            print(e, f"({seat})" if seat else "", f"count {self.count.running:+d}" if e.kind in ("placed", "renamed") else "")
            if e.kind == "round_over":
                self.hud.flash("round over: table cleared")
            if self.log:
                c = e.card
                self.log.writerow([f"{time.time():.3f}", e.frame, e.kind, c.order if c else "", c.label if c else "",
                                   seat or "", self.count.running, round(c.x) if c else "", round(c.y) if c else ""])
        return events

    # ------------------------------------------------------------ keys
    def handle_key(self, k: int, frame: np.ndarray) -> bool:
        """Act on a key code from cv2.waitKey. Returns False when the reader should quit."""
        if k == ord("q"):
            return False
        if ord("1") <= k <= ord(str(MAX_PLAYERS)):
            self.set_players(k - ord("0"), frame.shape[1], frame.shape[0])
        elif k == ord("e"):
            self.toggle_edit()
        elif k == ord("l"):
            self.hud.show_legend = not self.hud.show_legend
        elif k == ord("r"):
            self.tracker.new_round()
            print("--- new round ---")
            self.hud.flash("new round: numbering restarts at 1")
        elif k == ord("R"):
            self.tracker.new_shoe()
            self.count = shoe_count(self.tracker.shoe_seen)
            print("--- new shoe: count reset ---")
            self.hud.flash("new shoe: count reset to 0")
        elif k == ord("s"):
            self.save_frame(frame)
        return True

    def set_players(self, n: int, width: int, height: int):
        """Switch to n seats: the layout saved for that count, else the template."""
        saved = SeatLayout.load(self.seats_file, width, height, n)
        self.set_layout(saved or SeatLayout(n, width, height))
        self.hud.flash(f"{n} player{'s' if n > 1 else ''}: {'your saved layout' if saved else 'template layout'}")

    def toggle_edit(self):
        """Enter zone editing, or leave it and save the zones."""
        self.editor.active = not self.editor.active
        if self.editor.active:
            self.hud.flash("edit zones: drag corners or zones", 2.5)
            return
        self.layout.save(self.seats_file)
        self.editor.selected = None
        print(f"{self.layout.players}-player zones saved to {self.seats_file}")
        self.hud.flash(f"{self.layout.players}-player zones saved")

    def save_frame(self, frame: np.ndarray) -> Path:
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        p = self.frames_dir / f"frame_{int(time.time() * 1000)}.jpg"
        cv2.imwrite(str(p), frame)
        print("saved", p)
        self.hud.flash(f"frame saved: {p.name}")
        return p
