"""What the reader does with each frame and each key press.

read.py owns the camera, the window, and the command line; this class owns
everything in between, so it can be driven in a test without either:

    reader = Reader(tracker, layout, hud, seats_file, frames_dir, log=csv_writer)
    events = reader.process(frame, detections)     # tracker -> seats -> print, log, flash
    keep_going = reader.handle_key(key, frame)     # the keys listed in KEYS

Events are printed as they happen and appended to the CSV log as
    time, frame, event, order, card, seat, running_count, x, y
A duplicate event (a name already counted in an earlier round) is logged like
the rest, with the card's seat; it does not change the count.
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
        self.count: ShoeCount = self._recount()

    @property
    def layout(self) -> SeatLayout:
        return self.table.layout

    def _recount(self) -> ShoeCount:
        return shoe_count(self.tracker.shoe_seen, unrevealed=self.tracker.shoe_unrevealed)

    def set_layout(self, layout: SeatLayout):
        """Swap the seat layout everywhere it is held."""
        self.table.layout = self.editor.layout = layout
        self.editor.selected = None

    def process(self, frame: np.ndarray, dets: list[Detection]) -> list[Event]:
        """Feed one frame's detections to the tracker and report what changed.

        Follows a change of frame size by rescaling the seat layout. Prints each
        event, flashes "round over" on the overlay, and appends a row per event to
        the CSV log if there is one. Returns the tracker's events for this frame."""
        h, w = frame.shape[:2]
        if (w, h) != (self.layout.width, self.layout.height):
            self.set_layout(self.layout.resized(w, h))
        events = self.tracker.update(dets)
        self.count = self._recount()
        for event in events:
            card = event.card
            seat = self.table.seat_of(card) if card else None
            seat_text = f"({seat})" if seat else ""
            count_text = f"count {self.count.running:+d}" if event.kind in ("placed", "renamed") else ""
            print(event, seat_text, count_text)
            if event.kind == "round_over":
                self.hud.flash("round over: table cleared")
            if self.log:
                self.log.writerow([f"{time.time():.3f}", event.frame, event.kind,
                                   card.order if card else "", card.label if card else "", seat or "",
                                   self.count.running, round(card.x) if card else "", round(card.y) if card else ""])
        return events

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
            self.count = self._recount()
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
        """Write the raw frame, without the overlay, to frames_dir as frame_<ms>.jpg. Returns its path."""
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        path = self.frames_dir / f"frame_{int(time.time() * 1000)}.jpg"
        cv2.imwrite(str(path), frame)
        print("saved", path)
        self.hud.flash(f"frame saved: {path.name}")
        return path
