"""What the reader does with each frame and each key press.

read.py owns the camera, the window, and the command line; this class owns
everything in between, so it can be driven in a test without either:

    reader = Reader(tracker, layout, hud, seats_file, frames_dir, log=csv_writer)
    events = reader.process(frame, detections)     # tracker -> seats -> print, log, flash
    keep_going = reader.handle_key(key, frame)     # the keys listed in hud.KEY_LIST (C is read.py's)
    reader.rename_seat("P1", "Ann")                # from the browser page; "" restores P1
    reader.split_seat("P1"), reader.join_seat("P1") # from the browser page; keys Shift+1-5 toggle_split
    reader.set_count_system("uston_apc")           # from the browser page's dropdown: recounts the shoe
    reader.cycle_decks()                           # from the browser page's deck button: 1 -> 2 -> 3 -> 1

reader.count_system is the blackjack.COUNT_SYSTEMS id reader.count is made in.
The running count printed with each placed card and logged in the CSV follows it.
reader.decks is the decks in the shoe, from the tracker's config; the count's
cards left start from 52 per deck. cycle_decks changes it and recounts the
cards already seen. reader.count.cards_left is the true count's divisor, the
cards still in the shoe; cards_left_text() gives what every display shows,
the unseen copies with the face-down cards among them named beside it.

reader.seat_names maps seat ids to display names, loaded from and saved to the
seats file and shared by every player count. They are for people only: events
are printed and logged with the seat ids, so evaluate.py reads any run.

A player seat holding a pair can be split into two hands (see table.py). The
split is for the totals and hints only: both hands keep the seat id in the
terminal and the CSV log. Every split is cleared by keys R, X, Shift+R, a
change of player count, and the end of a round.

reader.preprocess (key P) and reader.show_enhanced (key Shift+V) tell read.py
how to preprocess each frame for the detector and which frame to show.
reader.rotation (keys < and >) is how many quarter turns clockwise read.py
turns each camera frame before anything else sees it. A turn moves the
tracker's cards with the frame and reloads the seat layout for the turned size.

Events are printed as they happen and appended to the CSV log as
    time, frame, event, order, card, seat, running_count, x, y
A duplicate event (a card whose name already has every copy dealt, one per
deck; see tracker.py for the placement and rename rules) and a removed
event (a card hidden too long) are logged like the rest, with the card's seat
where it was last seen; neither changes the count. With more than one deck,
a held event follows the placed event of a second copy beside a card of its
name: it is on the table but not in the count. A counted event, printed with
the running count, marks it counted at last. Key X (rescan) logs one
rescan row, with the running count after the rescan and no card, so
evaluate.py reads the round again under the same number. A rescan before the
round's first card logs nothing, since there is nothing to read again.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .blackjack import COUNT_SYSTEMS, DEFAULT_COUNT_SYSTEM, MAX_DECKS, ShoeCount, is_pair, shoe_count
from .cards import Card, Detection, Event
from .deck import BACK
from .hud import SPLIT_KEYS, VIEWS, DealtCards, Hud, seat_title
from .preprocess import MODES
from .seats import DEALER, MAX_PLAYERS, SeatLayout, ZoneEditor, load_seat_names, save_seat_names
from .table import Table
from .tracker import CardTracker

LOG_HEADER = ["time", "frame", "event", "order", "card", "seat", "running_count", "x", "y"]
MAX_NAME_LENGTH = 16  # characters in a seat's display name


class Reader:
    def __init__(self, tracker: CardTracker, layout: SeatLayout, hud: Hud,
                 seats_file: Path, frames_dir: Path, log=None, preprocess: str = "off", rotation: int = 0,
                 count_system: str = DEFAULT_COUNT_SYSTEM):
        """preprocess: the starting preprocess mode, one of preprocess.MODES (key P cycles it).
        rotation: the starting turn in clockwise quarter turns, 0 to 3 (settings.CAMERA_ROTATION // 90).
        count_system: the starting COUNT_SYSTEMS id (settings.COUNT_SYSTEM); set_count_system changes it."""
        self.tracker, self.hud, self.log = tracker, hud, log
        self.seats_file, self.frames_dir = Path(seats_file), Path(frames_dir)
        self.editor = ZoneEditor(layout)
        self.table = Table(layout)
        self.count_system = count_system  # the COUNT_SYSTEMS id every count is made in
        self.count: ShoeCount = self._recount()
        self.preprocess = preprocess  # read.py preprocesses each frame with this mode before detection
        self.show_enhanced = True     # False: the window shows the raw camera frame instead (key Shift+V)
        self.rotation = rotation % 4  # read.py turns each camera frame this many quarter turns clockwise
        self.seat_names = load_seat_names(self.seats_file)  # seat id -> display name, for renamed seats only

    @property
    def layout(self) -> SeatLayout:
        return self.table.layout

    @property
    def decks(self) -> int:
        """Decks in the shoe, held by the tracker's config (settings.DECKS at start-up)."""
        return self.tracker.cfg.decks

    def _recount(self) -> ShoeCount:
        return shoe_count(self._counted_cards(), decks=self.decks, unrevealed=self.tracker.shoe_unrevealed,
                          system=self.count_system)

    def _counted_cards(self) -> list[str]:
        """tracker.shoe_seen with each name kept at most `decks` times, its first copies.

        After fewer decks are chosen mid-shoe the tracker keeps every copy it saw, so a
        later return to more decks counts them again; only the count leaves out the extra
        copies, as the Cards Left grid and the odds do (hud.DealtCards.seen)."""
        copies: Counter = Counter()
        counted = []
        for name in self.tracker.shoe_seen:
            copies[name] += 1
            if copies[name] <= self.decks:
                counted.append(name)
        return counted

    def set_count_system(self, system: str) -> bool:
        """Count in another COUNT_SYSTEMS entry from now on, recounting the cards already seen this
        shoe; prints and flashes the new running count. Returns False, with a message and nothing
        changed, for an unknown system id."""
        if system not in COUNT_SYSTEMS:
            self.hud.flash(f"no counting system {system!r}", 2.5)
            return False
        self.count_system = system
        self.count = self._recount()
        name = COUNT_SYSTEMS[system].name
        print(f"--- counting with {name}: running count {self.count.running:+d} ---")
        self.hud.flash(f"counting with {name}")
        return True

    def cycle_decks(self) -> int:
        """Use the next deck count, 1 -> 2 -> ... -> MAX_DECKS -> 1, from the browser page's button.

        The cards already seen this shoe are recounted against the new shoe; nothing is cleared
        (Shift+R starts a new shoe). Copies of a name beyond the new deck count stay in the
        tracker but leave the count until more decks are chosen again. Prints and flashes the
        new deck count. Returns it."""
        decks = self.decks % MAX_DECKS + 1
        self.tracker.cfg = replace(self.tracker.cfg, decks=decks)
        self.count = self._recount()
        shoe = f"{decks} deck{'s' if decks > 1 else ''}"
        print(f"--- {shoe}: running count {self.count.running:+d}, {self.cards_left_text()} ---")
        self.hud.flash(shoe)
        return decks

    def cards_left_text(self) -> str:
        """Cards left for the terminal, as the page and the window show them: the unseen copies
        (hud.DealtCards.unseen), then the face-down cards never revealed among them when there
        are any, e.g. "99 cards left, 2 face down"."""
        dealt = DealtCards.from_tracker(self.tracker)
        text = f"{len(dealt.unseen())} cards left"
        return f"{text}, {dealt.unrevealed} face down" if dealt.unrevealed else text

    def set_layout(self, layout: SeatLayout):
        """Swap the seat layout everywhere it is held."""
        self.table.layout = self.editor.layout = layout
        self.editor.selected = None

    def process(self, frame: np.ndarray, dets: list[Detection]) -> list[Event]:
        """Feed one frame's detections to the tracker and report what changed.

        Follows a change of frame size by rescaling the seat layout. Prints each
        event, flashes "round over" on the overlay and clears every split, and appends
        a row per event to the CSV log if there is one. Returns the tracker's events for this frame."""
        h, w = frame.shape[:2]
        if (w, h) != (self.layout.width, self.layout.height):
            self.set_layout(self.layout.resized(w, h))
        events = self.tracker.update(dets)
        self.count = self._recount()
        for event in events:
            card = event.card
            seat = self.table.seat_of(card) if card else None
            seat_text = f"({seat})" if seat else ""
            count_text = f"count {self.count.running:+d}" if event.kind in ("placed", "renamed", "counted") else ""
            print(event, seat_text, count_text)
            if event.kind == "round_over":
                self.table.clear_splits()
                self.hud.flash("round over: table cleared")
            if self.log:
                self.log.writerow([f"{time.time():.3f}", event.frame, event.kind,
                                   card.order if card else "", card.label if card else "", seat or "",
                                   self.count.running, round(card.x) if card else "", round(card.y) if card else ""])
        return events

    def handle_key(self, k: int, frame: np.ndarray) -> bool:
        """Act on a key code from cv2.waitKey. Returns False when the reader should quit.

        frame: the camera frame as turned by rotation, before preprocessing; S saves it, and
        < and > turn from its size."""
        if k == ord("q"):
            return False
        if ord("1") <= k <= ord(str(MAX_PLAYERS)):
            self.set_players(k - ord("0"), frame.shape[1], frame.shape[0])
        elif k in map(ord, SPLIT_KEYS):
            self.toggle_split(f"P{SPLIT_KEYS.index(chr(k)) + 1}")
        elif k == ord("e"):
            self.toggle_edit()
        elif k == ord("l"):
            self.hud.show_legend = not self.hud.show_legend
        elif k == ord("g"):
            self.hud.show_unmatched = not self.hud.show_unmatched
            self.hud.flash("unmatched boxes shown" if self.hud.show_unmatched else "unmatched boxes hidden")
        elif k == ord("n"):
            self.hud.show_order = not self.hud.show_order
            self.hud.flash("placement numbers shown" if self.hud.show_order else "placement numbers hidden")
        elif k == ord("v"):
            self.hud.view = "game" if self.hud.view == "cards_left" else "cards_left"
            self.hud.flash(f"{VIEWS[self.hud.view]} view")
        elif k == ord("r"):
            self.tracker.new_round()
            self.table.clear_splits()
            print("--- new round ---")
            self.hud.flash("new round: numbering restarts at 1")
        elif k == ord("x"):
            placed_this_round = self.tracker.rescan()
            self.table.clear_splits()
            self.count = self._recount()
            print("--- rescan: this round is read again ---")
            self.hud.flash("rescan: reading this round again")
            # R logs nothing, so a rescan row before this round's first card would drop the round R ended.
            if self.log and placed_this_round:
                self.log.writerow([f"{time.time():.3f}", self.tracker.frame, "rescan", "", "", "",
                                   self.count.running, "", ""])
        elif k == ord("R"):
            self.tracker.new_shoe()
            self.table.clear_splits()
            self.count = self._recount()
            print("--- new shoe: count reset ---")
            self.hud.flash("new shoe: count reset to 0")
        elif k == ord("p"):
            self.preprocess = MODES[(MODES.index(self.preprocess) + 1) % len(MODES)]
            self.hud.flash(f"preprocess: {self.preprocess}")
        elif k == ord("V"):
            self.show_enhanced = not self.show_enhanced
            self.hud.flash("showing the frame the detector sees" if self.show_enhanced
                           else "showing the raw camera frame")
        elif k == ord("s"):
            self.save_frame(frame)
        elif k == ord(">"):
            self.turn(1, frame.shape[1], frame.shape[0])
        elif k == ord("<"):
            self.turn(-1, frame.shape[1], frame.shape[0])
        return True

    def turn(self, quarter_turns: int, width: int, height: int):
        """Turn the view clockwise by quarter_turns (-1: anticlockwise). width, height: the frame size before it.

        The tracker's cards turn with the frame, so they keep their numbers. The seat layout is
        reloaded for the turned size: the one saved for this player count, else the template."""
        self.rotation = (self.rotation + quarter_turns) % 4
        self.tracker.rotate(quarter_turns, width, height)
        if quarter_turns % 2:
            width, height = height, width
        players = self.layout.players
        self.set_layout(SeatLayout.load(self.seats_file, width, height, players) or SeatLayout(players, width, height))
        degrees = 90 * self.rotation
        print(f"camera turned to {degrees} degrees: set CAMERA_ROTATION = {degrees} in settings.py to start there")
        self.hud.flash(f"camera turned to {degrees} degrees: CAMERA_ROTATION = {degrees} keeps it", 2.5)

    def set_players(self, n: int, width: int, height: int):
        """Switch to n seats: the layout saved for that count, else the template. Clears every split."""
        saved = SeatLayout.load(self.seats_file, width, height, n)
        self.table.clear_splits()
        self.set_layout(saved or SeatLayout(n, width, height))
        self.hud.flash(f"{n} player{'s' if n > 1 else ''}: {'your saved layout' if saved else 'template layout'}")

    def rename_seat(self, seat: str, name: str) -> bool:
        """Give a seat of the current layout a display name, trimmed, and save it to seats_file.
        An empty name restores the default (P1, DEALER). Returns False, with a message and
        nothing saved, for a seat not at the table or a name over MAX_NAME_LENGTH characters."""
        default = seat_title(seat)
        name = name.strip()
        if seat not in self.layout.names():
            self.hud.flash(f"no seat {seat} at a {self.layout.players}-player table", 2.5)
            return False
        if len(name) > MAX_NAME_LENGTH:
            self.hud.flash(f"names can be at most {MAX_NAME_LENGTH} characters", 2.5)
            return False
        if name:
            self.seat_names[seat] = name
            self.hud.flash(f"{default} is now {name}")
        else:
            self.seat_names.pop(seat, None)
            self.hud.flash(f"{default} is {default} again")
        save_seat_names(self.seats_file, self.seat_names)
        return True

    def split_seat(self, seat: str) -> bool:
        """Split a player seat's pair into two hands, part 1 the card further left; prints and flashes it.

        Returns False, with a message and nothing changed, unless the seat is a player seat at
        the table, not split already, holding exactly two face-up cards of equal value."""
        if seat not in self.layout.names():
            self.hud.flash(f"no seat {seat} at a {self.layout.players}-player table", 2.5)
            return False
        if seat == DEALER:
            self.hud.flash("the dealer's hand cannot be split", 2.5)
            return False
        cards = self._seat_cards(seat)
        labels = [c.label for c in cards]
        if self.table.seeds(seat, cards):
            self.hud.flash(f"{seat} is already split", 2.5)
            return False
        if len(cards) != 2:
            self.hud.flash(f"{seat} can split two cards, not {len(cards)}", 2.5)
            return False
        if BACK in labels:
            self.hud.flash(f"{seat} cannot split a face-down card", 2.5)
            return False
        if not is_pair(labels):
            self.hud.flash(f"{seat} cannot split {labels[0]} and {labels[1]}: not a pair", 2.5)
            return False
        self.table.split(seat, cards)
        first, second = self.table.seeds(seat, cards)
        message = f"{seat} split: {first.label} | {second.label}"
        print(f"--- {message} ---")
        self.hud.flash(message)
        return True

    def join_seat(self, seat: str) -> bool:
        """Make a split seat one hand again; prints it with the seed cards and flashes it.
        Returns False, with a message and nothing changed, when the seat is not split."""
        seeds = self.table.seeds(seat, self._seat_cards(seat))
        if not seeds:
            self.hud.flash(f"{seat} is not split", 2.5)
            return False
        self.table.join(seat)
        print(f"--- {seat} joined: {seeds[0].label} | {seeds[1].label} ---")
        self.hud.flash(f"{seat} joined")
        return True

    def toggle_split(self, seat: str) -> bool:
        """Join the seat if it is split, else split it. Returns whether anything changed."""
        if self.table.seeds(seat, self._seat_cards(seat)):
            return self.join_seat(seat)
        return self.split_seat(seat)

    def _seat_cards(self, seat: str) -> list[Card]:
        return [c for c in self.tracker.round if self.table.seat_of(c) == seat]

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
