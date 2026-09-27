"""Turn noisy per-frame detections into the ordered list of cards on the table.

The detector gives a fresh list of boxes every frame, and that list flickers:
a card is missed for a frame, a label is misread, a hand covers the table.
The tracker smooths that out and remembers the order cards arrived in:

  * Each box is matched to the nearest existing track (one physical corner
    index followed across frames), or starts a new track.
  * A track must be seen for CONFIRM_FRAMES frames before it counts. This kills
    flicker and cards still in motion.
  * A track is only dropped after DROP_FRAMES consecutive misses, so a hand
    passing over the table does not erase a card.
  * A track's label is a running vote, so one misread frame does not flip it.
  * One deck: every confirmed track with the same face label is the same card.
    That collapses the two corner indices of one card, and a stray misread
    elsewhere cannot add a second King of Hearts. Face-down cards (BACK) are
    all different cards.

When a new card is confirmed it gets the next placement number. table() lists
the cards currently down in that order; history lists every card ever placed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .deck import BACK


@dataclass
class Detection:
    label: str
    conf: float
    cx: float
    cy: float
    box: tuple[float, float, float, float] = (0, 0, 0, 0)  # x1, y1, x2, y2


@dataclass
class Track:
    id: int
    cx: float
    cy: float
    votes: Counter = field(default_factory=Counter)
    seen: int = 0
    missed: int = 0
    confirmed: bool = False

    @property
    def label(self) -> str:
        return self.votes.most_common(1)[0][0]

    @property
    def card_key(self) -> str:
        """Tracks with the same key are the same physical card."""
        return self.label if self.label != BACK else f"{BACK}#{self.id}"


@dataclass
class Card:
    order: int                      # 1 for the first card placed, 2 for the next ...
    label: str
    x: float                        # latest position in pixels
    y: float
    placed_frame: int
    removed_frame: int | None = None


@dataclass
class Event:
    kind: str  # "placed" | "removed"
    card: Card
    frame: int

    def __str__(self):
        return f"[{self.frame:6d}] {self.kind:7s} #{self.card.order:<3d} {self.card.label}"


class CardTracker:
    def __init__(self, confirm_frames=8, drop_frames=30, match_dist=40.0):
        """match_dist: pixels; a detection closer than this to a track is that track."""
        self.confirm_frames = confirm_frames
        self.drop_frames = drop_frames
        self.match_dist = match_dist
        self.reset()

    def reset(self):
        """Forget everything, e.g. between rounds. Placement numbers restart at 1."""
        self.tracks: dict[int, Track] = {}
        self.history: list[Card] = []
        self._on_table: dict[str, Card] = {}  # card_key -> Card
        self._next_id = 1
        self.frame = 0

    def update(self, detections: list[Detection]) -> list[Event]:
        self.frame += 1
        self._match(detections)

        # Group confirmed tracks into cards. Track ids grow with first sighting,
        # so sorting by id puts cards that appeared earlier first.
        present: dict[str, Track] = {}
        for t in sorted(self.tracks.values(), key=lambda t: t.id):
            if t.confirmed:
                present.setdefault(t.card_key, t)

        events = []
        for key, t in present.items():
            card = self._on_table.get(key)
            if card is None:
                card = Card(len(self.history) + 1, t.label, t.cx, t.cy, self.frame)
                self.history.append(card)
                self._on_table[key] = card
                events.append(Event("placed", card, self.frame))
            card.x, card.y = t.cx, t.cy
        for key in [k for k in self._on_table if k not in present]:
            card = self._on_table.pop(key)
            card.removed_frame = self.frame
            events.append(Event("removed", card, self.frame))
        return events

    def table(self) -> list[Card]:
        """Cards on the table right now, in the order they were placed."""
        return sorted(self._on_table.values(), key=lambda c: c.order)

    def _match(self, detections: list[Detection]):
        # Greedy nearest-neighbour matching, closest pairs first.
        pairs = []
        for di, d in enumerate(detections):
            for tid, t in self.tracks.items():
                dist = ((d.cx - t.cx) ** 2 + (d.cy - t.cy) ** 2) ** 0.5
                if dist <= self.match_dist:
                    pairs.append((dist, di, tid))
        pairs.sort()
        used_d: set[int] = set()
        hit_t: set[int] = set()
        for _, di, tid in pairs:
            if di in used_d or tid in hit_t:
                continue
            used_d.add(di)
            hit_t.add(tid)
            self._hit(self.tracks[tid], detections[di])

        # Unmatched detections start new tracks, highest confidence first.
        for di in sorted(set(range(len(detections))) - used_d, key=lambda i: -detections[i].conf):
            t = Track(self._next_id, detections[di].cx, detections[di].cy)
            self._next_id += 1
            self._hit(t, detections[di])
            self.tracks[t.id] = t
            hit_t.add(t.id)

        # Unmatched tracks accumulate misses and are dropped after drop_frames.
        for tid in list(self.tracks):
            if tid in hit_t:
                continue
            t = self.tracks[tid]
            t.missed += 1
            if t.missed >= self.drop_frames:
                del self.tracks[tid]

    def _hit(self, t: Track, d: Detection):
        if t.seen:
            t.cx = 0.7 * t.cx + 0.3 * d.cx  # light smoothing against jitter
            t.cy = 0.7 * t.cy + 0.3 * d.cy
        t.votes[d.label] += 1
        t.seen += 1
        t.missed = 0
        if t.seen >= self.confirm_frames:
            t.confirmed = True
