"""Turn noisy per-frame detections into the cards of the current round, in order.

The detector gives a fresh list of boxes every frame, and that list flickers:
a card is missed for a frame, a label is misread, a hand covers the table.
The tracker smooths that out and remembers the order cards arrived in:

  * Each box is matched to the nearest existing track (one physical corner
    index followed across frames), or starts a new track.
  * A track must pass every confirmation check before it counts as a card:
      - seen steadily: in at least CONFIRM_FRAMES of the last CONFIRM_WINDOW
        frames, so a detection that flickers on and off never builds up enough
      - name agreement: its most common name has at least CONFIRM_AGREEMENT of
        all its reads, so a card read half as 6H and half as 9H waits
      - confidence: the model's average confidence over those frames is at
        least CONFIRM_CONFIDENCE, so borderline detections never count
      - held still: the corner has moved less than STILL_DISTANCE over those
        frames, so a card being slid across the table is counted where it lands
    A track that passes once stays confirmed.
  * A track is only dropped after DROP_FRAMES consecutive misses, so a hand
    passing over the table does not erase a card straight away.
  * A track's label is a running vote, so one misread frame does not flip it.
  * One deck: every confirmed track with the same face label is the same card.
    That collapses a card's corner indices, and a stray misread elsewhere
    cannot add a second King of Hearts. Face-down cards (BACK) are all
    different cards.

Memory, so a card is never counted twice:

  * A card that disappears is HIDDEN, not removed. It keeps its placement
    number and stays in the round. When it is seen again it is RETURNED, with
    the same number. A face card is recognised by its name anywhere on the
    table; a face-down card by reappearing near where it was.
  * The round ends when no card has been visible for ROUND_END_FRAMES (the
    dealer has swept the table), or when new_round() is called. Numbers
    restart at 1 for the next round.
  * shoe_seen lists every face card seen since the last shuffle, each once,
    in the order first seen. The Hi-Lo count is computed from it, so a card
    that leaves and comes back, even in a later round, is never counted again.
    new_shoe() clears it after a reshuffle.
"""
from __future__ import annotations

from collections import Counter, deque
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
    hits: deque = field(default_factory=lambda: deque(maxlen=200))  # (frame, conf, x, y) of recent sightings
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
    order: int                      # 1 for the first card placed this round, 2 for the next ...
    label: str
    x: float                        # latest known position in pixels
    y: float
    placed_frame: int
    hidden: bool = False            # not visible right now, but still on the table


@dataclass
class Event:
    kind: str  # "placed" | "hidden" | "returned" | "round_over"
    card: Card | None
    frame: int

    def __str__(self):
        if self.card is None:
            return f"[{self.frame:6d}] {self.kind}"
        return f"[{self.frame:6d}] {self.kind:8s} #{self.card.order:<3d} {self.card.label}"


class CardTracker:
    def __init__(self, confirm_frames=8, drop_frames=30, match_dist=40.0, round_end_frames=90,
                 confirm_window=None, confirm_agreement=0.0, confirm_confidence=0.0, still_distance=None):
        """match_dist: pixels; a detection closer than this to a track is that track.
        round_end_frames: frames with no card visible before the round is over.
        The confirmation checks (see the module docstring) default to off:
        confirm_window None = any frames count, still_distance None = movement allowed."""
        self.confirm_frames = confirm_frames
        self.confirm_window = confirm_window
        self.confirm_agreement = confirm_agreement
        self.confirm_confidence = confirm_confidence
        self.still_distance = still_distance
        self.drop_frames = drop_frames
        self.match_dist = match_dist
        self.round_end_frames = round_end_frames
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.frame = 0
        self.new_shoe()

    def new_round(self):
        """Start a new round: forget the table, numbering restarts at 1. The count is kept."""
        self.round: list[Card] = []
        self._by_key: dict[str, Card] = {}  # card_key -> Card
        self._empty_frames = 0

    def new_shoe(self):
        """After a reshuffle: a new round, and every card can be counted again."""
        self.new_round()
        self.shoe_seen: list[str] = []

    def update(self, detections: list[Detection]) -> list[Event]:
        self.frame += 1
        self._match(detections)
        for t in self.tracks.values():
            if not t.confirmed and self._passes_checks(t):
                t.confirmed = True

        # Group confirmed tracks into cards. Track ids grow with first sighting,
        # so sorting by id puts cards that appeared earlier first.
        present: dict[str, Track] = {}
        for t in sorted(self.tracks.values(), key=lambda t: t.id):
            if t.confirmed:
                present.setdefault(t.card_key, t)

        events = []
        visible: set[int] = set()
        for key, t in present.items():
            card = self._by_key.get(key)
            if card is None and t.label == BACK:
                card = self._hidden_back_near(t.cx, t.cy)  # a face-down card coming back
                if card is not None:
                    self._by_key[key] = card
            if card is None:
                card = Card(len(self.round) + 1, t.label, t.cx, t.cy, self.frame)
                self.round.append(card)
                self._by_key[key] = card
                events.append(Event("placed", card, self.frame))
                if t.label != BACK and t.label not in self.shoe_seen:
                    self.shoe_seen.append(t.label)
            elif card.hidden:
                card.hidden = False
                events.append(Event("returned", card, self.frame))
            card.x, card.y = t.cx, t.cy
            visible.add(card.order)

        for card in self.round:
            if card.order not in visible and not card.hidden:
                card.hidden = True
                events.append(Event("hidden", card, self.frame))

        self._empty_frames = self._empty_frames + 1 if self.round and not visible else 0
        if self._empty_frames >= self.round_end_frames:
            events.append(Event("round_over", None, self.frame))
            self.new_round()
        return events

    def table(self) -> list[Card]:
        """Cards visible right now, in the order they were placed."""
        return [c for c in self.round if not c.hidden]

    def _hidden_back_near(self, x: float, y: float) -> Card | None:
        best, best_d = None, 3 * self.match_dist  # a whole-card box can shift a little when picked up
        for c in self.round:
            if c.hidden and c.label == BACK:
                d = ((c.x - x) ** 2 + (c.y - y) ** 2) ** 0.5
                if d <= best_d:
                    best, best_d = c, d
        return best

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
        t.hits.append((self.frame, d.conf, d.cx, d.cy))
        t.seen += 1
        t.missed = 0

    def _passes_checks(self, t: Track) -> bool:
        """Has this track earned confirmation? See the module docstring for the checks."""
        window = self.confirm_window
        recent = [h for h in t.hits if window is None or h[0] > self.frame - window]
        if len(recent) < self.confirm_frames:
            return False  # not seen enough, or not steadily enough
        if t.votes[t.label] < self.confirm_agreement * sum(t.votes.values()):
            return False  # the reads disagree on which card this is
        if sum(h[1] for h in recent) / len(recent) < self.confirm_confidence:
            return False  # the model is not sure enough
        if self.still_distance is not None:
            xs, ys = [h[2] for h in recent], [h[3] for h in recent]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            if max(((x - mx) ** 2 + (y - my) ** 2) ** 0.5 for x, y in zip(xs, ys)) > self.still_distance:
                return False  # still moving
        return True
