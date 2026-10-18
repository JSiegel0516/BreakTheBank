"""Turn noisy per-frame detections into the cards of the current round, in order.

The detector gives a fresh list of boxes every frame, and that list flickers:
a card is missed for a frame, a label is misread, a hand covers the table.
The tracker smooths that out and remembers the order cards arrived in:

  * Each box is matched to the nearest existing track (one physical corner
    index followed across frames), or starts a new track.
  * A track must pass every confirmation check before it counts as a card:
      - seen steadily: in at least confirm_frames of the last confirm_window
        frames, so a detection that flickers on and off never builds up enough
      - name agreement: its most common name has at least confirm_agreement of
        its recent reads, so a card read half as 6H and half as 9H waits
      - confidence: the model's average confidence over those frames is at
        least confirm_confidence, so borderline detections never count
      - held still: the corner has moved less than still_distance over those
        frames, so a card being slid across the table is counted where it lands
    A track that passes once stays confirmed.
  * A track is only dropped after drop_frames consecutive misses, so a hand
    passing over the table does not erase a card straight away.
  * A track's label is a vote over its reads in the last label_window frames,
    so one misread frame does not flip it, but a card turned face up in place
    (a dealer's hole card) does change from BACK to its name after about half
    the window.
  * A confirmed track is bound to a card and stays with it. A card's label is
    the vote over all its tracks, so when the vote changes the card is RENAMED
    in place: same placement number, and the shoe count is corrected. No
    phantom second card appears.
  * One deck: a new track with a face label joins the card of that name if the
    round already has one. That collapses a card's corner indices, and a stray
    misread elsewhere cannot add a second King of Hearts. Face-down cards
    (BACK) are all different cards.

Memory, so a card is never counted twice:

  * A card that disappears is HIDDEN, not removed. It keeps its placement
    number and stays in the round. When it is seen again it is RETURNED, with
    the same number. A face card is recognised by its name anywhere on the
    table; a face-down card by reappearing near where it was.
  * The round ends when no card has been visible for round_end_frames (the
    dealer has swept the table), or when new_round() is called. Numbers
    restart at 1 for the next round.
  * shoe_seen lists every face card seen since the last shuffle, each once,
    in the order first seen. The Hi-Lo count is computed from it, so a card
    that leaves and comes back, even in a later round, is never counted again.
    new_shoe() clears it after a reshuffle.

All thresholds live in TrackerConfig; TrackerConfig.from_settings turns the
fractions of the frame width in settings.py into pixels.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field, replace

from .cards import Card, Detection, Event
from .deck import BACK


@dataclass
class TrackerConfig:
    """Every tracker threshold, in frames and pixels. The defaults are permissive, for tests."""
    confirm_frames: int = 8            # sightings needed ...
    confirm_window: int | None = None  # ... within this many frames (None: ever)
    confirm_agreement: float = 0.0     # share of recent reads that must agree on the name
    confirm_confidence: float = 0.0    # minimum average confidence over those sightings
    still_distance: float | None = None  # pixels the corner may move over them (None: any)
    drop_frames: int = 30              # misses in a row before a track is forgotten
    match_dist: float = 40.0           # pixels: a detection closer than this to a track is that track
    round_end_frames: int = 90         # frames with no card visible before the round is over
    label_window: int | None = None    # frames the name vote looks back (None: every read ever)

    @classmethod
    def from_settings(cls, s, frame_width: int) -> "TrackerConfig":
        """From settings.py, with distances given as fractions of the frame width."""
        return cls(confirm_frames=s.CONFIRM_FRAMES, confirm_window=s.CONFIRM_WINDOW,
                   confirm_agreement=s.CONFIRM_AGREEMENT, confirm_confidence=s.CONFIRM_CONFIDENCE,
                   still_distance=s.STILL_DISTANCE * frame_width, drop_frames=s.DROP_FRAMES,
                   match_dist=s.MATCH_DISTANCE * frame_width, round_end_frames=s.ROUND_END_FRAMES,
                   label_window=s.LABEL_WINDOW)


@dataclass
class Track:
    id: int
    cx: float
    cy: float
    hits: deque = field(default_factory=lambda: deque(maxlen=200))  # (frame, conf, x, y, label) of recent sightings
    votes: Counter = field(default_factory=Counter)  # reads per label, within the label window
    label: str = ""                                   # the winning vote
    seen: int = 0
    missed: int = 0
    confirmed: bool = False


class CardTracker:
    def __init__(self, config: TrackerConfig | None = None, **overrides):
        """CardTracker(TrackerConfig.from_settings(settings, width)), or thresholds as keywords."""
        self.cfg = replace(config or TrackerConfig(), **overrides)
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.frame = 0
        self.new_shoe()

    def new_round(self):
        """Start a new round: forget the table, numbering restarts at 1. The count is kept."""
        self.round: list[Card] = []
        self._card_of: dict[int, Card] = {}  # track id -> the card it belongs to
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
        events: list[Event] = []
        # Track ids grow with first sighting, so earlier cards get lower numbers.
        confirmed = [t for t in sorted(self.tracks.values(), key=lambda t: t.id) if t.confirmed]

        # A track already bound to a card stays with it. Any other joins the card
        # with its name (one deck), a hidden face-down card near it, or starts a new one.
        tracks_of: dict[int, list[Track]] = {}  # card order -> its tracks this frame
        for t in confirmed:
            card = self._card_of.get(t.id)
            if card is None:
                card = self._card_for(t)
                if card is None:
                    card = Card(len(self.round) + 1, t.label, t.cx, t.cy, self.frame)
                    self.round.append(card)
                    self._count(card)
                    events.append(Event("placed", card, self.frame))
                self._card_of[t.id] = card
            tracks_of.setdefault(card.order, []).append(t)

        for card in self.round:
            ts = tracks_of.get(card.order)
            if not ts:
                if not card.hidden:
                    card.hidden = True
                    events.append(Event("hidden", card, self.frame))
                continue
            label = sum((t.votes for t in ts), Counter()).most_common(1)[0][0]
            if label != card.label:
                was = card.label
                self._rename(card, label)
                events.append(Event("renamed", card, self.frame, was=was))
            if card.hidden:
                card.hidden = False
                events.append(Event("returned", card, self.frame))
            card.x, card.y = ts[0].cx, ts[0].cy

        self._empty_frames = self._empty_frames + 1 if self.round and not tracks_of else 0
        if self._empty_frames >= self.cfg.round_end_frames:
            events.append(Event("round_over", None, self.frame))
            self.new_round()
        return events

    def table(self) -> list[Card]:
        """Cards visible right now, in the order they were placed."""
        return [c for c in self.round if not c.hidden]

    # ------------------------------------------------------------ cards
    def _card_for(self, t: Track) -> Card | None:
        """The card of this round an unbound track belongs to, if any."""
        if t.label != BACK:
            return next((c for c in self.round if c.label == t.label), None)
        return self._hidden_back_near(t.cx, t.cy)

    def _hidden_back_near(self, x: float, y: float) -> Card | None:
        best, best_d = None, 3 * self.cfg.match_dist  # a whole-card box can shift a little when picked up
        for c in self.round:
            if c.hidden and c.label == BACK:
                d = ((c.x - x) ** 2 + (c.y - y) ** 2) ** 0.5
                if d <= best_d:
                    best, best_d = c, d
        return best

    def _count(self, card: Card, at: int | None = None):
        """Add the card's name to the shoe count, once per shoe. at: position to keep the order."""
        if card.label != BACK and card.label not in self.shoe_seen:
            self.shoe_seen.insert(len(self.shoe_seen) if at is None else at, card.label)
            card.counted = True

    def _rename(self, card: Card, label: str):
        """The vote changed its mind: same card, new name, and the count follows."""
        at = None
        if card.counted:  # the old name was never on the table
            at = self.shoe_seen.index(card.label)
            self.shoe_seen.pop(at)
            card.counted = False
        card.label = label
        self._count(card, at)

    # ------------------------------------------------------------ tracks
    def _match(self, detections: list[Detection]):
        # Greedy nearest-neighbour matching, closest pairs first.
        pairs = []
        for di, d in enumerate(detections):
            for tid, t in self.tracks.items():
                dist = ((d.cx - t.cx) ** 2 + (d.cy - t.cy) ** 2) ** 0.5
                if dist <= self.cfg.match_dist:
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
            if t.missed >= self.cfg.drop_frames:
                del self.tracks[tid]
                self._card_of.pop(tid, None)

    def _hit(self, t: Track, d: Detection):
        if t.seen:
            t.cx = 0.7 * t.cx + 0.3 * d.cx  # light smoothing against jitter
            t.cy = 0.7 * t.cy + 0.3 * d.cy
        t.hits.append((self.frame, d.conf, d.cx, d.cy, d.label))
        t.seen += 1
        t.missed = 0
        w = self.cfg.label_window
        if w is None:
            t.votes[d.label] += 1
        else:
            t.votes = Counter(h[4] for h in t.hits if h[0] > self.frame - w)
        t.label = t.votes.most_common(1)[0][0]

    def _passes_checks(self, t: Track) -> bool:
        """Has this track earned confirmation? See the module docstring for the checks."""
        cfg = self.cfg
        recent = [h for h in t.hits if cfg.confirm_window is None or h[0] > self.frame - cfg.confirm_window]
        if len(recent) < cfg.confirm_frames:
            return False  # not seen enough, or not steadily enough
        if t.votes[t.label] < cfg.confirm_agreement * sum(t.votes.values()):
            return False  # the reads disagree on which card this is
        if sum(h[1] for h in recent) / len(recent) < cfg.confirm_confidence:
            return False  # the model is not sure enough
        if cfg.still_distance is not None:
            xs, ys = [h[2] for h in recent], [h[3] for h in recent]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            if max(((x - mx) ** 2 + (y - my) ** 2) ** 0.5 for x, y in zip(xs, ys)) > cfg.still_distance:
                return False  # still moving
        return True
