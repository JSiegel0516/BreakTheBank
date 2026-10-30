"""Turn noisy per-frame detections into the cards of the current round, in order.

The detector gives a fresh list of boxes every frame, and that list flickers:
a card is missed for a frame, a label is misread, a hand covers the table.
The tracker smooths that out and remembers the order cards arrived in:

  * Each box is matched to the nearest existing track (one physical corner
    index followed across frames), or starts a new track.
  * A track must pass every confirmation check before it counts as a card:
      - seen steadily: in at least confirm_frames of the last confirm_window
        frames, so a detection that flickers on and off never builds up enough
      - rank agreement: the rank of its most common name has at least
        confirm_agreement of its recent reads (BACK counts as a rank of its
        own), so a card read half as 6H and half as 9H waits. A card read as
        both 10H and 10D does not: the suit changes neither its value nor the
        count. Its name is still the most common full name.
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
  * A hole card turned face up keeps its number. A BACK box is centred on the
    card, but face boxes are corner indices about half the card's diagonal
    away, so they start new tracks. A new face track with no card of its name
    joins the nearest face-down card within 3 * match_dist that is hidden or
    missing this frame (every one of its tracks missed it); the card's other
    corner then joins it by name. A face-down card still in view is never
    taken, so neighbouring face-down cards keep their own reveals.

Memory, so a card is never counted twice:

  * A card that disappears is HIDDEN, not removed. It keeps its placement
    number and stays in the round. When it is seen again it is RETURNED, with
    the same number. A face card is recognised by its name anywhere on the
    table; a face-down card by reappearing near where it was.
  * A card hidden for hidden_frames is REMOVED: it was picked up (a hand the
    dealer collected mid-round). It leaves the round but stays in the shoe
    count. Its number is not reused; if it comes back it is placed again
    under a new number, and its name is not counted again.
  * The round ends when no card has been visible for round_end_frames (the
    dealer has swept the table, or every card was removed), or when
    new_round() is called. Numbers restart at 1 for the next round.
  * shoe_seen lists every face card seen since the last shuffle, each once,
    in the order first seen. The Hi-Lo count is computed from it, so a card
    that leaves and comes back, even in a later round, is never counted again.
    A name stays in it while any card of the round holds it, so renaming one of
    two cards that share a name does not uncount the other.
  * shoe_unrevealed counts the face-down cards placed this shoe that were never
    renamed to a face (a hole card swept face down, say). They have left the
    deck, so shoe_count subtracts them from cards_left, but they add nothing to
    the running count. A face-down card still on the table across new_round()
    gets a new number but is not added again.
  * shoe_pending and shoe_locked split shoe_seen in two. A name first counted
    this round is pending: a rename can still take it back. A name counted
    before this round began is locked: it stays counted until the next shoe.
  * A card placed or renamed with a face name already in shoe_seen before
    this round is a DUPLICATE: one deck holds one of each card, so it is a
    misread or a reshuffle nobody marked with new_shoe(). It is reported, not
    counted again, and its name is added to shoe_duplicates. A card still on
    the table when the round began, and two corners of one card joining by
    name, are not duplicates.
  * rescan() takes back a misread while the round is still open. It forgets
    every card and track of this round and puts shoe_seen, shoe_unrevealed,
    and shoe_duplicates back as they were when the round began, so only the
    pending names leave the count. The cards on the table are then confirmed
    again from scratch and numbered from 1. A round that already ended is
    locked: rescan cannot reach back into it. A card still on the table from
    an earlier round is placed again after a rescan, so it is reported as a
    duplicate (face up) or added to shoe_unrevealed again (face down).
  * new_shoe() clears shoe_seen, shoe_unrevealed, and shoe_duplicates after a
    reshuffle.

All thresholds live in TrackerConfig, in frames and pixels.
TrackerConfig.from_settings turns the seconds in settings.py into frames at
settings.FPS, and the fractions of the frame width into pixels.

When the camera view is turned a quarter turn (keys < and > in read.py),
rotate() turns every track and card with the frame and rescales the pixel
thresholds to the new width, so cards on the table keep their numbers.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field, replace

from .cards import Card, Detection, Event
from .deck import BACK, rank_of


@dataclass
class TrackerConfig:
    """Every tracker threshold, in frames and pixels. The defaults are permissive, for tests."""
    confirm_frames: int = 8            # sightings needed ...
    confirm_window: int | None = None  # ... within this many frames (None: ever)
    confirm_agreement: float = 0.0     # share of recent reads that must agree on the rank
    confirm_confidence: float = 0.0    # minimum average confidence over those sightings
    still_distance: float | None = None  # pixels the corner may move over them (None: any)
    drop_frames: int = 30              # misses in a row before a track is forgotten
    match_dist: float = 40.0           # pixels: a detection closer than this to a track is that track
    round_end_frames: int = 90         # frames with no card visible before the round is over
    label_window: int | None = None    # frames the name vote looks back (None: every read ever)
    hidden_frames: int | None = None   # frames a card may stay hidden before it is removed (None: never)

    @classmethod
    def from_settings(cls, s, frame_width: int) -> TrackerConfig:
        """From settings.py: distances are fractions of the frame width, times are seconds at s.FPS.

        Each time becomes at least one frame, except HIDDEN_SECONDS, where 0 frames
        removes a card on the frame it is hidden."""
        def frames(seconds: float) -> int:
            return max(1, round(seconds * s.FPS))

        return cls(confirm_frames=frames(s.CONFIRM_SECONDS), confirm_window=frames(s.CONFIRM_WINDOW_SECONDS),
                   confirm_agreement=s.CONFIRM_AGREEMENT, confirm_confidence=s.CONFIRM_CONFIDENCE,
                   still_distance=s.STILL_DISTANCE * frame_width, drop_frames=frames(s.DROP_SECONDS),
                   match_dist=s.MATCH_DISTANCE * frame_width, round_end_frames=frames(s.ROUND_END_SECONDS),
                   label_window=frames(s.LABEL_WINDOW_SECONDS), hidden_frames=round(s.HIDDEN_SECONDS * s.FPS))


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
        self._card_of: dict[int, Card] = {}  # track id -> the card it belongs to
        self._dealt_backs: set[int] = set()  # ids of tracks of face-down cards already in shoe_unrevealed
        self.bound: list[Card | None] = []   # per detection of the last update: the confirmed card it matched
        self.new_shoe()

    def new_round(self):
        """Start a new round: forget the table, numbering restarts at 1. The count is kept.

        A card still on the table is placed again with a new number, but it is
        the card already dealt: it is not reported as a duplicate, and a
        face-down one is not added to shoe_unrevealed twice."""
        self._dealt_backs |= {track_id for track_id, card in self._card_of.items() if card.label == BACK}
        self._carried_tracks = set(self._card_of)  # tracks of cards still on the table as the round began
        self._seen_before_round = set(self.shoe_seen)
        self._unrevealed_before_round = self.shoe_unrevealed
        self._duplicates_before_round = set(self.shoe_duplicates)
        self.round: list[Card] = []
        self._card_of = {}
        self._next_order = 1  # placement numbers are never reused within a round, even after a removal
        self._empty_frames = 0

    def new_shoe(self):
        """After a reshuffle: a new round, and every card can be counted again."""
        self.shoe_seen: list[str] = []
        self.shoe_unrevealed = 0  # face-down cards placed this shoe and never renamed to a face
        self.shoe_duplicates: set[str] = set()  # face names reported as a duplicate this shoe
        self.new_round()
        self._dealt_backs = set()

    def rescan(self) -> bool:
        """Look at the table again: forget every card and track of this round, keep the rounds before it.

        Names first counted this round leave the count; names counted earlier stay, in order.
        shoe_unrevealed and shoe_duplicates go back to their values when the round began.
        Cards on the table are confirmed again from scratch and numbered from 1.
        Returns True when this round had placed a card, even one since removed: only then
        was there anything to take back."""
        placed_this_round = self._next_order > 1
        self.shoe_seen = [name for name in self.shoe_seen if name in self._seen_before_round]
        self.shoe_unrevealed = self._unrevealed_before_round
        self.shoe_duplicates = set(self._duplicates_before_round)
        self.tracks = {}
        self._card_of = {}
        self._dealt_backs = set()
        self._carried_tracks = set()
        self.bound = []
        self.round = []
        self._next_order = 1
        self._empty_frames = 0
        return placed_this_round

    @property
    def shoe_pending(self) -> set[str]:
        """Face names first counted this round. A rename can still take them back."""
        return {name for name in self.shoe_seen if name not in self._seen_before_round}

    @property
    def shoe_locked(self) -> set[str]:
        """Face names counted before this round began. They stay counted until new_shoe()."""
        return {name for name in self.shoe_seen if name in self._seen_before_round}

    def update(self, detections: list[Detection]) -> list[Event]:
        """Advance one frame with that frame's detections.

        Returns the events this frame caused: placed cards first, then renamed,
        hidden, and returned cards in placement order, then removed cards, and
        round_over last. A placed or renamed event is followed by duplicate when
        the card's name was already counted before this round. A removed card
        has left the round; round_over also starts a new round.

        Also sets bound: one entry per detection, in the same order, giving the
        confirmed card its track belongs to, or None while that track is not
        confirmed. A detection misread as another name is still bound to its card."""
        self.frame += 1
        det_tracks = self._match(detections)
        for track in self.tracks.values():
            if not track.confirmed and self._passes_checks(track):
                track.confirmed = True
        events: list[Event] = []
        # Track ids grow with first sighting, so earlier cards get lower numbers.
        by_age = sorted(self.tracks.values(), key=lambda track: track.id)
        confirmed = [track for track in by_age if track.confirmed]

        # A track already bound to a card stays with it. Any other joins the card
        # with its name (one deck), a face-down card near it, or starts a new one.
        tracks_of: dict[int, list[Track]] = {}  # card order -> its tracks this frame
        for track in confirmed:
            card = self._card_of.get(track.id)
            if card is None:
                card = self._card_for(track)
                if card is None:
                    card = Card(self._next_order, track.label, track.cx, track.cy, self.frame)
                    self._next_order += 1
                    self.round.append(card)
                    self._count(card)
                    if card.label == BACK and track.id not in self._dealt_backs:
                        self.shoe_unrevealed += 1
                    events.append(Event("placed", card, self.frame))
                    if track.id not in self._carried_tracks and self._is_duplicate(card.label):
                        self.shoe_duplicates.add(card.label)
                        events.append(Event("duplicate", card, self.frame))
                self._card_of[track.id] = card
            tracks_of.setdefault(card.order, []).append(track)

        for card in self.round:
            card_tracks = tracks_of.get(card.order)
            if not card_tracks:
                if not card.hidden:
                    card.hidden = True
                    card.hidden_since = self.frame
                    events.append(Event("hidden", card, self.frame))
                continue
            votes = sum((track.votes for track in card_tracks), Counter())
            label = votes.most_common(1)[0][0]
            if label != card.label:
                was = card.label
                self._rename(card, label)
                events.append(Event("renamed", card, self.frame, was=was))
                if self._is_duplicate(label):
                    self.shoe_duplicates.add(label)
                    events.append(Event("duplicate", card, self.frame))
            if card.hidden:
                card.hidden = False
                events.append(Event("returned", card, self.frame))
            card.x, card.y = card_tracks[0].cx, card_tracks[0].cy
        for card in self._expired_cards():
            self._remove(card)
            events.append(Event("removed", card, self.frame))
        self.bound = [self._card_of.get(track_id) for track_id in det_tracks]

        # Counts on after every card was removed, so a collected table still ends the round.
        placed_this_round = self._next_order > 1
        if placed_this_round and not tracks_of:
            self._empty_frames += 1
        else:
            self._empty_frames = 0
        if self._empty_frames >= self.cfg.round_end_frames:
            events.append(Event("round_over", None, self.frame))
            self.new_round()
        return events

    def rotate(self, quarter_turns: int, width: int, height: int):
        """Follow the frame turned clockwise by quarter_turns (negative: anticlockwise), as camera.rotate turns it.

        width, height: the frame size before the turn. Every track, its recent sightings, and every
        card of the round move with the frame, so the next frame's detections land on them and no
        card is placed again. still_distance and match_dist are rescaled to the new frame width."""
        for _ in range(quarter_turns % 4):
            for track in self.tracks.values():
                track.cx, track.cy = _quarter_turn(track.cx, track.cy, height)
                turned_hits = [(frame, conf, *_quarter_turn(x, y, height), label)
                               for frame, conf, x, y, label in track.hits]
                track.hits = deque(turned_hits, maxlen=track.hits.maxlen)
            for card in self.round:
                card.x, card.y = _quarter_turn(card.x, card.y, height)
            still = self.cfg.still_distance
            self.cfg = replace(self.cfg, match_dist=self.cfg.match_dist * height / width,
                               still_distance=None if still is None else still * height / width)
            width, height = height, width

    def table(self) -> list[Card]:
        """Cards visible right now, in the order they were placed."""
        return [c for c in self.round if not c.hidden]

    def _card_for(self, track: Track) -> Card | None:
        """The card of this round an unbound track belongs to, if any.

        A face track joins the card with its name, or a face-down card being
        revealed whose other corner already reads that name. Otherwise it joins a
        nearby face-down card that is out of sight this frame: the hole card it
        was before it was turned over. A BACK track only joins a hidden face-down
        card near it."""
        if track.label == BACK:
            return self._back_card_near(track.cx, track.cy, include_unseen=False)
        for card in self.round:
            if card.label == track.label:
                return card
            if card.label == BACK and any(t.label == track.label for t in self._tracks_of_card(card)):
                return card
        return self._back_card_near(track.cx, track.cy, include_unseen=True)

    def _tracks_of_card(self, card: Card) -> list[Track]:
        return [self.tracks[track_id] for track_id, bound in self._card_of.items() if bound is card]

    def _back_card_near(self, x: float, y: float, include_unseen: bool) -> Card | None:
        """The nearest face-down card within 3 match distances of (x, y) that is hidden or,
        with include_unseen, missing this frame: every one of its tracks missed it.

        The limit allows for a whole-card BACK box being centred on the card while
        a corner index sits about half the card's diagonal away from that centre."""
        best, best_dist = None, 3 * self.cfg.match_dist
        for card in self.round:
            if card.label != BACK:
                continue
            unseen = all(track.missed > 0 for track in self._tracks_of_card(card))
            if not (card.hidden or (include_unseen and unseen)):
                continue
            dist = ((card.x - x) ** 2 + (card.y - y) ** 2) ** 0.5
            if dist <= best_dist:
                best, best_dist = card, dist
        return best

    def _expired_cards(self) -> list[Card]:
        """Cards of the round hidden for at least hidden_frames, in placement order."""
        limit = self.cfg.hidden_frames
        if limit is None:
            return []
        return [card for card in self.round if card.hidden and self.frame - card.hidden_since >= limit]

    def _remove(self, card: Card):
        """Take a card off the table. Its name stays in the shoe count.

        A counted card hands the count to another card of the round with its
        name, if there is one, so a later rename of that card still corrects it."""
        self.round.remove(card)
        if card.counted:
            card.counted = False
            holder = next((c for c in self.round if c.label == card.label), None)
            if holder is not None:
                holder.counted = True

    def _is_duplicate(self, label: str) -> bool:
        """A face name already counted before this round: one deck has only one such card."""
        return label != BACK and label in self._seen_before_round

    def _count(self, card: Card, at: int | None = None):
        """Add the card's name to the shoe count, once per shoe. at: position to keep the order."""
        if card.label != BACK and card.label not in self.shoe_seen:
            self.shoe_seen.insert(len(self.shoe_seen) if at is None else at, card.label)
            card.counted = True

    def _rename(self, card: Card, label: str):
        """The vote changed its mind: same card, new name, and the count follows.

        The old name leaves the count only if no other card of the round still
        holds it; otherwise that card keeps it counted."""
        at = None
        if card.counted:
            card.counted = False
            holder = next((c for c in self.round if c is not card and c.label == card.label), None)
            if holder is not None:
                holder.counted = True
            else:  # the old name was never on the table
                at = self.shoe_seen.index(card.label)
                self.shoe_seen.pop(at)
        if card.label == BACK and label != BACK:
            self.shoe_unrevealed -= 1
        elif card.label != BACK and label == BACK:
            self.shoe_unrevealed += 1
        card.label = label
        self._count(card, at)

    def _match(self, detections: list[Detection]) -> list[int]:
        """Hit each detection's track, starting new tracks for the unmatched, and age the rest.
        Returns the track id of each detection, in detection order."""
        # Greedy nearest-neighbour matching, closest pairs first.
        det_tracks = [0] * len(detections)
        pairs = []
        for det_i, det in enumerate(detections):
            for track_id, track in self.tracks.items():
                dist = ((det.cx - track.cx) ** 2 + (det.cy - track.cy) ** 2) ** 0.5
                if dist <= self.cfg.match_dist:
                    pairs.append((dist, det_i, track_id))
        pairs.sort()
        used_dets: set[int] = set()
        hit_tracks: set[int] = set()
        for _, det_i, track_id in pairs:
            if det_i in used_dets or track_id in hit_tracks:
                continue
            used_dets.add(det_i)
            hit_tracks.add(track_id)
            det_tracks[det_i] = track_id
            self._hit(self.tracks[track_id], detections[det_i])

        # Unmatched detections start new tracks, highest confidence first.
        unmatched = set(range(len(detections))) - used_dets
        for det_i in sorted(unmatched, key=lambda i: -detections[i].conf):
            track = Track(self._next_id, detections[det_i].cx, detections[det_i].cy)
            self._next_id += 1
            self._hit(track, detections[det_i])
            self.tracks[track.id] = track
            hit_tracks.add(track.id)
            det_tracks[det_i] = track.id

        # Unmatched tracks accumulate misses and are dropped after drop_frames.
        for track_id in list(self.tracks):
            if track_id in hit_tracks:
                continue
            track = self.tracks[track_id]
            track.missed += 1
            if track.missed >= self.cfg.drop_frames:
                del self.tracks[track_id]
                self._card_of.pop(track_id, None)
                self._dealt_backs.discard(track_id)
        return det_tracks

    def _hit(self, track: Track, det: Detection):
        if track.seen:
            track.cx = 0.7 * track.cx + 0.3 * det.cx  # light smoothing against jitter
            track.cy = 0.7 * track.cy + 0.3 * det.cy
        track.hits.append((self.frame, det.conf, det.cx, det.cy, det.label))
        track.seen += 1
        track.missed = 0
        window = self.cfg.label_window
        if window is None:
            track.votes[det.label] += 1
        else:
            track.votes = Counter(hit[4] for hit in track.hits if hit[0] > self.frame - window)
        track.label = track.votes.most_common(1)[0][0]

    def _passes_checks(self, track: Track) -> bool:
        """Has this track earned confirmation? See the module docstring for the checks."""
        cfg = self.cfg
        window = cfg.confirm_window
        recent = [hit for hit in track.hits if window is None or hit[0] > self.frame - window]
        if len(recent) < cfg.confirm_frames:
            return False  # not seen enough, or not steadily enough
        rank_votes: Counter = Counter()
        for label, reads in track.votes.items():
            rank_votes[_rank_group(label)] += reads
        if rank_votes[_rank_group(track.label)] < cfg.confirm_agreement * track.votes.total():
            return False  # the reads disagree on the card's rank
        if sum(hit[1] for hit in recent) / len(recent) < cfg.confirm_confidence:
            return False  # the model is not sure enough
        if cfg.still_distance is not None:
            xs, ys = [hit[2] for hit in recent], [hit[3] for hit in recent]
            mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
            offsets = [((x - mean_x) ** 2 + (y - mean_y) ** 2) ** 0.5 for x, y in zip(xs, ys, strict=True)]
            if max(offsets) > cfg.still_distance:
                return False  # still moving
        return True


def _quarter_turn(x: float, y: float, height: int) -> tuple[float, float]:
    """Where (x, y) lands when a frame `height` pixels tall is turned a quarter turn clockwise."""
    return height - 1 - y, x


def _rank_group(label: str) -> str:
    """What the agreement check compares: the rank, with BACK as a group of its own."""
    return label if label == BACK else rank_of(label)
