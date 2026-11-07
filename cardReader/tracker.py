"""Turn noisy per-frame detections into the cards of the current round, in order.

The detector gives a fresh list of boxes every frame, and that list flickers:
a card is missed for a frame, a label is misread, a hand covers the table.
The tracker smooths that out and remembers the order cards arrived in:

  * Each box is matched to the nearest existing track (one physical corner
    index followed across frames), or starts a new track.
  * A box scoring below start_confidence is weak (read.py runs the detector
    down to SUSTAIN_CONFIDENCE). It is matched only to a track no confident box
    hit this frame, and only keeps it seen: the track moves with it and does not
    miss, so a corner that scores low for a few frames is not dropped and
    started again. A weak box adds nothing to the name vote or to the
    confirmation checks below, and never starts a track.
  * A track must pass every confirmation check before it counts as a card:
      - seen steadily: for confirm_seconds in total within the last
        confirm_window_seconds, so a detection that flickers on and off never
        builds up enough. Each sighting stands for the time since the frame
        before it, so a slow camera confirms a card after as long as a fast one
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
  * A track is only dropped once it has gone unseen for drop_seconds, weak
    sightings counting as seen, so a hand passing over the table does not
    erase a card straight away.
  * A track's label is a vote over its reads in the last label_window_seconds,
    so one misread frame does not flip it, but a card turned face up in place
    (a dealer's hole card) does change from BACK to its name after about half
    the window.
  * A confirmed track is bound to a card and stays with it. A card's label is
    the vote over all its tracks, so when the vote changes the card is RENAMED
    in place: same placement number, and the shoe count is corrected. No
    phantom second card appears. A name that differs from the card's only in
    suit needs SUIT_RENAME_SHARE of the reads to win, so a card whose two
    corners read two suits keeps one name instead of flipping with every
    missed corner; the suit changes neither the hand nor the count.
  * With any number of decks, a new face track within SAME_CORNER_MATCH_DISTS
    match distances of a face track bound to a card is that corner index read
    again, whatever name it reads, and joins the card: a second box on one
    corner is never a card of its own.
  * On the card: a face track reading a card's rank that lies on the card's
    body, between MID_CARD_ALONG of the way along the line between its two
    corner indices in view and within MID_CARD_ACROSS of that line, is a box on
    the card itself (a 2's pips read as an index) and joins it, whatever suit
    it reads. A card showing two face tracks also takes a third when one of the
    two lies on the body between the other and the new one: the pip came into
    view before the corner. And a card whose every live face track lies on the
    body of another card of its rank showing both corners is folded into that
    card with a MERGED event, held or counted: the pip was confirmed before the
    card's second corner and placed as a copy. The one layout this gets wrong
    is two cards of one rank fanned diagonally by about half a card, where the
    top card's near corner lands on the bottom card's body; the top card is
    then placed from its far corner alone.
  * One deck: a new track with a face label joins the card of that name if the
    round already has one. That collapses a card's corner indices, and a stray
    misread elsewhere cannot add a second King of Hearts. Face-down cards
    (BACK) are all different cards.
  * Two or three decks (TrackerConfig.decks): a second King of Hearts is a
    real card, so a new face track joins a card of its name only when it lies
    within the same-card distance of that card or one of its tracks: the
    card's other corner. That distance is TrackerConfig.same_card_dist
    (settings.SAME_CARD_DISTANCE), else SAME_CARD_MATCH_DISTS match distances,
    and at least DIAGONAL_SLACK times the card diagonal measured from face-down
    boxes (see card_diagonal), so a camera close to the table does not split
    every card in two. Farther away it starts a new card of the same name, a
    second copy. A card has two corner indices, so it takes at most two face
    tracks seen this frame: a third track of its name, even close by, starts a
    second copy. That keeps two copies fanned in one hand apart. A face-down
    card being revealed holds fewer than two live face tracks and still joins.
    Three rules keep one physical card from becoming two:
      - same corner (above): a second box on a corner, or a misread one, is
        not a copy; copies fanned a corner's width apart stay apart.
      - stale tracks: a bound track that missed this frame does not count
        toward the two, so a corner whose track was lost takes a new one.
      - out of sight: a card of its name whose every track missed this frame,
        or that is hidden, takes the new track at any distance: the card was
        moved. A real second copy arrives while the first is in view.
    The costs: beside a card already showing both corners, a stray misread of
    its name can place a phantom copy once it passes the confirmation checks;
    and a second copy confirmed while the first is covered joins the first.
  * The corner span: the tracker measures the distance between the two corner
    indices of the cards it sees with both in view (corner_span, the median,
    after SPAN_SAMPLES frames). Once known, two more rules use it:
      - a track of a card's name joining a card showing one corner must lie
        at least (1 - SPAN_TOLERANCE) of the span from that corner, else it is
        a second copy fanned over the card, and is placed (and held) as one;
      - a track reading the card's rank in another suit, about a span from a
        card's one corner in view, is the card's other corner misread (the
        most common misread swaps a suit of the same colour): it joins the
        card, whose name is the vote over both corners. Without the span, or
        with the card's two corners in view, such a track is a card of its own.
  * A card read twice is merged: a face card renamed to the name of a card
    that its tracks would have joined under that name (the nearest hidden card
    of that name anywhere, else the nearest within the same-card distance
    with which it shares at most two face tracks in view) is that card read
    twice, one corner misread until the vote corrected it. The card placed
    later is folded into the one placed first, as merge() folds it: it leaves
    the round with a MERGED event, its copy leaves the count, and its tracks
    join the card kept. A revealed hole card is never folded: it is a whole
    card. A renamed card not folded is held beside a card of its new name as
    a card placed there would be (below).
  * A held copy (with decks > 1 and copy_hold_seconds above 0): a face card
    placed within the same-card distance of another card of its
    name, holding no locked copy, while a copy of that name is left to count,
    may be that card read twice. It is placed and numbered but HELD: not
    counted. Once it and that card have both been seen together for
    copy_hold_seconds, it is COUNTED, and reported as a duplicate if every
    copy was dealt meanwhile. If that card is removed or renamed away, the held copy takes
    over its count and is counted at once. No other card hands a held copy its
    count, so it is never counted before its hold ends. A held copy
    renamed is placed again under its new name: held, or counted. A held copy
    still on the table across new_round() is placed afresh, as it holds no copy.
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
  * A card hidden for hidden_seconds is REMOVED: it was picked up (a hand the
    dealer collected mid-round). It leaves the round but stays in the shoe
    count. Its number is not reused; if it comes back it is placed again
    under a new number, and its name is not counted again.
  * The round ends when no card has been visible for round_end_seconds (the
    dealer has swept the table, or every card was removed), or when
    new_round() is called. Numbers restart at 1 for the next round.
  * shoe_seen lists every face card seen since the last shuffle, in the order
    first seen: each name once per copy, so at most `decks` times. The count
    is computed from it, so a card that leaves and comes back, even in a later
    round, is never counted again. A copy stays in it while a card of the round
    holds it, so renaming one of two cards that share a name does not uncount
    the other. A card still on the table as a new round begins is not counted
    again under the name it had.
  * shoe_unrevealed counts the face-down cards placed this shoe that were never
    renamed to a face (a hole card swept face down, say). They have left the
    deck, so shoe_count subtracts them from cards_left, but they add nothing to
    the running count. A face-down card still on the table across new_round()
    gets a new number but is not added again. Nor is a removed face-down card
    placed again in the same round within 3 * match_dist of where it lay: it
    gets a new number, as any removed card does, but is the card already dealt.
    Each spot stands for one card, so a second face-down card there is added.
    Cards put back close together are matched to spots one to one, nearest
    first, and an earlier match moves to another spot in its reach if that
    lets a later card keep the only spot it can reach.
  * shoe_pending and shoe_locked split shoe_seen in two, copy by copy. A copy
    counted this round is pending: a rename can still take it back. A copy
    counted before this round began is locked: it stays counted until the
    next shoe. With more than one deck a name can be both.
  * A card is counted only while its name has fewer than `decks` copies in
    shoe_seen. A card placed with a face name whose every copy is already
    dealt, counted before this round or held by another card of this round,
    is a DUPLICATE: a misread or a reshuffle nobody marked with new_shoe().
    A card renamed to a name with every copy counted before this round is one
    too; a rename may share a name with another card of this round while the
    vote settles which one is misread. A duplicate is reported, not counted
    again, and its name is added to shoe_duplicates. A card still on the
    table when the round began, and two corners of one card joining by name,
    are not duplicates.
  * rescan() takes back a misread while the round is still open. It forgets
    every card and track of this round and puts shoe_seen, shoe_unrevealed,
    and shoe_duplicates back as they were when the round began, so only the
    pending names leave the count. The cards on the table are then confirmed
    again from scratch and numbered from 1. A round that already ended is
    locked: rescan cannot reach back into it. A card still on the table from
    an earlier round is placed again after a rescan. Face up, it is counted
    again as another copy while the deck count allows, then reported as a
    duplicate. Face down, within 3 * match_dist of where it lay as the round
    began, it is not added to shoe_unrevealed again: the restored total holds it.
  * undo_last() and merge() fix one wrong placement and keep every other card.
    undo_last() takes back the card placed last, as if it was never placed: its
    copy leaves the count, or goes to a card of its name holding none, as on a
    rename. Its face-down share of shoe_unrevealed and a duplicate only it
    reported go too. Its tracks are forgotten, so a box still there is confirmed
    again and placed under a new number. A face-down card dealt before it was
    placed (carried into the round) stays dealt, and its spot is remembered as a
    removed card's is, so it is not added again when placed there anew. merge(4, 2) folds card 4 into card 2,
    one card placed twice: card 4 is taken back the same way, but its tracks
    join card 2. Number 4 is not used again this round.
  * new_shoe() clears shoe_seen, shoe_unrevealed, and shoe_duplicates after a
    reshuffle.

All thresholds live in TrackerConfig, in seconds and pixels.
TrackerConfig.from_settings keeps the seconds in settings.py as they are and
turns the fractions of the frame width into pixels.

Times run on the clock update() is given: read.py passes each frame's capture
time, so every setting holds in real time whatever the frame rate. An update()
given no time moves the clock one frame at nominal_fps, as in the tests.
With times given, a setting is reached once that many seconds have passed,
allowing only for float rounding (TIME_TOLERANCE). Without them, each update
is a whole frame, and half a frame short of a setting counts as reaching it:
each setting takes the whole number of frames nearest to it, as it did when
settings were turned into frames. Either way every time takes at least one
frame, except hidden_seconds or copy_hold_seconds of 0.

A track keeps its sightings for as long as its windows look back, however
many frames that is: the label window, and the confirm window until it is
confirmed. With no confirm window it keeps every sighting until then.

When the camera view is turned a quarter turn (keys < and > in read.py),
rotate() turns every track and card with the frame and rescales the pixel
thresholds to the new width, so cards on the table keep their numbers.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field, replace
from typing import NamedTuple

from .cards import Card, Detection, Event
from .deck import BACK, rank_of

SAME_CARD_MATCH_DISTS = 4  # with decks > 1: match distances within which a face track of a card's name is that card,
                           # when TrackerConfig.same_card_dist is not set (settings.SAME_CARD_DISTANCE sets it)
SAME_CORNER_MATCH_DISTS = 0.5  # with decks > 1: match distances within which a face track is a bound track's corner
SPAN_SAMPLES = 12       # frames showing both corners of a card before the measured corner span is trusted
SPAN_TOLERANCE = 0.35   # share of the corner span a track may be nearer or farther than it from a card's lone corner
                        # and still be the card's other corner. Generous: a tilted camera shows near cards bigger.
DIAGONAL_SAMPLES = 12   # face-down boxes measured before their diagonal may widen the same-card distance
DIAGONAL_SLACK = 1.2    # the same-card distance is at least this times the measured card diagonal
SUIT_RENAME_SHARE = 2 / 3  # share of a card's reads a name differing from its name only in suit needs to rename it:
                           # two corners read as two suits do not flip the name every few frames
MID_CARD_ALONG = (0.25, 0.75)  # a track this far along the line between a card's two corner indices, reading the
                               # card's rank, is on the card's body (its pips: a 2's lie at 0.31 and 0.69), not a copy.
                               # A card fanned straight over another lands outside this; one fanned diagonally by
                               # half a card lands inside, so the rule is kept to the card's own rank.
MID_CARD_ACROSS = 0.15         # ... and within this share of that line's length to either side of it
TIME_TOLERANCE = 1e-9  # seconds a time may fall short of a setting from float rounding in summed frame intervals

@dataclass
class TrackerConfig:
    """Every tracker threshold, in seconds and pixels, and the decks in the shoe.
    The defaults are permissive, for tests."""
    confirm_seconds: float = 0.65      # time seen needed ...
    confirm_window_seconds: float | None = None  # ... within this long (None: ever, keeping every sighting)
    confirm_agreement: float = 0.0     # share of recent reads that must agree on the rank
    confirm_confidence: float = 0.0    # minimum average confidence over those sightings
    still_distance: float | None = None  # pixels the corner may move over them (None: any)
    drop_seconds: float = 2.4          # time unseen before a track is forgotten
    match_dist: float = 40.0           # pixels: a detection closer than this to a track is that track
    round_end_seconds: float = 7.25    # time with no card visible before the round is over
    label_window_seconds: float | None = None  # time the name vote looks back (None: every read ever)
    hidden_seconds: float | None = None  # time a card may stay hidden before it is removed (None: never)
    decks: int = 1                     # decks in the shoe: copies of each card that may be counted
    copy_hold_seconds: float = 0.0     # with decks > 1: time a copy beside a card of its name waits (0: none)
    nominal_fps: float = 12.4          # the reader's usual frame rate: an update() given no time is one frame
                                       # at it, and its times are compared to the nearest such frame
    start_confidence: float = 0.0      # a detection scoring less is weak: it only keeps a track seen (see _match)
    same_card_dist: float | None = None  # pixels within which a face track of a card's name is its other corner
                                         # (None: SAME_CARD_MATCH_DISTS match distances); see CardTracker.same_card_dist

    @classmethod
    def from_settings(cls, s, frame_width: int) -> TrackerConfig:
        """From settings.py: distances are fractions of the frame width, times are seconds.
        decks keeps its default; pass settings.DECKS to CardTracker as a keyword.
        HIDDEN_SECONDS = 0 removes a card on the frame it is hidden; COPY_HOLD_SECONDS = 0
        counts a copy at once. CONFIDENCE becomes start_confidence."""
        return cls(confirm_seconds=s.CONFIRM_SECONDS, confirm_window_seconds=s.CONFIRM_WINDOW_SECONDS,
                   confirm_agreement=s.CONFIRM_AGREEMENT, confirm_confidence=s.CONFIRM_CONFIDENCE,
                   still_distance=s.STILL_DISTANCE * frame_width, drop_seconds=s.DROP_SECONDS,
                   match_dist=s.MATCH_DISTANCE * frame_width, round_end_seconds=s.ROUND_END_SECONDS,
                   label_window_seconds=s.LABEL_WINDOW_SECONDS, hidden_seconds=s.HIDDEN_SECONDS,
                   copy_hold_seconds=s.COPY_HOLD_SECONDS, start_confidence=s.CONFIDENCE,
                   same_card_dist=s.SAME_CARD_DISTANCE * frame_width)


class Hit(NamedTuple):
    """One sighting of a track."""
    time: float      # the tracker's clock on the frame it was seen
    interval: float  # seconds since the frame before: how long this sighting stands for
    conf: float
    x: float
    y: float
    label: str


@dataclass
class Track:
    id: int
    cx: float
    cy: float
    hits: deque = field(default_factory=deque)  # Hit of each sighting its windows still need, oldest first
    votes: Counter = field(default_factory=Counter)  # reads per label, within the label window
    label: str = ""                                   # the winning vote
    seen: int = 0          # sightings counted: weak ones are not
    missed: int = 0        # frames since the last sighting, weak ones included
    last_seen: float = 0.0  # the clock on the last sighting, weak ones included
    confirmed: bool = False


class CardTracker:
    def __init__(self, config: TrackerConfig | None = None, **overrides):
        """CardTracker(TrackerConfig.from_settings(settings, width)), or thresholds as keywords."""
        self.cfg = replace(config or TrackerConfig(), **overrides)
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.frame = 0       # updates so far: numbers each frame for events and the log
        self.now = 0.0       # the clock on the last update, in seconds
        self._interval = 0.0  # seconds from the update before to the last one
        self._slack = 0.0     # seconds a time may fall short of a setting on the last update (see _reached)
        self._card_of: dict[int, Card] = {}  # track id -> the card it belongs to
        self._dealt_backs: set[int] = set()  # ids of tracks of face-down cards already in shoe_unrevealed
        self.bound: list[Card | None] = []   # per detection of the last update: the confirmed card it matched
        # The cards' size in this camera's pixels, measured as they are seen (see corner_span, card_diagonal).
        # Kept across rounds and shoes: the camera does not move when the cards do.
        self._corner_spans: deque = deque(maxlen=600)    # (card, pixels between its two corner indices), per frame
        self._card_diagonals: deque = deque(maxlen=600)  # pixels corner to corner of a face-down card's box
        self.new_shoe()

    def new_round(self):
        """Start a new round: forget the table, numbering restarts at 1. The count is kept.

        A card still on the table is placed again with a new number, but it is
        the card already dealt: it is not reported as a duplicate, and a
        face-down one is not added to shoe_unrevealed twice."""
        self._dealt_backs |= {track_id for track_id, card in self._card_of.items() if card.label == BACK}
        # where the face-down cards still on the table as the round began lay, for rescan()
        carried_backs = {card.order: card for card in self._card_of.values() if card.label == BACK}
        self._carried_back_spots = [(card.x, card.y) for card in carried_backs.values()]
        self._dealt_back_spots: list[tuple[float, float]] = []  # face-down cards in shoe_unrevealed, off the table
        self._back_spot_holders: dict[int, tuple[float, float]] = {}  # spot index -> where the card put back lies
        # track id -> name its card was counted under, for cards still on the table as the round began.
        # A held copy was never counted, so it is placed afresh.
        self._carried_tracks = {track_id: card.label for track_id, card in self._card_of.items() if not card.held}
        self._held_against: dict[int, Card] = {}  # order of a held copy -> the card of its name it is held against
        self._seen_before_round = list(self.shoe_seen)  # the locked copies: shoe_seen always starts with them
        self._unrevealed_before_round = self.shoe_unrevealed
        self._duplicates_before_round = set(self.shoe_duplicates)
        self._unrevealed_by: dict[int, int] = {}  # order -> what that card added to shoe_unrevealed this round
        self._duplicate_reporters: dict[str, set[int]] = {}  # name -> orders of the cards reporting it this round
        self.round: list[Card] = []
        self._card_of = {}
        self._carried_cards: dict[int, str] = {}  # order -> name counted under, for cards placed from _carried_tracks
        self._next_order = 1  # placement numbers are never reused within a round, even after a removal
        self._last_busy = self.now  # the last time a card was visible, or nothing was placed yet

    def new_shoe(self):
        """After a reshuffle: a new round, and every card can be counted again."""
        self.shoe_seen: list[str] = []
        self.shoe_unrevealed = 0  # face-down cards placed this shoe and never renamed to a face
        self.shoe_duplicates: set[str] = set()  # face names reported as a duplicate this shoe
        self.new_round()
        self._carried_tracks = dict.fromkeys(self._carried_tracks)  # still not duplicates, but counted afresh
        self._dealt_backs = set()
        self._carried_back_spots = []

    def rescan(self) -> bool:
        """Look at the table again: forget every card and track of this round, keep the rounds before it.

        Names first counted this round leave the count; names counted earlier stay, in order.
        shoe_unrevealed and shoe_duplicates go back to their values when the round began.
        Cards on the table are confirmed again from scratch and numbered from 1. A face-down
        card that was on the table as the round began is already in shoe_unrevealed, so one
        placed again near where it lay then is not added to it again.
        Returns True when this round had placed a card, even one since removed: only then
        was there anything to take back."""
        placed_this_round = self._next_order > 1
        self.shoe_seen = list(self._seen_before_round)
        self.shoe_unrevealed = self._unrevealed_before_round
        self.shoe_duplicates = set(self._duplicates_before_round)
        self.tracks = {}
        self._card_of = {}
        self._dealt_backs = set()
        self._dealt_back_spots = list(self._carried_back_spots)
        self._back_spot_holders = {}
        self._carried_tracks = {}
        self._carried_cards = {}
        self._held_against = {}
        self._unrevealed_by = {}
        self._duplicate_reporters = {}
        self.bound = []
        self.round = []
        self._next_order = 1
        self._last_busy = self.now
        return placed_this_round

    def undo_last(self) -> Card | None:
        """Take back the card placed last of those still in the round, hidden or not, as if it was never placed.

        The copy it counted leaves the count, or goes to another card of its name holding none, as
        a rename hands it over. What it added to shoe_unrevealed comes off, and a duplicate only it
        reported is no longer one. Every other card keeps its number and its count. Its tracks are
        forgotten, so a box still on its spot must pass the confirmation checks again and is then
        placed anew, under a new number. A face-down card that added nothing, being one dealt
        already (carried into the round, or put back on a removed card's spot), leaves its spot in
        _dealt_back_spots as a removed card does, so placed again there it is not added again.
        Returns the card taken back, or None when the round has none."""
        if not self.round:
            return None
        card = self.round[-1]
        track_ids = [track_id for track_id, bound in self._card_of.items() if bound is card]
        dealt_before = card.label == BACK and not self._unrevealed_by.get(card.order, 0)
        self._take_back(card)
        if dealt_before:
            self._dealt_back_spots.append((card.x, card.y))
        for track_id in track_ids:
            del self.tracks[track_id]
            del self._card_of[track_id]
            self._dealt_backs.discard(track_id)
            self._carried_tracks.pop(track_id, None)
        self.bound = [None if bound is card else bound for bound in self.bound]
        return card

    def merge(self, order_from: int, order_to: int) -> Card | None:
        """Fold card order_from into card order_to: one physical card that was placed twice.

        Card order_from leaves the round as undo_last takes a card back, but its tracks are bound
        to card order_to, whose name is then the vote over all of them. So the copy it counted
        goes to card order_to when that card has its name and holds none. order_from's number is
        not used again this round. Returns the card folded in, or None, with nothing changed, unless both numbers
        are different cards of the round."""
        cards = {card.order: card for card in self.round}
        merged, survivor = cards.get(order_from), cards.get(order_to)
        if merged is None or survivor is None or merged is survivor:
            return None
        self._take_back(merged)
        for track_id, bound in self._card_of.items():
            if bound is merged:
                self._card_of[track_id] = survivor
        self.bound = [survivor if bound is merged else bound for bound in self.bound]
        return merged

    def _take_back(self, card: Card):
        """Take a card out of the round and undo what it did to the shoe (see undo_last). Its tracks
        stay bound to it: the caller forgets or rebinds them. A card held against it is counted on
        the next update, as when a card is removed."""
        self._release_copy(card)
        if card.held:
            card.held, card.held_seconds = False, 0.0
            del self._held_against[card.order]
        self.shoe_unrevealed -= self._unrevealed_by.pop(card.order, 0)
        for label, reporters in self._duplicate_reporters.items():
            reporters.discard(card.order)
            if not reporters and label not in self._duplicates_before_round:
                self.shoe_duplicates.discard(label)
        self._carried_cards.pop(card.order, None)
        self.round.remove(card)

    @property
    def shoe_pending(self) -> set[str]:
        """Face names with a copy first counted this round. A rename can still take it back."""
        return set(self.pending_copies)

    @property
    def shoe_locked(self) -> set[str]:
        """Face names with a copy counted before this round began. It stays counted until new_shoe()."""
        return set(self._seen_before_round)

    @property
    def pending_copies(self) -> Counter:
        """Copies of each face name first counted this round."""
        return Counter(self.shoe_seen) - Counter(self._seen_before_round)

    @property
    def locked_copies(self) -> Counter:
        """Copies of each face name counted before this round began."""
        return Counter(self._seen_before_round)

    def update(self, detections: list[Detection], now: float | None = None) -> list[Event]:
        """Advance one frame with that frame's detections.

        now: the frame's capture time in seconds, on a clock that only moves forward
        (time.monotonic(), or a video file's position); each setting then takes at least
        its seconds, and the first frame, with no frame before it, stands for no time.
        None moves the clock one frame at nominal_fps, so a run of updates without times
        is a camera at that rate whose settings round to whole frames.
        Give a time with every frame or with none.

        Returns the events this frame caused: placed cards first, then renamed,
        hidden, and returned cards in placement order, then removed cards, then
        held copies counted, and round_over last. A placed or renamed event is
        followed by held when the card is held as a possible second read of a card
        beside it, or by duplicate when every copy of the card's name was already
        dealt; a counted event may be followed by duplicate too (see the module
        docstring). A removed card has left the round; round_over also starts a new round.

        Also sets bound: one entry per detection, in the same order, giving the
        confirmed card its track belongs to, or None while that track is not
        confirmed or for a weak detection matched to no track. A detection misread
        as another name, or a weak one, is still bound to its card."""
        self.frame += 1
        nominal_interval = 1 / self.cfg.nominal_fps
        if now is None:
            now = self.frame * nominal_interval
            self._slack = nominal_interval / 2
            self._interval = nominal_interval  # every untimed update, the first too, is one nominal frame
        else:
            self._slack = TIME_TOLERANCE
            # The first timed frame has no frame before it, so it stands for no time: a sighting on it
            # never counts for time that was not measured.
            self._interval = max(now - self.now, 0.0) if self.frame > 1 else 0.0
        self.now = now
        det_tracks = self._match(detections)
        for track in self.tracks.values():
            if not track.confirmed and self._passes_checks(track):
                track.confirmed = True
        events: list[Event] = []
        # Track ids grow with first sighting, so earlier cards get lower numbers.
        by_age = sorted(self.tracks.values(), key=lambda track: track.id)
        confirmed = [track for track in by_age if track.confirmed]

        # A track already bound to a card stays with it. Any other joins the card
        # with its name (with more decks: nearby, or out of sight), a face-down card near it, or starts a new one.
        tracks_of: dict[int, list[Track]] = {}  # card order -> its tracks this frame
        for track in confirmed:
            card = self._card_of.get(track.id)
            if card is None:
                card = self._card_for(track)
                if card is None:
                    card = Card(self._next_order, track.label, track.cx, track.cy, self.frame)
                    self._next_order += 1
                    self.round.append(card)
                    carried = track.id in self._carried_tracks
                    carried_label = self._carried_tracks.get(track.id)
                    if carried_label is not None:
                        self._carried_cards[card.order] = carried_label
                    if card.label == BACK and not self._already_dealt_back(track):
                        self._add_unrevealed(card, 1)
                    events.append(Event("placed", card, self.frame))
                    if not carried and self._hold_if_copy(card):
                        events.append(Event("held", card, self.frame))
                    else:
                        if not self._holds_locked_copy(card):
                            self._count(card)
                        if not carried:
                            events += self._duplicate_events(card)
                self._card_of[track.id] = card
            tracks_of.setdefault(card.order, []).append(track)

        for card in self.round:
            self._measure_span(card, tracks_of.get(card.order, []))

        for card in list(self.round):
            if card not in self.round:
                continue  # folded into a card renamed earlier in this loop (see _fold)
            card_tracks = tracks_of.get(card.order)
            if not card_tracks:
                if not card.hidden:
                    card.hidden = True
                    card.hidden_since = self.now
                    events.append(Event("hidden", card, self.frame))
                continue
            votes = sum((track.votes for track in card_tracks), Counter())
            label = self._voted_label(card, votes)
            if label != card.label:
                was, was_held = card.label, card.held
                # A face card renamed to the name of a card it would have joined under that name was that
                # card read twice: one physical card, two numbers. A revealed hole card is a whole card.
                twin = self._read_twice_of(card, label) if was != BACK and label != BACK else None
                if twin is not None:
                    kept, fold_events = self._fold(card, twin, label)
                    events += fold_events
                    if kept is not card:
                        continue
                    card_tracks = card_tracks + tracks_of.pop(twin.order, [])
                    if card.held:
                        events.append(Event("held", card, self.frame))
                else:
                    self._rename(card, label)
                    events.append(Event("renamed", card, self.frame, was=was))
                    if card.held:
                        events.append(Event("held", card, self.frame))
                    elif was_held:
                        events += self._duplicate_events(card)
                    elif self._is_duplicate(label):
                        events.append(self._report_duplicate(card))
            if card.hidden:
                card.hidden = False
                events.append(Event("returned", card, self.frame))
            card.x, card.y = card_tracks[0].cx, card_tracks[0].cy
        for card in self._expired_cards():
            self._remove(card)
            events.append(Event("removed", card, self.frame))
        for card in list(self.round):
            events += self._fold_if_on_card(card)
        for card in self.round:
            if card.held:
                events += self._advance_hold(card)
        self.bound = [self._card_of.get(track_id) for track_id in det_tracks]

        # Counts on after every card was removed, so a collected table still ends the round.
        placed_this_round = self._next_order > 1
        if not placed_this_round or tracks_of:
            self._last_busy = self.now
        elif self._reached(self.now - self._last_busy, self.cfg.round_end_seconds):
            events.append(Event("round_over", None, self.frame))
            self.new_round()
        return events

    def _reached(self, elapsed: float, seconds: float) -> bool:
        """Whether elapsed seconds reach `seconds`. With times given, only float rounding may fall
        short; without, half a nominal frame may, so a time rounds to whole frames (see the module docstring)."""
        return elapsed >= seconds - self._slack

    def _within(self, then: float, seconds: float | None) -> bool:
        """Whether the time `then` lies within the last `seconds` (None: ever), rounded as _reached
        rounds. This frame always does."""
        return seconds is None or then >= self.now or not self._reached(self.now - then, seconds)

    def rotate(self, quarter_turns: int, width: int, height: int):
        """Follow the frame turned clockwise by quarter_turns (negative: anticlockwise), as camera.rotate turns it.

        width, height: the frame size before the turn. Every track, its recent sightings, and every
        card of the round move with the frame, so the next frame's detections land on them and no
        card is placed again. still_distance and match_dist are rescaled to the new frame width."""
        for _ in range(quarter_turns % 4):
            for track in self.tracks.values():
                track.cx, track.cy = _quarter_turn(track.cx, track.cy, height)
                turned_hits = []
                for hit in track.hits:
                    x, y = _quarter_turn(hit.x, hit.y, height)
                    turned_hits.append(hit._replace(x=x, y=y))
                track.hits = deque(turned_hits)
            for card in self.round:
                card.x, card.y = _quarter_turn(card.x, card.y, height)
            self._dealt_back_spots = [_quarter_turn(x, y, height) for x, y in self._dealt_back_spots]
            self._carried_back_spots = [_quarter_turn(x, y, height) for x, y in self._carried_back_spots]
            self._back_spot_holders = {index: _quarter_turn(x, y, height)
                                       for index, (x, y) in self._back_spot_holders.items()}
            still, same_card = self.cfg.still_distance, self.cfg.same_card_dist
            self.cfg = replace(self.cfg, match_dist=self.cfg.match_dist * height / width,
                               still_distance=None if still is None else still * height / width,
                               same_card_dist=None if same_card is None else same_card * height / width)
            width, height = height, width

    def table(self) -> list[Card]:
        """Cards visible right now, in the order they were placed."""
        return [c for c in self.round if not c.hidden]

    def _card_for(self, track: Track) -> Card | None:
        """The card of this round an unbound track belongs to, if any.

        A face track within SAME_CORNER_MATCH_DISTS match distances of a face
        track bound to a card is that card's corner index read again, whatever
        name it reads. A face track on the body of a card of its rank showing
        both corners (see _card_under) is a box on that card, a pip read as an
        index, and joins it. Otherwise it joins the card with its name, or a
        face-down card being revealed whose other corner already reads that
        name. With one deck that card may be anywhere. With more, in this order:
          1. the nearest such card out of sight this frame, at any distance:
             the card moved;
          2. the nearest such card within same_card_dist with a corner index
             free for it (see _takes_corner): fewer than two face tracks seen
             this frame, since a card has only two, and the track not much
             nearer its one corner than the corner span, once that is measured.
        With no card of its name it joins a card of its rank reading another
        suit, a corner span from that card's one corner in view (see
        _card_of_rank): one corner misread. Otherwise it joins a nearby
        face-down card that is out of sight this frame: the hole card it was
        before it was turned over. A BACK track only joins a hidden face-down
        card near it."""
        if track.label == BACK:
            return self._back_card_near(track.cx, track.cy, include_unseen=False)
        same_corner = self._card_at_corner(track)
        if same_corner is not None:
            return same_corner
        under = self._card_under(track)
        if under is not None:
            return under
        named = [card for card in self.round if card.label == track.label
                 or (card.label == BACK and any(t.label == track.label for t in self._tracks_of_card(card)))]
        if self.cfg.decks > 1:
            by_distance = sorted((self._distance_to(card, track.cx, track.cy), card.order, card) for card in named)
            unseen = [card for _, _, card in by_distance if self._out_of_sight(card)]
            if unseen:
                return unseen[0]
            limit = self.same_card_dist()
            named = [card for dist, _, card in by_distance if dist <= limit and self._takes_corner(card, track)]
        if named:
            return named[0]
        other_suit = self._card_of_rank(track)
        if other_suit is not None:
            return other_suit
        return self._back_card_near(track.cx, track.cy, include_unseen=True)

    def same_card_dist(self) -> float:
        """Pixels within which a face track of a card's name is that card's other corner: the config's
        same_card_dist, else SAME_CARD_MATCH_DISTS match distances; and at least DIAGONAL_SLACK times
        the measured card diagonal, so cards bigger than the setting allows for, a camera close to the
        table, still join their own corners once a face-down card has shown how big they are."""
        cfg = self.cfg
        limit = cfg.same_card_dist if cfg.same_card_dist is not None else SAME_CARD_MATCH_DISTS * cfg.match_dist
        diagonal = self.card_diagonal
        return limit if diagonal is None else max(limit, DIAGONAL_SLACK * diagonal)

    @property
    def corner_span(self) -> float | None:
        """Pixels between the two corner indices of one card: the median over the last frames that
        showed a card's two corners, once SPAN_SAMPLES have; None before. Measured from the cards
        themselves, so it fits the camera's height and zoom without a setting."""
        return self._span_excluding(None)

    def _span_excluding(self, card: Card | None) -> float | None:
        """The corner span measured from every card but `card` (see corner_span). A rule deciding
        whether one of a card's own tracks is a pip must not lean on spans measured from that pip."""
        spans = [span for owner, span in self._corner_spans if owner is not card]
        return _median(spans) if len(spans) >= SPAN_SAMPLES else None

    @property
    def card_diagonal(self) -> float | None:
        """Pixels corner to corner of a face-down card's box, the median over the last BACK boxes with a
        size, once DIAGONAL_SAMPLES have been seen; None before."""
        return _median(self._card_diagonals) if len(self._card_diagonals) >= DIAGONAL_SAMPLES else None

    def _forget_spans(self, card: Card):
        """Drop the corner span samples measured from `card`: one of its two tracks has turned out to
        be a pip on the card, not its corner (see _takes_corner)."""
        kept = [(owner, span) for owner, span in self._corner_spans if owner is not card]
        self._corner_spans = deque(kept, maxlen=self._corner_spans.maxlen)

    def _corners_of(self, card: Card) -> tuple[Track, Track] | None:
        """The card's two corner indices in view: of its face tracks seen this frame, the two farthest
        apart, so a pip bound to the card between them is left out. None with fewer than two."""
        faces = self._face_tracks_in_view(card)
        if len(faces) < 2:
            return None
        pairs = [(faces[i], faces[j]) for i in range(len(faces)) for j in range(i + 1, len(faces))]
        return max(pairs, key=lambda pair: _distance(*pair))

    def _measure_span(self, card: Card, card_tracks: list[Track]):
        """Record the distance between the card's two corner indices when both are seen this frame. Two
        boxes within a match distance are one corner read twice, not two corners, and are skipped."""
        faces = [track for track in card_tracks if track.label != BACK and track.missed == 0]
        if len(faces) == 2:
            span = _distance(faces[0], faces[1])
            if span > self.cfg.match_dist:
                self._corner_spans.append((card, span))

    def _card_under(self, track: Track) -> Card | None:
        """The card the track lies on the body of: a face card of the track's rank with two face tracks
        seen this frame, the track between them as _on_body says, nearest to the line between them when
        several. A box on a card's pips reads the card's rank and sits mid-card; a second copy fanned
        straight over a card puts its corner near one end of that line, outside the band."""
        if track.label == BACK:
            return None
        best: tuple[float, Card] | None = None
        for card in self.round:
            if card.label == BACK or rank_of(card.label) != rank_of(track.label):
                continue
            corners = self._corners_of(card)
            if corners is None:
                continue
            across = _on_body(*corners, track)
            if across is not None and (best is None or across < best[0]):
                best = (across, card)
        return None if best is None else best[1]

    def _takes_corner(self, card: Card, track: Track) -> bool:
        """Whether the card has a corner index free for the track: fewer than two face tracks seen this
        frame and, with one and the corner span measured, the track at least (1 - SPAN_TOLERANCE) of the
        span from it. A track of a card's name much nearer its one corner than its other corner could
        be is a second copy fanned over it, not that corner. A card showing two face tracks takes a
        third when one of the two lies on the body between the other and the new track: that one was
        a pip taken for the corner before the corner came into view, and the spans measured from it
        while it passed for a corner are forgotten. Once other cards have given the span, the new track
        must also lie about a span from the remaining corner: a copy of the same rank fanned along the
        card's diagonal puts the card's far corner where such a pip would be, and its own far corner
        well beyond a span."""
        faces = self._face_tracks_in_view(card)
        if len(faces) == 2:
            for corner, pip in ((faces[0], faces[1]), (faces[1], faces[0])):
                if _on_body(corner, track, pip) is None:
                    continue
                span = self._span_excluding(card)
                if span is not None and abs(_distance(corner, track) - span) > SPAN_TOLERANCE * span:
                    return False
                self._forget_spans(card)
                return True
            return False
        if len(faces) > 2:
            return False
        span = self.corner_span
        if len(faces) == 1 and span is not None:
            return _distance(faces[0], track) >= (1 - SPAN_TOLERANCE) * span
        return True

    def _card_of_rank(self, track: Track) -> Card | None:
        """The card the track is the other corner of though it reads another suit: a face card of its
        rank, reading another suit, with one face track seen this frame, the track within SPAN_TOLERANCE
        of a corner span from it; the nearest when several. None until the corner span is measured.
        A card with one corner read as the wrong suit then stays one card, its name the vote over both
        corners, instead of being counted under both suits. A card of its rank fanned beside it, a pair,
        shows its corner far nearer or farther than a span and is not taken."""
        span = self.corner_span
        if span is None:
            return None
        best: tuple[float, Card] | None = None
        for card in self.round:
            if card.label in (BACK, track.label) or rank_of(card.label) != rank_of(track.label):
                continue
            faces = self._face_tracks_in_view(card)
            if len(faces) != 1:
                continue
            dist = _distance(faces[0], track)
            if abs(dist - span) <= SPAN_TOLERANCE * span and (best is None or dist < best[0]):
                best = (dist, card)
        return None if best is None else best[1]

    def _read_twice_of(self, card: Card, label: str) -> Card | None:
        """The card of the round that card, a face card about to be renamed to label, is a second read
        of: the card of that name its tracks would have joined had they read it from the start (see
        _card_for). The nearest hidden card of that name, at any distance: it was moved and misread
        where it landed. Else the nearest within same_card_dist with which it shares at most two face
        tracks seen this frame, one card's two corners. None when there is none."""
        tracks = self._tracks_of_card(card)
        candidates = []
        for other in self.round:
            if other is card or other.label != label:
                continue
            if tracks:
                dist = min(self._distance_to(other, track.cx, track.cy) for track in tracks)
            else:
                dist = self._distance_to(other, card.x, card.y)
            candidates.append((dist, other.order, other))
        candidates.sort()
        hidden = [other for _, _, other in candidates if other.hidden]
        if hidden:
            return hidden[0]
        limit = self.same_card_dist()
        in_view = len(self._face_tracks_in_view(card))
        for dist, _, other in candidates:
            if dist <= limit and len(self._face_tracks_in_view(other)) + in_view <= 2:
                return other
        return None

    def _fold(self, card: Card, twin: Card, label: str) -> tuple[Card, list[Event]]:
        """card, about to be renamed to label, was twin read twice (see _read_twice_of): fold the one
        placed later into the one placed first, so one physical card keeps one number, the lower.
        Folding uses merge(): the folded card's copy leaves the count and its tracks join the card kept,
        whose name is then the vote over both. When card is kept it is renamed to label first.
        Returns the card kept and the events: renamed when card is kept, then merged."""
        if card.order < twin.order:
            was = card.label
            self.merge(twin.order, card.order)  # before the rename, so the copy twin gives up is not card's
            self._rename(card, label)
            return card, [Event("renamed", card, self.frame, was=was), Event("merged", twin, self.frame, into=card)]
        self.merge(card.order, twin.order)
        return twin, [Event("merged", card, self.frame, into=twin)]

    def _voted_label(self, card: Card, votes: Counter) -> str:
        """The name the card's tracks vote for. A name differing from the card's only in suit needs
        SUIT_RENAME_SHARE of the reads to win while the card's name still has some: a card whose two
        corners read two suits keeps one name instead of flipping whenever one corner is missed.
        The suit changes neither the hand nor the count, so nothing waits on it."""
        label = votes.most_common(1)[0][0]
        current = card.label
        if (label != current and BACK not in (label, current) and rank_of(label) == rank_of(current)
                and votes[current] and votes[label] < SUIT_RENAME_SHARE * votes.total()):
            return current
        return label

    def _distance_to(self, card: Card, x: float, y: float) -> float:
        """Pixels from (x, y) to the nearest of the card's last position and its tracks."""
        points = [(card.x, card.y)] + [(t.cx, t.cy) for t in self._tracks_of_card(card)]
        return min(((px - x) ** 2 + (py - y) ** 2) ** 0.5 for px, py in points)

    def _tracks_of_card(self, card: Card) -> list[Track]:
        return [self.tracks[track_id] for track_id, bound in self._card_of.items() if bound is card]

    def _face_tracks_in_view(self, card: Card) -> list[Track]:
        """Tracks bound to the card that read a face name and were seen this frame.
        A track that missed this frame is left out, so a corner it lost can be taken by a new track."""
        return [track for track in self._tracks_of_card(card) if track.label != BACK and track.missed == 0]

    def _out_of_sight(self, card: Card) -> bool:
        """Whether every track of the card missed this frame. A hidden card, whose tracks were dropped, is too."""
        return all(track.missed > 0 for track in self._tracks_of_card(card))

    def _card_at_corner(self, track: Track) -> Card | None:
        """The card of this round with a bound face track within SAME_CORNER_MATCH_DISTS match
        distances of the track, nearest first: the same corner index, whatever name it reads.
        With any number of decks: a second box on one corner is never a card of its own."""
        best, best_dist = None, SAME_CORNER_MATCH_DISTS * self.cfg.match_dist
        for track_id, card in self._card_of.items():
            other = self.tracks[track_id]
            if other.label == BACK or card not in self.round:
                continue
            dist = ((other.cx - track.cx) ** 2 + (other.cy - track.cy) ** 2) ** 0.5
            if dist <= best_dist:
                best, best_dist = card, dist
        return best

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

    def _already_dealt_back(self, track: Track) -> bool:
        """Whether a face-down card placed from this track is one shoe_unrevealed already holds:
        a card carried into the round on this track, or one off the table (removed this round, or
        carried in before a rescan) whose spot lies within 3 match distances. See _claim_back_spot."""
        return track.id in self._dealt_backs or self._claim_back_spot((track.cx, track.cy), set())

    def _claim_back_spot(self, point: tuple[float, float], tried: set[int]) -> bool:
        """Give the face-down card placed at point a spot of _dealt_back_spots within 3 match
        distances, nearest first. Each spot takes one card: a spot already taken is given up when
        the card holding it can take another in its own reach, so cards put back close together
        each find one. tried: spots already considered in this search. Returns whether one was found."""
        limit = 3 * self.cfg.match_dist
        in_reach = []
        for index, (x, y) in enumerate(self._dealt_back_spots):
            dist = ((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5
            if dist <= limit and index not in tried:
                in_reach.append((dist, index))
        for _, index in sorted(in_reach):
            tried.add(index)
            holder = self._back_spot_holders.get(index)
            if holder is None or self._claim_back_spot(holder, tried):
                self._back_spot_holders[index] = point
                return True
        return False

    def _expired_cards(self) -> list[Card]:
        """Cards of the round hidden for at least hidden_seconds, in placement order."""
        limit = self.cfg.hidden_seconds
        if limit is None:
            return []
        return [card for card in self.round if card.hidden and self._reached(self.now - card.hidden_since, limit)]

    def _remove(self, card: Card):
        """Take a card off the table. Its name stays in the shoe count.

        A counted card hands the count to a card of the round with its name that
        holds no copy yet, if there is one (see _copy_taker), so a later rename of that
        card still corrects it.
        A face-down card stays in shoe_unrevealed, and its spot is kept for _already_dealt_back."""
        self.round.remove(card)
        self._held_against.pop(card.order, None)
        if card.label == BACK:
            self._dealt_back_spots.append((card.x, card.y))
        if card.counted:
            card.counted = False
            holder = self._copy_taker(card)
            if holder is not None:
                holder.counted = True

    def _holds_locked_copy(self, card: Card) -> bool:
        """A card still on the table as the round began, under the name it was counted by then."""
        return self._carried_cards.get(card.order) == card.label

    def _copy_taker(self, card: Card) -> Card | None:
        """The card of the round to take over the copy of card's name that card gives up, if any.

        It holds no copy yet. A copy held against card comes first: it was most likely card,
        moved or misread. A copy held against another card takes none, so it is not counted
        before its hold ends."""
        takers = [c for c in self.round if c is not card and c.label == card.label
                  and not c.counted and not self._holds_locked_copy(c)]
        held_copy = next((c for c in takers if self._held_against.get(c.order) is card), None)
        if held_copy is not None:
            return held_copy
        return next((c for c in takers if not c.held), None)

    def _is_duplicate(self, label: str, in_round: int = 0) -> bool:
        """A face name with every copy already dealt: `decks` copies counted before this
        round, together with in_round other cards of this round placed under that name.
        in_round leaves out cards carried in from the last round: their copies are locked."""
        return label != BACK and self._seen_before_round.count(label) + in_round >= self.cfg.decks

    def _count(self, card: Card, at: int | None = None):
        """Add the card's name to the shoe count, at most once per deck. at: position to keep the order."""
        if card.label != BACK and self.shoe_seen.count(card.label) < self.cfg.decks:
            self.shoe_seen.insert(len(self.shoe_seen) if at is None else at, card.label)
            card.counted = True

    def _rename(self, card: Card, label: str):
        """The vote changed its mind: same card, new name, and the count follows.

        The old name's copy leaves the count unless another card of the round with that
        name holds no copy yet (see _copy_taker); that card then keeps the copy counted. A card carried in
        from the last round already holds a locked copy, so it takes none, and renamed
        back to the name it was carried under it is not counted again. Beside a card of its
        new name it is held (see _hold_if_copy), as a card placed there would be, whether or
        not it was held before; else it is counted at once."""
        if card.held:
            card.held, card.held_seconds = False, 0.0
            del self._held_against[card.order]
        at = self._release_copy(card)
        if card.label == BACK and label != BACK:
            self._add_unrevealed(card, -1)
        elif card.label != BACK and label == BACK:
            self._add_unrevealed(card, 1)
        card.label = label
        if self._hold_if_copy(card):
            return
        if not self._holds_locked_copy(card):
            self._count(card, at)

    def _release_copy(self, card: Card) -> int | None:
        """The card gives up the copy of its name it counted, if it counted one. Another card of the
        round with that name holding no copy takes it over (see _copy_taker); else the name's latest
        copy, the one counted this round, leaves shoe_seen. Returns the index it left, else None."""
        if not card.counted:
            return None
        card.counted = False
        holder = self._copy_taker(card)
        if holder is not None:
            holder.counted = True
            return None
        at = len(self.shoe_seen) - 1 - self.shoe_seen[::-1].index(card.label)
        self.shoe_seen.pop(at)
        return at

    def _add_unrevealed(self, card: Card, change: int):
        """Change shoe_unrevealed for this card, remembering the change so undo_last can take it back."""
        self.shoe_unrevealed += change
        self._unrevealed_by[card.order] = self._unrevealed_by.get(card.order, 0) + change

    def _report_duplicate(self, card: Card) -> Event:
        """Add the card's name to shoe_duplicates, remembering which card reported it, and return the event."""
        self.shoe_duplicates.add(card.label)
        self._duplicate_reporters.setdefault(card.label, set()).add(card.order)
        return Event("duplicate", card, self.frame)

    def _hold_if_copy(self, card: Card) -> bool:
        """Hold a face card, just placed or renamed while held, that may be a card of its name read a
        second time. That needs decks > 1, copy_hold_seconds above 0, a copy of its name still left to
        count, and another card of its name holding no locked copy within SAME_CARD_MATCH_DISTS match
        distances. The card is held against the nearest such card and not counted. Returns whether it was held."""
        cfg = self.cfg
        if cfg.decks == 1 or cfg.copy_hold_seconds == 0 or card.label == BACK:
            return False
        if self.shoe_seen.count(card.label) >= cfg.decks:
            return False  # nothing left to count: the duplicate check reports it at once
        limit = self.same_card_dist()
        near = []
        for other in self.round:
            if other is card or other.label != card.label or self._holds_locked_copy(other):
                continue
            dist = self._distance_to(other, card.x, card.y)
            if dist <= limit:
                near.append((dist, other.order, other))
        if not near:
            return False
        card.held, card.held_seconds = True, 0.0
        self._held_against[card.order] = min(near)[2]
        return True

    def _fold_if_on_card(self, card: Card) -> list[Event]:
        """Fold a face card whose every live face track lies on the body of another card of its rank
        showing both corners this frame into that card (see _card_under): it is a box on that card's
        pips that was confirmed before the card's second corner and so could not be told from a copy.
        Held or already counted, it is folded as merge() folds a card, with a merged event. A card
        with a track off that body, a corner of its own, is a card of its own. Returns the events."""
        if card.label == BACK or card not in self.round:
            return []
        faces = [track for track in self._tracks_of_card(card) if track.label != BACK]
        if not faces or not any(track.missed == 0 for track in faces):
            return []
        for other in self.round:
            if other is card or other.label == BACK or rank_of(other.label) != rank_of(card.label):
                continue
            corners = self._corners_of(other)
            if corners is None or not all(_on_body(*corners, track) is not None for track in faces):
                continue
            span = self._span_excluding(other)  # other's corners must be a span apart, as other cards show it
            if span is not None and abs(_distance(*corners) - span) > SPAN_TOLERANCE * span:
                continue
            self.merge(card.order, other.order)
            return [Event("merged", card, self.frame, into=other)]
        return []

    def _advance_hold(self, card: Card) -> list[Event]:
        """One frame of a held copy's wait. A frame on which it and the card it is held against both
        have a track seen adds its interval; once those add up to copy_hold_seconds it is counted. It is counted
        at once when that card has left the round, handing its copy over (see _remove), or changed its
        name. Returns a counted event, followed by duplicate when every copy of its name was already dealt."""
        against = self._held_against[card.order]
        if against in self.round and against.label == card.label:
            if not self._out_of_sight(card) and not self._out_of_sight(against):
                card.held_seconds += self._interval
            if not self._reached(card.held_seconds, self.cfg.copy_hold_seconds):
                return []
        card.held, card.held_seconds = False, 0.0
        del self._held_against[card.order]
        events = [Event("counted", card, self.frame)]
        if not card.counted:
            self._count(card)
            events += self._duplicate_events(card)
        return events

    def _duplicate_events(self, card: Card) -> list[Event]:
        """For a card just placed or counted after a hold: a duplicate event when every copy of its
        name was already dealt, counted before this round or placed by another card of this round
        that is not held. The name joins shoe_duplicates. Cards carried in from the last round are
        left out of the cards of this round: their copies are locked."""
        same_name = sum(1 for other in self.round if other is not card and other.label == card.label
                        and not self._holds_locked_copy(other) and not other.held)
        if not self._is_duplicate(card.label, same_name):
            return []
        return [self._report_duplicate(card)]

    def _match(self, detections: list[Detection]) -> list[int]:
        """Hit each detection's track, starting new tracks for the unmatched, and age the rest.

        A detection scoring at least start_confidence is strong: it is paired with the nearest track
        in reach, or starts a new one. A weaker one is weak: it is paired only with a track no strong
        detection hit this frame, nearest first, and then only sustains it (see _sustain); unpaired,
        it starts no track. Returns the track id of each detection, in detection order, or 0 for a
        weak detection paired with none."""
        det_tracks = [0] * len(detections)
        strong = [i for i, det in enumerate(detections) if det.conf >= self.cfg.start_confidence]
        weak = [i for i, det in enumerate(detections) if det.conf < self.cfg.start_confidence]
        hit_tracks: set[int] = set()
        for det_i, track_id in self._nearest_pairs(detections, strong, hit_tracks):
            hit_tracks.add(track_id)
            det_tracks[det_i] = track_id
            self._hit(self.tracks[track_id], detections[det_i])
        for det_i, track_id in self._nearest_pairs(detections, weak, hit_tracks):
            hit_tracks.add(track_id)
            det_tracks[det_i] = track_id
            self._sustain(self.tracks[track_id], detections[det_i])

        # Unmatched strong detections start new tracks, highest confidence first.
        unmatched = [det_i for det_i in strong if not det_tracks[det_i]]
        for det_i in sorted(unmatched, key=lambda i: -detections[i].conf):
            track = Track(self._next_id, detections[det_i].cx, detections[det_i].cy)
            self._next_id += 1
            self._hit(track, detections[det_i])
            self.tracks[track.id] = track
            hit_tracks.add(track.id)
            det_tracks[det_i] = track.id

        # Unmatched tracks accumulate misses and are dropped once unseen for drop_seconds.
        for track_id in list(self.tracks):
            if track_id in hit_tracks:
                continue
            track = self.tracks[track_id]
            track.missed += 1
            if self._reached(self.now - track.last_seen, self.cfg.drop_seconds):
                del self.tracks[track_id]
                self._card_of.pop(track_id, None)
                self._dealt_backs.discard(track_id)
        return det_tracks

    def _nearest_pairs(self, detections: list[Detection], det_indices: list[int],
                       taken: set[int]) -> list[tuple[int, int]]:
        """Greedy nearest-neighbour pairs (detection index, track id) of the detections at det_indices
        and the tracks not in taken, closest first, one to one, each within match_dist."""
        candidates = []
        for det_i in det_indices:
            det = detections[det_i]
            for track_id, track in self.tracks.items():
                if track_id in taken:
                    continue
                dist = ((det.cx - track.cx) ** 2 + (det.cy - track.cy) ** 2) ** 0.5
                if dist <= self.cfg.match_dist:
                    candidates.append((dist, det_i, track_id))
        candidates.sort()
        pairs: list[tuple[int, int]] = []
        used_dets: set[int] = set()
        used_tracks: set[int] = set()
        for _, det_i, track_id in candidates:
            if det_i in used_dets or track_id in used_tracks:
                continue
            used_dets.add(det_i)
            used_tracks.add(track_id)
            pairs.append((det_i, track_id))
        return pairs

    def _sustain(self, track: Track, det: Detection):
        """A weak sighting: the track moves and is seen this frame, so it does not miss or drop.
        It adds no Hit, so the name vote and the confirmation checks never see it."""
        self._move(track, det)
        track.missed = 0
        track.last_seen = self.now

    def _move(self, track: Track, det: Detection):
        track.cx = 0.7 * track.cx + 0.3 * det.cx  # light smoothing against jitter
        track.cy = 0.7 * track.cy + 0.3 * det.cy

    def _hit(self, track: Track, det: Detection):
        if track.seen:
            self._move(track, det)
        track.last_seen = self.now
        track.hits.append(Hit(self.now, self._interval, det.conf, det.cx, det.cy, det.label))
        if det.label == BACK and any(det.box):  # a whole-card box: how big the cards are (see card_diagonal)
            x1, y1, x2, y2 = det.box
            self._card_diagonals.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        keep = self._history_seconds(track)
        while len(track.hits) > 1 and not self._within(track.hits[0].time, keep):
            track.hits.popleft()
        track.seen += 1
        track.missed = 0
        window = self.cfg.label_window_seconds
        if window is None:
            track.votes[det.label] += 1
        else:
            track.votes = Counter(hit.label for hit in track.hits if self._within(hit.time, window))
        track.label = track.votes.most_common(1)[0][0]

    def _history_seconds(self, track: Track) -> float | None:
        """How far back the track's sightings are still needed (None: all of them). The label window
        needs them, and so does the confirm window until the track is confirmed; with no confirm
        window that is every sighting. 0: only the latest."""
        cfg = self.cfg
        if not track.confirmed and cfg.confirm_window_seconds is None:
            return None
        windows = []
        if cfg.label_window_seconds is not None:
            windows.append(cfg.label_window_seconds)
        if not track.confirmed:
            windows.append(cfg.confirm_window_seconds)
        return max(windows, default=0.0)

    def _passes_checks(self, track: Track) -> bool:
        """Has this track earned confirmation? See the module docstring for the checks."""
        cfg = self.cfg
        recent = [hit for hit in track.hits if self._within(hit.time, cfg.confirm_window_seconds)]
        seen_seconds = sum(hit.interval for hit in recent)
        if not recent or not self._reached(seen_seconds, cfg.confirm_seconds):
            return False  # not seen long enough, or not steadily enough
        rank_votes: Counter = Counter()
        for label, reads in track.votes.items():
            rank_votes[_rank_group(label)] += reads
        if rank_votes[_rank_group(track.label)] < cfg.confirm_agreement * track.votes.total():
            return False  # the reads disagree on the card's rank
        if sum(hit.conf for hit in recent) / len(recent) < cfg.confirm_confidence:
            return False  # the model is not sure enough
        if cfg.still_distance is not None:
            xs, ys = [hit.x for hit in recent], [hit.y for hit in recent]
            mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
            offsets = [((x - mean_x) ** 2 + (y - mean_y) ** 2) ** 0.5 for x, y in zip(xs, ys, strict=True)]
            if max(offsets) > cfg.still_distance:
                return False  # still moving
        return True


def _quarter_turn(x: float, y: float, height: int) -> tuple[float, float]:
    """Where (x, y) lands when a frame `height` pixels tall is turned a quarter turn clockwise."""
    return height - 1 - y, x


def _distance(a: Track, b: Track) -> float:
    """Pixels between two tracks."""
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


def _on_body(a: Track, b: Track, m: Track) -> float | None:
    """Whether track m lies on the body of a card whose corner indices are tracks a and b: between
    MID_CARD_ALONG of the way from a to b, and within MID_CARD_ACROSS of their distance to either side of
    the line between them. Returns how far to the side it lies, as that share, or None when it is not."""
    dx, dy = b.cx - a.cx, b.cy - a.cy
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return None
    px, py = m.cx - a.cx, m.cy - a.cy
    along = (px * dx + py * dy) / length2
    across = abs(px * dy - py * dx) / length2
    if MID_CARD_ALONG[0] <= along <= MID_CARD_ALONG[1] and across <= MID_CARD_ACROSS:
        return across
    return None


def _median(values) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _rank_group(label: str) -> str:
    """What the agreement check compares: the rank, with BACK as a group of its own."""
    return label if label == BACK else rank_of(label)
