"""Stage 4: turn noisy per-frame detections into a stable table state.

The detector gives a fresh list of boxes every frame, and that list flickers:
a card is missed for a frame, a label is misread, a hand covers the table.
The tracker smooths all of that out with a few rules:

  * A detection must be seen in roughly the same spot for CONFIRM_FRAMES
    consecutive frames before it counts. Kills flicker and cards in motion.
  * A confirmed card is only dropped after DROP_FRAMES consecutive misses,
    so a hand passing over the table does not erase it.
  * The label of a track is a running vote, so one misread frame does not flip it.
  * Deck uniqueness: a face label can only be confirmed in one zone at a time.
    If it shows up twice, the higher-confidence zone wins and the other is suppressed.
  * A card lying flat exposes two corner indices, so two tracks with the same
    label in the same zone collapse to one card in the state.

Output is state() -> {zone: [labels]} plus a list of "placed"/"removed" Events.
"""
from __future__ import annotations

from collections import Counter, defaultdict
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
    zone: str | None
    votes: Counter = field(default_factory=Counter)
    conf_sum: float = 0.0
    seen: int = 0
    missed: int = 0
    confirmed: bool = False
    suppressed: bool = False

    @property
    def label(self) -> str:
        return self.votes.most_common(1)[0][0]

    @property
    def mean_conf(self) -> float:
        return self.conf_sum / max(self.seen, 1)


@dataclass
class Event:
    kind: str  # "placed" | "removed"
    label: str
    zone: str | None
    frame: int

    def __str__(self):
        return f"[{self.frame:6d}] {self.kind:7s} {self.label:>4s} in {self.zone}"


class CardTracker:
    def __init__(self, zone_fn, confirm_frames=10, drop_frames=30, match_dist=40.0):
        """
        zone_fn: (cx, cy) -> zone name or None
        match_dist: pixels; a detection closer than this to a track is that track
        """
        self.zone_fn = zone_fn
        self.confirm_frames = confirm_frames
        self.drop_frames = drop_frames
        self.match_dist = match_dist
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.frame = 0
        self.events: list[Event] = []

    def update(self, detections: list[Detection]) -> list[Event]:
        self.frame += 1
        before = self.state()

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

        # Unmatched detections start new tracks.
        for di, d in enumerate(detections):
            if di not in used_d:
                t = Track(self._next_id, d.cx, d.cy, None)
                self._next_id += 1
                self._hit(t, d)
                self.tracks[t.id] = t

        # Unmatched tracks accumulate misses; drop after drop_frames.
        for tid in list(self.tracks):
            if tid in hit_t:
                continue
            t = self.tracks[tid]
            t.missed += 1
            if t.missed >= self.drop_frames:
                del self.tracks[tid]

        self._apply_deck_uniqueness()

        after = self.state()
        new_events: list[Event] = []
        for zone in sorted(set(before) | set(after), key=str):
            for label in after.get(zone, []):
                if label not in before.get(zone, []):
                    new_events.append(Event("placed", label, zone, self.frame))
            for label in before.get(zone, []):
                if label not in after.get(zone, []):
                    new_events.append(Event("removed", label, zone, self.frame))
        self.events.extend(new_events)
        return new_events

    def _hit(self, t: Track, d: Detection):
        if t.seen:
            t.cx = 0.7 * t.cx + 0.3 * d.cx  # light smoothing against jitter
            t.cy = 0.7 * t.cy + 0.3 * d.cy
        else:
            t.cx, t.cy = d.cx, d.cy
        t.votes[d.label] += 1
        t.conf_sum += d.conf
        t.seen += 1
        t.missed = 0
        t.zone = self.zone_fn(t.cx, t.cy)
        if t.seen >= self.confirm_frames:
            t.confirmed = True

    def _apply_deck_uniqueness(self):
        by_label: dict[str, list[Track]] = defaultdict(list)
        for t in self.tracks.values():
            t.suppressed = False
            if t.confirmed and t.label != BACK:
                by_label[t.label].append(t)
        for ts in by_label.values():
            zones = {t.zone for t in ts}
            if len(zones) <= 1:
                continue  # same zone: the two corners of one card
            # A corner that fell outside every zone never outranks one inside a seat.
            named = {z for z in zones if z is not None}
            candidates = named or zones
            best_zone = max(candidates, key=lambda z: max(t.mean_conf for t in ts if t.zone == z))
            for t in ts:
                if t.zone != best_zone:
                    t.suppressed = True

    def state(self) -> dict[str | None, list[str]]:
        """zone -> sorted list of confirmed card labels. Faces are unique per zone; BACK may repeat."""
        faces: dict[str | None, set[str]] = defaultdict(set)
        backs: Counter = Counter()
        for t in self.tracks.values():
            if not t.confirmed or t.suppressed:
                continue
            if t.label == BACK:
                backs[t.zone] += 1
            else:
                faces[t.zone].add(t.label)
        out: dict[str | None, list[str]] = {}
        for z in set(faces) | set(backs):
            out[z] = sorted(faces.get(z, ())) + [BACK] * backs.get(z, 0)
        return out

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.confirmed and not t.suppressed]
