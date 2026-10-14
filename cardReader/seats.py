"""Where each seat is on the table, from the number of players alone.

The camera looks straight down with the dealer at the top of the image. The
dealer gets a strip across the top; the players sit on an arc below it, like a
real blackjack table, P1 on the left. Each player seat is a rectangle turned to
face the dealer. The layout is rebuilt whenever the player count changes (keys
1-7 in read.py), so nothing has to be drawn by hand.

Any zone can then be adjusted with the mouse in edit mode (ZoneEditor, key E in
read.py): drag a corner, drag the whole zone, or right-click it to reset it to
the template. Edited layouts are saved to data/seats.json as fractions of the
frame, so they work at any camera resolution.

    +------------------------------------+
    |        [      DEALER      ]        |
    | [P1]                          [P5] |
    |     [P2]                  [P4]     |
    |           [P3]                     |
    +------------------------------------+
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MAX_PLAYERS = 7
DEALER = "dealer"

# Layout as fractions of the frame, tuned for a 16:9 overhead view.
DEALER_BOX = (0.30, 0.03, 0.70, 0.28)  # x1, y1, x2, y2
ARC_CENTRE = (0.50, 0.10)               # the arc curves around the dealer
ARC_RADIUS = (0.42, 0.68)               # horizontal, vertical
ARC_ANGLES = (160, 20)                  # degrees, P1 end to last-player end; 90 is straight below
SEAT_DEPTH = 0.30                       # seat size towards the dealer, fraction of frame height
SEAT_WIDTH_MAX = 0.20                   # fraction of frame width, shrinks as players are added


@dataclass
class Seat:
    name: str
    polygon: np.ndarray  # pixel corners, int32, shape (4, 2)


class SeatLayout:
    def __init__(self, players: int, width: int, height: int, polygons: list | None = None):
        """polygons: optional pixel corners for each seat (dealer first), e.g. from an edited layout."""
        self.players = max(1, min(players, MAX_PLAYERS))
        self.width, self.height = width, height
        self.seats = [Seat(DEALER if i == 0 else f"P{i}", self.template(i)) for i in range(self.players + 1)]
        if polygons is not None:
            for seat, poly in zip(self.seats, polygons):
                seat.polygon = np.array(poly, np.int32)

    def names(self) -> list[str]:
        return [s.name for s in self.seats]

    def assign(self, x: float, y: float) -> str | None:
        """Name of the seat containing pixel (x, y), or None if it is in none."""
        for s in self.seats:
            if cv2.pointPolygonTest(s.polygon.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0:
                return s.name
        return None

    def template(self, i: int) -> np.ndarray:
        """Template corners of seat i (0 is the dealer, 1 is P1 ...), in pixels."""
        return self._dealer() if i == 0 else self._player(i - 1)

    def resized(self, width: int, height: int) -> "SeatLayout":
        """The same layout for a different frame size."""
        sx, sy = width / self.width, height / self.height
        return SeatLayout(self.players, width, height, [s.polygon * [sx, sy] for s in self.seats])

    def save(self, path: str | Path):
        W, H = self.width, self.height
        data = {"players": self.players,
                "seats": [{"name": s.name, "polygon": [[round(x / W, 5), round(y / H, 5)] for x, y in s.polygon]}
                          for s in self.seats]}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path, width: int, height: int) -> "SeatLayout | None":
        """The layout saved at path, scaled to this frame size, or None if there is none."""
        path = Path(path)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        polys = [[(x * width, y * height) for x, y in s["polygon"]] for s in data["seats"]]
        return cls(data["players"], width, height, polys)

    def _dealer(self) -> np.ndarray:
        x1, y1, x2, y2 = DEALER_BOX
        W, H = self.width, self.height
        return np.array([(x1 * W, y1 * H), (x2 * W, y1 * H), (x2 * W, y2 * H), (x1 * W, y2 * H)], np.int32)

    def _player(self, i: int) -> np.ndarray:
        W, H = self.width, self.height
        a0, a1 = ARC_ANGLES
        angle = 90.0 if self.players == 1 else a0 + (a1 - a0) * i / (self.players - 1)
        cx, cy = ARC_CENTRE[0] * W, ARC_CENTRE[1] * H
        px = cx + ARC_RADIUS[0] * W * math.cos(math.radians(angle))
        py = cy + ARC_RADIUS[1] * H * math.sin(math.radians(angle))
        # v points from the seat towards the dealer, u runs along the arc
        vx, vy = cx - px, cy - py
        n = math.hypot(vx, vy)
        vx, vy = vx / n, vy / n
        ux, uy = -vy, vx
        half_w = min(SEAT_WIDTH_MAX, 0.84 / self.players) * W / 2
        half_d = SEAT_DEPTH * H / 2
        pts = [(px + sx * ux * half_w + sy * vx * half_d, py + sx * uy * half_w + sy * vy * half_d)
               for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1))]
        return np.clip(np.array(pts), [0, 0], [W - 1, H - 1]).astype(np.int32)


class ZoneEditor:
    """Adjust seat zones with the mouse while the reader runs.

    Only active while `active` is True (key E in read.py):
      * left-drag a corner handle: move that corner anywhere, the others stay put
      * left-drag inside a zone: move the whole zone
      * right-click a zone: reset it to the template shape
    Pass on_mouse to cv2.setMouseCallback. Clicks beyond the video (on the sidebar) are ignored."""

    def __init__(self, layout: SeatLayout):
        self.layout = layout
        self.active = False
        self.selected: int | None = None   # index of the zone last clicked, for highlighting
        self._drag: tuple | None = None    # (seat index, corner index or None, last x, last y)

    def handle_radius(self) -> int:
        return max(12, round(18 * self.layout.width / 1600))

    def on_mouse(self, event, x, y, flags=0, param=None):
        if not self.active:
            self._drag = None
            return
        W, H = self.layout.width, self.layout.height
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) and not (0 <= x < W and 0 <= y < H):
            return  # a click on the sidebar or bottom bar
        x, y = min(max(x, 0), W - 1), min(max(y, 0), H - 1)  # a drag past the edge stops at the edge
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self._hit(x, y)
            self.selected = hit[0] if hit else None
            self._drag = (*hit, x, y) if hit else None
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            seat, corner, lx, ly = self._drag
            poly = self.layout.seats[seat].polygon
            if corner is not None:
                poly[corner] = (x, y)
            else:
                dx = int(np.clip(x - lx, -poly[:, 0].min(), W - 1 - poly[:, 0].max()))
                dy = int(np.clip(y - ly, -poly[:, 1].min(), H - 1 - poly[:, 1].max()))
                poly += (dx, dy)
                x, y = lx + dx, ly + dy
            self._drag = (seat, corner, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            hit = self._hit(x, y)
            if hit:
                self.layout.seats[hit[0]].polygon = self.layout.template(hit[0])
                self.selected = hit[0]

    def _hit(self, x, y) -> tuple[int, int | None] | None:
        """(seat index, corner index) for a corner handle under (x, y), else (seat index, None)
        for a zone containing it, else None. Corners win over zone interiors, and player seats
        over the dealer, so a handle can always be grabbed."""
        r = self.handle_radius()
        order = list(range(len(self.layout.seats)))[::-1]
        for i in order:
            d = np.hypot(*(self.layout.seats[i].polygon - (x, y)).T)
            if d.min() <= r:
                return i, int(d.argmin())
        for i in order:
            poly = self.layout.seats[i].polygon.reshape(-1, 1, 2)
            if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                return i, None
        return None
