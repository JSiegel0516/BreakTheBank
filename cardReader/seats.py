"""Where each seat is on the table, from the number of players alone.

The camera looks straight down with the dealer at the top of the image. The
dealer gets a strip across the top; the players sit on an arc below it, like a
real blackjack table, P1 on the left. Each player seat is a rectangle turned to
face the dealer. The layout is rebuilt whenever the player count changes (keys
1-7 in read.py), so nothing has to be drawn by hand.

    +------------------------------------+
    |        [      DEALER      ]        |
    | [P1]                          [P5] |
    |     [P2]                  [P4]     |
    |           [P3]                     |
    +------------------------------------+
"""
from __future__ import annotations

import math
from dataclasses import dataclass

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
    def __init__(self, players: int, width: int, height: int):
        self.players = max(1, min(players, MAX_PLAYERS))
        self.width, self.height = width, height
        self.seats = [self._dealer()] + [self._player(i) for i in range(self.players)]

    def names(self) -> list[str]:
        return [s.name for s in self.seats]

    def assign(self, x: float, y: float) -> str | None:
        """Name of the seat containing pixel (x, y), or None if it is in none."""
        for s in self.seats:
            if cv2.pointPolygonTest(s.polygon.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0:
                return s.name
        return None

    def _dealer(self) -> Seat:
        x1, y1, x2, y2 = DEALER_BOX
        W, H = self.width, self.height
        pts = [(x1 * W, y1 * H), (x2 * W, y1 * H), (x2 * W, y2 * H), (x1 * W, y2 * H)]
        return Seat(DEALER, np.array(pts, np.int32))

    def _player(self, i: int) -> Seat:
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
        pts = np.clip(np.array(pts), [0, 0], [W - 1, H - 1])
        return Seat(f"P{i + 1}", pts.astype(np.int32))
