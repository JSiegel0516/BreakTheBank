"""The data that flows through the reader.

    Detection   one box the detector drew in one frame
    Card        one physical card of the current round, with its placement number
    Event       something that changed: placed, hidden, returned, removed (hidden
                too long, taken off the table), renamed, duplicate (a name
                already counted in an earlier round), held (a second copy
                placed beside a card of its name, not counted yet), counted (a
                held copy counted at last), round_over

detector.py makes Detections, tracker.py turns them into Cards and Events,
table.py groups Cards into hands, hud.py draws all three.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    label: str
    conf: float
    cx: float
    cy: float
    box: tuple[float, float, float, float] = (0, 0, 0, 0)  # x1, y1, x2, y2


@dataclass
class Card:
    order: int                      # 1 for the first card placed this round, 2 for the next ...
    label: str
    x: float                        # latest known position in pixels
    y: float
    placed_frame: int
    hidden: bool = False            # not visible right now, but still on the table
    hidden_since: float = 0.0       # the tracker's clock when it went hidden, while hidden
    counted: bool = False           # this card added its label to the shoe count
    held: bool = False              # a copy beside a card of its name: not counted until both are seen together
    held_seconds: float = 0.0       # while held: seconds this card and that card were both seen


@dataclass
class Event:
    # "placed" | "hidden" | "returned" | "removed" | "renamed" | "duplicate" | "held" | "counted" | "round_over"
    kind: str
    card: Card | None
    frame: int
    was: str | None = None  # renamed: the label before

    def __str__(self):
        if self.card is None:
            return f"[{self.frame:6d}] {self.kind}"
        s = f"[{self.frame:6d}] {self.kind:8s} #{self.card.order:<3d} {self.card.label}"
        return f"{s} (was {self.was})" if self.was else s
