"""The data that flows through the reader.

    Detection   one box the detector drew in one frame
    Card        one physical card of the current round, with its placement number
    Event       something that changed: placed, hidden, returned, renamed,
                duplicate (a name already counted in an earlier round), round_over

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
    counted: bool = False           # this card added its label to the shoe count


@dataclass
class Event:
    kind: str  # "placed" | "hidden" | "returned" | "renamed" | "duplicate" | "round_over"
    card: Card | None
    frame: int
    was: str | None = None  # renamed: the label before

    def __str__(self):
        if self.card is None:
            return f"[{self.frame:6d}] {self.kind}"
        s = f"[{self.frame:6d}] {self.kind:8s} #{self.card.order:<3d} {self.card.label}"
        return f"{s} (was {self.was})" if self.was else s
