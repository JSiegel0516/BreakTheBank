"""Stage 3: which seat is a detection in?

A zone is a named polygon on the table (dealer, player1 ... player4). Polygons
are stored as fractions of the frame size (0..1) so one zones.json works at any
camera resolution. Make them with scripts/1_calibrateZones.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ZONE_NAMES = ["dealer", "player1", "player2", "player3", "player4"]


@dataclass
class Zone:
    name: str
    polygon: list[tuple[float, float]] = field(default_factory=list)  # normalised

    def pixels(self, w: int, h: int) -> np.ndarray:
        return np.array([(x * w, y * h) for x, y in self.polygon], dtype=np.float32)

    def contains(self, x: float, y: float, w: int, h: int) -> bool:
        if len(self.polygon) < 3:
            return False
        return cv2.pointPolygonTest(self.pixels(w, h), (float(x), float(y)), False) >= 0


class ZoneMap:
    def __init__(self, zones: list[Zone] | None = None):
        self.zones = zones or []

    @classmethod
    def load(cls, path: str | Path) -> "ZoneMap":
        raw = json.loads(Path(path).read_text())
        return cls([Zone(z["name"], [tuple(p) for p in z["polygon"]]) for z in raw["zones"]])

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(
            {"zones": [{"name": z.name, "polygon": z.polygon} for z in self.zones]}, indent=2))

    def assign(self, x: float, y: float, w: int, h: int) -> str | None:
        for z in self.zones:
            if z.contains(x, y, w, h):
                return z.name
        return None

    def names(self) -> list[str]:
        return [z.name for z in self.zones]

    def draw(self, frame: np.ndarray, color=(0, 200, 255)):
        h, w = frame.shape[:2]
        for z in self.zones:
            if len(z.polygon) < 2:
                continue
            pts = z.pixels(w, h).astype(np.int32)
            cv2.polylines(frame, [pts], isClosed=len(z.polygon) >= 3, color=color, thickness=2)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(frame, z.name, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame


def example_zones() -> ZoneMap:
    """Dealer strip along the top, four player boxes along the bottom.
    A starting point for calibrate_zones.py, not a substitute for it."""
    zones = [Zone("dealer", [(0.25, 0.05), (0.75, 0.05), (0.75, 0.40), (0.25, 0.40)])]
    for i in range(4):
        x0 = 0.02 + i * 0.24
        zones.append(Zone(f"player{i + 1}", [(x0, 0.55), (x0 + 0.22, 0.55), (x0 + 0.22, 0.95), (x0, 0.95)]))
    return ZoneMap(zones)
