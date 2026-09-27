"""Play a scripted blackjack round through the tracker. No camera or model needed.

    python scripts/simulateRound.py

Prints the "placed"/"removed" event stream and each seat's hand total, with
random missed detections, an occasional misread, and a hand passing over the
table, to show how the tracker copes. Good first thing to run.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardVision.blackjack import table_summary  # noqa: E402
from cardVision.tracker import CardTracker, Detection  # noqa: E402
from cardVision.zones import example_zones  # noqa: E402

W, H = 1920, 1080
zones = example_zones()
tracker = CardTracker(lambda x, y: zones.assign(x, y, W, H), confirm_frames=8, drop_frames=25, match_dist=40)

# Where each seat's cards land (pixel centres), and the script of the round.
SEAT = {"dealer": (960, 250), "player1": (250, 800), "player2": (710, 800), "player3": (1170, 800), "player4": (1630, 800)}
SCRIPT = [  # (frame, zone, label)
    (10, "player1", "9H"), (20, "player2", "AS"), (30, "player3", "5D"), (40, "player4", "KC"),
    (50, "dealer", "7S"),
    (60, "player1", "8C"), (70, "player2", "KD"), (80, "player3", "6H"), (90, "player4", "10S"),
    (100, "dealer", "BACK"),
    (140, "player3", "JH"),  # player3 hits and busts
    (180, "dealer", "9C"),   # dealer reveals hole card (BACK is removed below)
]
REMOVE = [(180, "dealer", "BACK")]
OCCLUDED = range(115, 135)  # a hand passes over the table

rng = random.Random(0)
on_table: list[tuple[str, str, float, float]] = []
placed_at: dict[int, list] = {}
for f, z, l in SCRIPT:
    placed_at.setdefault(f, []).append((z, l))
removed_at: dict[int, list] = {}
for f, z, l in REMOVE:
    removed_at.setdefault(f, []).append((z, l))

for frame in range(1, 260):
    for z, l in placed_at.get(frame, []):
        sx, sy = SEAT[z]
        n = sum(1 for c in on_table if c[0] == z)
        on_table.append((z, l, sx + n * 90, sy))
    for z, l in removed_at.get(frame, []):
        on_table = [c for c in on_table if not (c[0] == z and c[1] == l)]

    dets = []
    if frame not in OCCLUDED:
        for z, l, x, y in on_table:
            if rng.random() < 0.08:
                continue  # random missed detection
            label = l if rng.random() > 0.03 else "8H"  # occasional misread
            jitter = lambda: rng.uniform(-3, 3)
            dets.append(Detection(label, rng.uniform(0.6, 0.99), x + jitter(), y + jitter()))
            if l != "BACK":  # a face-up card shows two corner indices; a back is one whole-card box
                dets.append(Detection(label, rng.uniform(0.6, 0.99), x + 60 + jitter(), y + 90 + jitter()))
    for e in tracker.update(dets):
        print(e)
    if frame in (105, 130, 200):
        print(f"--- frame {frame} table:", table_summary(tracker.state(), zones.names()))
