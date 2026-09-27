"""Stage 5: score a blackjack hand from a list of card labels.

Aces count 11 unless that busts, then 10. Face-down cards (BACK) are ignored
for the total but reported as hidden.
"""
from __future__ import annotations

from .deck import BACK, rank_of


def card_value(label: str) -> int:
    r = rank_of(label)
    if r == "A":
        return 11
    if r in ("J", "Q", "K"):
        return 10
    return int(r)


def hand_value(labels: list[str]) -> tuple[int, bool]:
    """Return (best total, is_soft). Face-down cards (BACK) are ignored."""
    faces = [l for l in labels if l != BACK]
    total = sum(card_value(l) for l in faces)
    aces = sum(1 for l in faces if rank_of(l) == "A")
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total, aces > 0


def is_blackjack(labels: list[str]) -> bool:
    faces = [l for l in labels if l != BACK]
    return len(faces) == 2 and hand_value(faces)[0] == 21


def describe(labels: list[str]) -> str:
    if not labels:
        return "-"
    total, soft = hand_value(labels)
    hidden = labels.count(BACK)
    s = f"{total}"
    if soft:
        s = f"soft {s}"
    if is_blackjack(labels):
        s = "BLACKJACK"
    elif total > 21:
        s = f"BUST ({total})"
    if hidden:
        s += f" +{hidden} hidden"
    return s


def table_summary(state: dict[str | None, list[str]], zone_names: list[str]) -> dict[str, str]:
    return {z: describe(state.get(z, [])) for z in zone_names}
