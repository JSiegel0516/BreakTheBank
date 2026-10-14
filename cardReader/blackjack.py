"""Blackjack rules on top of the card labels: hand totals and the Hi-Lo count.

Hand totals: aces count 11 unless that busts, then 1. Face-down cards (BACK)
are left out of the total and reported as hidden.

Hi-Lo: every card seen adds to the running count, 2-6 as +1, 7-9 as 0, and
10-A as -1. The true count divides it by the decks still to be dealt, which
for one deck is (52 - cards seen) / 52. A positive true count means the cards
left are rich in tens and aces, which favours the player.
"""
from __future__ import annotations

from dataclasses import dataclass

from .deck import BACK, rank_of

CARDS_PER_DECK = 52


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
    """Short text for a hand, e.g. '17', 'soft 18', 'BLACKJACK', 'BUST (24)', '10 +1 hidden'."""
    if not labels:
        return "-"
    total, soft = hand_value(labels)
    hidden = labels.count(BACK)
    if is_blackjack(labels):
        s = "BLACKJACK"
    elif total > 21:
        s = f"BUST ({total})"
    else:
        s = f"soft {total}" if soft else str(total)
    if hidden:
        s = f"{s} +{hidden} hidden" if total else f"{hidden} hidden"
    return s


def hi_lo(label: str) -> int:
    """Hi-Lo value of one card: +1 for 2-6, 0 for 7-9, -1 for 10, J, Q, K, A. BACK is 0."""
    if label == BACK:
        return 0
    r = rank_of(label)
    if r in ("2", "3", "4", "5", "6"):
        return 1
    if r in ("7", "8", "9"):
        return 0
    return -1


@dataclass
class ShoeCount:
    running: int      # sum of Hi-Lo values of every card seen this shoe
    true: float       # running count per deck still to be dealt
    cards_left: int   # cards not yet seen


def shoe_count(seen: list[str], decks: int = 1) -> ShoeCount:
    """Count for the face cards seen since the last shuffle."""
    running = sum(hi_lo(l) for l in seen)
    cards_left = max(decks * CARDS_PER_DECK - len(seen), 0)
    decks_left = cards_left / CARDS_PER_DECK
    true = running / decks_left if decks_left else 0.0
    return ShoeCount(running, true, cards_left)
