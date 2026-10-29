"""Blackjack rules on top of the card labels: hand totals and the Hi-Lo count.

Hand totals: aces count 11 unless that busts, then 1. Face-down cards (BACK)
are left out of the total and reported as hidden.

Hi-Lo: every card seen adds to the running count, 2-6 as +1, 7-9 as 0, and
10-A as -1. The true count divides it by the decks still to be dealt. Cards
left is the deck minus the face cards seen and minus the face-down cards that
were dealt and never turned over: those are gone from the deck, but nobody saw
what they were, so they do not change the running count. Decks left is cards
left / 52, but never less than MIN_DECKS_LEFT (a quarter deck, 13 cards), so
the true count does not blow up as the last few cards come out. A positive
true count means the cards left are rich in tens and aces, which favours the
player.
"""
from __future__ import annotations

from dataclasses import dataclass

from .deck import BACK, rank_of

CARDS_PER_DECK = 52
MIN_DECKS_LEFT = 0.25  # floor for the true count's divisor, so a few cards left cannot make it huge


def card_value(label: str) -> int:
    rank = rank_of(label)
    if rank == "A":
        return 11
    if rank in ("J", "Q", "K"):
        return 10
    return int(rank)


def hand_value(labels: list[str]) -> tuple[int, bool]:
    """Return (best total, is_soft). Face-down cards (BACK) are ignored."""
    faces = [label for label in labels if label != BACK]
    total = sum(card_value(label) for label in faces)
    aces = sum(1 for label in faces if rank_of(label) == "A")
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total, aces > 0


def is_blackjack(labels: list[str]) -> bool:
    faces = [label for label in labels if label != BACK]
    return len(faces) == 2 and hand_value(faces)[0] == 21


def describe(labels: list[str]) -> str:
    """Short text for a hand, e.g. '17', 'soft 18', 'BLACKJACK', 'BUST (24)', '10 +1 hidden'."""
    if not labels:
        return "-"
    total, soft = hand_value(labels)
    hidden = labels.count(BACK)
    if is_blackjack(labels):
        text = "BLACKJACK"
    elif total > 21:
        text = f"BUST ({total})"
    else:
        text = f"soft {total}" if soft else str(total)
    if hidden:
        text = f"{text} +{hidden} hidden" if total else f"{hidden} hidden"
    return text


def hi_lo(label: str) -> int:
    """Hi-Lo value of one card: +1 for 2-6, 0 for 7-9, -1 for 10, J, Q, K, A. BACK is 0."""
    if label == BACK:
        return 0
    rank = rank_of(label)
    if rank in ("2", "3", "4", "5", "6"):
        return 1
    if rank in ("7", "8", "9"):
        return 0
    return -1


@dataclass
class ShoeCount:
    running: int      # sum of Hi-Lo values of every card seen this shoe
    true: float       # running count per deck still to be dealt (at least MIN_DECKS_LEFT)
    cards_left: int   # cards not yet dealt: the deck minus faces seen and face-down cards never revealed


def shoe_count(seen: list[str], decks: int = 1, unrevealed: int = 0) -> ShoeCount:
    """Count for the face cards seen since the last shuffle.

    unrevealed: face-down cards dealt this shoe and never turned over. They
    leave the deck, so they lower cards_left, but add nothing to the running count."""
    running = sum(hi_lo(label) for label in seen)
    cards_left = max(decks * CARDS_PER_DECK - len(seen) - unrevealed, 0)
    decks_left = max(cards_left / CARDS_PER_DECK, MIN_DECKS_LEFT)
    return ShoeCount(running, running / decks_left, cards_left)
