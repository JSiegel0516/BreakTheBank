"""The table: which seat each card is in, and the hand at every seat.

The tracker knows cards and positions; the seat layout knows zones. This is
the one place that puts them together, so the overlay, the terminal, and the
CSV log all agree on which hand a card belongs to and what it totals.

A split seat has two hands, parts 1 and 2, each grown from one card of the
pair it was split from (its seed). Part 1's seed is the card further left in
the image. Every later card in the seat joins the hand whose seed lies nearest.
Both hands keep the seat id, so the CSV log and evaluate.py see one seat.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .blackjack import describe
from .cards import Card
from .seats import SeatLayout

OTHER = "OTHER"  # cards that lie in no seat


@dataclass
class Hand:
    seat: str               # "dealer", "P1" ... or OTHER
    cards: list[Card]       # in placement order, hidden ones included
    part: int | None = None  # 1 or 2 for the two hands of a split seat, else None

    @property
    def labels(self) -> list[str]:
        return [c.label for c in self.cards]

    @property
    def total(self) -> str:
        """'17', 'soft 18', 'BLACKJACK', 'BUST (24)', '10 +1 hidden', or '' for loose cards."""
        return describe(self.labels) if self.seat != OTHER else ""


def _distance(a: Card, b: Card) -> float:
    return math.dist((a.x, a.y), (b.x, b.y))


class Table:
    def __init__(self, layout: SeatLayout):
        self.layout = layout
        self.splits: dict[str, tuple[int, int]] = {}  # seat id -> placement numbers of the seeds of parts 1 and 2

    def seat_of(self, card: Card) -> str | None:
        """The seat the card's corner index lies in, or None."""
        return self.layout.assign(card.x, card.y)

    def split(self, seat: str, cards: list[Card]):
        """Split seat into two hands seeded by its two cards; part 1's seed is the one further left."""
        left, right = sorted(cards, key=lambda c: c.x)
        self.splits[seat] = (left.order, right.order)

    def join(self, seat: str):
        """Make a split seat one hand again."""
        self.splits.pop(seat, None)

    def clear_splits(self):
        self.splits.clear()

    def hands(self, cards: list[Card]) -> list[Hand]:
        """One hand per seat in layout order (dealer first), then OTHER if any card is loose.

        A split seat gives two hands in a row, parts 1 and 2. A split whose seed card has
        left the seat is dropped (see seeds), and the seat gives one hand again."""
        seat_of = {c.order: self.seat_of(c) for c in cards}
        hands = []
        for name in self.layout.names():
            seated = [c for c in cards if seat_of[c.order] == name]
            seeds = self.seeds(name, seated)
            hands.extend(self._split_hands(name, seated, seeds) if seeds else [Hand(name, seated)])
        loose = [c for c in cards if seat_of[c.order] is None]
        if loose:
            hands.append(Hand(OTHER, loose))
        return hands

    def seeds(self, seat: str, cards: list[Card]) -> tuple[Card, Card] | None:
        """The seed cards of parts 1 and 2 of a split seat, found among cards (the seat's cards).

        None when the seat is not split. A split whose seed is not among cards is dropped
        from splits and gives None too."""
        if seat not in self.splits:
            return None
        by_order = {c.order: c for c in cards}
        first, second = (by_order.get(order) for order in self.splits[seat])
        if first is None or second is None:
            del self.splits[seat]
            return None
        return first, second

    @staticmethod
    def _split_hands(seat: str, cards: list[Card], seeds: tuple[Card, Card]) -> list[Hand]:
        """The two hands of a split seat: each seed in its own part, every other card with the nearer seed."""
        first_seed, second_seed = seeds
        first, second = Hand(seat, [], part=1), Hand(seat, [], part=2)
        for card in cards:
            nearer_second = _distance(card, second_seed) < _distance(card, first_seed)
            if card is second_seed or (card is not first_seed and nearer_second):
                second.cards.append(card)
            else:
                first.cards.append(card)
        return [first, second]
