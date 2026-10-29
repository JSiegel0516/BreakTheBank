"""The table: which seat each card is in, and the hand at every seat.

The tracker knows cards and positions; the seat layout knows zones. This is
the one place that puts them together, so the overlay, the terminal, and the
CSV log all agree on which hand a card belongs to and what it totals.
"""
from __future__ import annotations

from dataclasses import dataclass

from .blackjack import describe
from .cards import Card
from .seats import SeatLayout

OTHER = "OTHER"  # cards that lie in no seat


@dataclass
class Hand:
    seat: str          # "dealer", "P1" ... or OTHER
    cards: list[Card]  # in placement order, hidden ones included

    @property
    def labels(self) -> list[str]:
        return [c.label for c in self.cards]

    @property
    def total(self) -> str:
        """'17', 'soft 18', 'BLACKJACK', 'BUST (24)', '10 +1 hidden', or '' for loose cards."""
        return describe(self.labels) if self.seat != OTHER else ""


class Table:
    def __init__(self, layout: SeatLayout):
        self.layout = layout

    def seat_of(self, card: Card) -> str | None:
        """The seat the card's corner index lies in, or None."""
        return self.layout.assign(card.x, card.y)

    def hands(self, cards: list[Card]) -> list[Hand]:
        """One hand per seat in layout order (dealer first), then OTHER if any card is loose."""
        seat_of = {c.order: self.seat_of(c) for c in cards}
        hands = [Hand(name, [c for c in cards if seat_of[c.order] == name]) for name in self.layout.names()]
        loose = [c for c in cards if seat_of[c.order] is None]
        if loose:
            hands.append(Hand(OTHER, loose))
        return hands
