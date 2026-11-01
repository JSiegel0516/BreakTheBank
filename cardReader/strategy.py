"""Basic strategy: the best play for a player hand against the dealer's up card.

Rules assumed: the common multi-deck table, the dealer stands on soft 17,
double on any first two cards, double after a split allowed, no surrender.

Each table row lists the play against the dealer's up card 2, 3, 4, 5, 6, 7,
8, 9, 10, A: H hit, S stand, P split, D double (hit with more than two cards),
Ds double (stand with more than two cards). Hard 8 or less always hits, hard
17 or more and soft 19 or more always stand. A pair is two cards of the same
value, so a jack and a queen are a pair of tens.
"""
from __future__ import annotations

from .blackjack import card_value, hand_value, is_blackjack
from .deck import BACK

HARD = {
    9:  "H  D  D  D  D  H  H  H  H  H",
    10: "D  D  D  D  D  D  D  D  H  H",
    11: "D  D  D  D  D  D  D  D  D  H",
    12: "H  H  S  S  S  H  H  H  H  H",
    13: "S  S  S  S  S  H  H  H  H  H",
    14: "S  S  S  S  S  H  H  H  H  H",
    15: "S  S  S  S  S  H  H  H  H  H",
    16: "S  S  S  S  S  H  H  H  H  H",
}

SOFT = {
    13: "H  H  H  D  D  H  H  H  H  H",
    14: "H  H  H  D  D  H  H  H  H  H",
    15: "H  H  D  D  D  H  H  H  H  H",
    16: "H  H  D  D  D  H  H  H  H  H",
    17: "H  D  D  D  D  H  H  H  H  H",
    18: "S  Ds Ds Ds Ds S  S  H  H  H",
}

PAIRS = {  # keyed by the value of one card, aces as 11
    2:  "P  P  P  P  P  P  H  H  H  H",
    3:  "P  P  P  P  P  P  H  H  H  H",
    4:  "H  H  H  P  P  H  H  H  H  H",
    5:  "D  D  D  D  D  D  D  D  H  H",
    6:  "P  P  P  P  P  H  H  H  H  H",
    7:  "P  P  P  P  P  P  H  H  H  H",
    8:  "P  P  P  P  P  P  P  P  P  P",
    9:  "P  P  P  P  P  S  P  P  S  S",
    10: "S  S  S  S  S  S  S  S  S  S",
    11: "P  P  P  P  P  P  P  P  P  P",
}

ALWAYS_HIT = "H  H  H  H  H  H  H  H  H  H"
ALWAYS_STAND = "S  S  S  S  S  S  S  S  S  S"

PLAYS = {"H": "hit", "S": "stand", "P": "split"}


def _table_row(player_labels: list[str]) -> str:
    total, soft = hand_value(player_labels)
    is_pair = len(player_labels) == 2 and card_value(player_labels[0]) == card_value(player_labels[1])
    if is_pair:
        return PAIRS[card_value(player_labels[0])]
    if soft:
        return ALWAYS_STAND if total >= 19 else SOFT[total]
    if total <= 8:
        return ALWAYS_HIT
    if total >= 17:
        return ALWAYS_STAND
    return HARD[total]


def hint(player_labels: list[str], dealer_labels: list[str]) -> str | None:
    """Return "hit", "stand", "double" or "split" for a player hand, or None for no hint.

    The dealer's up card is the first face-up label in dealer_labels. There is
    no hint when the player hand has a face-down card (BACK), fewer than two
    cards, is bust or is a blackjack, or when the dealer has no face-up card.
    """
    up_card = next((label for label in dealer_labels if label != BACK), None)
    if up_card is None or BACK in player_labels or len(player_labels) < 2:
        return None
    if hand_value(player_labels)[0] > 21 or is_blackjack(player_labels):
        return None

    column = card_value(up_card) - 2
    play = _table_row(player_labels).split()[column]
    first_two_cards = len(player_labels) == 2
    if play == "D":
        return "double" if first_two_cards else "hit"
    if play == "Ds":
        return "double" if first_two_cards else "stand"
    return PLAYS[play]
