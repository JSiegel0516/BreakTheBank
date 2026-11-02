"""Blackjack rules on top of the card labels: hand totals and the card count.

Hand totals: aces count 11 unless that busts, then 1. Face-down cards (BACK)
are left out of the total and reported as hidden.

Outcomes: a complete hand (two or more cards, none face down) shows a bust,
blackjack, or 21 at once. Win, push, or lose against the dealer is known only
once the dealer is finished: no card face down and a total of DEALER_STANDS_AT
or more (the dealer stands on soft 17), or bust. A blackjack beats a finished
dealer unless the dealer also has one, which is a push.

Counting: every card seen adds its value in the chosen system (COUNT_SYSTEMS)
to the running count. Hi-Lo counts 2-6 as +1, 7-9 as 0, and 10-A as -1;
Hi-Opt II and Uston APC give some ranks two or three points. Each system sums
to zero over a deck. The true count divides the running count by the decks
still to be dealt (Hi-Lo, Hi-Opt II) or the half-decks (Uston APC). Cards
left is the shoe (1 to MAX_DECKS decks of 52) minus the face cards seen and
minus the face-down cards that were dealt and never turned over: those are
gone, but nobody saw what they were, so they do not change the running count. The divisor never
counts fewer cards than MIN_DECKS_LEFT of a deck (a quarter deck, 13 cards,
which is half a half-deck), so the true count does not blow up as the last
few cards come out. A positive
true count means the cards left are rich in high cards (tens and aces), which
favours the player; a negative one means low cards are left, which favours the
dealer.

Odds: player_odds, dealer_odds, dealer_final_odds, and stand_odds give a hand's
chances as whole percentages, exact against the unseen cards: the shoe minus the
faces seen, one entry per copy left. A face-down card never revealed is one of
those unseen cards, so it counts as a random draw from them. dealer_final_odds
plays the dealer out, drawing without replacement until DEALER_STANDS_AT or bust;
stand_odds compares a standing player hand with those finals. insurance says
whether the 2-to-1 insurance bet pays on average against the unseen cards.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cache, lru_cache

from .deck import BACK, rank_of

CARDS_PER_DECK = 52
MAX_DECKS = 3  # most decks a shoe may hold (settings.DECKS)
TEN_RANKS = ("10", "J", "Q", "K")  # the ranks worth ten
MIN_DECKS_LEFT = 0.25  # floor for the true count's divisor, so a few cards left cannot make it huge
DEALER_STANDS_AT = 17  # soft 17 included


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


def is_pair(labels: list[str]) -> bool:
    """Whether labels are exactly two face-up cards of equal value, so a jack and a queen are a pair of tens."""
    return len(labels) == 2 and BACK not in labels and card_value(labels[0]) == card_value(labels[1])


def dealer_finished(labels: list[str]) -> bool:
    """Whether the dealer is done drawing: no card face down and DEALER_STANDS_AT or more (soft or hard), or bust."""
    return BACK not in labels and hand_value(labels)[0] >= DEALER_STANDS_AT


def milestone(labels: list[str]) -> str | None:
    """'bust', 'blackjack', or 'twenty-one' for a complete hand that reached one, else None.

    A hand with fewer than two cards or any face-down card is not complete."""
    if len(labels) < 2 or BACK in labels:
        return None
    if is_blackjack(labels):
        return "blackjack"
    total = hand_value(labels)[0]
    if total > 21:
        return "bust"
    return "twenty-one" if total == 21 else None


def outcome(player_labels: list[str], dealer_labels: list[str]) -> str | None:
    """A player hand's state against the dealer: 'bust', 'blackjack', 'twenty-one', 'win', 'push', 'lose', or None.

    A milestone (bust, blackjack, 21) shows while the dealer is unfinished. Once the
    dealer is finished, a blackjack pushes against a dealer blackjack and stays
    'blackjack' otherwise; any other standing hand, 21 included, wins, pushes, or
    loses on the totals, loses to a dealer blackjack, and wins on a dealer bust.
    None for an incomplete hand (see milestone), or for a hand without a milestone
    while the dealer is unfinished."""
    if len(player_labels) < 2 or BACK in player_labels:
        return None
    mark = milestone(player_labels)
    if mark == "bust" or not dealer_finished(dealer_labels):
        return mark
    if mark == "blackjack":
        return "push" if is_blackjack(dealer_labels) else "blackjack"
    if is_blackjack(dealer_labels):
        return "lose"
    player_total = hand_value(player_labels)[0]
    dealer_total = hand_value(dealer_labels)[0]
    if dealer_total > 21 or player_total > dealer_total:
        return "win"
    return "push" if player_total == dealer_total else "lose"


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


@dataclass(frozen=True)
class CountSystem:
    name: str               # display name, e.g. "Hi-Opt II"
    values: dict[str, int]  # rank -> what one card of it adds to the running count
    unit_cards: int         # cards in one unit of the true count's divisor: a deck or a half-deck

    def value(self, label: str) -> int:
        """What one card adds to the running count. BACK is 0."""
        return 0 if label == BACK else self.values[rank_of(label)]


def _by_rank(groups: dict[str, int]) -> dict[str, int]:
    """Values given for space-separated ranks, such as {"2 3": 1}, as one entry per rank."""
    return {rank: value for ranks, value in groups.items() for rank in ranks.split()}


DEFAULT_COUNT_SYSTEM = "hi_lo"
COUNT_SYSTEMS = {  # system id -> system, in the order the page's dropdown lists them
    "hi_lo": CountSystem("Hi-Lo", _by_rank({"2 3 4 5 6": 1, "7 8 9": 0, "10 J Q K A": -1}), CARDS_PER_DECK),
    "hi_opt_ii": CountSystem("Hi-Opt II", _by_rank({"2 3 6 7": 1, "4 5": 2, "8 9 A": 0, "10 J Q K": -2}),
                             CARDS_PER_DECK),
    "uston_apc": CountSystem("Uston APC",
                             _by_rank({"2 8": 1, "3 4 6 7": 2, "5": 3, "9": -1, "10 J Q K": -3, "A": 0}),
                             CARDS_PER_DECK // 2),
}


def hi_lo(label: str) -> int:
    """Hi-Lo value of one card: +1 for 2-6, 0 for 7-9, -1 for 10, J, Q, K, A. BACK is 0."""
    return COUNT_SYSTEMS["hi_lo"].value(label)


@dataclass
class ShoeCount:
    running: int      # sum of the system's values of every card seen this shoe
    true: float       # running count per deck (half-deck for Uston APC) still to be dealt, floored as above
    cards_left: int   # cards not yet dealt: the shoe minus faces seen and face-down cards never revealed
    system: str = DEFAULT_COUNT_SYSTEM  # the COUNT_SYSTEMS id it was counted in
    decks: int = 1    # decks in the shoe: cards_left starts from decks * CARDS_PER_DECK


def shoe_count(seen: list[str], decks: int = 1, unrevealed: int = 0,
               system: str = DEFAULT_COUNT_SYSTEM) -> ShoeCount:
    """Count for the face cards seen since the last shuffle, in the COUNT_SYSTEMS entry `system`.

    unrevealed: face-down cards dealt this shoe and never turned over. They
    leave the deck, so they lower cards_left, but add nothing to the running count.
    The true count divides by the system's units left (decks, or half-decks for Uston APC),
    counting never fewer cards than MIN_DECKS_LEFT of a deck. Raises KeyError for an unknown system."""
    counting = COUNT_SYSTEMS[system]
    running = sum(counting.value(label) for label in seen)
    cards_left = max(decks * CARDS_PER_DECK - len(seen) - unrevealed, 0)
    divisor_cards = max(cards_left, MIN_DECKS_LEFT * CARDS_PER_DECK)
    units_left = divisor_cards / counting.unit_cards
    return ShoeCount(running, running / units_left, cards_left, system, decks)


NEXT_CARD_TOTALS = (21, 20, 19, 18)  # the totals a live hand's odds give a chance for, then bust


def _percent(part: int | float, whole: int | float) -> int:
    return round(100 * part / whole)


def _completes_blackjack(label: str, unseen: list[str]) -> int:
    """The chance, as a whole percentage, that one card drawn from unseen makes a blackjack with label:
    a ten-value card for an ace, an ace for a ten-value card, else 0."""
    rank = rank_of(label)
    if rank == "A":
        wanted = sum(rank_of(card) in TEN_RANKS for card in unseen)
    elif rank in TEN_RANKS:
        wanted = sum(rank_of(card) == "A" for card in unseen)
    else:
        wanted = 0
    return _percent(wanted, len(unseen))


def player_odds(labels: list[str], dealer_labels: list[str], unseen: list[str]) -> dict[str, int] | None:
    """A player hand's chances from the unseen card labels, as whole percentages.

    No cards: {"blackjack": p}, the chance of a two-card blackjack, 2 * aces/N * tens/(N - 1).
    One face-up card: {"blackjack": p}, the chance the next card completes a blackjack.
    Two or more face-up cards, still live: the chance the next card makes each NEXT_CARD_TOTALS
    total and the chance it busts, e.g. {"21": 8, "20": 8, "19": 8, "18": 8, "bust": 31};
    totals follow hand_value, so soft hands count an ace as 1 once 11 busts.
    None when nothing is unseen, a card is face down, or the hand is finished: any outcome
    against dealer_labels (bust, blackjack, twenty-one, win, push, lose), which every
    two-card-or-more hand of 21 or over has."""
    if not unseen or BACK in labels or outcome(labels, dealer_labels) is not None:
        return None
    if not labels:
        if len(unseen) < 2:
            return {"blackjack": 0}
        aces = sum(rank_of(card) == "A" for card in unseen)
        tens = sum(rank_of(card) in TEN_RANKS for card in unseen)
        return {"blackjack": _percent(2 * aces * tens, len(unseen) * (len(unseen) - 1))}
    if len(labels) == 1:
        return {"blackjack": _completes_blackjack(labels[0], unseen)}
    totals = [hand_value(labels + [card])[0] for card in unseen]
    odds = {str(total): _percent(totals.count(total), len(totals)) for total in NEXT_CARD_TOTALS}
    odds["bust"] = _percent(sum(total > 21 for total in totals), len(totals))
    return odds


def dealer_odds(labels: list[str], unseen: list[str]) -> dict[str, int] | None:
    """{"blackjack": p}, the chance as a whole percentage that a dealer holding one face-up card
    and one face-down hole card has blackjack: the hole card drawn from the unseen card labels
    completes it (0 unless the up card is an ace or ten-value). None for any other dealer hand,
    such as no face-down card, or when nothing is unseen."""
    faces = [label for label in labels if label != BACK]
    if not unseen or len(labels) != 2 or len(faces) != 1:
        return None
    return {"blackjack": _completes_blackjack(faces[0], unseen)}


DEALER_FINALS = ("21", "20", "19", "18", "17", "bust")  # the dealer's possible finals, in the page's order


def _draw_counts(unseen: list[str]) -> tuple[int, ...]:
    """Unseen copies by the value a draw adds with aces as 1: index 0 is aces, 1 to 8 are 2-9, 9 is ten-value."""
    counts = [0] * 10
    for card in unseen:
        value = card_value(card)
        counts[0 if value == 11 else value - 1] += 1
    return tuple(counts)


def _best_total(hard: int, has_ace: bool) -> int:
    """hand_value's total for a hand worth hard with every ace as 1: one ace counts 11 when that does not bust."""
    return hard + 10 if has_ace and hard + 10 <= 21 else hard


@lru_cache(maxsize=8)
def _dealer_finals(faces: tuple[str, ...], hidden: int, counts: tuple[int, ...]) -> dict[str, Fraction] | None:
    """Exact chances of each DEALER_FINALS entry for a dealer with faces up and hidden cards face down,
    drawing from counts (_draw_counts). With exactly one card up and one down, "blackjack" comes first
    and "21" leaves it out. None when the cards can run out before the dealer finishes, on any path,
    since the finals would then be incomplete.

    Cached, so the dealer's row and every player's stand odds in one snapshot share one computation."""

    @cache
    def play_out(hard: int, has_ace: bool, left: tuple[int, ...], draws_due: int) -> dict[str, Fraction] | None:
        """The finals' chances from a hand worth hard (aces as 1) that must still draw draws_due cards,
        or None when the cards in left can run out first."""
        if hard > 21:
            return {"bust": Fraction(1)}
        total = _best_total(hard, has_ace)
        if not draws_due and total >= DEALER_STANDS_AT:
            return {str(total): Fraction(1)}
        cards_left = sum(left)
        if not cards_left:
            return None
        finals: dict[str, Fraction] = {}
        for index, copies in enumerate(left):
            if not copies:
                continue
            value = index + 1
            after = left[:index] + (copies - 1,) + left[index + 1:]
            drawn = play_out(hard + value, has_ace or value == 1, after, max(draws_due - 1, 0))
            if drawn is None:
                return None
            for final, share in drawn.items():
                finals[final] = finals.get(final, Fraction(0)) + Fraction(copies, cards_left) * share
        return finals

    hard = sum(1 if rank_of(label) == "A" else card_value(label) for label in faces)
    has_ace = any(rank_of(label) == "A" for label in faces)
    played = play_out(hard, has_ace, counts, hidden)
    if played is None:
        return None
    chances = dict.fromkeys(DEALER_FINALS, Fraction(0)) | played
    if len(faces) != 1 or hidden != 1:
        return chances
    # the hole card completes a blackjack: a ten-value card under an ace, an ace under a ten-value card
    completing = {11: counts[9], 10: counts[0]}.get(card_value(faces[0]), 0)
    blackjack = Fraction(completing, sum(counts))
    chances["21"] -= blackjack
    return {"blackjack": blackjack} | chances


def _dealer_chances(labels: list[str], unseen: list[str]) -> dict[str, Fraction] | None:
    """_dealer_finals for a dealer hand, or None when nothing is unseen, no card is up, the dealer is finished,
    or the unseen cards can run out before the dealer finishes."""
    faces = tuple(label for label in labels if label != BACK)
    if not unseen or not faces or dealer_finished(labels):
        return None
    return _dealer_finals(faces, labels.count(BACK), _draw_counts(unseen))


def dealer_final_odds(labels: list[str], unseen: list[str]) -> dict[str, int] | None:
    """The chance, as whole percentages, that the dealer ends on each of 21, 20, 19, 18, 17, or bust,
    e.g. {"21": 12, "20": 18, "19": 13, "18": 13, "17": 13, "bust": 31}.

    The dealer plays out from its face-up cards, drawing from the unseen card labels without replacement
    until DEALER_STANDS_AT (soft 17 included) or bust; each face-down card is one more random draw.
    With exactly one card up and one face down, "blackjack" comes first, the dealer_odds value, and
    "21" leaves blackjacks out, so the entries are separate outcomes. Each entry is rounded on its own,
    so they may sum to 99 or 101.
    None when nothing is unseen, no card is face up, the dealer is finished (dealer_finished), or some
    order of draws runs out of unseen cards before the dealer finishes, so the finals would be incomplete."""
    chances = _dealer_chances(labels, unseen)
    if chances is None:
        return None
    return {final: _percent(chance, 1) for final, chance in chances.items()}


def stand_odds(player_labels: list[str], dealer_labels: list[str], unseen: list[str]) -> dict[str, int] | None:
    """A live player hand's chances, as whole percentages, to win, push, or lose if it stands now,
    e.g. {"win": 42, "push": 6, "lose": 52}.

    The player's total meets each of the dealer's finals (dealer_final_odds): a dealer bust or a lower
    total wins, the same total pushes, and a higher total or a dealer blackjack loses.
    None for a hand with fewer than two cards and whenever player_odds is None (nothing unseen,
    a card face down, or the hand finished), and whenever dealer_final_odds is None for a dealer
    still to play, such as one with no face-up card or one the unseen cards could run out on."""
    if len(player_labels) < 2 or player_odds(player_labels, dealer_labels, unseen) is None:
        return None
    chances = _dealer_chances(dealer_labels, unseen)
    if chances is None:
        return None
    player_total = hand_value(player_labels)[0]
    win = chances["bust"]
    push = Fraction(0)
    lose = chances.get("blackjack", Fraction(0))
    for final in DEALER_FINALS[:-1]:
        dealer_total = int(final)
        if dealer_total < player_total:
            win += chances[final]
        elif dealer_total == player_total:
            push += chances[final]
        else:
            lose += chances[final]
    return {"win": _percent(win, 1), "push": _percent(push, 1), "lose": _percent(lose, 1)}


def insurance(dealer_labels: list[str], unseen: list[str]) -> str | None:
    """"take" or "skip" for the insurance bet while the dealer shows an ace with one card face down.

    Insurance pays 2 to 1 when the hole card is ten-value, so it pays on average only when more than
    one third of the unseen cards are ten-value: "take" then, "skip" at a third or less.
    None for any other dealer hand, or when nothing is unseen."""
    faces = [label for label in dealer_labels if label != BACK]
    if not unseen or len(dealer_labels) != 2 or len(faces) != 1 or rank_of(faces[0]) != "A":
        return None
    tens = sum(rank_of(card) in TEN_RANKS for card in unseen)
    return "take" if 3 * tens > len(unseen) else "skip"
