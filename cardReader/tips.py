"""Tips for the page: the play the unseen cards favour for a player hand, and how strongly, and a
bet tip for an empty seat from the true count.

    # two decks, less the three cards shown: 101 unseen
    play_values(["10S", "6H"], ["10D", "BACK"], unseen)  # {"stand": -0.54, "hit": -0.52, "double": -1.05}
    tip(["10S", "6H"], ["10D", "BACK"], unseen)          # Tip("hit", "Close: hit", "close", 2,
                                                         #     "hit +0.02 over stand, against 101 unseen cards")
    bet_tip(2.4, "hi_lo")                                # Tip("bet-more", "Bet more", "normal", None, ...)

Values are expected winnings in units of the bet, exact against the unseen card labels (the
shoe minus the faces seen, one entry per copy left) up to float rounding:

  * stand: the chance to win minus the chance to lose against the dealer's finals
    (blackjack.dealer_final_odds), played out from the dealer's cards and the unseen cards.
  * hit: one card drawn from the unseen cards without replacement, then the better of standing
    and hitting again, on and on; a bust is worth -1. The dealer's finals are not recomputed
    for the cards the player draws: an approximation that keeps the work to one dealer play-out.
  * double: twice the value of one card drawn, then standing. Only for a hand of two cards.

A dealer with one card up and one face down has peeked for blackjack before the player acts, so
the values leave the dealer's blackjack out: the other finals are scaled to sum to one.

The tip is the play with the largest value. Its tone says how far ahead it is of the runner-up:
"firm" at FIRM_GAP of a bet or more ("Stand."), "close" below CLOSE_GAP ("Close: stand"), and
"normal" between ("Stand"). gap is that lead in hundredths of a bet. Splitting has no value here:
when basic strategy (strategy.hint) says split, the tip is split; when the values are unavailable,
the tip is basic strategy's play. Both have a normal tone and no gap.

Bet tips use each CountSystem's bet_more_at and bet_less_at: "Bet more" at or above the first,
"Bet less" at or below the second, firm ("Bet more.", "Bet less.") at double the threshold, and
"Bet flat" between.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cache, lru_cache

from .blackjack import (
    COUNT_SYSTEMS,
    DEALER_FINALS,
    _best_total,
    _dealer_chances,
    _dealer_finals,
    _draw_counts,
    card_value,
    hand_value,
)
from .deck import BACK, rank_of
from .strategy import hint

FIRM_GAP = 0.10   # bets: a play this far ahead of the runner-up is firm
CLOSE_GAP = 0.03  # bets: a play less than this far ahead is close


@dataclass(frozen=True)
class Tip:
    play: str          # "stand", "hit", "double", "split", or "bet-more", "bet-less", "bet-flat"
    text: str          # what the pill says, e.g. "Stand.", "Hit", "Close: stand", "Bet more"
    tone: str          # "firm", "normal", or "close"
    gap: int | None    # the play's lead over the runner-up in hundredths of a bet, None without values
    title: str         # the hover sentence, e.g. "stand +0.12 over hit, against 99 unseen cards"


def play_values(labels: list[str], dealer_labels: list[str], unseen: list[str]) -> dict[str, float] | None:
    """The expected values of "stand", "hit", and, with exactly two cards, "double" for a player
    hand, in units of the bet, against the unseen card labels (see the module docstring).

    None for a hand with a face-down card, fewer than two cards, or a bust, and whenever the
    dealer's finals are unavailable: nothing unseen, no dealer card up, the dealer finished, or
    the unseen cards could run out before the dealer finishes. Memoised, like the dealer's finals,
    so a frame computes each hand once."""
    if BACK in labels or len(labels) < 2 or hand_value(labels)[0] > 21:
        return None
    if _dealer_chances(dealer_labels, unseen) is None:
        return None
    faces = tuple(label for label in dealer_labels if label != BACK)
    hard = sum(1 if rank_of(label) == "A" else card_value(label) for label in labels)
    has_ace = any(rank_of(label) == "A" for label in labels)
    values = _play_values(hard, has_ace, len(labels) == 2, faces, dealer_labels.count(BACK), _draw_counts(unseen))
    return None if values is None else dict(values)


@lru_cache(maxsize=64)
def _play_values(hard: int, has_ace: bool, two_cards: bool, faces: tuple[str, ...], hidden: int,
                 counts: tuple[int, ...]) -> tuple[tuple[str, float], ...] | None:
    """play_values for a hand worth hard (aces as 1) against a dealer with faces up and hidden cards
    down, drawing from counts (blackjack._draw_counts). None when the dealer surely has blackjack."""
    finals = _dealer_finals(faces, hidden, counts)
    blackjack = finals.get("blackjack", Fraction(0))
    if blackjack == 1:
        return None
    stand_by_total = {total: _stand_value(total, finals) / (1 - blackjack) for total in range(2, 22)}

    def stand(hard: int, has_ace: bool, left: tuple[int, ...]) -> float:
        return stand_by_total[_best_total(hard, has_ace)]

    @cache
    def best(hard: int, has_ace: bool, left: tuple[int, ...]) -> float:
        """The better of standing and hitting on, for a hand worth hard drawing from left."""
        standing = stand(hard, has_ace, left)
        return max(standing, _draw(hard, has_ace, left, best)) if sum(left) else standing

    values = {"stand": stand(hard, has_ace, counts), "hit": _draw(hard, has_ace, counts, best)}
    if two_cards:
        values["double"] = 2 * _draw(hard, has_ace, counts, stand)
    return tuple(values.items())


def _stand_value(total: int, finals: dict[str, Fraction]) -> float:
    """Win minus lose for a player total standing against the dealer's finals, leaving any
    "blackjack" entry out (the caller scales by the chance it is not one)."""
    value = finals["bust"]
    for final in DEALER_FINALS[:-1]:
        dealer_total = int(final)
        if dealer_total < total:
            value += finals[final]
        elif dealer_total > total:
            value -= finals[final]
    return float(value)


def _draw(hard: int, has_ace: bool, left: tuple[int, ...], then) -> float:
    """The value of drawing one card from left, then playing on with then(hard, has_ace, left);
    a bust is worth -1."""
    cards_left = sum(left)
    value = 0.0
    for index, copies in enumerate(left):
        if not copies:
            continue
        drawn = index + 1
        if hard + drawn > 21:
            outcome = -1.0
        else:
            after = left[:index] + (copies - 1,) + left[index + 1:]
            outcome = then(hard + drawn, has_ace or drawn == 1, after)
        value += copies / cards_left * outcome
    return value


def tip(labels: list[str], dealer_labels: list[str], unseen: list[str], can_split: bool = True) -> Tip | None:
    """The tip for a player hand against the unseen card labels, or None where basic strategy gives
    no hint (a face-down card, fewer than two cards, a bust, a blackjack, or no dealer card up).

    can_split: False for one hand of a split seat, whose pair is played as its total."""
    basic = hint(labels, dealer_labels, can_split=can_split)
    if basic is None:
        return None
    if basic == "split":
        return Tip("split", "Split", "normal", None, "split: basic strategy for this pair")
    values = play_values(labels, dealer_labels, unseen)
    if values is None:
        return Tip(basic, basic.capitalize(), "normal", None,
                   f"{basic}: basic strategy, as the unseen cards cannot play the dealer out")
    ranked = sorted(values.items(), key=lambda item: item[1], reverse=True)  # stable: stand first on a tie
    (play, best), (runner_up, second) = ranked[0], ranked[1]
    lead = best - second
    title = f"{play} {lead:+.2f} over {runner_up}, against {len(unseen)} unseen cards"
    return Tip(play, _text(play.capitalize(), _tone(lead), play), _tone(lead), round(100 * lead), title)


def _tone(lead: float) -> str:
    if lead >= FIRM_GAP:
        return "firm"
    return "close" if lead < CLOSE_GAP else "normal"


def _text(word: str, tone: str, play: str) -> str:
    """The pill's text: "Stand." when firm, "Close: stand" when close, else "Stand"."""
    if tone == "firm":
        return f"{word}."
    return f"Close: {play}" if tone == "close" else word


def bet_tip(true: float, system: str) -> Tip:
    """The bet tip for an empty player seat from the true count in the COUNT_SYSTEMS entry system:
    "bet-more" at or above its bet_more_at, "bet-less" at or below its bet_less_at, firm at double
    either threshold, else "bet-flat". Raises KeyError for an unknown system."""
    counting = COUNT_SYSTEMS[system]
    more, less = counting.bet_more_at, counting.bet_less_at
    title = f"true count {true:+.1f} ({counting.name}): bet more at {more:+g} or above, less at {less:+g} or below"
    if true >= more:
        tone = "firm" if true >= 2 * more else "normal"
        return Tip("bet-more", _text("Bet more", tone, "bet more"), tone, None, title)
    if true <= less:
        tone = "firm" if true <= 2 * less else "normal"
        return Tip("bet-less", _text("Bet less", tone, "bet less"), tone, None, title)
    return Tip("bet-flat", "Bet flat", "normal", None, title)
